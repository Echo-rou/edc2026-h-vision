#!/usr/bin/env python3
"""Orange Pi ball detection with a browser UI and PC-side recording."""

import argparse
import json
import os
import signal
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


class SharedState:
    def __init__(self):
        self.stop_event = threading.Event()
        self.frame_condition = threading.Condition()
        self.capture_condition = threading.Condition()
        self.captured_frame = None
        self.capture_seq = -1
        self.jpeg = None
        self.jpeg_seq = -1
        self.status = "starting"
        self.error = ""
        self.fps = 0.0

    def publish_capture(self, frame, seq):
        with self.capture_condition:
            self.captured_frame = frame
            self.capture_seq = seq
            self.capture_condition.notify_all()

    def wait_capture(self, last_seq):
        with self.capture_condition:
            self.capture_condition.wait_for(
                lambda: self.capture_seq != last_seq or self.stop_event.is_set(),
                timeout=0.5,
            )
            return self.capture_seq, self.captured_frame

    def publish_jpeg(self, jpeg, seq, fps):
        with self.frame_condition:
            self.jpeg = jpeg
            self.jpeg_seq = seq
            self.fps = fps
            self.status = "running"
            self.frame_condition.notify_all()


STATE = SharedState()
APP_DIR = Path(__file__).resolve().parent


def letterbox(image, size=640):
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_height = int(height * scale)
    resized_width = int(width * scale)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    resized = cv2.resize(image, (resized_width, resized_height))
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized
    return canvas, scale, x_offset, y_offset


class BallDetector:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Cannot load RKNN model: {model_path}")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            self.rknn.release()
            raise RuntimeError("Cannot initialize RKNN runtime")
        print(f"[NPU] Ready: {model_path}")

    def close(self):
        self.rknn.release()

    def detect(self, frame):
        image_height, image_width = frame.shape[:2]
        padded, scale, x_offset, y_offset = letterbox(frame)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(
            inputs=[np.expand_dims(rgb, axis=0)], data_format=["nhwc"]
        )
        if not outputs:
            raise RuntimeError("RKNN inference returned no output")
        output = (
            np.concatenate((outputs[0], outputs[1]), axis=1)
            if len(outputs) == 2
            else outputs[0]
        )
        data = output[0].T
        selected = data[data[:, 4] > 0.5]
        if not len(selected):
            return []

        xc, yc, width, height, confidence = selected.T
        x1 = np.clip(((xc - width / 2) - x_offset) / scale, 0, image_width)
        y1 = np.clip(((yc - height / 2) - y_offset) / scale, 0, image_height)
        x2 = np.clip(((xc + width / 2) - x_offset) / scale, 0, image_width)
        y2 = np.clip(((yc + height / 2) - y_offset) / scale, 0, image_height)
        nms_boxes = [
            [float(left), float(top), float(right - left), float(bottom - top)]
            for left, top, right, bottom in zip(x1, y1, x2, y2)
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidence.tolist(), 0.5, 0.45
        )
        if len(indices) == 0:
            return []
        return [
            (
                int(x1[index]), int(y1[index]),
                int(x2[index]), int(y2[index]),
                float(confidence[index]),
            )
            for index in indices.flatten()
        ]


def open_camera(camera_id):
    pipeline = (
        f"v4l2src device=/dev/video{camera_id} io-mode=2 ! "
        "image/jpeg,width=640,height=480,framerate=90/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if camera.isOpened():
        print("[Camera] GStreamer: 640x480 @ 90 FPS")
        return camera

    camera.release()
    camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{camera_id}")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 90)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[Camera] V4L2 fallback")
    return camera


def capture_loop(camera_id):
    camera = None
    try:
        camera = open_camera(camera_id)
        seq = 0
        while not STATE.stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                continue
            seq += 1
            STATE.publish_capture(frame, seq)
    except Exception as error:
        STATE.error = str(error)
        STATE.status = "error"
        STATE.stop_event.set()
        print(f"[Camera] Error: {error}")
    finally:
        if camera is not None:
            camera.release()


def vision_loop(model_path, jpeg_quality, stream_fps, detect_enabled):
    detector = None
    frame_times = deque()
    last_seq = -1
    last_publish = 0.0
    min_interval = 1.0 / max(stream_fps, 1.0)
    try:
        if detect_enabled:
            detector = BallDetector(model_path)
        while not STATE.stop_event.is_set():
            seq, source = STATE.wait_capture(last_seq)
            if source is None or seq == last_seq:
                continue
            last_seq = seq
            frame = source.copy()
            boxes = detector.detect(frame) if detector is not None else []

            now = time.perf_counter()
            frame_times.append(now)
            while frame_times and frame_times[0] < now - 1.0:
                frame_times.popleft()
            fps = float(len(frame_times))

            for x1, y1, x2, y2, confidence in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"ball {confidence:.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
            cv2.putText(
                frame, f"FPS {fps:.1f}  Balls {len(boxes)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA,
            )

            if now - last_publish < min_interval:
                continue
            ok, jpeg = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if ok:
                STATE.publish_jpeg(jpeg.tobytes(), seq, fps)
                last_publish = now
    except Exception as error:
        STATE.error = str(error)
        STATE.status = "error"
        STATE.stop_event.set()
        print(f"[Vision] Error: {error}")
    finally:
        if detector is not None:
            detector.close()


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BallVision</title>
<style>
:root{color-scheme:dark;font-family:Arial,"Microsoft YaHei",sans-serif}
*{box-sizing:border-box}body{margin:0;background:#101214;color:#eef1f4}
header{height:58px;display:flex;align-items:center;justify-content:space-between;padding:0 20px;background:#191c20;border-bottom:1px solid #30343a}
h1{font-size:20px;margin:0}.status{display:flex;align-items:center;gap:8px;color:#aeb6bf;font-size:14px}.dot{width:9px;height:9px;border-radius:50%;background:#27c56f}
main{max-width:1100px;margin:auto;padding:18px}.video{background:#000;aspect-ratio:4/3;max-height:calc(100vh - 170px);margin:auto;position:relative}
canvas{display:block;width:100%;height:100%;object-fit:contain}.source{position:absolute;width:1px;height:1px;opacity:0}
.toolbar{display:flex;align-items:center;justify-content:center;gap:16px;padding-top:15px}
button{height:46px;min-width:150px;border:1px solid #3977d1;background:#2868c7;color:#fff;padding:0 22px;border-radius:7px;font-size:17px;cursor:pointer}
button:hover{background:#3478dc}button.recording{background:#c93838;border-color:#dc4a4a}button.recording:hover{background:#dd4141}button:disabled{opacity:.6;cursor:wait}
.timer{min-width:85px;font-variant-numeric:tabular-nums;color:#f1c75b;text-align:center;font-size:18px}.hint{text-align:center;color:#9099a3;font-size:13px;margin-top:10px}
</style>
</head>
<body>
<header><h1>BallVision</h1><div class="status"><span class="dot"></span><span id="state">实时画面</span></div></header>
<main>
  <div class="video"><img id="source" class="source" src="/stream.mjpg"><canvas id="view" width="640" height="480"></canvas></div>
  <div class="toolbar">
    <button id="save">开始保存</button>
    <span id="timer" class="timer">00:00.0</span>
  </div>
  <div id="hint" class="hint">第一次点击开始录像，再次点击结束并保存；随后可继续下一次录像。</div>
</main>
<script>
const source=document.getElementById('source'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const saveBtn=document.getElementById('save'),timer=document.getElementById('timer'),hint=document.getElementById('hint');
let recorder=null,chunks=[],started=0,timerId=null;
function draw(){if(source.naturalWidth){ctx.drawImage(source,0,0,canvas.width,canvas.height)}requestAnimationFrame(draw)}draw();
function mime(){const choices=['video/mp4;codecs=h264','video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];return choices.find(x=>MediaRecorder.isTypeSupported(x))||''}
function format(ms){const s=ms/1000;return String(Math.floor(s/60)).padStart(2,'0')+':'+(s%60).toFixed(1).padStart(4,'0')}
function startSave(){
  const stream=canvas.captureStream(25),type=mime();
  chunks=[];
  recorder=new MediaRecorder(stream,type?{mimeType:type,videoBitsPerSecond:5000000}:undefined);
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  recorder.onstop=()=>{
    const blob=new Blob(chunks,{type:recorder.mimeType});
    const ext=recorder.mimeType.includes('mp4')?'mp4':'webm';
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='H题钢球测试_'+new Date().toISOString().replace(/[:.]/g,'-')+'.'+ext;
    a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    saveBtn.textContent='开始保存';
    saveBtn.classList.remove('recording');
    saveBtn.disabled=false;
    hint.textContent='录像已保存到电脑下载目录，可以再次点击开始下一次录像。';
    document.getElementById('state').textContent='实时画面';
  };
  recorder.start(1000);
  started=performance.now();
  timer.textContent='00:00.0';
  timerId=setInterval(()=>timer.textContent=format(performance.now()-started),100);
  saveBtn.textContent='结束并保存';
  saveBtn.classList.add('recording');
  hint.textContent='正在录像。测试结束后再次点击按钮。';
  document.getElementById('state').textContent='正在保存';
}
saveBtn.onclick=()=>{
  if(recorder&&recorder.state!=='inactive'){
    clearInterval(timerId);
    saveBtn.disabled=true;
    recorder.stop();
  }else{
    startSave();
  }
};
</script>
</body></html>"""


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/health":
            body = json.dumps(
                {"status": STATE.status, "error": STATE.error, "fps": STATE.fps}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_seq = -1
        try:
            while not STATE.stop_event.is_set():
                with STATE.frame_condition:
                    STATE.frame_condition.wait_for(
                        lambda: STATE.jpeg_seq != last_seq or STATE.stop_event.is_set(),
                        timeout=1.0,
                    )
                    jpeg = STATE.jpeg
                    last_seq = STATE.jpeg_seq
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class VideoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="BallVision browser application")
    parser.add_argument("--camera", type=int, default=int(os.getenv("BALLVISION_CAMERA", "0")))
    parser.add_argument(
        "--model",
        default=os.getenv(
            "BALLVISION_MODEL",
            str(APP_DIR / "ball_best_int8_split.rknn"),
        ),
    )
    parser.add_argument("--host", default=os.getenv("BALLVISION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BALLVISION_PORT", "8080")))
    parser.add_argument("--address", default=os.getenv("BALLVISION_ADDRESS", "192.168.5.1"))
    parser.add_argument("--fps", type=float, default=float(os.getenv("BALLVISION_FPS", "25")))
    parser.add_argument("--quality", type=int, default=int(os.getenv("BALLVISION_QUALITY", "65")))
    parser.add_argument("--no-detect", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.no_detect and not Path(args.model).is_file():
        raise FileNotFoundError(f"RKNN model not found: {args.model}")
    cv2.setNumThreads(2)
    capture_thread = threading.Thread(
        target=capture_loop, args=(args.camera,), name="capture"
    )
    vision_thread = threading.Thread(
        target=vision_loop,
        args=(args.model, args.quality, args.fps, not args.no_detect),
        name="vision",
    )
    capture_thread.start()
    vision_thread.start()
    server = VideoServer((args.host, args.port), WebHandler)
    server_thread = threading.Thread(target=server.serve_forever, name="web")
    server_thread.start()

    print("[Web] BallVision is ready")
    print(f"[Web] Open: http://{args.address}:{args.port}/")
    print("[Main] Press Ctrl+C to stop")

    def request_stop(*_):
        STATE.stop_event.set()
        with STATE.capture_condition:
            STATE.capture_condition.notify_all()
        with STATE.frame_condition:
            STATE.frame_condition.notify_all()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not STATE.stop_event.wait(0.5):
            pass
    finally:
        server.shutdown()
        server.server_close()
        capture_thread.join(timeout=3)
        vision_thread.join(timeout=3)
        server_thread.join(timeout=3)
        print("[Main] Exit")


if __name__ == "__main__":
    main()
