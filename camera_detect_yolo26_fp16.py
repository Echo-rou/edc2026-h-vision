#!/usr/bin/env python3
"""钢球检测 - 多线程 NPU 推理、MJPEG 图传和本地录像。"""
import os
import cv2, numpy as np, serial, time, threading
from datetime import datetime
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue, Empty
from rknnlite.api import RKNNLite

# ==================== 配置 ====================
MODEL_FILE  = "ball_yolo26_single_fp16.rknn"
CAMERA_ID   = 0
IMG_SIZE    = 640
CONF_THRESH = 0.5
UART_PORT   = "/dev/ttyS1"
UART_BAUD   = 115200
NPU_WORKERS = 2
# Remote runs use MJPEG. OpenCV GUI can abort when an SSH session has a stale DISPLAY.
LOCAL_DISPLAY = os.environ.get("LOCAL_DISPLAY") == "1"
MJPEG_ENABLED = True
MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = 8080
MJPEG_QUALITY = 50
MJPEG_MAX_FPS = 25
# Use STREAM_ONLY=1 to validate the camera and wireless link without RKNN.
STREAM_ONLY = os.environ.get("STREAM_ONLY") == "1"
# The required recording is performed by the off-board receiving PC.
RECORD_ENABLED = os.environ.get("LOCAL_RECORD") == "1"
RECORD_DIR = "recordings"
RECORD_FPS = 30

# ==================== 全局状态 ====================
frame_queue = Queue(maxsize=NPU_WORKERS)
result_lock  = threading.Lock()
latest_result = {
    "seq": -1,
    "frame": None,
    "balls": [],
    "fps": 0.0,
    "max_conf": 0.0,
    "capture_fps": 0.0,
    "capture_time": 0.0,
}
completion_times = deque()
capture_times = deque()
display_times = deque()
stream_condition = threading.Condition()
stream_frame = None
stream_seq = -1
running = True


def stop(reason):
    global running
    if running:
        print(f"[Main] Stopping: {reason}")
    running = False

# ==================== MJPEG HTTP 图传 ====================
class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>Ball Vision</title>"
                "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
                "main{min-height:100vh;display:grid;place-items:center;padding:12px;}"
                "img{max-width:100%;height:auto;background:#000;}</style></head>"
                "<body><main><img src='/stream.mjpg' alt='Ball Vision'></main></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        min_interval = 1.0 / max(MJPEG_MAX_FPS, 1)
        last_send = 0.0
        while running:
            with stream_condition:
                stream_condition.wait_for(
                    lambda: stream_seq != last_seq or not running, timeout=1.0
                )
                if not running:
                    break
                frame = stream_frame
                last_seq = stream_seq
            if frame is None:
                continue
            now = time.perf_counter()
            sleep_time = min_interval - (now - last_send)
            if sleep_time > 0:
                time.sleep(sleep_time)
            ok, jpg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY]
            )
            if not ok:
                continue
            data = jpg.tobytes()
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                last_send = time.perf_counter()
            except (BrokenPipeError, ConnectionResetError):
                break


def mjpeg_server_thread():
    global running
    server = ThreadingHTTPServer((MJPEG_HOST, MJPEG_PORT), MjpegHandler)
    server.timeout = 0.5
    print(f"[MJPEG] http://{MJPEG_HOST}:{MJPEG_PORT}/")
    while running:
        server.handle_request()
    server.server_close()


def publish_stream_frame(frame):
    global stream_frame, stream_seq
    if not MJPEG_ENABLED:
        return
    with stream_condition:
        stream_frame = frame.copy()
        stream_seq += 1
        stream_condition.notify_all()

# ==================== YOLO 工具函数 ====================
def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size/h, size/w)
    nh, nw = round(h*scale), round(w*scale)
    canvas = np.full((size,size,3), 114, dtype=np.uint8)
    resized = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh))
    canvas[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = resized
    return canvas, scale, (size-nw)//2, (size-nh)//2

def postprocess(output, scale, xo, yo, iw, ih):
    # FP16 YOLO26 end-to-end output: [batch, 300, 6].
    # Each row is [x1, y1, x2, y2, confidence, class_id].
    data = np.asarray(output)
    if data.shape == (1, 6, 300):
        data = data.transpose(0, 2, 1)
    if data.shape != (1, 300, 6):
        raise RuntimeError(f"Unexpected YOLO26 output shape: {data.shape}")
    data = data[0]
    classes = np.rint(data[:, 5]).astype(np.int32)
    mask = (data[:, 4] > CONF_THRESH) & (classes == 0)
    f = data[mask]
    if len(f) == 0:
        return [], []
    x1 = np.clip((f[:, 0] - xo) / scale, 0, iw)
    y1 = np.clip((f[:, 1] - yo) / scale, 0, ih)
    x2 = np.clip((f[:, 2] - xo) / scale, 0, iw)
    y2 = np.clip((f[:, 3] - yo) / scale, 0, ih)
    boxes = np.stack([x1,y1,x2,y2], axis=1)
    return f[:,4].tolist(), boxes.tolist()

# ==================== 线程 1: 摄像头采集 ====================
def capture_thread(camera_id):
    global running
    pipeline = (
        f"v4l2src device=/dev/video{camera_id} io-mode=2 ! "
        "image/jpeg,width=640,height=480,framerate=60/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    backend = "GStreamer"
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        backend = "V4L2"
    if not cap.isOpened():
        print(f"[Thread-1] Cannot open /dev/video{camera_id}")
        stop("camera cannot be opened")
        return
    if backend == "V4L2":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fmt = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"[Thread-1] Camera OK: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS, {fmt}, {backend}"
    )

    seq = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        # In detection mode, only the main thread publishes annotated frames.
        # Publishing both raw and annotated frames makes the MJPEG stream flicker.
        if STREAM_ONLY:
            publish_stream_frame(frame)
        now = time.perf_counter()
        with result_lock:
            capture_times.append(now)
            while capture_times and capture_times[0] < now - 1.0:
                capture_times.popleft()
            latest_result["capture_fps"] = float(len(capture_times))
        seq += 1
        # 队列满时丢弃最旧帧，优先保证低延迟。
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put((seq, now, frame))
    cap.release()

# ==================== NPU 推理线程 ====================
def inference_thread(worker_id, core_mask):
    global running, latest_result
    print(f"[NPU-{worker_id}] Loading RKNN: {MODEL_FILE}")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_FILE) != 0:
        print(f"[NPU-{worker_id}] Load failed")
        stop(f"NPU-{worker_id} model load failed")
        return
    if rknn.init_runtime(core_mask=core_mask) != 0:
        print(f"[NPU-{worker_id}] Runtime init failed")
        stop(f"NPU-{worker_id} runtime init failed")
        rknn.release()
        return
    print(f"[NPU-{worker_id}] Ready")

    while running:
        try:
            seq, capture_time, frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        oh, ow = frame.shape[:2]
        inp, s, xo, yo = letterbox(frame, IMG_SIZE)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)
        outputs = rknn.inference(inputs=[inp], data_format=["nhwc"])
        if not outputs:
            print(f"[NPU-{worker_id}] Inference failed")
            stop(f"NPU-{worker_id} inference failed")
            break
        if len(outputs) != 1:
            print(f"[NPU-{worker_id}] Expected 1 output, got {len(outputs)}")
            stop(f"NPU-{worker_id} invalid output count")
            break
        out = np.asarray(outputs[0])
        normalized_out = (
            out.transpose(0, 2, 1) if out.shape == (1, 6, 300) else out
        )
        if normalized_out.shape != (1, 300, 6):
            print(
                f"[NPU-{worker_id}] Unexpected output shape: "
                f"{normalized_out.shape}"
            )
            stop(f"NPU-{worker_id} invalid output shape")
            break
        max_conf = float(np.max(normalized_out[0, :, 4]))
        confs, boxes_raw = postprocess(out, s, xo, yo, ow, oh)
        balls = []

        if boxes_raw:
            # YOLO26 is end-to-end and already returns the final top detections.
            for box, confidence in zip(boxes_raw, confs):
                x1,y1,x2,y2 = map(int, box)
                cx,cy = (x1+x2)//2, (y1+y2)//2
                r = max((x2-x1)//2, (y2-y1)//2)
                balls.append((x1,y1,x2,y2,cx,cy,r,confidence))

        with result_lock:
            now = time.perf_counter()
            completion_times.append(now)
            while completion_times and completion_times[0] < now - 1.0:
                completion_times.popleft()
            fps = float(len(completion_times))
            if seq > latest_result["seq"]:
                capture_fps = latest_result["capture_fps"]
                latest_result = {
                    "seq": seq,
                    "frame": frame,
                    "balls": balls,
                    "fps": fps,
                    "max_conf": max_conf,
                    "capture_fps": capture_fps,
                    "capture_time": capture_time,
                }

    rknn.release()

# ==================== 主线程: 显示 + 串口 ====================
def main():
    global running

    cv2.setNumThreads(2)

    # 尝试开串口
    uart = None
    try:
        uart = serial.Serial(UART_PORT, UART_BAUD, timeout=0.1)
        print(f"[Main] UART: {UART_PORT} @ {UART_BAUD}")
    except:
        print("[Main] UART off")

    if MJPEG_ENABLED:
        threading.Thread(target=mjpeg_server_thread, daemon=True).start()

    recorder = None
    record_path = None
    last_record_time = 0.0

    # Each RK3588 NPU core uses an independent runtime on a separate frame.
    t_cap = threading.Thread(target=capture_thread, args=(CAMERA_ID,), daemon=True)
    t_cap.start()
    if STREAM_ONLY:
        print("[Main] Stream-only test. Press Ctrl+C to stop")
    else:
        core_masks = [
            RKNNLite.NPU_CORE_0,
            RKNNLite.NPU_CORE_1,
            RKNNLite.NPU_CORE_2,
        ]
        workers = [
            threading.Thread(
                target=inference_thread,
                args=(worker_id, core_masks[worker_id]),
                daemon=True,
            )
            for worker_id in range(NPU_WORKERS)
        ]
        for worker in workers:
            worker.start()
        print(f"[Main] Capture + {NPU_WORKERS} NPU workers. Press ESC to stop")

    # 创建窗口
    if LOCAL_DISPLAY:
        cv2.namedWindow("Ball Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Detection", 960, 540)

    displayed_seq = -1
    while running:
        if STREAM_ONLY:
            time.sleep(0.1)
            continue
        with result_lock:
            seq = latest_result["seq"]
            frame = latest_result["frame"]
            balls = latest_result["balls"]
            fps   = latest_result["fps"]
            max_conf = latest_result["max_conf"]
            capture_fps = latest_result["capture_fps"]
            capture_time = latest_result["capture_time"]

        if frame is None or seq == displayed_seq:
            if LOCAL_DISPLAY and cv2.waitKey(1) & 0xFF == 27:
                stop("ESC pressed")
            time.sleep(0.001)
            continue
        displayed_seq = seq
        frame = frame.copy()
        display_time = time.perf_counter()
        display_times.append(display_time)
        while display_times and display_times[0] < display_time - 1.0:
            display_times.popleft()
        e2e_fps = float(len(display_times))
        latency_ms = (display_time - capture_time) * 1000.0

        # 画检测框
        best = None
        for (x1,y1,x2,y2,cx,cy,r,conf) in balls:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)
            cv2.putText(frame, f"{conf:.2f}", (x1,y1-6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            if best is None or conf > best[3]:
                best = (cx,cy,r,conf)

        # 串口发送
        if uart:
            if best:
                uart.write(f"B{best[0]},{best[1]},{best[2]}\n".encode())
            else:
                uart.write(b"BNO\n")

        # 显示 FPS 和检出数
        cv2.putText(
            frame,
            f"E2E: {e2e_fps:.0f} FPS  Lat: {latency_ms:.1f} ms",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(
            frame,
            f"NPU: {fps:.0f}  CAM: {capture_fps:.0f}  "
            f"Balls: {len(balls)}  Max: {max_conf:.2f}",
            (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2
        )

        publish_stream_frame(frame)

        if RECORD_ENABLED:
            if recorder is None:
                os.makedirs(RECORD_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_path = os.path.join(RECORD_DIR, f"ball_{timestamp}.mp4")
                height, width = frame.shape[:2]
                recorder = cv2.VideoWriter(
                    record_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    RECORD_FPS,
                    (width, height),
                )
                if recorder.isOpened():
                    print(f"[Record] Saving: {record_path}")
                else:
                    print(f"[Record] Cannot open: {record_path}")
                    recorder.release()
                    recorder = None
                    record_path = None
            now = time.perf_counter()
            if recorder is not None and now - last_record_time >= 1.0 / RECORD_FPS:
                recorder.write(frame)
                last_record_time = now

        if LOCAL_DISPLAY:
            cv2.imshow("Ball Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop("ESC pressed")
                break

    # 清理
    with stream_condition:
        stream_condition.notify_all()
    if LOCAL_DISPLAY:
        cv2.destroyAllWindows()
    if recorder is not None:
        recorder.release()
        print(f"[Record] Saved: {record_path}")
    if uart: uart.close()
    print("[Main] Exit")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop("Ctrl+C")
        print("\n[Main] Interrupted")
