#!/usr/bin/env python3
"""Orange Pi 5 GHz low-latency UDP vision sender."""

import argparse
import os
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np

PACKET_MAGIC = b"BV5G"
PACKET_HEADER = struct.Struct("!4sIQHH")
MAX_DATAGRAM = 1200
MAX_PAYLOAD = MAX_DATAGRAM - PACKET_HEADER.size
DISCOVERY_MAGIC = b"BV5G_HELLO"


def parse_args():
    parser = argparse.ArgumentParser(description="Orange Pi UDP video sender")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--discovery-port", type=int, default=5601)
    parser.add_argument("--stream-fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def open_camera(camera_id):
    pipeline = (
        f"v4l2src device=/dev/video{camera_id} io-mode=2 ! "
        "image/jpeg,width=640,height=480,framerate=90/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[Camera] GStreamer 640x480@90")
        return cap

    cap.release()
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{camera_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 90)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[Camera] V4L2 fallback")
    return cap


def letterbox(image, size=640):
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_height, new_width = int(height * scale), int(width * scale)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    resized = image if (new_width, new_height) == (width, height) else cv2.resize(
        image, (new_width, new_height)
    )
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return canvas, scale, x_offset, y_offset


class BallDetector:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Cannot load RKNN model: {model_path}")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("Cannot initialize RKNN runtime")
        print("[NPU] Ready on core 0")

    def close(self):
        self.rknn.release()

    def detect(self, frame):
        image_height, image_width = frame.shape[:2]
        padded, scale, x_offset, y_offset = letterbox(frame)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(
            inputs=[np.expand_dims(rgb, axis=0)], data_format=["nhwc"]
        )
        if len(outputs) == 2:
            output = np.concatenate((outputs[0], outputs[1]), axis=1)
        else:
            output = outputs[0]

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
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidence.tolist(), 0.5, 0.45)
        if len(indices) == 0:
            return []
        return [
            (int(x1[index]), int(y1[index]), int(x2[index]), int(y2[index]), confidence[index])
            for index in indices.flatten()
        ]


def draw_overlay(frame, boxes, fps, timestamp_ms):
    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ball {confidence:.2f}", (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(
        frame, f"TX {fps:.1f} FPS  T {timestamp_ms}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
    )


def send_frame(sock, endpoint, frame_id, timestamp_ms, frame, quality):
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return
    data = encoded.tobytes()
    packet_count = (len(data) + MAX_PAYLOAD - 1) // MAX_PAYLOAD
    for packet_index in range(packet_count):
        start = packet_index * MAX_PAYLOAD
        payload = data[start:start + MAX_PAYLOAD]
        header = PACKET_HEADER.pack(
            PACKET_MAGIC, frame_id, timestamp_ms, packet_index, packet_count
        )
        sock.sendto(header + payload, endpoint)


def find_model(requested):
    candidates = []
    if requested:
        candidates.append(Path(requested))
    candidates.extend(
        [
            Path.cwd() / "ball_best_int8_split.rknn",
            Path(__file__).resolve().parent / "ball_best_int8_split.rknn",
            Path.home() / "ball_best_int8_split.rknn",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "ball_best_int8_split.rknn was not found in the current directory, "
        "the script directory, or the user's home directory"
    )


def main():
    args = parse_args()
    if args.display and not os.environ.get("DISPLAY"):
        raise RuntimeError("--display requires an Orange Pi local desktop terminal")

    cap = open_camera(args.camera)
    detector = None if args.no_detect else BallDetector(find_model(args.model))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    discovery = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    discovery.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    discovery.bind(("0.0.0.0", args.discovery_port))
    discovery.setblocking(False)
    endpoint = None
    frame_interval = 1.0 / max(args.stream_fps, 1.0)
    last_send = 0.0
    frame_id = 0
    fps_times = []

    print(f"[UDP] Waiting for receiver discovery on port {args.discovery_port}")
    print("[Main] Press Ctrl+C to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                message, receiver_address = discovery.recvfrom(128)
                if message == DISCOVERY_MAGIC:
                    new_endpoint = (receiver_address[0], args.port)
                    if new_endpoint != endpoint:
                        endpoint = new_endpoint
                        print(f"[UDP] Receiver: {endpoint[0]}:{endpoint[1]}")
            except BlockingIOError:
                pass
            timestamp_ms = int(time.time() * 1000)
            boxes = [] if detector is None else detector.detect(frame)
            now = time.perf_counter()
            fps_times.append(now)
            while fps_times and fps_times[0] < now - 1.0:
                fps_times.pop(0)
            draw_overlay(frame, boxes, len(fps_times), timestamp_ms)

            if endpoint is not None and now - last_send >= frame_interval:
                send_frame(sock, endpoint, frame_id, timestamp_ms, frame, args.jpeg_quality)
                frame_id = (frame_id + 1) & 0xFFFFFFFF
                last_send = now

            if args.display:
                cv2.imshow("Ball Vision 5G - Local", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        sock.close()
        discovery.close()
        if detector is not None:
            detector.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
