#!/usr/bin/env python3
"""Receive the Orange Pi MJPEG stream and save each test on the PC."""

import argparse
import os
import time
from datetime import datetime

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Ball Vision receiver and recorder")
    parser.add_argument(
        "--url",
        default="http://10.79.173.31:8080/stream.mjpg",
        help="Orange Pi MJPEG stream URL",
    )
    parser.add_argument("--output", default="received_recordings")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {args.url}")

    writer = None
    output_path = None
    last_write = 0.0
    frame_interval = 1.0 / max(args.fps, 1.0)

    print(f"[Receiver] Connected: {args.url}")
    print("[Receiver] Press ESC to stop and finish the recording")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Receiver] Stream interrupted")
                break

            if writer is None:
                height, width = frame.shape[:2]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    args.output, f"ball_test_{timestamp}.mp4"
                )
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {output_path}")
                print(f"[Receiver] Recording: {os.path.abspath(output_path)}")

            now = time.perf_counter()
            if now - last_write >= frame_interval:
                writer.write(frame)
                last_write = now

            cv2.imshow("Ball Vision - Receiving and Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Receiver] Saved: {os.path.abspath(output_path)}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
