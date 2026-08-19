#!/usr/bin/env python3
import time

import cv2


def measure(name, cap, seconds=5.0):
    if not cap.isOpened():
        print(f"{name}: OPEN FAILED")
        return
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        ok, _ = cap.read()
        if ok:
            count += 1
    elapsed = time.perf_counter() - started
    print(f"{name}: {count / elapsed:.2f} FPS ({count} frames)")
    cap.release()


pipeline = (
    "v4l2src device=/dev/video0 io-mode=2 ! "
    "image/jpeg,width=640,height=480,framerate=90/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)
measure("GStreamer MJPG 640x480@90", cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 90)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
measure("OpenCV V4L2 MJPG 640x480@90", cap)
