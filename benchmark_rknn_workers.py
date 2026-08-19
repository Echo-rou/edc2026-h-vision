#!/usr/bin/env python3
import threading
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


MODEL = "ball_best_int8_split.rknn"
SECONDS = 5.0


def capture_input():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise RuntimeError("cannot open /dev/video0")
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            frame = None
            break
    cap.release()
    if frame is None:
        raise RuntimeError("camera capture failed")

    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[80:560] = frame
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)


def run_case(worker_count, input_tensor):
    cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    runtimes = []
    try:
        for index in range(worker_count):
            rknn = RKNNLite(verbose=False)
            if rknn.load_rknn(MODEL) != 0:
                raise RuntimeError(f"worker {index}: load failed")
            if rknn.init_runtime(core_mask=cores[index]) != 0:
                raise RuntimeError(f"worker {index}: init failed")
            runtimes.append(rknn)

        stop_at = time.perf_counter() + SECONDS
        counts = [0] * worker_count
        errors = []

        def worker(index):
            try:
                while time.perf_counter() < stop_at:
                    outputs = runtimes[index].inference(
                        inputs=[input_tensor], data_format=["nhwc"]
                    )
                    if not outputs:
                        raise RuntimeError("inference returned no output")
                    counts[index] += 1
            except Exception as error:
                errors.append(f"worker {index}: {error}")

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started

        if errors:
            raise RuntimeError("; ".join(errors))
        total = sum(counts)
        print(
            f"workers={worker_count}: {total / elapsed:.2f} FPS "
            f"(counts={counts}, elapsed={elapsed:.2f}s)"
        )
    finally:
        for rknn in runtimes:
            rknn.release()


def main():
    input_tensor = capture_input()
    print("Model:", MODEL)
    print("Input:", input_tensor.shape, input_tensor.dtype)
    for worker_count in (1, 2, 3):
        run_case(worker_count, input_tensor)
        time.sleep(1)


if __name__ == "__main__":
    main()
