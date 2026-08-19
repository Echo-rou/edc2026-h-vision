#!/usr/bin/env python3
import inspect
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


MODELS = ["ball_best.rknn", "ball_best_int8_split.rknn"]


def letterbox(image, size=640):
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized
    return canvas


def capture_frame():
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
        raise RuntimeError("cannot capture camera frame")
    cv2.imwrite("compare_frame.jpg", frame)
    return frame


def confidence_values(output):
    if output.ndim != 3 or output.shape[0] != 1:
        return None
    if output.shape[1] == 5:
        return output[0, 4, :]
    if output.shape[2] == 5:
        return output[0, :, 4]
    return None


def inspect_model(model_path, input_tensor):
    print(f"\n===== {model_path} =====")
    rknn = RKNNLite(verbose=False)
    try:
        ret = rknn.load_rknn(model_path)
        print("load_rknn:", ret)
        if ret != 0:
            return
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print("init_runtime:", ret)
        if ret != 0:
            return

        print("inference signature:", inspect.signature(rknn.inference))
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[input_tensor], data_format=["nhwc"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"inference: {elapsed_ms:.2f} ms")

        if not outputs:
            print("no outputs")
            return
        for index, output in enumerate(outputs):
            array = np.asarray(output)
            print(
                f"output[{index}]: shape={array.shape}, dtype={array.dtype}, "
                f"min={array.min()}, max={array.max()}, mean={array.mean()}"
            )
            conf = confidence_values(array)
            if conf is None:
                print("confidence: unknown output layout")
                continue
            percentiles = np.percentile(conf.astype(np.float32), [50, 90, 99, 99.9])
            print(
                "confidence: "
                f"max={float(conf.max()):.6f}, "
                f">0.10={int(np.count_nonzero(conf > 0.10))}, "
                f">0.25={int(np.count_nonzero(conf > 0.25))}, "
                f">0.50={int(np.count_nonzero(conf > 0.50))}, "
                f"p50/p90/p99/p99.9={percentiles.tolist()}"
            )
        if len(outputs) == 2:
            combined = np.concatenate(
                (np.asarray(outputs[0]), np.asarray(outputs[1])), axis=1
            )
            conf = combined[0, 4, :]
            print(
                "combined confidence: "
                f"max={float(conf.max()):.6f}, "
                f">0.10={int(np.count_nonzero(conf > 0.10))}, "
                f">0.25={int(np.count_nonzero(conf > 0.25))}, "
                f">0.50={int(np.count_nonzero(conf > 0.50))}"
            )
    finally:
        rknn.release()


def main():
    frame = capture_frame()
    image = letterbox(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(image, axis=0)
    print("input:", input_tensor.shape, input_tensor.dtype)
    print("saved camera frame: compare_frame.jpg")
    for model in MODELS:
        inspect_model(model, input_tensor)


if __name__ == "__main__":
    main()
