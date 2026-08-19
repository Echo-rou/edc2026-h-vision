#!/usr/bin/env python3
"""Validate the YOLO26 512 INT8 raw-output RKNN on one image."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

import camera_detect_yolo26_int8_highfps as detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--output", default="/home/orangepi/int8_raw_test_result.jpg"
    )
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    input_tensor, scale, pad_x, pad_y = detector.letterbox(image)
    print(
        "INPUT:",
        input_tensor.shape,
        input_tensor.dtype,
        int(input_tensor.min()),
        int(input_tensor.max()),
    )

    rknn = RKNNLite(verbose=False)
    try:
        code = rknn.load_rknn(args.model)
        print("load_rknn:", code)
        if code != 0:
            raise RuntimeError("load_rknn failed")
        code = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print("init_runtime:", code)
        if code != 0:
            raise RuntimeError("init_runtime failed")
        outputs = rknn.inference(
            inputs=[input_tensor], data_format=["nhwc"]
        )
    finally:
        rknn.release()

    if not outputs:
        raise RuntimeError("RKNN returned no outputs")
    print("OUTPUT COUNT:", len(outputs))
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        print(
            f"OUTPUT[{index}]: shape={array.shape} dtype={array.dtype} "
            f"min={float(array.min()):.6f} "
            f"max={float(array.max()):.6f}"
        )

    _, logits = detector.normalize_outputs(outputs)
    confidence = detector.stable_sigmoid(logits)
    print("MAX RAW LOGIT:", float(logits.max()))
    print("MAX CONFIDENCE AFTER SIGMOID:", float(confidence.max()))

    detector.CONF_THRESH = args.conf
    balls, max_conf = detector.postprocess(
        outputs,
        scale,
        pad_x,
        pad_y,
        image.shape[1],
        image.shape[0],
    )
    print(f"DETECTIONS AFTER NMS ABOVE {args.conf}:", len(balls))
    print("MAX CONFIDENCE:", max_conf)

    result = image.copy()
    for x1, y1, x2, y2, cx, cy, radius, score in balls:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(
            result,
            f"ball {score:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if not cv2.imwrite(args.output, result):
        raise RuntimeError(f"failed to save {args.output}")
    print("SAVED:", args.output)


if __name__ == "__main__":
    main()
