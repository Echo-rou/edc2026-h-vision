import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def letterbox(image: np.ndarray, size: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))

    pad_width = size - resized_width
    pad_height = size - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def normalize_output(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output, dtype=np.float32)
    if output.shape == (1, 6, 300):
        output = output.transpose(0, 2, 1)
    if output.shape != (1, 300, 6):
        raise RuntimeError(f"unexpected output shape: {output.shape}")
    return output


def summarize(name: str, output: np.ndarray) -> None:
    detections = output[0]
    scores = detections[:, 4]
    top_index = int(np.argmax(scores))
    top = detections[top_index]
    print(
        f"{name}: shape={output.shape}, dtype={output.dtype}, "
        f"score>0.25={int(np.count_nonzero(scores > 0.25))}, "
        f"top=[{', '.join(f'{value:.6f}' for value in top)}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(letterbox(image), cv2.COLOR_BGR2RGB)
    input_nhwc = np.expand_dims(rgb, axis=0)
    input_nchw = input_nhwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    session = ort.InferenceSession(
        args.onnx, providers=["CPUExecutionProvider"]
    )
    onnx_output = normalize_output(
        session.run(None, {session.get_inputs()[0].name: input_nchw})[0]
    )

    if not Path(args.rknn).is_file():
        raise FileNotFoundError(args.rknn)

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"config failed: {ret}")
        ret = rknn.load_onnx(model=args.onnx)
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        outputs = rknn.inference(
            inputs=[input_nhwc], data_format=["nhwc"]
        )
        if not outputs:
            raise RuntimeError("RKNN returned no outputs")
        rknn_output = normalize_output(outputs[0])
    finally:
        rknn.release()

    summarize("ONNX", onnx_output)
    summarize("RKNN", rknn_output)

    onnx_top = onnx_output[0, int(np.argmax(onnx_output[0, :, 4]))]
    rknn_top = rknn_output[0, int(np.argmax(rknn_output[0, :, 4]))]
    coordinate_error = np.abs(onnx_top[:4] - rknn_top[:4])
    score_error = abs(float(onnx_top[4] - rknn_top[4]))
    class_match = int(round(float(onnx_top[5]))) == int(
        round(float(rknn_top[5]))
    )
    print(
        f"top coordinate max_abs_error={float(coordinate_error.max()):.6f}, "
        f"score_abs_error={score_error:.6f}, class_match={class_match}"
    )

    if not np.isfinite(rknn_output).all():
        raise RuntimeError("RKNN output contains NaN or Inf")
    if not class_match:
        raise RuntimeError("top detection class differs")
    if score_error > 0.05:
        raise RuntimeError("top detection score differs by more than 0.05")
    if float(coordinate_error.max()) > 8.0:
        raise RuntimeError("top detection coordinates differ by more than 8 px")
    print("VALIDATION_PASS")


if __name__ == "__main__":
    main()
