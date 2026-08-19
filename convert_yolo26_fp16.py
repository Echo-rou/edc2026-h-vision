from pathlib import Path

from rknn.api import RKNN


HOME = Path.home()
ONNX_FILE = HOME / "ball_yolo26_single.onnx"
RKNN_FILE = HOME / "ball_yolo26_single_fp16.rknn"


def require_success(step: str, code: int) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with code {code}")


def main() -> None:
    if not ONNX_FILE.is_file():
        raise FileNotFoundError(ONNX_FILE)

    rknn = RKNN(verbose=True)
    try:
        print("--> Config RK3588 FP16")
        require_success(
            "config",
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3588",
                optimization_level=3,
            ),
        )

        print("--> Load YOLO26 ONNX")
        require_success("load_onnx", rknn.load_onnx(model=str(ONNX_FILE)))

        print("--> Build RKNN")
        require_success("build", rknn.build(do_quantization=False))

        print("--> Export RKNN")
        require_success("export_rknn", rknn.export_rknn(str(RKNN_FILE)))
    finally:
        rknn.release()

    print(f"DONE: {RKNN_FILE}")


if __name__ == "__main__":
    main()
