from pathlib import Path

from rknn.api import RKNN


home = Path.home()
onnx_file = str(home / "ball_best.onnx")
rknn_file = str(home / "ball_best.rknn")

rknn = RKNN(verbose=True)

print("--> Config")
ret = rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
)
if ret != 0:
    raise RuntimeError(f"config failed: {ret}")

print("--> Load ONNX")
ret = rknn.load_onnx(model=onnx_file)
if ret != 0:
    raise RuntimeError(f"load_onnx failed: {ret}")

print("--> Build RKNN")
ret = rknn.build(do_quantization=False)
if ret != 0:
    raise RuntimeError(f"build failed: {ret}")

print("--> Export RKNN")
ret = rknn.export_rknn(rknn_file)
if ret != 0:
    raise RuntimeError(f"export failed: {ret}")

rknn.release()
print(f"DONE: {rknn_file}")
