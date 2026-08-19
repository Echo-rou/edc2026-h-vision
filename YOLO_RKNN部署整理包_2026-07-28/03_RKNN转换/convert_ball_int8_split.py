from pathlib import Path

from rknn.api import RKNN


home = Path.home()
onnx_file = str(home / "ball_best_split.onnx")
dataset_file = str(home / "ball_dataset_640.txt")
rknn_file = str(home / "ball_best_int8_split.rknn")

rknn = RKNN(verbose=True)

print("--> Config split-output INT8")
ret = rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
    quantized_dtype="asymmetric_quantized-8",
    quantized_algorithm="normal",
    optimization_level=3,
)
if ret != 0:
    raise RuntimeError(f"config failed: {ret}")

print("--> Load split-output ONNX")
ret = rknn.load_onnx(model=onnx_file)
if ret != 0:
    raise RuntimeError(f"load_onnx failed: {ret}")

print("--> Build split-output INT8 RKNN")
ret = rknn.build(do_quantization=True, dataset=dataset_file)
if ret != 0:
    raise RuntimeError(f"build failed: {ret}")

print("--> Export split-output INT8 RKNN")
ret = rknn.export_rknn(rknn_file)
if ret != 0:
    raise RuntimeError(f"export failed: {ret}")

rknn.release()
print(f"DONE: {rknn_file}")
