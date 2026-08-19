#!/usr/bin/env python3
"""Convert a non-end-to-end YOLO26 ONNX to RK3588 INT8 raw-output RKNN.

Required ONNX output before this script modifies it:
    decoded xywh boxes + sigmoid(class logits), shape [1, 5, 5376]

RKNN outputs created by this script:
    output[0] decoded xywh boxes, shape [1, 4, 5376]
    output[1] raw class logits, shape [1, 1, 5376]

Keeping logits before Sigmoid is important: quantizing the final small
probabilities/TopK output can collapse every confidence value to zero.
"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
from rknn.api import RKNN


HOME = Path.home()
ONNX_FILE = HOME / "ball_yolo26_raw_512.onnx"
SPLIT_ONNX_FILE = HOME / "ball_yolo26_raw_512_split.onnx"
DATASET_FILE = HOME / "ball_dataset_512.txt"
RKNN_FILE = HOME / "ball_yolo26_int8_512_raw.rknn"

EXPECTED_BOXES = [1, 4, 5376]
EXPECTED_LOGITS = [1, 1, 5376]


def require_success(step: str, code: int) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with code {code}")


def tensor_shapes(graph) -> dict[str, list[int | None]]:
    result = {}
    values = list(graph.input) + list(graph.value_info) + list(graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else None)
        result[value.name] = dims
    return result


def producer_map(graph):
    return {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }


def unwrap_passthrough(name, producers):
    """Walk through harmless exporter nodes between Concat and Sigmoid."""
    while True:
        node = producers.get(name)
        if node is None or node.op_type not in {"Identity", "Cast"}:
            return name, node
        name = node.input[0]


def make_raw_output_onnx(source: Path, output: Path) -> None:
    model = shape_inference.infer_shapes(onnx.load(str(source)))
    graph = model.graph
    shapes = tensor_shapes(graph)
    producers = producer_map(graph)

    if len(graph.output) != 1:
        raise RuntimeError(
            "ONNX must have one non-end-to-end YOLO26 output; "
            f"got {len(graph.output)}"
        )

    final_name = graph.output[0].name
    final_node = producers.get(final_name)
    if final_node is None or final_node.op_type != "Concat":
        raise RuntimeError(
            "Final ONNX output is not Concat. Export with "
            "nms=False end2end=False, then run this script."
        )

    axis = next(
        (
            helper.get_attribute_value(attr)
            for attr in final_node.attribute
            if attr.name == "axis"
        ),
        None,
    )
    if axis not in (1, -2) or len(final_node.input) != 2:
        raise RuntimeError(
            "Expected YOLO26 [decoded boxes, sigmoid scores] Concat "
            f"on axis 1; got axis={axis}, inputs={len(final_node.input)}"
        )

    boxes_source = None
    scores_source = None
    for name in final_node.input:
        shape = shapes.get(name)
        if shape == EXPECTED_BOXES:
            boxes_source = name
        elif shape == EXPECTED_LOGITS:
            scores_source = name

    if boxes_source is None or scores_source is None:
        details = ", ".join(
            f"{name}={shapes.get(name)}" for name in final_node.input
        )
        raise RuntimeError(
            "Unexpected 512px YOLO26 output branches. Expected "
            f"{EXPECTED_BOXES} and {EXPECTED_LOGITS}; got {details}"
        )

    sigmoid_input_name, score_node = unwrap_passthrough(
        scores_source, producers
    )
    if score_node is None or score_node.op_type != "Sigmoid":
        raise RuntimeError(
            "Score branch does not end in Sigmoid. The ONNX may not be "
            "a standard Ultralytics YOLO26 non-end-to-end export."
        )
    logits_source = score_node.input[0]
    logits_shape = shapes.get(logits_source)
    if logits_shape != EXPECTED_LOGITS:
        raise RuntimeError(
            f"Unexpected raw-logit shape {logits_shape}; "
            f"expected {EXPECTED_LOGITS}"
        )

    graph.node.extend(
        [
            helper.make_node(
                "Identity",
                inputs=[boxes_source],
                outputs=["yolo26_boxes_xywh"],
                name="YOLO26DecodedBoxesOutput",
            ),
            helper.make_node(
                "Identity",
                inputs=[logits_source],
                outputs=["yolo26_class_logits"],
                name="YOLO26RawLogitsOutput",
            ),
        ]
    )
    graph.ClearField("output")
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                "yolo26_boxes_xywh", TensorProto.FLOAT, EXPECTED_BOXES
            ),
            helper.make_tensor_value_info(
                "yolo26_class_logits", TensorProto.FLOAT, EXPECTED_LOGITS
            ),
        ]
    )

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    print(f"Raw-output ONNX: {output}")
    print(f"  output[0] decoded xywh boxes: {EXPECTED_BOXES}")
    print(f"  output[1] pre-Sigmoid logits: {EXPECTED_LOGITS}")


def validate_inputs() -> None:
    if not ONNX_FILE.is_file():
        raise FileNotFoundError(f"ONNX not found: {ONNX_FILE}")
    if not DATASET_FILE.is_file():
        raise FileNotFoundError(
            f"calibration dataset list not found: {DATASET_FILE}"
        )

    calibration_images = [
        Path(line.strip())
        for line in DATASET_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calibration_images:
        raise RuntimeError(f"calibration list is empty: {DATASET_FILE}")
    missing = [path for path in calibration_images if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} calibration images do not exist; "
            f"first missing file: {missing[0]}"
        )
    print(f"Calibration images: {len(calibration_images)}")


def main() -> None:
    validate_inputs()
    make_raw_output_onnx(ONNX_FILE, SPLIT_ONNX_FILE)

    rknn = RKNN(verbose=True)
    try:
        print("--> Config YOLO26 512 INT8 W8A8 raw outputs")
        require_success(
            "config",
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3588",
                quantized_dtype="asymmetric_quantized-8",
                quantized_algorithm="normal",
                optimization_level=3,
            ),
        )

        print("--> Load raw-output ONNX")
        require_success(
            "load_onnx",
            rknn.load_onnx(model=str(SPLIT_ONNX_FILE)),
        )

        print("--> Build INT8 RKNN")
        require_success(
            "build",
            rknn.build(
                do_quantization=True,
                dataset=str(DATASET_FILE),
            ),
        )

        print("--> Export INT8 RKNN")
        require_success(
            "export_rknn",
            rknn.export_rknn(str(RKNN_FILE)),
        )
    finally:
        rknn.release()

    print(f"DONE: {RKNN_FILE}")


if __name__ == "__main__":
    main()
