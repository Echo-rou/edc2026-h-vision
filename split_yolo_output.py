from pathlib import Path

import onnx
from onnx import TensorProto, helper


home = Path.home()
source_path = home / "ball_best.onnx"
output_path = home / "ball_best_split.onnx"

model = onnx.load(str(source_path))
graph = model.graph
if len(graph.output) != 1:
    raise RuntimeError(f"expected one output, got {len(graph.output)}")

source_output = graph.output[0].name
producer = next(
    (node for node in graph.node if source_output in node.output),
    None,
)
if producer is None or producer.op_type != "Concat" or len(producer.input) != 2:
    raise RuntimeError("expected final boxes/scores Concat node")

boxes_name, scores_name = producer.input

graph.ClearField("output")
graph.output.extend(
    [
        helper.make_tensor_value_info(
            boxes_name, TensorProto.FLOAT, [1, 4, 8400]
        ),
        helper.make_tensor_value_info(
            scores_name, TensorProto.FLOAT, [1, 1, 8400]
        ),
    ]
)

onnx.checker.check_model(model)
onnx.save(model, str(output_path))
print(f"DONE: {output_path}")
print(f"BOXES: {boxes_name}")
print(f"SCORES: {scores_name}")
