from pathlib import Path

import cv2
import numpy as np


source_dir = Path("/mnt/d/py_pro/27yolo/ball2/images")
output_dir = Path.home() / "ball_calibration_640"
dataset_file = Path.home() / "ball_dataset_640.txt"
output_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    path
    for path in source_dir.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

written = []
for index, image_path in enumerate(image_paths):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"SKIP: {image_path}")
        continue

    height, width = image.shape[:2]
    scale = min(640 / height, 640 / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    x_offset = (640 - new_width) // 2
    y_offset = (640 - new_height) // 2
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized

    output_path = output_dir / f"cal_{index:04d}.jpg"
    if cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        written.append(output_path)

dataset_file.write_text(
    "".join(f"{path}\n" for path in written),
    encoding="utf-8",
)
print(f"DONE: {len(written)} images")
print(f"DATASET: {dataset_file}")
