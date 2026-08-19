#!/usr/bin/env python3
"""Prepare representative 512x512 letterboxed images for RKNN INT8."""
from pathlib import Path

import cv2
import numpy as np


SOURCE_DIR = Path("/mnt/c/Users/baixi/Desktop/new shujuji/2ball")
OUTPUT_DIR = Path.home() / "ball_calibration_512"
DATASET_FILE = Path.home() / "ball_dataset_512.txt"
IMAGE_SIZE = 512
MAX_IMAGES = 200


def find_images(root: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    preferred_roots = [root / "train" / "images", root / "images" / "train"]
    search_root = next(
        (path for path in preferred_roots if path.is_dir()), root
    )
    return sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def letterbox(image):
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8
    )
    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized
    return canvas


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {SOURCE_DIR}")

    image_paths = find_images(SOURCE_DIR)
    if not image_paths:
        raise RuntimeError(f"no calibration images found under: {SOURCE_DIR}")

    # Evenly sample the dataset instead of taking only its first images.
    count = min(len(image_paths), MAX_IMAGES)
    indices = np.linspace(0, len(image_paths) - 1, count, dtype=int)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for output_index, source_index in enumerate(indices):
        image_path = image_paths[int(source_index)]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP: {image_path}")
            continue
        output_path = OUTPUT_DIR / f"cal_{output_index:04d}.jpg"
        if cv2.imwrite(
            str(output_path),
            letterbox(image),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            written.append(output_path)

    if not written:
        raise RuntimeError("no calibration images were written")
    DATASET_FILE.write_text(
        "".join(f"{path}\n" for path in written),
        encoding="utf-8",
    )
    print(f"DONE: {len(written)} images")
    print(f"DATASET: {DATASET_FILE}")


if __name__ == "__main__":
    main()
