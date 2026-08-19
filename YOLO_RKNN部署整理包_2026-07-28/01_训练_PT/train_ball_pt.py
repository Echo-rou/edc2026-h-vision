from pathlib import Path

from ultralytics import YOLO


DATASET_ROOT = Path(r"D:\py_pro\27yolo\ball2")
DATA_CONFIG = Path(__file__).with_name("ball2_data.yaml")
PROJECT_DIR = DATASET_ROOT / ".avls" / "runs"


def main():
    image_count = len(list((DATASET_ROOT / "images").glob("*")))
    label_count = len(list((DATASET_ROOT / "labels").glob("*.txt")))
    if image_count == 0 or label_count == 0:
        raise RuntimeError("dataset images or labels are missing")

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Images: {image_count}, labels: {label_count}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        patience=100,
        batch=16,
        imgsz=640,
        device=0,
        workers=2,
        project=str(PROJECT_DIR),
        name="quick_train_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        seed=0,
        deterministic=True,
        close_mosaic=10,
        plots=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"DONE: {best_model}")


if __name__ == "__main__":
    main()
