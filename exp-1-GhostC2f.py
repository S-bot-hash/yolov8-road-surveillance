import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/yolov8-GhostC2f.yaml")
    model.load("yolov8n.pt")

    results = model.train(
        data="./data/data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="runs/detect",
        name="v8n_Exp1_lineFull_GhostC2f-bbox",
        exist_ok=True,
    )
