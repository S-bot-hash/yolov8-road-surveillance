from ultralytics import YOLO

if __name__ == "__main__":
    baseline_str = "runs/detect/runs/detect/v8n_baseline-3/weights/best.pt"
    Exp1_str = "runs/detect/runs/detect/v8n_Exp1_lineFull_GhostC2f/weights/best.pt"
    Exp2_str = "runs/detect/runs/detect/v8n_Exp2-NeckOnly-GhostC2f/weights/best.pt"
    Exp3_str = "runs/detect/runs/detect/v8n_Exp3-GhostC2f-SEAtt/weights/best.pt"

    model = YOLO(Exp3_str)

    metrics = model.val(data="./data/data.yaml", device=0)
