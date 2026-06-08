from ultralytics import YOLO

if __name__=="__main__":
    baseline_str_bbox="runs/detect/runs/detect/v8n_baseline-bbox/weights/best.pt"
    baseline_str_innerbox="runs/detect/runs/detect/v8n_baseline-innerbox/weights/best.pt"
    Exp1_str="runs/detect/runs/detect/v8n_Exp1_lineFull_GhostC2f-bbox/weights/best.pt"
    Exp2_str="runs/detect/runs/detect/v8n_Exp2_NeckOnly_GhostC2f-bbox/weights/best.pt"
    Exp3_str="runs/detect/runs/detect/v8n_Exp3_GhostC2f_SEAtt-bbox/weights/best.pt"
    Exp4_str="runs/detect/runs/detect/v8n_Exp4_NeckOnly-GhostC2f_SEAtt-bbox/weights/best.pt"
    Exp6_str="runs/detect/runs/detect/v8n_Exp6_NeckOnly_GhostC2f-InnerIOU/weights/best.pt"
    model=YOLO(Exp2_str)

    metrics=model.val(data='./data/data.yaml',device=0)