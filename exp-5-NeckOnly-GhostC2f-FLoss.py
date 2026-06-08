#仅对neck层中的c2f模块进行替换

from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == '__main__':
    model=YOLO("ultralytics/cfg/models/v8/yolov8-NeckOnly-GhostC2f.yaml")
    model.load("yolov8n.pt")

    resluts=model.train(
        data='./data/data.yaml',
        epochs=120,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project='runs/detect',
        name='v8n_Exp2_NeckOnly_GhostC2f-bbox',
        exist_ok=True
    )