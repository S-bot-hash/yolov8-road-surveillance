#用于训练基线模型代码
import os

from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ =='__main__':
#初始化骨架模型
#传入yolov8n.yaml初始化配置，让系统去解析官方原生的yolov8的拓扑连线图
#启动nano的缩放因子：属于yolov8n的depth_factor和width_factor
    model=YOLO('yolov8n.yaml')

#启动基线模型开始训练

    model.train(
        data='./data/data.yaml', #指向数据集data.yaml文件的相对路径
        epochs=120, #完整的训练循环
        batch=16,# 每批次喂给显卡16张图
        imgsz=640,#输入提箱分辨率同一缩放为640x640
        workers=4,#多线程并行读取图片
        device=0,#代表调用电脑里的第一块英伟达显卡
        project='runs/detect',#结果保存的总主目录
        name='v8n_baseline',#本次基线训练结果的子文件夹名字
        exist_ok=True #如果文件夹已经存在，直接覆盖

    )