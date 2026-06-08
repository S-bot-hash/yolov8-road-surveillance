import os
import cv2
import random
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心配置区
# ==========================================
DATASET_DIR = "data/train"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

# 你要增强的目标类别ID (Bus 是 0，Pickup 是 3，请按需修改)
TARGET_CLASS = 2

# the path of output img
OUTPUT_IMG_DIR = os.path.join(DATASET_DIR, "images_aug")
OUTPUT_LBL_DIR = os.path.join(DATASET_DIR, "labels_aug")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# 目标生成数量
NUM_NEW_SAMPLES = 170

# 碰撞拦截阈值与最大尝试次数
IOU_THRESHOLD = 0.1 #if the actual IoU is bigger than the IOU_THREOLD,it will try again
MAX_ATTEMPTS = 50


# ==========================================
# 2. 底层算子
# ==========================================
def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    _, x1_min, y1_min, x1_max, y1_max = box1
    _, x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0.0
    return inter_area / union_area


def load_yolo_label(label_path, img_w, img_h):
    """读取 YOLO 标签并还原为绝对像素坐标"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            """
            the cx represents the relative position x(after normalization) of the target's center point
            the cy represents the relative position y(after normalization) of the target's center point
            the w is the width of the target(after normalization)
            the h is the height of the targer(after normalization)
            """

            xmin = int((cx - w / 2) * img_w)
            ymin = int((cy - h / 2) * img_h)
            xmax = int((cx + w / 2) * img_w)
            ymax = int((cy + h / 2) * img_h)

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_w, xmax), min(img_h, ymax)

            if xmax > xmin and ymax > ymin:
                boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return boxes


def create_yolo_txt(boxes, output_path, img_w, img_h):
    """将绝对像素坐标保存为 YOLO 格式"""
    with open(output_path, "w") as f:
        for box in boxes:
            cls_id, xmin, ymin, xmax, ymax = box
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# 【已回退至最稳妥的重型读取法】
def extract_target_objects(image_paths):
    """先读图获取真实宽高，再解算坐标，绝不漏掉任何目标"""
    obj_patches = []
    print(f"正在启动重型全库扫描，提取类别 ID={TARGET_CLASS} 的特征矩阵...")

    for img_path in image_paths:
        img_name = Path(img_path).stem
        lbl_path = os.path.join(LABEL_DIR, f"{img_name}.txt")

        if not os.path.exists(lbl_path):
            continue

        # 直接暴力读图获取真实物理尺寸
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        boxes = load_yolo_label(lbl_path, w, h)
        for box in boxes:
            cls_id, xmin, ymin, xmax, ymax = box
            if cls_id == TARGET_CLASS:
                patch = img[ymin:ymax, xmin:xmax]
                if patch.size > 10:
                    obj_patches.append(patch)

    print(f"提取完成！成功获取高纯度特征贴纸: {len(obj_patches)} 个。")
    return obj_patches


def visualize_augmented_image(img, boxes):
    vis_img = img.copy()
    for box in boxes:
        cls_id, xmin, ymin, xmax, ymax = box
        color = (0, 0, 255) if cls_id == TARGET_CLASS else (0, 255, 0)
        thickness = 3 if cls_id == TARGET_CLASS else 1
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(vis_img, str(cls_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img_rgb)
    plt.title(f"Copy-Paste Result (Red: Class {TARGET_CLASS})")
    plt.axis("off")
    plt.show()


# ==========================================
# 3. 主执行流程 (带空间碰撞拦截)
# ==========================================
if __name__ == "__main__":
    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

    patches = extract_target_objects(all_image_paths)
    if not patches:
        raise ValueError(f"严重错误：数据集中不存在类别为 {TARGET_CLASS} 的有效标注！")

    print(f"\n开始进行 Copy-Paste 自适应合成，目标生成数量：{NUM_NEW_SAMPLES}")

    success_count = 0
    visualize_count = 0

    while success_count < NUM_NEW_SAMPLES:
        bg_img_path = random.choice(all_image_paths)
        bg_img_name = Path(bg_img_path).stem
        bg_lbl_path = os.path.join(LABEL_DIR, f"{bg_img_name}.txt")

        bg_img = cv2.imread(bg_img_path)
        if bg_img is None: continue
        bg_h, bg_w = bg_img.shape[:2]

        bg_boxes = load_yolo_label(bg_lbl_path, bg_w, bg_h)
        patch = random.choice(patches)
        patch_h, patch_w = patch.shape[:2]

        # 模拟不同距离的透视缩放
        scale = random.uniform(0.6, 1.4)
        new_w, new_h = int(patch_w * scale), int(patch_h * scale)
        if new_w >= bg_w or new_h >= bg_h or new_w < 5 or new_h < 5:
            continue

        patch_resized = cv2.resize(patch, (new_w, new_h))

        # IoU 碰撞拦截器
        valid_position_found = False
        new_box = None

        for attempt in range(MAX_ATTEMPTS):
            paste_x = random.randint(0, bg_w - new_w)
            paste_y = random.randint(0, bg_h - new_h)

            candidate_box = (TARGET_CLASS, paste_x, paste_y, paste_x + new_w, paste_y + new_h)

            collision = False
            for bg_box in bg_boxes:
                if compute_iou(candidate_box, bg_box) > IOU_THRESHOLD:
                    collision = True
                    break

            if not collision:
                valid_position_found = True
                new_box = candidate_box
                break

        if not valid_position_found:
            continue

        # 像素替换
        aug_img = bg_img.copy()
        aug_img[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = patch_resized

        all_boxes = bg_boxes + [new_box]

        # 保存
        new_img_name = f"aug_cp_{TARGET_CLASS}_{success_count}_{bg_img_name}.jpg"
        new_img_path = os.path.join(OUTPUT_IMG_DIR, new_img_name)
        new_lbl_path = os.path.join(OUTPUT_LBL_DIR, f"aug_cp_{TARGET_CLASS}_{success_count}_{bg_img_name}.txt")

        cv2.imwrite(new_img_path, aug_img)
        create_yolo_txt(all_boxes, new_lbl_path, bg_w, bg_h)

        if visualize_count < 3:
            print(f"展示防碰撞合成效果 [{visualize_count + 1}/3]...")
            visualize_augmented_image(aug_img, all_boxes)
            visualize_count += 1

        success_count += 1
        if success_count % 50 == 0:
            print(f"进度: {success_count} / {NUM_NEW_SAMPLES}")

    print(f"\n合成完毕！请将 {OUTPUT_IMG_DIR} 和 {OUTPUT_LBL_DIR} 中的文件合入原训练集。")