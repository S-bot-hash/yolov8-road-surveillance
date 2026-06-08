import os
from collections import defaultdict

# 将路径替换为你的真实训练集标签路径
labels_dir = "./data/train/labels"

# 你的类别名称映射 (请确保与你的 classes.txt 或 yaml 对应)
class_names = {
    0: "Bus",
    1: "Car",
    2: "Motorbike",
    3: "Pickup",
    4: "Truck",
    5: "Van"
}

instances_count = defaultdict(int)
images_with_class = defaultdict(set)

print(f"正在扫描目录: {labels_dir} ...")

for filename in os.listdir(labels_dir):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(labels_dir, filename)
    with open(filepath, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])

            # 统计实例数
            instances_count[cls_id] += 1
            # 统计包含该类别的图片 (利用 set 自动去重)
            images_with_class[cls_id].add(filename)

# 打印最终统计结果
print("\n=== 训练集目标与图片分布统计 ===")
print(f"{'类别ID':<8} | {'类别名称':<12} | {'图片数(Images)':<15} | {'实例数(Instances)':<15}")
print("-" * 55)

for cls_id in sorted(class_names.keys()):
    name = class_names[cls_id]
    img_count = len(images_with_class[cls_id])
    inst_count = instances_count[cls_id]
    print(f"{cls_id:<8} | {name:<12} | {img_count:<15} | {inst_count:<15}")