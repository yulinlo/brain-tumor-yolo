"""
数据集准备辅助工具
功能：
  1. 检查data目录结构是否正确
  2. 统计各类别样本数量
  3. 划分训练/验证集（如果只有一个all文件夹）
  4. 可视化检查标注是否正确

用法:
    python utils/prepare_data.py --check
    python utils/prepare_data.py --split data/all_images --ratio 0.8
    python utils/prepare_data.py --visualize 5
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_COLORS = [(255, 0, 0), (0, 200, 0), (0, 100, 255)]


def check_structure(data_dir="data"):
    """检查数据集目录结构"""
    data_path = Path(data_dir)
    required = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
    ]

    print("检查数据集结构...")
    all_ok = True
    for folder in required:
        p = data_path / folder
        exists = p.exists()
        count = len(list(p.iterdir())) if exists and p.is_dir() else 0
        status = "✅" if exists and count > 0 else "❌"
        print(f"  {status} {folder}: {count} 个文件")
        if not exists or count == 0:
            all_ok = False

    if all_ok:
        # 统计图片和标签是否一一对应
        for split in ["train", "valid"]:
            img_dir = data_path / split / "images"
            lbl_dir = data_path / split / "labels"
            imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
            lbls = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}

            missing_lbl = imgs - lbls
            missing_img = lbls - imgs
            if missing_lbl:
                print(f"  ⚠️ {split}集有 {len(missing_lbl)} 张图没有标签")
            if missing_img:
                print(f"  ⚠️ {split}集有 {len(missing_img)} 个标签没有对应图片")
            if not missing_lbl and not missing_img:
                print(f"  ✅ {split}集图片与标签一一对应 ({len(imgs)}对)")

        # 统计类别分布
        count_classes(data_path)
    else:
        print("\n数据集不完整。请从Roboflow或Kaggle下载YOLO格式数据集并解压到data/目录。")

    return all_ok


def count_classes(data_path):
    """统计各类别标注数量"""
    print("\n类别分布统计:")
    counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for split in ["train", "valid"]:
        lbl_dir = data_path / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob("*.txt"):
            with open(txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in counts:
                            counts[cls_id] += 1

    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[cls_id]} 个标注")


def split_dataset(source_dir, output_dir="data", ratio=0.8):
    """将一个文件夹的图片和标签按比例划分为训练集和验证集"""
    source = Path(source_dir)
    if not source.exists():
        print(f"源目录不存在: {source}")
        return

    # 找图片和对应标签
    img_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        img_files.extend(source.glob(f"*{ext}"))
        img_files.extend(source.glob(f"*{ext.upper()}"))

    pairs = []
    for img in img_files:
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        print("未找到图片-标签对。请确保.txt文件和图片在同一目录且同名。")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * ratio)
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    for split_name, split_pairs in [("train", train_pairs), ("valid", valid_pairs)]:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img, lbl in split_pairs:
            shutil.copy2(img, output_dir / split_name / "images" / img.name)
            shutil.copy2(lbl, output_dir / split_name / "labels" / lbl.name)

    print(f"划分完成: 训练集 {len(train_pairs)} 对, 验证集 {len(valid_pairs)} 对")


def visualize_annotations(num_samples=6, data_dir="data", output="results/annotation_check.png"):
    """可视化检查标注是否正确"""
    data_path = Path(data_dir)
    img_dir = data_path / "train" / "images"
    lbl_dir = data_path / "train" / "labels"

    if not img_dir.exists():
        print("训练集图片目录不存在，请先准备数据。")
        return

    imgs = list(img_dir.iterdir())[:num_samples * 3]
    random.shuffle(imgs)
    samples = imgs[:num_samples]

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for idx, img_path in enumerate(samples):
        ax = axes[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, CLASS_NAMES[cls_id % len(CLASS_NAMES)],
                                (x1, max(y1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ax.imshow(img)
        ax.set_title(img_path.name[:30], fontsize=9)
        ax.axis("off")

    for idx in range(len(samples), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"标注可视化已保存: {output}")


def main():
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument("--check", action="store_true", help="检查数据集结构")
    parser.add_argument("--split", type=str, default=None, help="要划分的源目录")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--visualize", type=int, default=0, help="可视化N张训练样本")
    args = parser.parse_args()

    if args.check or (not args.split and args.visualize == 0):
        check_structure()

    if args.split:
        split_dataset(args.split, ratio=args.ratio)

    if args.visualize > 0:
        visualize_annotations(args.visualize)


if __name__ == "__main__":
    main()
