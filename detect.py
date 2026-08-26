"""
YOLOv8 脑肿瘤检测 - 推理脚本
对单张图片或文件夹进行检测，输出带标注框的结果图。

用法:
    python detect.py --source test.jpg --weights runs/detect/v8n_e100_b16/weights/best.pt
    python detect.py --source test_images/ --weights best.pt --conf 0.5 --save-txt
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2


# 类别名称与颜色（BGR格式）
CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "pituitary", 3: "notumor"}
CLASS_COLORS = {
    0: (0, 0, 255),    # 红色 - 神经胶质瘤
    1: (0, 200, 0),    # 绿色 - 脑膜瘤
    2: (255, 100, 0),  # 蓝色 - 垂体瘤
    3: (128, 128, 128),# 灰色 - 无肿瘤
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 脑肿瘤检测推理")
    parser.add_argument("--source", type=str, required=True,
                        help="输入图片路径或文件夹路径")
    parser.add_argument("--weights", type=str, required=True,
                        help="模型权重路径 (best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS的IoU阈值")
    parser.add_argument("--save-dir", type=str, default="results/predictions",
                        help="结果保存目录")
    parser.add_argument("--save-txt", action="store_true",
                        help="同时保存检测结果为txt文件")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: 0 / cpu")
    return parser.parse_args()


def draw_detections(image, boxes, class_ids, confidences):
    """在图像上绘制检测框和标签"""
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        # 画框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 画标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    model = YOLO(args.weights)

    # 收集图片
    source = Path(args.source)
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"输入路径不存在: {args.source}")

    print(f"共找到 {len(image_paths)} 张图片")

    for img_path in image_paths:
        # 推理
        results = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        result = results[0]
        image = cv2.imread(str(img_path))

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            image = draw_detections(image, boxes, class_ids, confidences)

            # 打印检测结果
            print(f"\n[{img_path.name}] 检测到 {len(boxes)} 个目标:")
            for box, cls_id, conf in zip(boxes, class_ids, confidences):
                print(f"  {CLASS_NAMES.get(cls_id, cls_id)}: {conf:.3f}")

            # 保存txt
            if args.save_txt:
                txt_path = Path(args.save_dir) / f"{img_path.stem}.txt"
                h, w = image.shape[:2]
                with open(txt_path, "w") as f:
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        # 转换为YOLO格式: class x_center y_center width height (归一化)
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
        else:
            print(f"\n[{img_path.name}] 未检测到肿瘤")

        # 保存结果图
        save_path = Path(args.save_dir) / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(save_path), image)
        print(f"  结果已保存: {save_path}")

    print(f"\n全部完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
