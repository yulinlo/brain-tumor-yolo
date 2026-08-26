"""
YOLOv8 脑肿瘤检测 - 训练脚本
支持模型对比和数据增强对比实验。

用法:
    python train.py --model yolov8n.pt --epochs 100 --batch 16 --imgsz 640
    python train.py --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --augment
    python train.py --model yolov8n.pt --epochs 200 --batch 8 --imgsz 640 --name exp4
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 脑肿瘤检测训练")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="预训练模型: yolov8n.pt(轻量) / yolov8s.pt(标准)")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小（显存不足时调小到8或4）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数（Windows下设为0或2）")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称（用于区分不同实验的输出目录）")
    parser.add_argument("--augment", action="store_true",
                        help="启用增强数据增强（mosaic+mixup+HSV+随机透视）")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备: 0(GPU) / cpu / 0,1(多卡)。不指定则自动检测")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续训练")
    return parser.parse_args()


def get_augment_params():
    """增强数据增强参数（在默认增强基础上额外开启/加强）"""
    return {
        "mosaic": 1.0,       # Mosaic拼接增强概率
        "mixup": 0.3,        # Mixup混合增强概率
        "hsv_h": 0.02,       # 色调增强
        "hsv_s": 0.7,        # 饱和度增强
        "hsv_v": 0.4,        # 明度增强
        "degrees": 10.0,     # 随机旋转角度范围
        "translate": 0.1,    # 随机平移比例
        "scale": 0.5,        # 随机缩放范围
        "shear": 5.0,        # 随机裁剪/剪切角度
        "perspective": 0.001,  # 透视变换
        "flipud": 0.2,       # 上下翻转概率（医学图像慎用，MRI可能不需要）
        "fliplr": 0.5,       # 左右翻转概率
        "copy_paste": 0.1,   # 复制粘贴增强
    }


def main():
    args = parse_args()

    # 生成实验名称
    if args.name is None:
        model_tag = args.model.replace(".pt", "").replace("yolov8", "v8")
        aug_tag = "_aug" if args.augment else ""
        args.name = f"{model_tag}_e{args.epochs}_b{args.batch}{aug_tag}"

    print("=" * 60)
    print(f"实验名称: {args.name}")
    print(f"模型: {args.model}")
    print(f"轮数: {args.epochs} | 批次: {args.batch} | 尺寸: {args.imgsz}")
    print(f"增强数据增强: {'是' if args.augment else '否'}")
    print("=" * 60)

    # 加载模型
    if args.resume:
        model = YOLO("runs/detect/last.pt")
        model.train(resume=True)
        return
    else:
        model = YOLO(args.model)

    # 基础训练参数
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        name=args.name,
        exist_ok=True,
        # 保存策略
        save=True,
        save_period=10,       # 每10轮保存一次checkpoint
        # 优化器
        optimizer="auto",     # 自动选择SGD/AdamW
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率 = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 早停
        patience=20,          # 20轮无提升则停止
        # 验证
        val=True,
        # 设备
        device=args.device,
    )

    # 如果启用增强，加入额外增强参数
    if args.augment:
        train_kwargs.update(get_augment_params())

    # 开始训练
    results = model.train(**train_kwargs)

    # 训练完成后验证
    print("\n" + "=" * 60)
    print("训练完成，在验证集上评估...")
    print("=" * 60)
    metrics = model.val()

    print(f"\n{'=' * 60}")
    print(f"实验 [{args.name}] 结果:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"{'=' * 60}")

    # 保存结果摘要到文件
    summary_path = f"runs/detect/{args.name}/summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"实验: {args.name}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"轮数: {args.epochs}\n")
        f.write(f"批次: {args.batch}\n")
        f.write(f"尺寸: {args.imgsz}\n")
        f.write(f"增强: {args.augment}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n")

    print(f"\n结果已保存到: {summary_path}")
    print(f"最佳权重: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
