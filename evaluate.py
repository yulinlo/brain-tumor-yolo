"""
YOLOv8 脑肿瘤检测 - 评估与对比可视化
对比不同实验的结果，生成对比表格和图表。

用法:
    python evaluate.py
    python evaluate.py --exps runs/detect/v8n_e100_b16 runs/detect/v8s_e100_b16
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="实验对比评估")
    parser.add_argument("--exps", nargs="+", default=None,
                        help="实验目录列表，不指定则自动扫描runs/detect/")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="数据集配置文件")
    parser.add_argument("--output", type=str, default="results",
                        help="图表输出目录")
    return parser.parse_args()


def load_exp_results(exp_dir):
    """从训练输出目录加载结果"""
    exp_path = Path(exp_dir)
    summary_file = exp_path / "summary.txt"
    results_csv = exp_path / "results.csv"

    info = {"name": exp_path.name}

    # 从summary.txt读取最终指标
    if summary_file.exists():
        with open(summary_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    try:
                        info[key] = float(val)
                    except ValueError:
                        info[key] = val

    # 从results.csv读取训练曲线数据
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        info["curve"] = {
            "epoch": df["epoch"].values if "epoch" in df.columns else np.arange(len(df)),
            "map50": df["metrics/mAP50(B)"].values if "metrics/mAP50(B)" in df.columns else None,
            "map": df["metrics/mAP50-95(B)"].values if "metrics/mAP50-95(B)" in df.columns else None,
            "precision": df["metrics/precision(B)"].values if "metrics/precision(B)" in df.columns else None,
            "recall": df["metrics/recall(B)"].values if "metrics/recall(B)" in df.columns else None,
            "box_loss": df["train/box_loss"].values if "train/box_loss" in df.columns else None,
        }
    return info


def auto_discover_experiments():
    """自动发现runs/detect下的实验目录"""
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        return []
    return [str(d) for d in sorted(runs_dir.iterdir())
            if d.is_dir() and (d / "weights").exists()]


def plot_comparison_bar(exp_data, output_dir):
    """绘制指标对比柱状图"""
    metrics = ["map50", "map", "precision", "recall"]
    metric_labels = ["mAP50", "mAP50-95", "Precision", "Recall"]

    names = [e["name"] for e in exp_data]
    values = {m: [e.get(m, 0) for e in exp_data] for m in metrics}

    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        offset = (i - len(names) / 2 + 0.5) * width
        vals = [values[m][i] for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("YOLOv8 Brain Tumor Detection - Experiment Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "comparison_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar chart saved: {path}")


def plot_training_curves(exp_data, output_dir):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    curve_configs = [
        ("map50", "mAP50", axes[0, 0]),
        ("map", "mAP50-95", axes[0, 1]),
        ("precision", "Precision", axes[1, 0]),
        ("recall", "Recall", axes[1, 1]),
    ]

    for exp in exp_data:
        if "curve" not in exp:
            continue
        curve = exp["curve"]
        epochs = curve["epoch"]
        for key, label, ax in curve_configs:
            if curve.get(key) is not None:
                ax.plot(epochs, curve[key], label=exp["name"], linewidth=1.5)

    for key, label, ax in curve_configs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Curves Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")


def plot_loss_curves(exp_data, output_dir):
    """绘制loss下降曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for exp in exp_data:
        if "curve" in exp and exp["curve"].get("box_loss") is not None:
            ax.plot(exp["curve"]["epoch"], exp["curve"]["box_loss"],
                    label=exp["name"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Box Loss")
    ax.set_title("Training Loss Curves Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "loss_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curves saved: {path}")


def print_comparison_table(exp_data):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print(f"{'Experiment':<30} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 80)
    for exp in exp_data:
        name = exp.get("name", "unknown")[:30]
        map50 = exp.get("map50", "N/A")
        map_ = exp.get("map", "N/A")
        prec = exp.get("precision", "N/A")
        rec = exp.get("recall", "N/A")

        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{name:<30} {fmt(map50):>8} {fmt(map_):>10} {fmt(prec):>10} {fmt(rec):>8}")
    print("=" * 80)

    # 保存为CSV
    rows = []
    for exp in exp_data:
        rows.append({
            "experiment": exp.get("name", ""),
            "mAP50": exp.get("map50", None),
            "mAP50-95": exp.get("map", None),
            "Precision": exp.get("precision", None),
            "Recall": exp.get("recall", None),
            "model": exp.get("模型", ""),
            "epochs": exp.get("轮数", ""),
            "augment": exp.get("增强", ""),
        })
    df = pd.DataFrame(rows)
    csv_path = Path("results/comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nComparison table saved: {csv_path}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 发现实验
    if args.exps:
        exp_dirs = args.exps
    else:
        exp_dirs = auto_discover_experiments()

    if not exp_dirs:
        print("No experiment results found. Please run train.py first.")
        print("Example: python train.py --model yolov8n.pt --epochs 100")
        return

    print(f"Found {len(exp_dirs)} experiments:")
    for d in exp_dirs:
        print(f"  - {d}")

    # 加载所有实验结果
    exp_data = [load_exp_results(d) for d in exp_dirs]

    # 打印表格
    print_comparison_table(exp_data)

    # 生成图表
    plot_comparison_bar(exp_data, args.output)
    plot_training_curves(exp_data, args.output)
    plot_loss_curves(exp_data, args.output)

    print(f"\nAll results saved to {args.output}/ directory")


if __name__ == "__main__":
    main()
