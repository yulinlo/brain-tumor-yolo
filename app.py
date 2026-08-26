"""
YOLOv8 脑肿瘤检测 - Gradio可视化演示
启动后在浏览器上传MRI图片即可查看检测结果。

用法:
    python app.py
    python app.py --weights runs/detect/v8n_e100_b16/weights/best.pt --port 7860
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


# 类别名称与颜色（RGB格式，Gradio用RGB）
CLASS_NAMES = {
    0: "Glioma（神经胶质瘤）",
    1: "Meningioma（脑膜瘤）",
    2: "Pituitary（垂体瘤）",
    3: "No Tumor（无肿瘤）",
}
CLASS_COLORS_RGB = {
    0: (255, 0, 0),    # 红
    1: (0, 200, 0),    # 绿
    2: (0, 100, 255),  # 蓝
    3: (160, 160, 160),# 灰
}


def parse_args():
    parser = argparse.ArgumentParser(description="脑肿瘤检测Gradio演示")
    parser.add_argument("--weights", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口")
    parser.add_argument("--share", action="store_true",
                        help="生成公网可访问链接")
    return parser.parse_args()


# 全局模型（启动时加载一次）
MODEL = None


def load_model(weights_path):
    global MODEL
    if not Path(weights_path).exists():
        print(f"[警告] 权重文件不存在: {weights_path}")
        print("请先训练模型，或通过 --weights 指定正确路径。")
        MODEL = None
        return
    MODEL = YOLO(weights_path)
    print(f"模型加载成功: {weights_path}")


def detect_image(image, conf_threshold, iou_threshold):
    """Gradio推理函数：接收numpy数组（RGB），返回标注后的图像和检测信息"""
    if MODEL is None:
        return image, "⚠️ 模型未加载。请先训练模型并指定正确的权重路径。"

    if image is None:
        return None, "请上传一张MRI图片。"

    # Gradio传入RGB，OpenCV需要BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = MODEL.predict(
        image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    result = results[0]
    detections_info = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            color_bgr = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))[::-1]
            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

            # 画框
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, 2)

            # 标签
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color_bgr, -1)
            cv2.putText(image_bgr, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            detections_info.append(f"• {label} — 置信度 {conf:.1%}，位置 [{x1}, {y1}, {x2}, {y2}]")

        info = f"✅ 检测到 {len(boxes)} 个病灶：\n" + "\n".join(detections_info)
    else:
        info = "✅ 未检测到明显肿瘤病灶。（注意：本系统仅供学习演示，不能替代专业医学诊断）"

    # 转回RGB给Gradio显示
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, info


def build_demo():
    """构建Gradio界面"""
    with gr.Blocks(
        title="脑肿瘤MRI检测系统",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 900px;}"
    ) as demo:
        gr.Markdown(
            """
            # 🧠 基于YOLOv8的脑肿瘤MRI检测系统
            上传脑部MRI图像，自动检测并定位神经胶质瘤、脑膜瘤、垂体瘤。
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="上传MRI图像",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=400,
                )
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="置信度阈值",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                        label="NMS IoU阈值",
                    )
                detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(
                    label="检测结果",
                    type="numpy",
                    height=400,
                )
                output_text = gr.Textbox(
                    label="检测详情",
                    lines=8,
                    interactive=False,
                )

        gr.Markdown(
            """
            ---
            ⚠️ **免责声明**：本系统仅用于计算机视觉学术研究与技术演示，检测结果不能作为医学诊断依据。
            如有健康疑虑请咨询专业医生。

            **检测类别**：
            - 🔴 Glioma（神经胶质瘤）
            - 🟢 Meningioma（脑膜瘤）
            - 🔵 Pituitary（垂体瘤）
            """
        )

        detect_btn.click(
            fn=detect_image,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_text],
        )

    return demo


def main():
    args = parse_args()
    load_model(args.weights)

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
