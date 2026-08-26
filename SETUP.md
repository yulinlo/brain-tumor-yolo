# 环境搭建与数据下载指南

## 第一步：安装环境（10分钟）

### 1. 下载安装 Anaconda
- 官网：https://www.anaconda.com/download
- 安装后打开 Anaconda Prompt

### 2. 创建虚拟环境
```bash
conda create -n tumor-yolo python=3.10 -y
conda activate tumor-yolo
```

### 3. 安装 PyTorch
**有NVIDIA显卡的：**
```bash
# 先运行 nvidia-smi 查看支持的CUDA版本，然后安装对应版本
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**没有独立显卡（纯CPU）：**
```bash
pip install torch torchvision torchaudio
```
CPU训练慢，但小数据集也能跑，或者用AutoDL租GPU。

### 4. 安装项目依赖
```bash
cd brain-tumor-yolo
pip install -r requirements.txt
```

### 5. 验证安装
```bash
python -c "from ultralytics import YOLO; print('YOLO安装成功')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')"
```

---

## 第二步：下载数据集（15分钟）

### 推荐方案：Kaggle "MRI for Brain Tumor with Bounding Boxes"
- 地址：https://www.kaggle.com/datasets/ahmedisoroul/mri-for-brain-tumor-with-bounding-boxes
- 5249张MRI图像，4类（glioma/meningioma/pituitary/notumor）
- **已经是YOLO格式标注**，下载后直接能用

### 下载方式

**方式A：网页手动下载（推荐新手）**
1. 注册/登录Kaggle账号
2. 打开上面的链接
3. 点击右上角 "Download" 按钮下载zip
4. 解压后按下面的目录结构整理

**方式B：命令行下载**
```bash
# 先在Kaggle设置页面生成API Token，下载kaggle.json
pip install kaggle
# 将kaggle.json放到 ~/.kaggle/ 目录（Windows: C:\Users\你的用户名\.kaggle\）
kaggle datasets download -d ahmedisoroul/mri-for-brain-tumor-with-bounding-boxes
```

### 数据目录整理

下载解压后，整理成以下结构：
```
brain-tumor-yolo/
└── data/
    ├── train/
    │   ├── images/    # 训练图片 (.jpg)
    │   └── labels/    # 训练标签 (.txt, YOLO格式)
    └── valid/
        ├── images/    # 验证图片
        └── labels/    # 验证标签
```

> **注意**：Kaggle下载的数据集可能已经分好了train/valid文件夹，直接把里面的内容复制到对应目录即可。如果文件名不同（如`labels`可能叫`Label`），重命名为小写的`labels`。

### 检查数据是否正确
```bash
python utils/prepare_data.py --check
```
输出应该显示各类别图片和标签数量，全部✅就对了。

### 可视化检查标注（可选）
```bash
python utils/prepare_data.py --visualize 6
```
会在 `results/annotation_check.png` 生成带标注框的样例图，确认框画对了。

---

## 第三步：没有GPU怎么办（AutoDL方案）

### 租GPU（推荐，总成本约5-10元）
1. 注册 AutoDL：https://www.autodl.com
2. 选择：RTX 3090（24G显存），约1.5元/小时
3. 选择镜像：PyTorch 2.1 + Python 3.10 + CUDA 11.8
4. 开机后用JupyterLab或SSH连接
5. 上传项目代码和数据集
6. 运行训练命令

### 训练时间估算
- YOLOv8n，5000张图，100 epochs，RTX 3090：约 **40-60分钟**
- YOLOv8s，5000张图，100 epochs，RTX 3090：约 **1.5-2小时**

---

## 常见问题

**Q: 报错 CUDA out of memory？**
A: 减小batch size：`--batch 8` 或 `--batch 4`，或者用yolov8n而非yolov8s。

**Q: 训练时mAP一直为0？**
A: 检查data.yaml中的nc和names是否和标签txt中的类别id一致；检查图片和标签是否一一对应；运行`python utils/prepare_data.py --check`。

**Q: Windows下 workers 报错？**
A: 训练时加 `--workers 0`。

**Q: Gradio启动后打不开？**
A: 确认端口7860没被占用；如果是云服务器，需要开放7860端口。

**Q: 模型权重在哪里？**
A: 训练完后在 `runs/detect/实验名/weights/best.pt`。
