# 🐛 Silkworm Segmentation & Disease Detection Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Learning framework for **silkworm instance segmentation**, **disease detection**, and **counting** using SOTA architectures (VM-UNet/Mamba, Attention U-Net + Watershed, and Silkynet).

---

## 📌 Features

- 🛰️ **VM-UNet (Mamba State-Space)**: Fast & accurate segmentation using Visual Mamba architecture.
- 🌊 **Deep Watershed**: Attention U-Net boundary prediction + Watershed algorithm to separate touching silkworms (includes an interactive GUI viewer).
- 📦 **Box-Guided Watershed**: Weakly supervised segmentation using bounding box labels.
- 🔍 **Silkynet & Counting**: Contour detection and silkworm counting algorithms.
- 🛠️ **YOLO Converter**: Convert YOLO bounding box datasets to binary masks & boundary maps.

---

## 📁 Repository Structure

```text
Silkworm_Segmentation/
├── methods/
│   ├── vmunet/               # VM-UNet (Mamba-based UNet)
│   ├── deep_watershed/       # Attention U-Net + Watershed Pipeline & Interactive GUI
│   ├── box_guided_watershed/ # Weakly supervised bounding-box segmentation
│   └── silkynet/             # Silkynet model & contour counting scripts
├── utils/
│   └── yolo_bbox_to_masks.py # Converter from YOLO bboxes to mask & boundary maps
├── data/                     # Dataset directory (see DATASET_STRUCTURE.md)
└── DATASET_STRUCTURE.md      # Detailed dataset structure & guidelines
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ngvuthtinh/Silkworm_Segmentation.git
cd Silkworm_Segmentation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision opencv-python matplotlib pillow scikit-image scipy tqdm
```

---

## 🚀 Quick Start

### 1. Convert YOLO Annotations to Masks
```bash
python -m utils.yolo_bbox_to_masks \
    --input-root "data/Silkworm Diseases.v1i.yolo26" \
    --output-root "data/converted"
```

### 2. Run VM-UNet (Mamba)
```bash
# Train VM-UNet
python methods/vmunet/train.py

# Inference & Visualization
python methods/vmunet/predict.py --max-images 20
```

### 3. Run Deep Watershed Pipeline
```bash
# Generate 2px boundary labels from masks
python -m methods.deep_watershed.boundary_pipeline prepare-labels \
    --mask-dir data/masks \
    --output-dir data/boundaries

# Train Attention U-Net boundary model
python -m methods.deep_watershed.boundary_pipeline train \
    --image-dir data/images \
    --boundary-dir data/boundaries \
    --save-path data/checkpoints/boundary_model.pt

# Launch Interactive GUI Viewer
python -m methods.deep_watershed.interactive_viewer \
    --image-dir "data/test/images" \
    --checkpoint data/checkpoints/boundary_model.pt
```

### 4. Run Silkynet & Silkworm Counting
```bash
python methods/silkynet/Count_contours.py
```

---

## 📄 License

This project is licensed under the MIT License.