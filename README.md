# 🐛 Silkworm Segmentation & Disease Detection Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture: 4-Tier](https://img.shields.io/badge/Design-4--Tier%20Modular-success)](PROJECT_STRUCTURE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance deep learning framework for **silkworm instance segmentation**, **disease classification** (Grasserie vs. Healthy), and **larva counting**, combining State-of-the-Art architectures (**VM-UNet / Visual Mamba**, **Swin-UNet**, **SAM 3**, and **Silkynet**).

---

## 📌 Features

- 🛰️ **VM-UNet (Mamba State-Space)**: Ultra-fast, long-range receptive field segmentation using Visual Mamba backbone.
- 🔀 **Two-Phase Multi-Task Learning**: 
  - **Phase 1**: Binary segmentation pre-training on silkworm bodies.
  - **Phase 2**: Joint disease classification fine-tuning with segmentation loss preservation.
- 🎯 **SAM 3 Auto-Labeling Engine**: Fast BFloat16 CUDA generation of pixel-accurate masks directly from YOLO bounding boxes using text prompt grounding (`silkworm`).
- 🪟 **Swin-UNet**: Shifted-window Vision Transformer baseline for dense silkworm segmentation.
- 🔍 **Silkynet & Counting**: Classical and CNN-based contour detection for silkworm density monitoring.
- 📁 **Clean 4-Tier Architecture**: Strict modular separation between Model Backbones, Shared Code, Experiments, and Runs.

---

## 🏗️ Repository Architecture

The project follows a strict **4-Tier Modular Design** (detailed in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)):

```text
Silkworm_Segmentation/
├── models/                  # [TẦNG 1] Kiến trúc model gốc (không sửa trực tiếp)
│   ├── vmunet/              #   VM-UNet / VMamba backbone repo
│   ├── swin_unet/           #   Swin-UNet backbone repo
│   ├── sam3/                #   SAM 3 backbone repo
│   └── silkynet/            #   Silkynet model & data
│
├── src/                     # [TẦNG 2] Code dùng chung (Dataset, Losses, Metrics)
│   ├── dataset.py           #   DataLoaders (SilkynetSegDataset, YOLOClsDataset)
│   ├── losses.py            #   Losses (DiceLoss, SegLoss, ClsLoss, SegFirstMultiTaskLoss)
│   └── metrics.py           #   Metrics (Dice, IoU, Accuracy, F1)
│
├── experiments/             # [TẦNG 3] Code thử nghiệm (Configs + Wrappers)
│   └── segfirst_vmunet/     #   Thử nghiệm VM-UNet Segmentation-First
│       ├── config.py        #     Cấu hình siêu tham số (hyperparameters)
│       ├── model.py         #     Wrapper kết nối models/vmunet với multi-task head
│       ├── train.py         #     Pipeline huấn luyện 2 pha
│       └── notes.md         #     Ghi chú thử nghiệm
│
├── runs/                    # [TẦNG 4] Toàn bộ kết quả tự sinh ra khi train
│   ├── segfirst_vmunet/     #   Checkpoints & logs của VM-UNet (theo timestamp/run)
│   ├── segfirst_swinunet/   #   Checkpoints & logs của Swin-UNet
│   ├── multitask_vmunet/    #   Kết quả huấn luyện multitask
│   └── vmunet_single_task/  #   Kết quả baseline đơn nhiệm
│
├── data/                    # Datasets (xem chi tiết tại DATASET_STRUCTURE.md)
│   ├── Silkworm Diseases.v1i.yolo26/ # YOLO bounding box dataset
│   ├── Silkworm_SAM3_Segmented/      # SAM 3 generated masks
│   └── silkworm_mixed_dataset/       # Mixed multi-task dataset
│
├── methods/                 # Code thử nghiệm & script độc lập
└── utils/                   # Công cụ tiền xử lý & sinh nhãn
    ├── sam3_label_from_yolo.py
    └── yolo_bbox_to_masks.py
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ngvuthtinh/Silkworm_Segmentation.git
cd Silkworm_Segmentation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python matplotlib pillow scikit-image scipy tqdm pyyaml
```

> **Ghi chú về VMamba & Mamba-SSM**: Để chạy mô hình VM-UNet trên GPU, cần cài đặt `selective_scan` theo hướng dẫn tại [models/vmunet](models/vmunet/README.md).

---

## 🚀 Quick Start

### 1. Auto-generate Masks using SAM 3
Tạo mặt nạ phân đoạn chất lượng cao từ nhãn bounding box YOLO:
```bash
python utils/sam3_label_from_yolo.py \
    --input-yaml "data/Silkworm Diseases.v1i.yolo26/data.yaml" \
    --output-dir "data/Silkworm_SAM3_Segmented" \
    --prompt "silkworm"
```

### 2. Huấn luyện SegFirst VM-UNet (Recommended)
Chạy pipeline huấn luyện 2 pha (Phase 1: Segmentation Pretrain → Phase 2: Multi-task Fine-tune):
```bash
python -m experiments.segfirst_vmunet.train
```
*Kết quả checkpoint, log và ảnh sample sẽ tự động được lưu vào `runs/segfirst_vmunet/<timestamp>_runX/`.*

### 3. Đánh giá hoặc Inference
Đánh giá checkpoint đã huấn luyện trên tập test:
```bash
python methods/segfirst_multitask_vmunet/evaluate_test.py \
    --checkpoint runs/segfirst_vmunet/2026-09-17_run4/checkpoints/phase2_best.pth
```

Chạy inference trên thư mục ảnh thực tế:
```bash
python methods/segfirst_multitask_vmunet/inference.py \
    --checkpoint runs/segfirst_vmunet/2026-09-17_run4/checkpoints/phase2_best.pth \
    --image-dir data/test/ \
    --output-dir runs/segfirst_vmunet/inference_output/
```

### 4. Đếm số lượng tằm bằng Silkynet
```bash
python models/silkynet/Count_contours.py
```

---

## 📖 Tài Liệu Tham Khảo Thêm

- 📐 [**PROJECT_STRUCTURE.md**](PROJECT_STRUCTURE.md): Quy tắc thiết kế 4 tầng, cách thêm model mới, cách viết loss/dataloader dùng chung, và quy tắc commit.
- 📊 [**DATASET_STRUCTURE.md**](DATASET_STRUCTURE.md): Chi tiết các tập dữ liệu, định dạng nhãn, số lượng mẫu và cách dùng DataLoaders trong `src/dataset.py`.

---

## 📄 License

This project is licensed under the MIT License.