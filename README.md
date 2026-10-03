# 🐛 Silkworm Joint Multi-Task Segmentation & Disease Detection Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture: 4-Tier](https://img.shields.io/badge/Design-4--Tier%20Modular-success)](PROJECT_STRUCTURE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance deep learning framework for **joint silkworm body segmentation**, **disease classification** (Grasserie vs. Healthy), and **larva counting**. Built on State-of-the-Art architectures (**VM-UNet / Visual Mamba**, **Swin-UNet**, **SAM 3**, and **Silkynet**).

---

## 📌 Features & Key Innovations

- 🔀 **End-to-End Joint Multi-Task Learning**:
  - Shared encoder backbone extracts rich joint representations.
  - **Dual Task Heads**: Reconstructs dense binary body segmentation masks (`[B, 1, H, W]`) while simultaneously predicting disease status (`[B, 2]`) from deep bottleneck features.
  - **Simultaneous Optimization**: Combined multi-task loss $\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \cdot \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}$.
- 🛰️ **VM-UNet (Visual Mamba State-Space Model)**: Fast, long-range receptive field backbone for silkworm body structure.
- 🎯 **SAM 3 Auto-Labeling Engine**: Fast CUDA generation of pixel-accurate masks directly from YOLO bounding boxes using text prompt grounding (`silkworm`).
- 🪟 **Swin-UNet**: Shifted-window Vision Transformer multi-task baseline.
- 🔍 **Silkynet & Counting**: Contour analysis for dense silkworm larvae counting.
- 📁 **Clean 4-Tier Architecture**: Strict modular separation between Upstream Models, Shared Modules, Experiment Wrappers, and Output Runs.

---

## 🏗️ Repository Architecture

The project follows a strict **4-Tier Modular Design** (detailed in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)):

```text
Silkworm_Segmentation/
├── models/                  # [TIER 1] Upstream Original Backbones (Submodules / Read-Only)
│   ├── vmunet/              #   VM-UNet / VMamba backbone repository
│   ├── swin_unet/           #   Swin-UNet backbone repository
│   ├── sam3/                #   SAM 3 backbone repository
│   └── silkynet/            #   Silkynet counting repository & dataset
│
├── src/                     # [TIER 2] Shared Reusable Core Modules
│   ├── dataset.py           #   Single-task DataLoaders (Seg & Cls)
│   ├── dataset_multitask.py #   Joint Multi-Task DataLoaders (SilkwormMultiTaskDataset)
│   ├── losses.py            #   Shared Loss functions (DiceLoss, SegLoss, ClsLoss, JointMultiTaskLoss)
│   ├── metrics.py           #   Evaluation metrics (Dice, IoU, Accuracy, F1, Precision, Recall)
│   └── visualizer.py        #   Prediction overlay and plotting tools
│
├── experiments/             # [TIER 3] Experiment Wrappers & Pipelines
│   ├── multitask_vmunet/    #   ⭐ Primary Joint Multi-Task VM-UNet Experiment
│   │   ├── config.py        #     Hyperparameter dataclass
│   │   ├── model.py         #     Multi-Task wrapper connecting VM-UNet + Dual Heads
│   │   ├── train.py         #     End-to-end multi-task training pipeline
│   │   ├── evaluate.py      #     Evaluation script on test split
│   │   ├── inference.py     #     Inference and visual prediction script
│   │   └── notes.md         #     Experiment logs and observations
│   │
│   ├── multitask_swinunet/  #   Joint Multi-Task Swin-UNet Baseline
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   │
│   ├── segfirst_vmunet/     #   Segmentation-First 2-Phase Staged Baseline
│   └── segfirst_swinunet/   #   Segmentation-First Swin-UNet Baseline
│
├── runs/                    # [TIER 4] Automatically Generated Artifacts (Checkpoints & Logs)
│   ├── multitask_vmunet/    #   Outputs saved per timestamp run
│   ├── multitask_swinunet/  #   Swin-UNet multi-task run outputs
│   └── segfirst_vmunet/     #   Staged baseline run outputs
│
├── data/                    # Datasets (See DATASET_STRUCTURE.md)
│   ├── yolo_bbox/                    # Bounding box & disease labels
│   ├── sam3_seg/                     # SAM 3 generated pseudo-masks
│   ├── silkynet_seg/                 # Silkynet multi-larvae images + masks
│   ├── mixed_10k/                    # Augmented 10k (SAM3 + Silkynet)
│   └── sam3_aug20k/                  # Augmented ~20k (SAM3 only)
│
└── utils/                   # Preprocessing & Auto-labeling utilities
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

# Install required dependencies
pip install -r requirements.txt
```

> ⚠️ **VMamba CUDA Extension**: To run VM-UNet models on GPU, compile the CUDA extension `selective_scan` according to instructions in [`models/vmunet/README.md`](models/vmunet/README.md).

---

## 🚀 Quick Start

### 1. Auto-generate Pseudo-Masks using SAM 3
Generate pixel-accurate binary masks from YOLO bounding boxes:
```bash
python utils/sam3_label_from_yolo.py \
    --input-yaml "data/yolo_bbox/data.yaml" \
    --output-dir "data/sam3_seg" \
    --prompt "silkworm" \
    --device cuda
```

### 2. Train Multi-Task Experiments
Always run training scripts as Python modules from the project root:

- **Train Joint Multi-Task VM-UNet (Primary Architecture)**:
  ```bash
  python -m experiments.multitask_vmunet.train
  ```

- **Train Joint Multi-Task Swin-UNet**:
  ```bash
  python -m experiments.multitask_swinunet.train
  ```

*Checkpoints, CSV logs, TensorBoard logs, and visualization outputs are automatically saved to `runs/<exp_name>/<timestamp>/`.*

### 3. Evaluate & Run Inference
Evaluate a trained checkpoint on the test set:
```bash
python -m experiments.multitask_vmunet.evaluate \
    --checkpoint runs/multitask_vmunet/<run_timestamp>/checkpoints/best_model.pth
```

Run prediction on a directory of unseen images:
```bash
python -m experiments.multitask_vmunet.inference \
    --checkpoint runs/multitask_vmunet/<run_timestamp>/checkpoints/best_model.pth \
    --image-dir data/test/ \
    --output-dir runs/multitask_vmunet/inference_output/
```

### 4. Larva Counting with Silkynet
```bash
python models/silkynet/Count_contours.py
```

---

## 📖 Detailed Documentation

- 📐 [**PROJECT_STRUCTURE.md**](PROJECT_STRUCTURE.md): Detailed 4-Tier design rules, adding new backbones, fallback override logic, and git guidelines.
- 📊 [**DATASET_STRUCTURE.md**](DATASET_STRUCTURE.md): Dataset organization, YOLO annotation specs, SAM 3 pseudo-masks, and multi-task DataLoaders.
- 🤖 [**CLAUDE.md**](CLAUDE.md): Developer & AI Assistant guidelines, mentor instructions, and quick command references.

---

## 📄 License

This project is licensed under the MIT License.