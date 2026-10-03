# 🐛 CLAUDE.md — Silkworm Multi-Task Segmentation & Disease Detection Framework

This document serves as the primary guidance file for AI Assistants (Claude / Antigravity) and developers working on the **Silkworm Segmentation & Disease Detection** repository.

---

## 🎯 1. Project Goal & Overview

- **Primary Goal**: A high-performance **End-to-End Joint Multi-Task Deep Learning Framework** designed to perform two simultaneous tasks from a single silkworm image:
  1. **Binary Body Segmentation**: Predicting pixel-level masks of silkworm bodies (`[B, 1, H, W]`).
  2. **Disease Classification**: Classifying silkworm health status as **Healthy (0)** or **Grasserie Disease (1)** (`[B, 2]`).
- **Core Architectural Design**:
  - **Shared Encoder Backbone**: Uses State-of-the-Art backbones (Visual Mamba / **VM-UNet** or Vision Transformer / **Swin-UNet**) to extract joint spatial representations.
  - **Dual Task Heads (Branching Architecture)**:
    - **Segmentation Head**: Conv decoder branching out to reconstruct the dense binary mask.
    - **Classification Head**: Bottleneck features extracted via forward hooks (or deepest encoder layer), routed through Global Average Pooling (GAP) + Fully Connected (FC) layers.
  - **Joint Multi-Task Loss**:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \cdot \mathcal{L}_{\text{seg}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}$$
    Trained end-to-end simultaneously so that shared features capture both geometric body contours and pathological disease indicators.
- **Auxiliary Tools & Models**:
  - **SAM 3 (Segment Anything Model 3)**: Automatically generates high-precision ground-truth pseudo-masks from YOLO bounding box labels.
  - **Silkynet**: Classical/CNN contour detection baseline for larva counting.

---

## 🏗️ 2. Repository 4-Tier Architecture

The project strictly follows a **4-Tier Modular Separation Design** (detailed in [`PROJECT_STRUCTURE.md`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/PROJECT_STRUCTURE.md)):

```text
Silkworm_Segmentation/
├── models/          ← [TIER 1] Upstream Original Backbones (READ-ONLY / Submodules)
├── src/             ← [TIER 2] Shared Core Code (Dataset, Losses, Metrics, Visualizer)
├── experiments/     ← [TIER 3] Experiment Wrappers & Pipelines (Config, Model, Train, Eval)
└── runs/            ← [TIER 4] Automatically Generated Artifacts (Checkpoints, Logs, Visuals)
```

### 🔒 Tier Rules & Responsibilities:

1. **TIER 1 — `models/` (Upstream Models)**:
   - Contains original author repositories (`vmunet`, `swin_unet`, `sam3`, `silkynet`).
   - 🔒 **STRICTLY READ-ONLY**: Do NOT modify files inside `models/` directly. Wrap backbones inside Tier 3 (`experiments/<exp_name>/model.py`).
2. **TIER 2 — `src/` (Shared Modules)**:
   - Maintained core logic shared across all experiments:
     - [`src/dataset_multitask.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/dataset_multitask.py): Multi-task DataLoaders (`SilkwormMultiTaskDataset`).
     - [`src/losses.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/losses.py): `DiceLoss`, `SegLoss`, `ClsLoss`, `JointMultiTaskLoss`.
     - [`src/metrics.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/metrics.py): Segmentation (Dice, IoU) and Classification (Accuracy, F1, Precision, Recall).
3. **TIER 3 — `experiments/` (Experiment Wrappers)**:
   - Each subdirectory represents an independent experiment approach (`multitask_vmunet`, `multitask_swinunet`, `segfirst_vmunet`).
   - Every experiment directory contains:
     - `config.py`: Hyperparameter dataclass.
     - `model.py`: Multi-task model definition connecting `models/` backbone with task heads.
     - `train.py`: Training loop script.
     - `evaluate.py`: Test dataset evaluation script.
     - `inference.py`: Prediction & visualization script for new unseen images.
     - `notes.md`: Experiment logs and observations.
   - **Override Mechanism**: If an experiment needs custom losses or augmentations, create `experiments/<exp_name>/losses.py` and use fallback imports:
     ```python
     try:
         from .losses import CustomLoss
     except ImportError:
         from src.losses import CustomLoss
     ```
4. **TIER 4 — `runs/` (Generated Artifacts)**:
   - 🤖 **Auto-generated during training**. Never write code here and never commit to Git.
   - Automatically organized by timestamp: `runs/<exp_name>/<YYYY-MM-DD_HH-MM-SS>/` containing:
     - `checkpoints/`: Model weights (`best_model.pth`, `last_model.pth`).
     - `logs/`: CSV metrics (`train_log.csv`) and TensorBoard logs.
     - `test_outputs/`: Visualizations of predictions vs. ground truth.
     - `config.yaml`: Snapshot of hyperparameters used during that run.

---

## 🛠️ 3. Essential Commands & Workflows

### 3.1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
> ⚠️ **VMamba CUDA Extension**: VM-UNet requires the C++/CUDA `selective_scan` module. See instructions in [`models/vmunet/README.md`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/models/vmunet/README.md).

### 3.2. Data Preparation & Pseudo-Mask Generation
```bash
# Generate high-quality segmentation masks from YOLO bounding boxes using SAM 3
python utils/sam3_label_from_yolo.py \
    --input-yaml "data/yolo_bbox/data.yaml" \
    --output-dir "data/sam3_seg" \
    --prompt "silkworm" \
    --device cuda
```

### 3.3. Training Experiments
> ⚠️ **CRITICAL RULE**: Always execute python scripts as modules (`python -m ...`) from the project root (`Silkworm_Segmentation/`).

```bash
# Train End-to-End Multi-Task VM-UNet (Primary Architecture)
python -m experiments.multitask_vmunet.train

# Train End-to-End Multi-Task Swin-UNet
python -m experiments.multitask_swinunet.train

# Train SegFirst VM-UNet Baseline
python -m experiments.segfirst_vmunet.train
```

### 3.4. Evaluation & Inference
```bash
# Run inference on test images using trained Multi-Task VM-UNet
python -m experiments.multitask_vmunet.inference \
    --checkpoint runs/multitask_vmunet/<run_timestamp>/checkpoints/best_model.pth \
    --image-dir data/test/ \
    --output-dir runs/multitask_vmunet/inference_output/

# Count larvae using Silkynet contour analysis
python models/silkynet/Count_contours.py
```

---

## 🎓 4. Mentor Instructions for AI Assistant (Student-Friendly Guidance)

> 💡 **User Profile Context**: The user is an undergraduate student who is learning deep learning and multi-task learning best practices. The AI assistant must act as a **patient, encouraging, and expert AI Research Mentor**.

### 🤝 AI Mentor Behavioral Guidelines:

1. **Explain the 'Why' Behind Concepts**:
   - Do not just output code changes. Explain *why* a particular layer, loss function, or hyperparameter is used (e.g., why we use `Dice Loss + BCE` for segmentation, or how `forward_hook` captures bottleneck features without editing source code).
2. **Step-by-Step Mentorship Workflow**:
   - **Step 1: Code Verification**: Before suggesting changes, view existing files using `view_file` or `grep_search`. Never guess variable names or class structures.
   - **Step 2: Guided Action**: Break down tasks into clear steps (Dataset setup -> Model building -> Training -> Metric evaluation).
   - **Step 3: Verification**: Always verify code syntax or run test commands after making changes.
   - **Step 4: Result Interpretation**: Teach the student how to read `train_log.csv` (monitoring Dice score vs Loss) and how to diagnose overfitting or loss imbalance.

3. **Common Learning & Debugging Scenarios**:
   - **Loss Imbalance in Multi-Task**: If segmentation training degrades while classification improves, explain how to adjust loss weights (`lambda_seg` vs `lambda_cls`).
   - **GPU VRAM / Batch Size**: If CUDA Out of Memory (OOM) occurs, guide the student to lower `batch_size` or enable `amp` (Automatic Mixed Precision) in `config.py`.
   - **Module Import Errors**: Remind the student to always run scripts using `python -m experiments.<exp_name>.<script>`.

---

## 📊 5. Dataset Architecture & Handling

Data details are documented in [`DATASET_STRUCTURE.md`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/DATASET_STRUCTURE.md):

- `data/yolo_bbox`: YOLO bounding boxes and disease class (`0: Grasserie`, `1: Healthy`).
- `data/sam3_seg`: SAM 3 generated pseudo-masks (`masks/`, `boundaries/`, `labels/`).
- `data/sam3_aug20k`: SAM3-only augmented Multi-Task Dataset (19,997 train / 499 valid / 498 test). `data/mixed_10k`: older 10k set incl. Silkynet.
- `models/silkynet/data`: Legacy Silkynet counting dataset.

### Rules for Dataset Management:
- Raw datasets inside `data/` are **read-only and immutable**.
- All paths referenced in code must be **relative to project root**.
- Corresponding image and mask files must maintain identical filenames (`sample_001.jpg` <-> `sample_001.png`).

---

## 💻 6. Coding Standards & Conventions

- **Language & Framework**: Python 3.10+, PyTorch >= 2.0.0.
- **Type Hinting**: Provide explicit type annotations for all function parameters and return values (`def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:`).
- **Configuration**: Store hyperparameters using Dataclasses in `experiments/<exp_name>/config.py`.
- **Device Safety**: Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Reproducibility**: Set seeds (`torch.manual_seed(cfg.seed)`, `np.random.seed(cfg.seed)`) at the start of training scripts.

---

## 🔍 7. Quick Reference File Mapping

| Task / Goal | File Location |
|---|---|
| Adjust Hyperparameters (LR, Batch Size, Epochs) | [`experiments/multitask_vmunet/config.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/experiments/multitask_vmunet/config.py) |
| Modify Multi-Task Architecture & Dual Heads | [`experiments/multitask_vmunet/model.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/experiments/multitask_vmunet/model.py) |
| Edit Core Multi-Task DataLoader | [`src/dataset_multitask.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/dataset_multitask.py) |
| Edit Shared Loss Functions | [`src/losses.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/losses.py) |
| Edit Evaluation Metrics (Dice, IoU, Accuracy, F1) | [`src/metrics.py`](file:///home/subnh5/nguyenvuthanhtinh/Silkworm_Segmentation/src/metrics.py) |
| View Saved Models & Checkpoints | `runs/<exp_name>/<timestamp>/checkpoints/` |
| View Training Logs & Metrics | `runs/<exp_name>/<timestamp>/logs/train_log.csv` |
