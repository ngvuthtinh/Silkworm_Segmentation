# 🐛 Silkworm Per-Larva Segmentation & Disease Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture: 4-Tier](https://img.shields.io/badge/Design-4--Tier%20Modular-success)](PROJECT_STRUCTURE.md)

Given a photo of silkworm larvae (e.g. on a production line), the system **segments every larva individually — even when larvae touch or overlap — and diagnoses each larva as Healthy or Grasserie**.

---

## 📌 Approach

- **VM-UNet (Visual Mamba) multi-task network** with one shared encoder–decoder and three per-pixel outputs:
  - **Body**: larva vs. background
  - **Boundary**: contour of each larva — separates touching/overlapping larvae
  - **Disease**: Grasserie vs. Healthy for every larva pixel
- **Per-larva post-processing**: (body − boundary) seeds → watershed → one region per larva → mean disease probability → diagnosis per larva.
- **Joint loss**: λ_body·(BCE + Dice) + λ_boundary·(BCE + Dice) + λ_disease·BCE (inside the body only).
- **Data**: SAM 3 turns YOLO boxes into pixel masks; an augmentation pipeline composes realistic scenes with several overlapping healthy and diseased larvae, leaf occlusion and photometric changes, and records an instance map + class map per image.
- **Evaluation per larva**: separation F1 (IoU ≥ 0.5), diagnosis accuracy, end-to-end F1 (separated *and* correctly diagnosed), Grasserie recall.

---

## 🏗️ Repository Layout

```text
Silkworm_Segmentation/
├── models/                  # [TIER 1] Upstream backbones (do not edit): vmunet, swin_unet, sam3, silkynet
├── src/                     # [TIER 2] Shared code
│   ├── dataset_instance.py  #   Per-larva dataset (body / boundary / disease / instance maps)
│   ├── dataset_multitask.py #   Legacy image-level dataset
│   ├── losses.py            #   InstanceMultiTaskLoss (+ legacy losses)
│   └── metrics.py           #   split_instances, per-larva matching & metrics
├── experiments/             # [TIER 3] One folder per experiment
│   ├── multitask_vmunet/     #   ⭐ Primary: per-larva segmentation + diagnosis
│   ├── multitask_vmunet_v1/    #   Legacy: binary mask + one label per image
│   ├── multitask_swinunet/  #   Legacy
│   └── segfirst_*/          #   Legacy staged baselines
├── runs/                    # [TIER 4] Checkpoints & logs (auto-generated, not committed)
├── data/                    # Datasets (not committed) — see DATASET_STRUCTURE.md
│   ├── yolo_bbox/           #   Raw images + YOLO boxes + disease class
│   ├── sam3_seg/            #   SAM 3 pseudo-masks
│   ├── sam3_aug20k/         #   ⭐ Per-larva training set (train / valid / test / test_real)
│   └── mixed_10k/           #   Legacy image-level set
└── utils/
    ├── sam3_label_from_yolo.py  # SAM 3 masks from YOLO boxes
    └── augment_mixed_10k.py     # Scene composition / augmentation
```

---

## ⚙️ Installation

VM-UNet needs the CUDA `selective_scan` kernel from `mamba_ssm` (~0.1 s/step instead of ~18 s). Prebuilt wheels exist only up to torch 2.4, hence the pinned versions.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-deps "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Pseudo-masks with SAM 3
python utils/sam3_label_from_yolo.py --input-yaml data/yolo_bbox/data.yaml --output-dir data/sam3_seg --prompt silkworm --device cuda

# 2. Build the per-larva dataset (repeat with --split valid/test and a smaller --goal)
python utils/augment_mixed_10k.py --sam3-only --name-prefix aug --split train --output-dir data/sam3_aug20k/train --goal 20000

# 3. Train (always from the project root, as a module)
python -m experiments.multitask_vmunet.train --smoke-test --gpu 1
python -m experiments.multitask_vmunet.train --gpu 1

# 4. Evaluate per larva
python -m experiments.multitask_vmunet.evaluate --checkpoint runs/multitask_vmunet/<run>/checkpoints/best.pth --split test_real
```

Outputs go to `runs/multitask_vmunet/<timestamp>/` (`checkpoints/`, `logs/train_log.csv`, `config.yaml`).

---

## 📖 Documentation

- [**PROJECT_STRUCTURE.md**](PROJECT_STRUCTURE.md) — 4-tier rules and where to change what.
- [**DATASET_STRUCTURE.md**](DATASET_STRUCTURE.md) — datasets, formats, generation commands, known issues.
- [**CLAUDE.md**](CLAUDE.md) — guidance for AI assistants and developers.
