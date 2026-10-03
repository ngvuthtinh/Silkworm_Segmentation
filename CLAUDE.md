# 🐛 CLAUDE.md — Silkworm Per-Larva Segmentation & Disease Detection

This document is the primary guidance file for AI Assistants (Claude / Antigravity) and developers working on this repository.

---

## 🎯 1. Project Goal & Overview

- **Use case**: a silkworm production line. Larvae pass under a camera; for every photo the system must **find each larva individually** — even when larvae touch or overlap — and **decide for each larva whether it is Healthy or has Grasserie disease**.
- **Primary model**: **VM-UNet** (Visual Mamba) trained as a **multi-task** network with one shared encoder–decoder and **three per-pixel outputs** (`[B, 3, H, W]`):
  1. **Body** — is this pixel part of a larva?
  2. **Boundary** — is this pixel on the contour of a larva? (separates touching/overlapping larvae)
  3. **Disease** — does this pixel belong to a Grasserie larva?
- **Per-larva post-processing** (`src/metrics.py::split_instances`): seeds = body − boundary → watershed back over the body → one region per larva → mean disease probability over the region decides **Healthy / Grasserie for that larva**.
- **Joint loss**:
  $$\mathcal{L} = \lambda_{\text{body}}(\text{BCE}+\text{Dice}) + \lambda_{\text{bnd}}(\text{BCE}+\text{Dice}) + \lambda_{\text{dis}}\,\text{BCE}_{\text{inside body}}$$
- **Class convention**: `0 = Grasserie`, `1 = Healthy` (label files); in `masks_cls`: `0 = background`, `1 = Grasserie`, `2 = Healthy`.
- **Auxiliary tools**: **SAM 3** generates pseudo-masks from YOLO boxes. **Silkynet is no longer used** (multi-larva images without disease labels).
- **Legacy**: `experiments/multitask_vmunet_v1` / `multitask_swinunet` (binary mask + one disease label per image) and `segfirst_*` are kept only to reproduce earlier reports; they are not developed further.

---

## 🏗️ 2. Repository 4-Tier Architecture

Detailed in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md):

```text
Silkworm_Segmentation/
├── models/          ← [TIER 1] Upstream backbones (do not edit)
├── src/             ← [TIER 2] Shared code: datasets, losses, metrics
├── experiments/     ← [TIER 3] One folder per experiment (config, model, train, evaluate, inference)
├── runs/            ← [TIER 4] Auto-generated checkpoints & logs (never commit)
├── data/            ← Datasets (never commit) — see DATASET_STRUCTURE.md
└── utils/           ← Data generation scripts (SAM 3 labelling, augmentation)
```

### 🔒 Tier Rules
1. **`models/`** (`vmunet`, `swin_unet`, `sam3`, `silkynet`): upstream code. Do **not** modify; wrap backbones in `experiments/<exp>/model.py`.
2. **`src/`**: shared modules
   - [`src/dataset_instance.py`](src/dataset_instance.py): `InstanceSilkwormDataset` → `(image, body, boundary, disease, inst)`.
   - [`src/losses.py`](src/losses.py): `InstanceMultiTaskLoss` (+ legacy `SegLoss`, `ClsLoss`, `SegFirstMultiTaskLoss`).
   - [`src/metrics.py`](src/metrics.py): `split_instances`, `match_instances`, `summarize_instance_counts` (+ legacy Dice/accuracy helpers).
   - [`src/dataset_multitask.py`](src/dataset_multitask.py): legacy image-level dataset.
3. **`experiments/`**: each folder has `config.py`, `model.py`, `train.py`, `evaluate.py`, (`inference.py`), `notes.md`. Experiment-specific overrides use the fallback import pattern:
   ```python
   try:
       from .losses import CustomLoss
   except ImportError:
       from src.losses import CustomLoss
   ```
4. **`runs/<exp>/<YYYY-MM-DD_HH-MM-SS>/`**: `checkpoints/{best,last}.pth`, `logs/train_log.csv`, `config.yaml`.

---

## 🛠️ 3. Essential Commands

> ⚠️ Always run scripts **as modules from the project root**: `python -m experiments.<exp>.<script>`.

### 3.1. Environment
```bash
source .venv/bin/activate   # Python 3.12, torch 2.4.1 + CUDA 12.4, mamba_ssm 2.2.2 (fast selective_scan kernel)
```
- VM-UNet needs the CUDA `selective_scan` kernel from `mamba_ssm`. Without it `vmamba.py` falls back to a pure-PyTorch loop: ~18 s per step instead of ~0.1 s (batch 8, 256²). At startup you must see `[VMAMBA] Fast C++ CUDA selective_scan_fn loaded successfully!`.
- torch is pinned to 2.4.1 because prebuilt `mamba_ssm` wheels only exist up to torch 2.4 (the server's `nvcc` 11.3 cannot build it for newer CUDA). Install steps: see README.

### 3.2. Data
```bash
# SAM 3 pseudo-masks from YOLO boxes
python utils/sam3_label_from_yolo.py --input-yaml data/yolo_bbox/data.yaml --output-dir data/sam3_seg --prompt silkworm --device cuda

# Augmented per-larva dataset (one call per split; sources never cross splits)
python utils/augment_mixed_10k.py --sam3-only --name-prefix aug --split train --output-dir data/sam3_aug20k/train --goal 20000 --workers 24 --seed 42
```

### 3.3. Train / evaluate (primary experiment)
```bash
python -m experiments.multitask_vmunet.train --smoke-test --gpu 1     # 5 train + 3 val batches
python -m experiments.multitask_vmunet.train --gpu 1 [--batch-size 4] [--epochs 40]
python -m experiments.multitask_vmunet.evaluate --checkpoint runs/multitask_vmunet/<run>/checkpoints/best.pth --split test
python -m experiments.multitask_vmunet.evaluate --checkpoint ... --split test_real
```

### 3.4. Metrics in `train_log.csv`
| Column | Meaning |
|---|---|
| `body_dice` | pixel Dice of the body channel |
| `inst_f1` | larvae correctly separated (IoU ≥ 0.5), regardless of disease |
| `cls_acc_matched` | among correctly separated larvae, fraction with the correct diagnosis |
| `e2e_f1` | separated **and** correctly diagnosed — **main score**, used to pick `best.pth` |
| `grasserie_recall` | fraction of diseased larvae found and flagged (missing a sick larva is the costly error) |

Objects smaller than `min_area` (100 px at 256²) are ignored on both sides (SAM 3 noise, slivers hidden by leaves).

---

## 🎓 4. Mentor Instructions for AI Assistant

> 💡 The user is an undergraduate student learning deep learning and multi-task learning. Act as a **patient, encouraging, expert research mentor**.

1. **Explain the "why"** — e.g. why Dice is added for the sparse boundary channel, why the disease loss is computed only inside the body, why watershed needs seeds.
2. **Workflow**: verify existing code before changing it → break work into steps (data → model → training → evaluation) → verify by running code → teach how to read `train_log.csv`.
3. **Common scenarios**:
   - **Loss imbalance**: if `inst_f1` stalls while the disease loss drops, raise `lambda_boundary`; if diagnosis lags, raise `lambda_disease`.
   - **Touching larvae merged**: check the boundary channel and `boundary_thr`; a thicker `boundary_width` separates more but may split thin larvae.
   - **OOM / slow training**: confirm the fast kernel is loaded first (≈0.1 s/step at batch 8; 40 epochs ≈ 3 h); then lower `--batch-size` or keep `amp=True`. GPUs on the server are shared — check `nvidia-smi`.
   - **Import errors**: run with `python -m ...` from the project root.

---

## 📊 5. Datasets

See [`DATASET_STRUCTURE.md`](DATASET_STRUCTURE.md).
- `data/yolo_bbox` — raw images + boxes + disease class.
- `data/sam3_seg` — SAM 3 masks.
- `data/sam3_aug20k` — **primary**: per-larva maps (`masks_inst`, `masks_cls`); splits `train` 20,000 / `valid` 1,000 / `test` 1,000 (synthetic, mixed healthy + diseased, overlapping larvae) and `test_real` 497 (original SAM 3 test images).
- `data/mixed_10k` — legacy image-level set (known issues, see DATASET_STRUCTURE.md).

Rules: raw data is read-only; paths in code are relative to the project root; `images/<stem>` ↔ `masks*/<stem>.png` ↔ `labels/<stem>.txt`.

---

## 💻 6. Coding Standards

- Python 3.12, PyTorch ≥ 2.0; explicit type hints on all functions.
- Hyperparameters live in dataclasses in `experiments/<exp>/config.py`; each run saves a `config.yaml` snapshot.
- Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Reproducibility: set `random`, `np.random`, `torch.manual_seed(cfg.seed)` at the start of training.
- Commit messages: short, no co-author trailer.

---

## 🔍 7. Quick Reference

| Task | File |
|---|---|
| Hyperparameters (LR, batch, epochs, λ, thresholds) | [`experiments/multitask_vmunet/config.py`](experiments/multitask_vmunet/config.py) |
| Model (3-channel VM-UNet) | [`experiments/multitask_vmunet/model.py`](experiments/multitask_vmunet/model.py) |
| Dataset / boundary generation | [`src/dataset_instance.py`](src/dataset_instance.py) |
| Loss | [`src/losses.py`](src/losses.py) → `InstanceMultiTaskLoss` |
| Per-larva splitting & metrics | [`src/metrics.py`](src/metrics.py) |
| Data generation | [`utils/augment_mixed_10k.py`](utils/augment_mixed_10k.py) |
| Run outputs | `runs/multitask_vmunet/<timestamp>/` |
