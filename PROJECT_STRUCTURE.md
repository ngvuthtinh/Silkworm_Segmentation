# 📁 Project Structure & Architecture Guidelines — Silkworm Segmentation

> **Design philosophy**: strict separation of **upstream models** / **shared code** / **experiments** / **generated artifacts**. Each file lives in exactly one tier.

---

## 🗺️ Overview

```text
Silkworm_Segmentation/
├── models/          ← [TIER 1] Upstream model code (do not edit)
├── src/             ← [TIER 2] Shared modules (datasets, losses, metrics)
├── experiments/     ← [TIER 3] One folder per experiment
├── runs/            ← [TIER 4] Auto-generated checkpoints & logs
├── data/            ← Datasets (see DATASET_STRUCTURE.md)
└── utils/           ← Data generation scripts
```

---

## TIER 1 — `models/`: Upstream Backbones

> 🔒 Do not edit. These folders are plain copies of the original repositories (not git submodules). Wrap them in `experiments/<exp>/model.py` instead.

```text
models/
├── vmunet/      # VM-UNet / VMamba — only models/vmunet/models/vmunet/vmamba.py (VSSM) is used
├── swin_unet/   # Swin-UNet network + config + pretrained checkpoint (+ legacy Silkworm scripts)
├── sam3/        # SAM 3, used by utils/sam3_label_from_yolo.py
└── silkynet/    # Legacy, no longer used
```

---

## TIER 2 — `src/`: Shared Modules

```text
src/
├── dataset_instance.py   # InstanceSilkwormDataset → (image, body, boundary, disease, inst)
├── dataset_multitask.py  # Legacy image-level dataset (MultitaskSilkwormDataset)
├── dataset.py            # Legacy single-task datasets
├── losses.py             # InstanceMultiTaskLoss, SegLoss, DiceLoss, ClsLoss, SegFirstMultiTaskLoss
└── metrics.py            # split_instances, match_instances, summarize_instance_counts (+ legacy)
```

| Scenario | Where |
|---|---|
| Loss shared by several experiments | `src/losses.py` |
| Loss only one experiment needs | `experiments/<exp>/losses.py` (override) |
| Standard data loading | `src/dataset_instance.py` |
| Experiment-specific augmentation | `experiments/<exp>/dataset.py` (override) |

Override pattern:
```python
try:
    from .losses import CustomLoss       # experiment-specific version
except ImportError:
    from src.losses import CustomLoss    # shared fallback
```

---

## TIER 3 — `experiments/`

```text
experiments/
├── multitask_vmunet/      # ⭐ Primary: per-larva segmentation + per-larva diagnosis
│   ├── config.py         #   MultiTaskConfig dataclass (data, model, λ, thresholds)
│   ├── model.py          #   MultiTaskVMUNet: VSSM backbone, 3 output channels (body, boundary, disease)
│   ├── train.py          #   Training loop, per-epoch per-larva validation, CSV log, best/last checkpoints
│   ├── evaluate.py       #   Per-larva evaluation on valid / test / test_real
│   └── notes.md
│
├── multitask_vmunet_v1/     # Legacy: binary mask + image-level disease head (forward hook on VSSM.layers[3])
├── multitask_swinunet/   # Legacy (train.py only, reuses segfirst_swinunet model)
├── segfirst_vmunet/      # Legacy staged baseline (data path no longer exists)
└── segfirst_swinunet/    # Legacy staged baseline (data path no longer exists)
```

### 🧠 Primary model (`multitask_vmunet/model.py`)
One VSSM encoder–decoder outputs `[B, 3, H, W]` logits:
1. **Body** — larva vs. background.
2. **Boundary** — contour of each larva; cutting it out of the body separates touching larvae.
3. **Disease** — Grasserie probability per pixel, trained only inside the body.

`src/metrics.split_instances` turns these maps into one region per larva and one diagnosis per larva.

---

## TIER 4 — `runs/`

```text
runs/multitask_vmunet/<YYYY-MM-DD_HH-MM-SS>/
├── config.yaml              # snapshot of MultiTaskConfig
├── checkpoints/best.pth     # best validation e2e_f1
├── checkpoints/last.pth
└── logs/train_log.csv       # losses + body_dice, inst_f1, cls_acc_matched, e2e_f1, grasserie_recall
```

---

## 📊 Which file should I modify?

| Task | File |
|---|---|
| Learning rate, batch size, epochs, λ, thresholds | `experiments/multitask_vmunet/config.py` |
| Model outputs / backbone wrapper | `experiments/multitask_vmunet/model.py` |
| How boundaries are built from instance maps | `src/dataset_instance.py` |
| Loss | `src/losses.py` (`InstanceMultiTaskLoss`) |
| Splitting larvae / per-larva metrics | `src/metrics.py` |
| Generate or regenerate data | `utils/augment_mixed_10k.py` |
| Inspect a run | `runs/<exp>/<timestamp>/` |

---

## 🚫 Critical Rules

- ❌ Never edit code inside `models/`.
- ❌ Never duplicate shared logic (datasets, losses, metrics) inside experiments.
- ❌ Never write checkpoints or generated images into `experiments/` or `src/`.
- ❌ Never commit `runs/` or `data/` (enforced by `.gitignore`).
