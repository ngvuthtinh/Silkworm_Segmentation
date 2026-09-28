# 📁 Project Structure & Architecture Guidelines — Silkworm Segmentation

> **Design Philosophy**: Complete separation of **Upstream Models** / **Shared Core Code** / **Experiment Wrappers** / **Generated Artifacts**.  
> Every component resides strictly in its designated tier without overlap or circular dependencies.

---

## 🗺️ High-Level Overview

```text
Silkworm_Segmentation/
│
├── models/          ← [TIER 1] Original Upstream Model Code (Submodules, READ-ONLY)
├── src/             ← [TIER 2] Shared Core Framework Modules (Maintained by you)
├── experiments/     ← [TIER 3] Experiment Wrappers & Pipelines (Configs & Dual Task Heads)
└── runs/            ← [TIER 4] Automatically Generated Artifacts (Checkpoints & Logs)
```

---

## TIER 1 — `models/` : Original Upstream Backbones

> 🔒 **Rule: STRICTLY READ-ONLY — DO NOT MODIFY DIRECTLY.**  
> These are official upstream repositories imported via `git submodule`.  
> Updating backbones is done cleanly via `git submodule update`.

```text
models/
├── vmunet/          # git submodule → Original VM-UNet / Visual Mamba repo
├── swin_unet/       # git submodule → Original Swin-UNet repo
├── sam3/            # git submodule → Original SAM 3 repo
└── silkynet/        # Submodule / repository → Silkynet model & counting data
```

**Adding a new model backbone:**
```bash
git submodule add https://github.com/author/NewModel.git models/new_model
```

---

## TIER 2 — `src/` : Shared Core Modules

> ✅ **Code maintained by you and shared across all experiments.**  
> Fixing a bug or adding a feature in `src/` instantly improves all experiments.

```text
src/
├── dataset.py            # Single-task DataLoaders (SilkynetSegDataset, YOLOClsDataset)
├── dataset_multitask.py  # Joint Multi-Task DataLoaders (SilkwormMultiTaskDataset)
├── losses.py             # Shared loss functions (DiceLoss, SegLoss, ClsLoss, JointMultiTaskLoss)
├── metrics.py            # Evaluation metrics (Dice, IoU, Accuracy, F1, Precision, Recall)
└── visualizer.py         # Prediction visualization tools (Overlay, RGB, Masks)
```

### ❓ When to put code in `src/` vs. inside `experiments/`?

| Scenario | Action |
|---|---|
| Loss function shared across multiple multi-task models | Put in `src/losses.py` |
| 1 experiment needs a custom loss (e.g., Focal Loss specifically for VM-UNet) | Create `experiments/multitask_vmunet/losses.py` to **override** |
| Standard dataset loading logic | Put in `src/dataset_multitask.py` |
| 1 experiment requires specialized data augmentations | Override inside `experiments/<exp_name>/dataset.py` |

**Code Fallback / Override Mechanism:**
```python
# experiments/multitask_vmunet/train.py

try:
    from .losses import CustomLoss       # 👈 Use custom loss if defined in experiment
except ImportError:
    from src.losses import CustomLoss    # 👈 Fallback to shared loss in src/
```

---

## TIER 3 — `experiments/` : Experiment Wrappers & Pipelines

> 🧪 **Each folder represents an independent research approach.**  
> Experiment files are concise wrappers linking `models/` backbones with dual task heads (`SegmentationHead` + `ClassificationHead`) and shared `src/` utilities.

```text
experiments/
├── multitask_vmunet/         # ⭐ Primary Joint Multi-Task VM-UNet Experiment
│   ├── config.py             #   ← Hyperparameter dataclass & output directory setup
│   ├── model.py              #   ← MultiTaskVMUNet wrapper connecting VSSM backbone with dual heads
│   ├── train.py              #   ← End-to-end multi-task training loop
│   ├── evaluate.py           #   ← Test split evaluation script
│   ├── inference.py          #   ← Inference script for unseen images
│   └── notes.md              #   ← Experiment notes, hyperparameter logs, observations
│
├── multitask_swinunet/       # Joint Multi-Task Swin-UNet Experiment
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── segfirst_vmunet/          # Segmentation-First 2-Phase Staged Baseline
└── segfirst_swinunet/        # Segmentation-First Swin-UNet Baseline
```

### 🧠 Dual-Task Head Model Architecture Design (`model.py`)

In joint multi-task learning (`multitask_vmunet`), a single shared backbone processes the image:
1. **Segmentation Head**: Decoder outputs dense binary mask `[B, 1, H, W]`.
2. **Classification Head**: Registered forward hook captures deepest bottleneck features (`VSSM.layers[3]`), passes through Adaptive Average Pooling -> FC layers to output `[B, 2]` disease logits.

---

## TIER 4 — `runs/` : Generated Artifacts (Auto-Created)

> 🤖 **Contains NO code. Auto-generated during training.**  
> Every execution creates a unique timestamped directory, ensuring no run overwrites another.

```text
runs/
├── multitask_vmunet/
│   ├── 2026-09-28_14-30-00/      # Run timestamp directory
│   │   ├── config.yaml           #   Snapshot copy of configuration used for this run
│   │   ├── checkpoints/
│   │   │   ├── best_model.pth    #   Best checkpoint based on validation score
│   │   │   └── last_model.pth    #   Final epoch checkpoint
│   │   ├── logs/
│   │   │   ├── train_log.csv     #   Per-epoch losses and metrics
│   │   │   └── events.out.tfevents.* # Tensorboard log file
│   │   └── test_outputs/
│   │       ├── sample_001.png    #   Visualization of prediction vs ground truth
│   │       └── sample_002.png
│   │
│   └── 2026-09-28_18-00-00/      # Subsequent training run
│
└── multitask_swinunet/
    └── 2026-09-28_11-00-00/
```

---

## 📊 Quick Lookup Table: "Which file should I modify?"

| Task | File Path |
|---|---|
| Add a new backbone architecture | `git submodule add` into `models/` |
| Modify shared Multi-Task DataLoader | `src/dataset_multitask.py` |
| Add a new shared loss function | `src/losses.py` |
| Override loss for 1 specific experiment | `experiments/<exp_name>/losses.py` |
| Change learning rate, batch size, epochs | `experiments/<exp_name>/config.py` |
| Modify dual-head branching architecture | `experiments/<exp_name>/model.py` |
| Check results of a past training run | `runs/<exp_name>/<timestamp>/` |
| Compare metrics across runs | Open `runs/<exp_name>/<timestamp>/logs/train_log.csv` |

---

## 🚫 Critical Rules

- ❌ **Never modify code inside `models/` directly** (Upstream submodules).
- ❌ **Never duplicate `dataset.py` or `losses.py` across experiments** when shared logic belongs in `src/`.
- ❌ **Never write checkpoints or generated images inside `experiments/` or `src/`**.
- ❌ **Never commit the `runs/` directory to Git** (enforced by `.gitignore`).
