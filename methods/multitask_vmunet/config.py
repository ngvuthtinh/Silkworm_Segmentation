"""
config.py — Configuration dataclass for multi-task VM-UNet training.
"""


from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class MultiTaskConfig:
    """
    All hyperparameters and paths for multi-task silkworm training.

    Paths are relative to the project root (Silkworm_Segmentation/).
    Override any field when instantiating:

        cfg = MultiTaskConfig(epochs=100, batch_size_seg=8)
    """

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    # Dataset A — Silkynet segmentation data
    silkynet_data_dir: str = "models/silkynet/data"

    # Dataset B — YOLO classification data
    yolo_data_dir: str = "data/Silkworm Diseases.v1i.yolo26"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    input_size: int = 128                  # H = W (resize all images to this for fast scan speed)
    num_seg_classes: int = 1               # binary: foreground/background
    num_cls_classes: int = 2               # healthy / diseased
    depths: list = field(default_factory=lambda: [2, 2, 2, 2])
    depths_decoder: list = field(default_factory=lambda: [2, 2, 2, 1])
    drop_path_rate: float = 0.2
    cls_hidden: int = 256                  # hidden dim of classification FC
    cls_dropout: float = 0.3

    # Pretrained VMamba checkpoint (relative to project root)
    pretrained_path: str = "models/vmunet/pre_trained_weights/vmamba_small_e238_ema.pth"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 50
    batch_size_seg: int = 2                # batch size for Dataset A (seg)
    batch_size_cls: int = 2                # batch size for Dataset B (cls)
    max_cls_samples: int | None = 500      # subsample Dataset B to 500 images (250 healthy / 250 diseased)
    num_workers: int = 4
    seed: int = 42

    # Loss weights
    lambda_seg: float = 1.0
    lambda_cls: float = 1.0
    w_bce: float = 1.0                     # weight of BCE inside SegLoss
    w_dice: float = 1.0                    # weight of Dice inside SegLoss
    label_smoothing: float = 0.0           # label smoothing for ClsLoss

    # ------------------------------------------------------------------
    # Optimizer (AdamW)
    # ------------------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)

    # ------------------------------------------------------------------
    # Scheduler (CosineAnnealingLR)
    # ------------------------------------------------------------------
    T_max: int = 50                        # = epochs by default
    eta_min: float = 1e-6

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    seg_threshold: float = 0.5            # binarise mask_pred for IoU/Dice
    val_interval: int = 1                  # validate every N epochs
    save_interval: int = 5                 # save checkpoint every N epochs

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    work_dir: str = field(default="")      # set automatically in __post_init__
    gpu_id: str = "0"
    amp: bool = True                       # automatic mixed precision

    # ------------------------------------------------------------------

    def __post_init__(self):
        if not self.work_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = os.path.join(
                "runs", "multitask_vmunet", f"run_{timestamp}"
            )
        # T_max should match epochs
        if self.T_max == 50 and self.epochs != 50:
            self.T_max = self.epochs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.work_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.work_dir, "logs")

    @property
    def vis_dir(self) -> str:
        return os.path.join(self.work_dir, "visualizations")

    def create_dirs(self) -> None:
        for d in [self.checkpoint_dir, self.log_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        lines = ["MultiTaskConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
