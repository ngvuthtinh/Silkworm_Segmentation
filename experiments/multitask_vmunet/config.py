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
    """

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    silkynet_data_dir: str = "models/silkynet/data"
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
    batch_size_seg: int = 2
    batch_size_cls: int = 2
    max_cls_samples: int | None = 500
    num_workers: int = 4
    seed: int = 42

    # Loss weights
    lambda_seg: float = 1.0
    lambda_cls: float = 1.0
    w_bce: float = 1.0
    w_dice: float = 1.0
    label_smoothing: float = 0.0

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"
    T_max: int = 50
    eta_min: float = 1e-6

    # Evaluation
    seg_threshold: float = 0.5
    val_interval: int = 1
    save_interval: int = 5

    # Output & GPU
    work_dir: str = field(default="")
    gpu_id: str = "0"
    amp: bool = True

    def __post_init__(self):
        if not self.work_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.work_dir = os.path.join(
                "runs", "multitask_vmunet", timestamp
            )
        if self.T_max == 50 and self.epochs != 50:
            self.T_max = self.epochs

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
