"""
config.py — Configuration dataclass for Segmentation-First Multi-Task VM-UNet training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SegFirstConfig:
    """
    Hyperparameters and paths for Segmentation-First Multi-Task VM-UNet training.

    Paths are relative to the project root (Silkworm_Segmentation/).
    """

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    # Dataset A — Silkynet segmentation data (image, mask)
    silkynet_data_dir: str = "data/silkworm_mixed_dataset"

    # Dataset B — YOLO classification data (image, class_label, bbox)
    yolo_data_dir: str = "data/Silkworm Diseases.v1i.yolo26"

    # ------------------------------------------------------------------
    # Model Hyperparameters
    # ------------------------------------------------------------------
    input_size: int = 128                  # Image resolution H=W
    num_seg_classes: int = 1               # Binary foreground / background
    num_cls_classes: int = 2               # Healthy (0) / Diseased (1)
    depths: list = field(default_factory=lambda: [2, 2, 2, 2])
    depths_decoder: list = field(default_factory=lambda: [2, 2, 2, 1])
    drop_path_rate: float = 0.2
    cls_hidden: int = 128                  # Hidden dimension of classification FC layers
    cls_dropout: float = 0.3

    # Pretrained VMamba checkpoint (relative to project root)
    pretrained_path: str = "models/vmunet/pre_trained_weights/vmamba_small_e238_ema.pth"

    # ------------------------------------------------------------------
    # Two-Phase Training Strategy
    # ------------------------------------------------------------------
    # Phase 1: Segmentation Pre-training
    phase1_epochs: int = 25
    phase1_lr: float = 1e-4

    # Phase 2: Multi-Task Fine-Tuning with Segmentation Protection
    phase2_epochs: int = 25
    freeze_backbone_phase2: bool = False   # If True, freeze backbone parameters in Phase 2
    lr_backbone: float = 1e-5             # Small LR for backbone to preserve seg features
    lr_seg_head: float = 1e-4             # Standard LR for segmentation head
    lr_cls_head: float = 1e-3             # Larger LR for classification head

    # Loss weights (Segmentation-first priority)
    lambda_seg: float = 1.0
    lambda_cls: float = 0.1               # Low impact [0.05, 0.2] to protect segmentation
    w_bce: float = 1.0                    # Weight of BCE in SegLoss
    w_dice: float = 1.0                   # Weight of Dice in SegLoss

    # ------------------------------------------------------------------
    # General Training Params
    # ------------------------------------------------------------------
    batch_size_seg: int = 2                # Batch size for Dataset A
    batch_size_cls: int = 2                # Batch size for Dataset B
    max_cls_samples: int | None = 500      # Subsample Dataset B to N images (None for all)
    num_workers: int = 4
    seed: int = 42
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)

    # Evaluation
    seg_threshold: float = 0.5            # Binarization threshold for IoU/Dice
    val_interval: int = 1                  # Validate every N epochs
    save_interval: int = 5                 # Save checkpoint every N epochs

    # Output & GPU
    work_dir: str = field(default="")      # Set automatically in __post_init__
    gpu_id: str = "0"
    amp: bool = True                       # Automatic mixed precision

    # ------------------------------------------------------------------

    def __post_init__(self):
        if not self.work_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = os.path.join(
                "runs", "segfirst_vmunet", f"run_{timestamp}"
            )

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
