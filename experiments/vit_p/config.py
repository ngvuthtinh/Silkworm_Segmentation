"""
config.py — Configuration dataclass for ViT-P silkworm experiment.
Strictly aligned with segfirst_swinunet hyperparameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ViTPConfig:
    # Dataset
    data_dir: str = "data/silkworm_mixed_dataset"
    input_size: int = 224
    num_classes: int = 2          # 0: Background, 1: Silkworm
    num_points: int = 32          # Central / mask sampled points per image

    # Optimization (strictly matching segfirst_swinunet)
    batch_size: int = 6
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    amp: bool = True

    # Backbone
    arch: str = "dinov2_vits14"
    patch_size: int = 14

    # System & Runtime
    gpu_id: str = "1"             # GPU 1 dedicated for ViT-P (GPU 0 for Mask2Former)
    num_workers: int = 4
    seed: int = 42
    output_dir: str = "runs/vit_p"
    save_interval: int = 5

    def create_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
