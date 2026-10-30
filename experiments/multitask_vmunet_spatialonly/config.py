"""
config.py — Siêu tham số cho VM-UNet + spatial embeddings KHÔNG có kênh biên / watershed.

Kế thừa `SpatialEmbConfig`: mọi siêu tham số (lr, epoch, batch, λ thân/bệnh/spatial, σ, seed_thr...) giống hệt run
`multitask_vmunet_spatialemb`. Khác duy nhất: không còn kênh biên, nên `lambda_boundary` = 0 (không dùng).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from experiments.multitask_vmunet_spatialemb.config import SpatialEmbConfig


@dataclass
class SpatialOnlyConfig(SpatialEmbConfig):
    lambda_boundary: float = 0.0       # không có kênh biên; giữ trường này ở 0 để config.yaml ghi rõ

    def __post_init__(self) -> None:
        if not self.work_dir:
            self.work_dir = os.path.join("runs", "multitask_vmunet_spatialonly", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
