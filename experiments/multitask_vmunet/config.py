"""
config.py — Siêu tham số cho VM-UNet phân đoạn TỪNG CON + chẩn đoán TỪNG CON.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class MultiTaskConfig:
    # Dữ liệu (đường dẫn tính từ thư mục gốc dự án)
    data_root: str = "data/sam3_aug20k"
    img_size: int = 256
    boundary_width: int = 1            # độ dày viền (px, ở kích thước img_size)

    # Mô hình
    depths: list = field(default_factory=lambda: [2, 2, 2, 2])
    depths_decoder: list = field(default_factory=lambda: [2, 2, 2, 1])
    drop_path_rate: float = 0.2
    pretrained_path: str = "models/vmunet/pre_trained_weights/vmamba_small_e238_ema.pth"

    # Huấn luyện
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    amp: bool = True

    # Trọng số loss: L = λ_body·L_body + λ_boundary·L_boundary + λ_disease·L_disease
    lambda_body: float = 1.0
    lambda_boundary: float = 1.0
    lambda_disease: float = 1.0

    # Hậu xử lý & đánh giá theo từng con
    body_thr: float = 0.5
    boundary_thr: float = 0.5
    disease_thr: float = 0.5
    iou_thr: float = 0.5               # IoU tối thiểu để coi là tách đúng 1 con
    min_area: int = 100                # bỏ qua mảnh < min_area px khi chấm điểm

    work_dir: str = ""

    def __post_init__(self) -> None:
        if not self.work_dir:
            self.work_dir = os.path.join("runs", "multitask_vmunet", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.work_dir, "checkpoints")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.work_dir, "logs")

    def to_dict(self) -> dict:
        return asdict(self)
