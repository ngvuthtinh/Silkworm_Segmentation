"""
config.py — Siêu tham số cho VM-UNet + spatial embeddings (Neven et al., CVPR 2019).

Kế thừa nguyên cấu hình baseline `MultiTaskConfig`, chỉ THÊM các tham số của nhánh spatial embeddings. Mọi siêu tham số khác
(lr, epoch, batch, λ thân/biên/bệnh, ngưỡng...) luôn trùng baseline và trùng thí nghiệm `multitask_vmunet_disc`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from experiments.multitask_vmunet.config import MultiTaskConfig


@dataclass
class SpatialEmbConfig(MultiTaskConfig):
    # Nhánh spatial embeddings (mặc định theo repo gốc train_config.py, trừ n_sigma)
    n_sigma: int = 2                   # 1 = vùng gaussian tròn; 2 = elip theo trục x, y (hợp con dài; paper: σxy tốt hơn σ)
    to_center: bool = False            # False: tâm học được = trung bình embedding (paper, Bảng 2: tốt hơn); True: trung bình toạ độ của con
                                       # (con cong hoặc bị che thành nhiều mảnh có thể có tâm hình học nằm ngoài thân, nên mặc định là False)
    foreground_weight: float = 10.0    # trọng số seed loss trong vùng con (repo gốc: 10)
    w_inst: float = 1.0                # Lovász-hinge (IoU) của mỗi con
    w_var: float = 10.0                # σ đồng đều trong một con
    w_seed: float = 1.0                # seed map
    lambda_spatial: float = 1.0        # trọng số của cả nhánh này trong tổng loss (cùng thang với λ_body, λ_boundary, λ_disease)

    # Trộn data khi train (rỗng = chỉ data_root như cũ). Mỗi mục '<tên>:<số lần lặp>', tên trong src.dataset_instance.MIX_SOURCES.
    # valid vẫn là data_root/valid để so sánh được với B1, E1-E3. Mỗi ảnh được co ngẫu nhiên theo tỉ lệ của nguồn rồi cắt cửa sổ `crop_size`.
    train_mix: list = field(default_factory=list)
    crop_size: int = 256

    # Suy luận: tách con bằng clustering quanh các seed
    seed_thr: float = 0.5              # chỉ lấy seed có điểm >= seed_thr làm tâm một con (chỉnh trên valid)
    cluster_min_area: int = 30         # vùng/đề xuất nhỏ hơn (px ở 256²) bị bỏ

    def __post_init__(self) -> None:
        if not self.work_dir:
            self.work_dir = os.path.join("runs", "multitask_vmunet_spatialemb", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
