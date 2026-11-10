"""
config.py — Siêu tham số cho các biến thể KIẾN TRÚC của VM-UNet + Spatial (baseline = `multitask_vmunet_spatialonly`).

Kế thừa nguyên `SpatialOnlyConfig`: loss, data, lr, epoch, batch, seed_thr... không đổi. Chỉ thêm `arch`, chọn cách đổi kiến trúc:

    base    giống hệt baseline (kiểm tra folder này tái tạo đúng baseline; dùng để đo nhiễu giữa các seed)
    deep3   encoder stage 3 sâu hơn: depths [2,2,2,2] → [2,2,4,2]
    branch  tách decoder: hai tầng decoder cuối + final_up nhân đôi; nhánh 1 ra thân/bệnh, nhánh 2 ra offset/σ/seed
    x0skip  nối skip 64×64 (skip_list[0], baseline bỏ phí) vào đầu ra tầng decoder cuối

Mỗi lần chỉ đổi MỘT thứ so với baseline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from experiments.multitask_vmunet_spatialonly.config import SpatialOnlyConfig

ARCHS = ("base", "deep3", "branch", "x0skip")
ARCH_DEPTHS = {"deep3": [2, 2, 4, 2]}            # các arch còn lại dùng depths mặc định [2, 2, 2, 2]


@dataclass
class ArchConfig(SpatialOnlyConfig):
    arch: str = "base"

    def __post_init__(self) -> None:
        if self.arch not in ARCHS:
            raise ValueError(f"arch phải là một trong {ARCHS}, nhận {self.arch!r}")
        if self.arch in ARCH_DEPTHS:
            self.depths = list(ARCH_DEPTHS[self.arch])
        if not self.work_dir:
            self.work_dir = os.path.join("runs", "multitask_vmunet_arch", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
