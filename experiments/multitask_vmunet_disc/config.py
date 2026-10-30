"""
config.py — Siêu tham số cho VM-UNet + embedding (discriminative loss).

Kế thừa nguyên cấu hình baseline `MultiTaskConfig`, chỉ THÊM các tham số của embedding. Nhờ vậy mọi siêu tham số
khác (lr, epoch, batch, λ cũ, ngưỡng...) luôn trùng baseline và so sánh với baseline là "một thay đổi duy nhất".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from experiments.multitask_vmunet.config import MultiTaskConfig


@dataclass
class DiscConfig(MultiTaskConfig):
    # Embedding theo pixel (De Brabandere et al., 2017)
    n_embed: int = 8                   # số chiều embedding (paper dùng 8 cho Cityscapes, 16 cho CVPPP)
    lambda_embed: float = 1.0          # trọng số của discriminative loss trong tổng loss
    delta_var: float = 0.5             # biên kéo: pixel cách tâm cụm ≤ delta_var thì không bị kéo nữa
    delta_dist: float = 1.5            # biên đẩy: tâm hai cụm cách nhau ≥ 2·delta_dist thì không bị đẩy nữa (cần > 2·delta_var)

    # Hậu xử lý: gộp hai vùng rời nếu embedding trung bình cách nhau < merge_thr (xem src.metrics.merge_fragments)
    merge_thr: float = 1.5

    def __post_init__(self) -> None:
        if not self.work_dir:
            self.work_dir = os.path.join("runs", "multitask_vmunet_disc", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
