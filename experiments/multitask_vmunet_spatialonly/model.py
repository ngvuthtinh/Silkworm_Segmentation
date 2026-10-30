"""
model.py — VM-UNet với 2 + 2 + n_sigma + 1 kênh đầu ra theo pixel (không có kênh biên).

    kênh 0              — thân (logit)
    kênh 1              — bệnh (logit)
    kênh 2..3           — offset (x, y) trỏ về tâm con  → embedding = tanh(offset) + toạ độ
    kênh 4..4+n_sigma-1 — σ (độ rộng vùng gaussian của con, thô)
    kênh cuối           — seed (điểm "gần tâm", logit)

Cùng backbone và cùng khởi tạo lớp cuối với `SpatialEmbVMUNet` (offset = 0, σ thô = 1); chỉ thiếu kênh biên.
"""

from __future__ import annotations

import torch

from experiments.multitask_vmunet.model import MultiTaskVMUNet

BODY, DISEASE = 0, 1
N_BASE = 2                              # thân, bệnh

__all__ = ["BODY", "DISEASE", "N_BASE", "SpatialOnlyVMUNet"]


class SpatialOnlyVMUNet(MultiTaskVMUNet):
    def __init__(
        self,
        n_sigma: int = 2,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
    ):
        super().__init__(depths, depths_decoder, drop_path_rate, num_outputs=N_BASE + 2 + n_sigma + 1)
        self.n_sigma = n_sigma
        conv = self.backbone.final_conv
        with torch.no_grad():
            conv.weight[N_BASE:N_BASE + 2 + n_sigma].fill_(0)
            conv.bias[N_BASE:N_BASE + 2].fill_(0)                 # offset khởi đầu = 0
            conv.bias[N_BASE + 2:N_BASE + 2 + n_sigma].fill_(1)   # σ thô khởi đầu = 1

    def split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 2 + 2 + n + 1, H, W] → (logit thân/bệnh [B, 2, H, W], spatial [B, 2 + n + 1, H, W] = offset, σ, seed)."""
        return out[:, :N_BASE], out[:, N_BASE:]
