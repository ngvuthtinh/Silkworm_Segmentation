"""
model.py — VM-UNet với 3 + 2 + n_sigma + 1 kênh đầu ra theo pixel.

    kênh 0..2          — thân, biên, bệnh (y hệt baseline `MultiTaskVMUNet`; biên dùng cho đường watershed)
    kênh 3..4          — offset (x, y) trỏ về tâm con      → embedding = tanh(offset) + toạ độ
    kênh 5..5+n_sigma-1 — σ (độ rộng vùng gaussian của con, thô)
    kênh cuối          — seed (điểm "gần tâm", logit)

Chỉ khác baseline ở số kênh của lớp cuối. Khởi tạo lớp cuối như repo gốc: offset = 0, σ thô = 1 (xem BranchedERFNet.init_output).
"""

from __future__ import annotations

import torch

from experiments.multitask_vmunet.model import BODY, BOUNDARY, DISEASE, MultiTaskVMUNet

N_BASE = 3                              # thân, biên, bệnh

__all__ = ["BODY", "BOUNDARY", "DISEASE", "N_BASE", "SpatialEmbVMUNet"]


class SpatialEmbVMUNet(MultiTaskVMUNet):
    def __init__(
        self,
        n_sigma: int = 2,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
    ):
        super().__init__(depths, depths_decoder, drop_path_rate, num_outputs=N_BASE + 2 + n_sigma + 1)
        self.n_sigma = n_sigma
        conv = self.backbone.final_conv                       # Conv2d 1×1: [out, in, 1, 1]
        with torch.no_grad():
            conv.weight[N_BASE:N_BASE + 2 + n_sigma].fill_(0)
            conv.bias[N_BASE:N_BASE + 2].fill_(0)             # offset khởi đầu = 0
            conv.bias[N_BASE + 2:N_BASE + 2 + n_sigma].fill_(1)   # σ thô khởi đầu = 1

    def split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 3 + 2 + n + 1, H, W] → (logit thân/biên/bệnh [B, 3, H, W], spatial [B, 2 + n + 1, H, W] = offset, σ, seed)."""
        return out[:, :N_BASE], out[:, N_BASE:]
