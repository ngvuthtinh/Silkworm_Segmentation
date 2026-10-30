"""
model.py — VM-UNet với 3 + D kênh đầu ra theo pixel.

    kênh 0..2  — thân, biên, bệnh (y hệt baseline `MultiTaskVMUNet`)
    kênh 3..   — embedding D chiều (không qua sigmoid), học bằng DiscriminativeLoss

Chỉ khác baseline ở số kênh của lớp cuối; backbone, pretrained và forward dùng lại từ baseline.
"""

from __future__ import annotations

import torch

from experiments.multitask_vmunet.model import BODY, BOUNDARY, DISEASE, MultiTaskVMUNet

N_BASE = 3                              # số kênh đầu ra cũ (thân, biên, bệnh)

__all__ = ["BODY", "BOUNDARY", "DISEASE", "N_BASE", "DiscVMUNet"]


class DiscVMUNet(MultiTaskVMUNet):
    def __init__(
        self,
        n_embed: int = 8,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
    ):
        super().__init__(depths, depths_decoder, drop_path_rate, num_outputs=N_BASE + n_embed)
        self.n_embed = n_embed

    @staticmethod
    def split(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 3 + D, H, W] → (logit thân/biên/bệnh [B, 3, H, W], embedding [B, D, H, W])."""
        return out[:, :N_BASE], out[:, N_BASE:]
