"""
model.py — Các biến thể kiến trúc của VM-UNet + Spatial. Đầu ra LUÔN là [B, 2 + 2 + n_sigma + 1, H, W] như `SpatialOnlyVMUNet`
(thân, bệnh, offset x/y, σ, seed), nên loss, hàm `split`, suy luận và chấm điểm dùng lại nguyên.

    base / deep3   cùng lớp `SpatialOnlyVMUNet` (deep3 chỉ khác `depths` trong config)
    branch         BranchVMUNet : encoder và 2 tầng decoder đầu dùng chung; 2 tầng decoder cuối + Final_PatchExpand nhân đôi.
                   Nhánh chính ra thân/bệnh (2 kênh), nhánh embedding ra offset/σ/seed (2 + n_sigma + 1 kênh).
    x0skip         X0SkipVMUNet : cộng thêm skip 64×64 (đầu ra patch embed, baseline bỏ phí) vào đầu ra tầng decoder cuối.

Không sửa `models/vmunet`: các lớp con gọi lại các khối của VSSM (`forward_features`, `layers_up`, `final_up`, `final_conv`).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.multitask_vmunet_spatialonly.model import N_BASE, SpatialOnlyVMUNet

__all__ = ["BranchVMUNet", "X0SkipVMUNet", "build_model", "count_params"]


def _init_embedding_conv(conv: nn.Conv2d, n_sigma: int) -> None:
    """Giống SpatialOnlyVMUNet: offset khởi đầu = 0 (weight 0, bias 0), σ thô khởi đầu = 1 (weight 0, bias 1); kênh seed để mặc định."""
    with torch.no_grad():
        conv.weight[: 2 + n_sigma].fill_(0)
        conv.bias[:2].fill_(0)
        conv.bias[2: 2 + n_sigma].fill_(1)


def _resize_like(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return out if out.shape[-2:] == x.shape[-2:] else F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class BranchVMUNet(SpatialOnlyVMUNet):
    """Tách decoder: chung tới hết `layers_up[1]` (16×16), rồi hai nhánh độc lập (32×32 → 64×64 → 256×256)."""

    N_SHARED = 2                                          # số tầng decoder dùng chung (layers_up[0], layers_up[1])

    def __init__(self, n_sigma: int = 2, depths: list[int] | None = None, depths_decoder: list[int] | None = None,
                 drop_path_rate: float = 0.2):
        super().__init__(n_sigma, depths, depths_decoder, drop_path_rate)
        bb = self.backbone
        self.emb_layers_up = nn.ModuleList([copy.deepcopy(bb.layers_up[i]) for i in range(self.N_SHARED, len(bb.layers_up))])
        self.emb_final_up = copy.deepcopy(bb.final_up)
        c_in = bb.final_conv.in_channels
        bb.final_conv = nn.Conv2d(c_in, N_BASE, 1)                                  # nhánh chính: chỉ thân, bệnh
        self.emb_final_conv = nn.Conv2d(c_in, 2 + n_sigma + 1, 1)                   # nhánh embedding: offset, σ, seed
        _init_embedding_conv(self.emb_final_conv, n_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        h, skip = bb.forward_features(x)
        for i in range(self.N_SHARED):
            h = bb.layers_up[i](h if i == 0 else h + skip[-i])
        hm, he = h, h
        for j, i in enumerate(range(self.N_SHARED, len(bb.layers_up))):
            hm = bb.layers_up[i](hm + skip[-i])
            he = self.emb_layers_up[j](he + skip[-i])
        main = bb.final_conv(bb.final_up(hm).permute(0, 3, 1, 2))
        emb = self.emb_final_conv(self.emb_final_up(he).permute(0, 3, 1, 2))
        return _resize_like(torch.cat([main, emb], dim=1), x)


class X0SkipVMUNet(SpatialOnlyVMUNet):
    """Cộng `skip_list[0]` (64×64×96, đầu ra patch embed) vào đầu ra tầng decoder cuối, như các skip còn lại (cộng, không nối kênh)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        h, skip = bb.forward_features(x)
        h = bb.forward_features_up(h, skip)
        h = bb.forward_final(h + skip[0])
        return _resize_like(h, x)


def build_model(cfg) -> SpatialOnlyVMUNet:
    args = (cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate)
    if cfg.arch == "branch":
        return BranchVMUNet(*args)
    if cfg.arch == "x0skip":
        return X0SkipVMUNet(*args)
    return SpatialOnlyVMUNet(*args)                                                  # base, deep3


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
