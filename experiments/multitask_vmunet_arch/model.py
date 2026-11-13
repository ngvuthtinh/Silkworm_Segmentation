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

__all__ = ["AttnVMUNet", "BranchVMUNet", "X0SkipVMUNet", "build_model", "count_params"]


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


def _sincos_2d(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mã vị trí sin/cos 2D cố định, [h*w, dim]; tính theo cỡ bản đồ nên đổi cỡ ảnh (256² → 512²) vẫn dùng được."""
    q = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    ang_y, ang_x = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
    return torch.cat([ang_y.sin(), ang_y.cos(), ang_x.sin(), ang_x.cos()], dim=1).to(dtype)


class GlobalAttnBlock(nn.Module):
    """Một khối transformer (pre-norm self-attention + MLP) trên bản đồ [B, H, W, C]: mọi vị trí trao đổi thông tin với mọi vị trí.
    Mã vị trí chỉ cộng vào q, k. Lớp chiếu ra của attention và của MLP khởi tạo = 0 nên lúc bắt đầu khối là phép đồng nhất (mạng khởi động như baseline)."""

    def __init__(self, dim: int, heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        t = self.norm1(x).reshape(b, h * w, c)
        qk = t + _sincos_2d(h, w, c, t.device, t.dtype)
        x = x + self.attn(qk, qk, t, need_weights=False)[0].reshape(b, h, w, c)
        return x + self.mlp(self.norm2(x))


class AttnVMUNet(SpatialOnlyVMUNet):
    """Thêm một khối self-attention toàn cục ở bản đồ 16×16×384 (đầu ra `layers_up[1]`, trước khi cộng skip tiếp theo), nơi các mảnh của một con
    có thể trao đổi thông tin để thống nhất một tâm. Phần còn lại giống baseline."""

    def __init__(self, n_sigma: int = 2, depths: list[int] | None = None, depths_decoder: list[int] | None = None,
                 drop_path_rate: float = 0.2):
        super().__init__(n_sigma, depths, depths_decoder, drop_path_rate)
        self.global_attn = GlobalAttnBlock(384)                                      # kênh của bản đồ 16×16 (dims 96/192/384/768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        h, skip = bb.forward_features(x)
        for i, layer_up in enumerate(bb.layers_up):
            h = layer_up(h) if i == 0 else layer_up(h + skip[-i])
            if i == 1:                                                           # đầu ra tầng decoder thứ hai: 16×16×384
                h = self.global_attn(h)
        return _resize_like(bb.forward_final(h), x)


def build_model(cfg) -> SpatialOnlyVMUNet:
    args = (cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate)
    if cfg.arch == "attn":
        return AttnVMUNet(*args)
    if cfg.arch == "branch":
        return BranchVMUNet(*args)
    if cfg.arch == "x0skip":
        return X0SkipVMUNet(*args)
    return SpatialOnlyVMUNet(*args)                                                  # base, deep3


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
