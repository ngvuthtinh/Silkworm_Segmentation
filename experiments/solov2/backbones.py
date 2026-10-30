"""
backbones.py — Hai backbone dựa trên VM-UNet (VSSM) cho SOLOv2, đăng ký vào detectron2.

Cách A — `build_mamba_backbone`: encoder VSSM (4 tầng, bước 4/8/16/32, kênh 96/192/384/768, giống hệt res2..res5 của ResNet)
         thay ResNet; FPN của detectron2 giữ nguyên → p2..p6. Giống cách VMamba làm detection.
Cách B — `build_vmunet_backbone`: CẢ VM-UNet. Decoder của VM-UNet vốn là nhánh từ trên xuống có skip connection (như FPN), nên
         lấy đầu ra 4 tầng decoder (bước 32/16/8/4) làm p5..p2 (+ conv 1×1 và 3×3 về 256 kênh như đầu ra FPN), p6 = max-pool của p5.

Cả hai chỉ đọc mã VSSM ở models/vmunet (không sửa). Tensor bên trong VSSM có dạng kênh cuối (B, H, W, C).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from torch import nn

from experiments.solov2 import adet_shim  # noqa: F401  (đặt sys.path cho models/vmunet)
from vmunet.vmamba import VSSM

DIMS = [96, 192, 384, 768]


def _vssm(cfg) -> VSSM:
    return VSSM(in_chans=3, num_classes=1, depths=list(cfg.MODEL.VSSM.DEPTHS), depths_decoder=list(cfg.MODEL.VSSM.DEPTHS_DECODER),
                dims=DIMS, drop_path_rate=cfg.MODEL.VSSM.DROP_PATH_RATE)


class VSSMEncoder(Backbone):
    """Encoder VSSM → {res2, res3, res4, res5}; lấy đặc trưng SAU các khối VSS, TRƯỚC khi gộp patch xuống tầng sau."""

    def __init__(self, cfg):
        super().__init__()
        vssm = _vssm(cfg)
        self.patch_embed, self.pos_drop, self.layers = vssm.patch_embed, vssm.pos_drop, vssm.layers   # bỏ decoder: không dùng
        self.out_norms = nn.ModuleList(nn.LayerNorm(d) for d in DIMS)
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = dict(zip(self._out_features, DIMS))
        self._out_feature_strides = dict(zip(self._out_features, [4, 8, 16, 32]))

    @property
    def size_divisibility(self) -> int:
        return 32

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pos_drop(self.patch_embed(x))
        outs = {}
        for i, layer in enumerate(self.layers):
            for blk in layer.blocks:
                x = blk(x)
            outs[self._out_features[i]] = self.out_norms[i](x).permute(0, 3, 1, 2).contiguous()
            if layer.downsample is not None:
                x = layer.downsample(x)
        return outs

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {k: ShapeSpec(channels=self._out_feature_channels[k], stride=self._out_feature_strides[k]) for k in self._out_features}


@BACKBONE_REGISTRY.register()
def build_mamba_backbone(cfg, input_shape: ShapeSpec) -> Backbone:
    """Cách A: encoder VSSM + FPN → p2..p6."""
    return FPN(bottom_up=VSSMEncoder(cfg), in_features=cfg.MODEL.FPN.IN_FEATURES, out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
               norm=cfg.MODEL.FPN.NORM, top_block=LastLevelMaxPool(), fuse_type=cfg.MODEL.FPN.FUSE_TYPE)


class VMUNetPyramid(Backbone):
    """Cách B: cả VM-UNet; p5..p2 = đầu ra 4 tầng decoder (đã cộng skip từ encoder), p6 = max-pool(p5)."""

    def __init__(self, cfg):
        super().__init__()
        vssm = _vssm(cfg)
        del vssm.final_up, vssm.final_conv                                  # đầu ra phân đoạn của VM-UNet không dùng
        self.vssm = vssm
        c = cfg.MODEL.FPN.OUT_CHANNELS
        dec_dims = DIMS[::-1]                                               # 768, 384, 192, 96 = bước 32, 16, 8, 4
        self.out_norms = nn.ModuleList(nn.LayerNorm(d) for d in dec_dims)   # chuẩn hoá đặc trưng decoder (như cách A); thiếu nó mask không học được
        self.lateral = nn.ModuleList(Conv2d(d, c, 1) for d in dec_dims)
        self.output = nn.ModuleList(Conv2d(c, c, 3, padding=1) for _ in dec_dims)
        for conv in [*self.lateral, *self.output]:
            weight_init.c2_xavier_fill(conv)                                # khởi tạo như FPN của detectron2
        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {k: c for k in self._out_features}
        self._out_feature_strides = dict(zip(self._out_features, [4, 8, 16, 32, 64]))

    @property
    def size_divisibility(self) -> int:
        return 32

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x, skips = self.vssm.forward_features(x)
        dec = []
        for i, layer_up in enumerate(self.vssm.layers_up):                 # giống VSSM.forward_features_up, giữ từng tầng
            x = layer_up(x) if i == 0 else layer_up(x + skips[-i])
            dec.append(x)
        p5, p4, p3, p2 = (out(lat(norm(f).permute(0, 3, 1, 2).contiguous()))
                          for f, norm, lat, out in zip(dec, self.out_norms, self.lateral, self.output))
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": F.max_pool2d(p5, kernel_size=1, stride=2)}

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {k: ShapeSpec(channels=self._out_feature_channels[k], stride=self._out_feature_strides[k]) for k in self._out_features}


@BACKBONE_REGISTRY.register()
def build_vmunet_backbone(cfg, input_shape: ShapeSpec) -> Backbone:
    """Cách B: cả VM-UNet làm backbone + neck → p2..p6."""
    return VMUNetPyramid(cfg)
