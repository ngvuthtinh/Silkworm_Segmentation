"""
model.py — Segmentation-First Multi-Task Swin-Unet Architecture.

Architecture:
    Backbone & Decoder: SwinTransformerSys (Shifted Windows U-Net)
    Seg Head: Native Patch Expanding decoder output (Conv2d 1x1 -> 1 channel)
    Cls Head: Bottleneck feature (768) -> Global Average Pooling -> Linear(128) -> ReLU -> Dropout -> Linear(2)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve Swin-Unet import path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_SWIN_DIR = _PROJECT_ROOT / "Swin-Unet"
sys.path.insert(0, str(_SWIN_DIR))
sys.path.insert(0, str(_HERE))

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from config import SegFirstSwinConfig


class ClassificationHead(nn.Module):
    """
    Classification head attached to the bottleneck feature map:
    Global Average Pooling -> Linear(in_channels, hidden) -> ReLU -> Dropout -> Linear(hidden, num_classes).
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, L, C] -> mean over tokens -> [B, C]
        if x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class SegFirstSwinUnet(nn.Module):
    """
    Segmentation-First Multi-Task Swin-Unet.

    Maintains the Swin-Unet decoder for segmentation quality
    and attaches a lightweight classification head to the bottleneck features.
    """

    def __init__(
        self,
        config: SegFirstSwinConfig | None = None,
        input_size: int = 224,
        num_seg_classes: int = 1,
        num_cls_classes: int = 2,
        cls_hidden: int = 128,
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        if config is None:
            config = SegFirstSwinConfig()

        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        # Setup SwinTransformerSys backbone & decoder
        self.swin_unet = SwinTransformerSys(
            img_size=input_size,
            patch_size=4,
            in_chans=3,
            num_classes=num_seg_classes,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=config.drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )

        # Bottleneck dimension: 96 * 2^(4-1) = 768 for Swin-T
        bottleneck_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))
        self.cls_head = ClassificationHead(
            in_channels=bottleneck_dim,
            hidden=cls_hidden,
            num_classes=num_cls_classes,
            dropout=cls_dropout,
        )

    def forward(
        self, x: torch.Tensor, phase: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        phase == 1: returns (mask_logits, None)
        phase == 2: returns (mask_logits, class_logits)
        """
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. Forward encoder to bottleneck
        bottleneck, x_downsample = self.swin_unet.forward_features(x)  # [B, L, C]

        # 2. Forward decoder to segmentation mask
        x_up = self.swin_unet.forward_up_features(bottleneck, x_downsample)
        mask_logits = self.swin_unet.up_x4(x_up)  # [B, num_seg_classes, H, W]

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if phase == 1:
            return mask_logits, None

        # 3. Forward classification head from bottleneck
        class_logits = self.cls_head(bottleneck)

        return mask_logits, class_logits

    def get_parameter_groups(
        self, lr_backbone: float, lr_seg: float, lr_cls: float
    ) -> list[dict]:
        """
        Return parameter groups with differential learning rates to protect
        the backbone during Phase 2 multi-task fine-tuning.
        """
        backbone_params = []
        seg_head_params = []
        cls_head_params = list(self.cls_head.parameters())

        for name, param in self.swin_unet.named_parameters():
            if any(k in name for k in ["layers_up", "concat_back_dim", "norm_up", "up", "output"]):
                seg_head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": seg_head_params, "lr": lr_seg, "name": "seg_head"},
            {"params": cls_head_params, "lr": lr_cls, "name": "cls_head"},
        ]

    def load_from(self, pretrained_path: str) -> None:
        """Load pretrained Swin-T ImageNet weights."""
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained Swin-T backbone weights from: {pretrained_path}")
            device = next(self.parameters()).device
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" in pretrained_dict:
                pretrained_dict = pretrained_dict["model"]
            else:
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}

            model_dict = self.swin_unet.state_dict()
            full_dict = model_dict.copy()
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    full_dict[k] = v
            self.swin_unet.load_state_dict(full_dict, strict=False)
            print(" Pretrained backbone weights loaded successfully!\n")
