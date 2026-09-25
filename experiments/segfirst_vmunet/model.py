"""
model.py — Segmentation-First Multi-Task VM-UNet Architecture

Architecture:
    Backbone: VSSM (VMamba encoder-decoder)
    Seg Head: Native VM-UNet decoder output (1×1 Conv → sigmoid)
    Cls Head: Bottleneck feature → AdaptiveAvgPool2d → FC(128) → ReLU → Dropout → FC(2)

Features:
    - Forward hook captures deepest encoder stage (bottleneck feature).
    - Phase 1 & Phase 2 forward pass flexibility.
    - Parameter group builder for differential learning rates (protects backbone).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve VM-UNet import path (models/vmunet/ at project root)
_HERE = Path(__file__).resolve().parent                         # experiments/segfirst_vmunet/
_PROJECT_ROOT = _HERE.parent.parent                             # Silkworm_Segmentation/
_VMUNET_DIR = _PROJECT_ROOT / "models" / "vmunet"              # models/vmunet/ (repo gốc)
sys.path.insert(0, str(_VMUNET_DIR))

from models.vmunet.vmamba import VSSM  # noqa: E402


class ClassificationHead(nn.Module):
    """
    Classification head attached to the bottleneck feature map:
    Global Average Pooling → FC(hidden=128) → ReLU → Dropout → FC(num_classes=2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.fc(x)


class SegFirstVMUNet(nn.Module):
    """
    Segmentation-First Multi-Task VM-UNet.

    Prioritizes segmentation quality by maintaining the original VM-UNet decoder
    for segmentation and attaching a lightweight classification head to the bottleneck.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_seg_classes: int = 1,
        num_cls_classes: int = 2,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
        cls_hidden: int = 128,
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 2, 2]
        if depths_decoder is None:
            depths_decoder = [2, 2, 2, 1]

        self.num_seg_classes = num_seg_classes

        # --- Backbone ---
        self.backbone = VSSM(
            in_chans=input_channels,
            num_classes=num_seg_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

        # --- Bottleneck Feature Hook ---
        self._bottleneck_feat: torch.Tensor | None = None
        self._hook_handle = self.backbone.layers[3].register_forward_hook(
            self._capture_bottleneck
        )

        # Infer bottleneck channels (768 for default embed_dim=96)
        bottleneck_channels = self._infer_bottleneck_channels()

        # --- Task Heads ---
        self.cls_head = ClassificationHead(
            in_channels=bottleneck_channels,
            hidden=cls_hidden,
            num_classes=num_cls_classes,
            dropout=cls_dropout,
        )

    def _infer_bottleneck_channels(self) -> int:
        try:
            embed_dim = self.backbone.embed_dim
            num_stages = len(self.backbone.layers)
            return embed_dim * (2 ** (num_stages - 1))
        except AttributeError:
            return 768

    def _capture_bottleneck(self, module, input, output):  # noqa: A002
        self._bottleneck_feat = output

    def forward(
        self, x: torch.Tensor, phase: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Parameters
        ----------
        x : [B, 3, H, W]
        phase : int (1 = seg only, 2 = multi-task)

        Returns
        -------
        mask_pred    : [B, 1, H, W] (sigmoid probability map)
        class_logits : [B, 2] (or None if phase=1)
        """
        # 1. Run backbone (fires hook at bottleneck stage)
        mask_logits = self.backbone(x)

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        if phase == 1:
            return mask_logits, None

        # 2. Classification prediction from bottleneck feature
        bot_feat = self._bottleneck_feat
        bot_feat = self._to_spatial(bot_feat)
        class_logits = self.cls_head(bot_feat)

        return mask_logits, class_logits

    @staticmethod
    def _to_spatial(feat: torch.Tensor) -> torch.Tensor:
        """Convert VSSM feature tensor [B, H', W', C] to [B, C, H', W']."""
        if feat.dim() == 4:
            if feat.shape[1] != feat.shape[3] and feat.shape[3] in {96, 192, 384, 768}:
                return feat.permute(0, 3, 1, 2).contiguous()
            return feat
        elif feat.dim() == 3:
            B, L, C = feat.shape
            H = W = int(L**0.5)
            return feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return feat

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

        # Separate backbone vs segmentation final conv head
        for name, param in self.backbone.named_parameters():
            if "final_conv" in name:
                seg_head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": seg_head_params, "lr": lr_seg, "name": "seg_head"},
            {"params": cls_head_params, "lr": lr_cls, "name": "cls_head"},
        ]

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters to protect segmentation features."""
        for name, param in self.backbone.named_parameters():
            if "final_conv" not in name:
                param.requires_grad = not freeze
