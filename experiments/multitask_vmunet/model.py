"""
model.py — Multi-Task VM-UNet

Architecture:
    Backbone : VSSM (from models/vmunet/models/vmunet/vmamba.py)
    Seg head : 1×1 Conv → sigmoid  → mask_pred  [B, 1, H, W]
    Cls head : GAP → FC(256) → FC(2) → class_logits [B, 2]

The bottleneck feature (deepest encoder stage) is extracted via a
forward hook registered on `VSSM.layers[3]` so we do NOT modify
vmamba.py at all.

Returns:
    (mask_pred, class_logits)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_VMUNET_DIR = _PROJECT_ROOT / "models" / "vmunet"
if str(_VMUNET_DIR) not in sys.path:
    sys.path.insert(0, str(_VMUNET_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.vmunet.vmamba import VSSM


class ClassificationHead(nn.Module):
    """
    Global-Average-Pool → FC(hidden) → ReLU → Dropout → FC(num_classes).
    """

    def __init__(self, in_channels: int, hidden: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.head(x)


class SegmentationHead(nn.Module):
    """1×1 conv for binary segmentation."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiTaskVMUNet(nn.Module):
    """
    VM-UNet backbone with two task heads:
        1. Segmentation head  → mask_pred   (binary sigmoid map)
        2. Classification head → class_logits (healthy / diseased)
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_seg_classes: int = 1,
        num_cls_classes: int = 2,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
        load_ckpt_path: str | None = None,
        cls_hidden: int = 256,
        cls_dropout: float = 0.3,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 2, 2]
        if depths_decoder is None:
            depths_decoder = [2, 2, 2, 1]

        self.num_seg_classes = num_seg_classes
        self.load_ckpt_path = load_ckpt_path

        self.backbone = VSSM(
            in_chans=input_channels,
            num_classes=num_seg_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

        self._bottleneck_feat: torch.Tensor | None = None
        self._hook_handle = self.backbone.layers[3].register_forward_hook(
            self._capture_bottleneck
        )

        bottleneck_channels = self._infer_bottleneck_channels()
        self.cls_head = ClassificationHead(bottleneck_channels, cls_hidden, num_cls_classes, cls_dropout)

    def _infer_bottleneck_channels(self) -> int:
        try:
            embed_dim = self.backbone.embed_dim
            num_stages = len(self.backbone.layers)
            return embed_dim * (2 ** (num_stages - 1))
        except AttributeError:
            return self._probe_bottleneck_channels()

    def _probe_bottleneck_channels(self) -> int:
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        self.backbone.eval()
        with torch.no_grad():
            self.backbone(dummy)
        self.backbone.train()
        assert self._bottleneck_feat is not None, "Hook did not fire."
        feat = self._bottleneck_feat
        if feat.dim() in [3, 4]:
            return feat.shape[-1]
        raise RuntimeError(f"Unexpected bottleneck shape: {feat.shape}")

    def _capture_bottleneck(self, module, input, output):
        self._bottleneck_feat = output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_logits = self.backbone(x)

        if self.num_seg_classes == 1:
            mask_pred = torch.sigmoid(mask_logits)
        else:
            mask_pred = torch.softmax(mask_logits, dim=1)

        if mask_pred.shape[-2:] != x.shape[-2:]:
            mask_pred = F.interpolate(
                mask_pred, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        bot_feat = self._bottleneck_feat
        bot_feat = self._to_spatial(bot_feat, x.shape[-2:])
        class_logits = self.cls_head(bot_feat)

        return mask_pred, class_logits

    @staticmethod
    def _to_spatial(feat: torch.Tensor, img_hw: tuple[int, int]) -> torch.Tensor:
        if feat.dim() == 4:
            if feat.shape[1] != feat.shape[3]:
                return feat.permute(0, 3, 1, 2)
            return feat

        if feat.dim() == 3:
            B, L, C = feat.shape
            H_img, W_img = img_hw
            ratio = H_img / W_img
            W_prime = int((L / ratio) ** 0.5)
            H_prime = L // W_prime
            if H_prime * W_prime != L:
                H_prime = W_prime = int(L ** 0.5)
            feat = feat.permute(0, 2, 1).reshape(B, C, H_prime, W_prime)
            return feat

        raise RuntimeError(f"Cannot convert bottleneck feature of shape {feat.shape} to spatial.")

    def load_pretrained(self) -> None:
        if self.load_ckpt_path is None or not os.path.exists(self.load_ckpt_path):
            print("[MultiTaskVMUNet] No pretrained checkpoint loaded.")
            return

        model_dict = self.backbone.state_dict()
        checkpoint = torch.load(self.load_ckpt_path, map_location="cpu")
        pretrained_dict = checkpoint.get("model", checkpoint)

        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(new_dict)
        print(f"[Pretrained] Encoder: loaded {len(new_dict)}/{len(pretrained_dict)} keys")

        mirror_dict = {}
        for k, v in pretrained_dict.items():
            for i in range(4):
                if f"layers.{i}" in k:
                    new_k = k.replace(f"layers.{i}", f"layers_up.{3 - i}")
                    mirror_dict[new_k] = v
                    break
        new_dict2 = {k: v for k, v in mirror_dict.items() if k in model_dict}
        model_dict.update(new_dict2)
        print(f"[Pretrained] Decoder mirror: loaded {len(new_dict2)} keys")

        self.backbone.load_state_dict(model_dict)
        print("[Pretrained] Weights loaded successfully.")

    def remove_hooks(self) -> None:
        self._hook_handle.remove()
