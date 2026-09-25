"""
model.py — ViT-P Point-based Segmentation Classifier Architecture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VITP_DINOV2_ROOT = PROJECT_ROOT / "models" / "ViT-P" / "ViT-P"

if str(VITP_DINOV2_ROOT) not in sys.path:
    sys.path.insert(0, str(VITP_DINOV2_ROOT))

from dinov2.models import vision_transformer as vits


class ViTPSegClassifier(nn.Module):
    """
    ViT-P Classifier:
    - Backbone: DinoVisionTransformer (ViT-Small or ViT-Base) with point positional embeddings
    - Head: Linear / MLP point classification head
    """

    def __init__(
        self,
        arch: str = "vit_small",
        num_classes: int = 2,
        num_points: int = 32,
        img_size: int = 224,
        patch_size: int = 14,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.img_size = img_size

        size_tuple = (img_size, img_size)
        if arch == "vit_small" or arch == "dinov2_vits14":
            self.backbone = vits.vit_small(
                patch_size=patch_size,
                n_points=num_points,
                img_size=size_tuple,
                num_classes=num_classes,
            )
            embed_dim = 384
            pretrained_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
        else:
            self.backbone = vits.vit_base(
                patch_size=patch_size,
                n_points=num_points,
                img_size=size_tuple,
                num_classes=num_classes,
            )
            embed_dim = 768
            pretrained_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, map_location="cpu")
            # Filter out cls_token if shape differs, load everything else
            model_dict = self.backbone.state_dict()
            filtered_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict, strict=False)

        # Classification head for sampled points
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
            points: (B, N, 2) normalized coordinates in [-1, 1]
        Returns:
            logits: (B, N, num_classes)
        """
        features = self.backbone.forward_features(images, points)
        point_tokens = features["x_norm_point_tokens"]  # (B, N, embed_dim)
        logits = self.head(point_tokens)                # (B, N, num_classes)
        return logits
