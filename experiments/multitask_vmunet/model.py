"""
model.py — VM-UNet multitask cho phân đoạn TỪNG CON + chẩn đoán TỪNG CON.

Một backbone VSSM (encoder + decoder dùng chung), đầu ra 3 kênh theo pixel:
    kênh 0 — thân tằm   : pixel có thuộc con tằm nào không
    kênh 1 — biên       : pixel có nằm trên viền của một con không (tách các con chạm/đè nhau)
    kênh 2 — bệnh       : pixel thuộc con bị Grasserie hay không

Từng con được tách và chẩn đoán ở bước hậu xử lý: src.metrics.split_instances.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VMUNET_DIR = _PROJECT_ROOT / "models" / "vmunet" / "models"
for p in (_VMUNET_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vmunet.vmamba import VSSM

BODY, BOUNDARY, DISEASE = 0, 1, 2


class MultiTaskVMUNet(nn.Module):
    def __init__(
        self,
        depths: list[int] | None = None,
        depths_decoder: list[int] | None = None,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.backbone = VSSM(
            in_chans=3,
            num_classes=3,
            depths=depths or [2, 2, 2, 2],
            depths_decoder=depths_decoder or [2, 2, 2, 1],
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logit thô [B, 3, H, W] (thân, biên, bệnh)."""
        logits = self.backbone(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def load_pretrained(self, ckpt_path: str) -> None:
        """Nạp trọng số VMamba cho encoder và soi gương sang decoder (như VM-UNet gốc)."""
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"[MultiTaskVMUNet] Không có pretrained ({ckpt_path}) → train từ đầu.")
            return
        model_dict = self.backbone.state_dict()
        pre = torch.load(ckpt_path, map_location="cpu")
        pre = pre.get("model", pre)
        enc = {k: v for k, v in pre.items() if k in model_dict and v.shape == model_dict[k].shape}
        dec = {}
        for k, v in pre.items():
            for i in range(4):
                if f"layers.{i}" in k:
                    nk = k.replace(f"layers.{i}", f"layers_up.{3 - i}")
                    if nk in model_dict and v.shape == model_dict[nk].shape:
                        dec[nk] = v
                    break
        model_dict.update(enc)
        model_dict.update(dec)
        self.backbone.load_state_dict(model_dict)
        print(f"[MultiTaskVMUNet] Pretrained: encoder {len(enc)} keys, decoder {len(dec)} keys.")
