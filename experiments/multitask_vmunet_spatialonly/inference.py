"""
experiments/multitask_vmunet_spatialonly/inference.py — nạp model và dự đoán cho một ảnh RGB bất kỳ.

Các script vẽ ảnh (Reports/Meeting_2026-10-30/results/comparison/source/make_figures_spatialonly.py, dense_photos.py) dùng hai hàm này.
Tách con và vẽ dùng `spatial_instances` (evaluate.py) và `src.visualize_instances.draw_instances_colored`.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from experiments.multitask_vmunet_spatialonly.config import SpatialOnlyConfig
from experiments.multitask_vmunet_spatialonly.model import SpatialOnlyVMUNet
from src.dataset_multitask import _normalize


def load_model(ckpt: str, cfg: SpatialOnlyConfig, device: torch.device) -> SpatialOnlyVMUNet:
    model = SpatialOnlyVMUNet(cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def predict(model: SpatialOnlyVMUNet, rgb: np.ndarray, cfg: SpatialOnlyConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """rgb [H, W, 3] uint8 → (xác suất [2, S, S] thân/bệnh, đầu ra thô [2 + n + 1, S, S] offset/σ/seed)."""
    s = cfg.img_size
    x = _normalize(cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR) / 255.0)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
        logits, spatial = model.split(model(t))
    return torch.sigmoid(logits.float())[0].cpu().numpy(), spatial.float()[0].cpu().numpy()
