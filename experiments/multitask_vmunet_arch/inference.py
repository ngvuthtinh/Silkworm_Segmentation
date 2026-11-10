"""
experiments/multitask_vmunet_arch/inference.py — nạp model (theo `arch` trong config.yaml của run) và dự đoán cho một ảnh RGB bất kỳ.
`predict` dùng lại nguyên của spatialonly (cùng định dạng đầu ra).
"""

from __future__ import annotations

import torch

from experiments.multitask_vmunet_arch.config import ArchConfig
from experiments.multitask_vmunet_arch.model import build_model
from experiments.multitask_vmunet_spatialonly.inference import predict  # noqa: F401


def load_model(ckpt: str, cfg: ArchConfig, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()
