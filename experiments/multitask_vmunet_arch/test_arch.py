"""
Kiểm tra nhanh các biến thể kiến trúc (không cần data):
  1. đầu ra [B, 2 + 2 + n_sigma + 1, H, W] đúng kích thước, hữu hạn
  2. `base` có đúng số tham số của SpatialOnlyVMUNet; in số tham số các arch
  3. khởi tạo lớp cuối: offset = 0, σ thô = 1 (như baseline)
  4. lan truyền ngược chạm mọi tham số (không có nhánh chết)

  python -m experiments.multitask_vmunet_arch.test_arch --gpu 1
"""

from __future__ import annotations

import argparse

import torch

from experiments.multitask_vmunet_arch.config import ARCHS, ArchConfig
from experiments.multitask_vmunet_arch.model import build_model, count_params
from experiments.multitask_vmunet_spatialonly.model import SpatialOnlyVMUNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    n_base = count_params(SpatialOnlyVMUNet(2))
    print(f"baseline SpatialOnlyVMUNet: {n_base:.2f}M params")
    for arch in ARCHS:
        cfg = ArchConfig(arch=arch)
        model = build_model(cfg).to(dev).train()
        n = count_params(model)
        x = torch.randn(2, 3, a.size, a.size, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.autocast(device_type="cuda"):
            out = model(x)
        assert out.shape == (2, 5 + cfg.n_sigma, a.size, a.size), out.shape
        assert torch.isfinite(out.float()).all(), "đầu ra có NaN/Inf"
        logits, spatial = model.split(out)
        assert logits.shape[1] == 2 and spatial.shape[1] == 3 + cfg.n_sigma
        out.float().mean().backward()
        dead = [k for k, p in model.named_parameters() if p.grad is None]
        assert not dead, f"tham số không nhận gradient: {dead[:5]}"
        if arch == "base":
            assert abs(n - n_base) < 1e-9, "arch=base phải có đúng số tham số của baseline"
        model.zero_grad()
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            o0 = model(torch.zeros(1, 3, a.size, a.size, device=dev)).float()
        # khởi đầu: offset ≈ 0 và σ thô = 1 (trọng số lớp cuối của các kênh này = 0)
        assert o0[:, 2:4].abs().max() < 1e-3, "offset khởi đầu phải = 0"
        assert (o0[:, 4:4 + cfg.n_sigma] - 1).abs().max() < 1e-3, "σ thô khởi đầu phải = 1"
        print(f"  {arch:7s} OK | {n:6.2f}M params ({n - n_base:+.2f}M) | depths {cfg.depths} | out {tuple(out.shape)} | "
              f"đỉnh VRAM (fwd, batch 2) {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
        del model
        torch.cuda.empty_cache()
    print("TẤT CẢ OK")


if __name__ == "__main__":
    main()
