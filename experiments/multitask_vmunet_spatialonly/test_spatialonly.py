#!/usr/bin/env python3
"""
experiments/multitask_vmunet_spatialonly/test_spatialonly.py — kiểm tra nhanh model không biên và BodyDiseaseLoss.

  python -m experiments.multitask_vmunet_spatialonly.test_spatialonly      # cần GPU cho phần model (VM-UNet dùng kernel CUDA)
"""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.multitask_vmunet_spatialonly.config import SpatialOnlyConfig
from experiments.multitask_vmunet_spatialonly.model import N_BASE, SpatialOnlyVMUNet
import numpy as np

from src.dataset_instance import InstanceSilkwormDataset
from src.losses import BodyDiseaseLoss, InstanceMultiTaskLoss
from src.metrics import gt_instance_classes, match_instances


def test_config() -> None:
    cfg = SpatialOnlyConfig()
    assert cfg.lambda_boundary == 0.0 and cfg.n_sigma == 2 and cfg.lambda_spatial == 1.0 and cfg.w_var == 10.0
    print("OK  config: λ_boundary = 0, các tham số spatial giữ như SpatialEmbConfig")


def test_loss_matches_base_without_boundary() -> None:
    """BodyDiseaseLoss = InstanceMultiTaskLoss với λ_boundary = 0 (cùng thân và bệnh)."""
    g = torch.Generator().manual_seed(0)
    b, h, w = 2, 24, 24
    logits3 = torch.randn(b, 3, h, w, generator=g)
    body = (torch.rand(b, 1, h, w, generator=g) > 0.5).float()
    bnd = (torch.rand(b, 1, h, w, generator=g) > 0.9).float()
    dis = (torch.rand(b, 1, h, w, generator=g) > 0.5).float()
    valid = torch.ones(b, 1, h, w)
    valid[:, :, :4] = 0
    want, l_body, _, l_dis = InstanceMultiTaskLoss(1.0, 0.0, 1.0)(logits3, body, bnd, dis, valid)
    logits2 = logits3[:, [0, 2]].clone().requires_grad_(True)
    got, g_body, g_dis = BodyDiseaseLoss(1.0, 1.0)(logits2, body, dis, valid)
    assert torch.allclose(got, want) and torch.allclose(g_body, l_body) and torch.allclose(g_dis, l_dis)
    got.backward()
    assert logits2.grad[:, :, :4].abs().sum() == 0, "vùng bỏ qua không được nhận gradient"
    empty, *_ = BodyDiseaseLoss()(torch.randn(1, 2, 8, 8, requires_grad=True), torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    assert torch.isfinite(empty)
    print("OK  BodyDiseaseLoss khớp InstanceMultiTaskLoss (λ_boundary = 0), vùng bỏ qua không có gradient, ảnh trống ổn")


def test_model_channels() -> None:
    if not torch.cuda.is_available():
        print("BỎ QUA  kiểm tra model (không có GPU)")
        return
    model = SpatialOnlyVMUNet(n_sigma=2).cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256).cuda())
        logits, sp = model.split(out)
    assert out.shape == (1, 2 + 2 + 2 + 1, 256, 256), out.shape
    assert logits.shape == (1, N_BASE, 256, 256) and sp.shape == (1, 5, 256, 256)
    assert sp[:, 0:2].abs().max() < 1e-6, "offset khởi đầu phải bằng 0 (init lớp cuối)"
    assert torch.allclose(sp[:, 2:4], torch.ones_like(sp[:, 2:4])), "σ thô khởi đầu phải bằng 1"
    print(f"OK  model: {tuple(out.shape)} = thân, bệnh | offset ×2, σ ×2, seed; init offset = 0, σ = 1")


def test_unknown_disease() -> None:
    """Ảnh không có nhãn bệnh (disease = -1): loss bệnh bỏ qua, chấm điểm không tính sai, data thật đọc đúng."""
    b, h, w = 2, 16, 16
    logits3 = torch.randn(b, 3, h, w, requires_grad=True)
    body = torch.ones(b, 1, h, w)
    bnd = torch.zeros(b, 1, h, w)
    dis = torch.full((b, 1, h, w), -1.0)
    dis[0] = 1.0                                              # ảnh 0 có nhãn bệnh, ảnh 1 chưa biết
    total, _, _, l_dis = InstanceMultiTaskLoss()(logits3, body, bnd, dis)
    total.backward()
    assert torch.isfinite(total) and logits3.grad[1, 2].abs().sum() == 0, "pixel chưa biết bệnh không được nhận gradient bệnh"
    assert logits3.grad[0, 2].abs().sum() > 0
    all_unknown = InstanceMultiTaskLoss()(torch.randn(b, 3, h, w), body, bnd, torch.full((b, 1, h, w), -1.0))[3]
    assert float(all_unknown) == 0.0, "toàn ảnh chưa biết bệnh → loss bệnh = 0, không NaN"
    l2 = torch.randn(b, 2, h, w, requires_grad=True)
    BodyDiseaseLoss()(l2, body, dis)[0].backward()
    assert l2.grad[1, 1].abs().sum() == 0 and l2.grad[0, 1].abs().sum() > 0

    gi = np.zeros((20, 20), int); gi[:, :10] = 1; gi[:, 10:] = 2
    gd = np.zeros((20, 20), np.float32); gd[:, :10] = -1.0; gd[:, 10:] = 1.0     # con 1: chưa biết; con 2: bệnh
    gcls = gt_instance_classes(gi, gd)
    assert gcls == {1: -1, 2: 0}, gcls
    wrong = {1: 0, 2: 1}                                      # đoán sai cả hai nhãn: con chưa biết vẫn "đúng", con bệnh sai
    pred = gi.copy()
    c = match_instances(pred, wrong, gi, gcls, min_area=10)
    assert (c["tp"], c["cls_correct"], c["gt_grasserie"], c["hit_grasserie"]) == (2, 1, 1, 0), c
    ds_root = Path("data/dense_seg_v1")
    if ds_root.exists():
        ds = InstanceSilkwormDataset(ds_root, "train", 256, augment=False)
        _, body_t, _, dis_t, inst_t, _ = ds[0]
        assert (dis_t[body_t > 0.5] == -1).all() and (dis_t[body_t < 0.5] == 0).all() and inst_t.max() > 0
        print(f"OK  chưa biết bệnh: loss bỏ qua, chấm điểm không phạt; data/dense_seg_v1 đọc đúng ({len(ds)} ảnh train)")
    else:
        print("OK  chưa biết bệnh: loss bỏ qua, chấm điểm không phạt")


if __name__ == "__main__":
    test_config()
    test_loss_matches_base_without_boundary()
    test_model_channels()
    test_unknown_disease()
    print("\nTất cả kiểm tra đều đạt.")
