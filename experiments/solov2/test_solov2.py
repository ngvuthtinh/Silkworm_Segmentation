#!/usr/bin/env python3
"""
experiments/solov2/test_solov2.py — kiểm tra nhanh trước khi train.

  python -m experiments.solov2.test_solov2      # cần GPU (VSSM dùng kernel CUDA selective_scan)
"""

from __future__ import annotations

from pathlib import Path

import torch
from fvcore.nn import sigmoid_focal_loss_jit

from experiments.solov2 import backbones  # noqa: F401
from experiments.solov2.config import SOLOv2Config, build_cfg
from experiments.solov2.data import image_record
from experiments.solov2.model import IGNORE, UNKNOWN, silk_cate_loss


def test_backbones() -> None:
    from detectron2.modeling import build_backbone
    for variant in ("mamba", "vmunet"):
        cfg = build_cfg(SOLOv2Config(variant=variant), ["x"], "y")
        bb = build_backbone(cfg).cuda().eval()
        with torch.no_grad():
            out = bb(torch.randn(2, 3, 256, 320, device="cuda"))
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        want = {"p2": (2, 256, 64, 80), "p3": (2, 256, 32, 40), "p4": (2, 256, 16, 20), "p5": (2, 256, 8, 10), "p6": (2, 256, 4, 5)}
        assert shapes == want, (variant, shapes)
        assert {k: s.stride for k, s in bb.output_shape().items()} == {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}
        n = sum(p.numel() for p in bb.parameters()) / 1e6
        print(f"OK  backbone {variant}: p2..p6 đúng cỡ (bước 4..64, 256 kênh), {n:.1f}M tham số")


def test_cate_loss() -> None:
    g = torch.Generator().manual_seed(0)
    preds = torch.randn(6, 2, generator=g, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 2, UNKNOWN, IGNORE])          # 2 = nền khi num_classes = 2
    loss = silk_cate_loss(preds, labels, 2, 0.25, 2.0)
    loss.backward()
    assert preds.grad[5].abs().sum() == 0, "ô IGNORE không được nhận gradient"
    assert (preds.grad[4] < 0).all(), "ô UNKNOWN: đẩy xác suất có tằm lên (gradient âm ở mọi lớp)"
    # không có UNKNOWN / IGNORE → đúng bằng focal loss gốc của SOLOv2
    p2 = torch.randn(5, 2, generator=g)
    l2 = torch.tensor([0, 1, 2, 2, 0])
    oh = torch.zeros(5, 2)
    oh[[0, 1, 4], [0, 1, 0]] = 1
    want = sigmoid_focal_loss_jit(p2, oh, gamma=2.0, alpha=0.25, reduction="sum")
    assert torch.allclose(silk_cate_loss(p2, l2, 2, 0.25, 2.0), want)
    print("OK  loss phân loại: IGNORE không có gradient, UNKNOWN được dạy là 'có tằm', còn lại khớp focal loss gốc")


def test_data() -> None:
    for root, split, want in (("data/sam3_aug20k_v3", "test_real", {0, 1}), ("data/dense_seg_v2", "test", {UNKNOWN})):
        imgs = sorted((Path(root) / split / "images").glob("*.jpg"))[:20]
        if not imgs:
            print(f"BỎ QUA  {root}/{split} không có")
            continue
        cats = set()
        for p in imgs:
            r = image_record(Path(root), split, p)
            cats |= {a["category_id"] for a in r["annotations"]}
            for an in r["annotations"]:
                x0, y0, x1, y1 = an["bbox"]
                assert 0 <= x0 < x1 <= r["width"] and 0 <= y0 < y1 <= r["height"]
        assert (cats - {IGNORE}) <= want and cats & want, (root, cats)
        print(f"OK  data {root}/{split}: lớp {sorted(cats)} (−1 = chưa biết bệnh, −2 = vùng bỏ qua)")


if __name__ == "__main__":
    test_cate_loss()
    test_data()
    test_backbones()
    print("\nTất cả kiểm tra đều đạt.")
