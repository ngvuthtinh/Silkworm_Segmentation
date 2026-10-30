#!/usr/bin/env python3
"""
experiments/multitask_vmunet_spatialemb/test_spatialemb.py — kiểm tra SpatialEmbeddingLoss và cluster_spatial_embeddings.

Chạy (từ thư mục gốc dự án, không cần GPU):
  python -m experiments.multitask_vmunet_spatialemb.test_spatialemb

Các phép so sánh với repo gốc (models/SpatialEmbeddings) chỉ dùng làm CHUẨN ĐỐI CHIẾU trong test, không được import ở code chính;
nếu repo không có trên máy thì các test đó bị bỏ qua.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from src.losses import SpatialEmbeddingLoss, lovasz_hinge_flat
from src.metrics import cluster_spatial_embeddings

REF_SRC = Path("models/SpatialEmbeddings/src")


def _load_reference():
    if not REF_SRC.exists():
        return None
    sys.path.insert(0, str(REF_SRC.resolve()))
    try:
        import criterions.my_loss as ref_loss
        from criterions.lovasz_losses import lovasz_hinge_flat as ref_flat
    except Exception as e:                                   # repo gốc cũ, thiếu phụ thuộc...
        print(f"BỎ QUA đối chiếu với repo gốc: {type(e).__name__}: {e}")
        return None
    orig = ref_loss.lovasz_hinge
    ref_loss.lovasz_hinge = lambda logits, mask: orig(logits, mask.float())   # repo gốc trừ trên tensor bool → lỗi ở PyTorch mới
    return ref_flat, ref_loss.SpatialEmbLoss


def _scene(h: int = 40, w: int = 48) -> torch.Tensor:
    """Hai con; con 1 gồm HAI MẢNH RỜI NHAU (như bị lá che) cùng id, con 2 một mảnh."""
    inst = torch.zeros(2, h, w, dtype=torch.long)
    inst[0, 3:12, 3:20] = 1
    inst[0, 26:36, 26:44] = 1                                # mảnh rời của con 1
    inst[0, 4:12, 30:44] = 2
    inst[1, 10:30, 10:25] = 5                                # id không liên tục, một con
    return inst


def test_lovasz_matches_reference() -> None:
    ref = _load_reference()
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(500, generator=g)
    labels = (torch.rand(500, generator=g) > 0.7).float()
    got = lovasz_hinge_flat(logits, labels).item()
    if ref is not None:
        want = ref[0](logits, labels).item()
        assert abs(got - want) < 1e-5, (got, want)
        print(f"OK  lovasz_hinge_flat khớp bản gốc ({got:.6f})")
    else:
        assert 0.0 <= got <= 2.0
        print("OK  lovasz_hinge_flat chạy (không có repo gốc để đối chiếu)")
    assert lovasz_hinge_flat(torch.tensor([5.0, 5.0, -5.0]), torch.tensor([1.0, 1.0, 0.0])).item() == 0.0   # dự đoán hoàn hảo
    assert lovasz_hinge_flat(torch.zeros(0), torch.zeros(0)).item() == 0.0


def test_matches_reference_loss() -> None:
    ref = _load_reference()
    if ref is None:
        print("BỎ QUA  đối chiếu loss với repo gốc")
        return
    inst = _scene()
    h, w = inst.shape[1:]
    for n_sigma in (1, 2):
        for to_center in (True, False):
            g = torch.Generator().manual_seed(n_sigma * 10 + int(to_center))
            pred = torch.cat([torch.randn(2, 2, h, w, generator=g) * 0.3,
                              0.3 + torch.randn(2, n_sigma, h, w, generator=g) * 0.1,
                              torch.randn(2, 1, h, w, generator=g)], 1)
            mine = SpatialEmbeddingLoss(to_center, n_sigma, 10.0)
            total, *_ = mine(pred, inst)
            theirs = ref[1](to_center=to_center, n_sigma=n_sigma, foreground_weight=10)
            theirs.xym = SpatialEmbeddingLoss.coordinate_map(h, w, pred.device)    # cùng hệ toạ độ chuẩn hoá cho ảnh nhỏ
            want = theirs(pred, inst, (inst > 0).long(), w_inst=1, w_var=10, w_seed=1).item()
            assert abs(total.item() - want) < 1e-4 * max(1.0, abs(want)), (n_sigma, to_center, total.item(), want)
    print("OK  SpatialEmbeddingLoss khớp loss gốc (n_sigma 1/2, tâm = toạ độ / embedding; có mảnh rời cùng id)")


def _perfect_pred(inst: torch.Tensor, s_raw: float = 0.35, break_fragment: bool = False) -> torch.Tensor:
    """Dự đoán 'hoàn hảo': pixel của con k trỏ đúng về tâm con (trung bình toạ độ cả các mảnh); nền không dịch; seed = φ."""
    b, h, w = inst.shape
    xym = SpatialEmbeddingLoss.coordinate_map(h, w, inst.device)
    pred = torch.zeros(b, 2 + 2 + 1, h, w)
    pred[:, 2:4] = s_raw
    for i in range(b):
        for k in torch.unique(inst[i][inst[i] > 0]):
            m = inst[i] == k
            center = xym[:, m].mean(1)
            if break_fragment:                                   # mảnh dưới (hàng >= h/2) trỏ lệch sang chỗ khác
                low = m.clone()
                low[: h // 2] = False
                shifted = center + torch.tensor([0.0, 0.4])
                off = torch.where(low[None], shifted[:, None, None] - xym, center[:, None, None] - xym)
            else:
                off = center[:, None, None] - xym
            pred[i, 0:2] = torch.where(m[None], torch.atanh(off.clamp(-0.99, 0.99)), pred[i, 0:2])
    pred[:, 4] = 6.0 * (inst > 0).float() - 6.0 * (inst == 0).float()     # seed cao trong con, thấp ở nền
    return pred


def test_fragments_share_center() -> None:
    inst = _scene()
    loss = SpatialEmbeddingLoss(True, 2, 10.0)
    good = loss(_perfect_pred(inst), inst)
    bad = loss(_perfect_pred(inst, break_fragment=True), inst)
    assert good[1].item() < bad[1].item(), (good[1].item(), bad[1].item())            # L_inst: mảnh cùng trỏ về một tâm tốt hơn
    assert good[0].item() < bad[0].item()
    rnd = loss(torch.randn(2, 5, *inst.shape[1:]), inst)
    assert good[0].item() < rnd[0].item()
    print(f"OK  mảnh rời cùng id trỏ về một tâm → loss thấp hơn lệch tâm và hơn ngẫu nhiên "
          f"(L_inst {good[1].item():.3f} < {bad[1].item():.3f}; tổng {good[0].item():.3f} < {rnd[0].item():.3f})")


def test_gradients_and_edge_cases() -> None:
    inst = _scene()
    pred = torch.randn(2, 5, *inst.shape[1:], requires_grad=True)
    total, *_ = SpatialEmbeddingLoss()(pred, inst)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    p0 = torch.randn(1, 5, 16, 16, requires_grad=True)                                   # ảnh không có con
    t0, *_ = SpatialEmbeddingLoss()(p0, torch.zeros(1, 16, 16, dtype=torch.long))
    t0.backward()
    assert torch.isfinite(t0) and p0.grad is not None
    valid = torch.ones(2, 1, *inst.shape[1:])                                            # vùng bỏ qua không có gradient
    valid[:, 0, :, :6] = 0
    p1 = torch.randn(2, 5, *inst.shape[1:], requires_grad=True)
    SpatialEmbeddingLoss()(p1, inst, valid)[0].backward()
    assert p1.grad[:, :, :, :6].abs().sum() == 0, "pixel trong vùng bỏ qua không được nhận gradient"
    print("OK  gradient hữu hạn, ảnh trống, vùng bỏ qua không có gradient")


def test_cluster_one_id_per_larva() -> None:
    """Dự đoán hoàn hảo → mỗi con đúng MỘT id, kể cả hai mảnh rời của con 1."""
    inst = _scene()
    pred = _perfect_pred(inst)
    i = 0
    out = cluster_spatial_embeddings(pred[i, 0:2].numpy(), pred[i, 2:4].numpy(), torch.sigmoid(pred[i, 4]).numpy(),
                                     (inst[i] > 0).numpy(), seed_thr=0.5, min_area=30)
    gt = inst[i].numpy()
    frag_a, frag_b, other = out[5, 10], out[30, 35], out[8, 36]
    assert frag_a > 0 and frag_a == frag_b, "hai mảnh rời của cùng một con phải cùng id"
    assert other > 0 and other != frag_a, "con khác phải có id khác"
    assert len(np.unique(out[gt > 0])) == 2, np.unique(out[gt > 0])
    assert (out[gt == 0] == 0).all()
    # ảnh không có vùng thân → không có con nào
    assert cluster_spatial_embeddings(pred[i, 0:2].numpy(), pred[i, 2:4].numpy(), torch.sigmoid(pred[i, 4]).numpy(),
                                      np.zeros_like(gt, bool)).max() == 0
    print("OK  clustering cho đúng 1 id mỗi con (hai mảnh rời cùng id), nền = 0, ảnh trống = 0 con")


if __name__ == "__main__":
    test_lovasz_matches_reference()
    test_matches_reference_loss()
    test_fragments_share_center()
    test_gradients_and_edge_cases()
    test_cluster_one_id_per_larva()
    print("\nTất cả kiểm tra đều đạt.")
