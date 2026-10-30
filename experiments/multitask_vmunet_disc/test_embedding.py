#!/usr/bin/env python3
"""
experiments/multitask_vmunet_disc/test_embedding.py — kiểm tra DiscriminativeLoss và merge_fragments.

Chạy (từ thư mục gốc dự án, không cần GPU):
  python -m experiments.multitask_vmunet_disc.test_embedding
"""

from __future__ import annotations

import numpy as np
import torch

from src.losses import DiscriminativeLoss
from src.metrics import diagnose_instances, merge_fragments


def _reference(emb: torch.Tensor, inst: torch.Tensor, valid, dv: float, dd: float):
    """Bản tính tay, từng cụm một, đúng công thức (1)-(3) của paper (không vector hoá) để đối chiếu."""
    out = []
    for b in range(emb.shape[0]):
        keep = inst[b] > 0
        if valid is not None:
            keep = keep & (valid[b, 0] > 0.5)
        ids = [int(i) for i in torch.unique(inst[b][keep])]
        if not ids:
            continue
        mus, var_c = [], []
        for c in ids:
            m = keep & (inst[b] == c)
            x = emb[b][:, m]                                   # [D, N_c]
            mu = x.mean(1)
            mus.append(mu)
            var_c.append(((x - mu[:, None]).norm(dim=0) - dv).clamp_min(0).pow(2).mean())
        l_var = torch.stack(var_c).mean()
        n = len(ids)
        dist = [(2 * dd - (mus[a] - mus[c]).norm()).clamp_min(0).pow(2) for a in range(n) for c in range(n) if a != c]
        l_dist = torch.stack(dist).sum() / (n * (n - 1)) if n > 1 else torch.zeros(())
        l_reg = torch.stack([m.norm() for m in mus]).mean()
        out.append((l_var, l_dist, l_reg))
    return [torch.stack(v).mean().item() for v in zip(*out)]


def test_matches_reference() -> None:
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(3, 8, 24, 24, generator=g) * 0.6
    inst = torch.zeros(3, 24, 24, dtype=torch.long)
    inst[0, 2:10, 2:12], inst[0, 14:22, 12:22], inst[0, 3:8, 15:20] = 1, 2, 3
    inst[1, 5:15, 5:15] = 7                                    # id không liên tục, một con
    # ảnh 2 có hai con, ảnh 3 không có con nào (kiểm tra bỏ qua ảnh trống)
    inst[2] = 0
    inst[1, 18:23, 1:9] = 9
    valid = torch.ones(3, 1, 24, 24)
    valid[0, 0, :, :3] = 0
    for v in (None, valid):
        total, l_var, l_dist, l_reg = DiscriminativeLoss(0.5, 1.5)(emb, inst, v)
        ref = _reference(emb, inst, v, 0.5, 1.5)
        for got, want in zip((l_var.item(), l_dist.item(), l_reg.item()), ref):
            assert abs(got - want) < 1e-4 * max(1.0, abs(want)), (got, want)
        assert abs(total.item() - (ref[0] + ref[1] + 0.001 * ref[2])) < 1e-4
    print("OK  khớp bản tính tay theo công thức paper (có/không vùng bỏ qua, id không liên tục, ảnh trống)")


def test_perfect_embedding_and_connectivity() -> None:
    """Hai mảnh RỜI NHAU của cùng một con, cùng embedding → L_var = 0 và không bị phạt dù không liền kề."""
    inst = torch.zeros(1, 20, 20, dtype=torch.long)
    inst[0, 2:8, 2:8] = 1                                      # mảnh A của con 1
    inst[0, 12:18, 12:18] = 1                                  # mảnh B của con 1 (rời mảnh A, như bị lá che)
    inst[0, 2:8, 12:18] = 2                                    # con 2
    emb = torch.zeros(1, 4, 20, 20)
    emb[0, 0][inst[0] == 1] = 0.0
    emb[0, 0][inst[0] == 2] = 4.0                              # hai tâm cách 4 > 2·δ_d = 3
    total, l_var, l_dist, l_reg = DiscriminativeLoss(0.5, 1.5)(emb, inst)
    assert l_var.item() < 1e-6 and l_dist.item() < 1e-6, (l_var.item(), l_dist.item())
    # cùng bố cục nhưng mảnh B bị lệch embedding → phải bị phạt
    emb2 = emb.clone()
    emb2[0, 1][(inst[0] == 1) & (torch.arange(20)[:, None] >= 10)] = 3.0
    assert DiscriminativeLoss(0.5, 1.5)(emb2, inst)[1].item() > 0.1
    # hai con khác nhau nhưng embedding trùng → L_dist phải lớn
    emb3 = torch.zeros(1, 4, 20, 20)
    assert DiscriminativeLoss(0.5, 1.5)(emb3, inst)[2].item() > 1.0
    print("OK  hai mảnh rời cùng một con, cùng embedding: không bị phạt; lệch embedding hoặc trùng con khác: bị phạt")


def test_gradients_and_edge_cases() -> None:
    inst = torch.zeros(2, 16, 16, dtype=torch.long)
    inst[0, 2:8, 2:8], inst[0, 9:14, 9:14] = 1, 2
    emb = torch.zeros(2, 8, 16, 16, requires_grad=True)        # đúng 0 → kiểm tra không ra NaN ở điểm trùng tâm
    total, *_ = DiscriminativeLoss()(emb, inst)
    total.backward()
    assert torch.isfinite(emb.grad).all() and torch.isfinite(total)
    # toàn bộ ảnh không có con: trả 0 nhưng vẫn gắn đồ thị gradient
    emb0 = torch.randn(1, 8, 8, 8, requires_grad=True)
    total0, *_ = DiscriminativeLoss()(emb0, torch.zeros(1, 8, 8, dtype=torch.long))
    total0.backward()
    assert total0.item() == 0.0 and emb0.grad is not None
    # pixel có valid = 0 không được nhận gradient
    inst1 = torch.zeros(1, 8, 8, dtype=torch.long)
    inst1[0, :, :4], inst1[0, :, 4:] = 1, 2
    valid = torch.ones(1, 1, 8, 8)
    valid[0, 0, :, 0] = 0
    e = torch.randn(1, 4, 8, 8, requires_grad=True)
    DiscriminativeLoss()(e, inst1, valid)[0].backward()
    assert e.grad[0, :, :, 0].abs().sum() == 0 and e.grad[0, :, :, 1:].abs().sum() > 0
    try:
        DiscriminativeLoss(delta_var=0.8, delta_dist=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("phải từ chối delta_dist <= 2*delta_var")
    print("OK  gradient hữu hạn (kể cả khi trùng tâm), ảnh trống, vùng bỏ qua không có gradient, kiểm tra δ_d > 2δ_v")


def test_merge_fragments() -> None:
    inst = np.zeros((20, 20), np.int32)
    inst[2:6, 2:6] = 1            # con A, mảnh 1
    inst[12:16, 12:16] = 2        # con A, mảnh 2 (rời mảnh 1)
    inst[2:6, 12:16] = 3          # con B
    emb = np.zeros((4, 20, 20), np.float32)
    emb[0][inst == 3] = 4.0
    out = merge_fragments(inst, emb, merge_thr=1.5)
    assert out[3, 3] == out[13, 13] != 0 and out[3, 3] != out[3, 13]
    assert len(np.unique(out)) == 3                           # nền + 2 con
    assert np.array_equal(merge_fragments(inst, emb, merge_thr=0.0), inst)  # ngưỡng 0: không gộp gì
    prob = np.zeros((20, 20), np.float32)
    prob[out == out[3, 3]] = 0.9
    info = diagnose_instances(out, prob, 0.5)
    assert info[int(out[3, 3])][0] == 0 and info[int(out[3, 13])][0] == 1
    print("OK  merge_fragments gộp đúng hai mảnh rời cùng embedding, giữ riêng con khác; diagnose_instances đúng lớp")


if __name__ == "__main__":
    test_matches_reference()
    test_perfect_embedding_and_connectivity()
    test_gradients_and_edge_cases()
    test_merge_fragments()
    print("\nTất cả kiểm tra đều đạt.")
