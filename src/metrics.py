"""
src/metrics.py — Shared Evaluation Metrics for Silkworm Segmentation experiments.

Functions:
    evaluate_seg  — mIoU & Dice Score on segmentation val set
    evaluate_cls  — Accuracy on classification val set

Usage:
    from src.metrics import evaluate_seg, evaluate_cls
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def masked_cls_correct(
    class_logits: torch.Tensor,
    class_gt: torch.Tensor,
    ignore_index: int = -1,
) -> tuple[int, int]:
    """
    Đếm (số dự đoán đúng, số mẫu có nhãn) — bỏ qua mẫu không có nhãn bệnh (-1).
    Cộng dồn qua các batch rồi chia ở cuối epoch để accuracy không bị lệch theo kích thước batch.
    """
    valid = class_gt != ignore_index
    correct = (class_logits.argmax(1)[valid] == class_gt[valid]).sum().item()
    return int(correct), int(valid.sum().item())


def evaluate_seg(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Evaluate segmentation quality (mIoU and Dice Score) on a val DataLoader.

    Parameters
    ----------
    model     : model with forward(x, phase) returning (mask_logits, _)
    loader    : DataLoader yielding (imgs, masks)
    device    : torch device
    threshold : binarization threshold for predicted mask

    Returns
    -------
    (mean_iou, mean_dice)
    """
    model.eval()
    ious, dices = [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            mask_logits, _ = model(imgs, phase=1)
            mask_pred = torch.sigmoid(mask_logits)

            pred_bin = (mask_pred >= threshold).float()

            for p, g in zip(pred_bin, masks):
                p_b = p.squeeze().bool()
                g_b = g.squeeze().bool()

                inter = (p_b & g_b).sum().item()
                union = (p_b | g_b).sum().item()

                iou = (inter + 1e-6) / (union + 1e-6)
                dice = (2.0 * inter + 1e-6) / (p_b.sum().item() + g_b.sum().item() + 1e-6)

                ious.append(iou)
                dices.append(dice)

    return float(np.mean(ious)), float(np.mean(dices))


def evaluate_cls(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate classification accuracy on a val DataLoader.

    Parameters
    ----------
    model  : model with forward(x, phase) returning (_, class_logits)
    loader : DataLoader yielding (imgs, labels)
    device : torch device

    Returns
    -------
    (accuracy, accuracy)  — second value reserved for future F1 metric
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, class_logits = model(imgs, phase=2)

            preds = torch.argmax(class_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    acc = correct / max(1, total)
    return float(acc), float(acc)  # Second slot reserved for F1


# =====================================================================
# Phân đoạn TỪNG CON + chẩn đoán TỪNG CON
# =====================================================================

def split_instances(
    body_prob: np.ndarray,
    boundary_prob: np.ndarray,
    disease_prob: np.ndarray,
    body_thr: float = 0.5,
    boundary_thr: float = 0.5,
    disease_thr: float = 0.5,
    min_area: int = 30,
) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    """
    Tách từng con từ 3 bản đồ xác suất [H, W] rồi chẩn đoán từng con.

    1. Hạt giống = thân − biên → mỗi thành phần liền kề là lõi của một con
       (biên cắt rời 2 con đang chạm/đè nhau).
    2. Watershed lan hạt giống ra lại toàn bộ vùng thân để lấy lại pixel biên.
    3. Mỗi con: xác suất bệnh = trung bình disease_prob trên pixel của con đó.

    Trả về (inst_map, {id: (class_id, p_grasserie)}), class_id: 0 = Grasserie, 1 = Healthy.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    body = body_prob >= body_thr
    seeds, n = ndi.label(body & (boundary_prob < boundary_thr))
    if n:
        sizes = ndi.sum(np.ones_like(seeds), seeds, index=np.arange(1, n + 1))
        for i, sz in enumerate(sizes, start=1):
            if sz < min_area:
                seeds[seeds == i] = 0
    inst = watershed(-body_prob, markers=seeds, mask=body)

    info: dict[int, tuple[int, float]] = {}
    for i in np.unique(inst):
        if i == 0:
            continue
        p = float(disease_prob[inst == i].mean())
        info[int(i)] = (0 if p >= disease_thr else 1, p)
    return inst, info


def gt_instance_classes(inst_gt: np.ndarray, disease_gt: np.ndarray) -> dict[int, int]:
    """{id: class_id} cho GT (0 = Grasserie, 1 = Healthy), lấy theo đa số pixel của mỗi con."""
    out = {}
    for i in np.unique(inst_gt):
        if i == 0:
            continue
        out[int(i)] = 0 if disease_gt[inst_gt == i].mean() >= 0.5 else 1
    return out


def match_instances(
    pred_inst: np.ndarray,
    pred_cls: dict[int, int],
    gt_inst: np.ndarray,
    gt_cls: dict[int, int],
    iou_thr: float = 0.5,
    min_area: int = 100,
) -> dict[str, int]:
    """
    Ghép con dự đoán với con thật theo IoU (tham lam, IoU cao trước, mỗi con ghép tối đa 1 lần).
    Đối tượng nhỏ hơn `min_area` pixel (mảnh nhiễu, mẩu bị lá che gần hết) bị bỏ qua ở CẢ hai phía.

    Trả về số đếm:
      tp / fp / fn          — tách con đúng (IoU ≥ iou_thr), bất kể bệnh
      cls_correct           — trong các cặp ghép được, số con chẩn đoán ĐÚNG
      e2e_tp                — tách đúng VÀ chẩn đoán đúng (thước đo cho dây chuyền)
      gt_grasserie, hit_grasserie — số con bệnh thật, và số con bệnh được tách + báo bệnh đúng
    """
    gt_cls = {g: c for g, c in gt_cls.items() if (gt_inst == g).sum() >= min_area}
    pred_cls = {q: c for q, c in pred_cls.items() if (pred_inst == q).sum() >= min_area}

    pairs = []
    for g in gt_cls:
        gm = gt_inst == g
        cand, counts = np.unique(pred_inst[gm], return_counts=True)
        for p, inter in zip(cand, counts):
            if p == 0 or int(p) not in pred_cls:
                continue
            union = gm.sum() + (pred_inst == p).sum() - inter
            iou = inter / union
            if iou >= iou_thr:
                pairs.append((iou, g, int(p)))
    pairs.sort(reverse=True)

    used_g, used_p = set(), set()
    tp = cls_correct = hit_g = 0
    for _, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g); used_p.add(p)
        tp += 1
        if pred_cls.get(p) == gt_cls[g]:
            cls_correct += 1
            if gt_cls[g] == 0:
                hit_g += 1
    return {
        "tp": tp,
        "fp": len(pred_cls) - tp,
        "fn": len(gt_cls) - tp,
        "cls_correct": cls_correct,
        "e2e_tp": cls_correct,
        "gt_grasserie": sum(1 for c in gt_cls.values() if c == 0),
        "hit_grasserie": hit_g,
    }


def summarize_instance_counts(c: dict[str, int]) -> dict[str, float]:
    """Từ số đếm cộng dồn → các chỉ số theo từng con."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    e2e = c["e2e_tp"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "inst_precision": prec,
        "inst_recall": rec,
        "inst_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "cls_acc_matched": c["cls_correct"] / max(tp, 1),
        "e2e_f1": 2 * e2e / max(2 * e2e + (tp + fp - e2e) + (tp + fn - e2e), 1),
        "grasserie_recall": c["hit_grasserie"] / max(c["gt_grasserie"], 1),
    }
