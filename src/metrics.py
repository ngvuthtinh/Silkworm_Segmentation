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


def diagnose_instances(
    inst: np.ndarray, disease_prob: np.ndarray, disease_thr: float = 0.5
) -> dict[int, tuple[int, float]]:
    """{id: (class_id, p_grasserie)} từ bản đồ id; xác suất bệnh = trung bình disease_prob trên pixel của con (0 = Grasserie, 1 = Healthy)."""
    info: dict[int, tuple[int, float]] = {}
    for i in np.unique(inst):
        if i == 0:
            continue
        p = float(disease_prob[inst == i].mean())
        info[int(i)] = (0 if p >= disease_thr else 1, p)
    return info


def merge_fragments(inst: np.ndarray, embedding: np.ndarray, merge_thr: float = 1.5) -> np.ndarray:
    """
    Gộp các mảnh RỜI NHAU của cùng một con thành một id, dựa trên embedding (DiscriminativeLoss).

    `split_instances` trả về một vùng cho mỗi thành phần liền kề, nên con bị lá che thành 2 mảnh ra 2 id.
    Ở đây tính embedding trung bình của từng vùng rồi gộp hai vùng nếu khoảng cách < `merge_thr` (liên kết đơn:
    A~B và B~C thì cả ba thành một). Lý do chọn ngưỡng: loss ép pixel cách tâm cụm ≤ δ_v nên hai mảnh cùng con có
    tâm cách nhau ≲ 2δ_v, còn hai con khác nhau có tâm cách ≳ 2δ_d − 2δ_v; với δ_v = 0.5, δ_d = 1.5 thì ngưỡng
    hợp lệ nằm trong khoảng (1.0, 2.0), mặc định 1.5 là điểm giữa. Nên chỉnh trên tập valid.

    inst: [H, W] id từng vùng (0 = nền); embedding: [D, H, W]. Trả về bản đồ id mới (id không nhất thiết liên tục).
    """
    ids = [int(i) for i in np.unique(inst) if i != 0]
    if len(ids) < 2:
        return inst
    means = np.stack([embedding[:, inst == i].mean(axis=1) for i in ids])        # [n, D]
    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if np.linalg.norm(means[a] - means[b]) < merge_thr:
                parent[find(b)] = find(a)
    out = inst.copy()
    for k, i in enumerate(ids):
        out[inst == i] = ids[find(k)]
    return out


def cluster_spatial_embeddings(
    offset: np.ndarray,
    sigma: np.ndarray,
    seed: np.ndarray,
    fg_mask: np.ndarray,
    seed_thr: float = 0.5,
    min_area: int = 30,
    coord_scale: int | None = None,
) -> np.ndarray:
    """
    Tách từng con từ đầu ra "spatial embeddings" (Neven et al., 2019; mục 3.6), KHÔNG dùng watershed và không cần các mảnh liền kề.

    offset [2, H, W] và sigma [n, H, W] là giá trị thô của mô hình, seed [H, W] là xác suất (sau sigmoid), fg_mask [H, W] bool
    là vùng thân tằm (từ kênh thân). Lặp: chọn pixel chưa gán có seed cao nhất → embedding của nó là tâm, σ lấy tại chính điểm
    đó → mọi pixel có φ = exp(−Σ (e − tâm)²·exp(10σ)) > 0.5 thuộc con này. Hai mảnh rời của cùng một con có cùng tâm nên
    cùng một id; một id = một con.

    Khác repo gốc: vùng nền lấy từ kênh thân (không phải seed > 0.5); chỉ gán các pixel còn chưa gán (không ghi đè con trước);
    ngưỡng diện tích tính theo `min_area` (ảnh 256²) thay vì 128 px của ảnh 2048×1024. Trả về bản đồ id [H, W] int32 (0 = nền).
    """
    height, width = seed.shape
    if coord_scale is None:                                                             # mặc định: chuẩn hoá [0, 1] theo cả ảnh (đúng khi ảnh = cỡ lúc train)
        xm, ym = np.meshgrid(np.linspace(0, 1, width, dtype=np.float32), np.linspace(0, 1, height, dtype=np.float32))
    else:                                                                               # cỡ pixel như lúc train: 1/(coord_scale-1) mỗi pixel, bất kể cỡ ảnh lúc suy luận
        xm, ym = np.meshgrid(np.arange(width, dtype=np.float32) / (coord_scale - 1), np.arange(height, dtype=np.float32) / (coord_scale - 1))
    emb = np.tanh(offset.astype(np.float32)) + np.stack([xm, ym])                       # [2, H, W]
    flat = np.flatnonzero(fg_mask.ravel())
    out = np.zeros(height * width, np.int32)
    if flat.size < min_area:
        return out.reshape(height, width)
    e = emb.reshape(2, -1)[:, flat]                                                     # [2, M]
    s_all = sigma.reshape(sigma.shape[0], -1)[:, flat].astype(np.float32)               # [n, M]
    sd = seed.ravel()[flat]
    free = np.ones(flat.size, bool)
    labels = np.zeros(flat.size, np.int32)
    count = 1
    while free.sum() >= min_area:
        k = int(np.argmax(np.where(free, sd, -1.0)))
        if sd[k] < seed_thr:
            break
        phi = np.exp(-(((e - e[:, k:k + 1]) ** 2) * np.exp(s_all[:, k:k + 1] * 10.0)).sum(0))
        prop = phi > 0.5                                    # luôn gồm chính điểm hạt giống (φ = 1)
        take = prop & free
        if prop.sum() >= min_area and take.sum() / prop.sum() > 0.5:
            labels[take] = count
            count += 1
        free &= ~prop
    out[flat] = labels
    return out.reshape(height, width)


def gt_instance_classes(inst_gt: np.ndarray, disease_gt: np.ndarray) -> dict[int, int]:
    """{id: class_id} cho GT (0 = Grasserie, 1 = Healthy, -1 = chưa biết bệnh), lấy theo đa số pixel của mỗi con."""
    out = {}
    for i in np.unique(inst_gt):
        if i == 0:
            continue
        m = disease_gt[inst_gt == i].mean()
        out[int(i)] = -1 if m < 0 else 0 if m >= 0.5 else 1
    return out


def match_instances(
    pred_inst: np.ndarray,
    pred_cls: dict[int, int],
    gt_inst: np.ndarray,
    gt_cls: dict[int, int],
    iou_thr: float = 0.5,
    min_area: int = 100,
    ignore: np.ndarray | None = None,
) -> dict[str, int]:
    """
    Ghép con dự đoán với con thật theo IoU (tham lam, IoU cao trước, mỗi con ghép tối đa 1 lần).
    Đối tượng nhỏ hơn `min_area` pixel (mảnh nhiễu, mẩu bị lá che gần hết) bị bỏ qua ở CẢ hai phía.
    `ignore` (bool [H, W], tuỳ chọn): con dự đoán có ≥ 50% pixel nằm trong vùng bỏ qua (con tằm thật nhưng
    không có nhãn) không bị tính là dự đoán sai.

    Trả về số đếm:
      tp / fp / fn          — tách con đúng (IoU ≥ iou_thr), bất kể bệnh
      cls_correct           — trong các cặp ghép được, số con chẩn đoán ĐÚNG
      e2e_tp                — tách đúng VÀ chẩn đoán đúng (thước đo cho dây chuyền)
      gt_grasserie, hit_grasserie — số con bệnh thật, và số con bệnh được tách + báo bệnh đúng
    """
    gt_cls = {g: c for g, c in gt_cls.items() if (gt_inst == g).sum() >= min_area}
    pred_cls = {q: c for q, c in pred_cls.items() if (pred_inst == q).sum() >= min_area}
    if ignore is not None and ignore.any():
        pred_cls = {q: c for q, c in pred_cls.items() if ignore[pred_inst == q].mean() < 0.5}

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
        if gt_cls[g] < 0 or pred_cls.get(p) == gt_cls[g]:      # con chưa biết bệnh: không bao giờ tính là chẩn đoán sai (e2e_f1 = inst_f1)
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
