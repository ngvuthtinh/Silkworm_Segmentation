"""
evaluator.py — Chấm SOLOv2 theo TỪNG CON, đúng cách đang chấm VM-UNet (để so được với B1, E1–E3):
nhãn và dự đoán đưa về 256² (nearest), con < min_area = 100 px bị bỏ ở cả hai phía, ghép theo IoU ≥ 0.5, dự đoán nằm >= 50% trong
vùng bỏ qua không bị tính sai, con chưa biết bệnh không bao giờ tính là chẩn đoán sai (src.metrics.match_instances).

Dự đoán của SOLOv2 là các mask có điểm; giữ mask có điểm >= score_thr, vẽ theo điểm tăng dần (pixel thuộc mask điểm cao nhất).
Có thể chấm nhiều ngưỡng trong một lượt (dùng để chọn ngưỡng trên valid).
"""

from __future__ import annotations

import collections
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.evaluation import DatasetEvaluator

from src.metrics import gt_instance_classes, match_instances, summarize_instance_counts

SIZE, MIN_AREA, IOU_THR = 256, 100, 0.5


def gt_at_256(root: str, split: str, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inst [S,S], disease [S,S] với -1 = chưa biết bệnh, valid [S,S] bool) như InstanceSilkwormDataset ở 256²."""
    d = os.path.join(root, split)
    inst = cv2.imread(os.path.join(d, "masks_inst", f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    cls = cv2.imread(os.path.join(d, "masks_cls", f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    ign_p = os.path.join(d, "masks_ignore", f"{stem}.png")
    ign = cv2.imread(ign_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(ign_p) else np.zeros(cls.shape, np.uint8)
    inst = cv2.resize(inst, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    cls = cv2.resize(cls, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    ign = cv2.resize(ign, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    disease = (cls == 1).astype(np.float32)
    disease[cls == 3] = -1.0
    valid = (ign <= 127) | (inst > 0)
    return inst, disease, valid


def instances_to_map(instances, score_thr: float) -> tuple[np.ndarray, dict[int, int]]:
    """Instances của SOLOv2 (mask ở cỡ ảnh gốc) → (bản đồ id [S,S], {id: lớp 0 = Grasserie / 1 = Healthy})."""
    out = np.zeros((SIZE, SIZE), np.int64)
    if len(instances) == 0 or instances.pred_masks.numel() == 0:
        return out, {}
    scores = instances.scores
    keep = (scores >= score_thr).nonzero().flatten()
    if len(keep) == 0:
        return out, {}
    keep = keep[torch.argsort(scores[keep])]                               # tăng dần: mask điểm cao vẽ sau, đè lên
    masks = F.interpolate(instances.pred_masks[keep].float()[None], size=(SIZE, SIZE), mode="nearest")[0] > 0.5
    classes = instances.pred_classes[keep].tolist()
    masks = masks.cpu().numpy()
    cls_of = {}
    for k, (m, c) in enumerate(zip(masks, classes), 1):
        out[m] = k
        cls_of[k] = int(c)
    return out, {k: c for k, c in cls_of.items() if (out == k).any()}


class SilkEvaluator(DatasetEvaluator):
    def __init__(self, score_thrs: list[float] | float = 0.3, log_csv: str | None = None):
        self.thrs = [score_thrs] if isinstance(score_thrs, float) else list(score_thrs)
        self.log_csv = log_csv
        self._iter = 0

    def reset(self) -> None:
        self.counts = {t: collections.Counter() for t in self.thrs}
        self.n_regions = {t: 0 for t in self.thrs}
        self.inter = {t: 0.0 for t in self.thrs}
        self.denom = {t: 0.0 for t in self.thrs}

    def process(self, inputs, outputs) -> None:
        for inp, out in zip(inputs, outputs):
            gi, gd, vm = gt_at_256(inp["silk_root"], inp["silk_split"], inp["image_id"])
            gcls = gt_instance_classes(gi, gd)
            inst = out["instances"].to("cpu")
            for t in self.thrs:
                pi, pcls = instances_to_map(inst, t)
                self.n_regions[t] += len(pcls)
                self.counts[t].update(match_instances(pi, pcls, gi, gcls, IOU_THR, MIN_AREA, ~vm))
                pb, gb = pi > 0, gi > 0
                self.inter[t] += (pb & gb & vm).sum()
                self.denom[t] += (pb & vm).sum() + (gb & vm).sum()

    def evaluate(self) -> dict:
        res = {}
        for t in self.thrs:
            r = summarize_instance_counts(self.counts[t])
            r["body_dice"] = float(2 * self.inter[t] / max(self.denom[t], 1))
            r["n_gt_worms"] = int(self.counts[t]["tp"] + self.counts[t]["fn"])
            r["n_regions"] = int(self.n_regions[t])
            res[t] = r
        main = dict(res[self.thrs[0]])
        if len(self.thrs) > 1:
            main["by_score_thr"] = {str(t): res[t] for t in self.thrs}
        if self.log_csv:
            os.makedirs(os.path.dirname(self.log_csv), exist_ok=True)
            row = {"eval": self._iter, "score_thr": self.thrs[0], **{k: v for k, v in res[self.thrs[0]].items()}}
            new = not os.path.exists(self.log_csv)
            with open(self.log_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            self._iter += 1
        return {"silk": main}
