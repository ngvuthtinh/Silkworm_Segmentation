"""
src/losses.py — Shared Loss Functions for Silkworm Segmentation experiments.

Implements:
    - DiceLoss (Soft Dice Loss)
    - SegLoss (BCE + Dice)
    - ClsLoss (CrossEntropyLoss)
    - SegFirstMultiTaskLoss (handles Dataset A, Dataset B, or Joint loss)

Usage:
    from src.losses import DiceLoss, SegLoss, ClsLoss, SegFirstMultiTaskLoss

Override per-experiment:
    Create losses.py inside your experiment folder to override these defaults.
    The experiment's train.py will prefer the local version via try/except import.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset_multitask import IGNORE_LABEL


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : [B, 1, H, W] (probabilities or logits after sigmoid)
        target : [B, 1, H, W] (binary ground truth)
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class SegLoss(nn.Module):
    """Combined BCEWithLogitsLoss + Dice Loss for segmentation."""

    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, mask_logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(mask_logits, mask_gt)
        mask_pred = torch.sigmoid(mask_logits)
        loss_dice = self.dice(mask_pred, mask_gt)
        return self.w_bce * loss_bce + self.w_dice * loss_dice


def masked_cross_entropy(
    class_logits: torch.Tensor,
    class_gt: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = IGNORE_LABEL,
) -> torch.Tensor:
    """
    CrossEntropy chỉ tính trên các mẫu CÓ nhãn bệnh (class_gt != ignore_index).

    Ảnh Silkynet không có nhãn bệnh (-1) sẽ không đóng góp gradient cho head phân loại,
    nhưng vẫn huấn luyện head phân đoạn bình thường. Nếu cả batch không có nhãn, trả về 0
    (có gradient graph) thay vì NaN — vì CE mean-reduction trên tập rỗng cho ra NaN.
    """
    valid = class_gt != ignore_index
    if not valid.any():
        return class_logits.sum() * 0.0
    return F.cross_entropy(
        class_logits[valid], class_gt[valid], label_smoothing=label_smoothing
    )


class ClsLoss(nn.Module):
    """CrossEntropyLoss for binary classification (bỏ qua mẫu không có nhãn = -1)."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, class_logits: torch.Tensor, class_gt: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(class_logits, class_gt, self.label_smoothing)


class SegFirstMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with Segmentation-First priority.

    Weighting:
        L = lambda_seg * L_seg + lambda_cls * L_cls

    Default lambda_seg = 1.0, lambda_cls = 0.1 (low impact to protect backbone).
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 0.1,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.seg_loss_fn = SegLoss(w_bce=w_bce, w_dice=w_dice)
        self.cls_loss_fn = ClsLoss()

    def forward(
        self,
        mask_pred: torch.Tensor | None = None,
        mask_gt: torch.Tensor | None = None,
        class_logits: torch.Tensor | None = None,
        class_gt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes total loss depending on available targets.

        Returns
        -------
        (total_loss, seg_loss_val, cls_loss_val)
        """
        device = (
            mask_pred.device
            if mask_pred is not None
            else (class_logits.device if class_logits is not None else torch.device("cpu"))
        )

        l_seg = torch.tensor(0.0, device=device)
        l_cls = torch.tensor(0.0, device=device)

        if mask_pred is not None and mask_gt is not None:
            l_seg = self.seg_loss_fn(mask_pred, mask_gt)

        if class_logits is not None and class_gt is not None:
            l_cls = self.cls_loss_fn(class_logits, class_gt)

        total_loss = self.lambda_seg * l_seg + self.lambda_cls * l_cls
        return total_loss, l_seg, l_cls


def _masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """BCE + soft Dice chỉ trên pixel có weight = 1 (pixel weight = 0 không đóng góp gradient)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / weight.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits) * weight
    tgt = target * weight
    dice = 1.0 - (2.0 * (prob * tgt).sum() + eps) / (prob.sum() + tgt.sum() + eps)
    return bce + dice


class InstanceMultiTaskLoss(nn.Module):
    """
    Loss cho mô hình phân đoạn TỪNG CON + chẩn đoán TỪNG CON (3 kênh đầu ra theo pixel).

        L = λ_body·(BCE+Dice)(thân) + λ_bnd·(BCE+Dice)(biên) + λ_dis·BCE(bệnh | trong thân)

    - Biên chiếm rất ít pixel nên cần Dice để không bị "học" thành toàn 0.
    - Loss bệnh chỉ tính trên pixel thuộc thân tằm (GT): nền không có khái niệm khỏe/bệnh.
    - `valid` (tuỳ chọn): 0 = vùng bỏ qua (con tằm không có nhãn) → không tính vào bất kỳ loss nào.
    """

    def __init__(self, lambda_body: float = 1.0, lambda_boundary: float = 1.0, lambda_disease: float = 1.0):
        super().__init__()
        self.lambda_body = lambda_body
        self.lambda_boundary = lambda_boundary
        self.lambda_disease = lambda_disease

    def forward(
        self,
        logits: torch.Tensor,
        body_gt: torch.Tensor,
        boundary_gt: torch.Tensor,
        disease_gt: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits: [B, 3, H, W] — kênh 0 = thân, 1 = biên, 2 = bệnh (logit thô)
        Trả về (total, l_body, l_boundary, l_disease).
        """
        logits = logits.float()
        w = torch.ones_like(body_gt) if valid is None else valid.float()
        l_body = _masked_bce_dice(logits[:, 0:1], body_gt, w)
        l_bnd = _masked_bce_dice(logits[:, 1:2], boundary_gt, w)

        inside = (body_gt > 0.5) & (w > 0.5)
        if inside.any():
            l_dis = F.binary_cross_entropy_with_logits(logits[:, 2:3][inside], disease_gt[inside])
        else:
            l_dis = logits[:, 2:3].sum() * 0.0

        total = self.lambda_body * l_body + self.lambda_boundary * l_bnd + self.lambda_disease * l_dis
        return total, l_body, l_bnd, l_dis
