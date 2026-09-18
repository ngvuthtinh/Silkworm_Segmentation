"""
losses.py — Loss functions for multi-task silkworm model.

    SegLoss     : BCE + Dice for binary segmentation (reuses BceDiceLoss logic)
    ClsLoss     : CrossEntropyLoss for healthy/diseased classification
    MultiTaskLoss : Combines both with configurable λ weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segmentation Loss  (BCE + Dice, mirrors BceDiceLoss from vmunet/utils.py)
# ---------------------------------------------------------------------------

class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (pred must be sigmoid probabilities)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        B = pred.size(0)
        pred_ = pred.view(B, -1)
        tgt_  = target.view(B, -1)
        inter = (pred_ * tgt_).sum(1)
        dice  = (2 * inter + smooth) / (pred_.sum(1) + tgt_.sum(1) + smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """
    Combined BCE + Dice loss for binary segmentation.

    pred   : [B, 1, H, W]  sigmoid probabilities (output of model seg head)
    target : [B, 1, H, W]  binary ground-truth mask in [0, 1]
    """

    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce  = w_bce
        self.w_dice = w_dice
        self.bce    = nn.BCELoss()
        self.dice   = BinaryDiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device_type = pred.device.type
        with torch.amp.autocast(device_type, enabled=False):
            pred_f32   = pred.float()
            target_f32 = target.float()
            pred_   = torch.clamp(pred_f32.view(pred_f32.size(0), -1), 1e-7, 1.0 - 1e-7)
            target_ = torch.clamp(target_f32.view(target_f32.size(0), -1), 0.0, 1.0)
            loss_bce  = self.bce(pred_, target_)
            loss_dice = self.dice(pred_f32, target_f32)
            return self.w_bce * loss_bce + self.w_dice * loss_dice


# ---------------------------------------------------------------------------
# Classification Loss
# ---------------------------------------------------------------------------

class ClsLoss(nn.Module):
    """
    CrossEntropyLoss for 2-class (healthy / diseased) classification.

    logits : [B, 2]   raw logits from classification head
    labels : [B]      integer class indices  (0=healthy, 1=diseased)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels.long())


# ---------------------------------------------------------------------------
# Combined Multi-Task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Conditional multi-task loss.

    For a batch from Dataset A (segmentation):
        loss = lambda_seg * SegLoss(mask_pred, mask_gt)

    For a batch from Dataset B (classification):
        loss = lambda_cls * ClsLoss(cls_logits, cls_label)

    For a batch with BOTH (future):
        loss = lambda_seg * SegLoss(...) + lambda_cls * ClsLoss(...)

    Parameters
    ----------
    lambda_seg : float   weight for segmentation loss
    lambda_cls : float   weight for classification loss
    w_bce      : float   BCE weight inside SegLoss
    w_dice     : float   Dice weight inside SegLoss
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 1.0,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.seg_loss   = SegLoss(w_bce=w_bce, w_dice=w_dice)
        self.cls_loss   = ClsLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        mask_pred: torch.Tensor,
        class_logits: torch.Tensor,
        mask_gt: torch.Tensor | None = None,
        class_gt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns
        -------
        total_loss : scalar tensor
        loss_dict  : dict with individual loss values (for logging)
        """
        has_seg = mask_gt is not None
        has_cls = class_gt is not None

        loss_seg_val = torch.tensor(0.0, device=mask_pred.device)
        loss_cls_val = torch.tensor(0.0, device=class_logits.device)

        if has_seg:
            loss_seg_val = self.seg_loss(mask_pred, mask_gt)

        if has_cls:
            loss_cls_val = self.cls_loss(class_logits, class_gt)

        total = self.lambda_seg * loss_seg_val + self.lambda_cls * loss_cls_val

        return total, {
            "loss_seg": loss_seg_val.item(),
            "loss_cls": loss_cls_val.item(),
            "loss_total": total.item(),
        }
