"""
losses.py — Segmentation-First Multi-Task Loss Functions for Swin-Unet.

Implements:
    - DiceLoss (Soft Dice Loss)
    - SegLoss (BCE + Dice)
    - ClsLoss (CrossEntropyLoss)
    - SegFirstMultiTaskLoss (handles Dataset A, Dataset B, or Joint loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ClsLoss(nn.Module):
    """CrossEntropyLoss for binary classification."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, class_logits: torch.Tensor, class_gt: torch.Tensor) -> torch.Tensor:
        return self.ce(class_logits, class_gt)


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
        """
        device = mask_pred.device if mask_pred is not None else class_logits.device
        zero = torch.tensor(0.0, device=device)

        loss_seg = zero
        loss_cls = zero

        if mask_pred is not None and mask_gt is not None:
            loss_seg = self.seg_loss_fn(mask_pred, mask_gt)

        if class_logits is not None and class_gt is not None:
            loss_cls = self.cls_loss_fn(class_logits, class_gt)

        total_loss = self.lambda_seg * loss_seg + self.lambda_cls * loss_cls

        return total_loss, loss_seg, loss_cls
