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
