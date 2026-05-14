"""
train.py — Training Script for Silkworm Anomaly Segmentation
=============================================================
Pipeline: dataset.py → model.py → train.py

Key design decisions
--------------------
Loss    : BCEWithLogitsLoss + Dice Loss  (handles class imbalance)
Optim   : AdamW  (decoupled weight decay — better than Adam for segmentation)
Sched   : ReduceLROnPlateau  (reduces LR when val loss plateaus)
Metrics : IoU (Jaccard) + Dice Score  (standard for binary segmentation)
Ckpt    : Saves models/best_model.pt only when val IoU improves

WHY BCE + DICE?
---------------
In silkworm anomaly images, diseased pixels can be <5% of the image.
BCE alone will optimise pixel accuracy — the model learns to predict
"all healthy" and still achieves 95%+ accuracy while completely missing
every lesion (high False Negative = catastrophic for disease control).

Dice Loss directly optimises the overlap ratio between predicted and
ground-truth masks:
    Dice = 2 * |P ∩ G| / (|P| + |G|)

When the mask is sparse, Dice focuses gradient energy on the rare
positive pixels.  BCE provides stable gradient signals across the full
image.  The combination gives both global calibration (BCE) and
lesion-focus (Dice).

Usage
-----
python train.py \\
    --train_dir data/train \\
    --val_dir   data/val   \\
    --epochs    50         \\
    --batch     8          \\
    --lr        3e-4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import build_dataloaders
from model import AttentionUNet


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Accepts raw logits — applies Sigmoid internally so it pairs cleanly
    with BCEWithLogitsLoss (no double-sigmoid).

    smooth : Laplace smoothing constant — prevents division by zero
             when both prediction and target are all-zero (no lesion image).
             Typical value: 1.0
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits  : (B, 1, H, W) — raw model output
        # targets : (B, 1, H, W) — binary mask {0.0, 1.0}
        probs = torch.sigmoid(logits)                        # (B, 1, H, W)

        # Flatten spatial dims for element-wise dot product
        probs_f   = probs.view(probs.size(0), -1)           # (B, H*W)
        targets_f = targets.view(targets.size(0), -1)       # (B, H*W)

        intersection = (probs_f * targets_f).sum(dim=1)     # (B,)
        union        = probs_f.sum(dim=1) + targets_f.sum(dim=1)  # (B,)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_sample.mean()                 # scalar


class CombinedLoss(nn.Module):
    """
    L_total = alpha * BCE + (1 - alpha) * Dice

    alpha=0.5 gives equal weight.  You can tune toward Dice (alpha<0.5)
    if your dataset is highly imbalanced (lesion area < 2% of image).

    pos_weight : Passed to BCEWithLogitsLoss.  If lesion pixels are ~5%
                 of total, set pos_weight=19.0 to up-weight positive class.
                 Leave as None to disable (Dice alone handles imbalance).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha    = alpha
        self.bce      = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice     = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute IoU (Jaccard) and Dice Score from raw logits.

    Parameters
    ----------
    logits    : (B, 1, H, W) raw model output — Sigmoid applied here.
    targets   : (B, 1, H, W) binary float mask {0.0, 1.0}.
    threshold : Binarisation threshold after Sigmoid.
    smooth    : Numerical stability constant.

    Returns
    -------
    (iou, dice) — Python floats, averaged over the batch.

    WHY IoU AND DICE?
    -----------------
    IoU = |P ∩ G| / |P ∪ G|   — penalises False Positives and False
                                   Negatives equally.  The standard metric
                                   for segmentation competitions (COCO, VOC).

    Dice = 2|P ∩ G| / (|P| + |G|) — related to IoU but less sensitive
                                      to large background regions.  Directly
                                      corresponds to the F1 score.

    Both are reported so we can monitor whether the model is precise
    (high IoU) and complete (high Dice).
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()  # (B, 1, H, W) binary

        preds_f   = preds.view(preds.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)

        intersection = (preds_f * targets_f).sum(dim=1)          # (B,)
        pred_sum     = preds_f.sum(dim=1)                        # (B,)
        tgt_sum      = targets_f.sum(dim=1)                      # (B,)

        union = pred_sum + tgt_sum - intersection                # (B,)

        iou  = (intersection + smooth) / (union + smooth)        # (B,)
        dice = (2.0 * intersection + smooth) / (pred_sum + tgt_sum + smooth)  # (B,)

    return iou.mean().item(), dice.mean().item()


# ============================================================================
# Device Detection
# ============================================================================

def get_device() -> torch.device:
    """
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    Prints the selected device for transparency.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        name   = "Apple MPS"
    else:
        device = torch.device("cpu")
        name   = "CPU"

    print(f"[Device] Using {device.type.upper()} — {name}")
    return device


# ============================================================================
# One Epoch — Train or Validate
# ============================================================================

def run_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    optimizer:  torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    Run one full pass over the DataLoader (train or val).

    Parameters
    ----------
    optimizer : Pass optimizer for training phase; None for validation.

    Returns
    -------
    Dict with keys: loss, iou, dice  (epoch averages)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc="  train" if is_train else "  val  ", leave=False)
        for images, masks in pbar:
            # images : (B, 3, H, W) float32
            # masks  : (B, 1, H, W) float32 {0, 1}
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            logits = model(images)              # (B, 1, H, W) raw logits

            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — prevents exploding gradients on small
                # batches where all pixels may be healthy (zero-mask batches)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_iou, batch_dice = compute_metrics(logits, masks)

            total_loss += loss.item()
            total_iou  += batch_iou
            total_dice += batch_dice
            n_batches  += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{batch_iou:.4f}")

    return {
        "loss": total_loss / n_batches,
        "iou":  total_iou  / n_batches,
        "dice": total_dice / n_batches,
    }


# ============================================================================
# Training Orchestrator
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Full training loop.

    Epoch flow
    ----------
    for epoch in range(num_epochs):
        train_metrics = run_epoch(train_loader, optimizer)
        val_metrics   = run_epoch(val_loader)
        scheduler.step(val_metrics["loss"])
        if val_iou improved:
            save checkpoint
        print epoch summary
    """

    device = get_device()

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print(f"\n[Data] Building DataLoaders...")
    train_loader, val_loader = build_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch,
        # num_workers=0 on Windows avoids multiprocessing deadlocks in some envs
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        mask_suffix=args.mask_suffix,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n[Model] Instantiating Attention U-Net...")
    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        base_features=args.base_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Optional: compute pos_weight from dataset stats for severe imbalance.
    # pos_weight = torch.tensor([19.0]).to(device)  # example: 5% lesion pixels
    criterion = CombinedLoss(alpha=args.loss_alpha).to(device)
    print(f"\n[Loss] CombinedLoss  alpha={args.loss_alpha} (BCE) / {1-args.loss_alpha:.2f} (Dice)")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # AdamW decouples L2 regularisation from the adaptive LR update.
    # This prevents weight decay from being amplified on params with small
    # gradient magnitudes — a common failure mode of standard Adam.
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    # ReduceLROnPlateau monitors val_loss and reduces LR when it stops
    # improving.  This is preferable to CosineAnnealing for segmentation
    # because disease datasets often have noisy val curves.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-7,
        verbose=True,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_iou = -1.0
    history: list[dict] = []

    SEP = "-" * 72
    print(f"\n{SEP}")
    print(f"  Starting training for {args.epochs} epochs")
    print(f"  LR={args.lr}  WeightDecay={args.weight_decay}  BatchSize={args.batch}")
    print(SEP)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train phase ───────────────────────────────────────────────────
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)

        # ── Val phase ─────────────────────────────────────────────────────
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        # ── Scheduler step ────────────────────────────────────────────────
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Epoch summary ─────────────────────────────────────────────────
        improved = val_metrics["iou"] > best_val_iou
        marker   = "  <-- BEST" if improved else ""

        print(
            f"Ep {epoch:03d}/{args.epochs}  "
            f"| T-Loss {train_metrics['loss']:.4f}  T-IoU {train_metrics['iou']:.4f}  T-Dice {train_metrics['dice']:.4f}"
            f"  | V-Loss {val_metrics['loss']:.4f}  V-IoU {val_metrics['iou']:.4f}  V-Dice {val_metrics['dice']:.4f}"
            f"  | LR {current_lr:.2e}  {elapsed:.1f}s{marker}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if improved:
            best_val_iou = val_metrics["iou"]
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optim_state":  optimizer.state_dict(),
                    "val_iou":      best_val_iou,
                    "val_dice":     val_metrics["dice"],
                    "args":         vars(args),
                },
                best_ckpt,
            )

        # ── History (for post-run analysis) ───────────────────────────────
        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_iou":  train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss":   val_metrics["loss"],
            "val_iou":    val_metrics["iou"],
            "val_dice":   val_metrics["dice"],
            "lr":         current_lr,
        })

    # ── Final summary ─────────────────────────────────────────────────────────
    print(SEP)
    print(f"  Training complete.")
    print(f"  Best Val IoU  : {best_val_iou:.4f}")
    print(f"  Checkpoint    : {best_ckpt}")
    print(SEP)

    # Save training history as CSV for easy plotting
    history_path = ckpt_dir / "training_history.csv"
    with open(history_path, "w") as f:
        header = ",".join(history[0].keys())
        f.write(header + "\n")
        for row in history:
            f.write(",".join(str(v) for v in row.values()) + "\n")
    print(f"  Training history saved to: {history_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Attention U-Net for Silkworm Anomaly Segmentation"
    )

    # Paths
    p.add_argument("--train_dir",  type=str, required=True,        help="Path to training split root (images/ + masks/)")
    p.add_argument("--val_dir",    type=str, required=True,        help="Path to validation split root")
    p.add_argument("--output_dir", type=str, default="models",     help="Directory to save best_model.pt")
    p.add_argument("--mask_suffix",type=str, default="",           help="Optional mask filename suffix (e.g. '_mask')")

    # Model
    p.add_argument("--base_features", type=int, default=64,        help="Base channel count for Attention U-Net")
    p.add_argument("--image_size",    type=int, default=256,       help="Square image size fed to the model")

    # Training
    p.add_argument("--epochs",      type=int,   default=50,        help="Number of training epochs")
    p.add_argument("--batch",       type=int,   default=8,         help="Mini-batch size")
    p.add_argument("--workers",     type=int,   default=0,         help="DataLoader workers (0 = safe on Windows)")
    p.add_argument("--lr",          type=float, default=3e-4,      help="Initial learning rate for AdamW")
    p.add_argument("--weight_decay",type=float, default=1e-4,      help="L2 weight decay for AdamW")
    p.add_argument("--loss_alpha",  type=float, default=0.5,       help="BCE weight in CombinedLoss (0=pure Dice, 1=pure BCE)")
    p.add_argument("--lr_factor",   type=float, default=0.5,       help="ReduceLROnPlateau factor")
    p.add_argument("--lr_patience", type=int,   default=5,         help="ReduceLROnPlateau patience (epochs)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
