#!/usr/bin/env python3
"""
experiments/multitask_vmunet/train.py
End-to-End Multi-Task VM-UNet Training (Segmentation + Classification).

Goal: Trên 1 tấm ảnh, phân đoạn con tằm VÀ chẩn đoán bệnh cùng lúc.
  - Nhánh Seg:  Dự đoán Mask thân tằm (nhị phân 0/1)
  - Nhánh Cls:  Dự đoán nhãn bệnh (0=Grasserie, 1=Healthy)
  - Loss:       Dice + BCE (seg) + CrossEntropy (cls) tối ưu đồng thời trong 1 lần backward

Chạy thử (smoke-test 5 batch):
  ./.venv/bin/python experiments/multitask_vmunet/train.py --smoke-test

Chạy thật:
  CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python experiments/multitask_vmunet/train.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models" / "vmunet") not in sys.path:
    sys.path.insert(0, str(ROOT / "models" / "vmunet"))

from src.dataset_multitask import MultitaskSilkwormDataset
from experiments.multitask_vmunet.model import MultiTaskVMUNet

# ─── Hàm mất mát ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.contiguous().view(-1)
        t = target.contiguous().view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class MultitaskLoss(nn.Module):
    """Loss = lambda_seg*(BCE+Dice) + lambda_cls*CrossEntropy."""
    def __init__(self, lambda_seg: float = 1.0, lambda_cls: float = 1.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(
        self,
        mask_logits:   torch.Tensor,
        mask_gt:       torch.Tensor,
        cls_logits:    torch.Tensor,
        cls_gt:        torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_bce  = self.bce(mask_logits, mask_gt)
        l_dice = self.dice(torch.sigmoid(mask_logits), mask_gt)
        l_seg  = l_bce + l_dice
        l_cls  = self.ce(cls_logits, cls_gt)
        total  = self.lambda_seg * l_seg + self.lambda_cls * l_cls
        return total, l_seg, l_cls


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return (2.0 * inter / (denom + 1e-6)).item()


def cls_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    DATA_DIR  = ROOT / "data" / "Silkworm_mixed_dataset_10k"
    IMG_SIZE  = 256
    BATCH     = 8 if not args.smoke_test else 2
    EPOCHS    = 40 if not args.smoke_test else 1
    LR        = 1e-4
    WORKERS   = 4
    SMOKE_N   = 5

    device = _pick_device(args)
    print(f"\n{'='*70}")
    print(f" 🐛 Multi-Task VM-UNet — End-to-End Training")
    print(f"    Dataset : {DATA_DIR}")
    print(f"    Img Size: {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH}  |  Epochs: {EPOCHS}")
    print(f"    Device  : {device}")
    print(f"    Mode    : {'SMOKE-TEST (5 batch)' if args.smoke_test else 'FULL TRAIN'}")
    print(f"{'='*70}\n")

    ds_train = MultitaskSilkwormDataset(DATA_DIR, "train", IMG_SIZE, augment=True)
    ds_val   = MultitaskSilkwormDataset(DATA_DIR, "valid", IMG_SIZE, augment=False)
    dl_train = DataLoader(ds_train, BATCH, shuffle=True,  num_workers=WORKERS, pin_memory=True)
    dl_val   = DataLoader(ds_val,   BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True)
    print(f"  Train: {len(ds_train)} mẫu ({len(dl_train)} batches)  |  Val: {len(ds_val)} mẫu")

    model = MultiTaskVMUNet(
        input_channels=3,
        num_seg_classes=1,
        num_cls_classes=2,
        load_ckpt_path=None,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params : {total_params:.1f}M\n")

    loss_fn   = MultitaskLoss(lambda_seg=1.0, lambda_cls=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler()

    run_tag  = "smoke_test" if args.smoke_test else datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir  = ROOT / "runs" / "multitask_vmunet" / run_tag
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = t_seg = t_cls = 0.0
        n_batches = min(SMOKE_N, len(dl_train)) if args.smoke_test else len(dl_train)
        pct_epoch = epoch / EPOCHS * 100

        pbar = tqdm(
            enumerate(dl_train),
            total=n_batches,
            desc=f"[VM-UNet] Epoch {epoch:02d}/{EPOCHS} ({pct_epoch:5.1f}%) [TRAIN]",
            ncols=110,
            leave=True,
        )
        for i, (imgs, masks, labels) in pbar:
            if i >= n_batches:
                break
            imgs   = imgs.to(device)
            masks  = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                mask_logits, cls_logits = model(imgs)
                loss, l_seg, l_cls = loss_fn(mask_logits, masks, cls_logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item(); t_seg += l_seg.item(); t_cls += l_cls.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "seg":  f"{l_seg.item():.3f}",
                "cls":  f"{l_cls.item():.3f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
            })

        scheduler.step()
        mean_loss = t_loss / n_batches

        if args.smoke_test:
            print(f"  [SMOKE] Epoch {epoch} OK — loss={mean_loss:.4f}")
            break

        model.eval()
        val_dice = val_acc = 0.0
        v_steps  = 0
        with torch.no_grad():
            for imgs, masks, labels in tqdm(dl_val, desc=f"  → [VM-UNet] Val ", ncols=110, leave=False):
                imgs   = imgs.to(device)
                masks  = masks.to(device)
                labels = labels.to(device)
                with autocast(device_type="cuda"):
                    mask_logits, cls_logits = model(imgs)
                val_dice += dice_score(mask_logits, masks)
                val_acc  += cls_accuracy(cls_logits, labels)
                v_steps  += 1

        val_dice /= v_steps
        val_acc  /= v_steps
        combined  = 0.7 * val_dice + 0.3 * val_acc

        best_tag = ""
        if combined > best_score:
            best_score = combined
            torch.save({"epoch": epoch, "model": model.state_dict(), "score": best_score},
                       ckpt_dir / "best.pth")
            best_tag = " ★ BEST"

        print(
            f"  [VM-UNet] Ep {epoch:02d}/{EPOCHS} "
            f"({pct_epoch:5.1f}%) | "
            f"Loss={mean_loss:.4f} | Dice={val_dice:.4f} | Acc={val_acc:.4f} | "
            f"Score={combined:.4f}{best_tag}"
        )

        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / f"ep{epoch:03d}.pth")

    torch.save({"epoch": EPOCHS, "model": model.state_dict()}, ckpt_dir / "last.pth")
    if not args.smoke_test:
        print(f"\n✅ [VM-UNet] Hoàn thành! Best score={best_score:.4f} | Saved: {out_dir}")
    else:
        print(f"\n✅ [VM-UNet] Smoke-test PASSED!")


def _pick_device(args: argparse.Namespace) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if hasattr(args, "gpu") and str(args.gpu).isdigit():
        idx = int(args.gpu)
        if idx < torch.cuda.device_count():
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
    best, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best = i
    torch.cuda.set_device(best)
    return torch.device(f"cuda:{best}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-Task VM-UNet Trainer")
    p.add_argument("--smoke-test", action="store_true", help="Chạy thử 5 batch")
    p.add_argument("--gpu", type=str, default="auto", help="GPU index hoặc 'auto'")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
