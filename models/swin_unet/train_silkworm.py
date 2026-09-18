#!/usr/bin/env python3
"""
train_silkworm.py — Train Swin-Unet on Silkworm Mixed Dataset
============================================================
Architecture: Swin-Unet (Shifted Windows Transformer U-Net)
Dataset: data/silkworm_mixed_dataset (RGB images + Binary masks)
Classes: 2 (0: Background, 1: Silkworm)
Loss: 0.4 * CrossEntropy + 0.6 * DiceLoss
Progress Logging: Real-time percentage (% completed) logged directly to terminal.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Add project paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import DiceLoss


class SilkwormMixedDataset(Dataset):
    """
    Loads (image, mask) pairs from silkworm_mixed_dataset.
    Images: .jpg or .png
    Masks: .png (values: 0=bg, 255=silkworm)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.is_train = split == "train"

        root = Path(data_root) / split
        img_dir = root / "images"
        mask_dir = root / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Missing images or masks directory in {root}")

        self.samples: List[Tuple[Path, Path]] = []
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            stem = img_p.stem
            # Match corresponding mask
            mask_p = mask_dir / f"{stem}.png"
            if not mask_p.exists():
                mask_p = mask_dir / f"{stem}.jpg"
            if mask_p.exists():
                self.samples.append((img_p, mask_p))

        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs found in {root}")

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p = self.samples[idx]

        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")

        # Resize
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        # Training data augmentation
        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        # Convert to tensors
        img_tensor = TF.to_tensor(img)
        img_tensor = self.norm(img_tensor)

        mask_np = np.array(mask)
        mask_tensor = (torch.from_numpy(mask_np) > 127).long()

        return img_tensor, mask_tensor


def compute_dice_score(pred_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> float:
    """Compute binary Dice score for foreground class (silkworm)."""
    preds = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
    pred_fg = (preds == 1).float()
    target_fg = (targets == 1).float()

    intersection = (pred_fg * target_fg).sum()
    total = pred_fg.sum() + target_fg.sum()
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return float(dice.item())


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train_silkworm(args) -> None:
    # Setup Device
    gpu_id = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"

    # Logger setup
    logger = logging.getLogger("SwinUnet_Silkworm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    def log_print(msg: str) -> None:
        print(msg, flush=True)
        logger.info(msg)

    log_print("=" * 80)
    log_print("   🐛 SWIN-UNET TRAINING PIPELINE — SILKWORM SEGMENTATION")
    log_print("=" * 80)
    log_print(f" Dataset Root     : {args.data_dir}")
    log_print(f" Model Config     : {args.cfg}")
    log_print(f" Output Directory : {args.output_dir}")
    log_print(f" Batch Size       : {args.batch_size}")
    log_print(f" Image Size       : {args.img_size}x{args.img_size}")
    log_print(f" Max Epochs       : {args.max_epochs}")
    log_print(f" Base LR          : {args.base_lr}")
    log_print(f" Device           : {device} (GPU ID: {gpu_id})")
    log_print("=" * 80)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Datasets & DataLoaders
    train_dataset = SilkwormMixedDataset(args.data_dir, split="train", img_size=args.img_size)
    val_dataset = SilkwormMixedDataset(args.data_dir, split="valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    log_print(f" Loaded {len(train_dataset)} training samples ({len(train_loader)} batches/epoch)")
    log_print(f" Loaded {len(val_dataset)} validation samples ({len(val_loader)} batches/epoch)")

    # Model
    config = get_config(args)
    model = ViT_seg(config, img_size=args.img_size, num_classes=2).to(device)
    model.load_from(config)
    log_print(" Swin-Unet initialized & pretrained weights loaded successfully!\n")

    # Loss & Optimizer
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    total_epochs = args.max_epochs
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch

    global_step = 0
    best_val_loss = float("inf")
    best_val_dice = 0.0
    start_train_time = time.time()

    log_print("🚀 STARTING TRAINING LOOP...")
    log_print("=" * 80)

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()

        train_ce_total = 0.0
        train_dice_total = 0.0
        train_loss_total = 0.0

        for step, (images, masks) in enumerate(train_loader, 1):
            global_step += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss_ce = ce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Poly LR decay
            lr_current = args.base_lr * (1.0 - (global_step / total_steps)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_current

            train_ce_total += loss_ce.item()
            train_dice_total += loss_dice.item()
            train_loss_total += loss.item()

            # Progress % calculations
            step_pct = (step / steps_per_epoch) * 100.0
            epoch_pct = ((epoch + step / steps_per_epoch) / total_epochs) * 100.0

            # Real-time percentage logging to terminal every 5 steps or first/last step
            if step == 1 or step % 5 == 0 or step == steps_per_epoch:
                log_msg = (
                    f"[{epoch_pct:5.1f}% TOTAL] [Epoch {epoch+1:03d}/{total_epochs:03d} | Step {step:02d}/{steps_per_epoch:02d} ({step_pct:5.1f}%)] "
                    f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Dice: {loss_dice.item():.4f}) | LR: {lr_current:.5f}"
                )
                print(log_msg, flush=True)

        # Epoch Train Metrics
        train_loss_epoch = train_loss_total / steps_per_epoch
        train_ce_epoch = train_ce_total / steps_per_epoch
        train_dice_epoch = train_dice_total / steps_per_epoch

        # Validation Phase
        model.eval()
        val_ce_total = 0.0
        val_dice_loss_total = 0.0
        val_loss_total = 0.0
        val_dice_scores = []

        with torch.no_grad():
            for val_imgs, val_masks in val_loader:
                val_imgs = val_imgs.to(device, non_blocking=True)
                val_masks = val_masks.to(device, non_blocking=True)

                val_preds = model(val_imgs)
                v_ce = ce_loss(val_preds, val_masks)
                v_dice = dice_loss(val_preds, val_masks, softmax=True)
                v_loss = 0.4 * v_ce + 0.6 * v_dice

                val_ce_total += v_ce.item()
                val_dice_loss_total += v_dice.item()
                val_loss_total += v_loss.item()

                batch_dice = compute_dice_score(val_preds, val_masks)
                val_dice_scores.append(batch_dice)

        val_loss_epoch = val_loss_total / len(val_loader)
        val_ce_epoch = val_ce_total / len(val_loader)
        val_dice_epoch = val_dice_loss_total / len(val_loader)
        val_dice_score_epoch = float(np.mean(val_dice_scores))

        epoch_time = time.time() - epoch_start_time
        elapsed_total = time.time() - start_train_time
        completed_epochs = epoch + 1
        eta_seconds = (elapsed_total / completed_epochs) * (total_epochs - completed_epochs)
        overall_pct = (completed_epochs / total_epochs) * 100.0

        # Checkpoint Saving
        is_best = val_loss_epoch < best_val_loss
        checkpoint_status = ""
        if is_best:
            best_val_loss = val_loss_epoch
            best_val_dice = val_dice_score_epoch
            best_path = out_dir / "best_model.pth"
            torch.save(model.state_dict(), str(best_path))
            checkpoint_status = f"🌟 NEW BEST! Saved to {best_path.name}"
        else:
            checkpoint_status = f"Best Val Loss remains: {best_val_loss:.4f} (Dice: {best_val_dice:.2%})"

        # Always save latest
        latest_path = out_dir / "latest_model.pth"
        torch.save(model.state_dict(), str(latest_path))

        # Epoch Summary Block
        log_print("-" * 80)
        log_print(
            f"🎯 [HOÀN THÀNH EPOCH {completed_epochs:03d}/{total_epochs:03d} — TIẾN ĐỘ: {overall_pct:5.1f}%] (Thời gian: {format_time(epoch_time)} | Còn lại: ~{format_time(eta_seconds)})"
        )
        log_print(
            f"   Train Loss: {train_loss_epoch:.4f} (CE: {train_ce_epoch:.4f}, Dice: {train_dice_epoch:.4f})"
        )
        log_print(
            f"   Val Loss  : {val_loss_epoch:.4f} (CE: {val_ce_epoch:.4f}, Dice: {val_dice_epoch:.4f}) | Silkworm Val Dice: {val_dice_score_epoch:.2%}"
        )
        log_print(f"   Checkpoint: {checkpoint_status}")
        log_print("-" * 80)

    total_time = time.time() - start_train_time
    log_print("\n" + "=" * 80)
    log_print(f"🎉 HUẤN LUYỆN HOÀN TẤT 100%! Tổng thời gian: {format_time(total_time)}")
    log_print(f" Best Val Loss: {best_val_loss:.4f} | Best Silkworm Dice: {best_val_dice:.2%}")
    log_print(f" Checkpoint tốt nhất: {out_dir / 'best_model.pth'}")
    log_print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train Swin-Unet on Silkworm Mixed Dataset")
    parser.add_argument("--data_dir", type=str, default="../data/silkworm_mixed_dataset")
    parser.add_argument("--cfg", type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--output_dir", type=str, default="./silkworm_model_out")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)

    # Required Swin-Unet dummy config arguments
    parser.add_argument("--dataset", type=str, default="silkworm")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O1")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", default=None, nargs="+")

    args = parser.parse_args()
    train_silkworm(args)


if __name__ == "__main__":
    main()
