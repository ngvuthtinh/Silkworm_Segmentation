"""
train.py — Training script for ViT-P Point-based Segmentation Classifier on Silkworm Mixed Dataset.
Hyperparameters strictly synchronized with segfirst_swinunet:
  - Input: 224x224
  - Batch size: 6
  - LR: 1e-4 (AdamW, betas=(0.9, 0.999), weight_decay=0.01)
  - Epochs: 50
  - AMP: Enabled
  - Real-time tqdm progress bar in terminal
"""

from __future__ import annotations

import os
import sys
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vit_p.config import ViTPConfig
from experiments.vit_p.dataset_adapter import SilkwormViTPDataset
from experiments.vit_p.model import ViTPSegClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] (Train)",
        leave=True,
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)  # (B, N)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images, points)  # (B, N, num_classes)
            # Flatten to (B * N, num_classes)
            loss = criterion(logits.view(-1, 2), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        total_loss += loss.item() * images.size(0)

        current_acc = (correct / total) * 100.0 if total > 0 else 0.0
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    avg_loss = total_loss / len(dataloader.dataset)
    epoch_acc = (correct / total) * 100.0
    return avg_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] (Valid)",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images, points)
            loss = criterion(logits.view(-1, 2), labels.view(-1))

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        total_loss += loss.item() * images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total)*100.0:.2f}%")

    avg_loss = total_loss / len(dataloader.dataset)
    epoch_acc = (correct / total) * 100.0
    return avg_loss, epoch_acc


def main():
    cfg = ViTPConfig()
    cfg.create_dirs()
    set_seed(cfg.seed)

    # Set CUDA device (CUDA_VISIBLE_DEVICES maps the chosen GPU to cuda:0)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 70)
    print(f"🚀 Launching ViT-P Training for Silkworm Mixed Dataset")
    print(f"   Resolution   : {cfg.input_size}x{cfg.input_size}")
    print(f"   Batch Size   : {cfg.batch_size}")
    print(f"   Epochs       : {cfg.epochs}")
    print(f"   LR           : {cfg.lr} (AdamW, wd={cfg.weight_decay})")
    print(f"   AMP          : {cfg.amp}")
    print(f"   Device       : {device_str} ({torch.cuda.get_device_name(device)})")
    print(f"   Output Dir   : {cfg.output_dir}")
    print("=" * 70)

    # 1. Dataset & DataLoaders
    train_dataset = SilkwormViTPDataset(
        split_dir=os.path.join(cfg.data_dir, "train"),
        input_size=cfg.input_size,
        num_points=cfg.num_points,
        is_train=True,
    )
    valid_dataset = SilkwormViTPDataset(
        split_dir=os.path.join(cfg.data_dir, "valid"),
        input_size=cfg.input_size,
        num_points=cfg.num_points,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Dataset loaded: Train={len(train_dataset)} samples, Valid={len(valid_dataset)} samples")

    # 2. Build Model
    model = ViTPSegClassifier(
        arch=cfg.arch,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        img_size=cfg.input_size,
        patch_size=cfg.patch_size,
        pretrained=True,
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "best_model.pth")
    final_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "model_final.pth")

    # 4. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )

        val_loss, val_acc = evaluate_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=cfg.epochs,
        )

        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )

        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} Summary | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% "
            f"{'(⭐ New Best)' if is_best else ''}"
        )

    # Save final model
    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
        },
        final_ckpt_path,
    )
    print(f"\n🎉 ViT-P Training Completed!")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best checkpoint saved to: {best_ckpt_path}")
    print(f"   Final checkpoint saved to: {final_ckpt_path}")


if __name__ == "__main__":
    main()
