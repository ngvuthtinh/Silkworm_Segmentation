"""
train.py — Joint Multi-Task Training for VM-UNet (Segmentation + Classification).

Trains VM-UNet on both segmentation and classification simultaneously.
Results are saved to: runs/multitask_vmunet/<timestamp>/
    ├── checkpoints/    ← best.pth, latest.pth
    ├── logs/           ← TensorBoard events & training log
    └── visualizations/ ← Prediction sample plots
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Resolve project root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Experiment-local imports ──────────────────────────────────────────────────
from experiments.multitask_vmunet.config import MultiTaskConfig
from experiments.multitask_vmunet.model import MultiTaskVMUNet

# ── Shared src imports ────────────────────────────────────────────────────────
from src.dataset import SilkynetSegDataset, YOLOClsDataset
from src.losses import SegLoss, ClsLoss
from src.metrics import evaluate_seg, evaluate_cls


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: MultiTaskConfig) -> None:
    if torch.cuda.is_available():
        if cfg.gpu_id.lower() == "auto":
            best_gpu = 0
            max_free_mem = -1
            for i in range(torch.cuda.device_count()):
                free_mem, _ = torch.cuda.mem_get_info(i)
                if free_mem > max_free_mem:
                    max_free_mem = free_mem
                    best_gpu = i
            gpu_idx = best_gpu
        else:
            gpu_idx = int(cfg.gpu_id) if cfg.gpu_id.isdigit() else 0
            if gpu_idx >= torch.cuda.device_count():
                gpu_idx = 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    set_seed(cfg.seed)
    cfg.create_dirs()

    print("=" * 80)
    print(" 🚀 Joint Multi-Task VM-UNet Training Pipeline")
    print(f" Output Directory: {cfg.work_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f" Epochs: {cfg.epochs} | LR: {cfg.lr}")
    print(f" Lambda Seg: {cfg.lambda_seg} | Lambda Cls: {cfg.lambda_cls}")
    print("=" * 80)

    # 1. Datasets
    ds_seg_train = SilkynetSegDataset(
        data_root=os.path.join(_PROJECT_ROOT, cfg.silkynet_data_dir),
        img_size=cfg.input_size,
        is_train=True,
    )
    ds_seg_val = SilkynetSegDataset(
        data_root=os.path.join(_PROJECT_ROOT, cfg.silkynet_data_dir),
        img_size=cfg.input_size,
        is_train=False,
    )
    ds_cls_train = YOLOClsDataset(
        data_root=os.path.join(_PROJECT_ROOT, cfg.yolo_data_dir),
        split="train",
        img_size=cfg.input_size,
        max_samples=cfg.max_cls_samples,
        seed=cfg.seed,
    )
    ds_cls_val = YOLOClsDataset(
        data_root=os.path.join(_PROJECT_ROOT, cfg.yolo_data_dir),
        split="valid",
        img_size=cfg.input_size,
        max_samples=cfg.max_cls_samples,
        seed=cfg.seed,
    )

    loader_seg_train = DataLoader(
        ds_seg_train, batch_size=cfg.batch_size_seg, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    loader_seg_val = DataLoader(
        ds_seg_val, batch_size=cfg.batch_size_seg, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    loader_cls_train = DataLoader(
        ds_cls_train, batch_size=cfg.batch_size_cls, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    loader_cls_val = DataLoader(
        ds_cls_val, batch_size=cfg.batch_size_cls, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    print(f"  Dataset A (Seg): {len(ds_seg_train)} train | {len(ds_seg_val)} val")
    print(f"  Dataset B (Cls): {len(ds_cls_train)} train | {len(ds_cls_val)} val\n")

    # 2. Model
    model = MultiTaskVMUNet(
        input_channels=3,
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        depths=cfg.depths,
        depths_decoder=cfg.depths_decoder,
        drop_path_rate=cfg.drop_path_rate,
        load_ckpt_path=os.path.join(_PROJECT_ROOT, cfg.pretrained_path),
        cls_hidden=cfg.cls_hidden,
        cls_dropout=cfg.cls_dropout,
    ).to(device)

    model.load_pretrained()

    seg_loss_fn = SegLoss(w_bce=cfg.w_bce, w_dice=cfg.w_dice)
    cls_loss_fn = ClsLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")
    writer = SummaryWriter(cfg.log_dir)

    best_combined_score = 0.0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        iter_seg = iter(loader_seg_train)
        iter_cls = iter(loader_cls_train)
        max_steps = max(len(loader_seg_train), len(loader_cls_train))

        pbar = tqdm(range(max_steps), desc=f"Epoch [{ep:02d}/{cfg.epochs:02d}]", ncols=90)
        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)

            try:
                imgs_A, masks_A = next(iter_seg)
            except StopIteration:
                iter_seg = iter(loader_seg_train)
                imgs_A, masks_A = next(iter_seg)

            try:
                imgs_B, labels_B = next(iter_cls)
            except StopIteration:
                iter_cls = iter(loader_cls_train)
                imgs_B, labels_B = next(iter_cls)

            imgs_A, masks_A = imgs_A.to(device), masks_A.to(device)
            imgs_B, labels_B = imgs_B.to(device), labels_B.to(device)

            with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                mask_pred_A, _ = model(imgs_A)
                loss_seg = seg_loss_fn(mask_pred_A, masks_A)

                _, class_logits_B = model(imgs_B)
                loss_cls = cls_loss_fn(class_logits_B, labels_B)

                loss_step = cfg.lambda_seg * loss_seg + cfg.lambda_cls * loss_cls

            scaler.scale(loss_step).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_step.item()
            pbar.set_postfix({"seg": f"{loss_seg.item():.4f}", "cls": f"{loss_cls.item():.4f}"})

        scheduler.step()
        mean_loss = total_loss / max_steps

        # Validation
        val_iou, val_dice = evaluate_seg(model, loader_seg_val, device, cfg.seg_threshold)
        val_acc, val_f1 = evaluate_cls(model, loader_cls_val, device)
        combined_score = val_dice + 0.5 * val_acc

        writer.add_scalar("Train/Loss", mean_loss, ep)
        writer.add_scalar("Val/Dice", val_dice, ep)
        writer.add_scalar("Val/IoU", val_iou, ep)
        writer.add_scalar("Val/Accuracy", val_acc, ep)
        writer.add_scalar("Val/F1", val_f1, ep)

        is_best = combined_score > best_combined_score
        if is_best:
            best_combined_score = combined_score
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "best.pth"))

        if ep % cfg.save_interval == 0 or ep == cfg.epochs:
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"epoch_{ep:03d}.pth"))

        best_tag = " (NEW BEST!)" if is_best else ""
        print(f"  [Ep {ep:02d}/{cfg.epochs:02d}] Loss={mean_loss:.4f} | Seg Dice={val_dice:.4f} | IoU={val_iou:.4f} | Cls Acc={val_acc:.4f} (Score={combined_score:.4f}){best_tag}")

    print("\n" + "=" * 80)
    print("🎉 Huấn luyện Joint Multi-Task VM-UNet hoàn tất!")
    print(f"  Best Checkpoint: {os.path.join(cfg.checkpoint_dir, 'best.pth')} (Score: {best_combined_score:.4f})")
    print("=" * 80)
    writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Task VM-UNet Training")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size-seg", type=int, default=2)
    p.add_argument("--batch-size-cls", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-seg", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MultiTaskConfig(
        epochs=args.epochs,
        batch_size_seg=args.batch_size_seg,
        batch_size_cls=args.batch_size_cls,
        lr=args.lr,
        lambda_seg=args.lambda_seg,
        lambda_cls=args.lambda_cls,
        gpu_id=args.gpu,
        seed=args.seed,
        amp=not args.no_amp,
    )
    train(cfg)
