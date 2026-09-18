"""
train.py — Two-Phase Training Pipeline for Segmentation-First Multi-Task Swin-Unet.

Strategy:
  Phase 1: Pure segmentation pre-training on silkworm mask data.
  Phase 2: Multi-task fine-tuning with differential learning rates:
           - Small LR on backbone (preserve learned representations)
           - Standard LR on segmentation head
           - Higher LR on classification head
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Resolve project root
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import SilkynetSegDataset, YOLOClsDataset
from src.losses import SegFirstMultiTaskLoss
from src.metrics import evaluate_seg, evaluate_cls
from experiments.segfirst_swinunet.config import SegFirstSwinConfig
from experiments.segfirst_swinunet.model import SegFirstSwinUnet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: SegFirstSwinConfig) -> None:
    if torch.cuda.is_available():
        if cfg.gpu_id.lower() == "auto":
            best_gpu, max_free_mem = 0, -1
            for i in range(torch.cuda.device_count()):
                free_mem, _ = torch.cuda.mem_get_info(i)
                if free_mem > max_free_mem:
                    max_free_mem, best_gpu = free_mem, i
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
    print(" 🌟 Segmentation-First Multi-Task Swin-Unet Training Pipeline")
    print(f" Output Directory: {cfg.work_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f" Phase 1 Epochs: {cfg.phase1_epochs} | Phase 2 Epochs: {cfg.phase2_epochs}")
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
    model = SegFirstSwinUnet(
        config=cfg,
        input_size=cfg.input_size,
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        cls_hidden=cfg.cls_hidden,
        cls_dropout=cfg.cls_dropout,
    ).to(device)

    # Load pretrained Swin-T ImageNet weights
    pretrained_full_path = os.path.join(_PROJECT_ROOT, cfg.pretrained_path)
    model.load_from(pretrained_full_path)

    loss_fn = SegFirstMultiTaskLoss(
        lambda_seg=cfg.lambda_seg,
        lambda_cls=cfg.lambda_cls,
        w_bce=cfg.w_bce,
        w_dice=cfg.w_dice,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    writer = SummaryWriter(cfg.log_dir)

    # =========================================================================
    # PHASE 1: SEGMENTATION PRE-TRAINING
    # =========================================================================
    print("=" * 80)
    print("🚀 [PHASE 1] SEGMENTATION PRE-TRAINING (Silkynet / Mixed Dataset)")
    print("=" * 80)

    opt_p1 = torch.optim.AdamW(
        model.swin_unet.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay, betas=cfg.betas
    )
    sch_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=cfg.phase1_epochs, eta_min=1e-6)

    best_p1_dice = 0.0
    for ep in range(1, cfg.phase1_epochs + 1):
        model.train()
        total_p1_loss = 0.0
        pbar = tqdm(loader_seg_train, desc=f"Phase1 [{ep:02d}/{cfg.phase1_epochs:02d}]", ncols=85)

        for step, (imgs, masks) in enumerate(pbar, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            opt_p1.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                mask_logits, _ = model(imgs, phase=1)
                loss_tot, loss_seg, _ = loss_fn(mask_pred=mask_logits, mask_gt=masks)

            scaler.scale(loss_tot).backward()
            scaler.step(opt_p1)
            scaler.update()

            total_p1_loss += loss_seg.item()
            pbar.set_postfix({"seg_loss": f"{loss_seg.item():.4f}"})

        sch_p1.step()
        mean_p1_loss = total_p1_loss / len(loader_seg_train)

        val_iou, val_dice = evaluate_seg(model, loader_seg_val, device, cfg.seg_threshold)
        writer.add_scalar("Phase1/TrainLoss", mean_p1_loss, ep)
        writer.add_scalar("Phase1/ValDice", val_dice, ep)
        writer.add_scalar("Phase1/ValIoU", val_iou, ep)

        is_best = val_dice > best_p1_dice
        if is_best:
            best_p1_dice = val_dice
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "phase1_best.pth"))

        best_tag = " (NEW BEST!)" if is_best else ""
        pct = (ep / cfg.phase1_epochs) * 100.0
        print(f"  [{pct:5.1f}% Phase 1 Ep {ep:02d}/{cfg.phase1_epochs:02d}] Loss={mean_p1_loss:.4f} | Val Dice={val_dice:.4f} | IoU={val_iou:.4f}{best_tag}")

    # =========================================================================
    # PHASE 2: MULTI-TASK FINE-TUNING (DIFFERENTIAL LEARNING RATES)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🚀 [PHASE 2] MULTI-TASK FINE-TUNING (Differential LR + Joint Loss)")
    print("=" * 80)

    best_p1_path = os.path.join(cfg.checkpoint_dir, "phase1_best.pth")
    if os.path.exists(best_p1_path):
        model.load_state_dict(torch.load(best_p1_path, map_location=device))
        print(f"  Loaded best Phase 1 checkpoint from: {best_p1_path}")

    param_groups = model.get_parameter_groups(
        lr_backbone=cfg.lr_backbone, lr_seg=cfg.lr_seg_head, lr_cls=cfg.lr_cls_head
    )
    opt_p2 = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=cfg.betas)
    sch_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=cfg.phase2_epochs, eta_min=1e-6)

    best_p2_score = 0.0
    for ep in range(1, cfg.phase2_epochs + 1):
        model.train()
        total_p2_loss = 0.0

        iter_seg = iter(loader_seg_train)
        iter_cls = iter(loader_cls_train)
        max_steps = max(len(loader_seg_train), len(loader_cls_train))

        pbar = tqdm(range(max_steps), desc=f"Phase2 [{ep:02d}/{cfg.phase2_epochs:02d}]", ncols=85)
        for _ in pbar:
            opt_p2.zero_grad()

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

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                mask_logits_A, _ = model(imgs_A, phase=1)
                loss_A, _, _ = loss_fn(mask_pred=mask_logits_A, mask_gt=masks_A)

                _, class_logits_B = model(imgs_B, phase=2)
                loss_B, _, _ = loss_fn(class_logits=class_logits_B, class_gt=labels_B)

                loss_step = loss_A + loss_B

            scaler.scale(loss_step).backward()
            scaler.step(opt_p2)
            scaler.update()

            total_p2_loss += (loss_A.item() + loss_B.item())
            pbar.set_postfix({"seg_loss": f"{loss_A.item():.4f}", "cls_loss": f"{loss_B.item():.4f}"})

        sch_p2.step()
        mean_p2_loss = total_p2_loss / max_steps

        val_iou, val_dice = evaluate_seg(model, loader_seg_val, device, cfg.seg_threshold)
        val_acc, _ = evaluate_cls(model, loader_cls_val, device)
        combined_score = val_dice + 0.5 * val_acc

        writer.add_scalar("Phase2/TrainLoss", mean_p2_loss, ep)
        writer.add_scalar("Phase2/ValSegDice", val_dice, ep)
        writer.add_scalar("Phase2/ValSegIoU", val_iou, ep)
        writer.add_scalar("Phase2/ValClsAcc", val_acc, ep)

        is_best = combined_score > best_p2_score
        if is_best:
            best_p2_score = combined_score
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "phase2_best.pth"))

        best_tag = " (NEW BEST!)" if is_best else ""
        pct = (ep / cfg.phase2_epochs) * 100.0
        print(f"  [{pct:5.1f}% Phase 2 Ep {ep:02d}/{cfg.phase2_epochs:02d}] Loss={mean_p2_loss:.4f} | Seg Dice={val_dice:.4f} | IoU={val_iou:.4f} | Cls Acc={val_acc:.4f} (Score={combined_score:.4f}){best_tag}")

    print("\n" + "=" * 80)
    print("🎉 Huấn luyện SegFirst Multi-Task Swin-Unet hoàn tất!")
    print(f"  Phase 1 Best: {os.path.join(cfg.checkpoint_dir, 'phase1_best.pth')}")
    print(f"  Phase 2 Best: {os.path.join(cfg.checkpoint_dir, 'phase2_best.pth')} (Score: {best_p2_score:.4f})")
    print("=" * 80)
    writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmentation-First Multi-Task Swin-Unet Training")
    p.add_argument("--phase1-epochs", type=int, default=25)
    p.add_argument("--phase2-epochs", type=int, default=25)
    p.add_argument("--batch-size-seg", type=int, default=6)
    p.add_argument("--batch-size-cls", type=int, default=6)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-seg", type=float, default=1e-4)
    p.add_argument("--lr-cls", type=float, default=1e-3)
    p.add_argument("--lambda-seg", type=float, default=1.0)
    p.add_argument("--lambda-cls", type=float, default=0.1)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--data-dir", type=str, default="data/silkworm_mixed_dataset")
    p.add_argument("--yolo-dir", type=str, default="data/Silkworm Diseases.v1i.yolo26")
    p.add_argument("--gpu", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SegFirstSwinConfig(
        silkynet_data_dir=args.data_dir,
        yolo_data_dir=args.yolo_dir,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size_seg=args.batch_size_seg,
        batch_size_cls=args.batch_size_cls,
        lr_backbone=args.lr_backbone,
        lr_seg_head=args.lr_seg,
        lr_cls_head=args.lr_cls,
        lambda_seg=args.lambda_seg,
        lambda_cls=args.lambda_cls,
        input_size=args.input_size,
        gpu_id=args.gpu,
        seed=args.seed,
        amp=not args.no_amp,
    )
    train(cfg)
