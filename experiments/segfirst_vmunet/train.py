"""
train.py — Two-Phase Training Script for Segmentation-First Multi-Task VM-UNet.

Phase 1: Segmentation Pre-training on Dataset A (Silkynet) until stable.
Phase 2: Multi-Task Fine-Tuning on Dataset A (Seg) & Dataset B (Cls) with differential
         learning rates (lr_backbone=1e-5, lr_seg=1e-4, lr_cls=1e-3) and low lambda_cls=0.1.

Results are saved to: runs/segfirst_vmunet/<timestamp>/
    ├── checkpoints/    ← phase1_best.pth, phase2_best.pth
    ├── logs/           ← TensorBoard events
    └── test_outputs/   ← Visualization images
"""

from __future__ import annotations

import argparse
import csv
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

# ── Resolve project root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # experiments/segfirst_vmunet/
_PROJECT_ROOT = _HERE.parent.parent              # Silkworm_Segmentation/
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Experiment-local imports ──────────────────────────────────────────────────
from experiments.segfirst_vmunet.config import SegFirstConfig
from experiments.segfirst_vmunet.model import SegFirstVMUNet

# ── Shared src imports (with per-experiment override support) ─────────────────
try:
    from experiments.segfirst_vmunet.losses import SegFirstMultiTaskLoss  # local override
except ImportError:
    from src.losses import SegFirstMultiTaskLoss  # shared default

try:
    from experiments.segfirst_vmunet.dataset import SilkynetSegDataset, YOLOClsDataset  # local override
except ImportError:
    from src.dataset import SilkynetSegDataset, YOLOClsDataset  # shared default

from src.metrics import evaluate_seg, evaluate_cls


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: SegFirstConfig) -> None:
    # ── Device Setup ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        if cfg.gpu_id.lower() == "auto":
            best_gpu = 0
            max_free_mem = -1
            print("[INFO] Checking GPU availability:")
            for i in range(torch.cuda.device_count()):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024 ** 3)
                total_gb = total_mem / (1024 ** 3)
                print(f"  - GPU {i}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
                if free_mem > max_free_mem:
                    max_free_mem = free_mem
                    best_gpu = i
            gpu_idx = best_gpu
            print(f"  ==> Selected GPU {gpu_idx} (most free VRAM: {max_free_mem / (1024**3):.2f} GB)")
        else:
            gpu_idx = int(cfg.gpu_id) if cfg.gpu_id.isdigit() else 0
            if gpu_idx >= torch.cuda.device_count():
                print(f"[WARN] Requested GPU {gpu_idx} but only {torch.cuda.device_count()} GPU(s) visible. Using GPU 0.")
                gpu_idx = 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    set_seed(cfg.seed)
    cfg.create_dirs()

    print(f"\n{'='*60}")
    print(f" Segmentation-First Multi-Task VM-UNet Training Pipeline")
    print(f" Output: {cfg.work_dir}")
    print(f" Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f" Phase 1 Epochs: {cfg.phase1_epochs} | Phase 2 Epochs: {cfg.phase2_epochs}")
    print(f" Lambda Seg: {cfg.lambda_seg} | Lambda Cls: {cfg.lambda_cls}")
    print(f"{'='*60}\n")

    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── CSV log setup ─────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["phase", "epoch", "train_loss", "val_iou", "val_dice", "val_cls_acc"])

    # ── 1. Load Datasets ──────────────────────────────────────────────────────
    print("[1/5] Loading Datasets...")
    ds_seg_train = SilkynetSegDataset(cfg.silkynet_data_dir, img_size=cfg.input_size, is_train=True)
    ds_seg_val   = SilkynetSegDataset(cfg.silkynet_data_dir, img_size=cfg.input_size, is_train=False)
    ds_cls_train = YOLOClsDataset(cfg.yolo_data_dir, split="train", img_size=cfg.input_size, max_samples=cfg.max_cls_samples)
    ds_cls_val   = YOLOClsDataset(cfg.yolo_data_dir, split="val",   img_size=cfg.input_size, max_samples=200)

    loader_seg_train = DataLoader(ds_seg_train, batch_size=cfg.batch_size_seg, shuffle=True,  num_workers=cfg.num_workers)
    loader_seg_val   = DataLoader(ds_seg_val,   batch_size=cfg.batch_size_seg, shuffle=False, num_workers=cfg.num_workers)
    loader_cls_train = DataLoader(ds_cls_train, batch_size=cfg.batch_size_cls, shuffle=True,  num_workers=cfg.num_workers)
    loader_cls_val   = DataLoader(ds_cls_val,   batch_size=cfg.batch_size_cls, shuffle=False, num_workers=cfg.num_workers)

    print(f"  Dataset A (Seg): {len(ds_seg_train)} train / {len(ds_seg_val)} val")
    print(f"  Dataset B (Cls): {len(ds_cls_train)} train / {len(ds_cls_val)} val")

    # ── 2. Build Model ────────────────────────────────────────────────────────
    print("\n[2/5] Initializing SegFirstVMUNet...")
    model = SegFirstVMUNet(
        input_channels=3,
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        depths=cfg.depths,
        depths_decoder=cfg.depths_decoder,
        drop_path_rate=cfg.drop_path_rate,
        cls_hidden=cfg.cls_hidden,
        cls_dropout=cfg.cls_dropout,
    ).to(device)

    pretrained_full_path = _PROJECT_ROOT / cfg.pretrained_path
    if pretrained_full_path.exists():
        print(f"  Loading pretrained weights: {pretrained_full_path}")
        ckpt = torch.load(str(pretrained_full_path), map_location=device)
        state = ckpt.get("model", ckpt)
        model.backbone.load_state_dict(state, strict=False)

    loss_fn = SegFirstMultiTaskLoss(
        lambda_seg=cfg.lambda_seg,
        lambda_cls=cfg.lambda_cls,
        w_bce=cfg.w_bce,
        w_dice=cfg.w_dice,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    # ── 3. PHASE 1: Segmentation Pre-training ────────────────────────────────
    print("\n[3/5] Starting Phase 1: Segmentation Pre-training...")
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    sch_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=cfg.phase1_epochs)

    best_p1_dice = 0.0

    for ep in range(1, cfg.phase1_epochs + 1):
        model.train()
        ep_loss = 0.0

        pbar = tqdm(loader_seg_train, desc=f"Phase 1 [{ep:02d}/{cfg.phase1_epochs:02d}]", ncols=110, dynamic_ncols=True)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            opt_p1.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                mask_pred, _ = model(imgs, phase=1)
                loss, _, _ = loss_fn(mask_pred=mask_pred, mask_gt=masks)

            scaler.scale(loss).backward()
            scaler.step(opt_p1)
            scaler.update()

            ep_loss += loss.item() * len(imgs)
            pbar.set_postfix({"seg_loss": f"{loss.item():.4f}"})

        sch_p1.step()
        mean_loss = ep_loss / len(ds_seg_train)

        val_iou, val_dice = evaluate_seg(model, loader_seg_val, device, cfg.seg_threshold)
        writer.add_scalar("Phase1/TrainLoss", mean_loss, ep)
        writer.add_scalar("Phase1/ValIoU",    val_iou,   ep)
        writer.add_scalar("Phase1/ValDice",   val_dice,  ep)

        # CSV log
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([1, ep, f"{mean_loss:.6f}", f"{val_iou:.4f}", f"{val_dice:.4f}", ""])

        is_best = val_dice > best_p1_dice
        if is_best:
            best_p1_dice = val_dice
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "phase1_best.pth"))

        best_tag = " ★ NEW BEST!" if is_best else ""
        print(f"  --> [Phase 1 {ep:02d}/{cfg.phase1_epochs:02d}] Loss={mean_loss:.4f} | Dice={val_dice:.4f} | IoU={val_iou:.4f}{best_tag}")

    # Load best Phase 1 checkpoint for Phase 2
    phase1_ckpt = os.path.join(cfg.checkpoint_dir, "phase1_best.pth")
    if os.path.exists(phase1_ckpt):
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device))
        print(f"  Phase 1 complete. Loaded best weights from {phase1_ckpt}")

    # ── 4. PHASE 2: Multi-Task Fine-Tuning ───────────────────────────────────
    print("\n[4/5] Starting Phase 2: Multi-Task Fine-Tuning (Segmentation Protection)...")
    if cfg.freeze_backbone_phase2:
        model.freeze_backbone(True)
        print("  Backbone frozen to protect segmentation features.")

    param_groups = model.get_parameter_groups(
        lr_backbone=cfg.lr_backbone, lr_seg=cfg.lr_seg_head, lr_cls=cfg.lr_cls_head
    )
    opt_p2 = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    sch_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=cfg.phase2_epochs)

    best_p2_score = 0.0

    for ep in range(1, cfg.phase2_epochs + 1):
        model.train()
        total_p2_loss = 0.0

        iter_seg = iter(loader_seg_train)
        iter_cls = iter(loader_cls_train)
        max_steps = max(len(loader_seg_train), len(loader_cls_train))

        pbar = tqdm(range(max_steps), desc=f"Phase 2 [{ep:02d}/{cfg.phase2_epochs:02d}]", ncols=110, dynamic_ncols=True)
        for step in pbar:
            # Dataset A — Segmentation step
            try:
                imgs_A, masks_A = next(iter_seg)
            except StopIteration:
                iter_seg = iter(loader_seg_train)
                imgs_A, masks_A = next(iter_seg)

            imgs_A, masks_A = imgs_A.to(device), masks_A.to(device)
            opt_p2.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                mask_pred_A, _ = model(imgs_A, phase=2)
                loss_A, _, _ = loss_fn(mask_pred=mask_pred_A, mask_gt=masks_A)
            scaler.scale(loss_A).backward()
            scaler.step(opt_p2)
            scaler.update()

            # Dataset B — Classification step
            try:
                imgs_B, class_gt_B = next(iter_cls)
            except StopIteration:
                iter_cls = iter(loader_cls_train)
                imgs_B, class_gt_B = next(iter_cls)

            imgs_B, class_gt_B = imgs_B.to(device), class_gt_B.to(device)
            opt_p2.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                _, class_logits_B = model(imgs_B, phase=2)
                loss_B, _, _ = loss_fn(class_logits=class_logits_B, class_gt=class_gt_B)
            scaler.scale(loss_B).backward()
            scaler.step(opt_p2)
            scaler.update()

            total_p2_loss += (loss_A.item() + loss_B.item())
            pbar.set_postfix({"seg": f"{loss_A.item():.4f}", "cls": f"{loss_B.item():.4f}"})

        sch_p2.step()
        mean_p2_loss = total_p2_loss / max_steps

        val_iou, val_dice = evaluate_seg(model, loader_seg_val, device, cfg.seg_threshold)
        val_acc, _        = evaluate_cls(model, loader_cls_val, device)
        combined_score    = val_dice + 0.5 * val_acc

        writer.add_scalar("Phase2/TrainLoss",  mean_p2_loss, ep)
        writer.add_scalar("Phase2/ValSegDice", val_dice,     ep)
        writer.add_scalar("Phase2/ValSegIoU",  val_iou,      ep)
        writer.add_scalar("Phase2/ValClsAcc",  val_acc,      ep)

        # CSV log
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([2, ep, f"{mean_p2_loss:.6f}", f"{val_iou:.4f}", f"{val_dice:.4f}", f"{val_acc:.4f}"])

        is_best = combined_score > best_p2_score
        if is_best:
            best_p2_score = combined_score
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "phase2_best.pth"))

        # Save periodic checkpoint
        if ep % cfg.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"phase2_ep{ep:03d}.pth"))

        best_tag = " ★ NEW BEST!" if is_best else ""
        print(f"  --> [Phase 2 {ep:02d}/{cfg.phase2_epochs:02d}] Loss={mean_p2_loss:.4f} | Dice={val_dice:.4f} | IoU={val_iou:.4f} | Cls={val_acc:.4f}{best_tag}")

    print(f"\n[5/5] Training Complete!")
    print(f"  Results saved to: {cfg.work_dir}/")
    print(f"  Best Phase 1: {os.path.join(cfg.checkpoint_dir, 'phase1_best.pth')}")
    print(f"  Best Phase 2: {os.path.join(cfg.checkpoint_dir, 'phase2_best.pth')}")
    print(f"  Training log: {csv_path}")
    writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmentation-First Multi-Task VM-UNet Training")
    p.add_argument("--phase1-epochs",  type=int,   default=25)
    p.add_argument("--phase2-epochs",  type=int,   default=25)
    p.add_argument("--batch-size-seg", type=int,   default=2)
    p.add_argument("--batch-size-cls", type=int,   default=2)
    p.add_argument("--lr-backbone",    type=float, default=1e-5)
    p.add_argument("--lr-seg",         type=float, default=1e-4)
    p.add_argument("--lr-cls",         type=float, default=1e-3)
    p.add_argument("--lambda-seg",     type=float, default=1.0)
    p.add_argument("--lambda-cls",     type=float, default=0.1)
    p.add_argument("--input-size",     type=int,   default=128)
    p.add_argument("--data-dir",       type=str,   default="data/silkworm_mixed_dataset",
                   help="Path to segmentation dataset (Dataset A)")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--gpu",            type=str,   default="auto",
                   help="GPU id ('0', '1', or 'auto' to pick GPU with most free VRAM)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no-amp",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SegFirstConfig(
        silkynet_data_dir=args.data_dir,
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
        freeze_backbone_phase2=args.freeze_backbone,
        gpu_id=args.gpu,
        seed=args.seed,
        amp=not args.no_amp,
    )
    train(cfg)
