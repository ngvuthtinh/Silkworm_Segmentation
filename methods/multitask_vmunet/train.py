"""
train.py — Multi-task training loop for VM-UNet segmentation + classification.

Usage (from project root Silkworm_Segmentation/):

    python -m methods.multitask_vmunet.train

Or with custom overrides:
    python -m methods.multitask_vmunet.train \
        --epochs 100 \
        --batch-size-seg 8 \
        --batch-size-cls 32 \
        --lr 5e-5 \
        --gpu 0
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve module paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "models" / "vmunet"))


from methods.multitask_vmunet.config  import MultiTaskConfig   # noqa
from methods.multitask_vmunet.dataset import build_seg_loaders, build_cls_loaders  # noqa
from methods.multitask_vmunet.losses  import MultiTaskLoss     # noqa
from methods.multitask_vmunet.model   import MultiTaskVMUNet   # noqa


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_seg_metrics(
    preds: np.ndarray,
    gts: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mIoU and Dice from flattened arrays."""
    pred_bin = (preds >= threshold).astype(np.int32)
    gt_bin   = (gts   >= threshold).astype(np.int32)

    TP = ((pred_bin == 1) & (gt_bin == 1)).sum()
    FP = ((pred_bin == 1) & (gt_bin == 0)).sum()
    FN = ((pred_bin == 0) & (gt_bin == 1)).sum()
    TN = ((pred_bin == 0) & (gt_bin == 0)).sum()

    dice = (2 * TP + 1e-6) / (2 * TP + FP + FN + 1e-6)
    iou  = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    acc  = (TP + TN + 1e-6) / (TP + TN + FP + FN + 1e-6)

    return {"dice": float(dice), "miou": float(iou), "acc_seg": float(acc)}


def compute_cls_metrics(
    preds: list[int],
    gts: list[int],
) -> dict[str, float]:
    """Compute accuracy and F1 for binary classification."""
    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds, average="binary", zero_division=0)
    return {"acc_cls": float(acc), "f1_cls": float(f1)}


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: MultiTaskVMUNet,
    loader_seg,
    loader_cls,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    config: MultiTaskConfig,
    logger: logging.Logger,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    """
    Interleaved training: one batch from Dataset A then one from Dataset B.
    Uses two separate backward() calls per step to keep task gradients clean.
    """
    model.train()

    seg_losses, cls_losses = [], []
    amp_enabled = config.amp and device.type == "cuda"

    # Cycle shorter loader so both are fully consumed
    iter_seg = iter(loader_seg)
    iter_cls = iter(loader_cls)
    n_steps = max(len(loader_seg), len(loader_cls))

    pbar = tqdm(range(n_steps), desc=f"Epoch {epoch:03d} [train]", leave=False)

    for step in pbar:
        batch_seg = next(iter_seg, None)
        batch_cls = next(iter_cls, None)

        # If one runs out, cycle it
        if batch_seg is None:
            iter_seg = iter(loader_seg)
            batch_seg = next(iter_seg)
        if batch_cls is None:
            iter_cls = iter(loader_cls)
            batch_cls = next(iter_cls)

        # ── Dataset A: Segmentation ──────────────────────────────────────
        imgs_a  = batch_seg["image"].to(device, non_blocking=True).float()
        masks_a = batch_seg["mask"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            mask_pred_a, cls_pred_a = model(imgs_a)
            loss_a, info_a = criterion(
                mask_pred=mask_pred_a,
                class_logits=cls_pred_a,
                mask_gt=masks_a,
                class_gt=None,
            )

        scaler.scale(loss_a).backward()
        scaler.step(optimizer)
        scaler.update()

        seg_losses.append(info_a["loss_seg"])
        del imgs_a, masks_a, mask_pred_a, cls_pred_a, loss_a

        # ── Dataset B: Classification ────────────────────────────────────
        imgs_b  = batch_cls["image"].to(device, non_blocking=True).float()
        lbls_b  = batch_cls["class_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            mask_pred_b, cls_pred_b = model(imgs_b)
            loss_b, info_b = criterion(
                mask_pred=mask_pred_b,
                class_logits=cls_pred_b,
                mask_gt=None,
                class_gt=lbls_b,
            )

        scaler.scale(loss_b).backward()
        scaler.step(optimizer)
        scaler.update()

        cls_losses.append(info_b["loss_cls"])
        del imgs_b, lbls_b, mask_pred_b, cls_pred_b, loss_b

        # ── Logging ───────────────────────────────────────────────────────
        global_step += 1
        writer.add_scalar("train/loss_seg", info_a["loss_seg"], global_step)
        writer.add_scalar("train/loss_cls", info_b["loss_cls"], global_step)

        pbar.set_postfix({
            "seg": f"{info_a['loss_seg']:.4f}",
            "cls": f"{info_b['loss_cls']:.4f}",
        })

    mean_seg = float(np.mean(seg_losses)) if seg_losses else 0.0
    mean_cls = float(np.mean(cls_losses)) if cls_losses else 0.0

    log = f"[Epoch {epoch:03d}][Train] loss_seg={mean_seg:.4f}  loss_cls={mean_cls:.4f}"
    logger.info(log)

    return {"loss_seg": mean_seg, "loss_cls": mean_cls}, global_step


@torch.no_grad()
def validate(
    model: MultiTaskVMUNet,
    loader_seg,
    loader_cls,
    criterion: MultiTaskLoss,
    epoch: int,
    config: MultiTaskConfig,
    logger: logging.Logger,
    writer: SummaryWriter,
    device: torch.device,
) -> dict[str, float]:
    """Validate on both tasks, return dict of metrics."""
    model.eval()
    amp_enabled = config.amp and device.type == "cuda"

    # ── Segmentation validation ──────────────────────────────────────────
    seg_preds_all, seg_gts_all, seg_losses = [], [], []

    for batch in tqdm(loader_seg, desc=f"Epoch {epoch:03d} [val-seg] ", leave=False):
        imgs  = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).float()

        with autocast("cuda", enabled=amp_enabled):
            mask_pred, cls_pred = model(imgs)
            loss, _ = criterion(mask_pred=mask_pred, class_logits=cls_pred, mask_gt=masks)

        seg_losses.append(loss.item())
        seg_preds_all.append(mask_pred.cpu().numpy().reshape(-1))
        seg_gts_all.append(masks.cpu().numpy().reshape(-1))

    seg_preds_np = np.concatenate(seg_preds_all)
    seg_gts_np   = np.concatenate(seg_gts_all)
    seg_metrics  = compute_seg_metrics(seg_preds_np, seg_gts_np, config.seg_threshold)
    seg_metrics["loss_seg"] = float(np.mean(seg_losses))

    # ── Classification validation ─────────────────────────────────────────
    cls_preds_all, cls_gts_all, cls_losses = [], [], []

    for batch in tqdm(loader_cls, desc=f"Epoch {epoch:03d} [val-cls] ", leave=False):
        imgs = batch["image"].to(device, non_blocking=True).float()
        lbls = batch["class_label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            mask_pred, cls_pred = model(imgs)
            loss, _ = criterion(mask_pred=mask_pred, class_logits=cls_pred, class_gt=lbls)

        cls_losses.append(loss.item())
        cls_preds_all.extend(cls_pred.argmax(dim=1).cpu().tolist())
        cls_gts_all.extend(lbls.cpu().tolist())

    cls_metrics = compute_cls_metrics(cls_preds_all, cls_gts_all)
    cls_metrics["loss_cls"] = float(np.mean(cls_losses))

    # ── Log & tensorboard ────────────────────────────────────────────────
    all_metrics = {**seg_metrics, **cls_metrics}

    log = (
        f"[Epoch {epoch:03d}][Val] "
        f"loss_seg={seg_metrics['loss_seg']:.4f}  "
        f"dice={seg_metrics['dice']:.4f}  "
        f"miou={seg_metrics['miou']:.4f} | "
        f"loss_cls={cls_metrics['loss_cls']:.4f}  "
        f"acc={cls_metrics['acc_cls']:.4f}  "
        f"f1={cls_metrics['f1_cls']:.4f}"
    )
    logger.info(log)
    print(log)

    for k, v in all_metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)

    return all_metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: MultiTaskConfig) -> None:
    # ── Setup ────────────────────────────────────────────────────────────
    config.create_dirs()
    logger = get_logger("train", config.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(config.work_dir, "tb"))

    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        gpu_idx = int(config.gpu_id) if str(config.gpu_id).isdigit() else 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.seed)
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)} (device {device})")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available or disabled in PyTorch build. Falling back to CPU.")

    logger.info("=" * 60)
    logger.info("Multi-Task VM-UNet Training")
    logger.info(repr(config))
    logger.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info("Building dataloaders …")
    seg_train, seg_val, seg_test = build_seg_loaders(
        config.silkynet_data_dir,
        img_size=config.input_size,
        batch_size=config.batch_size_seg,
        num_workers=config.num_workers,
    )
    cls_train, cls_val, cls_test = build_cls_loaders(
        config.yolo_data_dir,
        img_size=config.input_size,
        batch_size=config.batch_size_cls,
        num_workers=config.num_workers,
        max_train_samples=config.max_cls_samples,
    )

    logger.info(f"Dataset A (seg):  train={len(seg_train.dataset)}  val={len(seg_val.dataset)}  test={len(seg_test.dataset)}")
    logger.info(f"Dataset B (cls):  train={len(cls_train.dataset)}  val={len(cls_val.dataset)}  test={len(cls_test.dataset)}")
    logger.info(f"Class distribution (train): {cls_train.dataset.class_counts()}")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = MultiTaskVMUNet(
        input_channels=3,
        num_seg_classes=config.num_seg_classes,
        num_cls_classes=config.num_cls_classes,
        depths=config.depths,
        depths_decoder=config.depths_decoder,
        drop_path_rate=config.drop_path_rate,
        load_ckpt_path=config.pretrained_path if os.path.exists(config.pretrained_path) else None,
        cls_hidden=config.cls_hidden,
        cls_dropout=config.cls_dropout,
    ).to(device)

    if os.path.exists(config.pretrained_path):
        model.load_pretrained()
        logger.info(f"Pretrained weights loaded from: {config.pretrained_path}")
    else:
        logger.warning(f"Pretrained weights NOT found at: {config.pretrained_path} — training from scratch.")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    criterion = MultiTaskLoss(
        lambda_seg=config.lambda_seg,
        lambda_cls=config.lambda_cls,
        w_bce=config.w_bce,
        w_dice=config.w_dice,
        label_smoothing=config.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    amp_enabled = config.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # ── Resume checkpoint ─────────────────────────────────────────────────
    latest_ckpt = os.path.join(config.checkpoint_dir, "latest.pth")
    start_epoch = 1
    best_dice   = 0.0
    best_f1     = 0.0
    global_step = 0

    if os.path.exists(latest_ckpt):
        logger.info(f"Resuming from {latest_ckpt} …")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        best_f1     = ckpt.get("best_f1",   0.0)
        global_step = ckpt.get("global_step", 0)
        logger.info(f"  Resumed from epoch {ckpt['epoch']}, best_dice={best_dice:.4f}, best_f1={best_f1:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info(f"Starting training: epochs {start_epoch} → {config.epochs}")

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()

        train_metrics, global_step = train_one_epoch(
            model=model,
            loader_seg=seg_train,
            loader_cls=cls_train,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            config=config,
            logger=logger,
            writer=writer,
            global_step=global_step,
            device=device,
        )

        scheduler.step()

        # Validation
        if epoch % config.val_interval == 0:
            val_metrics = validate(
                model=model,
                loader_seg=seg_val,
                loader_cls=cls_val,
                criterion=criterion,
                epoch=epoch,
                config=config,
                logger=logger,
                writer=writer,
                device=device,
            )

            # Save best segmentation checkpoint
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best_seg.pth"))
                logger.info(f"  ★ New best seg (dice={best_dice:.4f}) → saved best_seg.pth")

            # Save best classification checkpoint
            if val_metrics["f1_cls"] > best_f1:
                best_f1 = val_metrics["f1_cls"]
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best_cls.pth"))
                logger.info(f"  ★ New best cls (f1={best_f1:.4f}) → saved best_cls.pth")

        # Periodic checkpoint
        if epoch % config.save_interval == 0 or epoch == config.epochs:
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_dice":   best_dice,
                "best_f1":     best_f1,
                "global_step": global_step,
            }, latest_ckpt)

        elapsed = time.time() - t0
        logger.info(f"  Epoch {epoch:03d} done in {elapsed:.1f}s  (lr={scheduler.get_last_lr()[0]:.2e})")

    # ── Final test evaluation ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Final test evaluation (best_seg model) …")
    best_seg_path = os.path.join(config.checkpoint_dir, "best_seg.pth")
    if os.path.exists(best_seg_path):
        model.load_state_dict(torch.load(best_seg_path, map_location=device))
        test_metrics = validate(
            model=model,
            loader_seg=seg_test,
            loader_cls=cls_test,
            criterion=criterion,
            epoch=0,
            config=config,
            logger=logger,
            writer=writer,
            device=device,
        )
        logger.info(f"[TEST] {test_metrics}")
    else:
        logger.warning("No best_seg.pth found — skipping test evaluation.")

    writer.close()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Task VM-UNet Training")
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--batch-size-seg",   type=int,   default=2)
    p.add_argument("--batch-size-cls",   type=int,   default=2)
    p.add_argument("--max-cls-samples",  type=int,   default=500, help="Subsample Dataset B train set to N images (0=all)")
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lambda-seg",       type=float, default=1.0)
    p.add_argument("--lambda-cls",       type=float, default=1.0)
    p.add_argument("--input-size",       type=int,   default=128)
    p.add_argument("--gpu",              type=str,   default="0")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-amp",           action="store_true")
    p.add_argument("--silkynet-dir",     type=str,   default="models/silkynet/data")
    p.add_argument("--yolo-dir",         type=str,   default="data/Silkworm Diseases.v1i.yolo26")
    p.add_argument("--pretrained",       type=str,   default="models/vmunet/pre_trained_weights/vmamba_small_e238_ema.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = MultiTaskConfig(
        epochs=args.epochs,
        batch_size_seg=args.batch_size_seg,
        batch_size_cls=args.batch_size_cls,
        max_cls_samples=args.max_cls_samples if args.max_cls_samples > 0 else None,
        lr=args.lr,
        lambda_seg=args.lambda_seg,
        lambda_cls=args.lambda_cls,
        input_size=args.input_size,
        gpu_id=args.gpu,
        seed=args.seed,
        amp=not args.no_amp,
        silkynet_data_dir=args.silkynet_dir,
        yolo_data_dir=args.yolo_dir,
        pretrained_path=args.pretrained,
    )
    train(cfg)
