#!/usr/bin/env python3
"""
experiments/multitask_vmunet_disc/train.py
VM-UNet multitask (thân + biên + bệnh) + embedding theo pixel học bằng discriminative loss.

    L = λ_body·L_body + λ_bnd·L_bnd + λ_dis·L_dis  +  λ_embed·(L_var + L_dist + 0.001·L_reg)

Phần trong ngoặc đầu y hệt baseline `multitask_vmunet`; chỉ THÊM kênh embedding và loss của nó.

Chạy thử (5 batch train + 3 batch val):
  python -m experiments.multitask_vmunet_disc.train --smoke-test --gpu 1
Chạy thật:
  python -m experiments.multitask_vmunet_disc.train --gpu 1 --out-dir Reports/Meeting_2026-10-30/checkpoints/multitask_vmunet_disc
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.multitask_vmunet_disc.config import DiscConfig
from experiments.multitask_vmunet_disc.evaluate import run_eval
from experiments.multitask_vmunet_disc.model import DiscVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.losses import DiscriminativeLoss, InstanceMultiTaskLoss

SMOKE_TRAIN_BATCHES, SMOKE_VAL_BATCHES = 5, 3
# các tham số của run cũ phải được giữ nguyên khi --resume (kích thước mô hình, δ, trọng số loss)
RESUME_KEYS = ("n_embed", "lambda_embed", "delta_var", "delta_dist", "merge_thr")


def _pick_device(gpu: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu.isdigit():
        idx = int(gpu)
    else:
        idx = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def train(args: argparse.Namespace) -> None:
    cfg = DiscConfig()
    if args.smoke_test:
        cfg.work_dir = "runs/multitask_vmunet_disc/smoke_test"
        cfg.epochs = 1
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.data_root:
        cfg.data_root = args.data_root
    if args.lambda_embed is not None:
        cfg.lambda_embed = args.lambda_embed
    if args.out_dir and not args.smoke_test:
        cfg.work_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # --resume: tiếp tục đúng thư mục run cũ (nhận thư mục run hoặc đường dẫn last.pth)
    resume_ck = None
    if args.resume:
        rp = Path(args.resume)
        if rp.is_dir():
            rp = rp / "checkpoints" / "last.pth"
        if not rp.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {rp}")
        cfg.work_dir = str(rp.parent.parent)
        old_cfg = Path(cfg.work_dir) / "config.yaml"
        if old_cfg.exists() and not args.smoke_test:
            saved = yaml.safe_load(old_cfg.read_text())
            if not args.epochs:
                cfg.epochs = saved.get("epochs", cfg.epochs)
            if not args.batch_size:
                cfg.batch_size = saved.get("batch_size", cfg.batch_size)
            for k in RESUME_KEYS:
                if k in saved and not (k == "lambda_embed" and args.lambda_embed is not None):
                    setattr(cfg, k, saved[k])
        resume_ck = torch.load(rp, map_location="cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _pick_device(args.gpu)

    ds_train = InstanceSilkwormDataset(cfg.data_root, "train", cfg.img_size, True, cfg.boundary_width)
    ds_val = InstanceSilkwormDataset(cfg.data_root, "valid", cfg.img_size, False, cfg.boundary_width)
    dl_train = DataLoader(ds_train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = DiscVMUNet(cfg.n_embed, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate)
    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
    else:
        model.load_pretrained(cfg.pretrained_path)
    model.to(device)

    base_loss_fn = InstanceMultiTaskLoss(cfg.lambda_body, cfg.lambda_boundary, cfg.lambda_disease)
    emb_loss_fn = DiscriminativeLoss(cfg.delta_var, cfg.delta_dist)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if resume_ck is None:
        with open(os.path.join(cfg.work_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
    log_path = os.path.join(cfg.log_dir, "train_log.csv")

    print(f"\n🐛 Instance VM-UNet + embedding({cfg.n_embed}) | device {device} | train {len(ds_train)} / val {len(ds_val)} ảnh "
          f"| {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | out: {cfg.work_dir}\n")

    start_epoch, best = 1, -1.0
    if resume_ck is not None:
        start_epoch = resume_ck["epoch"] + 1
        if "optimizer" in resume_ck:                       # checkpoint đầy đủ → khôi phục y nguyên
            optimizer.load_state_dict(resume_ck["optimizer"])
            scheduler.load_state_dict(resume_ck["scheduler"])
            scaler.load_state_dict(resume_ck["scaler"])
            best = resume_ck.get("best", -1.0)
        print(f"♻️  Resume từ epoch {resume_ck['epoch']} → tiếp tục epoch {start_epoch}/{cfg.epochs} | best E2E F1 = {best:.4f}")
        if start_epoch > cfg.epochs:
            print("Đã train đủ số epoch, không còn gì để chạy.")
            return

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        n_batches = min(SMOKE_TRAIN_BATCHES, len(dl_train)) if args.smoke_test else len(dl_train)
        sums = np.zeros(7)
        pbar = tqdm(enumerate(dl_train), total=n_batches, desc=f"Epoch {epoch:02d}/{cfg.epochs} [TRAIN]", ncols=140)
        for i, (imgs, body, bnd, dis, inst, valid) in pbar:
            if i >= n_batches:
                break
            imgs, body, bnd, dis, inst, valid = (
                t.to(device, non_blocking=True) for t in (imgs, body, bnd, dis, inst, valid))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, emb = model.split(model(imgs))
            loss_base, l_body, l_bnd, l_dis = base_loss_fn(logits, body, bnd, dis, valid)
            l_emb, l_var, l_dist, _l_reg = emb_loss_fn(emb, inst, valid)
            loss = loss_base + cfg.lambda_embed * l_emb
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums += [loss.item(), l_body.item(), l_bnd.item(), l_dis.item(), l_emb.item(), l_var.item(), l_dist.item()]
            overall = ((epoch - 1) + (i + 1) / n_batches) / cfg.epochs * 100
            pbar.set_postfix(tổng=f"{overall:.1f}%", loss=f"{loss.item():.3f}", body=f"{l_body.item():.3f}",
                             bnd=f"{l_bnd.item():.3f}", dis=f"{l_dis.item():.3f}", emb=f"{l_emb.item():.3f}")
        scheduler.step()
        mean = sums / n_batches

        val = run_eval(model, dl_val, cfg, device, SMOKE_VAL_BATCHES if args.smoke_test else None)
        is_best = val["e2e_f1"] > best
        if is_best:
            best = val["e2e_f1"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "val": val},
                       os.path.join(cfg.checkpoint_dir, "best.pth"))
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best": best, "val": val},
                   os.path.join(cfg.checkpoint_dir, "last.pth"))

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "loss": mean[0], "loss_body": mean[1],
               "loss_boundary": mean[2], "loss_disease": mean[3], "loss_embed": mean[4],
               "loss_embed_var": mean[5], "loss_embed_dist": mean[6], **val}
        new_file = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)

        print(f"  Ep {epoch:02d} | loss {mean[0]:.3f} (emb {mean[4]:.3f}) | body Dice {val['body_dice']:.3f} | "
              f"tách con F1 {val['inst_f1']:.3f} (không gộp {val['inst_f1_nomerge']:.3f}) | "
              f"chẩn đoán đúng {val['cls_acc_matched']:.3f} | "
              f"E2E F1 {val['e2e_f1']:.3f} (không gộp {val['e2e_f1_nomerge']:.3f}){' ★' if is_best else ''}")

    print(f"\n✅ Xong. Best E2E F1 = {best:.4f} | {cfg.work_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Instance VM-UNet + embedding trainer")
    p.add_argument("--smoke-test", action="store_true", help="Chạy thử 5 batch train + 3 batch val")
    p.add_argument("--gpu", type=str, default="auto", help="Chỉ số GPU hoặc 'auto'")
    p.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch trong config")
    p.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size (giảm nếu thiếu VRAM)")
    p.add_argument("--data-root", type=str, default=None, help="Ghi đè thư mục data trong config (vd data/sam3_aug20k_v3); khi --resume phải truyền lại")
    p.add_argument("--lambda-embed", type=float, default=None, help="Ghi đè λ_embed (trọng số discriminative loss)")
    p.add_argument("--out-dir", type=str, default=None, help="Thư mục chứa các run (vd Reports/Meeting_2026-10-30/checkpoints/multitask_vmunet_disc); mặc định runs/multitask_vmunet_disc")
    p.add_argument("--resume", type=str, default=None,
                   help="Tiếp tục train: thư mục run hoặc đường dẫn last.pth")
    train(p.parse_args())


if __name__ == "__main__":
    main()
