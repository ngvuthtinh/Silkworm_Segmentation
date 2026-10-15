#!/usr/bin/env python3
"""
Quét kỹ bằng SAM 3 (cùng cách như sam3_recheck.py) và CHỈ lưu phát hiện thô, mỗi ảnh một file dets/<stem>.npz.
Chạy được lại nhiều lần: ảnh nào đã có npz thì bỏ qua (kể cả ảnh không phát hiện gì: file rỗng vẫn được ghi).
Chia việc cho nhiều GPU bằng --shard / --num-shards.

  /home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_scan_dets.py --split train --shard 0 --num-shards 2 --gpu-id 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_recheck import scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/sam3_seg_v2")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-root", default="data/sam3_scan")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--checkpoint", default="models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--bpe-path", default="models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = ap.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.cuda.set_device(args.gpu_id)
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint,
                                   enable_inst_interactivity=False, device="cuda", load_from_HF=False)
    processor = Sam3Processor(model, device="cuda")

    out = Path(args.out_root) / args.split / "dets"
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (Path(args.data_root) / args.split / "images").glob("*.jpg"))[args.shard::args.num_shards]
    todo = [s for s in stems if not (out / f"{s}.npz").exists()]
    print(f"[shard {args.shard}/{args.num_shards}] {len(stems)} ảnh, còn {len(todo)} chưa quét", flush=True)
    for k, s in enumerate(todo):
        rgb = np.array(Image.open(Path(args.data_root) / args.split / "images" / f"{s}.jpg").convert("RGB"))
        masks, scores, votes, swmax = scan(processor, rgb, args.thr, args.min_area)
        tmp = out / f"{s}.tmp.npz"                      # ghi tạm rồi đổi tên: bị ngắt giữa chừng không để lại file hỏng
        np.savez_compressed(tmp, masks=np.packbits(masks, axis=-1), scores=scores, sw_votes=votes, sw_max=swmax,
                            shape=np.array(rgb.shape[:2]))
        tmp.rename(out / f"{s}.npz")
        print(f"[{k + 1}/{len(todo)}] {s[:50]}: {len(masks)} phát hiện", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
