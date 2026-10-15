#!/usr/bin/env python3
"""
Kiểm tra SAM 3 (chỉ dùng text prompt) trên một tập có nhãn từng con:
  1. Tách được bao nhiêu con (prompt "silkworm") so với nhãn thật (IoU >= 0.5)?
  2. Có tự phân biệt con bệnh / khỏe không (prompt "diseased silkworm" vs "healthy silkworm")?

  /home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_check_on_data.py --split test_real --num 100 --gpu-id 0 \
      --output-dir Reports/2026-10-13/sam3_check/test_real
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "sam3"))

PROMPT_ALL = "silkworm"
PROMPT_CLS = {0: "diseased silkworm", 1: "healthy silkworm"}  # 0 = Grasserie, 1 = Healthy
COLOR = {0: (255, 90, 30), 1: (60, 200, 60), -1: (200, 200, 200)}  # RGB
NAME = {0: "Benh", 1: "Khoe", -1: "?"}


def detect(processor, state, prompt: str, thr: float) -> tuple[np.ndarray, np.ndarray]:
    processor.confidence_threshold = thr
    out = processor.set_text_prompt(prompt, state)
    if len(out["masks"]) == 0:
        return np.zeros((0,) + state["original_hw"], bool), np.zeros(0)
    m = (out["masks"].float().cpu().numpy()[:, 0] > 0.5)
    return m, out["scores"].float().cpu().numpy()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    af, bf = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    inter = af @ bf.T
    union = af.sum(1)[:, None] + bf.sum(1)[None] - inter
    return inter / np.maximum(union, 1)


def draw(rgb: np.ndarray, masks: np.ndarray, labels: list[int], texts: list[str]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for m, c in zip(masks, labels):
        out[m] = 0.55 * out[m] + 0.45 * np.array(COLOR[c], np.float32)
    out = out.astype(np.uint8)
    for m, c, t in zip(masks, labels, texts):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, COLOR[c], 2)
        ys, xs = np.nonzero(m)
        cv2.putText(out, t, (int(xs.mean()) - 30, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/sam3_aug20k_v3")
    ap.add_argument("--split", default="test_real")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--num-vis", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--min-area", type=int, default=625)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stems-from", default=None, help="thư mục ảnh kết quả VM-UNet (NN_<stem>.png): chạy đúng các ảnh đó")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default="models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--bpe-path", default="models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = ap.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint,
                                   enable_inst_interactivity=False, device="cuda", load_from_HF=False)
    processor = Sam3Processor(model, device="cuda")

    split = Path(args.data_root) / args.split
    stems = sorted(p.stem for p in (split / "images").glob("*.jpg"))
    if args.stems_from:  # đúng các ảnh mà inference.py của VM-UNet đã vẽ
        want = [p.stem.split("_", 1)[1] for p in sorted(Path(args.stems_from).glob("*.png"))]
        stems = [s for s in want if s in set(stems)]
    else:
        random.Random(args.seed).shuffle(stems)
        stems = stems[:args.num]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gt = n_found = n_pred = n_fp = 0
    cls_ok = cls_tot = 0
    conf = np.zeros((2, 3), int)  # hàng: lớp thật (Gr, He); cột: đoán Gr, He, không rõ
    same_prompt_imgs = 0
    rows = []
    for k, s in enumerate(stems):
        rgb = np.array(Image.open(split / "images" / f"{s}.jpg").convert("RGB"))
        inst = cv2.imread(str(split / "masks_inst" / f"{s}.png"), cv2.IMREAD_UNCHANGED)
        cm = cv2.imread(str(split / "masks_cls" / f"{s}.png"), 0)
        ign = cv2.imread(str(split / "masks_ignore" / f"{s}.png"), 0) > 127
        ids = [i for i in np.unique(inst) if i > 0]
        gt = np.stack([inst == i for i in ids]) if ids else np.zeros((0,) + inst.shape, bool)
        gt_cls = [int(cm[inst == i][0]) - 1 for i in ids]  # 0 = Grasserie, 1 = Healthy

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(rgb))
            state["original_hw"] = rgb.shape[:2]
            pm, ps = detect(processor, state, PROMPT_ALL, args.thr)
            dm, ds = detect(processor, state, PROMPT_CLS[0], args.thr)
            hm, hs = detect(processor, state, PROMPT_CLS[1], args.thr)
        keep = [i for i in range(len(pm)) if pm[i].sum() >= args.min_area]
        pm, ps = pm[keep], ps[keep]

        # phân loại từng mask SAM 3: lấy prompt nào khớp (IoU >= 0.5) với điểm cao hơn
        pred_cls = []
        for m in pm:
            sc = []
            for dets, sco in ((dm, ds), (hm, hs)):
                io = iou_matrix(m[None], dets)[0] if len(dets) else np.zeros(0)
                sc.append(float(sco[io.argmax()]) if len(io) and io.max() >= 0.5 else 0.0)
            pred_cls.append(-1 if max(sc) == 0 else int(np.argmax(sc)))
        pred_cls = np.array(pred_cls, int)
        if len(dm) == len(hm) and len(dm) and np.allclose(ds, hs, atol=1e-3):
            same_prompt_imgs += 1

        io = iou_matrix(gt, pm)
        matched_gt, used = {}, set()
        for _ in range(min(io.shape)):
            if io.size == 0 or io.max() < 0.5:
                break
            g, p = np.unravel_index(io.argmax(), io.shape)
            matched_gt[g] = p; used.add(p)
            io[g, :] = 0; io[:, p] = 0
        n_gt += len(ids); n_found += len(matched_gt); n_pred += len(pm)
        for p in range(len(pm)):  # dự đoán thừa: không khớp con thật nào và không nằm trong vùng bỏ qua
            if p not in used and (pm[p] & ign).sum() / max(pm[p].sum(), 1) < 0.5:
                n_fp += 1
        for g, p in matched_gt.items():
            conf[gt_cls[g], pred_cls[p] if pred_cls[p] >= 0 else 2] += 1
            if pred_cls[p] >= 0:
                cls_tot += 1; cls_ok += int(pred_cls[p] == gt_cls[g])
        rows.append(dict(stem=s, gt=len(ids), pred=len(pm), found=len(matched_gt)))

        if k < args.num_vis:
            gt_img = draw(rgb, gt, gt_cls, [NAME[c] for c in gt_cls]) if len(gt) else rgb
            sam_img = draw(rgb, pm, list(pred_cls), [f"{NAME[c]} {x:.2f}" for c, x in zip(pred_cls, ps)]) if len(pm) else rgb
            cv2.imwrite(str(out_dir / f"{k:02d}_{s[:40]}.jpg"),
                        cv2.cvtColor(np.concatenate([rgb, gt_img, sam_img], 1), cv2.COLOR_RGB2BGR))
        print(f"[{k + 1}/{len(stems)}] {s[:40]}: gt={len(ids)} sam3={len(pm)} khop={len(matched_gt)}", flush=True)

    rec = n_found / max(n_gt, 1)
    prec = n_found / max(n_pred, 1)
    summ = dict(split=args.split, images=len(stems), gt_larvae=n_gt, sam3_masks=n_pred, found=n_found,
                recall=rec, precision=prec, f1=2 * rec * prec / max(rec + prec, 1e-9), extra_masks=n_fp,
                cls_acc_when_decided=cls_ok / max(cls_tot, 1), decided=cls_tot,
                undecided=int(conf[:, 2].sum()), images_with_identical_prompt_output=same_prompt_imgs,
                confusion_rows_true_Gr_He_cols_predGr_He_undecided=conf.tolist())
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
