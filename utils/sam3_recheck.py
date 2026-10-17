#!/usr/bin/env python3
"""
Quét kỹ bằng SAM 3 (chỉ text prompt) để tìm con tằm có trong ảnh mà nhãn không có.

Để không bỏ sót: nhiều prompt, ngưỡng thấp, ảnh gốc + ảnh lật ngang + 4 mảnh cắt phóng to (bắt con nhỏ / ở mép),
rồi gộp bằng NMS. Đối chiếu với masks_inst / masks_ignore:
  - candidate = phát hiện không nằm trong con có nhãn nào và không nằm trong vùng bỏ qua (nghi nhãn bị thiếu)
  - gt_unseen = con có nhãn mà SAM 3 không phát hiện được (nghi nhãn sai / SAM 3 bỏ sót)

  /home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_recheck.py --split test_real --gpu-id 0 \
      --output-dir Reports/Meeting_2026-10-17/results/sam3_check/sam3_recheck/test_real
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "sam3"))

PROMPTS = ["silkworm", "silkworm larva", "caterpillar", "worm", "larva"]
SW = (0, 1)  # chỉ hai prompt chứa "silkworm" mới được tính là bằng chứng có tằm (prompt chung chung bắt cả phân, mảnh lá)
RED, GREEN, GRAY = (255, 40, 40), (60, 200, 60), (150, 150, 150)


def detect(processor, state, hw, thr):
    """Mọi prompt trên một ảnh → (masks bool [N,H,W], scores [N])."""
    ms, ss, ps = [], [], []
    processor.confidence_threshold = thr
    for k, p in enumerate(PROMPTS):
        out = processor.set_text_prompt(p, state)
        if len(out["masks"]):
            ms.append(out["masks"].float().cpu().numpy()[:, 0] > 0.5)
            ss.append(out["scores"].float().cpu().numpy())
            ps.append(np.full(len(ss[-1]), k))
    if not ms:
        return np.zeros((0,) + hw, bool), np.zeros(0), np.zeros(0, int)
    return np.concatenate(ms), np.concatenate(ss), np.concatenate(ps)


def nms(masks, scores, prompts, iou_thr=0.5, contain_thr=0.8):
    """Gộp phát hiện trùng. Mỗi mask giữ lại kèm: số phiếu từ prompt chứa 'silkworm' và điểm cao nhất của các prompt đó."""
    order = np.argsort(-scores)
    keep, group = [], []
    for i in order:
        a = masks[i].sum()
        host = None
        for n, j in enumerate(keep):
            inter = (masks[i] & masks[j]).sum()
            if inter / max(a + masks[j].sum() - inter, 1) > iou_thr or inter / max(min(a, masks[j].sum()), 1) > contain_thr:
                host = n
                break
        if host is None:
            keep.append(i); group.append([i])
        else:
            group[host].append(i)
    sw_votes = np.array([sum(prompts[j] in SW for j in g) for g in group])
    sw_max = np.array([max([scores[j] for j in g if prompts[j] in SW], default=0.0) for g in group])
    return masks[keep], scores[keep], sw_votes, sw_max


def scan(processor, rgb, thr, min_area):
    h, w = rgb.shape[:2]
    all_m, all_s, all_p = [], [], []

    def run(img, place):
        state = processor.set_image(Image.fromarray(img))
        m, s, pr = detect(processor, state, img.shape[:2], thr)
        for mi, si, pi in zip(m, s, pr):
            full = place(mi)
            if full is not None:
                all_m.append(full); all_s.append(si); all_p.append(pi)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        run(rgb, lambda m: m)                                             # ảnh gốc
        run(rgb[:, ::-1].copy(), lambda m: m[:, ::-1])                    # lật ngang
        t = int(h * 0.625)                                                # 4 mảnh phóng to, chồng nhau 25%
        for y0 in (0, h - t):
            for x0 in (0, w - t):
                def place(m, y0=y0, x0=x0):
                    ys, xs = np.nonzero(m)
                    if len(ys) == 0:
                        return None
                    # bỏ phát hiện chạm biên mảnh cắt (không phải biên ảnh): con bị cắt ngang, đã có ở ảnh gốc
                    if (y0 > 0 and ys.min() == 0) or (y0 == 0 and ys.max() == t - 1 and h > t) \
                            or (x0 > 0 and xs.min() == 0) or (x0 == 0 and xs.max() == t - 1 and w > t):
                        return None
                    full = np.zeros((h, w), bool)
                    full[y0:y0 + t, x0:x0 + t] = m
                    return full
                run(rgb[y0:y0 + t, x0:x0 + t].copy(), place)
    if not all_m:
        return np.zeros((0, h, w), bool), np.zeros(0), np.zeros(0, int), np.zeros(0)
    m, s, pr = np.stack(all_m), np.array(all_s), np.array(all_p)
    keep = m.reshape(len(m), -1).sum(1) >= min_area
    return nms(m[keep], s[keep], pr[keep])


def shape_features(m):
    """diện tích, tỉ lệ dài/rộng của hình chữ nhật bao nghiêng, có chạm biên ảnh không."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    ys, xs = np.nonzero(m)
    border = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1)
    return int(m.sum()), float(max(rw, rh) / max(min(rw, rh), 1)), border


def overlay(rgb, items):
    out = rgb.astype(np.float32).copy()
    for m, c in items:
        out[m] = 0.55 * out[m] + 0.45 * np.array(c, np.float32)
    out = out.astype(np.uint8)
    for m, c in items:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, c, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/sam3_aug20k_v3")
    ap.add_argument("--split", default="test_real")
    ap.add_argument("--stems", nargs="*", default=None, help="chỉ chạy các ảnh này (mặc định: cả tập)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--cand-score", type=float, default=0.5, help="điểm tối thiểu của ứng viên bị thiếu nhãn (mảnh vụn, phân thường < 0.4)")
    ap.add_argument("--cand-votes", type=int, default=0, help="số lần phát hiện tối thiểu từ prompt có 'silkworm' (0 = không bắt buộc)")
    ap.add_argument("--cand-area", type=int, default=1000, help="diện tích tối thiểu của ứng viên (px ở ảnh 640)")
    ap.add_argument("--cand-aspect", type=float, default=1.5, help="tằm thì dài: tỉ lệ dài/rộng tối thiểu (bỏ qua nếu chạm biên ảnh)")
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default="models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--bpe-path", default="models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = ap.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.cuda.set_device(args.gpu_id)
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint,
                                   enable_inst_interactivity=False, device="cuda", load_from_HF=False)
    processor = Sam3Processor(model, device="cuda")

    split = Path(args.data_root) / args.split
    stems = args.stems or sorted(p.stem for p in (split / "images").glob("*.jpg"))
    if args.limit:
        stems = stems[:args.limit]
    out = Path(args.output_dir)
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "dets").mkdir(exist_ok=True)

    report, n_cand_img, n_cand, n_unseen = [], 0, 0, 0
    for k, s in enumerate(stems):
        rgb = np.array(Image.open(split / "images" / f"{s}.jpg").convert("RGB"))
        inst = cv2.imread(str(split / "masks_inst" / f"{s}.png"), cv2.IMREAD_UNCHANGED)
        ign = cv2.imread(str(split / "masks_ignore" / f"{s}.png"), 0) > 127
        gt_union = inst > 0
        ids = [i for i in np.unique(inst) if i > 0]
        masks, scores, votes, swmax = scan(processor, rgb, args.thr, args.min_area)
        if len(masks):  # lưu thô để chỉnh ngưỡng sau mà không phải quét lại
            np.savez_compressed(out / "dets" / f"{s}.npz", masks=np.packbits(masks, axis=-1), scores=scores,
                                sw_votes=votes, sw_max=swmax, shape=np.array(masks.shape[1:]))

        cands = []
        for m, sc, nv in zip(masks, scores, votes):
            a, aspect, border = shape_features(m)
            if (sc >= args.cand_score and nv >= args.cand_votes and a >= args.cand_area
                    and (aspect >= args.cand_aspect or border)
                    and (m & gt_union).sum() / a < 0.3 and (m & ign).sum() / a < 0.5):
                cands.append((m, float(sc)))
        unseen = []
        for i in ids:
            g = inst == i
            best = max(((g & m).sum() / (g | m).sum() for m in masks), default=0.0)
            if best < 0.3:
                unseen.append(int(i))
        n_cand += len(cands); n_unseen += len(unseen); n_cand_img += bool(cands)
        report.append(dict(stem=s, n_gt=len(ids), n_sam3=len(masks), candidates=[round(c[1], 2) for c in cands],
                           gt_unseen_by_sam3=unseen))

        if cands or unseen:
            gt_items = [(inst == i, GREEN if i not in unseen else RED) for i in ids] + [(ign, GRAY)]
            sam_items = [(m, (60, 140, 255)) for m, sc in zip(masks, scores) if sc >= args.cand_score and not any((m == c[0]).all() for c in cands)]
            sam_items += [(c[0], RED) for c in cands]
            img = np.concatenate([rgb, overlay(rgb, gt_items), overlay(rgb, sam_items)], 1)
            cv2.putText(img, "anh goc", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "nhan that (xanh=co SAM3 thay, do=SAM3 khong thay, xam=ignore)", (rgb.shape[1] + 8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"SAM3 quet ky (do=nghi thieu nhan: {len(cands)})", (2 * rgb.shape[1] + 8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(out / "review" / f"{s[:60]}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[{k + 1}/{len(stems)}] {s[:40]}: gt={len(ids)} sam3={len(masks)} nghi_thieu_nhan={len(cands)} gt_sam3_khong_thay={len(unseen)}", flush=True)

    summ = dict(split=args.split, images=len(stems), images_with_candidates=n_cand_img, candidates=n_cand,
                gt_larvae_unseen_by_sam3=n_unseen, thr=args.thr, cand_score=args.cand_score, cand_votes=args.cand_votes,
                cand_area=args.cand_area, cand_aspect=args.cand_aspect, prompts=PROMPTS)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (out / "per_image.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
