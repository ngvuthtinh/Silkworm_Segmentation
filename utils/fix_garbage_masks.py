#!/usr/bin/env python3
"""
Sửa mask rác ở một số ảnh cụ thể của data per-larva bằng SAM 3 (chỉ những ảnh được chỉ định).

  /home/subnh5/miniconda3/envs/sam3/bin/python utils/fix_garbage_masks.py --split train --stems-json report.json \
      --backup-dir data/legacy/sam3_aug20k_v3_before_fix --panels <thư mục ảnh trước/sau>

Với mỗi ảnh: tìm con có mask rác (viền răng cưa: chu vi/sqrt(diện tích) > 14), cho SAM 3 quét kỹ ảnh đó
(sam3_recheck.scan), lấy phát hiện tốt nhất (điểm >= 0.5, nằm >= 50% trong mask rác) thay cho mask rác.
Pixel còn lại của mask rác trở thành nền. Không tìm được phát hiện phù hợp thì KHÔNG đụng vào ảnh và báo lại.
Trước khi ghi đè, sao lưu masks, masks_inst, masks_cls, masks_ignore của các ảnh đó vào --backup-dir.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_recheck import scan  # noqa: E402

DIRS = ["masks", "masks_inst", "masks_cls", "masks_ignore"]
MIN_AREA = 625


def jagged(piece: np.ndarray) -> float:
    cs, _ = cv2.findContours(piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(c, True) for c in cs) / np.sqrt(max(int(piece.sum()), 1))


def garbage_ids(inst: np.ndarray) -> list[int]:
    out = []
    for i in [int(i) for i in np.unique(inst) if i > 0]:
        n, lab = cv2.connectedComponents((inst == i).astype(np.uint8))
        if any((lab == j).sum() >= MIN_AREA and jagged(lab == j) > 14 for j in range(1, n)):
            out.append(i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sam3_aug20k_v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stems-json", required=True, help="report.json của check_dataset.py (lấy mục speckled của split)")
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--panels", default=None)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--checkpoint", default="models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--bpe-path", default="models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    args = ap.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    d = Path(args.root) / args.split
    stems = json.loads(Path(args.stems_json).read_text())[args.split]["flagged"]["speckled"]
    torch.cuda.set_device(args.gpu_id)
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint,
                                   enable_inst_interactivity=False, device="cuda", load_from_HF=False)
    proc = Sam3Processor(model, device="cuda")
    backup = Path(args.backup_dir) / args.split
    results = []
    for s in stems:
        rgb = np.array(Image.open(d / "images" / f"{s}.jpg").convert("RGB"))
        inst = cv2.imread(str(d / "masks_inst" / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cm = cv2.imread(str(d / "masks_cls" / f"{s}.png"), 0)
        ign = cv2.imread(str(d / "masks_ignore" / f"{s}.png"), 0) > 127
        bad = garbage_ids(inst)
        masks, scores, _, _ = scan(proc, rgb, 0.1, 300)
        new_inst, new_cm, used, log = inst.copy(), cm.copy(), set(), []
        for g in bad:
            gm = inst == g
            cls_val = int(np.bincount(cm[gm]).argmax())
            best, best_score = None, 0.0
            for k, (m, sc) in enumerate(zip(masks, scores)):
                if k in used or sc < args.min_score or m.sum() < 1000:
                    continue
                if (m & gm).sum() / m.sum() >= 0.5 and sc > best_score:
                    best, best_score = k, float(sc)
            if best is None:
                log.append({"id": g, "result": "khong_tim_duoc_phat_hien", "garbage_area": int(gm.sum())})
                continue
            used.add(best)
            others = (inst > 0) & ~gm                       # con khác trong ảnh giữ nguyên chỗ của chúng
            new = masks[best] & ~others
            new_inst[gm] = 0
            new_cm[gm] = 0
            new_inst[new] = g
            new_cm[new] = cls_val
            log.append({"id": g, "result": "da_thay", "score": round(best_score, 2),
                        "garbage_area": int(gm.sum()), "new_area": int(new.sum())})
        done = [x for x in log if x["result"] == "da_thay"]
        if done:
            for sub in DIRS:                                # sao lưu trước khi ghi đè
                (backup / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy(d / sub / f"{s}.png", backup / sub / f"{s}.png")
            cv2.imwrite(str(d / "masks_inst" / f"{s}.png"), new_inst.astype(np.uint16))
            cv2.imwrite(str(d / "masks_cls" / f"{s}.png"), new_cm.astype(np.uint8))
            cv2.imwrite(str(d / "masks" / f"{s}.png"), np.where(new_inst > 0, 255, 0).astype(np.uint8))
            cv2.imwrite(str(d / "masks_ignore" / f"{s}.png"), np.where(ign & (new_inst == 0), 255, 0).astype(np.uint8))
            if args.panels:
                def draw(i: np.ndarray, title: str) -> np.ndarray:
                    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                    for v in [int(v) for v in np.unique(i) if v > 0]:
                        m = (i == v).astype(np.uint8)
                        out[m > 0] = (0.5 * out[m > 0] + 0.5 * np.array((40, 40, 255))).astype(np.uint8)
                        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(out, cs, -1, (0, 0, 255), 2)
                    cv2.putText(out, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    return out
                Path(args.panels).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.panels) / f"{s[:50]}.jpg"),
                            np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), draw(inst, "truoc khi sua"),
                                            draw(new_inst, "sau khi sua (SAM 3)")], 1))
        results.append({"stem": s, "garbage_ids": bad, "log": log})
        print(s[:50], [(x["id"], x["result"], x.get("score")) for x in log], flush=True)
    Path(args.backup_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.backup_dir) / f"fix_log_{args.split}.json").write_text(json.dumps(results, indent=1))
    print("Đã thay:", sum(1 for r in results for x in r["log"] if x["result"] == "da_thay"),
          "| không tìm được:", sum(1 for r in results for x in r["log"] if x["result"] != "da_thay"))


if __name__ == "__main__":
    main()
