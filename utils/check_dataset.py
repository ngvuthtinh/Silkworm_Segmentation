#!/usr/bin/env python3
"""
Kiểm tra lỗi mask của một bộ data per-larva (images, masks, masks_inst, masks_cls, masks_ignore).

  python utils/check_dataset.py                                   # data/sam3_aug20k_v3, mọi split
  python utils/check_dataset.py --root data/sam3_aug20k_v3 --splits test_real --panels <thư mục>

Hai nhóm kiểm tra:
  Cấu trúc (lỗi chắc chắn): thiếu file, kích thước lệch, masks != (masks_inst > 0), một con nhiều lớp,
    lớp ngoài con, con < min_area, nhãn đè lên vùng ignore, ảnh không có con, lớp ≠ tên file (ảnh thật).
  Hình dạng (nghi ngờ, cần xem ảnh), tính trên từng MẢNH liền >= min_area của mỗi con:
    lốm đốm (viền răng cưa: chu vi/sqrt(diện tích) > 14; KHÔNG dùng độ đặc vì con cong chữ C cũng có độ đặc thấp), hình hộp (gần bằng hình chữ nhật bao, kiểu lỗi
    box-prompt cũ), quá lớn (> 40% ảnh), thủng lỗ (> 15% diện tích; thường là lá dán đè giữa thân ở ảnh ghép, không phải lỗi), nhiều mảnh vụn (>= 5 mảnh < 100 px).
--panels: lưu ảnh 3 cột (ảnh | mọi con | con bị nghi, tô đỏ) cho tối đa --max-panels ảnh mỗi loại lỗi.
"""
from __future__ import annotations

import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

SPLITS = ["train", "valid", "test", "valid_real", "test_real"]
STRUCT = ["size_mismatch", "mask_vs_inst", "multi_class_inst", "class_outside_inst", "small_inst",
          "inst_in_ignore", "no_larva", "class_vs_filename"]
SHAPE = ["speckled", "boxy", "huge", "holes", "many_slivers"]


def check_image(args: tuple[Path, str, int]) -> dict:
    split_dir, stem, min_area = args
    img = cv2.imread(str(split_dir / "images" / f"{stem}.jpg"))
    mask = cv2.imread(str(split_dir / "masks" / f"{stem}.png"), 0)
    inst = cv2.imread(str(split_dir / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    cls = cv2.imread(str(split_dir / "masks_cls" / f"{stem}.png"), 0)
    ign_p = split_dir / "masks_ignore" / f"{stem}.png"
    ign = cv2.imread(str(ign_p), 0) > 127 if ign_p.exists() else np.zeros(inst.shape, bool)
    flags: dict[str, list[int]] = collections.defaultdict(list)   # loại lỗi → các id con (0 = cả ảnh)

    shapes = {a.shape[:2] for a in (img, mask, inst, cls, ign)}
    if len(shapes) > 1:
        flags["size_mismatch"].append(0)
        return {"stem": stem, "flags": dict(flags)}
    h, w = inst.shape
    if ((mask > 127) != (inst > 0)).sum() > 0:
        flags["mask_vs_inst"].append(0)
    if ((cls > 0) & (inst == 0)).sum() > 0:
        flags["class_outside_inst"].append(0)
    ids = [int(i) for i in np.unique(inst) if i > 0]
    if not ids:
        flags["no_larva"].append(0)
    real = not stem.startswith("aug")
    want = 1 if stem.lower().startswith("grasserie") else 2
    for i in ids:
        m = inst == i
        a = int(m.sum())
        c = set(np.unique(cls[m]).tolist())
        if len(c) != 1 or 0 in c:
            flags["multi_class_inst"].append(i)
        elif real and c != {want}:
            flags["class_vs_filename"].append(i)
        if a < min_area:
            flags["small_inst"].append(i)
        if (m & ign).any():
            flags["inst_in_ignore"].append(i)
        if a > 0.4 * h * w:
            flags["huge"].append(i)
        lab, n = ndi.label(m)
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else []
        if sum(1 for s in sizes if s < 100) >= 5:
            flags["many_slivers"].append(i)
        for j, s in enumerate(sizes, 1):
            if s < min_area:
                continue
            piece = (lab == j).astype(np.uint8)
            cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c0 = max(cs, key=cv2.contourArea)
            hull = cv2.contourArea(cv2.convexHull(c0))
            per = cv2.arcLength(c0, True)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c0)
            solidity, jagged, rect = s / max(hull, 1), per / np.sqrt(s), s / max(rw * rh, 1)
            if jagged > 14:                      # viền răng cưa: mask rác (con cong vẫn có viền trơn, < ~10)
                flags["speckled"].append(i)
            if rect > 0.97 and solidity > 0.97 and min(rw, rh) > 0.25 * max(rw, rh):
                flags["boxy"].append(i)
            filled = ndi.binary_fill_holes(piece).sum()
            if (filled - s) > 0.15 * s:
                flags["holes"].append(i)
    return {"stem": stem, "flags": {k: sorted(set(v)) for k, v in flags.items()}}


def panel(split_dir: Path, stem: str, bad: list[int], kind: str, out: Path) -> None:
    img = cv2.imread(str(split_dir / "images" / f"{stem}.jpg"))
    inst = cv2.imread(str(split_dir / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    ign = cv2.imread(str(split_dir / "masks_ignore" / f"{stem}.png"), 0)
    allc, badc = img.copy(), img.copy()
    rng = np.random.default_rng(0)
    for i in np.unique(inst):
        if i == 0:
            continue
        m = (inst == i).astype(np.uint8)
        col = tuple(int(x) for x in rng.integers(60, 255, 3))
        allc[m > 0] = (0.5 * allc[m > 0] + 0.5 * np.array(col)).astype(np.uint8)
        if i in bad or 0 in bad:
            badc[m > 0] = (0.4 * badc[m > 0] + 0.6 * np.array((40, 40, 255))).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(allc, cs, -1, col, 2)
    if ign is not None:
        g = ign > 127
        allc[g] = (0.5 * allc[g] + 0.5 * np.array((160, 160, 160))).astype(np.uint8)
    for im, t in ((img, "anh"), (allc, "moi con (xam = ignore)"), (badc, f"nghi loi: {kind}")):
        cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{kind}_{split_dir.name}_{stem[:50]}.jpg"), np.concatenate([img, allc, badc], 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sam3_aug20k_v3")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--min-area", type=int, default=625)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--panels", default=None, help="thư mục lưu ảnh các ca nghi ngờ")
    ap.add_argument("--max-panels", type=int, default=12)
    ap.add_argument("--json", default=None, help="lưu danh sách ảnh bị gắn cờ")
    args = ap.parse_args()

    report = {}
    for sp in args.splits:
        d = Path(args.root) / sp
        sets = {k: {p.stem for p in (d / k).glob("*.*")} for k in ["images", "masks", "masks_inst", "masks_cls", "masks_ignore"]}
        missing = {k: len(sets["images"] ^ v) for k, v in sets.items() if k != "images" and sets["images"] != v}
        stems = sorted(sets["images"] & sets["masks"] & sets["masks_inst"] & sets["masks_cls"])
        with Pool(args.workers) as pool:
            res = pool.map(check_image, [(d, s, args.min_area) for s in stems], chunksize=32)
        flagged = [r for r in res if r["flags"]]
        img_cnt = collections.Counter(k for r in flagged for k in r["flags"])
        inst_cnt = collections.Counter({k: sum(len(r["flags"].get(k, [])) for r in flagged) for k in img_cnt})
        print(f"\n== {sp}: {len(stems)} ảnh | lệch tập file: {missing or 'không'}")
        for k in STRUCT + SHAPE:
            if img_cnt[k]:
                print(f"   {k:20s} {img_cnt[k]:5d} ảnh ({100 * img_cnt[k] / len(stems):.2f}%), {inst_cnt[k]} con")
        if not flagged:
            print("   không có lỗi nào")
        report[sp] = {"images": len(stems), "file_set_mismatch": missing,
                      "flagged": {k: [r["stem"] for r in flagged if k in r["flags"]] for k in img_cnt}}
        if args.panels:
            out = Path(args.panels); out.mkdir(parents=True, exist_ok=True)
            for k in img_cnt:
                for r in [r for r in flagged if k in r["flags"]][:args.max_panels]:
                    panel(d, r["stem"], r["flags"][k], k, out)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
