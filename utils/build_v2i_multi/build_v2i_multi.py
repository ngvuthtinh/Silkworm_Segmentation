#!/usr/bin/env python3
"""
Dựng data/silkworm_v2i_multi: các ảnh NHIỀU TẰM (>= --min-larvae con) của Roboflow "silkworm" v2 (sicau-wjypc, CC BY 4.0), có nhãn từng con.

Nguồn:
  data/legacy/silkworm.v2i.yolov8/train/{images,labels}      ảnh + hộp gốc (heathly / sick)
  data/sam3_seg_silkworm_v2i/train/labels                    polygon từng con do SAM 3 tạo từ hộp (lớp lấy từ hộp khớp)
Đầu ra cùng định dạng với sam3_aug20k_v3 (images/, masks/, masks_inst/, masks_cls/, masks_ignore/) + manifest.json, README.md, previews/:
  masks_cls: 1 = sick (KHÔNG healthy, bệnh bất kỳ), 2 = heathly (Healthy)  — cùng quy ước số với repo (1 = lớp bệnh, 2 = Healthy)
  masks_ignore (255): (a) mảnh con quá nhỏ (< 100/256² ảnh); (b) vùng của HỘP GỐC không có mask nào của SAM 3 (con SAM 3 bỏ sót hoặc hộp lệch
                      do mosaic) → không bị học là nền, không tính điểm.
Bỏ: ảnh rỗng (SAM 3 không ra con nào) và ảnh có < --min-larvae con (ảnh tằm đơn / ít con). Con chồng nhau: con lớn vẽ trước, con nhỏ đè lên.

  python utils/build_v2i_multi/build_v2i_multi.py [--min-larvae 5] [--out data/silkworm_v2i_multi]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/legacy/silkworm.v2i.yolov8/train")
SAM = Path("data/sam3_seg_silkworm_v2i/train/labels")
CLS_SICK, CLS_HEALTHY = 1, 2
YOLO_TO_CLS = {"0": CLS_HEALTHY, "1": CLS_SICK}          # 0 = heathly, 1 = sick (data.yaml của nguồn)
BOX_COVER = 0.15                                          # hộp được coi là "có mask" nếu >= 15% diện tích hộp đã thuộc một con có nhãn


def rd(p: Path) -> list[list[str]]:
    return [l.split() for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/silkworm_v2i_multi")
    ap.add_argument("--min-larvae", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists() or not any(out.iterdir()), f"{out} đã có dữ liệu; xoá hoặc đổi --out"
    for d in ("images", "masks", "masks_inst", "masks_cls", "masks_ignore"):
        (out / "train" / d).mkdir(parents=True, exist_ok=True)
    manifest, dropped = [], {"rong": 0, "it_con": 0}
    for lab in sorted(SAM.glob("*.txt")):
        stem = lab.stem
        polys = rd(lab)
        if not polys:
            dropped["rong"] += 1
            continue
        if len(polys) < a.min_larvae:                                         # nhanh: ít dòng nhãn hơn ngưỡng thì chắc chắn bị loại
            dropped["it_con"] += 1
            continue
        img = cv2.imread(str(SRC / "images" / f"{stem}.jpg"))
        h, w = img.shape[:2]
        pts = [(YOLO_TO_CLS[p[0]], (np.array(p[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)) for p in polys]
        pts.sort(key=lambda t: -cv2.contourArea(t[1]))                        # lớn trước, nhỏ đè lên
        inst = np.zeros((h, w), np.uint16)
        cls = np.zeros((h, w), np.uint8)
        for k, (c, poly) in enumerate(pts, 1):
            cv2.fillPoly(inst, [poly], k)
            cv2.fillPoly(cls, [poly], c)
        min_area = max(1, round(100 / (256 * 256) * h * w))
        ign = np.zeros((h, w), bool)
        for k in range(1, len(pts) + 1):
            m = inst == k
            n = int(m.sum())
            if n and n < min_area:                                           # mảnh vụn → vùng bỏ qua
                ign |= m
                inst[m] = 0
                cls[m] = 0
        if len([i for i in np.unique(inst) if i > 0]) < a.min_larvae:        # đếm lại SAU khi dọn mảnh vụn / con bị đè hẳn
            dropped["it_con"] += 1
            continue
        # hộp gốc không có mask → vùng bỏ qua (con SAM 3 bỏ sót hoặc hộp lệch)
        labelled = inst > 0
        n_unmatched = 0
        for b in rd(SRC / "labels" / f"{stem}.txt"):
            cx, cy, bw, bh = (float(v) for v in b[1:5])
            x0, x1 = int(max(0, (cx - bw / 2) * w)), int(min(w, (cx + bw / 2) * w))
            y0, y1 = int(max(0, (cy - bh / 2) * h)), int(min(h, (cy + bh / 2) * h))
            area = max(1, (x1 - x0) * (y1 - y0))
            if labelled[y0:y1, x0:x1].sum() / area < BOX_COVER:
                n_unmatched += 1
                ign[y0:y1, x0:x1] |= True
        ign &= ~labelled                                                     # con có nhãn không nằm trong vùng bỏ qua
        ids = [int(i) for i in np.unique(inst) if i > 0]
        shutil.copy2(SRC / "images" / f"{stem}.jpg", out / "train" / "images" / f"{stem}.jpg")
        cv2.imwrite(str(out / "train" / "masks_inst" / f"{stem}.png"), inst)
        cv2.imwrite(str(out / "train" / "masks" / f"{stem}.png"), ((inst > 0) * 255).astype(np.uint8))
        cv2.imwrite(str(out / "train" / "masks_cls" / f"{stem}.png"), cls)
        cv2.imwrite(str(out / "train" / "masks_ignore" / f"{stem}.png"), (ign * 255).astype(np.uint8))
        n_sick = sum(1 for i in ids if np.bincount(cls[inst == i]).argmax() == CLS_SICK)
        manifest.append({"split": "train", "stem": stem, "source": "v2i", "base": re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", stem),
                         "larvae_in_label": len(ids), "n_sick": n_sick, "n_healthy": len(ids) - n_sick,
                         "unmatched_boxes": n_unmatched, "tier": "10+" if len(ids) >= 10 else "8-9" if len(ids) >= 8 else "5-7"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tiers = {t: sum(1 for x in manifest if x["tier"] == t) for t in ("5-7", "8-9", "10+")}
    nl = sum(x["larvae_in_label"] for x in manifest)
    ns = sum(x["n_sick"] for x in manifest)
    (out / "previews").mkdir(exist_ok=True)
    for x in random.Random(0).sample(manifest, min(24, len(manifest))):
        im = cv2.imread(str(out / "train" / "images" / f"{x['stem']}.jpg"))
        ins = cv2.imread(str(out / "train" / "masks_inst" / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
        cl = cv2.imread(str(out / "train" / "masks_cls" / f"{x['stem']}.png"), 0)
        ig = cv2.imread(str(out / "train" / "masks_ignore" / f"{x['stem']}.png"), 0) > 127
        ov = im.astype(np.float32)
        for k in np.unique(ins[ins > 0]):
            m = ins == k
            col = np.array((60, 60, 235) if np.bincount(cl[m]).argmax() == CLS_SICK else (60, 190, 60), np.float32)   # đỏ = sick, xanh = healthy
            ov[m] = 0.5 * ov[m] + 0.5 * col
        ov[ig] = 0.6 * ov[ig] + 0.4 * 128
        cv2.imwrite(str(out / "previews" / f"{x['stem'][:60]}.jpg"), np.hstack([im, np.clip(ov, 0, 255).astype(np.uint8)]))
    (out / "README.md").write_text(
        "# silkworm_v2i_multi\n\nẢnh nhiều tằm (>= %d con) của Roboflow \"silkworm\" v2 (workspace sicau-wjypc, **CC BY 4.0**, "
        "https://universe.roboflow.com/sicau-wjypc/silkworm/dataset/2), mask từng con do SAM 3 tạo từ hộp gốc, lớp từ hộp: `sick` = 1 (không healthy, bệnh bất kỳ), "
        "`heathly` = 2 (Healthy). Hộp gốc không có mask và mảnh quá nhỏ nằm trong `masks_ignore`.\n\n"
        "- Ảnh: %d (nhóm 5-7 con: %d, 8-9: %d, ≥ 10: %d); con: %d (sick %d, healthy %d).\n"
        "- Bỏ: %d ảnh rỗng (SAM 3 không ra con nào), %d ảnh có < %d con (đếm sau khi dọn mảnh vụn).\n"
        "- Cảnh báo: nhiều ảnh là mosaic có lớp trong suốt chồng nhau và con bị cắt ở biên các ô ghép, nên nhãn nhiễu hơn ảnh dày của dense_seg.\n"
        "- Mỗi ảnh gốc có nhiều bản augment (mosaic, xoay, nhiễu); chỉ dùng cho TRAIN (không có valid/test, không tách theo cảnh được).\n"
        "- Sinh bởi `utils/build_v2i_multi/build_v2i_multi.py`. Đã đo: không trùng với ảnh nguồn của sam3_aug20k_v3 (SIFT + RANSAC, 0/2412).\n"
        % (a.min_larvae, len(manifest), tiers["5-7"], tiers["8-9"], tiers["10+"], nl, ns, nl - ns, dropped["rong"], dropped["it_con"], a.min_larvae))
    print(f"ảnh giữ: {len(manifest)} {tiers} | con: {nl} (sick {ns}, healthy {nl - ns}) | bỏ: {dropped}")
    print(f"hộp gốc không có mask → vùng bỏ qua: {sum(x['unmatched_boxes'] for x in manifest)} hộp")


if __name__ == "__main__":
    main()
