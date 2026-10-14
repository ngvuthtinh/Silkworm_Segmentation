"""
Chép ảnh THẬT (không ghép) từ <src_root>/<split_in> sang <out_root>/<split_out> kèm id từng con, lớp, vùng bỏ qua.

  python utils/build_sam3_seg_v2/build_real.py data/sam3_seg_v2 train data/sam3_aug20k train      625
  python utils/build_sam3_seg_v2/build_real.py data/sam3_seg_v2 valid data/sam3_aug20k valid_real 625
  python utils/build_sam3_seg_v2/build_real.py data/sam3_seg_v2 test  data/sam3_aug20k test_real  625

Mỗi mảng liền của mask là một con (ảnh gốc chỉ có một mask chung, nên các con chạm nhau bị gộp — khoảng 1–2% ảnh).
Mảng nhỏ hơn min_area (px ở ảnh 640) → vùng bỏ qua. Lớp lấy theo tên file.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

src_root, split_in, out_root, split_out, min_area = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
SRC, OUT = Path(src_root) / split_in, Path(out_root) / split_out
for d in ["images", "masks", "labels", "masks_inst", "masks_cls", "masks_ignore"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
n = frag = skipped = 0
for mp in sorted((SRC / "masks").glob("*.png")):
    s = mp.stem
    m = (cv2.imread(str(mp), 0) > 127).astype(np.uint8)
    ign = cv2.imread(str(SRC / "masks_ignore" / mp.name), 0) > 127
    cls = 1 if s.lower().startswith("healthy") else 0
    k, cc = cv2.connectedComponents(m)
    inst = np.zeros(m.shape, np.uint16); cm = np.zeros(m.shape, np.uint8); nid = 1
    for j in range(1, k):
        r = cc == j
        if r.sum() < min_area:
            ign |= r; frag += 1
            continue
        inst[r] = nid; cm[r] = cls + 1; nid += 1
    if nid == 1:
        skipped += 1
        continue
    shutil.copy(SRC / "images" / f"{s}.jpg", OUT / "images" / f"{s}.jpg")
    shutil.copy(SRC / "labels" / f"{s}.txt", OUT / "labels" / f"{s}.txt")
    cv2.imwrite(str(OUT / "masks" / mp.name), np.where(inst > 0, 255, 0).astype(np.uint8))
    cv2.imwrite(str(OUT / "masks_inst" / mp.name), inst)
    cv2.imwrite(str(OUT / "masks_cls" / mp.name), cm)
    cv2.imwrite(str(OUT / "masks_ignore" / mp.name), np.where(ign & (inst == 0), 255, 0).astype(np.uint8))
    n += 1
print(f"{split_in} → {split_out}: {n} ảnh thật | mảnh nhỏ → vùng bỏ qua: {frag} | bỏ (không còn con nào): {skipped}")
