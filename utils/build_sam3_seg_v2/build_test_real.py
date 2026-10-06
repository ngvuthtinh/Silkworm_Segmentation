"""Dựng <out>/test_real từ <src>/test (ảnh thật, không ghép): id từng con, lớp, vùng bỏ qua."""
import sys, shutil, cv2, numpy as np
from pathlib import Path
SRC, OUT, MIN_AREA = Path(sys.argv[1]) / "test", Path(sys.argv[2]) / "test_real", int(sys.argv[3])
for d in ["images", "masks", "labels", "masks_inst", "masks_cls", "masks_ignore"]: (OUT / d).mkdir(parents=True, exist_ok=True)
n = frag = 0
for mp in sorted((SRC / "masks").glob("*.png")):
    s = mp.stem; m = (cv2.imread(str(mp), 0) > 127).astype(np.uint8)
    ign = cv2.imread(str(SRC / "masks_ignore" / mp.name), 0) > 127
    cls = 1 if s.lower().startswith("healthy") else 0
    k, cc = cv2.connectedComponents(m); inst = np.zeros(m.shape, np.uint16); cm = np.zeros(m.shape, np.uint8); nid = 1
    for j in range(1, k):
        r = cc == j
        if r.sum() < MIN_AREA: ign |= r; frag += 1; continue
        inst[r] = nid; cm[r] = cls + 1; nid += 1
    if nid == 1: continue
    shutil.copy(SRC / "images" / f"{s}.jpg", OUT / "images" / f"{s}.jpg"); shutil.copy(SRC / "labels" / f"{s}.txt", OUT / "labels" / f"{s}.txt")
    cv2.imwrite(str(OUT / "masks" / mp.name), np.where(inst > 0, 255, 0).astype(np.uint8))
    cv2.imwrite(str(OUT / "masks_inst" / mp.name), inst); cv2.imwrite(str(OUT / "masks_cls" / mp.name), cm)
    cv2.imwrite(str(OUT / "masks_ignore" / mp.name), np.where(ign & (inst == 0), 255, 0).astype(np.uint8)); n += 1
print(f"test_real: {n} ảnh | mảnh nhỏ chuyển thành vùng bỏ qua: {frag}")
