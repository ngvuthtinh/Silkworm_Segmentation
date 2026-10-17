"""
Sửa nhãn ảnh THẬT bằng kết quả quét kỹ của SAM 3 (utils/sam3_recheck.py, thư mục dets/). Ghi ra split MỚI, không đụng split cũ.

  python utils/build_sam3_seg_v2/fix_real_labels.py data/sam3_aug20k test_real Reports/Meeting_2026-10-17/results/sam3_check/sam3_recheck/test_real test_real_v2
  python utils/build_sam3_seg_v2/fix_real_labels.py data/sam3_aug20k valid_real Reports/Meeting_2026-10-17/results/sam3_check/sam3_recheck/valid_real valid_real_v2

Quy tắc (mỗi ảnh):
  1. Con có nhãn bị coi là HỎNG nếu SAM 3 không phát hiện được (IoU < 0.3 với mọi phát hiện) VÀ mask lốm đốm
     (độ đặc < 0.5 hoặc chu vi/sqrt(diện tích) > 14). Nếu ảnh đã có một con hỏng thì mọi con SAM 3 không thấy
     trong ảnh đó cũng bị coi là hỏng (nhãn rác thường vỡ thành nhiều mảng). Con hỏng bị xóa khỏi nhãn.
  2. Phát hiện SAM 3 hợp lệ (điểm >= 0.5, diện tích >= 1000, dài hoặc chạm biên) mà không nằm trong con có nhãn còn lại
     và không nằm trong vùng ignore:
       điểm >= 0.7  → thêm làm con có nhãn (lớp theo tên file, như cách cũ)
       0.5 - 0.7    → thêm vào vùng ignore (chưa chắc, không dạy cũng không chấm)
  3. Ảnh không còn con nào có nhãn thì bỏ.
Mọi thay đổi ghi trong changes.json (thư mục split mới).
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

root, split_in, scan_dir, split_out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
LABEL_T, IGNORE_T, MIN_AREA, DET_AREA, ASPECT = 0.7, 0.5, 625, 1000, 1.5
SRC, OUT = root / split_in, root / split_out
for d in ["images", "masks", "masks_inst", "masks_cls", "masks_ignore"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)


def shape(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    ys, xs = np.nonzero(m)
    border = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cs)))
    per = sum(cv2.arcLength(x, True) for x in cs)
    a = int(m.sum())
    return a, max(rw, rh) / max(min(rw, rh), 1), border, a / max(hull, 1), per / np.sqrt(a)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


changes, kept, dropped = [], 0, 0
n_removed = n_added = n_ignored = 0
for ip in sorted((SRC / "images").glob("*.jpg")):
    s = ip.stem
    inst = cv2.imread(str(SRC / "masks_inst" / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    ign = cv2.imread(str(SRC / "masks_ignore" / f"{s}.png"), 0) > 127
    cls = 0 if s.lower().startswith("grasserie") else 1            # 0 = Grasserie, 1 = Healthy (như cách cũ)
    npz = scan_dir / "dets" / f"{s}.npz"
    dets, sc = np.zeros((0,) + inst.shape, bool), np.zeros(0)
    if npz.exists():
        d = np.load(npz)
        dets = np.unpackbits(d["masks"], axis=-1)[..., :inst.shape[1]].astype(bool)
        sc = d["scores"]
    valid = [i for i in range(len(dets)) if sc[i] >= IGNORE_T]
    log = {}

    # 1. xóa con hỏng: SAM 3 không thấy VÀ (mask lốm đốm, hoặc ảnh này đã có con lốm đốm → các mảng còn lại cũng là rác)
    info = {}
    for g in [g for g in np.unique(inst) if g > 0]:
        gm = inst == g
        seen = max((iou(gm, dets[i]) for i in valid), default=0.0) >= 0.3
        _, _, _, sol, comp = shape(gm)
        info[int(g)] = (seen, sol < 0.5 or comp > 14)
    garbage_image = any((not seen) and noisy for seen, noisy in info.values())
    for g, (seen, noisy) in info.items():
        if not seen and (noisy or garbage_image):
            inst[inst == g] = 0
            log.setdefault("removed_bad", []).append(g)
            n_removed += 1

    # 2. con SAM 3 thấy mà nhãn không có
    nxt = int(inst.max()) + 1
    gt_union = inst > 0
    new_inst = []
    for i in sorted(valid, key=lambda i: -sc[i]):
        m = dets[i] & ~gt_union
        a, aspect, border, _, _ = shape(dets[i])
        if a < DET_AREA or not (aspect >= ASPECT or border):
            continue
        if (dets[i] & gt_union).sum() / a >= 0.3 or (dets[i] & ign).sum() / a >= 0.5:
            continue
        if sc[i] >= LABEL_T and m.sum() >= MIN_AREA:
            inst[m] = nxt; gt_union |= m; nxt += 1
            log.setdefault("added_labelled", []).append(round(float(sc[i]), 2)); n_added += 1
        else:
            ign |= m
            log.setdefault("added_ignore", []).append(round(float(sc[i]), 2)); n_ignored += 1

    ids = [g for g in np.unique(inst) if g > 0]
    if not ids:
        dropped += 1
        changes.append({"stem": s, "dropped": True, **log})
        continue
    out_inst = np.zeros(inst.shape, np.uint16)
    for k, g in enumerate(ids, 1):
        out_inst[inst == g] = k
    cm = np.where(out_inst > 0, cls + 1, 0).astype(np.uint8)
    shutil.copy(ip, OUT / "images" / ip.name)
    cv2.imwrite(str(OUT / "masks" / f"{s}.png"), np.where(out_inst > 0, 255, 0).astype(np.uint8))
    cv2.imwrite(str(OUT / "masks_inst" / f"{s}.png"), out_inst)
    cv2.imwrite(str(OUT / "masks_cls" / f"{s}.png"), cm)
    cv2.imwrite(str(OUT / "masks_ignore" / f"{s}.png"), np.where(ign & (out_inst == 0), 255, 0).astype(np.uint8))
    kept += 1
    if log:
        changes.append({"stem": s, **log})

(OUT / "changes.json").write_text(json.dumps(changes, indent=1))
print(f"{split_in} → {split_out}: giữ {kept} ảnh, bỏ {dropped} | xóa con hỏng {n_removed} | thêm con có nhãn {n_added} | thêm ignore {n_ignored} | ảnh bị đổi {len(changes)}")
