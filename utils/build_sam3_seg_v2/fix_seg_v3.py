"""
Sửa mask nguồn bằng kết quả quét kỹ của SAM 3 (utils/sam3_scan_dets.py) → ghi data/sam3_seg_v3 (không ghi đè v2).

  python utils/build_sam3_seg_v2/fix_seg_v3.py valid test train

Quy tắc (mỗi ảnh; "con" = mảng liền của mask >= 625 px, giống build_real.py / augment_mixed_10k.py):
  1. Con bị coi là HỎNG nếu SAM 3 không phát hiện được (IoU < 0.3 với mọi phát hiện điểm >= 0.5) VÀ mask lốm đốm
     (độ đặc < 0.5 hoặc chu vi/sqrt(diện tích) > 14). Nếu ảnh đã có một con hỏng thì mọi con SAM 3 không thấy
     trong ảnh đó cũng bị coi là hỏng. Con hỏng bị xóa khỏi mask.
  2. Phát hiện SAM 3 hợp lệ (điểm >= 0.5, diện tích >= 1000, dài hoặc chạm biên) mà không nằm trong con còn lại
     và không nằm trong vùng ignore:
       điểm >= 0.7  → thêm vào mask (chừa khe 2 px quanh con cũ để không dính vào nhau)
       0.5 - 0.7    → thêm vào vùng ignore
  3. Ảnh không còn mảng nào >= 625 px thì bỏ.
Ghi changes.json từng split; labels/ (hộp YOLO) không được sao chép vì không còn khớp mask.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT, SCAN_ROOT, OUT_ROOT = Path("data/sam3_seg_v2"), Path("data/sam3_scan"), Path("data/sam3_seg_v3")
LABEL_T, IGNORE_T, MIN_AREA, DET_AREA, ASPECT, GAP = 0.7, 0.5, 625, 1000, 1.5, 2
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1, 2 * GAP + 1))


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


def fix_split(split: str) -> None:
    src, out, scan = SRC_ROOT / split, OUT_ROOT / split, SCAN_ROOT / split / "dets"
    for d in ["images", "masks", "masks_ignore"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    changes, kept, dropped, missing = [], 0, 0, 0
    n_removed = n_added = n_ignored = 0
    for ip in sorted((src / "images").glob("*.jpg")):
        s = ip.stem
        mask = cv2.imread(str(src / "masks" / f"{s}.png"), 0) > 127
        ign = cv2.imread(str(src / "masks_ignore" / f"{s}.png"), 0) > 127
        npz = scan / f"{s}.npz"
        if npz.exists():
            d = np.load(npz)
            dets = np.unpackbits(d["masks"], axis=-1)[..., :mask.shape[1]].astype(bool)
            sc = d["scores"]
        else:                                           # chưa quét: giữ nguyên ảnh, không sửa
            dets, sc = np.zeros((0,) + mask.shape, bool), np.zeros(0)
            missing += 1
        valid = [i for i in range(len(dets)) if sc[i] >= IGNORE_T]
        log = {}

        # 1. xóa con hỏng
        k, cc = cv2.connectedComponents(mask.astype(np.uint8))
        info = {}
        for j in range(1, k):
            r = cc == j
            if r.sum() < MIN_AREA:
                continue
            seen = max((iou(r, dets[i]) for i in valid), default=0.0) >= 0.3
            _, _, _, sol, comp = shape(r)
            info[j] = (seen, sol < 0.5 or comp > 14)
        garbage_image = npz.exists() and any((not seen) and noisy for seen, noisy in info.values())
        for j, (seen, noisy) in info.items():
            if npz.exists() and not seen and (noisy or garbage_image):
                mask = mask & (cc != j)
                log["removed_bad"] = log.get("removed_bad", 0) + 1
                n_removed += 1

        # 2. con SAM 3 thấy mà mask không có
        for i in sorted(valid, key=lambda i: -sc[i]):
            a, aspect, border, _, _ = shape(dets[i])
            if a < DET_AREA or not (aspect >= ASPECT or border):
                continue
            if (dets[i] & mask).sum() / a >= 0.3 or (dets[i] & ign).sum() / a >= 0.5:
                continue
            if sc[i] >= LABEL_T:
                new = dets[i] & ~cv2.dilate(mask.astype(np.uint8), KERNEL).astype(bool)
                if new.sum() >= MIN_AREA:
                    mask = mask | new
                    log.setdefault("added_labelled", []).append(round(float(sc[i]), 2)); n_added += 1
                    continue
            ign |= dets[i] & ~mask
            log.setdefault("added_ignore", []).append(round(float(sc[i]), 2)); n_ignored += 1

        k, cc = cv2.connectedComponents(mask.astype(np.uint8))
        if not any((cc == j).sum() >= MIN_AREA for j in range(1, k)):
            dropped += 1
            changes.append({"stem": s, "dropped": True, **log})
            continue
        shutil.copy(ip, out / "images" / ip.name)
        cv2.imwrite(str(out / "masks" / f"{s}.png"), np.where(mask, 255, 0).astype(np.uint8))
        cv2.imwrite(str(out / "masks_ignore" / f"{s}.png"), np.where(ign & ~mask, 255, 0).astype(np.uint8))
        kept += 1
        if log:
            changes.append({"stem": s, **log})
    (out / "changes.json").write_text(json.dumps(changes, indent=1))
    print(f"{split}: giữ {kept}, bỏ {dropped}, chưa quét {missing} | xóa con hỏng {n_removed} | thêm vào mask {n_added} "
          f"| thêm ignore {n_ignored} | ảnh bị đổi {len(changes)}", flush=True)


if __name__ == "__main__":
    OUT_ROOT.mkdir(exist_ok=True)
    shutil.copy(SRC_ROOT / "split_manifest.json", OUT_ROOT / "split_manifest_from_v2.json")
    for sp in sys.argv[1:]:
        fix_split(sp)
