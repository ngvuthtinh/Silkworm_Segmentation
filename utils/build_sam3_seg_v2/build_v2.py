"""Dựng data/sam3_seg_v2: split theo cảnh + mask đã cứu + bỏ khung lỗi + vùng bỏ qua (tằm không có nhãn)."""
import json, sys, shutil, collections, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "utils")
from sam3_label_from_yolo import compute_iou
SCR, OUT = Path(sys.argv[1]), Path(sys.argv[2]); only = set(sys.argv[3:]) or {"train", "valid", "test"}
M = json.load(open(SCR / "split_manifest_final.json")); rescued = json.load(open(SCR / "rescued_ok.json"))
SCORE, IOU_BOX, OVL_MASK, DIL = 0.4, 0.3, 0.3, 7
stats = collections.Counter()
for r in M:
    sp, src, stem = r["split"], r["src_split"], r["stem"]
    if sp not in only: continue
    for d in ["images", "masks", "labels", "masks_ignore"]: (OUT / sp / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(f"data/yolo_bbox/{src}/images/{stem}.jpg", OUT / sp / "images" / f"{stem}.jpg")
    mp = SCR / "rescue_out" / src / "masks" / f"{stem}.png" if stem in set(rescued[src]) else Path(f"data/sam3_seg/{src}/masks/{stem}.png")
    m = cv2.imread(str(mp), 0) > 127; h, w = m.shape
    stats["mask đã cứu"] += stem in set(rescued[src])
    lines, boxes = [], []
    for ln in Path(f"data/yolo_bbox/{src}/labels/{stem}.txt").read_text().splitlines():
        t = ln.split()
        if len(t) != 5: continue
        c, xc, yc, bw, bh = int(t[0]), *map(float, t[1:])
        if bw * w < 8 or bh * h < 8 or bw * bh < 0.0005: stats["khung YOLO lỗi đã bỏ"] += 1; continue
        lines.append(ln.strip()); boxes.append(np.array([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
    (OUT / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    cv2.imwrite(str(OUT / sp / "masks" / f"{stem}.png"), np.where(m, 255, 0).astype(np.uint8))
    ign = np.zeros((h, w), bool)
    z = np.load(SCR / "sam3_det" / src / f"{stem}.npz")
    if len(z["scores"]):
        n, H_, W_ = z["shape"]; dm = np.unpackbits(z["masks"], axis=-1)[..., :W_].astype(bool)
        for box, sc, md in zip(z["boxes"], z["scores"], dm):
            if sc < SCORE: continue
            a = md.sum()
            if a < 300 or a > 0.25 * h * w: continue
            if boxes and max(compute_iou(box, b) for b in boxes) >= IOU_BOX: continue
            if (md & m).sum() / a >= OVL_MASK: continue
            ign |= md; stats["con tằm không nhãn → vùng bỏ qua"] += 1
    if ign.any():
        ign = cv2.dilate(ign.astype(np.uint8), np.ones((DIL, DIL), np.uint8)) > 0
        ign &= ~m; stats[f"ảnh có vùng bỏ qua ({sp})"] += 1
    cv2.imwrite(str(OUT / sp / "masks_ignore" / f"{stem}.png"), np.where(ign, 255, 0).astype(np.uint8))
    stats[f"ảnh {sp}"] += 1
print(dict(stats))
