#!/usr/bin/env python3
"""
Thêm các con SAM 3 thấy mà nhãn của data/dense_seg_v1 không có → data/dense_seg_v2 (v1 giữ nguyên để so sánh).

Tiêu chuẩn chọn con thiếu và ngưỡng điểm giống utils/sam3_scan_dets + fix_seg_v3.py (xem check_missing_with_sam3.py):
  một phát hiện của SAM 3 là "con thiếu" nếu: diện tích >= 100/256² ảnh, dài (dài/rộng >= 1.5) hoặc chạm biên ảnh,
  < 30% nằm trong con đã có nhãn, < 50% nằm trong vùng bỏ qua.
  điểm >= --label-thr (0.7)  → THÊM VÀO NHÃN thành con mới (masks_inst, lớp 3 = chưa biết bệnh); chừa khe --gap px quanh con cũ
  điểm trong [--ignore-thr (0.5), --label-thr) → thêm vào VÙNG BỎ QUA (masks_ignore): không loss, không tính điểm
  Các phát hiện chồng nhau: điểm cao lấy trước; con mới không đè lên con cũ, con mới khác, hoặc vùng bỏ qua.

Đầu vào: dets của SAM 3 theo tên ảnh (data/legacy/sam3_scan_dense/all/dets/<stem>.npz); ảnh không có dets giữ nguyên và được đếm.

  python utils/build_dense_seg/apply_sam3_missing.py [--src data/legacy/dense_seg_v1] [--out data/dense_seg_v2]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualize_instances import instance_color  # noqa: E402

CLS_UNKNOWN = 3
ASPECT, LABEL_OVERLAP, IGNORE_OVERLAP = 1.5, 0.3, 0.5


def shape(m: np.ndarray) -> tuple[int, float, bool]:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    ys, xs = np.nonzero(m)
    border = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1)
    return int(m.sum()), float(max(rw, rh) / max(min(rw, rh), 1)), border


def keep_pieces(region: np.ndarray, min_area: int) -> np.ndarray:
    """Chỉ giữ các mảnh liền >= min_area (bỏ vụn do bị cắt bớt khi tránh con cũ)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    out = np.zeros_like(region)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out |= lab == k
    return out


def process_image(inst: np.ndarray, ign: np.ndarray, dets: np.ndarray, scores: np.ndarray, label_thr: float, ignore_thr: float, gap: int):
    """Trả về (inst mới, ign mới, số con thêm vào nhãn, số vùng thêm vào bỏ qua)."""
    h, w = inst.shape
    min_area = max(1, round(100 / (256 * 256) * h * w))
    lab0 = inst > 0
    ign0 = ign > 127
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    near_old = cv2.dilate(lab0.astype(np.uint8), kernel) > 0
    taken = lab0.copy()                         # pixel đã thuộc con (cũ hoặc mới thêm)
    new_inst, new_ign = inst.copy(), ign0.copy()
    next_id = int(inst.max()) + 1
    n_lab = n_ign = 0
    for i in np.argsort(-scores):
        s = float(scores[i])
        if s < ignore_thr:
            break
        m = dets[i]
        area, aspect, border = shape(m) if m.any() else (0, 0.0, False)
        if area < min_area or not (aspect >= ASPECT or border):
            continue
        if (m & lab0).sum() / area >= LABEL_OVERLAP or (m & ign0).sum() / area >= IGNORE_OVERLAP:
            continue
        if s >= label_thr:
            region = keep_pieces(m & ~near_old & ~taken & ~new_ign, min_area)
            if region.sum() < min_area:
                continue
            new_inst[region] = next_id
            next_id += 1
            taken |= region
            n_lab += 1
        else:
            region = keep_pieces(m & ~lab0 & ~taken & ~new_ign, min_area)
            if region.sum() < min_area:
                continue
            new_ign |= region
            n_ign += 1
    return new_inst, new_ign, n_lab, n_ign


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def colored(im: np.ndarray, inst: np.ndarray, ign_m: np.ndarray, before: np.ndarray | None = None) -> np.ndarray:
    """MỖI CON MỘT MÀU RIÊNG (kể cả hai con chồng lên nhau). Màu theo thứ tự id nên con cũ giữ nguyên màu giữa v1 và v2.
    Nếu có `before`: con mới thêm (không có trong before) được viền trắng dày. Xám = vùng bỏ qua."""
    ov = im.astype(np.float32)
    ids = [int(i) for i in np.unique(inst) if i > 0]
    for k, i in enumerate(ids):
        m = inst == i
        ov[m] = 0.5 * ov[m] + 0.5 * np.array(instance_color(k)[::-1], np.float32)      # RGB → BGR
    ov[ign_m] = 0.6 * ov[ign_m] + 0.4 * 128
    out = np.clip(ov, 0, 255).astype(np.uint8)
    for k, i in enumerate(ids):
        is_new = before is not None and not (before[inst == i] > 0).any()
        cnts, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255) if is_new else tuple(int(c) for c in instance_color(k)[::-1]), 3 if is_new else 1)
    return out


def titled(img: np.ndarray, label: str, side: int = 480) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    sc = side / max(img.shape[:2])
    r = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    bar = Image.new("RGB", (r.shape[1], 36), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    f = ImageFont.truetype(FONT, 19)
    d.text(((r.shape[1] - d.textlength(label, font=f)) / 2, 6), label, font=f, fill=(20, 20, 20))
    return np.vstack([r, np.array(bar)[:, :, ::-1]])


def render_preview(im: np.ndarray, before: np.ndarray, after: np.ndarray, ign_before: np.ndarray, ign_after: np.ndarray) -> np.ndarray:
    gap = np.full((0, 6, 3), 255, np.uint8)
    panels = [titled(im, "Ảnh gốc"), titled(colored(im, before, ign_before), "v1"), titled(colored(im, after, ign_after, before), "v2")]
    h = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, h - p.shape[0]), (0, 6), (0, 0)), constant_values=255) for p in panels]
    return np.hstack(panels)


def write_previews(src: Path, out: Path, manifest: list[dict], n: int, like: Path | None = None) -> None:
    """Ghi previews/: Ảnh gốc | v1 | v2. Nếu có `like`: dùng đúng các ảnh đang có trong thư mục previews đó (vd previews của v1)."""
    (out / "previews").mkdir(exist_ok=True)
    for old in (out / "previews").glob("*.jpg"):
        old.unlink()
    if like is not None:
        names = {f.stem for f in like.glob("*.jpg")}
        chosen = [y for y in manifest if f"{y['split']}__{y['stem'][:50]}" in names]
    else:
        changed = [y for y in manifest if y["added_to_label"]]
        chosen = random.Random(1).sample(changed, min(n, len(changed)))
    for y in chosen:
        sp, stem = y["split"], y["stem"]
        im = cv2.imread(str(out / sp / "images" / f"{stem}.jpg"))
        before = cv2.imread(str(src / sp / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        after = cv2.imread(str(out / sp / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        ign_b = cv2.imread(str(src / sp / "masks_ignore" / f"{stem}.png"), 0) > 127
        ign_a = cv2.imread(str(out / sp / "masks_ignore" / f"{stem}.png"), 0) > 127
        cv2.imwrite(str(out / "previews" / f"{sp}__{stem[:50]}.jpg"), render_preview(im, before, after, ign_b, ign_a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/legacy/dense_seg_v1")
    ap.add_argument("--out", default="data/dense_seg_v2")
    ap.add_argument("--dets-dir", nargs="+", default=["data/legacy/sam3_scan_dense/all/dets"], help="một hoặc nhiều thư mục dets (tìm theo tên ảnh)")
    ap.add_argument("--readme-title", default="dense_seg_v2", help="tên bộ data ghi trong README")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--label-thr", type=float, default=0.7)
    ap.add_argument("--ignore-thr", type=float, default=0.5)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--previews", type=int, default=60, help="số ảnh before/after ghi vào previews/")
    ap.add_argument("--previews-only", action="store_true", help="chỉ vẽ lại previews/ từ --out đã dựng, không dựng lại data")
    ap.add_argument("--previews-like", default=None, help="thư mục previews cần bắt chước (vd data/dense_seg_v1/previews): dùng đúng các ảnh đó")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if a.previews_only:
        write_previews(src, out, json.loads((out / "manifest.json").read_text()), a.previews, Path(a.previews_like) if a.previews_like else None)
        print(f"đã vẽ lại previews ở {out / 'previews'}")
        return
    assert not out.exists() or not any(out.iterdir()), f"{out} đã có dữ liệu; xoá hoặc đổi --out"
    manifest = json.loads((src / "manifest.json").read_text())
    new_manifest, no_dets = [], 0
    tot = {"images": 0, "labelled": 0, "ignored": 0, "images_changed": 0}
    for x in manifest:
        sp, stem = x["split"], x["stem"]
        if sp not in a.splits:
            continue
        for d in ("images", "masks", "masks_inst", "masks_cls", "masks_ignore"):
            (out / sp / d).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / sp / "images" / f"{stem}.jpg", out / sp / "images" / f"{stem}.jpg")
        inst = cv2.imread(str(src / sp / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        inst0 = inst.copy()
        cls0 = cv2.imread(str(src / sp / "masks_cls" / f"{stem}.png"), 0)
        ign = cv2.imread(str(src / sp / "masks_ignore" / f"{stem}.png"), 0)
        npz = next((Path(d) / f"{stem}.npz" for d in a.dets_dir if (Path(d) / f"{stem}.npz").exists()), Path(a.dets_dir[0]) / f"{stem}.npz")
        n_lab = n_ign = 0
        if npz.exists():
            z = np.load(npz)
            h, w = (int(v) for v in z["shape"])
            if len(z["scores"]):
                dets = np.unpackbits(z["masks"], axis=-1)[..., :w].astype(bool)
                inst, ign_b, n_lab, n_ign = process_image(inst, ign, dets, z["scores"], a.label_thr, a.ignore_thr, a.gap)
                ign = (ign_b * 255).astype(np.uint8)
        else:
            no_dets += 1
        cv2.imwrite(str(out / sp / "masks_inst" / f"{stem}.png"), inst)
        cv2.imwrite(str(out / sp / "masks" / f"{stem}.png"), ((inst > 0) * 255).astype(np.uint8))
        cls_new = cls0.copy()                                           # con cũ giữ lớp (Khỏe / Bệnh / chưa biết); con mới thêm = chưa biết bệnh
        cls_new[(inst > 0) & (inst0 == 0)] = CLS_UNKNOWN
        cv2.imwrite(str(out / sp / "masks_cls" / f"{stem}.png"), cls_new)
        cv2.imwrite(str(out / sp / "masks_ignore" / f"{stem}.png"), ign)
        y = dict(x)
        y.update({"larvae_in_label_v1": x["larvae_in_label"], "larvae_in_label": x["larvae_in_label"] + n_lab,
                  "added_to_label": n_lab, "added_to_ignore": n_ign, "sam3_scanned": npz.exists()})
        new_manifest.append(y)
        tot["images"] += 1; tot["labelled"] += n_lab; tot["ignored"] += n_ign; tot["images_changed"] += bool(n_lab or n_ign)
    (out / "manifest.json").write_text(json.dumps(new_manifest, indent=1, ensure_ascii=False))

    rows = ["| Split | Ảnh | Con trong nhãn v1 | Thêm vào nhãn | Con trong nhãn v2 | Vùng thêm vào bỏ qua | Ảnh bị đổi |", "|---|---|---|---|---|---|---|"]
    for sp in a.splits:
        m = [y for y in new_manifest if y["split"] == sp]
        if not m:
            continue
        rows.append(f"| {sp} | {len(m)} | {sum(y['larvae_in_label_v1'] for y in m)} | {sum(y['added_to_label'] for y in m)} | "
                    f"{sum(y['larvae_in_label'] for y in m)} | {sum(y['added_to_ignore'] for y in m)} | {sum(1 for y in m if y['added_to_label'] or y['added_to_ignore'])} |")
    write_previews(src, out, new_manifest, a.previews, Path(a.previews_like) if a.previews_like else None)
    (out / "README.md").write_text(
        f"# {a.readme_title}\n\nBộ `{a.src}` + các con SAM 3 thấy mà nhãn gốc không có (sinh bởi `utils/build_dense_seg/apply_sam3_missing.py`).\n"
        "Ảnh, split giống bộ gốc; con cũ giữ lớp, con mới thêm có `masks_cls` = 3 (chưa biết bệnh). Quy tắc thêm: điểm SAM 3 ≥ 0.7 → thêm vào nhãn; 0.5-0.7 → vùng bỏ qua.\n\n"
        + "\n".join(rows) + f"\n\nẢnh không có kết quả quét SAM 3 (giữ nguyên như v1): {no_dets}.\n\n"
        "## Nguồn và giấy phép\nNhư v1: `fl_*` = Roboflow \"fl\" (asiak), CC BY 4.0; `silkynet_*` = dự án Silkynet. Mask bổ sung do SAM 3 tạo, chưa ai kiểm tra bằng tay.\n\n"
        "## Xem ảnh\n`previews/`: Ảnh gốc | v1 | v2; mỗi con một màu riêng, con mới thêm vào v2 có viền trắng dày, xám = vùng bỏ qua.\n")
    print("\n".join(rows))
    print(f"ảnh không có dets: {no_dets} | tổng thêm vào nhãn: {tot['labelled']} | vùng bỏ qua thêm: {tot['ignored']} | ảnh bị đổi: {tot['images_changed']}/{tot['images']}")


if __name__ == "__main__":
    main()
