#!/usr/bin/env python3
"""
Dựng data/dense_seg_v1: ảnh NHIỀU TẰM trong một khung hình, mask TỪNG CON, KHÔNG có nhãn bệnh.

Nguồn (đã có polygon từng con):
  data/legacy/fl.v2i.coco/{train,valid}  Roboflow "fl" (asiak), CC BY 4.0, polygon COCO; mỗi ảnh gốc có 3 bản augment
  data/legacy/silkynet_seg/20221127   Silkynet, labelme, 1280×720, ~16 con/ảnh; 69 khung hình của CÙNG MỘT cảnh

Đầu ra cùng định dạng với sam3_aug20k_v3 (xem DATASET_STRUCTURE.md) để InstanceSilkwormDataset đọc được:
  <split>/images/<stem>.jpg   kích thước gốc (không resize: sau này có thể cắt ô nhỏ)
  <split>/masks/<stem>.png    0/255 (= masks_inst > 0)
  <split>/masks_inst/<stem>.png  uint16, một id cho mỗi con
  <split>/masks_cls/<stem>.png   uint8: 0 = nền, 3 = con tằm CHƯA BIẾT BỆNH (src.dataset_instance.CLS_UNKNOWN)
  <split>/masks_ignore/<stem>.png  255 = mảnh quá nhỏ (đã bỏ khỏi nhãn)
  manifest.json, README.md, previews/

Quy ước:
  - Polygon chồng nhau: vẽ con lớn trước, con nhỏ đè lên (phần nhìn thấy của con nhỏ giữ nguyên).
  - Con có diện tích nhìn thấy < 100/256² diện tích ảnh (≈ 625 px ở 640²) bị bỏ khỏi nhãn và đánh dấu vùng bỏ qua.
  - Chia split THEO CẢNH (không theo ảnh): ảnh gần giống nhau (pixel; cùng ảnh gốc Roboflow) thuộc cùng một nhóm; cả 69 ảnh Silkynet là
    một nhóm và luôn ở train. valid/test chỉ lấy MỘT bản cho mỗi ảnh gốc (3 bản augment không được đếm ba lần khi chấm điểm).

  python utils/build_dense_seg/build_dense_seg_v1.py [--out data/dense_seg_v1] [--valid-frac 0.1] [--test-frac 0.1] [--seed 42]
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

COCO_ROOT = Path("data/legacy/fl.v2i.coco")
SILKY_ROOT = Path("data/legacy/silkynet_seg/20221127")
CLS_UNKNOWN = 3
DENSE_MIN = 10                       # một cảnh có >= ngần này con thì coi là "dày đặc" (dùng để chia đều valid/test)
NCC_THR, COS_PRE = 0.95, 0.90        # cùng ngưỡng với utils/build_sam3_seg_v2/group_scenes.py (chỉ dùng tín hiệu pixel, không SIFT)


def poly_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def load_coco() -> list[dict]:
    recs = []
    for split in ("train", "valid"):
        j = json.loads((COCO_ROOT / split / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in j["categories"]}
        by_img = collections.defaultdict(list)
        for a in j["annotations"]:
            if names.get(a["category_id"]) != "silkworm":
                continue
            seg = a.get("segmentation")
            if not isinstance(seg, list) or not seg:
                continue
            rings = [np.array(s, np.float32).reshape(-1, 2) for s in seg if len(s) >= 6]
            if rings:
                by_img[a["image_id"]].append(rings)
        for im in j["images"]:
            if not by_img[im["id"]]:
                continue
            recs.append({"source": "fl_coco", "src_path": str(COCO_ROOT / split / im["file_name"]),
                         "stem": "fl_" + Path(im["file_name"]).stem, "base": re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", im["file_name"]),      # tên ảnh gốc: bỏ phần "_<đuôi>.rf.<mã>" của Roboflow (có cả png, jpeg)
                         
                         "larvae": by_img[im["id"]], "rf_split": split})
    return recs


def load_silkynet() -> list[dict]:
    recs = []
    for jf in sorted(SILKY_ROOT.glob("*.json")):
        j = json.loads(jf.read_text())
        larvae, skipped = [], 0
        for s in j["shapes"]:
            if s.get("shape_type") == "polygon" and len(s["points"]) >= 3:
                larvae.append([np.array(s["points"], np.float32)])
            else:
                skipped += 1
        img_p = jf.with_suffix(".jpg")
        recs.append({"source": "silkynet", "src_path": str(img_p) if img_p.exists() else None, "image_data": j.get("imageData"),
                     "stem": f"silkynet_{jf.stem}", "base": "silkynet", "larvae": larvae, "rf_split": None, "skipped_shapes": skipped})
    return recs


def read_image(rec: dict) -> np.ndarray:
    if rec["src_path"]:
        return cv2.imread(rec["src_path"])
    return cv2.imdecode(np.frombuffer(base64.b64decode(rec["image_data"]), np.uint8), cv2.IMREAD_COLOR)


def rasterize(rec: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(inst uint16, ignore uint8, số con tính trên nhãn). Con lớn vẽ trước, con nhỏ đè lên."""
    order = sorted(rec["larvae"], key=lambda rings: -sum(poly_area(r) for r in rings))
    inst = np.zeros((h, w), np.uint16)
    for k, rings in enumerate(order, 1):
        for r in rings:
            cv2.fillPoly(inst, [np.round(r).astype(np.int32)], k)
    min_area = max(1, round(100 / (256 * 256) * h * w))
    ign = np.zeros((h, w), np.uint8)
    kept = 0
    for k in range(1, len(order) + 1):
        m = inst == k
        n = int(m.sum())
        if n == 0:
            continue
        if n < min_area:
            ign[m] = 255
            inst[m] = 0
        else:
            kept += 1
    return inst, ign, kept


def scene_groups(recs: list[dict]) -> dict[str, int]:
    """Nhóm cảnh cho các ảnh gốc fl (theo `base`): cùng base hoặc ảnh gần như giống nhau về pixel → cùng nhóm. Silkynet = một nhóm."""
    bases = sorted({r["base"] for r in recs if r["source"] == "fl_coco"})
    first = {b: next(r for r in recs if r["base"] == b) for b in bases}
    gray = {}
    for b in bases:
        g = cv2.cvtColor(read_image(first[b]), cv2.COLOR_BGR2GRAY)
        gray[b] = (cv2.resize(g, (24, 24)).astype(np.float32).ravel(), cv2.resize(g, (96, 96)).astype(np.float32).ravel())
    parent = {b: b for b in bases}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def ncc(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            ca, cb = gray[a][0], gray[b][0]
            if ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9) >= COS_PRE and ncc(gray[a][1], gray[b][1]) >= NCC_THR:
                parent[find(a)] = find(b)
    ids = {r: k for k, r in enumerate(sorted({find(b) for b in bases}))}
    return {b: ids[find(b)] for b in bases}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dense_seg_v1")
    ap.add_argument("--valid-frac", type=float, default=0.1, help="tỉ lệ NHÓM CẢNH fl vào valid")
    ap.add_argument("--test-frac", type=float, default=0.1, help="tỉ lệ NHÓM CẢNH fl vào test")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists() or not any(out.iterdir()), f"{out} đã có dữ liệu; xoá hoặc đổi --out"

    fl, silky = load_coco(), load_silkynet()
    grp = scene_groups(fl)
    n_larvae = {g: 0 for g in set(grp.values())}
    for r in fl:
        n_larvae[grp[r["base"]]] = max(n_larvae[grp[r["base"]]], len(r["larvae"]))
    rng = random.Random(a.seed)
    split_of: dict[int, str] = {}
    for dense in (True, False):                      # chia đều ảnh dày đặc và ảnh thưa vào valid/test
        gs = sorted(g for g in n_larvae if (n_larvae[g] >= DENSE_MIN) == dense)
        rng.shuffle(gs)
        nv, nt = round(len(gs) * a.valid_frac), round(len(gs) * a.test_frac)
        for k, g in enumerate(gs):
            split_of[g] = "valid" if k < nv else "test" if k < nv + nt else "train"

    manifest, used_bases = [], set()
    for sp in ("train", "valid", "test"):
        for d in ("images", "masks", "masks_inst", "masks_cls", "masks_ignore"):
            (out / sp / d).mkdir(parents=True, exist_ok=True)
    for r in sorted(fl + silky, key=lambda r: r["stem"]):
        if r["source"] == "silkynet":
            sp, group = "train", "silkynet"
        else:
            sp, group = split_of[grp[r["base"]]], f"fl_{grp[r['base']]:03d}"
            if sp != "train":                          # valid/test: một bản cho mỗi ảnh gốc
                if r["base"] in used_bases:
                    continue
                used_bases.add(r["base"])
        img = read_image(r)
        h, w = img.shape[:2]
        inst, ign, kept = rasterize(r, h, w)
        if kept == 0:
            continue
        dst = out / sp
        if r["src_path"]:
            shutil.copy2(r["src_path"], dst / "images" / f"{r['stem']}.jpg")
        else:
            cv2.imwrite(str(dst / "images" / f"{r['stem']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(dst / "masks_inst" / f"{r['stem']}.png"), inst)
        cv2.imwrite(str(dst / "masks" / f"{r['stem']}.png"), ((inst > 0) * 255).astype(np.uint8))
        cv2.imwrite(str(dst / "masks_cls" / f"{r['stem']}.png"), np.where(inst > 0, CLS_UNKNOWN, 0).astype(np.uint8))
        cv2.imwrite(str(dst / "masks_ignore" / f"{r['stem']}.png"), ign)
        manifest.append({"split": sp, "stem": r["stem"], "source": r["source"], "base": r["base"], "scene_group": group,
                         "width": w, "height": h, "larvae_in_label": kept, "larvae_in_polygons": len(r["larvae"])})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    # tóm tắt + preview + README
    rows = ["| Split | Ảnh | Ảnh Silkynet | Ảnh fl | Con (nhãn) | Con / ảnh | Ảnh ≥ 10 con | Cảnh (nhóm) |", "|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "valid", "test"):
        m = [x for x in manifest if x["split"] == sp]
        nl = sum(x["larvae_in_label"] for x in m)
        rows.append(f"| {sp} | {len(m)} | {sum(x['source'] == 'silkynet' for x in m)} | {sum(x['source'] == 'fl_coco' for x in m)} | {nl} | "
                    f"{nl / max(len(m), 1):.1f} | {sum(x['larvae_in_label'] >= DENSE_MIN for x in m)} | {len({x['scene_group'] for x in m})} |")
    (out / "previews").mkdir(exist_ok=True)
    pr = random.Random(0)
    pick = pr.sample(manifest, min(24, len(manifest)))
    for x in pick:
        im = cv2.imread(str(out / x["split"] / "images" / f"{x['stem']}.jpg"))
        ins = cv2.imread(str(out / x["split"] / "masks_inst" / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
        ov = im.copy().astype(np.float32)
        for k in np.unique(ins[ins > 0]):
            col = np.array(cv2.cvtColor(np.uint8([[[int(k * 47) % 180, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0], np.float32)
            ov[ins == k] = 0.5 * ov[ins == k] + 0.5 * col
        cv2.imwrite(str(out / "previews" / f"{x['split']}__{x['stem'][:50]}.jpg"), np.clip(ov, 0, 255).astype(np.uint8))
    (out / "README.md").write_text(
        "# dense_seg_v1\n\nẢnh nhiều tằm trong một khung hình, mask từng con (polygon → masks_inst), **không có nhãn bệnh** "
        "(`masks_cls` = 3 trên mọi con = chưa biết bệnh).\nSinh bởi `utils/build_dense_seg/build_dense_seg_v1.py`; mô tả đầy đủ ở "
        "DATASET_STRUCTURE.md §2.7.\n\n" + "\n".join(rows) + "\n\n## Nguồn và giấy phép\n"
        "- `fl_*`: Roboflow \"fl\" (asiak), https://universe.roboflow.com/asiak/fl-bdar2, **CC BY 4.0**. Đã đổi polygon thành mask từng con, "
        "gom lại theo cảnh, bỏ các bản augment thừa ở valid/test.\n"
        "- `silkynet_*`: dữ liệu của dự án Silkynet (repo `models/silkynet`, giấy phép GPL-3.0 cho mã; kiểm tra giấy phép dữ liệu trước khi công bố). "
        "Đã đổi polygon labelme thành mask từng con.\n\n"
        "## Lưu ý\n- Cả 69 ảnh Silkynet là **một cảnh** (cùng khay tằm), nên chỉ ở `train`.\n"
        "- Ảnh gốc có kích thước 640×639 (fl) và 1280×720 (Silkynet); `InstanceSilkwormDataset` resize về vuông khi đọc.\n"
        "- Nhãn là polygon của người khác, chưa được kiểm tra bởi chúng ta.\n")
    print("\n".join(rows))
    print(f"Dựng xong {out}: {len(manifest)} ảnh; preview ở {out / 'previews'}")


if __name__ == "__main__":
    main()
