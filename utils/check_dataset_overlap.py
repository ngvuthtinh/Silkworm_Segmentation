#!/usr/bin/env python3
"""
Kiểm tra ảnh của một bộ data (query) có TRÙNG ảnh của các bộ khác (db) không, kể cả khi ảnh query là mosaic / xoay / phóng to / cắt.

Cách làm (không so toàn ảnh vì mosaic ghép nhiều ảnh):
  1. SIFT (tối đa --q-feats điểm cho ảnh query, --db-feats cho ảnh db, ảnh co về cạnh dài 320).
  2. Với mỗi ảnh query: so khớp từng descriptor với TOÀN BỘ descriptor của db trên GPU (ratio test 0.75), mỗi descriptor bỏ phiếu cho ảnh db
     chứa điểm khớp; lấy các ảnh db nhiều phiếu nhất làm ứng viên.
  3. Kiểm chứng từng cặp ứng viên bằng RANSAC (phép đồng dạng: xoay + co giãn + tịnh tiến); cặp có >= --min-inliers điểm đồng nhất là TRÙNG.
  4. Đối chứng độ nhạy: lấy --controls ảnh db, xoay 30° + co 0.7 + cắt rồi dùng làm query; phải tìm lại đúng ảnh gốc.

  python utils/check_dataset_overlap.py --query data/sam3_seg_silkworm_v2i/train/images \
      --db data/legacy/sam3_seg_v3/train/images data/legacy/sam3_seg_v3/valid/images data/legacy/sam3_seg_v3/test/images \
      --out Reports/Meeting_2026-10-30/results/label_audit/v2i_vs_v3_overlap --gpu 1
Chỉ ĐỌC data; ghi báo cáo, matches.csv và vài ảnh cặp trùng vào --out.
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

LONG = 320


def feats(args: tuple[str, int]):
    path, n = args
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    s = LONG / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(im, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], np.float32), d.astype(np.float32)


def extract(paths: list[str], n: int, workers: int):
    with Pool(workers) as p:
        return p.map(feats, [(x, n) for x in paths], chunksize=16)


def transformed(path: str, seed: int) -> str:
    """Ảnh đối chứng: xoay 30°, co 0.7, cắt phần giữa 70% (giả lập tile bị xoay / cắt trong mosaic)."""
    rng = random.Random(seed)
    im = cv2.imread(path)
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.choice([-30, 30, 90, 180]), 0.7)
    out = cv2.warpAffine(im, M, (w, h), borderValue=(0, 0, 0))
    out = out[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    p = f"/tmp/overlap_ctrl_{seed}.jpg"
    cv2.imwrite(p, out)
    return p


@torch.no_grad()
def vote(q_desc: np.ndarray, D: torch.Tensor, Dn: torch.Tensor, owner: torch.Tensor, n_db: int, ratio: float = 0.75, chunk: int = 300_000):
    q = torch.from_numpy(q_desc).cuda()
    qn = (q * q).sum(1, keepdim=True)
    best = torch.full((len(q), 2), float("inf"), device="cuda")
    arg = torch.zeros((len(q),), dtype=torch.long, device="cuda")
    for s in range(0, len(D), chunk):
        d2 = qn + Dn[None, s:s + chunk] - 2 * q @ D[s:s + chunk].T
        v, i = d2.topk(2, dim=1, largest=False)
        cat = torch.cat([best, v], 1)
        top, idx = cat.topk(2, dim=1, largest=False)
        arg = torch.where(idx[:, 0] >= 2, i[:, 0] + s, arg)
        best = top
    ok = best[:, 0].clamp_min(0).sqrt() < ratio * best[:, 1].clamp_min(0).sqrt()
    return torch.bincount(owner[arg[ok]], minlength=n_db).cpu().numpy()


def verify(qk, qd, dk, dd) -> int:
    if len(qd) < 4 or len(dd) < 4:
        return 0
    m = cv2.BFMatcher().knnMatch(qd, dd, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 6:
        return 0
    src = qk[[g.queryIdx for g in good]]
    dst = dk[[g.trainIdx for g in good]]
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    return int(inl.sum()) if inl is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--db", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-feats", type=int, default=300)
    ap.add_argument("--db-feats", type=int, default=150)
    ap.add_argument("--min-votes", type=int, default=6)
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    torch.cuda.set_device(int(a.gpu))
    out = Path(a.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    ext = {".jpg", ".jpeg", ".png"}
    db_paths = sorted(str(p) for d in a.db for p in Path(d).iterdir() if p.suffix.lower() in ext)
    q_paths = sorted(str(p) for p in Path(a.query).iterdir() if p.suffix.lower() in ext)
    print(f"db {len(db_paths)} ảnh, query {len(q_paths)} ảnh", flush=True)
    db = extract(db_paths, a.db_feats, a.workers)
    q = extract(q_paths, a.q_feats, a.workers)
    D = torch.from_numpy(np.concatenate([d for _, d in db])).cuda()
    Dn = (D * D).sum(1)
    owner = torch.from_numpy(np.concatenate([np.full(len(d), i) for i, (_, d) in enumerate(db)])).cuda()
    print(f"descriptor db: {len(D)}", flush=True)

    def search(qk, qd):
        votes = vote(qd, D, Dn, owner, len(db_paths))
        res = []
        for j in np.argsort(-votes)[:5]:
            if votes[j] >= a.min_votes:
                res.append((int(j), int(votes[j]), verify(qk, qd, db[j][0], db[j][1])))
        return res

    # đối chứng độ nhạy
    rng = random.Random(0)
    ctrl = rng.sample(range(len(db_paths)), min(a.controls, len(db_paths)))
    hit = 0
    for k, j in enumerate(ctrl):
        qk, qd = feats((transformed(db_paths[j], k), a.q_feats))
        hit += any(jj == j and inl >= a.min_inliers for jj, _, inl in search(qk, qd))
    print(f"đối chứng: tìm lại đúng ảnh gốc {hit}/{len(ctrl)} (ảnh đã xoay / co / cắt)", flush=True)

    rows, by_q = [], {}
    for n, (path, (qk, qd)) in enumerate(zip(q_paths, q)):
        found = [(j, v, inl) for j, v, inl in search(qk, qd) if inl >= a.min_inliers]
        by_q[path] = found
        for j, v, inl in found:
            rows.append({"query": Path(path).name, "match": db_paths[j], "votes": v, "inliers": inl})
        if n % 300 == 0:
            print(f"  {n}/{len(q_paths)}", flush=True)
    with open(out / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "match", "votes", "inliers"])
        w.writeheader()
        w.writerows(rows)

    def where(p: str) -> str:
        parts = Path(p).parts
        return "/".join(parts[-3:-1])
    n_match = sum(1 for v in by_q.values() if v)
    split_cnt = collections.Counter(where(r["match"]) for r in rows)
    lines = [f"# Độ trùng của `{a.query}` với `{', '.join(a.db)}`", "",
             f"- Phương pháp: SIFT + RANSAC (đồng dạng), trùng nếu ≥ {a.min_inliers} điểm đồng nhất. db: {len(db_paths)} ảnh; query: {len(q_paths)} ảnh.",
             f"- **Đối chứng độ nhạy:** ảnh db bị xoay/co/cắt rồi tìm lại: **{hit}/{len(ctrl)}** tìm lại đúng.",
             f"- **Ảnh query có ≥ 1 ảnh trùng trong db: {n_match}/{len(q_paths)}** ({100 * n_match / len(q_paths):.1f}%)",
             f"- Cặp trùng theo thư mục db: {dict(split_cnt)}"]
    for th in (12, 20, 40):
        lines.append(f"- Số ảnh query có cặp trùng với ≥ {th} điểm đồng nhất: {sum(1 for v in by_q.values() if any(i >= th for _, _, i in v))}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    top = sorted(rows, key=lambda r: -r["inliers"])[:12]
    for k, r in enumerate(top):
        a_im, b_im = cv2.imread(str(Path(a.query) / r["query"])), cv2.imread(r["match"])
        s = 400 / max(a_im.shape[:2] + b_im.shape[:2])
        cv2.imwrite(str(out / "pairs" / f"{k:02d}_inl{r['inliers']}.jpg"),
                    np.hstack([cv2.resize(a_im, (400, 400)), cv2.resize(b_im, (400, 400))]))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
