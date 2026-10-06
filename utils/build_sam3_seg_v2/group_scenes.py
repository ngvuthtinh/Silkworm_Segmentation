"""
Gom 8.975 ảnh yolo_bbox thành NHÓM CẢNH (cùng con tằm / cùng phiên chụp) để chia split không rò rỉ.

Hai tín hiệu, nối bằng union-find:
  1. Pixel (bền với độ sáng): cosine ảnh xám 24px ≥ 0.90 rồi tương quan chuẩn hoá ảnh xám 96px ≥ 0.95.
  2. Cảnh (bền với việc con tằm bò đi): với 30 láng giềng gần nhất, SIFT + RANSAC ≥ 10 điểm khớp
     VÀ cùng lớp (một phiên chụp = một con = một lớp; ràng buộc này chặn việc nối nhầm thành nhóm khổng lồ).

Chạy (từ gốc dự án):  python utils/build_sam3_seg_v2/group_scenes.py <thư mục làm việc>
Xuất: F24.npy, H96.npy, cand_pairs.npy, cand_inl.npy, scene_groups_th10.json
"""
import collections, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from scene_match import inliers, sift_feats

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
NCC_THR, COS_PRE, K, SIFT_THR = 0.95, 0.90, 30, 10

items = [(sp, p.stem) for sp in ["train", "valid", "test"] for p in sorted(Path(f"data/yolo_bbox/{sp}/images").glob("*.jpg"))]
N = len(items)
paths = [f"data/yolo_bbox/{s}/images/{st}.jpg" for s, st in items]


def global_feats(path: str) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out = []
    for size in (24, 96):
        v = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean(); v /= max(np.linalg.norm(v), 1e-6); out.append(v)
    return out[0], out[1]


with Pool(24) as pool:
    GF = pool.map(global_feats, paths, chunksize=64)
    FE = pool.map(sift_feats, paths, chunksize=64)
F = np.stack([a for a, _ in GF]); H = np.stack([b for _, b in GF])
cls = np.array([1 if st.lower().startswith("healthy") else 0 for _, st in items])

parent = list(range(N))
def find(a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a

# 1) pixel
for i0 in range(0, N, 1024):
    S = F[i0:i0 + 1024] @ F.T
    for i, j in zip(*np.where(S >= COS_PRE)):
        i += i0
        if j > i and float(H[i] @ H[j]) >= NCC_THR:
            parent[find(i)] = find(j)
# 2) SIFT trên K láng giềng gần nhất
cand = set()
for i0 in range(0, N, 1024):
    S = np.maximum(F[i0:i0 + 1024] @ F.T, H[i0:i0 + 1024] @ H.T)
    for r, row in enumerate(S):
        i = i0 + r; row[i] = -9
        for j in np.argpartition(-row, K)[:K]:
            cand.add((min(i, int(j)), max(i, int(j))))
cand = sorted(cand)
with Pool(24) as pool:
    R = np.array(pool.starmap(inliers, [(FE[a], FE[b]) for a, b in cand], chunksize=512))
for (a, b), r in zip(cand, R):
    if r >= SIFT_THR and cls[a] == cls[b]:
        parent[find(a)] = find(b)

groups = collections.defaultdict(list)
for i in range(N):
    groups[find(i)].append(i)
G = list(groups.values())
np.save(OUT / "F24.npy", F); np.save(OUT / "H96.npy", H)
np.save(OUT / "cand_pairs.npy", np.array(cand)); np.save(OUT / "cand_inl.npy", R)
json.dump({"items": items, "groups": G}, open(OUT / "scene_groups_th10.json", "w"))
print(f"{N} ảnh → {len(G)} nhóm cảnh | nhóm lớn nhất {max(map(len, G))}")
