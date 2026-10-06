"""So khớp cảnh bằng SIFT + RANSAC (bền với việc con tằm di chuyển và thay đổi độ sáng)."""
from __future__ import annotations
import cv2, numpy as np

_sift = None

def sift_feats(path: str, size: int = 320, nfeat: int = 600):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=nfeat)
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)          # bớt phụ thuộc độ sáng
    kp, des = _sift.detectAndCompute(g, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, (des if des is not None else np.zeros((0, 128), np.float32))

def inliers(fa, fb, ratio: float = 0.75) -> int:
    (pa, da), (pb, db) = fa, fb
    if len(da) < 8 or len(db) < 8:
        return 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if len(good) < 8:
        return len(good)
    src = pa[[g.queryIdx for g in good]]; dst = pb[[g.trainIdx for g in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0
