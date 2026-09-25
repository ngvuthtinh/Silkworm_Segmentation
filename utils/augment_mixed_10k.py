#!/usr/bin/env python3
"""
utils/augment_mixed_10k.py

Hệ thống Data Augmentation thế hệ mới cho bài toán Phân đoạn Sâu tằm (Silkworm Segmentation).
Mục tiêu: Đạt 10.000 mẫu đa dạng, chất lượng cao cho tập huấn luyện.

Tính năng nổi bật:
  1. Đồng bộ hóa không gian (Spatial Synchronization) 100% giữa Ảnh, Mask và Polygon Labels YOLO.
  2. Ràng buộc bệnh lý nghiêm ngặt (Pathology Constraint): Giữ nguyên màu da hoại tử (Grasserie) và da trắng (Healthy).
  3. [Nhóm 4] Giả lập phiến lá dâu che thân tằm (Leaf Occlusion) với kết cấu lá tự nhiên, tự động khấu trừ mask.
  4. [Nhóm 5] Giả lập tằm nằm đè lên nhau (Silkworm Overlap Copy-Paste) với thuật toán giải quyết đè lớp (Occlusion Resolution).
"""

from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# =====================================================================
# 1. QUẢN LÝ ĐỐI TƯỢNG TẰM (SILKWORM OBJECT BANK)
# =====================================================================

@dataclass
class SilkwormCutout:
    """Đại diện cho một cá thể tằm được cắt ra kèm mask alpha và class id."""
    image_crop: np.ndarray  # HxWx3 (BGR)
    mask_crop: np.ndarray   # HxW (uint8, 0 hoặc 255)
    class_id: int           # 0: Grasserie, 1: Healthy
    width: int
    height: int


class SilkwormObjectBank:
    """
    Kho chứa các cá thể tằm đơn lẻ được trích xuất từ dữ liệu SAM3/YOLO.
    Dùng cho kỹ thuật Copy-Paste mô phỏng tằm nằm đè lên nhau.
    """

    def __init__(self, sam3_mask_dir: Path, yolo_img_dir: Path, max_objects: int = 1500, seed: int = 42):
        self.objects: List[SilkwormCutout] = []
        self._load_bank(sam3_mask_dir, yolo_img_dir, max_objects, seed)

    def _load_bank(self, mask_dir: Path, img_dir: Path, max_objects: int, seed: int):
        random.seed(seed)
        mask_files = sorted(list(mask_dir.glob("*.png")))
        if not mask_files:
            print(f"⚠️ Cảnh báo: Không tìm thấy mask trong {mask_dir}")
            return

        random.shuffle(mask_files)
        print(f"📦 Đang khởi tạo Object Bank từ {len(mask_files)} mẫu nguồn...")

        for m_path in mask_files:
            if len(self.objects) >= max_objects:
                break

            stem = m_path.stem
            img_p = img_dir / f"{stem}.jpg"
            if not img_p.exists():
                img_p = img_dir / f"{stem}.png"
            if not img_p.exists():
                continue

            mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Kiểm tra contour tằm
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Lấy contour lớn nhất (thân tằm chính)
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)
            if area < 1200:  # Quá nhỏ, bỏ qua
                continue

            x, y, w, h = cv2.boundingRect(largest_cnt)
            pad = 6
            h_img, w_img = mask.shape
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            img = cv2.imread(str(img_p))
            if img is None:
                continue

            img_crop = img[y1:y2, x1:x2].copy()
            mask_crop = binary[y1:y2, x1:x2].copy()

            # Class ID: 0: Grasserie, 1: Healthy
            class_id = 1 if "healthy" in stem.lower() else 0

            self.objects.append(
                SilkwormCutout(
                    image_crop=img_crop,
                    mask_crop=mask_crop,
                    class_id=class_id,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )

        print(f"✅ Đã nạp thành công {len(self.objects)} đối tượng tằm chất lượng cao vào Object Bank!")

    def sample(self) -> Optional[SilkwormCutout]:
        if not self.objects:
            return None
        return random.choice(self.objects)


# =====================================================================
# 2. GIẢ LẬP LÁ DÂU CHE (LEAF OCCLUSION — NHÓM 4)
# =====================================================================

def generate_mulberry_leaf(canvas_size: Tuple[int, int], target_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo ra một phiến lá dâu nhân tạo có kết cấu tự nhiên che lên một phần vùng thân tằm.
    Trả về:
      - leaf_bgr: ảnh màu BGR của phiến lá dâu
      - leaf_alpha: mặt nạ nhị phân của phiến lá (255 tại vùng lá)
    """
    h_canvas, w_canvas = canvas_size
    bx, by, bw, bh = target_bbox

    leaf_bgr = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
    leaf_alpha = np.zeros((h_canvas, w_canvas), dtype=np.uint8)

    # Đặt tâm phiến lá ở rìa hoặc góc của con tằm để che một phần
    center_x = int(bx + bw * random.uniform(0.1, 0.9))
    center_y = int(by + bh * random.uniform(0.1, 0.9))

    # Kích thước lá tỉ lệ với kích thước thân tằm
    leaf_w = int(max(35, bw * random.uniform(0.35, 0.75)))
    leaf_h = int(max(25, bh * random.uniform(0.35, 0.75)))
    angle = random.uniform(0, 180)

    # Tạo hình dạng lá dâu bằng đa giác biến thiên (phiến lá có cuống và chóp nhọn)
    num_points = 16
    pts = []
    for i in range(num_points):
        theta = 2.0 * math.pi * i / num_points
        # Hàm elip kết hợp gợn sóng mô phỏng viền lá dâu
        r_base = 0.5 * (leaf_w * leaf_h) / math.sqrt((leaf_h * math.cos(theta)) ** 2 + (leaf_w * math.sin(theta)) ** 2 + 1e-5)
        jitter = random.uniform(0.88, 1.08)
        if math.pi * 0.7 < theta < math.pi * 1.3:
            jitter *= 1.25

        rx = r_base * jitter * math.cos(theta)
        ry = r_base * jitter * math.sin(theta)

        rad = math.radians(angle)
        rot_x = rx * math.cos(rad) - ry * math.sin(rad) + center_x
        rot_y = rx * math.sin(rad) + ry * math.cos(rad) + center_y

        pts.append([int(rot_x), int(rot_y)])

    pts_arr = np.array([pts], dtype=np.int32)
    cv2.fillPoly(leaf_alpha, pts_arr, 255)

    # Làm mềm biên lá
    leaf_alpha = cv2.GaussianBlur(leaf_alpha, (3, 3), 0)

    # Tạo màu xanh lá dâu tự nhiên (BGR): B=[22..40], G=[90..140], R=[28..55]
    base_b = random.randint(22, 40)
    base_g = random.randint(90, 140)
    base_r = random.randint(28, 55)

    leaf_texture = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
    leaf_texture[:] = [base_b, base_g, base_r]

    noise = np.random.randint(-12, 12, (h_canvas, w_canvas, 3), dtype=np.int16)
    leaf_texture = np.clip(leaf_texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Vẽ vài đường gân lá mảnh
    cv2.polylines(leaf_texture, pts_arr, False, (max(0, base_b - 10), max(0, base_g - 20), max(0, base_r - 10)), 1)

    leaf_bgr[leaf_alpha > 127] = leaf_texture[leaf_alpha > 127]

    return leaf_bgr, leaf_alpha


# =====================================================================
# 3. GIẢ LẬP TẰM NẰM ĐÈ LÊN NHAU (SILKWORM OVERLAP — NHÓM 5)
# =====================================================================

def paste_silkworm_with_overlap(
    bg_image: np.ndarray,
    bg_mask: np.ndarray,
    instances: List[Tuple[np.ndarray, int]],  # List of (instance_mask, class_id)
    cutout: SilkwormCutout,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, int]]]:
    """
    Dán một con tằm mới lên ảnh nền, có thể đè lên con tằm đã có.
    Giải quyết đè lớp:
      - Vùng tằm mới sẽ ghi đè lên ảnh nền.
      - Mask của tằm mới được cộng vào tổng thể.
      - Vùng đè của các con tằm nằm dưới bị cắt đi (Occlusion Resolution).
    """
    h_bg, w_bg = bg_image.shape[:2]
    crop_img = cutout.image_crop
    crop_mask = cutout.mask_crop

    scale = random.uniform(0.85, 1.15)
    rot_angle = random.uniform(0, 360)
    flip_lr = random.choice([True, False])

    if flip_lr:
        crop_img = cv2.flip(crop_img, 1)
        crop_mask = cv2.flip(crop_mask, 1)

    ch, cw = crop_img.shape[:2]
    m_rot = cv2.getRotationMatrix2D((cw / 2, ch / 2), rot_angle, scale)
    cos = np.abs(m_rot[0, 0])
    sin = np.abs(m_rot[0, 1])
    new_w = int((ch * sin) + (cw * cos))
    new_h = int((ch * cos) + (cw * sin))
    m_rot[0, 2] += (new_w / 2) - cw / 2
    m_rot[1, 2] += (new_h / 2) - ch / 2

    trans_img = cv2.warpAffine(crop_img, m_rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    trans_mask = cv2.warpAffine(crop_mask, m_rot, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    _, trans_mask = cv2.threshold(trans_mask, 127, 255, cv2.THRESH_BINARY)

    # Ưu tiên đặt đè lên vùng tằm hiện có nếu có mask
    bg_tamm_pts = np.argwhere(bg_mask > 127)
    if len(bg_tamm_pts) > 100 and random.random() < 0.65:
        anchor_y, anchor_x = bg_tamm_pts[random.randint(0, len(bg_tamm_pts) - 1)]
        pos_x = anchor_x - new_w // 2 + random.randint(-25, 25)
        pos_y = anchor_y - new_h // 2 + random.randint(-25, 25)
    else:
        pos_x = random.randint(10, max(11, w_bg - new_w - 10))
        pos_y = random.randint(10, max(11, h_bg - new_h - 10))

    src_x1 = max(0, -pos_x)
    src_y1 = max(0, -pos_y)
    src_x2 = min(new_w, w_bg - pos_x)
    src_y2 = min(new_h, h_bg - pos_y)

    dst_x1 = max(0, pos_x)
    dst_y1 = max(0, pos_y)
    dst_x2 = min(w_bg, pos_x + new_w)
    dst_y2 = min(h_bg, pos_y + new_h)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return bg_image, bg_mask, instances

    sub_trans_img = trans_img[src_y1:src_y2, src_x1:src_x2]
    sub_trans_mask = trans_mask[src_y1:src_y2, src_x1:src_x2]

    new_instance_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    new_instance_mask[dst_y1:dst_y2, dst_x1:dst_x2] = sub_trans_mask

    # Cắt bỏ phần mask của các con tằm cũ bị đè lên
    updated_instances = []
    inv_new_mask = cv2.bitwise_not(new_instance_mask)
    for old_mask, old_cls in instances:
        clipped_old_mask = cv2.bitwise_and(old_mask, inv_new_mask)
        if cv2.countNonZero(clipped_old_mask) > 100:
            updated_instances.append((clipped_old_mask, old_cls))

    updated_instances.append((new_instance_mask, cutout.class_id))

    out_img = bg_image.copy()
    canvas_new_mask = (new_instance_mask > 127)
    out_img[canvas_new_mask] = sub_trans_img[sub_trans_mask > 127]

    out_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    for inst_m, _ in updated_instances:
        out_mask = cv2.bitwise_or(out_mask, inst_m)

    return out_img, out_mask, updated_instances


# =====================================================================
# 4. TRÍCH XUẤT POLYGON YOLO CHO TỪNG INSTANCE
# =====================================================================

def instances_to_yolo_polygons(instances: List[Tuple[np.ndarray, int]], min_area: float = 80.0) -> List[str]:
    """
    Sinh các dòng nhãn YOLO segmentation từ danh sách (instance_mask, class_id).
    Mỗi con tằm có 1 dòng riêng: <class_id> x1 y1 x2 y2 ...
    """
    lines = []
    for inst_mask, cls_id in instances:
        h, w = inst_mask.shape[:2]
        _, binary = cv2.threshold(inst_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            coords = []
            for pt in approx:
                x, y = pt[0]
                norm_x = np.clip(x / float(w), 0.0, 1.0)
                norm_y = np.clip(y / float(h), 0.0, 1.0)
                coords.append(f"{norm_x:.6f}")
                coords.append(f"{norm_y:.6f}")

            lines.append(f"{cls_id} " + " ".join(coords))

    return lines


# =====================================================================
# 5. PIPELINE AUGMENTATION TỔNG HỢP
# =====================================================================

def get_base_spatial_photometric_augmenter() -> A.Compose:
    """Biến đổi hình học + quang học an toàn (Nhóm 1, 2, 3) có kiểm soát bệnh học."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.04, 0.04),
                rotate=(-35, 35),
                interpolation=cv2.INTER_LINEAR,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.6,
            ),
            A.HueSaturationValue(
                hue_shift_limit=6,   # Giới hạn bệnh lý an toàn
                sat_shift_limit=14,
                val_shift_limit=12,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=0.6),
                    A.MotionBlur(blur_limit=(3, 5), p=0.4),
                ],
                p=0.25,
            ),
            A.GaussNoise(p=0.2),
        ],
        is_check_shapes=False,
    )


def generate_single_sample(
    sample_idx: int,
    base_sample: Tuple[Path, Path, int],
    bank: SilkwormObjectBank,
    augmenter: A.Compose,
    target_img_dir: Path,
    target_mask_dir: Path,
    target_lbl_dir: Path,
    preview_dir: Optional[Path] = None,
) -> bool:
    """
    Tạo một mẫu dữ liệu mới kết hợp ngẫu nhiên các kỹ thuật.
    """
    img_p, mask_p, base_class_id = base_sample

    img = cv2.imread(str(img_p))
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False

    if img.shape[:2] != (640, 640):
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Biến đổi Hình học & Quang học cơ bản
    augmented = augmenter(image=img, mask=mask)
    cur_img = augmented["image"]
    cur_mask = augmented["mask"]
    _, cur_mask = cv2.threshold(cur_mask, 127, 255, cv2.THRESH_BINARY)

    instances = [(cur_mask.copy(), base_class_id)] if cv2.countNonZero(cur_mask) > 100 else []

    # 2. Quyết định chế độ tăng cường nâng cao:
    mode_rand = random.random()

    # [NHÓM 5]: Tằm đè lên nhau (Copy-Paste)
    if mode_rand < 0.50 and bank.objects:
        num_pastes = random.choice([1, 2])
        for _ in range(num_pastes):
            sample_cutout = bank.sample()
            if sample_cutout:
                cur_img, cur_mask, instances = paste_silkworm_with_overlap(
                    cur_img, cur_mask, instances, sample_cutout
                )

    # [NHÓM 4]: Giả lập lá dâu che (Leaf Occlusion)
    if (0.35 <= mode_rand < 0.85) and cv2.countNonZero(cur_mask) > 200:
        contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            target_cnt = random.choice(contours)
            x, y, w, h = cv2.boundingRect(target_cnt)
            leaf_bgr, leaf_alpha = generate_mulberry_leaf((640, 640), (x, y, w, h))

            leaf_mask_bool = leaf_alpha > 127
            cur_img[leaf_mask_bool] = leaf_bgr[leaf_mask_bool]

            inv_leaf = cv2.bitwise_not(leaf_alpha)
            cur_mask = cv2.bitwise_and(cur_mask, inv_leaf)

            updated_instances = []
            for inst_m, cls_id in instances:
                cut_inst = cv2.bitwise_and(inst_m, inv_leaf)
                if cv2.countNonZero(cut_inst) > 80:
                    updated_instances.append((cut_inst, cls_id))
            instances = updated_instances

    if cv2.countNonZero(cur_mask) < 150 or not instances:
        return False

    out_stem = f"silkworm_aug10k_{sample_idx:06d}"
    out_img_path = target_img_dir / f"{out_stem}.jpg"
    out_mask_path = target_mask_dir / f"{out_stem}.png"
    out_lbl_path = target_lbl_dir / f"{out_stem}.txt"

    cv2.imwrite(str(out_img_path), cur_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_mask_path), cur_mask)

    yolo_lines = instances_to_yolo_polygons(instances)
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n")

    if preview_dir and sample_idx < 50:
        overlay = cur_img.copy()
        for inst_m, cls_id in instances:
            color = [0, 69, 255] if cls_id == 0 else [0, 255, 128]
            m_bool = inst_m > 127
            overlay[m_bool] = (overlay[m_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
            cnts, _ = cv2.findContours(inst_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color, 2)
        cv2.imwrite(str(preview_dir / f"{out_stem}_preview.jpg"), overlay)

    return True


# =====================================================================
# 6. THU THẬP NGUỒN DỮ LIỆU VÀ CHẠY TOÀN BỘ QUY TRÌNH
# =====================================================================

def collect_all_source_samples() -> List[Tuple[Path, Path, int]]:
    """
    Thu thập toàn bộ dữ liệu từ cả 2 nguồn:
      1. Silkworm_SAM3_Segmented (kèm ảnh từ Silkworm_Yolo_BoundingBox)
      2. Silkworm_Silkynet_Segmented (larvaTrain + output20221127)
    """
    samples: List[Tuple[Path, Path, int]] = []

    # 1. Nguồn SAM3
    sam3_mask_dir = Path("data/Silkworm_SAM3_Segmented/train/masks")
    yolo_img_dir = Path("data/Silkworm_Yolo_BoundingBox/train/images")

    if sam3_mask_dir.exists() and yolo_img_dir.exists():
        for m_p in sam3_mask_dir.glob("*.png"):
            stem = m_p.stem
            img_p = yolo_img_dir / f"{stem}.jpg"
            if not img_p.exists():
                img_p = yolo_img_dir / f"{stem}.png"
            if img_p.exists():
                cls_id = 1 if "healthy" in stem.lower() else 0
                samples.append((img_p, m_p, cls_id))

    # 2. Nguồn Silkynet (larvaTrain & output20221127)
    silkynet_pairs = [
        (Path("data/Silkworm_Silkynet_Segmented/larvaTrain/img"), Path("data/Silkworm_Silkynet_Segmented/larvaTrain/label")),
        (Path("data/Silkworm_Silkynet_Segmented/output20221127/JPEGImages"), Path("data/Silkworm_Silkynet_Segmented/output20221127/SegmentationClassPNG")),
    ]

    for s_img_dir, s_mask_dir in silkynet_pairs:
        if s_img_dir.exists() and s_mask_dir.exists():
            for img_p in s_img_dir.glob("*.*"):
                if img_p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                stem = img_p.stem
                for ext in [".png", ".jpg"]:
                    cand = s_mask_dir / f"{stem}{ext}"
                    if cand.exists():
                        samples.append((img_p, cand, 1))
                        break

    print(f"📊 Tổng số mẫu dữ liệu gốc thu thập được: {len(samples)} mẫu")
    return samples


def _worker_process_batch(
    worker_id: int,
    indices: List[int],
    source_samples: List[Tuple[Path, Path, int]],
    bank: SilkwormObjectBank,
    target_img_dir: Path,
    target_mask_dir: Path,
    target_lbl_dir: Path,
    preview_dir: Optional[Path],
    seed: int,
) -> int:
    """Worker sinh một batch mẫu."""
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    augmenter = get_base_spatial_photometric_augmenter()

    local_success = 0
    for idx in indices:
        base_sample = random.choice(source_samples)
        ok = generate_single_sample(
            sample_idx=idx,
            base_sample=base_sample,
            bank=bank,
            augmenter=augmenter,
            target_img_dir=target_img_dir,
            target_mask_dir=target_mask_dir,
            target_lbl_dir=target_lbl_dir,
            preview_dir=preview_dir if idx <= 60 else None,
        )
        if ok:
            local_success += 1
    return local_success


def main():
    parser = argparse.ArgumentParser(description="Sinh 10.000 ảnh Silkworm Augmentation nâng cao.")
    parser.add_argument("--goal", type=int, default=10000, help="Mục tiêu số lượng ảnh (mặc định: 10000)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/Silkworm_mixed_dataset_10k/train"), help="Thư mục xuất kết quả")
    parser.add_argument("--preview", action="store_true", default=True, help="Tạo ảnh xem trước mask overlay")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--workers", type=int, default=8, help="Số worker tiến trình CPU song song")
    parser.add_argument("--dry-run", type=int, default=None, help="Chạy thử nghiệm N mẫu để kiểm tra trước")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    target_img_dir = args.output_dir / "images"
    target_mask_dir = args.output_dir / "masks"
    target_lbl_dir = args.output_dir / "labels"
    preview_dir = (args.output_dir / "aug_previews") if args.preview else None

    for d in [target_img_dir, target_mask_dir, target_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thu thập toàn bộ dữ liệu nguồn
    source_samples = collect_all_source_samples()
    if not source_samples:
        raise RuntimeError("Không tìm thấy dữ liệu nguồn từ SAM3 hoặc Silkynet!")

    # 2. Xây dựng Object Bank
    sam3_mask_dir = Path("data/Silkworm_SAM3_Segmented/train/masks")
    yolo_img_dir = Path("data/Silkworm_Yolo_BoundingBox/train/images")
    bank = SilkwormObjectBank(sam3_mask_dir, yolo_img_dir, max_objects=1200, seed=args.seed)

    total_target = args.dry_run if args.dry_run is not None else args.goal
    num_workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"\n🚀 Bắt đầu sinh {total_target} ảnh vào: {args.output_dir} sử dụng {num_workers} CPU workers...")

    # Phân bổ danh sách index cho các workers
    all_indices = list(range(1, total_target + 1))
    chunks = np.array_split(all_indices, num_workers)

    tasks = []
    with mp.Pool(processes=num_workers) as pool:
        for w_id, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            tasks.append(
                pool.apply_async(
                    _worker_process_batch,
                    args=(
                        w_id,
                        chunk.tolist(),
                        source_samples,
                        bank,
                        target_img_dir,
                        target_mask_dir,
                        target_lbl_dir,
                        preview_dir,
                        args.seed,
                    ),
                )
            )

        # Chờ và cập nhật tiến trình
        for t in tqdm(tasks, desc="Tiến độ các Batch Workers"):
            t.get()

    actual_images = len(list(target_img_dir.glob("*.jpg")))
    actual_masks = len(list(target_mask_dir.glob("*.png")))
    actual_labels = len(list(target_lbl_dir.glob("*.txt")))

    print("\n🎉 HOÀN THÀNH AUGMENTATION!")
    print(f"📁 Dữ liệu được lưu tại: {args.output_dir}")
    print(f"   - Images: {actual_images}")
    print(f"   - Masks:  {actual_masks}")
    print(f"   - Labels: {actual_labels}")
    if preview_dir:
        print(f"   - Previews: {preview_dir}")


if __name__ == "__main__":
    main()
