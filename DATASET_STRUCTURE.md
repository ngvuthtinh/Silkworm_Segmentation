# 📊 Cấu Trúc Dataset (Dataset Structure) — Silkworm Segmentation

Tài liệu này mô tả chi tiết toàn bộ các tập dữ liệu đang được quản lý trong thư mục `data/` và `models/silkynet/data/`, định dạng nhãn (annotations), và cách nạp dữ liệu thông qua `src/dataset.py`.

---

## 📁 1. Sơ Đồ Tổng Quan Thư Mục Dữ Liệu

```text
Silkworm_Segmentation/
├── data/
│   ├── Silkworm Diseases.v1i.yolo26/    # Dataset YOLO gốc (Bounding Box + Disease Class)
│   │   ├── train/                       #   images/ và labels/
│   │   ├── valid/                       #   images/ và labels/
│   │   ├── test/                        #   images/ và labels/
│   │   └── data.yaml                    #   Cấu hình class: Grasserie (0), Healthy (1)
│   │
│   ├── Silkworm_SAM3_Segmented/         # Dataset sinh tự động bởi SAM 3 từ YOLO bboxes
│   │   ├── train/                       #   masks/, boundaries/, labels/, visualizations/
│   │   ├── valid/                       #   masks/, boundaries/, labels/, visualizations/
│   │   ├── test/                        #   masks/, boundaries/, labels/, visualizations/
│   │   └── dataset_segmentation.yaml
│   │
│   ├── silkworm_mixed_dataset/          # Dataset hỗn hợp (Multi-silkworm + SAM3 Single)
│   │   ├── train/                       #   images/, labels/, masks/
│   │   ├── valid/                       #   images/, labels/, masks/
│   │   ├── test/                        #   images/, labels/, masks/
│   │   └── dataset_summary.json         #   Thống kê số lượng mẫu train/val/test
│   │
│   └── test/                            # Thư mục chứa ảnh mẫu để test nhanh inference
│       ├── image.png
│       └── image copy.png
│
├── models/
│   └── silkynet/data/                   # Dataset gốc của Silkynet
│       ├── larvaTrain/                  #   Dữ liệu huấn luyện gốc Silkynet
│       ├── larvaTest/                   #   Dữ liệu kiểm thử gốc Silkynet
│       └── 20221127/                    #   Dữ liệu thực địa bổ sung
│
└── utils/
    ├── sam3_label_from_yolo.py          # Script sinh mask từ SAM 3 dựa trên YOLO bbox
    └── yolo_bbox_to_masks.py            # Script convert bbox YOLO thành binary mask
```

---

## 🏷️ 2. Chi Tiết Từng Tập Dữ Liệu

### 2.1. `data/Silkworm Diseases.v1i.yolo26` (YOLO Dataset)
- **Mục đích**: Huấn luyện phân loại bệnh tằm và xác định vị trí qua bounding box.
- **Số lớp phân loại (Classes)**: 2 lớp
  - `0`: **Grasserie** (Bệnh mủ tằm do virus NPV gây ra)
  - `1`: **Healthy** (Tằm khoẻ mạnh)
- **Định dạng nhãn**: YOLO standard format (`.txt` tương ứng với mỗi file ảnh)
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
  *(Các toạ độ được chuẩn hoá trong khoảng `[0.0, 1.0]`)*

---

### 2.2. `data/Silkworm_SAM3_Segmented` (SAM 3 Pseudo-Masks)
- **Mục đích**: Tập nhãn phân đoạn pixel (instance segmentation) độ chính xác cao được sinh tự động bởi **SAM 3 (Segment Anything Model 3)** thông qua `utils/sam3_label_from_yolo.py`.
- **Cấu trúc mỗi tập (train/valid/test)**:
  - `masks/`: File ảnh PNG đơn kênh (binary mask 0 = background, 255 = foreground tằm).
  - `boundaries/`: Ảnh biên (contour edge 2px) phục vụ các thuật toán boundary-guided / watershed.
  - `labels/`: Nhãn polygon YOLO segmentation format.
  - `visualizations/`: Ảnh overlay trực quan hoá kết quả mask trên ảnh gốc để inspect chất lượng.

---

### 2.3. `data/silkworm_mixed_dataset` (Mixed Dataset)
- **Mục đích**: Tập dữ liệu kết hợp chuẩn hóa dùng cho kiến trúc Multi-Task Segmentation & Classification (như `segfirst_vmunet`).
- **Thống kê (`dataset_summary.json`)**:
  - **Train**: 599 samples (99 ảnh nhiều tằm legacy + 500 ảnh tằm đơn SAM3)
  - **Valid**: 110 samples (10 ảnh nhiều tằm legacy + 100 ảnh tằm đơn SAM3)
  - **Test**: 169 samples (69 ảnh nhiều tằm legacy + 100 ảnh tằm đơn SAM3)
- **Cấu trúc mỗi split**:
  - `images/`: Ảnh RGB gốc.
  - `masks/`: Mặt nạ nhị phân ground-truth tương ứng (`.png`).
  - `labels/`: Nhãn YOLO cho phân loại / phát hiện.

---

### 2.4. `models/silkynet/data` (Silkynet Original Data)
- Bộ dữ liệu ban đầu phục vụ bài toán đếm số lượng tằm và trích xuất đường bao (contour detection) của tác giả Silkynet:
  - `larvaTrain/`: Ảnh chụp cụm tằm nuôi mật độ dày trên nong kèm mask.
  - `larvaTest/`: Tập kiểm thử phân đoạn tằm.

---

## 🔄 3. Cách Nạp Dữ Liệu Trong Code (`src/dataset.py`)

Tất cả các mô hình và thử nghiệm đều dùng chung DataLoader từ module `src/dataset.py`:

### 3.1. Phân đoạn ảnh (`SilkynetSegDataset`)
Dùng trong Phase 1 (Pretrain Segmentation):
```python
from src.dataset import SilkynetSegDataset

train_dataset = SilkynetSegDataset(
    image_dir="data/silkworm_mixed_dataset/train/images",
    mask_dir="data/silkworm_mixed_dataset/train/masks",
    image_size=(128, 128),
    augment=True,
)
```

### 3.2. Phân loại bệnh (`YOLOClsDataset`)
Dùng trong Phase 2 (Multi-Task Fine-Tuning):
```python
from src.dataset import YOLOClsDataset

cls_dataset = YOLOClsDataset(
    yolo_dir="data/Silkworm Diseases.v1i.yolo26",
    split="train",
    image_size=(128, 128),
    augment=True,
)
```

---

## 🛠️ 4. Công Cụ Xử Lý & Chuẩn Hóa Dữ Liệu (`utils/`)

### Tự động tạo Mask bằng SAM 3:
Nếu có thêm ảnh mới kèm file nhãn YOLO bounding box, chạy lệnh sau để sinh mask tự động:
```bash
python utils/sam3_label_from_yolo.py \
    --input-yaml "data/Silkworm Diseases.v1i.yolo26/data.yaml" \
    --output-dir "data/Silkworm_SAM3_Segmented" \
    --prompt "silkworm" \
    --device cuda
```

### Chuyển đổi BBox sang Mask cơ bản (Bounding-box to Mask):
```bash
python -m utils.yolo_bbox_to_masks \
    --input-root "data/Silkworm Diseases.v1i.yolo26" \
    --output-root "data/converted"
```

---

## 🛡️ 5. Quy Tắc Quản Lý Dữ Liệu

1. **Dữ liệu gốc bất biến**: Không trực tiếp sửa hay xoá file trong `Silkworm Diseases.v1i.yolo26/` hay `models/silkynet/data/`.
2. **Không commit ảnh lớn lên Git**: Các thư mục dữ liệu lớn (`data/`, `runs/`) được quản lý bằng `.gitignore`.
3. **Đồng bộ tên file**: Ảnh trong thư mục `images/` và mặt nạ trong `masks/` luôn giữ chung stem tên file (ví dụ: `image_001.jpg` tương ứng với `image_001.png`).
