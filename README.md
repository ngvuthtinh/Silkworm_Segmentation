# 🐛 Silkworm Segmentation & Disease Detection Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Task-Instance_Segmentation-blue?style=flat)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dự án nghiên cứu và phát triển các phương pháp Học sâu (Deep Learning) & Thị giác máy tính (Computer Vision) tiên tiến nhằm **Phân đoạn con tằm (Silkworm Segmentation)**, **Phát hiện bệnh ở tằm (Silkworm Disease Detection)**, và **Đếm số lượng con tằm (Silkworm Counting)** trong hình ảnh thực tế.

---

## 📌 Giới thiệu dự án

Con tằm là vật nuôi có giá trị kinh tế cao trong ngành dệt may tơ lụa. Tuy nhiên, việc quản lý mật độ và phát hiện sớm các loại bệnh ở tằm (như bệnh Grasserie, Flacherie, v.v.) bằng thủ công tốn nhiều chi phí và dễ sai sót. 

Hệ thống **Silkworm Segmentation Framework** tổng hợp nhiều phương pháp SOTA (State-of-the-Art) giúp:
- Tách bạch ranh giới giữa các con tằm nằm dính liền nhau (Instance Segmentation).
- Nhận diện tổn thương và chẩn đoán bệnh trên thân tằm.
- Tự động hóa quá trình đếm số lượng tằm trong khay nuôi.

---

## ✨ Các phương pháp hỗ trợ (Supported Methods)

Dự án được cấu trúc modular theo thư mục `methods/`, bao gồm 4 phương pháp chính:

### 1. 🛰️ VM-UNet (Visual State Space / Mamba Architecture)
- **Vị trí**: [`methods/vmunet/`](methods/vmunet/)
- **Mô tả**: Kiến trúc phân đoạn kết hợp **Mamba (State Space Model - SSM)** và **U-Net**, cho khả năng học ngữ cảnh toàn cục (long-range dependency) với độ phức tạp tuyến tính, giúp phân đoạn tằm chính xác cao với tốc độ nhanh.

### 2. 🌊 Deep Watershed Pipeline (Attention U-Net + Watershed)
- **Vị trí**: [`methods/deep_watershed/`](methods/deep_watershed/)
- **Mô tả**: Kết hợp mô hình **Attention U-Net** để dự đoán các bản đồ ranh giới (Boundary Maps) 2px, sau đó áp dụng thuật toán **Watershed Transformation** để tách biệt chính xác các cá thể tằm nằm đè đè/dính liền nhau. Đi kèm công cụ xem trực quan GUI tương tác ([`interactive_viewer.py`](methods/deep_watershed/interactive_viewer.py)).

### 3. 📦 Box-Guided Watershed
- **Vị trí**: [`methods/box_guided_watershed/`](methods/box_guided_watershed/)
- **Mô tả**: Giải pháp học giám sát yếu (Weakly Supervised Segmentation) dựa trên khung bao Bounding Box (YOLO) để tự động sinh ra Pseudo-Masks và nhãn ranh giới, giảm thiểu chi phí gán nhãn thủ công.

### 4. 🔍 Silkynet
- **Vị trí**: [`methods/silkynet/`](methods/silkynet/)
- **Mô tả**: Mô hình U-Net chuyên biệt hóa cho tác vụ xử lý ảnh tằm, hỗ trợ chuyển đổi dữ liệu nhãn LabelMe sang VOC Mask, dự đoán mask và đếm số lượng đường viền/con tằm (Contour Counting).

---

## 📁 Cấu trúc thư mục (Directory Structure)

```text
Silkworm_Segmentation/
├── methods/
│   ├── vmunet/                     # Phương pháp VM-UNet (Mamba State Space)
│   │   ├── models/                 # Kiến trúc mạng VMUNet & VMamba
│   │   ├── configs/                # Cấu hình tham số huấn luyện
│   │   ├── train.py                # Script huấn luyện VM-UNet
│   │   └── predict.py              # Script dự đoán & trực quan hóa
│   ├── deep_watershed/             # Phương pháp Attention U-Net + Watershed
│   │   ├── model.py                # Mô hình Attention U-Net & Dice Loss
│   │   ├── boundary_pipeline.py    # Pipeline sinh nhãn ranh giới & huấn luyện
│   │   ├── instance_pipeline.py    # Engine phân đoạn 4 bước
│   │   └── interactive_viewer.py   # Giao diện GUI trực quan hóa kết quả
│   ├── box_guided_watershed/       # Phân đoạn dựa trên Bounding Box
│   │   ├── generate_pseudo_masks.py# Sinh mask giả từ BBox
│   │   ├── train.py                # Huấn luyện mô hình
│   │   └── infer.py                # Chạy suy luận
│   └── silkynet/                   # Phương pháp Silkynet & Contour Counting
│       ├── Silkynet.py             # Kiến trúc mạng Silkynet
│       ├── Count_contours.py       # Script đếm đường viền/con tằm
│       └── labelme2voc_mask.py     # Chuyển nhãn LabelMe sang VOC
├── utils/
│   └── yolo_bbox_to_masks.py       # Công cụ chuyển BBox YOLO sang Mask & Boundary
├── data/                           # Thư mục lưu dữ liệu (xem chi tiết ở DATASET_STRUCTURE.md)
│   ├── images/                     # Ảnh đầu vào RGB
│   ├── masks/                      # Nhãn Mask nhị phân
│   └── boundaries/                 # Nhãn viền Boundary
├── logs/                           # Nhật ký huấn luyện
├── DATASET_STRUCTURE.md            # Hướng dẫn chi tiết cấu trúc dữ liệu
└── .gitignore                      # Cấu hình bỏ qua file rác / checkpoints
```

---

## ⚙️ Cài đặt (Installation)

### 1. Yêu cầu hệ thống
- Python >= 3.10
- PyTorch >= 2.0 (Khuyến nghị sử dụng GPU CUDA)

### 2. Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/ngvuthtinh/Silkworm_Segmentation.git
cd Silkworm_Segmentation

# Tạo và kích hoạt môi trường ảo (Virtual Environment)
python3 -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# .venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện phụ thuộc
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib pillow scikit-image scipy tqdm yolo-bokeh labelme
```

---

## 🚀 Hướng dẫn sử dụng (Usage Guide)

### 1. Chuyển đổi dữ liệu nhãn YOLO sang Mask & Boundary Maps
Nếu bạn sử dụng dataset định dạng YOLO Bounding Box:
```bash
python -m utils.yolo_bbox_to_masks \
    --input-root "data/Silkworm Diseases.v1i.yolo26" \
    --output-root "data/converted"
```

---

### 2. Sử dụng phương pháp VM-UNet (Mamba)

- **Huấn luyện mô hình (Training):**
```bash
python methods/vmunet/train.py
```

- **Chạy dự đoán & Trực quan hóa (Inference & Visualization):**
```bash
python methods/vmunet/predict.py \
    --image-dir data/val/images \
    --mask-dir data/val/masks \
    --output-dir methods/vmunet/visualizations \
    --max-images 20
```

---

### 3. Sử dụng phương pháp Deep Watershed Pipeline

- **Bước 1: Tạo bản đồ ranh giới (Boundary Maps):**
```bash
python -m methods.deep_watershed.boundary_pipeline prepare-labels \
    --mask-dir data/masks \
    --output-dir data/boundaries
```

- **Bước 2: Huấn luyện mô hình Attention U-Net:**
```bash
python -m methods.deep_watershed.boundary_pipeline train \
    --image-dir data/images \
    --boundary-dir data/boundaries \
    --save-path data/checkpoints/boundary_model.pt
```

- **Bước 3: Chạy giao diện tương tác (GUI / Terminal Interactive Viewer):**
```bash
python -m methods.deep_watershed.interactive_viewer \
    --image-dir "data/test/images" \
    --checkpoint data/checkpoints/boundary_model.pt
```

---

### 4. Sử dụng phương pháp Silkynet & Đếm con tằm

- **Đếm số lượng con tằm theo đường viền (Contour Counting):**
```bash
python methods/silkynet/Count_contours.py
```

---

## 📊 Quản lý Dữ liệu & Kết quả (Data & Checkpoints)

Vui lòng tham khảo chi tiết tại [`DATASET_STRUCTURE.md`](DATASET_STRUCTURE.md).

- Thư mục `data/`, `results/`, `visualizations/`, và các file trọng số lớn (`*.pth`, `*.pt`) mặc định được bỏ qua trong [.gitignore](.gitignore) để giữ cho bộ mã nguồn gọn nhẹ.
- Trọng số mô hình đã huấn luyện có thể lưu trữ tại `data/checkpoints/` hoặc `methods/vmunet/results/`.

---

## 📄 Giấy phép (License)

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).