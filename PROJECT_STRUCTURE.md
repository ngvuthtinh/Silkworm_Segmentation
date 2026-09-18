# 📁 Cấu trúc Dự án Silkworm Segmentation

> **Tư tưởng thiết kế**: Tách biệt hoàn toàn 4 thứ — **Code Model** / **Code Chung** / **Code Thử Nghiệm** / **Kết Quả**.  
> Mỗi thứ nằm đúng chỗ của nó, không lẫn lộn.

---

## 🗺️ Sơ đồ tổng quan

```
Silkworm_Segmentation/
│
├── models/          ← [TẦNG 1] Kiến trúc model gốc (không sửa)
├── src/             ← [TẦNG 2] Code dùng chung (bạn viết & duy trì)
├── experiments/     ← [TẦNG 3] Code thử nghiệm (wrapper + config)
└── runs/            ← [TẦNG 4] Kết quả tự sinh ra khi train
```

---

## TẦNG 1 — `models/` : Kiến trúc Model Gốc

> 🔒 **Quy tắc: KHÔNG BAO GIỜ SỬA CODE Ở ĐÂY.**  
> Đây là các repo gốc của tác giả, được kéo về bằng `git submodule`.  
> Khi tác giả ra bản vá/cập nhật, chỉ cần chạy `git submodule update` là xong.

```
models/
├── vmunet/          # git submodule → repo gốc VM-UNet/Mamba
├── swin_unet/       # git submodule → repo gốc Swin-Unet
└── sam3/            # git submodule → repo gốc SAM
```

**Cách thêm model mới:**
```bash
git submodule add https://github.com/author/NewModel.git models/new_model
```

---

## TẦNG 2 — `src/` : Code Dùng Chung

> ✅ **Đây là code BẠN viết, dùng chung cho mọi experiment.**  
> Sửa 1 chỗ → toàn bộ experiment đều được hưởng, không cần copy-paste.

```
src/
├── dataset.py       # DataLoader dùng chung (SilkynetSegDataset, YOLOClsDataset)
├── losses.py        # Loss functions mặc định (Dice + BCE + CrossEntropy)
├── metrics.py       # Dice, IoU, F1, Accuracy
└── visualizer.py    # Vẽ ảnh kết quả (input / mask / overlay / badge)
```

### ❓ Khi nào dùng `src/` vs. viết riêng trong `experiments/`?

| Tình huống | Làm gì |
|------------|--------|
| Loss function giống nhau ở mọi model | Viết vào `src/losses.py` |
| 1 model cần loss đặc biệt (VD: Focal Loss chỉ cho VM-UNet) | Tạo `experiments/segfirst_vmunet/losses.py` để **ghi đè** |
| Dataset load ảnh theo cách chung | Viết vào `src/dataset.py` |
| 1 experiment cần augmentation đặc biệt | Override trong `experiments/.../dataset.py` |

**Cơ chế ghi đè (Override) trong code:**
```python
# experiments/segfirst_vmunet/train.py

try:
    from .losses import MyCustomLoss      # 👈 dùng loss riêng nếu có
except ImportError:
    from src.losses import MyCustomLoss   # 👈 fallback về loss chung
```

---

## TẦNG 3 — `experiments/` : Code Thử Nghiệm

> 🧪 **Mỗi thư mục = 1 cách tiếp cận bài toán.**  
> Code ở đây ngắn gọn, chủ yếu là **wrapper** gọi lại `models/` và `src/`.  
> KHÔNG chứa toàn bộ logic nặng — logic đó đã ở `src/` hoặc `models/`.

```
experiments/
├── segfirst_vmunet/          # Thử nghiệm: VM-UNet + 2-phase training
│   ├── config.yaml           #   ← QUAN TRỌNG: mọi hyperparameter ở đây
│   ├── model.py              #   ← Wrap models/vmunet vào bài toán silkworm
│   ├── train.py              #   ← Script train (gọi src/ + model.py)
│   ├── evaluate.py           #   ← Script đánh giá
│   ├── losses.py             #   ← (tuỳ chọn) loss riêng, ghi đè src/losses.py
│   └── notes.md              #   ← Ghi chú thử nghiệm, kết quả, nhận xét
│
└── segfirst_swinunet/        # Thử nghiệm: Swin-UNet + 2-phase training
    ├── config.yaml
    ├── model.py
    ├── train.py
    └── notes.md              # (không có losses.py → tự dùng src/losses.py)
```

### 📄 Ví dụ `config.yaml`

```yaml
experiment_name: segfirst_vmunet_phase2

model:
  name: SegFirstVMUNet
  backbone: vmamba_tiny
  pretrained: runs/segfirst_vmunet/2026-09-18_run1/checkpoints/best.pth

data:
  train_images: data/silkynet/train/images
  train_masks:  data/silkynet/train/masks
  yolo_cls_dir: data/diseases/train

training:
  phase: 2
  epochs: 50
  batch_size: 8
  lr_backbone: 1.0e-5
  lr_seg:      1.0e-4
  lr_cls:      1.0e-3
  lambda_cls:  0.1
```

> 💡 Nhìn vào `config.yaml` là biết ngay experiment đó thử nghiệm cái gì — không cần đọc code.

---

## TẦNG 4 — `runs/` : Kết Quả (Tự Sinh Ra)

> 🤖 **Thư mục này KHÔNG chứa code. Hoàn toàn tự sinh ra khi chạy train.**  
> Mỗi lần train = 1 thư mục mới theo timestamp, không bao giờ ghi đè lên nhau.

```
runs/
├── segfirst_vmunet/
│   ├── 2026-09-18_14-30-00/      # Lần train 1
│   │   ├── config.yaml           #   snapshot config lúc train (để tra cứu sau)
│   │   ├── checkpoints/
│   │   │   ├── best.pth          #   checkpoint tốt nhất (val_dice cao nhất)
│   │   │   └── last.pth          #   checkpoint epoch cuối
│   │   ├── logs/
│   │   │   └── train_log.csv     #   loss, dice, f1 từng epoch
│   │   └── test_outputs/
│   │       ├── img_001.png       #   ảnh kết quả dự đoán
│   │       └── img_002.png
│   │
│   └── 2026-09-20_09-15-00/      # Lần train 2 (chỉnh config, chạy lại)
│       ├── config.yaml
│       ├── checkpoints/
│       └── test_outputs/
│
└── segfirst_swinunet/
    └── 2026-09-19_11-00-00/
        ├── config.yaml
        ├── checkpoints/
        └── test_outputs/
```

**Code tự tạo thư mục này trong `train.py`:**
```python
from datetime import datetime
from pathlib import Path
import shutil

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir  = Path(f"runs/{cfg.experiment_name}/{run_name}")

(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(run_dir / "test_outputs").mkdir(parents=True, exist_ok=True)
(run_dir / "logs").mkdir(parents=True, exist_ok=True)

shutil.copy(cfg.config_path, run_dir / "config.yaml")  # lưu snapshot config
```

---

## 🔄 Luồng hoạt động tổng thể

```
[Bạn chỉnh config.yaml]
        ↓
[Chạy: python experiments/segfirst_vmunet/train.py]
        ↓
        ├── import models/vmunet      (kiến trúc gốc)
        ├── import src/dataset.py     (data loader chung)
        ├── import src/losses.py      (hoặc experiments/.../losses.py nếu có)
        └── import src/metrics.py
        ↓
[Tự tạo: runs/segfirst_vmunet/2026-09-18_14-30-00/]
        ├── checkpoints/best.pth
        ├── logs/train_log.csv
        └── test_outputs/img_*.png
```

---

## 📊 Bảng tra cứu nhanh: "Tôi cần làm gì thì mở file nào?"

| Tôi muốn... | Mở file ở đâu |
|-------------|---------------|
| Thêm model mới | `git submodule add` vào `models/` |
| Sửa cách load dataset (tất cả model) | `src/dataset.py` |
| Thêm loss function mới dùng chung | `src/losses.py` |
| Thêm loss riêng cho 1 experiment | `experiments/<tên>/losses.py` |
| Chỉnh hyperparameter, learning rate | `experiments/<tên>/config.yaml` |
| Xem kết quả lần train cũ | `runs/<tên>/<timestamp>/` |
| So sánh 2 lần train | Mở 2 file `runs/.../logs/train_log.csv` |
| Xem ảnh dự đoán | `runs/<tên>/<timestamp>/test_outputs/` |

---

## 🚫 Quy tắc KHÔNG được làm

- ❌ Không sửa code trong `models/` (repo gốc của tác giả)
- ❌ Không copy-paste `dataset.py` hay `losses.py` giữa các experiment
- ❌ Không lưu checkpoint hay ảnh kết quả trong `experiments/` hoặc `src/`
- ❌ Không commit thư mục `runs/` lên git (thêm vào `.gitignore`)

```gitignore
# .gitignore
runs/
__pycache__/
*.pyc
.venv/
```
