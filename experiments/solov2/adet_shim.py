"""
adet_shim.py — Nạp AdelaiDet (models/AdelaiDet) mà KHÔNG build phần mã CUDA riêng của nó.

`import adet` kéo theo `adet.layers.bezier_align` / `def_roi_align`, hai lớp này cần `adet._C` (C++/CUDA). Mã đó dùng API THC cũ,
không build được với PyTorch 2.x; SOLOv2 không bao giờ gọi chúng (chỉ ABCNet dùng). Vì vậy đặt một module rỗng làm `adet._C`
trước khi import. Không sửa gì trong models/AdelaiDet (quy tắc tầng 1).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADET_DIR = ROOT / "models" / "AdelaiDet"
VMUNET_DIR = ROOT / "models" / "vmunet" / "models"


def setup_paths() -> None:
    for p in (ADET_DIR, VMUNET_DIR, ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if "adet._C" not in sys.modules:
        sys.modules["adet._C"] = types.ModuleType("adet._C")      # rỗng: SOLOv2 không dùng tới


setup_paths()

import adet  # noqa: E402,F401  (đăng ký META_ARCH "SOLOv2" vào detectron2)
from adet.config import get_cfg  # noqa: E402

__all__ = ["ADET_DIR", "ROOT", "get_cfg"]
