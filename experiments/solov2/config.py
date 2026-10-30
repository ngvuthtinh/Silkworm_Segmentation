"""
config.py — Siêu tham số SOLOv2 cho data tằm và hàm dựng cfg detectron2 cho từng biến thể backbone.

Ba biến thể chỉ khác BACKBONE (một thay đổi mỗi lần); mọi thứ khác (data, cỡ ảnh, optimizer, số vòng, đầu SOLOv2) giống nhau:
  r50       build_resnet_fpn_backbone (ResNet-50 + FPN, trọng số ImageNet của detectron2; --no-imagenet để train từ đầu)
  mamba  build_mamba_backbone   (cách A, train từ đầu: chưa có trọng số VMamba pretrained trong repo)
  vmunet    build_vmunet_backbone     (cách B, train từ đầu)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

from detectron2.config import CfgNode as CN

from experiments.solov2.adet_shim import ADET_DIR, get_cfg

BACKBONES = {"r50": "build_resnet_fpn_backbone", "mamba": "build_mamba_backbone", "vmunet": "build_vmunet_backbone"}


@dataclass
class SOLOv2Config:
    variant: str = "mamba"
    train_sets: list = field(default_factory=lambda: ["v3:train", "dense_v2:train*8"])   # <tên data>:<split>[*lặp]; dense ≈ 18% mẫu
    val_set: str = "v3:valid"
    img_size: int = 640                    # cạnh ngắn khi CHẠY CẢ ẢNH (valid/test): gần cỡ gốc (ảnh data 640², ảnh dày 640×639 / 1280×720)
    max_size: int = 1280
    crop_size: int = 512                   # train: cắt cửa sổ crop² từ ảnh đã co ngẫu nhiên (cắt chỉ để vừa bộ nhớ; suy luận chạy cả ảnh)
    train_short_edges: list = field(default_factory=lambda: [400, 480, 560, 640, 720, 800, 880])   # scale jitter ≈ 0.6-1.4× cỡ gốc
    batch_size: int = 8
    max_iter: int = 60000                  # ≈ 24 lượt qua 19,989 ảnh train của v3 ở batch 8
    lr: float = 1e-4                       # AdamW cho cả ba biến thể (backbone Mamba train từ đầu không hợp SGD)
    weight_decay: float = 0.05
    warmup_iters: int = 1000
    steps: list = field(default_factory=lambda: [40000, 52000])
    eval_period: int = 5000
    num_classes: int = 2                   # 0 = Grasserie, 1 = Healthy (giống quy ước của src.metrics)
    imagenet_r50: bool = True
    vssm_depths: list = field(default_factory=lambda: [2, 2, 2, 2])            # giống MultiTaskConfig của VM-UNet
    vssm_depths_decoder: list = field(default_factory=lambda: [2, 2, 2, 1])
    drop_path_rate: float = 0.2
    eval_score_thr: float = 0.3            # ngưỡng điểm SOLOv2 khi chấm theo từng con (chỉnh trên valid)
    seed: int = 42
    num_workers: int = 4
    amp: bool = True
    work_dir: str = ""

    def __post_init__(self) -> None:
        assert self.variant in BACKBONES, self.variant
        if not self.work_dir:
            self.work_dir = os.path.join("runs", f"solov2_{self.variant}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    def to_dict(self) -> dict:
        return asdict(self)


def build_cfg(c: SOLOv2Config, train_names: list[str], val_name: str) -> CN:
    cfg = get_cfg()
    cfg.merge_from_file(str(ADET_DIR / "configs" / "SOLOv2" / "Base-SOLOv2.yaml"))
    cfg.MODEL.META_ARCHITECTURE = "SilkSOLOv2"
    cfg.MODEL.BACKBONE.NAME = BACKBONES[c.variant]
    cfg.MODEL.VSSM = CN()
    cfg.MODEL.VSSM.DEPTHS = list(c.vssm_depths)
    cfg.MODEL.VSSM.DEPTHS_DECODER = list(c.vssm_depths_decoder)
    cfg.MODEL.VSSM.DROP_PATH_RATE = c.drop_path_rate
    if c.variant == "r50":
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl" if c.imagenet_r50 else ""
        # PIXEL_MEAN/STD mặc định của detectron2 khớp trọng số MSRA (BGR, std = 1)
    else:
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]                  # BGR, chuẩn hoá ImageNet như VM-UNet
        cfg.MODEL.PIXEL_STD = [57.375, 57.12, 58.395]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SOLOV2.NUM_CLASSES = c.num_classes
    cfg.DATASETS.TRAIN = tuple(train_names)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = c.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(c.train_short_edges)
    cfg.INPUT.MAX_SIZE_TRAIN = 2 * c.max_size
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = [c.crop_size, c.crop_size]
    cfg.INPUT.MIN_SIZE_TEST = c.img_size
    cfg.INPUT.MAX_SIZE_TEST = c.max_size
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = c.batch_size
    cfg.SOLVER.BASE_LR = c.lr
    cfg.SOLVER.WEIGHT_DECAY = c.weight_decay
    cfg.SOLVER.WARMUP_ITERS = c.warmup_iters
    cfg.SOLVER.STEPS = tuple(c.steps)
    cfg.SOLVER.MAX_ITER = c.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = c.eval_period
    cfg.SOLVER.AMP.ENABLED = c.amp
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False             # cắt kiểu detectron2 là TỪNG tham số riêng; dùng cắt toàn model ở train.py
    cfg.TEST.EVAL_PERIOD = c.eval_period
    cfg.SEED = c.seed
    cfg.OUTPUT_DIR = c.work_dir
    cfg.SILK = CN()
    cfg.SILK.CLIP_FULL_MODEL = 1.0                        # chuẩn gradient của CẢ model ≤ 1 (như Mask2Former); không có thì
                                                          # AdamW làm SOLOv2 "nổ" ngay các bước đầu (kiểm tra 2026-10-28)
    cfg.SILK.EVAL_SCORE_THR = c.eval_score_thr
    cfg.SILK.VARIANT = c.variant
    return cfg
