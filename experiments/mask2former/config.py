"""
config.py — Detectron2 / Mask2Former configuration matching swin_unet hyperparameters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add Mask2Former and models root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK2FORMER_ROOT = PROJECT_ROOT / "models" / "Mask2Former"

if str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config


def setup_mask2former_cfg(output_dir: str = "runs/mask2former") -> any:
    """
    Constructs and returns Detectron2 CfgNode configured for Silkworm Mixed Segmentation,
    aligning with hyperparameters from segfirst_swinunet:
      - Resolution: 224x224
      - Batch size: 6
      - Base LR: 1e-4
      - Weight decay: 0.01
      - Total iterations: 5000 (~50 epochs for 570 train images with batch size 6)
      - AMP: Enabled
      - Backbone: ResNet-50 / Swin-T
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    # Base configuration from Mask2Former ADE20K ResNet-50 semantic segmentation
    base_yaml = MASK2FORMER_ROOT / "configs" / "ade20k" / "semantic-segmentation" / "maskformer2_R50_bs16_160k.yaml"
    cfg.merge_from_file(str(base_yaml))

    # Dataset settings
    cfg.DATASETS.TRAIN = ("silkworm_mixed_train",)
    cfg.DATASETS.TEST = ("silkworm_mixed_valid",)

    # Input resolution: aligned to Swin-Unet (224x224)
    cfg.INPUT.MIN_SIZE_TRAIN = (224,)
    cfg.INPUT.MAX_SIZE_TRAIN = 224
    cfg.INPUT.MIN_SIZE_TEST = 224
    cfg.INPUT.MAX_SIZE_TEST = 224
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (224, 224)
    cfg.INPUT.IMAGE_FORMAT = "RGB"

    # Model settings
    # 2 classes: 0 = Background, 1 = Silkworm
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 50  # 50 queries is abundant for silkworm segmentation

    # Solver / Optimizer hyperparameters aligned with swin_unet
    cfg.SOLVER.IMS_PER_BATCH = int(os.environ.get("BATCH_SIZE", "6"))  # Batch size = 6
    cfg.SOLVER.BASE_LR = 1e-4                 # LR = 1e-4
    cfg.SOLVER.WEIGHT_DECAY = 0.01            # Weight decay = 0.01
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01

    # 570 images / batch 6 = 95 iters/epoch. 50 epochs = 4750 iters.
    cfg.SOLVER.MAX_ITER = int(os.environ.get("MAX_ITER", "5000"))
    cfg.SOLVER.STEPS = (3500, 4500)
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.GAMMA = 0.1

    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500

    # Automatic Mixed Precision (AMP)
    cfg.SOLVER.AMP.ENABLED = True

    # Output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SEED = 42

    return cfg
