"""
train.py — Training script for Mask2Former on Silkworm Mixed Dataset.
Hyperparameters aligned with segfirst_swinunet:
  - Input: 224x224
  - Batch size: 6
  - LR: 1e-4
  - Weight decay: 0.01
  - Max iters: 5000 (~50 epochs)
  - AMP: True
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
try:
    import distutils.version
except Exception:
    pass
import torch
import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK2FORMER_ROOT = PROJECT_ROOT / "models" / "Mask2Former"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import SemSegEvaluator, DatasetEvaluators
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.engine import HookBase
from detectron2.utils.events import JSONWriter, TensorboardXWriter, get_event_storage
from tqdm import tqdm

from mask2former import (
    MaskFormerSemanticDatasetMapper,
    add_maskformer2_config,
)

from experiments.mask2former.dataset_adapter import register_silkworm_mixed_datasets
from experiments.mask2former.config import setup_mask2former_cfg


class TQDMMask2FormerHook(HookBase):
    """
    Custom TQDM progress bar hook for Mask2Former with percentage, epoch, loss, and ETA.
    """
    def __init__(self, max_iter: int, iters_per_epoch: int = 95, total_epochs: int = 50):
        self.max_iter = max_iter
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.pbar = None

    def before_train(self):
        self.pbar = tqdm(
            total=self.max_iter,
            desc="Mask2Former [50 Epochs]",
            dynamic_ncols=True,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    def after_step(self):
        self.pbar.update(1)
        storage = get_event_storage()
        try:
            total_loss = storage.latest().get("total_loss", None)
            if total_loss is None and "total_loss" in storage.histories():
                total_loss = storage.histories()["total_loss"].latest()
            loss_mask = storage.latest().get("loss_mask", None)
            loss_dice = storage.latest().get("loss_dice", None)
            lr = storage.latest().get("lr", None)
            cur_iter = self.trainer.iter + 1
            cur_epoch = min(cur_iter // self.iters_per_epoch + 1, self.total_epochs)
            postfix = {"Ep": f"{cur_epoch:02d}/{self.total_epochs:02d}"}
            if total_loss is not None:
                postfix["loss"] = f"{total_loss:.3f}"
            if loss_mask is not None:
                postfix["mask"] = f"{loss_mask:.3f}"
            if loss_dice is not None:
                postfix["dice"] = f"{loss_dice:.3f}"
            if lr is not None:
                postfix["lr"] = f"{lr:.1e}"
            self.pbar.set_postfix(postfix)
        except Exception:
            pass

    def after_train(self):
        if self.pbar:
            self.pbar.close()


class SilkwormMask2FormerTrainer(DefaultTrainer):
    """
    Mask2Former Trainer specialized for silkworm semantic segmentation.
    """

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return SemSegEvaluator(
            dataset_name,
            distributed=False,
            output_dir=output_folder,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_writers(self):
        # Replace verbose metric printer with quiet JSON & TensorBoard writers
        return [
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(TQDMMask2FormerHook(self.max_iter, iters_per_epoch=95, total_epochs=50))
        return hooks


def main():
    # Setup logger
    setup_logger(name="mask2former")
    logger = logging.getLogger("mask2former.train")
    logger.info("Initializing Mask2Former training for Silkworm Mixed Dataset...")

    # 1. Register datasets
    register_silkworm_mixed_datasets()
    logger.info("Datasets registered in Detectron2 DatasetCatalog.")

    # 2. Setup config
    cfg = setup_mask2former_cfg(output_dir="runs/mask2former")
    default_setup(cfg, {})
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)

    logger.info(f"Target Resolution: {cfg.INPUT.MIN_SIZE_TRAIN}x{cfg.INPUT.MAX_SIZE_TRAIN}")
    logger.info(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"Learning Rate: {cfg.SOLVER.BASE_LR}")
    logger.info(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"AMP Enabled: {cfg.SOLVER.AMP.ENABLED}")
    logger.info(f"Output Directory: {cfg.OUTPUT_DIR}")

    # 3. Create Trainer & Train
    trainer = SilkwormMask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=False)

    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
