"""
__init__.py — Multi-task VM-UNet package.
"""
from .model   import MultiTaskVMUNet
from .losses  import MultiTaskLoss, SegLoss, ClsLoss
from .dataset import SegDataset, ClsDataset, build_seg_loaders, build_cls_loaders
from .config  import MultiTaskConfig

__all__ = [
    "MultiTaskVMUNet",
    "MultiTaskLoss", "SegLoss", "ClsLoss",
    "SegDataset", "ClsDataset", "build_seg_loaders", "build_cls_loaders",
    "MultiTaskConfig",
]
