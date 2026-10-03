"""
experiments/multitask_vmunet package — VM-UNet phân đoạn từng con + chẩn đoán từng con.
"""

from .config import MultiTaskConfig
from .model import MultiTaskVMUNet

__all__ = ["MultiTaskConfig", "MultiTaskVMUNet"]
