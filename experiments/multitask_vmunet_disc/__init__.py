"""
experiments/multitask_vmunet_disc package — VM-UNet + embedding theo pixel (discriminative loss) để gộp các mảnh
rời của cùng một con bị lá che.
"""

from .config import DiscConfig
from .model import DiscVMUNet

__all__ = ["DiscConfig", "DiscVMUNet"]
