"""
experiments/multitask_vmunet_spatialemb package — VM-UNet + spatial embeddings (offset về tâm con + vùng gaussian học được
+ seed map; Neven et al. 2019). Giữ nguyên đường watershed của baseline để so sánh trên cùng một model.
"""

from .config import SpatialEmbConfig
from .model import SpatialEmbVMUNet

__all__ = ["SpatialEmbConfig", "SpatialEmbVMUNet"]
