# forearm_meshnet/models/__init__.py
"""
Neural network models for ForearmMeshNet
"""

from .encoder import VariationalGraphEncoder
from .decoder import StructureAwareDecoder, AnthroAffine
from .meshnet import ForearmMeshNet
from .losses import (
    CombinedLoss,
    VolumeLoss,
    ChamferDistance,
    EdgeLengthLoss,
    NormalConsistencyLoss,
    LaplacianSmoothingLoss
)

__all__ = [
    "VariationalGraphEncoder",
    "StructureAwareDecoder",
    "AnthroAffine",
    "ForearmMeshNet",
    "CombinedLoss",
    "VolumeLoss",
    "ChamferDistance",
    "EdgeLengthLoss",
    "NormalConsistencyLoss",
    "LaplacianSmoothingLoss"
]