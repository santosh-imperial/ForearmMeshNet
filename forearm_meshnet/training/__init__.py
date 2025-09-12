# forearm_meshnet/training/__init__.py
"""
Training infrastructure for ForearmMeshNet
"""

from .trainer import Trainer
from .curriculum import CurriculumManager
from .metrics import MeshEvaluationMetrics

__all__ = [
    "Trainer",
    "CurriculumManager",
    "MeshEvaluationMetrics"
]

