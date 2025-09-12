from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 4
    lr: float = 1e-4
    num_epochs: int = 100
    data_root: str = "data/processed"
    out_dir: str = "models/checkpoints"
    volume_loss_weight: float = 1.0

@dataclass
class PrepConfig:
    raw_root: str = "data/raw"
    processed_root: str = "data/processed"
    val_ratio: float = 0.2
    norm_method: str = "zscore"
