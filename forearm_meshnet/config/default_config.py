from __future__ import annotations
from dataclasses import dataclass

# ── Full pipeline defaults (preprocessing → model → training) ─────────────────

DEFAULT_CONFIG = {
    # Preprocessing parameters
    "skin_mask": {
        "end_slice_fraction": 0.25,
        "fix_ghosting": True,
        "fix_connected_ghosting": True,
        "max_connected_ghosting_fix": 14,
        "iso_resolution": 0.5,
        "sdf_blur_sigma": 1.5,
        "max_edge_length": 15.0,
    },
    "skin_mesh": {
        "target_faces": 50000,
        "smooth_iterations": 50,
        "refinement_level": "medium",
    },
    "muscle_mesh": {
        "simplification_target": 10000,
        "smooth_iterations": 30,
    },
    # Template parameters
    "template": {
        "skin_vertices": 5000,
        "muscle_vertices": 500,
        "min_muscle_availability": 0.8,
    },
    # Model parameters
    "model": {
        "latent_dim": 256,
        "encoder_hidden_dims": [128, 256, 512],
        "decoder_hidden_dims": [512, 256, 128],
        "dropout_rate": 0.1,
        "conv_type": "gcn",
        "use_template_augmentation": True,
    },
    # Training parameters
    "training": {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 200,
        "val_split": 0.2,
        "checkpoint_dir": "./checkpoints",
        "log_interval": 10,
    },
}


# ── Typed dataclass configs (used by Trainer / TrainingDataPreparation) ───────

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
