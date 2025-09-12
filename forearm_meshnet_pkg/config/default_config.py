"""
Default configuration for ForearmMeshNet
"""

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
    }
}
