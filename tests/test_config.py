"""
test_config.py — Tests for default_config dataclasses and config __init__ exports.
"""

import pytest


# ── DEFAULT_CONFIG ────────────────────────────────────────────────────────────

class TestDefaultConfig:
    def test_importable(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG is not None

    def test_has_all_sections(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        for section in ("skin_mask", "skin_mesh", "muscle_mesh", "template", "model", "training"):
            assert section in DEFAULT_CONFIG, f"Missing section: {section}"

    def test_model_section_keys(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        model = DEFAULT_CONFIG["model"]
        assert "latent_dim" in model
        assert "encoder_hidden_dims" in model
        assert "decoder_hidden_dims" in model
        assert "conv_type" in model

    def test_training_section_keys(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        training = DEFAULT_CONFIG["training"]
        assert "batch_size" in training
        assert "learning_rate" in training
        assert "num_epochs" in training

    def test_reasonable_latent_dim(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["model"]["latent_dim"] > 0

    def test_reasonable_num_epochs(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["training"]["num_epochs"] > 0

    def test_is_dict(self):
        from forearm_meshnet.config import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG, dict)


# ── TrainConfig ───────────────────────────────────────────────────────────────

class TestTrainConfig:
    def test_importable_from_config(self):
        from forearm_meshnet.config import TrainConfig
        assert TrainConfig is not None

    def test_default_seed(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.seed == 42

    def test_default_batch_size(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.batch_size == 4

    def test_default_num_epochs(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.num_epochs == 100

    def test_default_lr(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.lr == pytest.approx(1e-4)

    def test_default_data_root(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig()
        assert cfg.data_root == "data/processed"

    def test_custom_override(self):
        from forearm_meshnet.config import TrainConfig
        cfg = TrainConfig(seed=0, batch_size=16, num_epochs=50)
        assert cfg.seed == 0
        assert cfg.batch_size == 16
        assert cfg.num_epochs == 50

    def test_is_dataclass(self):
        import dataclasses
        from forearm_meshnet.config import TrainConfig
        assert dataclasses.is_dataclass(TrainConfig)


# ── PrepConfig ────────────────────────────────────────────────────────────────

class TestPrepConfig:
    def test_importable_from_config(self):
        from forearm_meshnet.config import PrepConfig
        assert PrepConfig is not None

    def test_default_val_ratio(self):
        from forearm_meshnet.config import PrepConfig
        cfg = PrepConfig()
        assert cfg.val_ratio == pytest.approx(0.2)

    def test_default_norm_method(self):
        from forearm_meshnet.config import PrepConfig
        cfg = PrepConfig()
        assert cfg.norm_method == "zscore"

    def test_default_raw_root(self):
        from forearm_meshnet.config import PrepConfig
        cfg = PrepConfig()
        assert cfg.raw_root == "data/raw"

    def test_custom_override(self):
        from forearm_meshnet.config import PrepConfig
        cfg = PrepConfig(val_ratio=0.1, norm_method="minmax")
        assert cfg.val_ratio == pytest.approx(0.1)
        assert cfg.norm_method == "minmax"

    def test_is_dataclass(self):
        import dataclasses
        from forearm_meshnet.config import PrepConfig
        assert dataclasses.is_dataclass(PrepConfig)
