"""
test_curriculum.py — Tests for CurriculumManager.
"""

import pytest
import torch
import numpy as np


def _make_sample(deform_scale: float, n_structs: int = 2, n_verts: int = 8):
    """Create a minimal training sample with controllable deformation magnitude.

    Scoring formula:
        num_structs = max(1, len(deformations) - 1)   # -1 to discount 'combined'
        score = (deform_mag / 10.0) + (20 - num_structs) / 20.0
    Buckets: easy < 0.7, medium < 1.3, hard >= 1.3

    With n_structs=2:  score ≈ 0 + (20-1)/20 = 0.95  → medium regardless of magnitude
    With n_structs=10: score ≈ 0 + (20-9)/20 = 0.55  → easy for tiny deformations
    """
    return {
        "structure_deformations": {
            f"s{i}": torch.randn(n_verts, 3) * deform_scale
            for i in range(n_structs)
        },
        "anthropometric_features": torch.randn(16),
    }


def _make_samples(n_medium=4, n_hard=4):
    """Return a mixed batch: medium (few structs, small deform) and hard (large deform)."""
    medium = [_make_sample(deform_scale=0.01, n_structs=2) for _ in range(n_medium)]
    hard   = [_make_sample(deform_scale=50.0, n_structs=2) for _ in range(n_hard)]
    return medium + hard


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestCurriculumManagerInit:
    def test_instantiates(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        assert cm is not None

    def test_default_config(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples(), config=None)
        assert cm.cv_epochs == 200

    def test_custom_cv_epochs(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples(), config={"cv_epochs": 50})
        assert cm.cv_epochs == 50

    def test_initial_epoch_zero(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        assert cm.current_epoch == 0


# ── _bucketize ────────────────────────────────────────────────────────────────

class TestBucketize:
    def test_all_easy_many_structs_tiny_deformations(self):
        """With 10 structures and tiny deform, score ≈ 0.55 → easy bucket."""
        from forearm_meshnet.training.curriculum import CurriculumManager
        samples = [_make_sample(deform_scale=0.001, n_structs=10) for _ in range(10)]
        cm = CurriculumManager(samples)
        assert len(cm.easy) == 10
        assert len(cm.hard) == 0

    def test_all_medium_few_structs_small_deformations(self):
        """With 2 structures and tiny deform, score ≈ 0.95 → medium bucket."""
        from forearm_meshnet.training.curriculum import CurriculumManager
        samples = [_make_sample(deform_scale=0.001, n_structs=2) for _ in range(10)]
        cm = CurriculumManager(samples)
        assert len(cm.medium) == 10
        assert len(cm.easy) == 0

    def test_all_hard_large_deformations(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        samples = [_make_sample(deform_scale=200.0) for _ in range(10)]
        cm = CurriculumManager(samples)
        assert len(cm.hard) == 10
        assert len(cm.easy) == 0

    def test_combined_key_excluded_from_magnitude(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        # A sample with only 'combined' key should not crash
        samples = [
            {"structure_deformations": {"combined": torch.randn(30)},
             "anthropometric_features": torch.randn(16)}
        ]
        cm = CurriculumManager(samples)
        assert len(cm.easy) + len(cm.medium) + len(cm.hard) == 1

    def test_buckets_partition_all_samples(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        samples = _make_samples(n_medium=5, n_hard=5)
        cm = CurriculumManager(samples)
        total = len(cm.easy) + len(cm.medium) + len(cm.hard)
        assert total == len(samples)

    def test_numpy_deformations_accepted(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        samples = [
            {"structure_deformations": {"skin": np.random.randn(8, 3) * 0.01},
             "anthropometric_features": torch.randn(16)}
        ]
        cm = CurriculumManager(samples)
        assert len(cm.easy) + len(cm.medium) + len(cm.hard) == 1


# ── get_epoch_samples ──────────────────────────────────────────────────────────

class TestGetEpochSamples:
    def test_returns_list(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples(), config={"epoch_sample_size": 4})
        result = cm.get_epoch_samples()
        assert isinstance(result, list)

    def test_returns_epoch_sample_size(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples(), config={"epoch_sample_size": 5})
        result = cm.get_epoch_samples()
        assert len(result) == 5

    def test_early_epoch_draws_from_easy_pool(self):
        """At epoch 0 (p=0), pool is easy samples (10-struct, tiny deform)."""
        from forearm_meshnet.training.curriculum import CurriculumManager
        easy = [_make_sample(deform_scale=0.001, n_structs=10) for _ in range(20)]
        hard = [_make_sample(deform_scale=200.0) for _ in range(1)]
        cm = CurriculumManager(easy + hard, config={"cv_epochs": 100, "epoch_sample_size": 10})
        cm.update_epoch(0)
        result = cm.get_epoch_samples()
        # All returned samples should be from the easy bucket (use identity, not equality,
        # because dicts with tensor values can't be compared with ==)
        easy_ids = {id(s) for s in easy}
        for s in result:
            assert id(s) in easy_ids

    def test_late_epoch_uses_all_samples(self):
        """At epoch >= cv_epochs (p=1), pool is all_samples."""
        from forearm_meshnet.training.curriculum import CurriculumManager
        all_samples = _make_samples(n_medium=5, n_hard=5)
        cm = CurriculumManager(all_samples, config={"cv_epochs": 10, "epoch_sample_size": 20})
        cm.update_epoch(10)
        samples = cm.get_epoch_samples()
        assert len(samples) == 20

    def test_fallback_when_easy_empty(self):
        """If easy is empty (only medium/hard), get_epoch_samples should not raise."""
        from forearm_meshnet.training.curriculum import CurriculumManager
        # 2-struct samples → always medium or hard, easy bucket is empty
        samples = [_make_sample(deform_scale=0.001, n_structs=2) for _ in range(5)]
        cm = CurriculumManager(samples, config={"epoch_sample_size": 3})
        cm.update_epoch(0)
        result = cm.get_epoch_samples()
        assert len(result) == 3


# ── update_epoch ──────────────────────────────────────────────────────────────

class TestUpdateEpoch:
    def test_updates_current_epoch(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        cm.update_epoch(42)
        assert cm.current_epoch == 42


# ── get_phase_config / get_loss_weights / should_evaluate_structure ──────────

class TestHelpers:
    def test_get_phase_config_keys(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        cfg = cm.get_phase_config()
        assert "name" in cfg
        assert "epochs" in cfg
        assert "description" in cfg

    def test_get_loss_weights_keys(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        w = cm.get_loss_weights()
        assert "reconstruction" in w
        assert "geometric" in w
        assert "kl" in w

    def test_should_evaluate_structure_always_true(self):
        from forearm_meshnet.training.curriculum import CurriculumManager
        cm = CurriculumManager(_make_samples())
        assert cm.should_evaluate_structure("skin") is True
        assert cm.should_evaluate_structure("FCR") is True
