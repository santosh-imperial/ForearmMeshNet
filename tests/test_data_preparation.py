"""
test_data_preparation.py — Tests for the pure (no-I/O) helpers in
TrainingDataPreparation that can be exercised without loading real files.
"""

import pytest
import numpy as np
import torch


# ── _extract_subject_id ───────────────────────────────────────────────────────

class TestExtractSubjectId:
    """
    _extract_subject_id is a static-like method on TrainingDataPreparation.
    We test it by monkey-patching an instance that skips __init__.
    """
    @pytest.fixture
    def prep(self):
        from forearm_meshnet.data.data_preparation import TrainingDataPreparation
        obj = object.__new__(TrainingDataPreparation)
        return obj

    def test_subject_underscore_number(self, prep):
        # Leading zeros are preserved (regex captures raw digits)
        assert prep._extract_subject_id("subject_01.ply") == "subject_01"

    def test_subject_uppercase(self, prep):
        result = prep._extract_subject_id("Subject_42.obj")
        assert result == "subject_42"

    def test_bare_number(self, prep):
        # Leading zeros are preserved
        result = prep._extract_subject_id("007.ply")
        assert result == "subject_007"

    def test_no_number_uses_stem(self, prep):
        result = prep._extract_subject_id("alice.ply")
        assert result == "alice"

    def test_subject_hyphen(self, prep):
        result = prep._extract_subject_id("subject-10.ply")
        assert result == "subject_10"


# ── create_train_val_split ────────────────────────────────────────────────────

class TestCreateTrainValSplit:
    @pytest.fixture
    def prep(self):
        from forearm_meshnet.data.data_preparation import TrainingDataPreparation
        obj = object.__new__(TrainingDataPreparation)
        return obj

    def _make_samples(self, n):
        return [{"id": i} for i in range(n)]

    def test_total_count_preserved(self, prep):
        samples = self._make_samples(10)
        train, val = prep.create_train_val_split(samples, val_ratio=0.2)
        assert len(train) + len(val) == 10

    def test_val_count_correct(self, prep):
        samples = self._make_samples(10)
        _, val = prep.create_train_val_split(samples, val_ratio=0.2)
        assert len(val) == 2

    def test_no_overlap(self, prep):
        samples = self._make_samples(20)
        train, val = prep.create_train_val_split(samples, val_ratio=0.2)
        train_ids = {s["id"] for s in train}
        val_ids = {s["id"] for s in val}
        assert len(train_ids & val_ids) == 0

    def test_reproducible_with_seed(self, prep):
        samples = self._make_samples(20)
        _, val1 = prep.create_train_val_split(samples, val_ratio=0.3, random_seed=0)
        _, val2 = prep.create_train_val_split(samples, val_ratio=0.3, random_seed=0)
        assert [s["id"] for s in val1] == [s["id"] for s in val2]

    def test_different_seeds_differ(self, prep):
        samples = self._make_samples(20)
        _, val1 = prep.create_train_val_split(samples, val_ratio=0.3, random_seed=0)
        _, val2 = prep.create_train_val_split(samples, val_ratio=0.3, random_seed=99)
        # Very unlikely to be identical
        assert [s["id"] for s in val1] != [s["id"] for s in val2]

    def test_val_ratio_zero(self, prep):
        samples = self._make_samples(10)
        train, val = prep.create_train_val_split(samples, val_ratio=0.0)
        assert len(val) == 0
        assert len(train) == 10


# ── _combine_deformations ─────────────────────────────────────────────────────

class TestCombineDeformations:
    @pytest.fixture
    def prep(self):
        from forearm_meshnet.data.data_preparation import TrainingDataPreparation
        import trimesh
        obj = object.__new__(TrainingDataPreparation)
        # Minimal unified mesh: 10 vertices
        obj.unified_mesh = trimesh.creation.box()
        # Patch with 10-vertex mesh
        obj.unified_mesh = trimesh.Trimesh(
            vertices=np.zeros((10, 3)),
            faces=np.array([[0,1,2]]),
            process=False
        )
        obj.structure_info = {
            "skin": {"vertex_range": (0, 6)},
            "fcr":  {"vertex_range": (6, 10)},
        }
        return obj

    def test_output_shape(self, prep):
        deforms = {
            "skin": np.ones((6, 3), dtype=np.float32),
            "fcr":  np.ones((4, 3), dtype=np.float32) * 2.0,
        }
        combined = prep._combine_deformations(deforms)
        assert combined.shape == (10, 3)

    def test_correct_values_placed(self, prep):
        deforms = {
            "skin": np.ones((6, 3), dtype=np.float32),
            "fcr":  np.ones((4, 3), dtype=np.float32) * 2.0,
        }
        combined = prep._combine_deformations(deforms)
        np.testing.assert_allclose(combined[:6], 1.0)
        np.testing.assert_allclose(combined[6:], 2.0)

    def test_dtype_is_float32(self, prep):
        deforms = {"skin": np.ones((6, 3), dtype=np.float64)}
        combined = prep._combine_deformations(deforms)
        assert combined.dtype == np.float32

    def test_unknown_structure_ignored(self, prep):
        deforms = {
            "skin": np.ones((6, 3), dtype=np.float32),
            "unknown": np.ones((5, 3), dtype=np.float32),  # not in structure_info
        }
        combined = prep._combine_deformations(deforms)
        assert combined.shape == (10, 3)

    def test_size_mismatch_truncated(self, prep):
        """Deformation larger than expected range is truncated."""
        deforms = {
            "skin": np.ones((10, 3), dtype=np.float32),  # expected 6, got 10
        }
        combined = prep._combine_deformations(deforms)
        # Only first 6 rows should be filled
        np.testing.assert_allclose(combined[:6], 1.0)
        np.testing.assert_allclose(combined[6:], 0.0)

    def test_size_mismatch_padded(self, prep):
        """Deformation smaller than expected range is zero-padded."""
        deforms = {
            "skin": np.ones((3, 3), dtype=np.float32),  # expected 6, got 3
        }
        combined = prep._combine_deformations(deforms)
        assert combined.shape == (10, 3)
