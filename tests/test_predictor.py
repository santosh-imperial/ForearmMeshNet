"""
test_predictor.py — Tests for Predictor internal methods.

The Predictor.__init__ loads files from disk (checkpoint, template, normalizer),
so we bypass it using object.__new__ and set attributes directly, then test
each internal method in isolation.
"""

import pytest
import torch
import numpy as np
import trimesh
import tempfile
import json
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_predictor(structure_info=None, n_verts=32):
    """Build a Predictor instance without loading any files."""
    from forearm_meshnet.inference.predictor import Predictor
    from forearm_meshnet.features.anthropometric import AnthropometricExtractor

    if structure_info is None:
        structure_info = {
            "skin": {"vertex_range": (0, 20)},
            "FCR":  {"vertex_range": (20, 32)},
        }

    pred = object.__new__(Predictor)
    pred.device = torch.device("cpu")
    pred.structure_info = structure_info

    # Minimal template mesh (cylinder-like, just needs vertices/faces)
    mesh = trimesh.creation.cylinder(radius=20.0, height=250.0, sections=16)
    # Pad to n_verts if needed for deformation tests
    verts = np.zeros((n_verts, 3), dtype=np.float64)
    verts[:min(n_verts, len(mesh.vertices))] = mesh.vertices[:n_verts]
    faces = np.array([[0, 1, 2]], dtype=np.int64)  # minimal face
    pred.template_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    pred.template_graph = None
    pred.anthro_extractor = AnthropometricExtractor()
    pred.normalizer = {}   # empty — triggers passthrough in _denormalize_predictions
    pred.config = {}

    return pred


def _minimal_measurements():
    return {
        "forearm_length": 250.0,
        "wrist_circumference": 170.0,
        "mid_forearm_circumference": 210.0,
        "proximal_circumference": 240.0,
        "taper_ratio": 1.41,
        "length_width_ratio": 6.25,
        "width_depth_ratio": 1.0,
        "wrist_cross_sectional_area": 2300.0,
        "mid_cross_sectional_area": 3500.0,
        "proximal_cross_sectional_area": 4580.0,
        "max_dimension": 250.0,
        "min_dimension": 38.0,
        "bounding_box_volume": 362000.0,
        "surface_area": 17593.0,
        "volume": 314159.0,
    }


# ── _prepare_features ──────────────────────────────────────────────────────────

class TestPrepareFeatures:
    def test_returns_tensor(self):
        pred = _make_predictor()
        features = pred._prepare_features(_minimal_measurements())
        assert isinstance(features, torch.Tensor)

    def test_output_is_1d(self):
        pred = _make_predictor()
        features = pred._prepare_features(_minimal_measurements())
        assert features.dim() == 1

    def test_output_length(self):
        pred = _make_predictor()
        features = pred._prepare_features(_minimal_measurements())
        expected_dim = pred.anthro_extractor.get_feature_dim(include_categorical=True)
        assert features.shape[0] == expected_dim

    def test_missing_keys_filled_with_defaults(self):
        pred = _make_predictor()
        # Pass almost empty measurements — should not raise
        features = pred._prepare_features({})
        assert features.shape[0] > 0

    def test_float32_output(self):
        pred = _make_predictor()
        features = pred._prepare_features(_minimal_measurements())
        assert features.dtype == torch.float32


# ── _apply_deformations ────────────────────────────────────────────────────────

class TestApplyDeformations:
    def test_returns_dict(self):
        pred = _make_predictor(n_verts=32)
        deformations = {
            "skin": torch.zeros(20, 3),
            "FCR":  torch.zeros(12, 3),
        }
        result = pred._apply_deformations(deformations)
        assert isinstance(result, dict)

    def test_unified_key_present(self):
        pred = _make_predictor(n_verts=32)
        deformations = {"skin": torch.zeros(20, 3), "FCR": torch.zeros(12, 3)}
        result = pred._apply_deformations(deformations)
        assert "unified" in result

    def test_unified_is_trimesh(self):
        pred = _make_predictor(n_verts=32)
        deformations = {"skin": torch.zeros(20, 3)}
        result = pred._apply_deformations(deformations)
        assert isinstance(result["unified"], trimesh.Trimesh)

    def test_zero_deformations_preserves_template_verts(self):
        pred = _make_predictor(n_verts=32)
        deformations = {"skin": torch.zeros(20, 3)}
        result = pred._apply_deformations(deformations)
        orig = pred.template_mesh.vertices.copy()
        np.testing.assert_allclose(result["unified"].vertices, orig, atol=1e-6)

    def test_nonzero_deformations_shift_vertices(self):
        pred = _make_predictor(n_verts=32)
        shift = torch.ones(20, 3) * 5.0
        deformations = {"skin": shift}
        orig = pred.template_mesh.vertices[:20].copy()
        result = pred._apply_deformations(deformations)
        expected = orig + 5.0
        np.testing.assert_allclose(result["unified"].vertices[:20], expected, atol=1e-5)

    def test_with_affine_params(self):
        pred = _make_predictor(n_verts=32)
        deformations = {"skin": torch.zeros(20, 3)}
        affine = {
            "scale":       torch.tensor([[1.0, 1.0, 1.0]]),
            "translation": torch.tensor([[0.0, 0.0, 0.0]]),
        }
        result = pred._apply_deformations(deformations, affine_params=affine)
        assert "unified" in result


# ── _denormalize_predictions ──────────────────────────────────────────────────

class TestDenormalizePredictions:
    def test_passthrough_without_scalers(self):
        pred = _make_predictor()
        pred.normalizer = {}  # no scalers
        sample = {
            "structure_deformations": {
                "skin": torch.randn(1, 20, 3),
                "FCR":  torch.randn(1, 12, 3),
            }
        }
        result = pred._denormalize_predictions(sample)
        assert "skin" in result
        assert "FCR" in result

    def test_output_is_dict(self):
        pred = _make_predictor()
        sample = {"structure_deformations": {"skin": torch.randn(1, 20, 3)}}
        result = pred._denormalize_predictions(sample)
        assert isinstance(result, dict)

    def test_affine_params_key_excluded(self):
        pred = _make_predictor()
        sample = {
            "structure_deformations": {"skin": torch.randn(1, 20, 3)},
            "affine_params": {"scale": torch.ones(1, 3)},
        }
        result = pred._denormalize_predictions(sample)
        assert "affine_params" not in result

    def test_2d_tensor_strip_batch(self):
        """If tensor is 3D [1, V, 3], it should be stripped to [V, 3]."""
        pred = _make_predictor()
        sample = {"structure_deformations": {"skin": torch.randn(1, 20, 3)}}
        result = pred._denormalize_predictions(sample)
        assert result["skin"].shape == (20, 3)


# ── save_prediction ────────────────────────────────────────────────────────────

class TestSavePrediction:
    def _make_fake_prediction(self):
        pred = _make_predictor(n_verts=32)
        mesh = trimesh.creation.box()
        return {
            "predictions": [
                {"meshes": {"unified": mesh, "skin": mesh}, "deformations": {}},
            ],
            "anthropometric_measurements": {"forearm_length": 250.0},
            "template_info": {
                "num_vertices": 32,
                "num_faces": 1,
                "structures": ["skin", "FCR"],
            },
        }

    def test_creates_output_dir(self):
        pred = _make_predictor()
        fake_pred = self._make_fake_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "result")
            pred.save_prediction(fake_pred, out)
            assert Path(out).is_dir()

    def test_metadata_json_created(self):
        pred = _make_predictor()
        fake_pred = self._make_fake_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "result")
            pred.save_prediction(fake_pred, out)
            meta = Path(out) / "metadata.json"
            assert meta.exists()

    def test_metadata_json_is_valid(self):
        pred = _make_predictor()
        fake_pred = self._make_fake_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "result")
            pred.save_prediction(fake_pred, out)
            with open(Path(out) / "metadata.json") as f:
                data = json.load(f)
            assert "anthropometric_measurements" in data
            assert "num_samples" in data

    def test_mesh_files_created(self):
        pred = _make_predictor()
        fake_pred = self._make_fake_prediction()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "result")
            pred.save_prediction(fake_pred, out, format="ply")
            sample_dir = Path(out) / "sample_0"
            assert sample_dir.is_dir()
            ply_files = list(sample_dir.glob("*.ply"))
            assert len(ply_files) > 0

    def test_none_meshes_skipped(self):
        """Predictions with meshes=None should not cause errors."""
        pred = _make_predictor()
        fake_pred = {
            "predictions": [{"meshes": None, "deformations": {}}],
            "anthropometric_measurements": {},
            "template_info": {"num_vertices": 0, "num_faces": 0, "structures": []},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "result")
            pred.save_prediction(fake_pred, out)   # should not raise
            assert Path(out).is_dir()
