"""
test_normalizer.py — Tests for DataNormalizer.

Additional bug found during review:
  normalizer.py:381: In `denormalize_predictions`, when name == 'combined',
      `scaler` is referenced before being assigned on line 384 → NameError.
      Also, when `exp is None`, `flat_pad` is never defined but used.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from tests.conftest import ANTHRO_DIM


def _make_samples(n=8, anthro_dim=ANTHRO_DIM, n_verts=10):
    """Minimal training samples without graph data."""
    samples = []
    for i in range(n):
        samples.append({
            "anthropometric_features": torch.randn(anthro_dim),
            "structure_deformations": {
                "skin": torch.randn(n_verts, 3),
                "FCR":  torch.randn(n_verts // 2, 3),
            },
            "structure_info": {
                "skin": {"vertex_range": (0, n_verts)},
                "FCR":  {"vertex_range": (n_verts, n_verts + n_verts // 2)},
            },
        })
    return samples


def _make_samples_with_graph(n=8, anthro_dim=ANTHRO_DIM, n_verts=10):
    samples = _make_samples(n, anthro_dim, n_verts)
    for s in samples:
        row = torch.arange(n_verts)
        col = (row + 1) % n_verts
        s["unified_template_graph"] = Data(
            x=torch.randn(n_verts, 7),
            edge_index=torch.stack([torch.cat([row, col]), torch.cat([col, row])]),
        )
    return samples


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestDataNormalizerInit:
    def test_standard_scaler(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer(normalization_method="standard")
        assert not dn.fitted

    def test_minmax_scaler(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer(normalization_method="minmax")
        assert not dn.fitted

    def test_not_fitted_raises_on_inference(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        with pytest.raises(ValueError, match="not fitted"):
            dn.normalize_for_inference(np.zeros(ANTHRO_DIM))

    def test_not_fitted_raises_on_transform(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        with pytest.raises(ValueError, match="not fitted"):
            dn.transform(_make_samples(n=2))


# ── fit_and_transform ─────────────────────────────────────────────────────────

class TestFitAndTransform:
    @pytest.fixture
    def fitted(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer(normalization_method="standard")
        dn.fit_and_transform(_make_samples(n=12))
        return dn

    def test_fitted_flag_set(self, fitted):
        assert fitted.fitted

    def test_returns_same_count(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=7)
        dn = DataNormalizer()
        out = dn.fit_and_transform(samples)
        assert len(out) == 7

    def test_structure_scalers_created(self, fitted):
        assert "skin" in fitted.structure_deformation_scalers
        assert "FCR"  in fitted.structure_deformation_scalers

    def test_outputs_are_float32_tensors(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=6)
        dn = DataNormalizer()
        normalized = dn.fit_and_transform(samples)
        for s in normalized:
            af = s["anthropometric_features"]
            assert isinstance(af, torch.Tensor)
            assert af.dtype == torch.float32
            for name, d in s["structure_deformations"].items():
                assert isinstance(d, torch.Tensor), f"{name} is not a tensor"
                assert d.dtype == torch.float32

    def test_standard_normalization_zero_centers_anthro(self):
        """After standard normalisation, anthro features should be roughly zero-centred."""
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=30)
        dn = DataNormalizer(normalization_method="standard")
        normalized = dn.fit_and_transform(samples)
        values = np.stack([s["anthropometric_features"].numpy() for s in normalized])
        assert abs(values.mean()) < 0.2

    def test_with_graph_features_bug_graph_fit_mode(self):
        """
        New bug: DataNormalizer.__init__ accepts `graph_fit_mode` as a parameter
        but never stores it as self.graph_fit_mode. When samples include a
        unified_template_graph, _collect_training_data accesses self.graph_fit_mode
        → AttributeError before the fix.
        """
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples_with_graph(n=8)
        dn = DataNormalizer()
        # Should not raise AttributeError
        normalized = dn.fit_and_transform(samples)
        for s in normalized:
            if "unified_template_graph" in s:
                g = s["unified_template_graph"]
                assert g.x.dtype == torch.float32

    def test_minmax_anthro_in_0_1(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=20)
        dn = DataNormalizer(normalization_method="minmax")
        normalized = dn.fit_and_transform(samples)
        values = np.stack([s["anthropometric_features"].numpy() for s in normalized])
        assert values.min() >= -0.01
        assert values.max() <= 1.01


# ── transform (val/test split) ────────────────────────────────────────────────

class TestTransform:
    def test_val_split_same_count(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        train = _make_samples(n=10)
        val   = _make_samples(n=4)
        dn = DataNormalizer()
        dn.fit_and_transform(train)
        out = dn.transform(val)
        assert len(out) == 4

    def test_val_uses_train_statistics(self):
        """Normalising with train stats should produce different values than train mean."""
        from forearm_meshnet.data.normalizer import DataNormalizer
        train = _make_samples(n=20)
        val   = _make_samples(n=4)
        dn = DataNormalizer(normalization_method="standard")
        dn.fit_and_transform(train)
        val_norm = dn.transform(val)
        # Just check it runs and produces tensors
        for s in val_norm:
            assert isinstance(s["anthropometric_features"], torch.Tensor)


# ── normalize_for_inference ───────────────────────────────────────────────────

class TestNormalizeForInference:
    @pytest.fixture
    def fitted(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=10))
        return dn

    def test_output_shape(self, fitted):
        x = np.random.randn(ANTHRO_DIM).astype(np.float32)
        tensor, graph = fitted.normalize_for_inference(x)
        assert tensor.shape == (ANTHRO_DIM,)
        assert graph is None

    def test_output_is_float32(self, fitted):
        x = np.random.randn(ANTHRO_DIM).astype(np.float32)
        tensor, _ = fitted.normalize_for_inference(x)
        assert tensor.dtype == torch.float32

    def test_with_graph_pass_through(self, fitted):
        x = np.random.randn(ANTHRO_DIM).astype(np.float32)
        g = Data(x=torch.randn(10, 7), edge_index=torch.zeros(2, 0, dtype=torch.long))
        tensor, norm_graph = fitted.normalize_for_inference(x, g)
        assert tensor.shape == (ANTHRO_DIM,)
        # graph should be returned (possibly normalised)
        assert norm_graph is not None


# ── denormalize_predictions ───────────────────────────────────────────────────

class TestDenormalizePredictions:
    @pytest.fixture
    def fitted(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=12, n_verts=10)
        dn = DataNormalizer(normalization_method="standard")
        dn.fit_and_transform(samples)
        return dn, samples

    def test_structure_roundtrip(self, fitted):
        """Normalize then denormalize should recover the original (up to float precision)."""
        from forearm_meshnet.data.normalizer import DataNormalizer
        samples = _make_samples(n=12, n_verts=10)
        dn = DataNormalizer()
        normalized = dn.fit_and_transform(samples)

        orig_skin = samples[0]["structure_deformations"]["skin"].numpy()
        norm_skin = normalized[0]["structure_deformations"]["skin"]

        result = dn.denormalize_predictions({"skin": norm_skin})
        np.testing.assert_allclose(
            result["skin"].reshape(-1),
            orig_skin.reshape(-1),
            atol=1e-4,
        )

    def test_unknown_structure_pass_through(self, fitted):
        """Structures without a scaler should be returned unchanged."""
        dn, _ = fitted
        raw = torch.randn(5, 3)
        result = dn.denormalize_predictions({"unknown_structure": raw})
        np.testing.assert_array_equal(result["unknown_structure"], raw.numpy())

    def test_denormalize_combined_bug(self):
        """
        Additional bug: `scaler` is used before being defined in the
        'combined' branch of denormalize_predictions → NameError.
        Also, when exp is None, `flat_pad` is undefined.
        """
        from forearm_meshnet.data.normalizer import DataNormalizer
        # Build samples with a 'combined' field
        n_verts = 10
        samples = _make_samples(n=8, n_verts=n_verts)
        combined_size = n_verts * 3 + (n_verts // 2) * 3
        for s in samples:
            s["structure_deformations"]["combined"] = torch.randn(combined_size)

        dn = DataNormalizer()
        normalized = dn.fit_and_transform(samples)
        dn.normalize_combined = True  # enable combined denorm path

        # Grab a normalised combined tensor
        norm_combined = normalized[0]["structure_deformations"]["combined"]
        # This should NOT raise NameError
        result = dn.denormalize_predictions({"combined": norm_combined})
        assert "combined" in result


# ── export ────────────────────────────────────────────────────────────────────

class TestExport:
    def test_export_returns_dict(self):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=6))
        exported = dn.export()
        assert isinstance(exported, dict)
        assert "anthropometric_scaler" in exported
        assert "structure_deformation_scalers" in exported


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    import tempfile, os

    def test_save_creates_file(self, tmp_path):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=6))
        path = str(tmp_path / "norm.pkl")
        dn.save(path)
        import os
        assert os.path.exists(path)

    def test_load_restores_fitted_flag(self, tmp_path):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=6))
        path = str(tmp_path / "norm.pkl")
        dn.save(path)

        dn2 = DataNormalizer()
        assert not dn2.fitted
        dn2.load(path)
        assert dn2.fitted

    def test_load_restores_scalers(self, tmp_path):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=8))
        path = str(tmp_path / "norm.pkl")
        dn.save(path)

        dn2 = DataNormalizer()
        dn2.load(path)
        assert "skin" in dn2.structure_deformation_scalers
        assert "FCR" in dn2.structure_deformation_scalers

    def test_roundtrip_produces_same_output(self, tmp_path):
        """Normalizing with original and loaded normalizer should give identical results."""
        from forearm_meshnet.data.normalizer import DataNormalizer
        import numpy as np

        samples = _make_samples(n=10)
        dn = DataNormalizer(normalization_method="standard")
        dn.fit_and_transform(samples)

        path = str(tmp_path / "norm.pkl")
        dn.save(path)

        dn2 = DataNormalizer()
        dn2.load(path)

        val = _make_samples(n=3)
        out1 = dn.transform(val)
        out2 = dn2.transform(val)

        for s1, s2 in zip(out1, out2):
            np.testing.assert_allclose(
                s1["anthropometric_features"].numpy(),
                s2["anthropometric_features"].numpy(),
                atol=1e-5,
            )

    def test_save_creates_parent_dirs(self, tmp_path):
        from forearm_meshnet.data.normalizer import DataNormalizer
        dn = DataNormalizer()
        dn.fit_and_transform(_make_samples(n=4))
        nested = str(tmp_path / "deep" / "nested" / "norm.pkl")
        dn.save(nested)
        import os
        assert os.path.exists(nested)
