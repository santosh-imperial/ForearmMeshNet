"""
test_meshnet.py — Tests for the full ForearmMeshNet VAE.

Known bugs covered:
  Bug 5 (meshnet.py:159): TemplateAugmentor has no `augment_template_graph`
      method. Calling model.train() + forward() raises AttributeError when
      use_template_augmentation=True.
  Bug 6 (meshnet.py:208-217): The affine-transformed template vertices are
      computed per-structure but never stored; they are silently discarded
      instead of being returned in outputs (so the loss never sees them).
"""

import pytest
import torch
from tests.conftest import BATCH_SIZE, ANTHRO_DIM, LATENT_DIM, V_SKIN, V_FCR


def _build_model(extra=None):
    from forearm_meshnet.models.meshnet import ForearmMeshNet
    cfg = {
        "node_feature_dim":        7,
        "anthro_feature_dim":      ANTHRO_DIM,
        "encoder_hidden_dims":     [32, 64],
        "decoder_hidden_dims":     [64, 32],
        "latent_dim":              LATENT_DIM,
        "num_structures":          2,
        "structure_vertex_counts": {"skin": V_SKIN, "FCR": V_FCR},
        "dropout_rate":            0.0,
        "conv_type":               "gcn",
        "use_template_augmentation": False,
        "use_affine":              False,
        "latent_dropout_p":        0.0,
    }
    if extra:
        cfg.update(extra)
    return ForearmMeshNet(cfg)


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestForearmMeshNetInstantiation:
    def test_basic(self):
        assert _build_model() is not None

    def test_with_affine(self):
        m = _build_model({"use_affine": True})
        assert hasattr(m, "affine")

    def test_without_affine(self):
        m = _build_model({"use_affine": False})
        assert not hasattr(m, "affine")

    def test_parameter_counts_positive(self):
        m = _build_model()
        counts = m.get_num_parameters()
        assert counts["total"]     > 0
        assert counts["trainable"] > 0
        assert counts["encoder"]   > 0
        assert counts["decoder"]   > 0


# ── Forward pass (eval mode) ──────────────────────────────────────────────────

class TestForearmMeshNetForwardEval:
    @pytest.fixture
    def model(self):
        m = _build_model()
        m.eval()
        return m

    def test_required_output_keys(self, model, toy_batch_graph, anthro_features):
        out = model(toy_batch_graph, anthro_features)
        for key in ("structure_deformations", "z", "mu", "logvar", "prior_mu", "prior_logvar"):
            assert key in out, f"Missing key: {key}"

    def test_deformation_shapes(self, model, toy_batch_graph, anthro_features):
        out = model(toy_batch_graph, anthro_features)
        d = out["structure_deformations"]
        assert d["skin"].shape == (BATCH_SIZE, V_SKIN, 3)
        assert d["FCR"].shape  == (BATCH_SIZE, V_FCR,  3)

    def test_latent_shapes(self, model, toy_batch_graph, anthro_features):
        out = model(toy_batch_graph, anthro_features)
        assert out["z"].shape            == (BATCH_SIZE, LATENT_DIM)
        assert out["mu"].shape           == (BATCH_SIZE, LATENT_DIM)
        assert out["logvar"].shape       == (BATCH_SIZE, LATENT_DIM)
        assert out["prior_mu"].shape     == (BATCH_SIZE, LATENT_DIM)
        assert out["prior_logvar"].shape == (BATCH_SIZE, LATENT_DIM)

    def test_no_nan_in_outputs(self, model, toy_batch_graph, anthro_features):
        out = model(toy_batch_graph, anthro_features)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in output['{k}']"
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        assert not torch.isnan(sv).any(), f"NaN in output['{k}']['{sk}']"

    def test_eval_mode_uses_prior_mean(self, toy_batch_graph, anthro_features):
        """In eval mode z should equal prior_mu (no reparametrisation noise)."""
        model = _build_model()
        model.eval()
        out = model(toy_batch_graph, anthro_features)
        assert torch.allclose(out["z"], out["prior_mu"], atol=1e-6)


# ── Forward pass (train mode) ─────────────────────────────────────────────────

class TestForearmMeshNetForwardTrain:
    def test_train_no_augmentation(self, toy_batch_graph, anthro_features):
        model = _build_model({"use_template_augmentation": False})
        model.train()
        out = model(toy_batch_graph, anthro_features)
        assert "structure_deformations" in out

    def test_train_with_augmentation_bug5(self, toy_batch_graph, anthro_features):
        """
        Bug 5: model.train() + use_template_augmentation=True calls
        self.template_augmentor.augment_template_graph(...) which does not
        exist on TemplateAugmentor → AttributeError before the fix.
        """
        model = _build_model({"use_template_augmentation": True})
        model.train()
        out = model(toy_batch_graph, anthro_features)
        assert "structure_deformations" in out

    def test_train_mode_z_differs_from_prior_mean(self, toy_batch_graph, anthro_features):
        """Training draws z from the posterior, so z ≠ prior_mu in general."""
        torch.manual_seed(42)
        model = _build_model()
        model.train()
        out = model(toy_batch_graph, anthro_features)
        assert not torch.allclose(out["z"], out["prior_mu"], atol=1e-6)


# ── Affine transformation ─────────────────────────────────────────────────────

class TestForearmMeshNetAffine:
    def test_affine_params_in_outputs(self, toy_batch_graph, anthro_features):
        model = _build_model({"use_affine": True})
        model.eval()
        template_vertices = {
            "skin": torch.randn(BATCH_SIZE, V_SKIN, 3),
            "FCR":  torch.randn(BATCH_SIZE, V_FCR,  3),
        }
        out = model(toy_batch_graph, anthro_features, template_vertices=template_vertices)
        assert "affine_params" in out

    def test_affine_vertices_stored_in_outputs_bug6(self, toy_batch_graph, anthro_features):
        """
        Bug 6: affine_template is computed inside the loop but the result is
        discarded (never written to any dict or returned). The loss function
        therefore always receives the un-transformed template.
        After the fix, outputs should contain the per-structure affine vertices
        so they can be passed to the loss.
        """
        model = _build_model({"use_affine": True})
        model.eval()
        template_vertices = {
            "skin": torch.randn(BATCH_SIZE, V_SKIN, 3),
            "FCR":  torch.randn(BATCH_SIZE, V_FCR,  3),
        }
        out = model(toy_batch_graph, anthro_features, template_vertices=template_vertices)
        assert "affine_vertices" in out, (
            "Bug 6: affine-transformed template vertices are computed but "
            "never stored in outputs"
        )
        # Each structure should have a matching tensor
        for name in ("skin", "FCR"):
            assert name in out["affine_vertices"]
            assert out["affine_vertices"][name].shape == template_vertices[name].shape

    def test_affine_skipped_when_no_template_vertices(self, toy_batch_graph, anthro_features):
        model = _build_model({"use_affine": True})
        model.eval()
        out = model(toy_batch_graph, anthro_features, template_vertices=None)
        assert "structure_deformations" in out
        assert "affine_params" not in out


# ── Sampling ──────────────────────────────────────────────────────────────────

class TestForearmMeshNetSample:
    def test_returns_list_of_correct_length(self, anthro_features):
        model = _build_model()
        samples = model.sample(anthro_features[:1], n_samples=4)
        assert isinstance(samples, list)
        assert len(samples) == 4

    def test_sample_deformation_shapes(self, anthro_features):
        model = _build_model()
        samples = model.sample(anthro_features[:1], n_samples=2)
        for s in samples:
            assert s["skin"].shape == (1, V_SKIN, 3)
            assert s["FCR"].shape  == (1, V_FCR,  3)

    def test_1d_anthro_input_auto_batched(self, anthro_features):
        model = _build_model()
        samples = model.sample(anthro_features[0], n_samples=1)
        assert len(samples) == 1

    def test_samples_vary_across_draws(self, anthro_features):
        """Different samples from the prior should not be identical."""
        torch.manual_seed(1)
        model = _build_model()
        samples = model.sample(anthro_features[:1], n_samples=3)
        s0 = samples[0]["skin"]
        s1 = samples[1]["skin"]
        assert not torch.allclose(s0, s1), "Samples from the prior should differ"


# ── KL divergence helper ──────────────────────────────────────────────────────

class TestKLDivergence:
    def test_kl_zero_for_identical_distributions(self):
        model = _build_model()
        mu     = torch.randn(BATCH_SIZE, LATENT_DIM)
        logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
        kl = model.compute_kl_divergence(mu, logvar, mu, logvar)
        assert kl.abs().sum().item() < 1e-4

    def test_kl_non_negative(self):
        model = _build_model()
        mu          = torch.randn(BATCH_SIZE, LATENT_DIM)
        logvar      = torch.randn(BATCH_SIZE, LATENT_DIM)
        prior_mu    = torch.randn(BATCH_SIZE, LATENT_DIM)
        prior_logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
        kl = model.compute_kl_divergence(mu, logvar, prior_mu, prior_logvar)
        assert (kl >= -1e-4).all(), "KL divergence must be non-negative"

    def test_kl_output_shape(self):
        model = _build_model()
        mu     = torch.randn(BATCH_SIZE, LATENT_DIM)
        logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
        kl = model.compute_kl_divergence(mu, logvar, mu, logvar)
        # sum(dim=-1) → [batch_size]
        assert kl.shape == (BATCH_SIZE,)
