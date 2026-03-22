"""
test_decoder.py — Tests for StructureAwareDecoder, FiLMParams,
                  AnthroAffine, and TemplateAugmentor.
"""

import pytest
import torch
from tests.conftest import BATCH_SIZE, ANTHRO_DIM, LATENT_DIM, V_SKIN, V_FCR


# ── FiLMParams ───────────────────────────────────────────────────────────────

class TestFiLMParams:
    def test_output_list_length_matches_layers(self):
        from forearm_meshnet.models.decoder import FiLMParams
        channels = [32, 64, 128]
        film = FiLMParams(c_dim=ANTHRO_DIM, channels_per_layer=channels)
        gammas, betas = film(torch.randn(BATCH_SIZE, ANTHRO_DIM))
        assert len(gammas) == len(channels)
        assert len(betas)  == len(channels)

    def test_output_shapes_match_channels(self):
        from forearm_meshnet.models.decoder import FiLMParams
        channels = [32, 64]
        film = FiLMParams(c_dim=ANTHRO_DIM, channels_per_layer=channels)
        gammas, betas = film(torch.randn(BATCH_SIZE, ANTHRO_DIM))
        for i, ch in enumerate(channels):
            assert gammas[i].shape == (BATCH_SIZE, ch)
            assert betas[i].shape  == (BATCH_SIZE, ch)

    def test_zero_initialised_near_zero_output(self):
        """Last layer is zero-initialised → params ≈ 0 at init for zero input."""
        from forearm_meshnet.models.decoder import FiLMParams
        film = FiLMParams(c_dim=ANTHRO_DIM, channels_per_layer=[32])
        c = torch.zeros(1, ANTHRO_DIM)
        gammas, betas = film(c)
        assert gammas[0].abs().max().item() < 1e-5
        assert betas[0].abs().max().item()  < 1e-5


# ── AnthroAffine ─────────────────────────────────────────────────────────────

class TestAnthroAffine:
    @pytest.fixture
    def affine(self):
        from forearm_meshnet.models.decoder import AnthroAffine
        return AnthroAffine(c_dim=ANTHRO_DIM)

    def test_forward_output_shapes(self, affine):
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        scale, translation = affine(c)
        assert scale.shape       == (BATCH_SIZE, 3)
        assert translation.shape == (BATCH_SIZE, 3)

    def test_scale_strictly_positive(self, affine):
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        scale, _ = affine(c)
        assert (scale > 0).all(), "Scale must be strictly positive"

    def test_apply_transform_output_shape(self, affine):
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        scale, translation = affine(c)
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        out = affine.apply_transform(verts, scale, translation)
        assert out.shape == (BATCH_SIZE, V_SKIN, 3)

    def test_apply_transform_identity(self, affine):
        """scale=1, translation=0 → vertices unchanged."""
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        scale = torch.ones(BATCH_SIZE, 3)
        translation = torch.zeros(BATCH_SIZE, 3)
        out = affine.apply_transform(verts, scale, translation)
        assert torch.allclose(out, verts, atol=1e-6)


# ── StructureAwareDecoder ────────────────────────────────────────────────────

class TestStructureAwareDecoder:
    @pytest.fixture
    def decoder(self):
        from forearm_meshnet.models.decoder import StructureAwareDecoder
        return StructureAwareDecoder(
            latent_dim=LATENT_DIM,
            anthro_dim=ANTHRO_DIM,
            hidden_dims=[64, 32],
            num_vertices_per_structure={"skin": V_SKIN, "FCR": V_FCR},
            dropout_rate=0.0,
        )

    def test_instantiation(self, decoder):
        assert decoder is not None

    def test_forward_output_keys(self, decoder):
        decoder.eval()
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        out = decoder(z, c)
        assert "skin" in out
        assert "FCR"  in out

    def test_forward_output_shapes(self, decoder):
        decoder.eval()
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        out = decoder(z, c)
        assert out["skin"].shape == (BATCH_SIZE, V_SKIN, 3)
        assert out["FCR"].shape  == (BATCH_SIZE, V_FCR,  3)

    def test_no_nan_in_output(self, decoder):
        decoder.eval()
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        out = decoder(z, c)
        for name, deform in out.items():
            assert not torch.isnan(deform).any(), f"NaN in '{name}' deformation"

    def test_conditioning_affects_output(self, decoder):
        """
        After a gradient step, different anthropometric features should produce
        different deformations. At init, FiLM and scale heads are zero-initialised
        for stability, so we apply a small update first.
        """
        import torch.optim as optim
        decoder.train()
        opt = optim.SGD(decoder.parameters(), lr=0.1)
        z = torch.randn(1, LATENT_DIM)
        c_train = torch.randn(1, ANTHRO_DIM)
        # One gradient step to break zero init
        out_train = decoder(z, c_train)
        loss = sum(v.sum() for v in out_train.values())
        loss.backward()
        opt.step()

        decoder.eval()
        c1 = torch.zeros(1, ANTHRO_DIM)
        c2 = torch.ones(1, ANTHRO_DIM) * 5.0
        out1 = decoder(z, c1)
        out2 = decoder(z, c2)
        diffs = [not torch.allclose(out1[k], out2[k]) for k in out1]
        assert any(diffs), "Anthropometric conditioning has no effect after training step"

    def test_decode_deterministic_no_gradient(self, decoder):
        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        out = decoder.decode_deterministic(z, c)
        for name, deform in out.items():
            assert not deform.requires_grad, f"'{name}' should not require grad"

    def test_backward_gradients_exist(self, decoder):
        decoder.train()
        z = torch.randn(BATCH_SIZE, LATENT_DIM, requires_grad=True)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        out = decoder(z, c)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert z.grad is not None


# ── TemplateAugmentor ────────────────────────────────────────────────────────

class TestTemplateAugmentor:
    @pytest.fixture
    def augmentor(self):
        from forearm_meshnet.models.decoder import TemplateAugmentor
        return TemplateAugmentor({
            "augment_scale":     0.1,
            "augment_rotate":    0.1,
            "augment_translate": 5.0,
            "augment_noise":     1.0,
        })

    def test_output_shape_preserved(self, augmentor):
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        out = augmentor(verts, training=False)
        assert out.shape == verts.shape

    def test_eval_mode_is_identity(self, augmentor):
        augmentor.eval()
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        out = augmentor(verts, training=False)
        assert torch.equal(out, verts), "Eval mode should return vertices unchanged"

    def test_train_mode_modifies_vertices(self, augmentor):
        torch.manual_seed(0)
        augmentor.train()
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        out = augmentor(verts, training=True)
        assert not torch.equal(out, verts), "Training mode should modify vertices"

    def test_no_augmentation_config(self):
        from forearm_meshnet.models.decoder import TemplateAugmentor
        aug = TemplateAugmentor({
            "augment_scale": 0.0,
            "augment_rotate": 0.0,
            "augment_translate": 0.0,
            "augment_noise": 0.0,
        })
        aug.train()
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        out = aug(verts, training=True)
        assert torch.allclose(out, verts, atol=1e-6)
