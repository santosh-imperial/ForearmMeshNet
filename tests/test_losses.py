"""
test_losses.py — Tests for all loss functions in models/losses.py.

Known bugs covered:
  Bug 3 (losses.py:548): `device` used before assignment in
      _compute_structure_laplacian_loss → UnboundLocalError when
      affine_meshes is not None.
  Bug 4 (losses.py:595): `source_verts.device` called on a numpy array
      from template_meshes → AttributeError.
  Bug 7 (losses.py:271, 283): CombinedLoss.__init__ accesses raw `config`
      parameter (not self.config) when config=None → TypeError/AttributeError.
"""

import pytest
import torch
import numpy as np
from tests.conftest import make_sparse_laplacian, BATCH_SIZE, V_SKIN, V_FCR


# ── ChamferDistance ──────────────────────────────────────────────────────────

class TestChamferDistance:
    @pytest.fixture
    def loss(self):
        from forearm_meshnet.models.losses import ChamferDistance
        return ChamferDistance()

    def test_identical_clouds_zero(self, loss):
        pts = torch.randn(BATCH_SIZE, 10, 3)
        assert loss(pts, pts).item() < 1e-5

    def test_output_is_scalar(self, loss):
        a = torch.randn(BATCH_SIZE, 10, 3)
        b = torch.randn(BATCH_SIZE, 12, 3)
        assert loss(a, b).shape == torch.Size([])

    def test_non_negative(self, loss):
        a = torch.randn(BATCH_SIZE, 10, 3)
        b = torch.randn(BATCH_SIZE, 10, 3)
        assert loss(a, b).item() >= 0.0

    def test_gradients_flow(self, loss):
        a = torch.randn(BATCH_SIZE, 10, 3, requires_grad=True)
        b = torch.randn(BATCH_SIZE, 10, 3)
        out = loss(a, b)
        out.backward()
        assert a.grad is not None


# ── EdgeLengthLoss ───────────────────────────────────────────────────────────

class TestEdgeLengthLoss:
    @pytest.fixture
    def loss(self):
        from forearm_meshnet.models.losses import EdgeLengthLoss
        return EdgeLengthLoss()

    def test_identical_verts_zero(self, loss):
        verts = torch.randn(BATCH_SIZE, 10, 3)
        edges = torch.randint(0, 10, (8, 2)).long()
        assert loss(verts, verts, edges).item() < 1e-5

    def test_output_is_scalar(self, loss):
        verts = torch.randn(BATCH_SIZE, 10, 3)
        edges = torch.randint(0, 10, (8, 2)).long()
        out = loss(verts, verts + 0.5, edges)
        assert out.shape == torch.Size([])

    def test_non_negative(self, loss):
        verts = torch.randn(BATCH_SIZE, 10, 3)
        edges = torch.randint(0, 10, (8, 2)).long()
        assert loss(verts, verts + torch.randn_like(verts), edges).item() >= 0.0


# ── NormalConsistencyLoss ────────────────────────────────────────────────────

class TestNormalConsistencyLoss:
    @pytest.fixture
    def loss(self):
        from forearm_meshnet.models.losses import NormalConsistencyLoss
        return NormalConsistencyLoss()

    def test_identical_verts_zero(self, loss):
        # Use sequential, non-degenerate faces to avoid zero-area triangles
        verts = torch.randn(BATCH_SIZE, 10, 3)
        faces = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4],
                               [3, 4, 5], [4, 5, 6], [5, 6, 7]], dtype=torch.long)
        assert loss(verts, verts, faces).item() < 1e-4

    def test_output_is_scalar(self, loss):
        verts = torch.randn(BATCH_SIZE, 10, 3)
        faces = torch.randint(0, 10, (6, 3)).long()
        assert loss(verts, verts + 0.1, faces).shape == torch.Size([])

    def test_non_negative(self, loss):
        verts = torch.randn(BATCH_SIZE, 10, 3)
        faces = torch.randint(0, 10, (6, 3)).long()
        assert loss(verts, verts + torch.randn_like(verts) * 0.5, faces).item() >= 0.0


# ── LaplacianSmoothingLoss ───────────────────────────────────────────────────

class TestLaplacianSmoothingLoss:
    @pytest.fixture
    def loss(self):
        from forearm_meshnet.models.losses import LaplacianSmoothingLoss
        return LaplacianSmoothingLoss()

    def test_constant_verts_near_zero(self, loss):
        """All vertices at the same position → Lv ≈ 0."""
        n = V_SKIN
        lap = make_sparse_laplacian(n)
        verts = torch.ones(BATCH_SIZE, n, 3) * 3.7
        assert loss(verts, lap).item() < 1e-4

    def test_output_is_scalar(self, loss):
        lap = make_sparse_laplacian(V_SKIN)
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        assert loss(verts, lap).shape == torch.Size([])

    def test_non_negative(self, loss):
        lap = make_sparse_laplacian(V_SKIN)
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3)
        assert loss(verts, lap).item() >= 0.0

    def test_gradients_flow(self, loss):
        lap = make_sparse_laplacian(V_SKIN)
        verts = torch.randn(BATCH_SIZE, V_SKIN, 3, requires_grad=True)
        loss(verts, lap).backward()
        assert verts.grad is not None


# ── VolumeLoss ───────────────────────────────────────────────────────────────

class TestVolumeLoss:
    @pytest.fixture
    def loss(self):
        from forearm_meshnet.models.losses import VolumeLoss
        return VolumeLoss()

    def _chain_edges(self, n):
        return torch.stack([torch.arange(0, n - 1), torch.arange(1, n)]).long()

    def test_zero_displacement_equals_one(self, loss):
        """exp(div(0)) = exp(0) = 1 for every vertex → mean = 1."""
        n = V_SKIN
        X = torch.randn(n, 3)
        edges = self._chain_edges(n)
        out = loss(X, X, edges)
        assert abs(out.item() - 1.0) < 1e-4

    def test_output_is_scalar(self, loss):
        n = V_SKIN
        X = torch.randn(n, 3)
        Y = X + torch.randn(n, 3) * 0.01
        edges = self._chain_edges(n)
        assert loss(X, Y, edges).shape == torch.Size([])

    def test_positive_output(self, loss):
        n = V_SKIN
        X = torch.randn(n, 3)
        Y = X + torch.randn(n, 3) * 0.5
        edges = self._chain_edges(n)
        assert loss(X, Y, edges).item() > 0.0


# ── CombinedLoss ─────────────────────────────────────────────────────────────

class TestCombinedLossInit:
    def test_config_none_does_not_crash(self, structure_info):
        """Bug 7: config=None must not raise TypeError or AttributeError."""
        from forearm_meshnet.models.losses import CombinedLoss
        loss = CombinedLoss(structure_info, config=None)
        assert loss is not None

    def test_empty_config(self, structure_info):
        from forearm_meshnet.models.losses import CombinedLoss
        loss = CombinedLoss(structure_info, config={})
        assert loss is not None

    def test_custom_weights(self, structure_info):
        from forearm_meshnet.models.losses import CombinedLoss
        cfg = {"lambda_weights": {"reconstruction": 1.0, "chamfer": 0.0}}
        loss = CombinedLoss(structure_info, config=cfg)
        assert loss.lambda_weights["reconstruction"] == 1.0


class TestCombinedLossForward:
    @pytest.fixture
    def loss_fn(self, structure_info):
        from forearm_meshnet.models.losses import CombinedLoss
        return CombinedLoss(structure_info, config={})

    def test_basic_forward_returns_scalar(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        total, loss_dict = loss_fn(
            pred_deformations, target_deformations, template_meshes, epoch=0
        )
        assert isinstance(total, torch.Tensor)
        assert total.shape == torch.Size([])

    def test_forward_loss_dict_keys(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        _, loss_dict = loss_fn(
            pred_deformations, target_deformations, template_meshes, epoch=0
        )
        assert "reconstruction" in loss_dict
        assert "total" in loss_dict

    def test_forward_all_values_finite(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        _, loss_dict = loss_fn(
            pred_deformations, target_deformations, template_meshes, epoch=50
        )
        for k, v in loss_dict.items():
            assert np.isfinite(v), f"Loss '{k}' is not finite: {v}"

    def test_forward_with_kl(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        mu          = torch.randn(BATCH_SIZE, LATENT_DIM := 8)
        logvar      = torch.zeros(BATCH_SIZE, LATENT_DIM)
        prior_mu    = torch.zeros(BATCH_SIZE, LATENT_DIM)
        prior_logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
        total, loss_dict = loss_fn(
            pred_deformations, target_deformations, template_meshes,
            mu=mu, logvar=logvar, prior_mu=prior_mu, prior_logvar=prior_logvar,
            epoch=60,
        )
        assert "kl" in loss_dict
        assert loss_dict["kl"] >= 0.0

    def test_backward_gradients_exist(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        preds = {k: v.clone().requires_grad_(True) for k, v in pred_deformations.items()}
        total, _ = loss_fn(preds, target_deformations, template_meshes, epoch=0)
        total.backward()
        for name, p in preds.items():
            assert p.grad is not None, f"No gradient for '{name}'"

    # ── Bug 3 ────────────────────────────────────────────────────────────────

    def test_laplacian_loss_with_affine_meshes(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        """
        Bug 3: _compute_structure_laplacian_loss uses `device` before it is
        assigned on the same line where pred[struct_name].device is read.
        With affine_meshes provided the bug path is triggered immediately.
        """
        affine_meshes = {"skin": torch.randn(BATCH_SIZE, V_SKIN, 3)}
        # epoch >= 10 activates the Laplacian loss branch
        total, loss_dict = loss_fn(
            pred_deformations, target_deformations, template_meshes,
            affine_meshes=affine_meshes, epoch=15,
        )
        assert "laplacian" in loss_dict
        assert np.isfinite(loss_dict["laplacian"])

    def test_laplacian_loss_without_affine_meshes(
        self, loss_fn, pred_deformations, target_deformations, template_meshes
    ):
        """
        Bug 3 (else-branch): non-tensor template vertex also triggers the
        device lookup before assignment when affine_meshes is None.
        """
        # Replace skin vertices with numpy to stress-test the else branch
        meshes_numpy = dict(template_meshes)
        meshes_numpy["skin"] = dict(template_meshes["skin"])
        meshes_numpy["skin"]["vertices"] = template_meshes["skin"]["vertices"].numpy()
        total, loss_dict = loss_fn(
            pred_deformations, target_deformations, meshes_numpy,
            affine_meshes=None, epoch=15,
        )
        assert np.isfinite(loss_dict.get("laplacian", 0.0))

    # ── Bug 4 ────────────────────────────────────────────────────────────────

    def test_volume_loss_with_numpy_template_verts(
        self, loss_fn, pred_deformations, target_deformations
    ):
        """
        Bug 4: _compute_structure_volume_loss calls source_verts.device when
        source_verts may be a numpy array → AttributeError before fix.
        """
        from tests.conftest import _ring_edges
        edges_np = _ring_edges(V_SKIN).numpy()
        edges_fcr = torch.stack(
            [torch.arange(0, V_FCR - 1), torch.arange(1, V_FCR)], dim=0
        ).long().numpy()

        template_with_numpy = {
            "skin": {
                "vertices": np.random.randn(BATCH_SIZE, V_SKIN, 3).astype(np.float32),
                "edges":    edges_np,
            },
            "FCR": {
                "vertices": np.random.randn(BATCH_SIZE, V_FCR, 3).astype(np.float32),
                "edges":    edges_fcr,
            },
        }
        total, loss_dict = loss_fn(
            pred_deformations, target_deformations,
            template_with_numpy, affine_meshes=None, epoch=15,
        )
        assert isinstance(total, torch.Tensor)
        assert np.isfinite(loss_dict["total"])

    def test_volume_loss_with_affine_and_numpy_verts(
        self, loss_fn, pred_deformations, target_deformations
    ):
        """
        Bug 4 (affine branch): even with affine_meshes provided, the
        source_verts path for edge lookup must not crash.
        """
        from tests.conftest import _ring_edges
        template_with_numpy = {
            "skin": {
                "vertices": np.random.randn(BATCH_SIZE, V_SKIN, 3).astype(np.float32),
                "edges":    _ring_edges(V_SKIN).numpy(),
            },
        }
        affine_meshes = {"skin": torch.randn(BATCH_SIZE, V_SKIN, 3)}
        total, _ = loss_fn(
            {"skin": pred_deformations["skin"]},
            {"skin": target_deformations["skin"]},
            template_with_numpy, affine_meshes=affine_meshes, epoch=15,
        )
        assert isinstance(total, torch.Tensor)
