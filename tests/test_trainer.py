"""
test_trainer.py — Tests for Trainer helper functions and initialization.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from tests.conftest import (
    BATCH_SIZE, V_SKIN, V_FCR, ANTHRO_DIM, LATENT_DIM, NODE_FEAT_DIM,
    STRUCTURES, _ring_edges,
)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _build_edges_from_faces(faces_np):
    from forearm_meshnet.training.trainer import _build_edges_from_faces as _bef
    return _bef(faces_np)


def _uniform_graph_laplacian(n, edges):
    from forearm_meshnet.training.trainer import _uniform_graph_laplacian as _ugl
    return _ugl(n, edges)


# ── _build_edges_from_faces ───────────────────────────────────────────────────

class TestBuildEdgesFromFaces:
    def test_single_triangle(self):
        faces = np.array([[0, 1, 2]])
        edges = _build_edges_from_faces(faces)
        assert edges is not None
        assert edges.shape == (3, 2)   # 3 unique edges from 1 triangle

    def test_two_triangles_shared_edge(self):
        # triangles share edge (1,2)
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        edges = _build_edges_from_faces(faces)
        assert edges is not None
        # 4 unique edges: (0,1),(0,2),(1,2),(1,3),(2,3) = 5 but (1,2) shared → still unique
        assert edges.shape[1] == 2

    def test_none_faces_returns_none(self):
        assert _build_edges_from_faces(None) is None

    def test_empty_faces_returns_none(self):
        assert _build_edges_from_faces(np.zeros((0, 3), dtype=int)) is None

    def test_output_dtype_long(self):
        faces = np.array([[0, 1, 2]])
        edges = _build_edges_from_faces(faces)
        assert edges.dtype == torch.long

    def test_edges_are_undirected_canonical(self):
        """Each edge (i,j) should have i < j (canonical form)."""
        faces = np.array([[2, 0, 1]])
        edges = _build_edges_from_faces(faces)
        for i, j in edges.tolist():
            assert i < j, f"Edge ({i},{j}) not in canonical form"


# ── _uniform_graph_laplacian ──────────────────────────────────────────────────

class TestUniformGraphLaplacian:
    def test_returns_none_for_none_edges(self):
        L = _uniform_graph_laplacian(10, None)
        assert L is None

    def test_shape(self):
        n = 8
        edges = torch.tensor([[0,1],[1,2],[2,3]], dtype=torch.long)
        L = _uniform_graph_laplacian(n, edges)
        assert L.shape == (n, n)

    def test_row_sums_zero(self):
        """L = D - A → row sums = 0."""
        n = 5
        edges = torch.tensor([[0,1],[1,2],[2,3],[3,4]], dtype=torch.long)
        L = _uniform_graph_laplacian(n, edges)
        L_dense = L.to_dense()
        row_sums = L_dense.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(n), atol=1e-6)

    def test_diagonal_equals_degree(self):
        """Diagonal entry i should equal the degree of vertex i."""
        n = 4
        edges = torch.tensor([[0,1],[1,2],[0,3]], dtype=torch.long)
        L = _uniform_graph_laplacian(n, edges)
        L_dense = L.to_dense()
        # vertex 0 connects to 1 and 3 → degree 2
        assert L_dense[0, 0].item() == 2.0
        # vertex 1 connects to 0 and 2 → degree 2
        assert L_dense[1, 1].item() == 2.0
        # vertex 2 connects to 1 → degree 1
        assert L_dense[2, 2].item() == 1.0

    def test_is_sparse(self):
        n = 6
        edges = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        L = _uniform_graph_laplacian(n, edges)
        assert L.is_sparse


# ── Trainer initialisation ────────────────────────────────────────────────────

def _make_samples(n=6):
    """Minimal samples compatible with ForearmDataset."""
    from torch_geometric.data import Data
    samples = []
    for _ in range(n):
        row = torch.arange(V_SKIN)
        col = (row + 1) % V_SKIN
        samples.append({
            "anthropometric_features": torch.randn(ANTHRO_DIM),
            "structure_deformations": {
                "skin": torch.randn(V_SKIN, 3),
                "FCR":  torch.randn(V_FCR, 3),
            },
            "structure_info": {
                "skin": {"vertex_range": (0, V_SKIN),          "num_vertices": V_SKIN},
                "FCR":  {"vertex_range": (V_SKIN, V_SKIN + V_FCR), "num_vertices": V_FCR},
            },
            "unified_template_graph": Data(
                x=torch.randn(V_SKIN, NODE_FEAT_DIM),
                edge_index=torch.stack([torch.cat([row, col]), torch.cat([col, row])]),
            ),
        })
    return samples


def _make_model_config():
    return {
        "node_feature_dim":        NODE_FEAT_DIM,
        "anthro_feature_dim":      ANTHRO_DIM,
        "encoder_hidden_dims":     [32, 16],
        "decoder_hidden_dims":     [16, 32],
        "latent_dim":              LATENT_DIM,
        "num_structures":          len(STRUCTURES),
        "structure_vertex_counts": {s: (V_SKIN if s == "skin" else V_FCR) for s in STRUCTURES},
        "dropout_rate":            0.0,
        "conv_type":               "gcn",
        "use_template_augmentation": False,
        "use_affine":              False,
        "latent_dropout_p":        0.0,
    }


def _make_trainer(tmp_path):
    from forearm_meshnet.models.meshnet import ForearmMeshNet
    from forearm_meshnet.data.dataset import ForearmDataset
    from forearm_meshnet.training.trainer import Trainer

    samples = _make_samples(n=8)
    train_ds = ForearmDataset(samples[:6])
    val_ds   = ForearmDataset(samples[6:])
    model    = ForearmMeshNet(_make_model_config())

    trainer_cfg = {
        "batch_size": 2,
        "num_workers": 0,
        "optimizer": {"type": "Adam", "lr": 1e-3},
        "scheduler": {"type": "StepLR", "step_size": 10, "gamma": 0.5},
    }
    return Trainer(model, train_ds, val_ds, trainer_cfg, output_dir=str(tmp_path))


class TestTrainerInit:
    def test_instantiates(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer is not None

    def test_output_dirs_created(self, tmp_path):
        _make_trainer(tmp_path)
        assert (tmp_path / "checkpoints").is_dir()
        assert (tmp_path / "logs").is_dir()

    def test_optimizer_is_adam(self, tmp_path):
        import torch.optim as optim
        trainer = _make_trainer(tmp_path)
        assert isinstance(trainer.optimizer, optim.Adam)

    def test_scheduler_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.scheduler is not None

    def test_criterion_created(self, tmp_path):
        from forearm_meshnet.models.losses import CombinedLoss
        trainer = _make_trainer(tmp_path)
        assert isinstance(trainer.criterion, CombinedLoss)

    def test_curriculum_manager_created(self, tmp_path):
        from forearm_meshnet.training.curriculum import CurriculumManager
        trainer = _make_trainer(tmp_path)
        assert isinstance(trainer.curriculum_manager, CurriculumManager)

    def test_metrics_evaluator_created(self, tmp_path):
        from forearm_meshnet.training.metrics import MeshEvaluationMetrics
        trainer = _make_trainer(tmp_path)
        assert isinstance(trainer.metrics_evaluator, MeshEvaluationMetrics)

    def test_adamw_optimizer(self, tmp_path):
        import torch.optim as optim
        from forearm_meshnet.models.meshnet import ForearmMeshNet
        from forearm_meshnet.data.dataset import ForearmDataset
        from forearm_meshnet.training.trainer import Trainer

        samples = _make_samples(n=4)
        model = ForearmMeshNet(_make_model_config())
        cfg = {
            "batch_size": 2,
            "num_workers": 0,
            "optimizer": {"type": "AdamW", "lr": 1e-3},
        }
        trainer = Trainer(model, ForearmDataset(samples[:3]),
                          ForearmDataset(samples[3:]), cfg, output_dir=str(tmp_path))
        assert isinstance(trainer.optimizer, optim.AdamW)

    def test_unknown_optimizer_raises(self, tmp_path):
        from forearm_meshnet.models.meshnet import ForearmMeshNet
        from forearm_meshnet.data.dataset import ForearmDataset
        from forearm_meshnet.training.trainer import Trainer

        samples = _make_samples(n=4)
        model = ForearmMeshNet(_make_model_config())
        cfg = {"batch_size": 2, "num_workers": 0,
               "optimizer": {"type": "SGDMomentum"}}
        with pytest.raises(ValueError):
            Trainer(model, ForearmDataset(samples[:3]),
                    ForearmDataset(samples[3:]), cfg, output_dir=str(tmp_path))
