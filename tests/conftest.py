"""
Shared fixtures for ForearmMeshNet test suite.
"""

import sys
import os
import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 2
V_SKIN        = 20
V_FCR         = 12
NODE_FEAT_DIM = 7
ANTHRO_DIM    = 16
LATENT_DIM    = 8
STRUCTURES    = ["skin", "FCR"]


# ── Helper (also used directly in some test files) ────────────────────────────
def make_sparse_laplacian(n: int) -> torch.Tensor:
    """Ring-graph Laplacian as a sparse COO tensor of shape [n, n]."""
    rows, cols, vals = [], [], []
    for i in range(n):
        j = (i + 1) % n
        k = (i - 1) % n
        rows += [i, i, i]
        cols += [i, j, k]
        vals += [2.0, -1.0, -1.0]
    idx = torch.tensor([rows, cols], dtype=torch.long)
    v   = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, v, (n, n)).coalesce()


def _ring_edges(n: int) -> torch.Tensor:
    """Undirected ring-graph edge_index [2, 2n]."""
    row = torch.arange(n)
    col = (row + 1) % n
    return torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0).long()


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def model_config():
    return {
        "node_feature_dim":        NODE_FEAT_DIM,
        "anthro_feature_dim":      ANTHRO_DIM,
        "encoder_hidden_dims":     [32, 64],
        "decoder_hidden_dims":     [64, 32],
        "latent_dim":              LATENT_DIM,
        "num_structures":          len(STRUCTURES),
        "structure_vertex_counts": {s: (V_SKIN if s == "skin" else V_FCR) for s in STRUCTURES},
        "dropout_rate":            0.0,
        "conv_type":               "gcn",
        "use_template_augmentation": False,
        "use_affine":              False,
        "latent_dropout_p":        0.0,
    }


@pytest.fixture
def structure_info():
    return {
        "skin": {"vertex_range": (0, V_SKIN),         "num_vertices": V_SKIN},
        "FCR":  {"vertex_range": (V_SKIN, V_SKIN + V_FCR), "num_vertices": V_FCR},
    }


@pytest.fixture
def toy_graph():
    """Single PyG graph with ring connectivity and NODE_FEAT_DIM features."""
    n = V_SKIN
    return Data(
        x=torch.randn(n, NODE_FEAT_DIM),
        edge_index=_ring_edges(n),
        pos=torch.randn(n, 3),
    )


@pytest.fixture
def toy_batch_graph(toy_graph):
    """BATCH_SIZE copies of toy_graph, batched via PyG Batch."""
    return Batch.from_data_list([toy_graph.clone() for _ in range(BATCH_SIZE)])


@pytest.fixture
def combined_batch_graph():
    """Batched graph with V_SKIN + V_FCR nodes per sample (for compute_all_metrics)."""
    n = V_SKIN + V_FCR
    single = Data(
        x=torch.randn(n, NODE_FEAT_DIM),
        edge_index=_ring_edges(n),
        pos=torch.randn(n, 3),
    )
    return Batch.from_data_list([single.clone() for _ in range(BATCH_SIZE)])


@pytest.fixture
def anthro_features():
    return torch.randn(BATCH_SIZE, ANTHRO_DIM)


@pytest.fixture
def pred_deformations():
    return {
        "skin": torch.randn(BATCH_SIZE, V_SKIN, 3),
        "FCR":  torch.randn(BATCH_SIZE, V_FCR,  3),
    }


@pytest.fixture
def target_deformations():
    return {
        "skin": torch.randn(BATCH_SIZE, V_SKIN, 3),
        "FCR":  torch.randn(BATCH_SIZE, V_FCR,  3),
    }


@pytest.fixture
def sparse_laplacian():
    return make_sparse_laplacian(V_SKIN)


@pytest.fixture
def template_meshes():
    """
    Minimal template_meshes dict consumed by CombinedLoss.
    vertices are torch tensors with a batch dimension.
    edges are [2, E] long tensors.
    """
    n = V_SKIN
    edges_skin = _ring_edges(n)
    row = torch.arange(n - 2)
    faces_skin = torch.stack([row, row + 1, row + 2], dim=1).long()

    edges_fcr = torch.stack(
        [torch.arange(0, V_FCR - 1), torch.arange(1, V_FCR)], dim=0
    ).long()

    return {
        "skin": {
            "vertices":  torch.randn(BATCH_SIZE, n, 3),
            "edges":     edges_skin,
            "faces":     faces_skin,
            "laplacian": make_sparse_laplacian(n),
        },
        "FCR": {
            "vertices": torch.randn(BATCH_SIZE, V_FCR, 3),
            "edges":    edges_fcr,
        },
    }
