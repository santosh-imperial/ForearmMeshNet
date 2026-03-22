"""
test_graph_features.py — Tests for GraphFeatureExtractor.
"""

import pytest
import numpy as np
import torch
import trimesh


def _make_simple_mesh(n_segments: int = 4):
    """Return a simple trimesh cylinder with faces."""
    return trimesh.creation.cylinder(radius=10.0, height=50.0, sections=n_segments)


# ── Instantiation ──────────────────────────────────────────────────────────────

class TestGraphFeatureExtractorInit:
    def test_instantiates(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        assert gfe is not None


# ── mesh_to_graph ─────────────────────────────────────────────────────────────

class TestMeshToGraph:
    def test_returns_pyg_data(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        from torch_geometric.data import Data
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert isinstance(graph, Data)

    def test_x_has_correct_num_nodes(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert graph.x.shape[0] == len(mesh.vertices)

    def test_x_feature_dim_without_structure_info(self):
        """Without structure_info: xyz + normals + dist_to_centroid = 7 features."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert graph.x.shape[1] == 7

    def test_x_feature_dim_with_structure_info(self):
        """With S structures: 7 + S + 1 features."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        n = len(mesh.vertices)
        structure_info = {
            "part_a": {"vertex_range": (0, n // 2)},
            "part_b": {"vertex_range": (n // 2, n)},
        }
        graph = gfe.mesh_to_graph(mesh, structure_info=structure_info)
        # 7 base + 2 structures + 1 distance col = 10
        assert graph.x.shape[1] == 10

    def test_edge_index_shape(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.dtype == torch.long

    def test_pos_matches_vertices(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert graph.pos.shape == (len(mesh.vertices), 3)

    def test_edge_attr_shape(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        num_edges = graph.edge_index.shape[1]
        assert graph.edge_attr.shape == (num_edges, 1)

    def test_output_is_float32(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        graph = gfe.mesh_to_graph(mesh)
        assert graph.x.dtype == torch.float32


# ── _faces_to_edges ───────────────────────────────────────────────────────────

class TestFacesToEdges:
    def test_returns_list(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        edges = gfe._faces_to_edges(faces)
        assert isinstance(edges, list)

    def test_bidirectional(self):
        """Each edge (u,v) should also have (v,u)."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        faces = np.array([[0, 1, 2]])
        edges = set(gfe._faces_to_edges(faces))
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert (1, 2) in edges
        assert (2, 1) in edges

    def test_no_duplicate_edges(self):
        """Each directed edge appears exactly once."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        faces = np.array([[0, 1, 2], [0, 1, 3]])  # share edge (0,1)
        edges = gfe._faces_to_edges(faces)
        # directed edge (0,1) and (1,0) should each appear exactly once
        assert len(edges) == len(set(edges))

    def test_triangle_has_six_directed_edges(self):
        """Single triangle → 3 undirected = 6 directed edges."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        faces = np.array([[0, 1, 2]])
        edges = gfe._faces_to_edges(faces)
        assert len(edges) == 6


# ── _calculate_edge_features ──────────────────────────────────────────────────

class TestCalculateEdgeFeatures:
    def test_output_shape(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        edges = [(0, 1), (1, 0), (0, 2)]
        feats = gfe._calculate_edge_features(vertices, edges)
        assert feats.shape == (3, 1)

    def test_edge_length_correct(self):
        """Edge from (0,0,0) to (3,4,0) should have length 5."""
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        vertices = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        edges = [(0, 1)]
        feats = gfe._calculate_edge_features(vertices, edges)
        assert abs(feats[0, 0] - 5.0) < 1e-5

    def test_dtype_float32(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        edges = [(0, 1)]
        feats = gfe._calculate_edge_features(vertices, edges)
        assert feats.dtype == np.float32


# ── _calculate_node_features ──────────────────────────────────────────────────

class TestCalculateNodeFeatures:
    def test_feature_dim_no_structure(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        feats = gfe._calculate_node_features(mesh)
        assert feats.shape == (len(mesh.vertices), 7)

    def test_feature_dim_with_structure(self):
        from forearm_meshnet.features.graph_features import GraphFeatureExtractor
        gfe = GraphFeatureExtractor()
        mesh = _make_simple_mesh()
        n = len(mesh.vertices)
        si = {"A": {"vertex_range": (0, n)}}
        feats = gfe._calculate_node_features(mesh, structure_info=si)
        # 7 + 1 structure + 1 distance = 9
        assert feats.shape[1] == 9


# ── Public import via utils __init__ ──────────────────────────────────────────

class TestUtilsInit:
    def test_kabsch_align_importable(self):
        from forearm_meshnet.utils import kabsch_align
        assert callable(kabsch_align)

    def test_umeyama_align_importable(self):
        from forearm_meshnet.utils import umeyama_align
        assert callable(umeyama_align)

    def test_rigid_icp_importable(self):
        from forearm_meshnet.utils import rigid_icp
        assert callable(rigid_icp)

    def test_similarity_icp_importable(self):
        from forearm_meshnet.utils import similarity_icp
        assert callable(similarity_icp)
