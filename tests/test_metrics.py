"""
test_metrics.py — Tests for MeshEvaluationMetrics.

Known bugs covered:
  Bug 1 (metrics.py:18): The outer `MeshEvaluationMetrics` class contains a
      nested inner class with the exact same name. All real methods live in the
      inner class. Instantiating the outer class raises TypeError because it
      has no __init__; accessing methods requires going through
      MeshEvaluationMetrics.MeshEvaluationMetrics.
"""

import pytest
import torch
import numpy as np
from tests.conftest import V_SKIN, V_FCR


def _get_cls():
    from forearm_meshnet.training.metrics import MeshEvaluationMetrics
    return MeshEvaluationMetrics


# ── Instantiation (Bug 1) ─────────────────────────────────────────────────────

class TestMeshEvaluationMetricsInstantiation:
    def test_can_be_instantiated(self, structure_info):
        """Bug 1: outer class must be directly instantiable."""
        cls = _get_cls()
        metrics = cls(torch.device("cpu"), structure_info)
        assert metrics is not None

    def test_instance_has_device(self, structure_info):
        cls = _get_cls()
        metrics = cls(torch.device("cpu"), structure_info)
        assert hasattr(metrics, "device")

    def test_instance_has_structure_info(self, structure_info):
        cls = _get_cls()
        metrics = cls(torch.device("cpu"), structure_info)
        assert hasattr(metrics, "structure_info")

    def test_required_methods_accessible(self, structure_info):
        """Bug 1: methods must be reachable on the instance, not buried in a nested class."""
        cls = _get_cls()
        metrics = cls(torch.device("cpu"), structure_info)
        for method in ("chamfer_distance", "f_score", "mesh_iou", "compute_all_metrics"):
            assert callable(getattr(metrics, method, None)), \
                f"Method '{method}' not accessible (Bug 1: nested class)"


# ── chamfer_distance ──────────────────────────────────────────────────────────

@pytest.fixture
def metrics(structure_info):
    cls = _get_cls()
    return cls(torch.device("cpu"), structure_info)


class TestChamferDistance:
    def test_identical_clouds_zero(self, metrics):
        pts = torch.randn(10, 3)
        cd = metrics.chamfer_distance(pts, pts)
        assert cd.item() < 1e-5

    def test_output_scalar(self, metrics):
        a = torch.randn(10, 3)
        b = torch.randn(12, 3)
        assert metrics.chamfer_distance(a, b).shape == torch.Size([])

    def test_non_negative(self, metrics):
        a = torch.randn(8, 3)
        b = torch.randn(8, 3)
        assert metrics.chamfer_distance(a, b).item() >= 0.0

    def test_batched_input_accepted(self, metrics):
        """2-D inputs should be expanded internally."""
        a = torch.randn(5, 3)
        b = torch.randn(5, 3)
        cd = metrics.chamfer_distance(a, b)
        assert cd.shape == torch.Size([])


# ── f_score ───────────────────────────────────────────────────────────────────

class TestFScore:
    def test_identical_clouds_perfect_score(self, metrics):
        pts = torch.randn(10, 3)
        f, p, r = metrics.f_score(pts, pts, threshold=1.0)
        assert abs(f.item() - 100.0) < 1e-3

    def test_returns_three_values(self, metrics):
        a = torch.randn(8, 3)
        b = torch.randn(8, 3)
        result = metrics.f_score(a, b, threshold=1.0)
        assert len(result) == 3

    def test_values_in_0_100(self, metrics):
        a = torch.randn(8, 3)
        b = torch.randn(8, 3)
        f, p, r = metrics.f_score(a, b, threshold=1.0)
        for val in (f, p, r):
            assert 0.0 <= val.item() <= 100.0 + 1e-4

    def test_larger_threshold_higher_fscore(self, metrics):
        torch.manual_seed(0)
        a = torch.randn(20, 3) * 0.5
        b = torch.randn(20, 3) * 0.5
        f_tight, _, _ = metrics.f_score(a, b, threshold=0.1)
        f_loose, _, _ = metrics.f_score(a, b, threshold=5.0)
        assert f_loose.item() >= f_tight.item()


# ── mesh_iou ──────────────────────────────────────────────────────────────────

class TestMeshIoU:
    def _make_cube_verts_faces(self, scale=1.0):
        """Return a simple tetrahedral mesh for IoU testing."""
        verts = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=torch.float32) * scale
        faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long)
        return verts, faces

    def test_identical_mesh_iou_one(self, metrics):
        verts, faces = self._make_cube_verts_faces()
        iou = metrics.mesh_iou(verts, verts, faces, faces, resolution=16)
        assert iou.item() > 0.99

    def test_output_in_0_1(self, metrics):
        verts, faces = self._make_cube_verts_faces()
        verts2, _ = self._make_cube_verts_faces(scale=1.1)
        iou = metrics.mesh_iou(verts, verts2, faces, faces, resolution=16)
        assert 0.0 <= iou.item() <= 1.0 + 1e-6

    def test_non_overlapping_iou_zero(self, metrics):
        verts1, faces = self._make_cube_verts_faces(scale=1.0)
        verts2 = verts1 + 1000.0   # far away
        iou = metrics.mesh_iou(verts1, verts2, faces, faces, resolution=8)
        assert iou.item() < 1e-3


# ── compute_all_metrics ───────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_dict_with_structure_keys(self, metrics, structure_info,
                                              pred_deformations, target_deformations,
                                              template_meshes, combined_batch_graph):
        result = metrics.compute_all_metrics(
            pred_deformations=pred_deformations,
            target_deformations=target_deformations,
            template_meshes=template_meshes,
            affine_template_graph=combined_batch_graph,
        )
        assert isinstance(result, dict)
        assert "aggregate" in result

    def test_chamfer_present_per_structure(self, metrics, structure_info,
                                           pred_deformations, target_deformations,
                                           template_meshes, combined_batch_graph):
        result = metrics.compute_all_metrics(
            pred_deformations, target_deformations, template_meshes, combined_batch_graph
        )
        for name in pred_deformations:
            if name in result:
                assert "chamfer_distance" in result[name]

    def test_aggregate_contains_mean(self, metrics, pred_deformations,
                                     target_deformations, template_meshes,
                                     combined_batch_graph):
        result = metrics.compute_all_metrics(
            pred_deformations, target_deformations, template_meshes, combined_batch_graph
        )
        agg = result.get("aggregate", {})
        # At least some mean_ keys should be present
        mean_keys = [k for k in agg if k.startswith("mean_")]
        assert len(mean_keys) > 0
