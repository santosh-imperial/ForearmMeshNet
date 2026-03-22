"""
test_mesh_operations.py — Tests for kabsch_align, umeyama_align, rigid_icp,
                           and similarity_icp in utils/mesh_operations.py.
"""

import pytest
import numpy as np


RNG = np.random.default_rng(0)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _random_pointcloud(n: int = 30) -> np.ndarray:
    return RNG.standard_normal((n, 3))


# ── kabsch_align (_kabsch_rigid) ──────────────────────────────────────────────

class TestKabschAlign:
    def test_identity_transform(self):
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud()
        R, t = kabsch_align(A, A)
        assert np.allclose(R, np.eye(3), atol=1e-8)
        assert np.allclose(t, np.zeros(3), atol=1e-8)

    def test_pure_translation_recovered(self):
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud()
        shift = np.array([3.0, -1.0, 2.0])
        B = A + shift
        R, t = kabsch_align(A, B)
        assert np.allclose(R, np.eye(3), atol=1e-6)
        assert np.allclose(t, shift, atol=1e-6)

    def test_rotation_is_proper_rotation(self):
        """R must be orthogonal with det = +1 (no reflections)."""
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud()
        R_true = _rotation_x(np.pi / 4)
        B = (R_true @ A.T).T
        R, t = kabsch_align(A, B)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "R must be orthogonal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, "det(R) must be +1"

    def test_reconstruction_accuracy(self):
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud(50)
        R_true = _rotation_x(0.3)
        t_true = np.array([1.0, -2.0, 0.5])
        B = (R_true @ A.T).T + t_true
        R, t = kabsch_align(A, B)
        B_hat = (R @ A.T).T + t
        assert np.allclose(B_hat, B, atol=1e-6)

    def test_output_shapes(self):
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud()
        R, t = kabsch_align(A, A)
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_float32_input_accepted(self):
        from forearm_meshnet.utils.mesh_operations import kabsch_align
        A = _random_pointcloud().astype(np.float32)
        R, t = kabsch_align(A, A)
        assert R.shape == (3, 3)


# ── umeyama_align (_umeyama_similarity) ──────────────────────────────────────

class TestUmeyamaAlign:
    def test_identity_transform(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud()
        R, t, s = umeyama_align(A, A)
        assert np.allclose(R, np.eye(3), atol=1e-6)
        assert np.allclose(t, np.zeros(3), atol=1e-6)
        assert abs(s - 1.0) < 1e-6

    def test_scale_recovered(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud(50)
        scale_true = 2.5
        B = A * scale_true
        R, t, s = umeyama_align(A, B)
        assert abs(s - scale_true) < 0.01

    def test_rotation_is_proper(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud(50)
        R_true = _rotation_x(0.5)
        B = (R_true @ A.T).T * 1.5
        R, t, s = umeyama_align(A, B)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_reconstruction_accuracy(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud(50)
        R_true = _rotation_x(0.3)
        s_true = 1.8
        t_true = np.array([1.0, -1.0, 2.0])
        B = s_true * (R_true @ A.T).T + t_true
        R, t, s = umeyama_align(A, B)
        B_hat = s * (R @ A.T).T + t
        assert np.allclose(B_hat, B, atol=1e-5)

    def test_output_shapes(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud()
        R, t, s = umeyama_align(A, A)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert isinstance(s, float)

    def test_scale_positive(self):
        from forearm_meshnet.utils.mesh_operations import umeyama_align
        A = _random_pointcloud()
        B = A * 3.0
        _, _, s = umeyama_align(A, B)
        assert s > 0.0


# ── rigid_icp ─────────────────────────────────────────────────────────────────

class TestRigidICP:
    def test_identical_clouds_near_identity(self):
        from forearm_meshnet.utils.mesh_operations import rigid_icp
        A = _random_pointcloud(30)
        R, t = rigid_icp(A, A)
        assert np.allclose(R, np.eye(3), atol=1e-5)
        assert np.allclose(t, np.zeros(3), atol=1e-5)

    def test_recovers_pure_translation(self):
        """Use a structured grid so NN matching is unambiguous and ICP converges exactly."""
        from forearm_meshnet.utils.mesh_operations import rigid_icp
        # 4x4x4 grid — each point has a unique nearest neighbour after small shift
        g = np.array([[x, y, z]
                      for x in range(4) for y in range(4) for z in range(4)],
                     dtype=np.float64)
        t_true = np.array([0.05, -0.05, 0.03])
        B = g + t_true
        R, t = rigid_icp(g, B, max_iters=50)
        A_aligned = (R @ g.T).T + t
        mse = np.mean(np.sum((A_aligned - B) ** 2, axis=1))
        assert mse < 1e-8

    def test_output_shapes(self):
        from forearm_meshnet.utils.mesh_operations import rigid_icp
        A = _random_pointcloud()
        R, t = rigid_icp(A, A)
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    def test_rotation_is_proper(self):
        from forearm_meshnet.utils.mesh_operations import rigid_icp
        A = _random_pointcloud(30)
        R_true = _rotation_x(0.2)
        B = (R_true @ A.T).T
        R, t = rigid_icp(A, B)
        assert abs(np.linalg.det(R) - 1.0) < 1e-4

    def test_with_init_transform(self):
        """Providing a good init should still converge."""
        from forearm_meshnet.utils.mesh_operations import rigid_icp
        g = np.array([[x, y, z]
                      for x in range(4) for y in range(4) for z in range(4)],
                     dtype=np.float64)
        t_true = np.array([0.05, 0.0, 0.0])
        B = g + t_true
        R, t = rigid_icp(g, B, init_t=np.array([0.04, 0.0, 0.0]))
        A_aligned = (R @ g.T).T + t
        mse = np.mean(np.sum((A_aligned - B) ** 2, axis=1))
        assert mse < 1e-8


# ── similarity_icp ────────────────────────────────────────────────────────────

def _grid_cloud():
    """3×3×3 grid: structured enough for unambiguous ICP correspondence."""
    return np.array([[x, y, z]
                     for x in range(4) for y in range(4) for z in range(4)],
                    dtype=np.float64)


class TestSimilarityICP:
    def test_identical_clouds(self):
        from forearm_meshnet.utils.mesh_operations import similarity_icp
        A = _grid_cloud()
        R, t, s = similarity_icp(A, A)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert isinstance(s, float)

    def test_recovers_uniform_scale(self):
        """Similarity ICP with a warm-start scale init converges correctly."""
        from forearm_meshnet.utils.mesh_operations import similarity_icp
        A = _grid_cloud()
        A = A - A.mean(0)
        s_true = 1.3
        B = A * s_true
        # Provide init_s near the true scale so NN matching is unambiguous
        R, t, s = similarity_icp(A, B, max_iters=100, init_s=1.2)
        A_aligned = s * (R @ A.T).T + t
        mse = np.mean(np.sum((A_aligned - B) ** 2, axis=1))
        assert mse < 1e-6

    def test_scale_positive(self):
        from forearm_meshnet.utils.mesh_operations import similarity_icp
        A = _grid_cloud()
        R, t, s = similarity_icp(A, A)
        assert s > 0.0

    def test_output_shapes(self):
        from forearm_meshnet.utils.mesh_operations import similarity_icp
        A = _grid_cloud()
        R, t, s = similarity_icp(A, A)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert isinstance(s, float)
