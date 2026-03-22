"""
test_io_utils.py — Tests for load_mesh / save_mesh (and optional nifti functions).
"""

import pytest
import numpy as np
import tempfile
import os


# ── load_mesh / save_mesh ─────────────────────────────────────────────────────

class TestLoadSaveMesh:
    def _make_tetrahedron(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ], dtype=np.int32)
        return vertices, faces

    def test_save_creates_file(self, tmp_path):
        from forearm_meshnet.utils.io_utils import save_mesh
        vertices, faces = self._make_tetrahedron()
        path = str(tmp_path / "mesh.ply")
        save_mesh(path, vertices, faces)
        assert os.path.exists(path)

    def test_roundtrip_vertices(self, tmp_path):
        from forearm_meshnet.utils.io_utils import save_mesh, load_mesh
        vertices, faces = self._make_tetrahedron()
        path = str(tmp_path / "mesh.ply")
        save_mesh(path, vertices, faces)
        mesh = load_mesh(path)
        np.testing.assert_allclose(mesh.points, vertices, atol=1e-5)

    def test_roundtrip_faces(self, tmp_path):
        from forearm_meshnet.utils.io_utils import save_mesh, load_mesh
        vertices, faces = self._make_tetrahedron()
        path = str(tmp_path / "mesh.ply")
        save_mesh(path, vertices, faces)
        mesh = load_mesh(path)
        # meshio stores cells as list of CellBlock; get the triangle block
        tri_block = next(c for c in mesh.cells if c.type == "triangle")
        np.testing.assert_array_equal(tri_block.data, faces)

    def test_load_returns_meshio_mesh(self, tmp_path):
        import meshio
        from forearm_meshnet.utils.io_utils import save_mesh, load_mesh
        vertices, faces = self._make_tetrahedron()
        path = str(tmp_path / "mesh.ply")
        save_mesh(path, vertices, faces)
        mesh = load_mesh(path)
        assert isinstance(mesh, meshio.Mesh)

    def test_save_obj_format(self, tmp_path):
        from forearm_meshnet.utils.io_utils import save_mesh, load_mesh
        vertices, faces = self._make_tetrahedron()
        path = str(tmp_path / "mesh.obj")
        save_mesh(path, vertices, faces)
        assert os.path.exists(path)

    def test_save_accepts_path_object(self, tmp_path):
        from pathlib import Path
        from forearm_meshnet.utils.io_utils import save_mesh
        vertices, faces = self._make_tetrahedron()
        path = tmp_path / "mesh.ply"
        save_mesh(path, vertices, faces)
        assert path.exists()

    def test_load_accepts_path_object(self, tmp_path):
        from pathlib import Path
        from forearm_meshnet.utils.io_utils import save_mesh, load_mesh
        vertices, faces = self._make_tetrahedron()
        path = tmp_path / "mesh.ply"
        save_mesh(str(path), vertices, faces)
        mesh = load_mesh(path)
        assert mesh is not None


# ── load_nifti / save_nifti (skipped if nibabel not available) ────────────────

try:
    import nibabel  # noqa: F401
    _has_nibabel = True
except ImportError:
    _has_nibabel = False


@pytest.mark.skipif(not _has_nibabel, reason="nibabel not installed")
class TestLoadSaveNifti:
    def test_save_creates_file(self, tmp_path):
        from forearm_meshnet.utils.io_utils import save_nifti
        data = np.zeros((4, 4, 4), dtype=np.float32)
        affine = np.eye(4)
        path = str(tmp_path / "vol.nii.gz")
        save_nifti(path, data, affine)
        assert os.path.exists(path)

    def test_roundtrip_data(self, tmp_path):
        from forearm_meshnet.utils.io_utils import load_nifti, save_nifti
        data = np.random.randn(4, 4, 4).astype(np.float32)
        affine = np.eye(4)
        path = str(tmp_path / "vol.nii.gz")
        save_nifti(path, data, affine)
        loaded_data, loaded_affine, _ = load_nifti(path)
        np.testing.assert_allclose(loaded_data, data, atol=1e-5)

    def test_roundtrip_affine(self, tmp_path):
        from forearm_meshnet.utils.io_utils import load_nifti, save_nifti
        data = np.zeros((4, 4, 4), dtype=np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        path = str(tmp_path / "vol.nii.gz")
        save_nifti(path, data, affine)
        _, loaded_affine, _ = load_nifti(path)
        np.testing.assert_allclose(loaded_affine, affine, atol=1e-5)

    def test_load_returns_header(self, tmp_path):
        from forearm_meshnet.utils.io_utils import load_nifti, save_nifti
        data = np.zeros((4, 4, 4), dtype=np.float32)
        path = str(tmp_path / "vol.nii.gz")
        save_nifti(path, data, np.eye(4))
        _, _, header = load_nifti(path)
        assert header is not None
