from __future__ import annotations
from pathlib import Path
import numpy as np


def load_nifti(path: str | Path):
    import nibabel as nib
    img = nib.load(str(path))
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header


def save_nifti(path: str | Path, data: np.ndarray, affine: np.ndarray):
    import nibabel as nib
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def load_mesh(path: str | Path):
    import meshio
    return meshio.read(str(path))


def save_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray):
    import meshio
    mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces.astype(np.int32))])
    meshio.write(str(path), mesh)
