from __future__ import annotations
from pathlib import Path
import nibabel as nib
import meshio
import numpy as np

def load_nifti(path: str | Path):
    img = nib.load(str(path))
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header

def save_nifti(path: str | Path, data: np.ndarray, affine: np.ndarray):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))

def load_mesh(path: str | Path):
    return meshio.read(str(path))

def save_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray):
    mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces.astype(np.int32))])
    meshio.write(str(path), mesh)
