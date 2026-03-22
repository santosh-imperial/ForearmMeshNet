# forearm_meshnet/preprocessing/mesh_utils.py
"""
Utility functions for mesh operations
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

def boundary_vertex_indices(mesh: trimesh.Trimesh) -> np.ndarray:
    edges = mesh.edges_sorted.reshape(-1, 2)
    edges_view = edges.view([('a', edges.dtype), ('b', edges.dtype)])
    uniq, counts = np.unique(edges_view, return_counts=True)
    boundary_edges = uniq[counts == 1]
    return np.unique(np.hstack([boundary_edges['a'], boundary_edges['b']]))

def cap_then_polish(mesh_in: trimesh.Trimesh, target_faces: int = 50_000, smooth_iter: int = 10) -> trimesh.Trimesh:
    mesh = mesh_in.copy()
    mesh.fill_holes()

    # Open3D decimation
    m_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    if len(mesh.faces) > target_faces:
        m_o3d = m_o3d.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))

    # Save original boundary vertex positions before smoothing
    tm_before = trimesh.Trimesh(np.asarray(m_o3d.vertices), np.asarray(m_o3d.triangles), process=False)
    keep_idx = boundary_vertex_indices(tm_before)
    keep_xyz = tm_before.vertices[keep_idx].copy()

    #Taubin smoothing 
    m_o3d.filter_smooth_taubin(number_of_iterations=int(smooth_iter))

    # Snap boundary vertices back to their pre-smooth positions
    verts = np.asarray(m_o3d.vertices)
    tree = cKDTree(verts)
    _, nn = tree.query(keep_xyz)
    verts[nn] = keep_xyz
    m_o3d.vertices = o3d.utility.Vector3dVector(verts)

    mesh_pol = trimesh.Trimesh(np.asarray(m_o3d.vertices), np.asarray(m_o3d.triangles), process=False)
    return mesh_pol


def center_mesh(mesh: trimesh.Trimesh, inplace: bool = False) -> trimesh.Trimesh:
    """
    Center mesh at origin.
    
    Args:
        mesh: Input mesh
        inplace: Whether to modify mesh in place
        
    Returns:
        Centered mesh
    """
    if not inplace:
        mesh = mesh.copy()
    
    centroid = mesh.centroid
    mesh.vertices -= centroid
    
    return mesh


def scale_mesh(mesh: trimesh.Trimesh, 
               target_size: Optional[float] = None,
               scale_factor: Optional[float] = None,
               inplace: bool = False) -> trimesh.Trimesh:
    """
    Scale mesh to target size or by scale factor.
    
    Args:
        mesh: Input mesh
        target_size: Target bounding box diagonal
        scale_factor: Direct scale factor
        inplace: Whether to modify mesh in place
        
    Returns:
        Scaled mesh
    """
    if not inplace:
        mesh = mesh.copy()
    
    if target_size is not None:
        current_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        scale_factor = target_size / current_size
    
    if scale_factor is not None:
        mesh.vertices *= scale_factor
    
    return mesh


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Validate mesh properties.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_empty': len(mesh.vertices) == 0,
        'is_valid': mesh.is_valid,
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'is_manifold': mesh.is_winding_consistent,
        'has_unreferenced_vertices': len(mesh.vertices) != len(np.unique(mesh.faces)),
        'has_duplicate_faces': len(mesh.faces) != len(np.unique(mesh.faces, axis=0)),
        'has_degenerate_faces': np.any(mesh.area_faces == 0),
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'edge_count': len(mesh.edges_unique),
    }
    
    # Check for non-manifold edges
    edge_face_count = {}
    for face_idx, face in enumerate(mesh.faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge not in edge_face_count:
                edge_face_count[edge] = 0
            edge_face_count[edge] += 1
    
    non_manifold_edges = sum(1 for count in edge_face_count.values() if count > 2)
    validation['non_manifold_edges'] = non_manifold_edges
    
    return validation


def remove_artifacts(mesh: trimesh.Trimesh,
                    max_edge_length: float = 15.0,
                    max_face_area: float = 50.0,
                    min_component_size: int = 100) -> trimesh.Trimesh:
    """
    Remove various artifacts from mesh.
    
    Args:
        mesh: Input mesh
        max_edge_length: Maximum allowed edge length
        max_face_area: Maximum allowed face area
        min_component_size: Minimum vertices in connected component
        
    Returns:
        Cleaned mesh
    """
    mesh = mesh.copy()
    
    # Remove faces with long edges
    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    
    face_mask = np.ones(len(mesh.faces), dtype=bool)
    for face_idx, face in enumerate(mesh.faces):
        face_edges = [
            tuple(sorted([face[i], face[(i+1)%3]])) for i in range(3)
        ]
        for edge in face_edges:
            edge_idx = np.where(
                (edges == edge).all(axis=1) | 
                (edges == edge[::-1]).all(axis=1)
            )[0]
            if len(edge_idx) > 0 and edge_lengths[edge_idx[0]] > max_edge_length:
                face_mask[face_idx] = False
                break
    
    # Remove faces with large area
    face_areas = mesh.area_faces
    face_mask &= face_areas < max_face_area
    
    # Update mesh
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    
    # Remove small components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        large_components = [
            comp for comp in components 
            if len(comp.vertices) >= min_component_size
        ]
        if large_components:
            mesh = trimesh.util.concatenate(large_components)
    
    # Final cleanup
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh,
                iterations: int = 50,
                lambda_factor: float = 0.5,
                mu_factor: float = -0.53) -> trimesh.Trimesh:
    """
    Apply Taubin smoothing to mesh.
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        lambda_factor: Shrinking factor
        mu_factor: Expansion factor
        
    Returns:
        Smoothed mesh
    """
    mesh = mesh.copy()
    vertices = mesh.vertices.copy()
    
    # Build adjacency
    edges = mesh.edges
    adjacency = {i: set() for i in range(len(vertices))}
    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    
    # Apply Taubin smoothing
    for _ in range(iterations):
        # Forward pass (shrinking)
        vertices_new = vertices.copy()
        for i, neighbors in adjacency.items():
            if neighbors:
                neighbor_mean = vertices[list(neighbors)].mean(axis=0)
                vertices_new[i] += lambda_factor * (neighbor_mean - vertices[i])
        vertices = vertices_new
        
        # Backward pass (expansion)
        vertices_new = vertices.copy()
        for i, neighbors in adjacency.items():
            if neighbors:
                neighbor_mean = vertices[list(neighbors)].mean(axis=0)
                vertices_new[i] += mu_factor * (neighbor_mean - vertices[i])
        vertices = vertices_new
    
    mesh.vertices = vertices
    
    return mesh

def remove_spurious_triangles(mesh: trimesh.Trimesh, max_edge_length_mm: float = 20.0, max_face_area_mm2: float = 100.0) -> trimesh.Trimesh:
    logger.info("Removing spurious triangles...")
    faces = mesh.faces
    verts = mesh.vertices

    # Edge lengths per face (max of three edges)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e01 = np.linalg.norm(v0 - v1, axis=1)
    e12 = np.linalg.norm(v1 - v2, axis=1)
    e20 = np.linalg.norm(v2 - v0, axis=1)
    face_edge_lengths = np.max(np.stack([e01, e12, e20], axis=1), axis=1)

    # Face areas
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)

    good_faces_mask = (face_edge_lengths <= max_edge_length_mm) & (face_areas <= max_face_area_mm2)
    num_bad_faces = (~good_faces_mask).sum()
    logger.info(f"   Removing {num_bad_faces} faces out of {len(faces)}")

    mesh = mesh.copy()
    mesh.update_faces(good_faces_mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def detect_isolated_components(mesh: trimesh.Trimesh, min_volume_ratio: float = 0.01) -> trimesh.Trimesh:
    logger.info("Detecting isolated components...")
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        logger.info("   Single component, no removal needed")
        return mesh

    sizes = []
    for comp in components:
        if comp.is_watertight:
            sizes.append(comp.volume)
        else:
            sizes.append(len(comp.vertices))

    max_size = max(sizes)
    kept = []
    for comp, size in zip(components, sizes):
        if size >= min_volume_ratio * max_size:
            kept.append(comp)

    if len(kept) == 0:
        logger.warning("   All components removed, returning original")
        return mesh
    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)