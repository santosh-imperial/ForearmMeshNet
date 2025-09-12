# forearm_meshnet/template/skin_template.py
"""
Skin template generation module for ForearmMeshNet
"""

import numpy as np
import trimesh
import torch
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import pickle
import pymeshfix
try:
    from smplx import SMPLX
    _SMPLX_AVAILABLE = True
except Exception:
    _SMPLX_AVAILABLE = False

class SkinTemplateGenerator:
    """
    Generate skin template mesh for ForearmMeshNet.
    
    This class creates a standardized template mesh that serves
    as the reference for deformation learning.
    """
    
    def __init__(self):
        """Initialize the SkinTemplateGenerator."""
        self.template_mesh = None
        self.template_graph = None
        self.template_features = None
    
    def create_from_mesh(self,
                        mesh: trimesh.Trimesh,
                        target_vertices: Optional[int] = None) -> trimesh.Trimesh:
        """
        Create template from a reference mesh.
        
        Args:
            mesh: Reference mesh
            target_vertices: Target number of vertices (None to keep original)
            
        Returns:
            Template mesh
        """
        print("Creating skin template from reference mesh...")
        
        # Center and normalize mesh
        template = self._normalize_mesh(mesh.copy())
        
        # Simplify if requested
        if target_vertices and len(template.vertices) > target_vertices:
            print(f"  Simplifying from {len(template.vertices)} to {target_vertices} vertices...")
            template = template.simplify_quadric_decimation(target_vertices)

        template= self._clean_mesh_preserve_scale(template)
        
        # Store template
        self.template_mesh = template
        
        # Create graph representation
        self.template_graph = self._create_graph_representation(template)
        
        print(f"  Template created: {len(template.vertices)} vertices, {len(template.faces)} faces")
        
        return template
    def _clean_mesh_preserve_scale(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair(verbose=False, joincomp=True, remove_smallest_components=True)
            mesh = trimesh.Trimesh(mf.v, mf.f, process=False)
        except Exception:
            pass
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        return mesh
    
    def create_from_average(self,
                          mesh_list: List[trimesh.Trimesh],
                          target_vertices: Optional[int] = None) -> trimesh.Trimesh:
        """
        Create template from average of multiple meshes.
        
        Args:
            mesh_list: List of reference meshes
            target_vertices: Target number of vertices
            
        Returns:
            Average template mesh
        """
        print(f"Creating template from {len(mesh_list)} meshes...")
        
        if len(mesh_list) == 0:
            raise ValueError("No meshes provided")
        
        if len(mesh_list) == 1:
            return self.create_from_mesh(mesh_list[0], target_vertices)
        
        # Use first mesh as reference
        reference = mesh_list[0].copy()
        reference = self._normalize_mesh(reference)
        
        # Simplify to target vertices if needed
        if target_vertices and len(reference.vertices) > target_vertices:
            reference = reference.simplify_quadric_decimation(target_vertices)
        
        # Initialize average vertices
        avg_vertices = reference.vertices.copy()
        
        # Align and average other meshes
        from scipy.spatial import cKDTree
        
        for mesh in mesh_list[1:]:
            # Normalize mesh
            normalized = self._normalize_mesh(mesh.copy())
            
            # Find correspondences using nearest neighbors
            tree = cKDTree(normalized.vertices)
            distances, indices = tree.query(reference.vertices)
            
            # Add to average (using closest points)
            avg_vertices += normalized.vertices[indices]
        
        # Compute average
        avg_vertices /= len(mesh_list)
        
        # Create average template
        template = trimesh.Trimesh(
            vertices=avg_vertices,
            faces=reference.faces,
            process=False
        )
        
        # Clean up
        template.remove_duplicate_faces()
        template.remove_unreferenced_vertices()
        template.remove_degenerate_faces()
        
        # Store template
        self.template_mesh = template
        self.template_graph = self._create_graph_representation(template)
        
        print(f"  Average template created: {len(template.vertices)} vertices")
        
        return template
    
    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh to standard position and scale.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Normalized mesh
        """
        # Center at origin
        mesh = mesh.copy()
        mesh.vertices -= mesh.centroid
        
        
        # Align main axis with Z-axis (length direction)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        
        # Transform to principal axes
        mesh.vertices = mesh.vertices @ pca.components_.T
        
        # Ensure Z is the longest axis
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        main_axis = np.argmax(dimensions)
        
        if main_axis != 2:  # If not Z-axis
            # Swap axes
            new_vertices = mesh.vertices.copy()
            new_vertices[:, [2, main_axis]] = new_vertices[:, [main_axis, 2]]
            mesh.vertices = new_vertices
        
        return mesh
    
    def _create_graph_representation(self, mesh: trimesh.Trimesh) -> Data:
        """
        Create graph representation of template.
        
        Args:
            mesh: Template mesh
            
        Returns:
            PyTorch Geometric Data object
        """
        from ..features import GraphFeatureExtractor
        
        extractor = GraphFeatureExtractor()
        graph = extractor.mesh_to_graph(mesh)
        
        return graph
    
    def save(self, path: str):
        """
        Save template to file.
        
        Args:
            path: Output file path
        """
        if self.template_mesh is None:
            raise ValueError("No template to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mesh
        mesh_path = path.with_suffix('.ply')
        self.template_mesh.export(str(mesh_path))
        
        # Save additional data
        data_path = path.with_suffix('.pkl')
        data = {
            'vertices': self.template_mesh.vertices,
            'faces': self.template_mesh.faces,
            'graph': self.template_graph,
            'metadata': {
                'num_vertices': len(self.template_mesh.vertices),
                'num_faces': len(self.template_mesh.faces),
                'bounds': self.template_mesh.bounds.tolist(),
                'is_watertight': self.template_mesh.is_watertight,
            }
        }
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Template saved to {path}")
    
    def load(self, path: str):
        """
        Load template from file.
        
        Args:
            path: Input file path
        """
        path = Path(path)
        
        # Load mesh
        mesh_path = path.with_suffix('.ply')
        if mesh_path.exists():
            self.template_mesh = trimesh.load(str(mesh_path))
        
        # Load additional data
        data_path = path.with_suffix('.pkl')
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            if self.template_mesh is None:
                self.template_mesh = trimesh.Trimesh(
                    vertices=data['vertices'],
                    faces=data['faces'],
                    process=False
                )
            
            self.template_graph = data.get('graph')
        
        print(f"Template loaded from {path}")
    
    def compute_deformation(self,
                          target_mesh: trimesh.Trimesh,
                          method: str = 'nearest_neighbor') -> np.ndarray:
        """
        Compute deformation from template to target mesh.
        
        Args:
            target_mesh: Target mesh
            method: Correspondence method ('nearest_neighbor' or 'coherent_point_drift')
            
        Returns:
            Deformation vectors (N, 3)
        """
        if self.template_mesh is None:
            raise ValueError("No template mesh available")
        
        print(f"Computing deformation using {method}...")
        
        # Normalize target mesh same way as template
        target_normalized = self._normalize_mesh(target_mesh.copy())
        
        if method == 'nearest_neighbor':
            from scipy.spatial import cKDTree
            
            # Build KD-tree for target
            tree = cKDTree(target_normalized.vertices)
            
            # Find nearest neighbors
            distances, indices = tree.query(self.template_mesh.vertices)
            
            # Compute deformation
            deformation = target_normalized.vertices[indices] - self.template_mesh.vertices
            
        elif method == 'coherent_point_drift':
            # Use CPD for non-rigid registration
            try:
                from pycpd import DeformableRegistration
                
                reg = DeformableRegistration(
                    X=target_normalized.vertices,
                    Y=self.template_mesh.vertices
                )
                reg.register()
                
                # Deformation is the difference
                deformation = reg.TY - self.template_mesh.vertices
                
            except ImportError:
                print("  pycpd not available, falling back to nearest neighbor")
                return self.compute_deformation(target_mesh, 'nearest_neighbor')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  Deformation computed: max={np.max(np.abs(deformation)):.2f}mm")
        
        return deformation
    
    def create_from_smplx(
        self,
        model_path: str,
        side: str = 'right',
        target_vertices: Optional[int] = None,
        gender: str = 'neutral',
        device: str = 'cpu',
        # cylindrical crop in SMPL-X units (meters)
        radius: float = 0.06,
        elbow_buffer: float = 0.02,
        wrist_buffer: float = 0.02,
        # real-world scaling
        scale_factor: Optional[float] = None,
        ground_truth_meshes: Optional[List[trimesh.Trimesh]] = None,
    ) -> trimesh.Trimesh:
        """
        Create a skin template from SMPL-X by extracting the forearm mesh,
        optionally scaling to mm using GT meshes, and orienting via PCA.
        """
        if not _SMPLX_AVAILABLE:
            raise ImportError("smplx is not installed. Please add it to your environment.")

        # 1) SMPL-X forward (neutral pose)
        model = SMPLX(
            model_path=model_path,
            gender=gender,
            use_pca=False,
            use_face_contour=False,
            flat_hand_mean=True,
        ).to(device)

        with torch.no_grad():
            out = model(
                betas=torch.zeros(1, 10, device=device),
                global_orient=torch.zeros(1, 3, device=device),
                body_pose=torch.zeros(1, 63, device=device),
            )

        verts = out.vertices[0].detach().cpu().numpy()
        joints = out.joints[0].detach().cpu().numpy()
        faces = model.faces

        # Pick joints for the chosen side 
        if side.lower() == 'left':
            elbow_idx, wrist_idx = 18, 20
        else:  # right
            elbow_idx, wrist_idx = 19, 21

        elbow = joints[elbow_idx]
        wrist = joints[wrist_idx]
        seg_vec = wrist - elbow
        L = float(np.linalg.norm(seg_vec))
        if L <= 1e-8:
            raise ValueError("SMPL-X forearm length is zero or invalid.")
        u = seg_vec / L  # unit axis

        # Cylindrical crop around the elbow→wrist axis with buffers
        keep_ids = []
        for i, v in enumerate(verts):
            rel = v - elbow
            proj = float(np.dot(rel, u))
            proj_pt = elbow + proj * u
            dist = float(np.linalg.norm(v - proj_pt))
            within_rad = dist < radius
            within_span = (elbow_buffer <= proj <= (L - wrist_buffer))
            if within_rad and within_span:
                keep_ids.append(i)

        if len(keep_ids) < 50:
            raise RuntimeError(f"Too few vertices selected for the forearm ({len(keep_ids)}).")

        keep = set(keep_ids)
        face_mask = np.all(np.isin(faces, list(keep)), axis=1)
        sub_faces = faces[face_mask]
        if sub_faces.size == 0:
            raise RuntimeError("Cylindrical crop produced no faces; adjust radius/buffers.")

        # Remap vertex indices
        ordered_ids = sorted(keep)
        old2new = {old: i for i, old in enumerate(ordered_ids)}
        new_verts = verts[ordered_ids]
        new_faces = np.vectorize(old2new.get)(sub_faces).astype(np.int64)

        mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

        #  (Optional) scale to real-world size using GT meshes (avg forearm length)
        if scale_factor is None and ground_truth_meshes:
            gt_lengths = []
            for m in ground_truth_meshes:
                dims = (m.bounds[1] - m.bounds[0])
                gt_lengths.append(float(np.max(dims)))
            if gt_lengths:
                avg_gt_len = float(np.mean(gt_lengths))
                ref_len = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
                if ref_len > 0:
                    scale_factor = avg_gt_len / ref_len
        if scale_factor is not None:
            mesh.vertices *= float(scale_factor)

        # (Optional) decimate to target count
        if target_vertices and len(mesh.vertices) > target_vertices:
            mesh = mesh.simplify_quadric_decimation(int(target_vertices))

        template = self._clean_mesh_preserve_scale(template)

        # Orientation normalization (PCA), **no scaling** 
        mesh = self._normalize_mesh(mesh)


        # Store
        self.template_mesh = mesh
        self.template_graph = self._create_graph_representation(mesh)
        print(f"SMPL-X template created: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh
