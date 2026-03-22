# forearm_meshnet/template/unified_template.py
"""
Unified multi-structure template generation module for ForearmMeshNet
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class UnifiedTemplateGenerator:
    """
    Generate unified multi-structure template for ForearmMeshNet.
    
    This class creates a single template containing both skin and muscle
    structures.
    """
    
    def __init__(self):
        """Initialize the UnifiedTemplateGenerator."""
        self.skin_template = None
        self.muscle_templates = {}
        self.unified_mesh = None
        self.structure_info = {}
        self.unified_graph = None
    
    def create(self,
              skin_template: trimesh.Trimesh,
              muscle_templates: Dict[str, trimesh.Trimesh],
              skin_vertices: int = 5000,
              muscle_vertices: int = 500) -> trimesh.Trimesh:
        """
        Create unified template from skin and muscle templates.
        
        Args:
            skin_template: Skin template mesh
            muscle_templates: Dictionary of muscle template meshes
            skin_vertices: Target vertices for skin
            muscle_vertices: Target vertices per muscle
            
        Returns:
            Unified template mesh
        """
        
        logger.info("CREATING UNIFIED MULTI-STRUCTURE TEMPLATE")
        
        # reset structure info for a fresh build
        self.structure_info = {}

        # single global translation using skin centroid 
        skin_center = skin_template.vertices.mean(axis=0)
        skin_t = skin_template.copy()
        skin_t.vertices = skin_t.vertices - skin_center

        muscles_t = {}
        for name, m in muscle_templates.items():
            mt = m.copy()
            mt.vertices = mt.vertices - skin_center
            muscles_t[name] = mt

        
        # Process skin template
        logger.info("Processing skin template...")
        self.skin_template = self._process_template(
            skin_t, 
            skin_vertices,
            "skin"
        )
        
        # Process muscle templates
        logger.info(f"Processing {len(muscle_templates)} muscle templates...")
        self.muscle_templates = {}
        
        for muscle_name, muscle_mesh in muscles_t.items():
            logger.info(f"  Processing {muscle_name}...")
            processed = self._process_template(
                muscle_mesh.copy(),
                muscle_vertices,
                muscle_name
            )
            self.muscle_templates[muscle_name] = processed
        
        # Combine into unified mesh
        logger.info("Combining structures into unified template...")
        self.unified_mesh = self._combine_structures()
        
        # Create graph representation
        logger.info("Creating unified graph representation...")
        self.unified_graph = self._create_unified_graph()
        
        logger.info("Unified template created:")
        logger.info(f"  Total vertices: {len(self.unified_mesh.vertices):,}")
        logger.info(f"  Total faces: {len(self.unified_mesh.faces):,}")
        logger.info(f"  Structures: {len(self.structure_info)}")
        
        return self.unified_mesh
    
    def _process_template(self,
                         mesh: trimesh.Trimesh,
                         target_vertices: int,
                         structure_name: str) -> trimesh.Trimesh:
        """
        Process individual template mesh.
        
        Args:
            mesh: Input mesh
            target_vertices: Target number of vertices
            structure_name: Name of the structure
            
        Returns:
            Processed mesh
        """
        
        
        # Simplify if needed
        if len(mesh.vertices) > target_vertices:
            mesh = mesh.simplify_quadric_decimation(target_vertices)
        
        # Clean up
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()
        
        logger.info(f"    {structure_name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _combine_structures(self) -> trimesh.Trimesh:
        """
        Combine all structures into unified mesh.
        
        Returns:
            Unified mesh with all structures
        """
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # Add skin first
        all_vertices.append(self.skin_template.vertices)
        all_faces.append(self.skin_template.faces)
        
        self.structure_info['skin'] = {
            'vertex_range': (0, len(self.skin_template.vertices)),
            'face_range': (0, len(self.skin_template.faces)),
            'num_vertices': len(self.skin_template.vertices),
            'num_faces': len(self.skin_template.faces)
        }
        
        vertex_offset += len(self.skin_template.vertices)
        face_offset = len(self.skin_template.faces)
        
        # Add muscles
        for muscle_name, muscle_mesh in self.muscle_templates.items():
            # Offset faces
            muscle_faces = muscle_mesh.faces + vertex_offset
            
            all_vertices.append(muscle_mesh.vertices)
            all_faces.append(muscle_faces)
            
            self.structure_info[muscle_name] = {
                'vertex_range': (vertex_offset, vertex_offset + len(muscle_mesh.vertices)),
                'face_range': (face_offset, face_offset + len(muscle_mesh.faces)),
                'num_vertices': len(muscle_mesh.vertices),
                'num_faces': len(muscle_mesh.faces)
            }
            
            vertex_offset += len(muscle_mesh.vertices)
            face_offset += len(muscle_mesh.faces)
        
        # Combine arrays
        unified_vertices = np.vstack(all_vertices)
        unified_faces = np.vstack(all_faces)
        
        # Create unified mesh
        unified_mesh = trimesh.Trimesh(
            vertices=unified_vertices,
            faces=unified_faces,
            process=False
        )
        
        return unified_mesh
    
    def _create_unified_graph(self) -> Data:
        """
        Create graph representation of unified template.
        
        Returns:
            PyTorch Geometric Data object
        """
        from ..features import GraphFeatureExtractor
        
        extractor = GraphFeatureExtractor()
        graph = extractor.mesh_to_graph(
            self.unified_mesh,
            self.structure_info
        )
        
        return graph
    
    def get_structure_mesh(self, structure_name: str) -> trimesh.Trimesh:
        """
        Extract individual structure mesh from unified template.
        
        Args:
            structure_name: Name of structure to extract
            
        Returns:
            Structure mesh
        """
        if structure_name not in self.structure_info:
            raise ValueError(f"Structure {structure_name} not found")
        
        info = self.structure_info[structure_name]
        v_start, v_end = info['vertex_range']
        f_start, f_end = info['face_range']
        
        # Extract vertices and faces
        vertices = self.unified_mesh.vertices[v_start:v_end]
        faces = self.unified_mesh.faces[f_start:f_end]
        
        # Adjust face indices
        faces = faces - v_start
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    def save(self, path: str):
        """
        Save unified template to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mesh
        mesh_path = path.with_suffix('.ply')
        self.unified_mesh.export(str(mesh_path))
        
        # Save metadata and structure info
        data_path = path.with_suffix('.pkl')
        data = {
            'vertices': self.unified_mesh.vertices,
            'faces': self.unified_mesh.faces,
            'structure_info': self.structure_info,
            'graph': self.unified_graph,
            'metadata': {
                'total_vertices': len(self.unified_mesh.vertices),
                'total_faces': len(self.unified_mesh.faces),
                'num_structures': len(self.structure_info),
                'structure_names': list(self.structure_info.keys())
            }
        }
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Unified template saved to {path}")
    
    def load(self, path: str):
        """
        Load unified template from file.
        
        Args:
            path: Input file path
        """
        path = Path(path)
        
        # Load mesh
        mesh_path = path.with_suffix('.ply')
        if mesh_path.exists():
            self.unified_mesh = trimesh.load(str(mesh_path))
        
        # Load metadata
        data_path = path.with_suffix('.pkl')
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.structure_info = data['structure_info']
            self.unified_graph = data.get('graph')
            
            if self.unified_mesh is None:
                self.unified_mesh = trimesh.Trimesh(
                    vertices=data['vertices'],
                    faces=data['faces'],
                    process=False
                )
            
            # Reconstruct individual templates
            self.skin_template = self.get_structure_mesh('skin')
            
            self.muscle_templates = {}
            for structure_name in self.structure_info.keys():
                if structure_name != 'skin':
                    self.muscle_templates[structure_name] = self.get_structure_mesh(structure_name)
        
        logger.info(f"Unified template loaded from {path}")
        logger.info(f"  Structures: {list(self.structure_info.keys())}")
    
    def compute_structure_deformations(self,
                                      target_skin: trimesh.Trimesh,
                                      target_muscles: Dict[str, trimesh.Trimesh]) -> Dict[str, np.ndarray]:
        """
        Compute deformations for all structures.
        
        Args:
            target_skin: Target skin mesh
            target_muscles: Dictionary of target muscle meshes
            
        Returns:
            Dictionary of structure deformations
        """
        deformations = {}
        
        # Compute skin deformation
        if self.skin_template is not None and target_skin is not None:
            from .skin_template import SkinTemplateGenerator
            
            skin_gen = SkinTemplateGenerator()
            skin_gen.template_mesh = self.skin_template
            deformations['skin'] = skin_gen.compute_deformation(target_skin)
        
        # Compute muscle deformations
        for muscle_name, muscle_template in self.muscle_templates.items():
            if muscle_name in target_muscles:
                target_muscle = target_muscles[muscle_name]
                
                # Compute deformation for this muscle
                from scipy.spatial import cKDTree
                
                # Build KD-tree for target
                tree = cKDTree(target_muscle.vertices)
                
                # Find nearest neighbors
                distances, indices = tree.query(muscle_template.vertices)
                
                # Compute deformation
                deformation = target_muscle.vertices[indices] - muscle_template.vertices
                deformations[muscle_name] = deformation
        
        # Combine into single deformation vector matching unified template
        combined_deformation = np.zeros((len(self.unified_mesh.vertices), 3))
        
        for structure_name, deformation in deformations.items():
            if structure_name in self.structure_info:
                v_start, v_end = self.structure_info[structure_name]['vertex_range']
                combined_deformation[v_start:v_end] = deformation
        
        deformations['combined'] = combined_deformation
        
        return deformations