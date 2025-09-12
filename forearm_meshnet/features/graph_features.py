# forearm_meshnet/features/graph_features.py
"""
Graph feature extraction module for ForearmMeshNet
"""

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List


class GraphFeatureExtractor:
    """
    Extract graph features from meshes for neural network processing.
    """
    
    def __init__(self):
        """Initialize the GraphFeatureExtractor."""
        pass
    
    def mesh_to_graph(self, 
                     mesh: trimesh.Trimesh,
                     structure_info: Optional[Dict] = None) -> Data:
        """
        Convert mesh to PyTorch Geometric graph.
        
        Args:
            mesh: Input mesh
            structure_info: Optional structure information for multi-structure meshes
            
        Returns:
            PyTorch Geometric Data object
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Calculate node features
        node_features = self._calculate_node_features(mesh, structure_info)
        
        # Create edges from faces
        edges = self._faces_to_edges(faces)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Calculate edge features
        edge_features = self._calculate_edge_features(vertices, edges)
        
        # Create graph data
        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            pos=torch.tensor(vertices, dtype=torch.float),
            faces=torch.tensor(faces, dtype=torch.long)
        )
        
        return graph_data
    
    def _calculate_node_features(self,
                                mesh: trimesh.Trimesh,
                                structure_info: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate node features for each vertex.
        
        Args:
            mesh: Input mesh
            structure_info: Optional structure information
            
        Returns:
            Node feature matrix (N, F)
        """
        vertices = mesh.vertices
        
        # Basic geometric features
        centroid = np.mean(vertices, axis=0)
        distances_to_centroid = np.linalg.norm(vertices - centroid, axis=1)
        
        # Vertex normals
        try:
            if hasattr(mesh, 'vertex_normals'):
                vertex_normals = mesh.vertex_normals
            else:
                mesh.vertex_normals  # Trigger computation
                vertex_normals = mesh.vertex_normals
        except:
            vertex_normals = np.zeros_like(vertices)
        
        # Combine features
        node_features = np.concatenate([
            vertices,                               # x, y, z coordinates
            vertex_normals,                          # nx, ny, nz normals
            distances_to_centroid.reshape(-1, 1),   # distance to centroid
        ], axis=1)
        
        # Add structure-specific features if provided
        if structure_info is not None:
            structure_features = self._calculate_structure_features(
                vertices, structure_info
            )
            node_features = np.concatenate([node_features, structure_features], axis=1)
        
        return node_features
    
    def _calculate_structure_features(self,
                                     vertices: np.ndarray,
                                     structure_info: Dict) -> np.ndarray:
        """
        Calculate structure-specific features for multi-structure meshes.
        
        Args:
            vertices: Vertex positions
            structure_info: Structure information with vertex ranges
            
        Returns:
            Structure feature matrix (N, S+1) where S is number of structures
        """
        num_vertices = len(vertices)
        num_structures = len(structure_info)
        
        # One-hot encoding for structure membership + distance features
        structure_features = np.zeros((num_vertices, num_structures + 1))
        
        for i, (structure_name, info) in enumerate(structure_info.items()):
            start, end = info['vertex_range']
            
            # One-hot encoding
            structure_features[start:end, i] = 1.0
            
            # Distance to structure centroid
            structure_vertices = vertices[start:end]
            if len(structure_vertices) > 0:
                structure_centroid = np.mean(structure_vertices, axis=0)
                distances = np.linalg.norm(vertices - structure_centroid, axis=1)
                normalized_distances = distances / (distances.max() + 1e-8)
                structure_features[:, -1] = np.maximum(
                    structure_features[:, -1],
                    1.0 - normalized_distances
                )
        
        return structure_features
    
    def _faces_to_edges(self, faces: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert faces to unique edges.
        
        Args:
            faces: Face array (F, 3)
            
        Returns:
            List of unique edges (bidirectional)
        """
        edges = []
        
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edges.append([v1, v2])
                edges.append([v2, v1])  # Bidirectional
        
        # Remove duplicates
        edges = list(set(tuple(edge) for edge in edges))
        
        return edges
    
    def _calculate_edge_features(self,
                                vertices: np.ndarray,
                                edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate edge features.
        
        Args:
            vertices: Vertex positions
            edges: List of edges
            
        Returns:
            Edge feature matrix (E, F)
        """
        edge_features = []
        for v1, v2 in edges:
            edge_vector = vertices[v2] - vertices[v1]
            edge_length = float(np.linalg.norm(edge_vector))
            edge_features.append([edge_length])
        return np.array(edge_features, dtype=np.float32)


