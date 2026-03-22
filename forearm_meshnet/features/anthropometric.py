# forearm_meshnet/features/anthropometric.py
"""
Anthropometric feature extraction module for ForearmMeshNet
"""

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


class AnthropometricExtractor:
    """
    Extract anthropometric measurements from forearm meshes.
    
    This class computes measurements that can be easily obtained
    in practice (e.g., with tape measure) and used for inference.
    """
    
    def __init__(self, debug_plots: bool = False):
        """
        Initialize the AnthropometricExtractor.
        
        Args:
            debug_plots: Whether to generate debug visualizations
        """
        self.debug_plots = debug_plots
        
        # Define standard feature order for consistency
        self.feature_order = [
            'forearm_length',
            'wrist_circumference',
            'mid_forearm_circumference',
            'proximal_circumference',
            'taper_ratio',
            'length_width_ratio',
            'width_depth_ratio',
            'wrist_cross_sectional_area',
            'mid_cross_sectional_area',
            'proximal_cross_sectional_area',
            'max_dimension',
            'min_dimension',
            'bounding_box_volume','surface_area', 'volume'
        ]
        
        # Subject-specific features (added during data preparation)
        self.subject_features = [
            'subject_height',
            'subject_weight',
            'subject_age',
            'bmi'
        ]
        
        # Categorical features
        self.categorical_features = [
            'subject_gender',  # M/F/Unknown
            'dominant_hand'     # L/R/Unknown
        ]
    
    def extract_from_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Extract anthropometric measurements from a mesh.
        
        Args:
            mesh: Input forearm mesh
            
        Returns:
            Dictionary of anthropometric measurements
        """
        logger.info("Extracting anthropometric measurements...")
        
        vertices = mesh.vertices
        
        # 1. Basic dimensions
        bbox = mesh.bounds
        dimensions = bbox[1] - bbox[0]
        
        # Find main axis (forearm length direction)
        main_axis = np.argmax(dimensions)
        forearm_length = dimensions[main_axis]
        
        # 2. Robust circumference measurements
        circumferences = self._calculate_robust_circumferences(
            vertices, main_axis, bbox
        )
        
        # 3. Derived measurements
        cross_sectional_areas = [c**2 / (4 * np.pi) for c in circumferences]
        taper_ratio = circumferences[0] / circumferences[2] if circumferences[2] > 0 else 1.0
        
        # 4. Shape descriptors
        sorted_dims = sorted(dimensions)
        length_width_ratio = sorted_dims[2] / sorted_dims[1] if sorted_dims[1] > 0 else 1.0
        width_depth_ratio = sorted_dims[1] / sorted_dims[0] if sorted_dims[0] > 0 else 1.0
        
        # 5. Compile measurements
        anthropometric_data = {
            # Primary measurements
            'forearm_length': forearm_length,
            'wrist_circumference': circumferences[0],
            'mid_forearm_circumference': circumferences[1],
            'proximal_circumference': circumferences[2],
            
            # Derived shape descriptors
            'taper_ratio': taper_ratio,
            'length_width_ratio': length_width_ratio,
            'width_depth_ratio': width_depth_ratio,
            
            # Cross-sectional estimates
            'wrist_cross_sectional_area': cross_sectional_areas[0],
            'mid_cross_sectional_area': cross_sectional_areas[1],
            'proximal_cross_sectional_area': cross_sectional_areas[2],
            
            # Additional measurements
            'max_dimension': np.max(dimensions),
            'min_dimension': np.min(dimensions),
            'bounding_box_volume': np.prod(dimensions),
        }
        
        # Add mesh-based measurements if available
        if mesh.is_watertight:
            anthropometric_data['surface_area'] = mesh.area
            anthropometric_data['volume'] = mesh.volume
        
        logger.info(f"  Forearm length: {forearm_length:.1f} mm")
        logger.info(f"  Circumferences: {[f'{c:.1f}' for c in circumferences]} mm")
        logger.info(f"  Taper ratio: {taper_ratio:.3f}")
        
        return anthropometric_data
    
    def _calculate_robust_circumferences(self,
                                        vertices: np.ndarray,
                                        main_axis: int,
                                        bbox: np.ndarray) -> List[float]:
        """
        Calculate circumferences at multiple positions along forearm.
        
        Args:
            vertices: Mesh vertices
            main_axis: Index of main axis (length direction)
            bbox: Bounding box of mesh
            
        Returns:
            List of circumferences [wrist, mid, proximal]
        """
        min_coord = bbox[0][main_axis]
        max_coord = bbox[1][main_axis]
        total_length = (max_coord - min_coord)
        
        # Define measurement positions (25%, 50%, 75% along forearm)
        positions = [0.25, 0.5, 0.75]
        circumferences = []
        
        for p in positions:
            coord = min_coord + p * total_length
            c = self._calculate_circumference_at_cross_section(vertices, main_axis, coord, total_length)
            circumferences.append(c)
        return circumferences
    
    def _calculate_circumference_at_cross_section(self, vertices, main_axis, coord, total_length):
        # 3% of length (adaptive window)
        tolerance = 0.03 * total_length
        mask = np.abs(vertices[:, main_axis] - coord) < tolerance
        if np.sum(mask) < 4:
            return 0.0 

        section_vertices = vertices[mask]
        cross_section_2d = self._project_to_cross_section_plane(section_vertices, main_axis)
        return self._fit_ellipse_and_calculate_circumference(cross_section_2d)

    def _project_to_cross_section_plane(self, section_vertices, main_axis):
        # just drop the main axis and center 
        other_axes = [i for i in range(3) if i != main_axis]
        points_2d = section_vertices[:, other_axes]
        points_2d -= points_2d.mean(axis=0)
        return points_2d
    
    def _calculate_hull_perimeter(self, hull_points: np.ndarray) -> float:
        """
        Calculate perimeter of convex hull.
        
        Args:
            hull_points: Ordered hull vertices
            
        Returns:
            Perimeter length
        """
        perimeter = 0
        n_points = len(hull_points)
        
        for i in range(n_points):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n_points]
            perimeter += np.linalg.norm(p2 - p1)
        
        return perimeter
    
    
    def _fit_ellipse_and_calculate_circumference(self, points_2d):
        if len(points_2d) < 5:
            # fallback: circle
            r = np.linalg.norm(points_2d, axis=1).mean()
            return 2 * np.pi * r
        try:
            params = self._fit_ellipse_least_squares(points_2d)  # returns dict with semi_major, semi_minor
            a, b = params['semi_major'], params['semi_minor']
            # Ramanujan approximation
            h = ((a - b) / (a + b))**2
            return np.pi * (a + b) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h)))
        except Exception:
            # fallback: convex hull perimeter, then circle
            return self._calculate_convex_hull_perimeter(points_2d)

    def _fit_ellipse_least_squares(self, points):
        x, y = points[:, 0], points[:, 1]
        D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
        C = np.array([[0,0,2,0,0,0],
                    [0,-1,0,0,0,0],
                    [2,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0]])
        S = D.T @ D
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S) @ C)

        # Pick an eigenvector with ellipse constraint B^2 - 4AC < 0 and best (smallest |eigval|)
        candidates = []
        for i in range(len(eigvals)):
            A,B,Cc,Dc,E,F = eigvecs[:, i]
            if (B**2 - 4*A*Cc) < 0:
                candidates.append((abs(eigvals[i]), eigvecs[:, i]))
        if not candidates:
            raise ValueError("No valid ellipse solution")
        _, p = min(candidates, key=lambda t: t[0])
        A,B,Cc,Dc,E,F = p
        denom = (B**2 - 4*A*Cc)
        if denom >= 0:
            raise ValueError("Not an ellipse")
        xc = (2*Cc*Dc - B*E) / denom
        yc = (2*A*E - B*Dc) / denom
        term1 = 2 * (A*xc**2 + Cc*yc**2 + B*xc*yc - F)
        term2 = np.sqrt((A - Cc)**2 + B**2)
        a_sq = term1 / (A + Cc + term2)
        b_sq = term1 / (A + Cc - term2)
        a = np.sqrt(abs(a_sq))
        b = np.sqrt(abs(b_sq))
        if a < b: a, b = b, a
        return {'center': (xc, yc), 'semi_major': a, 'semi_minor': b}

    def _calculate_convex_hull_perimeter(self, points_2d):
        try:
            hull = ConvexHull(points_2d)
            hp = points_2d[hull.vertices]
            return float(np.sum(np.linalg.norm(hp[(np.arange(len(hp))+1)%len(hp)] - hp, axis=1)))
        except Exception:
            r = np.linalg.norm(points_2d, axis=1).mean()
            return 2 * np.pi * r
        
    def add_subject_data(self,
                        anthropometric_data: Dict[str, float],
                        subject_data: Dict[str, any]) -> Dict[str, float]:
        """
        Add subject-specific data to anthropometric measurements.
        
        Args:
            anthropometric_data: Mesh-based measurements
            subject_data: Subject information (height, weight, etc.)
            
        Returns:
            Updated anthropometric data
        """
        # Add basic subject data
        anthropometric_data['subject_height'] = subject_data.get('height', None)
        anthropometric_data['subject_weight'] = subject_data.get('weight', None)
        anthropometric_data['subject_age'] = subject_data.get('age', None)
        anthropometric_data['subject_gender'] = subject_data.get('gender', None)
        anthropometric_data['dominant_hand'] = subject_data.get('dominant_hand', None)
        
        # Calculate derived metrics
        height = subject_data.get('height')
        weight = subject_data.get('weight')
        
        if height and weight:
            # BMI calculation
            height_m = height / 100  # Convert cm to m
            bmi = weight / (height_m ** 2)
            anthropometric_data['bmi'] = bmi
            
            # Relative forearm length
            forearm_length = anthropometric_data.get('forearm_length', 0)
            if forearm_length > 0 and height > 0:
                # Forearm length as percentage of height
                relative_length = (forearm_length / 10) / height  # Convert mm to cm
                anthropometric_data['forearm_length_relative'] = relative_length
        
        return anthropometric_data
    
    def to_feature_vector(self,
                         anthropometric_data: Dict[str, any],
                         include_categorical: bool = True) -> torch.Tensor: 
        """
        Convert anthropometric dictionary to feature vector.
        
        Args:
            anthropometric_data: Dictionary of measurements
            include_categorical: Whether to include one-hot encoded categorical features
            
        Returns:
            Feature vector
        """
        features = []
        
        # Add numerical features in standard order
        for key in self.feature_order:
            value = anthropometric_data.get(key, 0.0)
            features.append(float(value) if value is not None else 0.0)
        
        # Add subject-specific numerical features
        for key in self.subject_features:
            value = anthropometric_data.get(key, 0.0)
            features.append(float(value) if value is not None else 0.0)
        
        # Add categorical features (one-hot encoded)
        if include_categorical:
            # Gender encoding: [Is_Male, Is_Female, Gender_Unknown]
            gender = anthropometric_data.get('subject_gender')
            if gender == 'M':
                features.extend([1.0, 0.0, 0.0])
            elif gender == 'F':
                features.extend([0.0, 1.0, 0.0])
            else:
                features.extend([0.0, 0.0, 1.0])
            
            # Dominant hand encoding: [Is_Left, Is_Right, Hand_Unknown]
            hand = anthropometric_data.get('dominant_hand')
            if hand == 'L':
                features.extend([1.0, 0.0, 0.0])
            elif hand == 'R':
                features.extend([0.0, 1.0, 0.0])
            else:
                features.extend([0.0, 0.0, 1.0])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def from_feature_vector(self,
                           feature_vector: Union[np.ndarray, torch.Tensor, Sequence[float]],
                           include_categorical: bool = True) -> Dict[str, any]:
        """
        Convert feature vector back to anthropometric dictionary.
        
        Args:
            feature_vector: Input feature vector
            include_categorical: Whether categorical features are included
            
        Returns:
            Dictionary of anthropometric measurements
        """
        # Make an indexable 1-D numpy array for convenience
        if isinstance(feature_vector, torch.Tensor):
            feature_vector = feature_vector.detach().cpu().numpy().ravel()
        else:
            feature_vector = np.asarray(feature_vector).ravel()

        anthropometric_data = {}
        idx = 0
        
        # Extract numerical features
        for key in self.feature_order:
            if idx < len(feature_vector):
                anthropometric_data[key] = float(feature_vector[idx])
                idx += 1
        
        # Extract subject features
        for key in self.subject_features:
            if idx < len(feature_vector):
                anthropometric_data[key] = float(feature_vector[idx])
                idx += 1
        
        # Extract categorical features
        if include_categorical and idx < len(feature_vector):
            # Gender (3 values: one-hot)
            gender_encoding = feature_vector[idx:idx+3]
            if gender_encoding[0] > 0.5:
                anthropometric_data['subject_gender'] = 'M'
            elif gender_encoding[1] > 0.5:
                anthropometric_data['subject_gender'] = 'F'
            else:
                anthropometric_data['subject_gender'] = None
            idx += 3
            
            # Dominant hand (3 values: one-hot)
            if idx + 3 <= len(feature_vector):
                hand_encoding = feature_vector[idx:idx+3]
                if hand_encoding[0] > 0.5:
                    anthropometric_data['dominant_hand'] = 'L'
                elif hand_encoding[1] > 0.5:
                    anthropometric_data['dominant_hand'] = 'R'
                else:
                    anthropometric_data['dominant_hand'] = None
        
        return anthropometric_data
    
    def get_feature_names(self, include_categorical: bool = True) -> List[str]:
        """
        Get names of all features in order.
        
        Args:
            include_categorical: Whether to include categorical feature names
            
        Returns:
            List of feature names
        """
        names = self.feature_order + self.subject_features
        
        if include_categorical:
            names.extend([
                'is_male', 'is_female', 'gender_unknown',
                'is_left_handed', 'is_right_handed', 'hand_unknown'
            ])
        
        return names
    
    def get_feature_dim(self, include_categorical: bool = True) -> int:
        """
        Get dimension of feature vector.
        
        Args:
            include_categorical: Whether categorical features are included
            
        Returns:
            Feature vector dimension
        """
        dim = len(self.feature_order) + len(self.subject_features)
        
        if include_categorical:
            dim += 6  # 3 for gender + 3 for dominant hand
        
        return dim