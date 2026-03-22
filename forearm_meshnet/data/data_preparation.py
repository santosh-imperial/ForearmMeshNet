# forearm_meshnet/data/data_preparation.py
"""
Training data preparation module for ForearmMeshNet
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from torch_geometric.data import Data

from ..utils.mesh_operations import rigid_icp, similarity_icp

logger = logging.getLogger(__name__)



class TrainingDataPreparation:
    """
    Prepare training data for ForearmMeshNet from processed meshes and templates.
    
    This class handles:
    - Loading ground truth meshes
    - Computing deformations from template to GT
    - Extracting anthropometric features
    - Creating graph representations
    - Organizing data for training
    """
    
    def __init__(self,
                unified_template_path: str,
                skin_gt_folder: str,
                muscle_gt_data_path: Optional[str] = None,
                subject_measurements_path: Optional[str] = None):
        """
        Initialize the TrainingDataPreparation.
        
        Args:
            unified_template_path: Path to unified template
            skin_gt_folder: Folder containing skin ground truth meshes
            muscle_gt_data_path: Path to muscle ground truth data (optional)
            subject_measurements_path: Path to CSV with subject measurements (optional)
        """
        self.unified_template_path = Path(unified_template_path)
        self.skin_gt_folder = Path(skin_gt_folder)
        self.muscle_gt_data_path = Path(muscle_gt_data_path) if muscle_gt_data_path else None
        self.subject_measurements_path = Path(subject_measurements_path) if subject_measurements_path else None
        
        # Load unified template
        logger.info("Loading unified template system...")
        self._load_unified_template()
        
        # Load muscle GT data if available
        if self.muscle_gt_data_path and self.muscle_gt_data_path.exists():
            logger.info("Loading muscle ground truth data...")
            with open(self.muscle_gt_data_path, 'rb') as f:
                self.muscle_gt_data = pickle.load(f)
        else:
            self.muscle_gt_data = None
        
        # Load subject measurements if available
        if self.subject_measurements_path and self.subject_measurements_path.exists():
            logger.info("Loading subject measurements...")
            self.subject_measurements = pd.read_csv(self.subject_measurements_path, index_col='subject_id')
        else:
            self.subject_measurements = None
        
        # Initialize feature extractors
        from ..features import AnthropometricExtractor, GraphFeatureExtractor
        self.anthro_extractor = AnthropometricExtractor()
        self.graph_extractor = GraphFeatureExtractor()
        
        logger.info("Training data preparation initialized")
        logger.info(f"  Template vertices: {len(self.unified_mesh.vertices)}")
        logger.info(f"  Structures: {len(self.structure_info)}")
    
    def _load_unified_template(self):
        """Load unified template system."""
        # Try loading from pickle first
        pkl_path = self.unified_template_path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                template_data = pickle.load(f)
            
            self.unified_mesh = trimesh.Trimesh(
                vertices=template_data['vertices'],
                faces=template_data['faces'],
                process=False
            )
            self.template_faces = np.asarray(self.unified_mesh.faces, dtype=np.int64)
            try:
                self.template_edges = np.asarray(self.unified_mesh.edges_unique, dtype=np.int64)
            except Exception:
                self.template_edges = None
            self.structure_info = template_data['structure_info']
            self.template_graph = template_data.get('graph')
            
            # Extract individual structure meshes
            self._extract_structure_meshes()
        else:
            # Load from mesh file
            mesh_path = self.unified_template_path.with_suffix('.ply')
            if not mesh_path.exists():
                mesh_path = self.unified_template_path.with_suffix('.obj')
            
            if mesh_path.exists():
                self.unified_mesh = trimesh.load(str(mesh_path))
                # Structure info would need to be loaded separately
                self.structure_info = {}
                self.template_graph = None
            else:
                raise FileNotFoundError(f"Template not found at {self.unified_template_path}")
            
    def _normalize_like_template(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        m = mesh.copy()
        m.vertices = m.vertices - m.vertices.mean(axis=0)  # center
        
        p = PCA(n_components=3).fit(m.vertices)
        m.vertices = m.vertices @ p.components_.T
        # ensure Z is the longest axis
        dims = (m.bounds[1] - m.bounds[0])
        main = int(np.argmax(dims))
        if main != 2:
            v = m.vertices.copy()
            v[:, [2, main]] = v[:, [main, 2]]
            m.vertices = v
        return m
    
    def _extract_structure_meshes(self):
        """Extract individual structure meshes from unified template."""
        self.structure_meshes = {}
        
        for structure_name, info in self.structure_info.items():
            v_start, v_end = info['vertex_range']
            f_start, f_end = info.get('face_range', (0, 0))
            
            if f_end > f_start:
                # Extract vertices and faces
                vertices = self.unified_mesh.vertices[v_start:v_end]
                faces = (self.unified_mesh.faces[f_start:f_end] - v_start) if f_end > f_start else np.zeros((0,3), dtype=np.int64)
                
                self.structure_meshes[structure_name] = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=False
                )
    
    def prepare_training_dataset(self,
                                output_folder: str,
                                max_subjects: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Prepare complete training dataset.
        
        Args:
            output_folder: Output folder for saving data
            max_subjects: Maximum number of subjects to process (None for all)
            
        Returns:
            training_samples: List of training samples
            statistics: Dataset statistics
        """
        logger.info("PREPARING TRAINING DATASET")
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all skin GT meshes
        skin_mesh_files = sorted(self.skin_gt_folder.glob("*.ply"))
        if not skin_mesh_files:
            skin_mesh_files = sorted(self.skin_gt_folder.glob("*.obj"))
        
        logger.info(f"Found {len(skin_mesh_files)} skin ground truth meshes")
        
        if max_subjects:
            skin_mesh_files = skin_mesh_files[:max_subjects]
        
        # Process each subject
        training_samples = []
        statistics = defaultdict(list)
        
        for mesh_file in skin_mesh_files:
            # Extract subject ID from filename
            subject_id = self._extract_subject_id(mesh_file.name)
            
            logger.info(f"Processing {subject_id}...")
            
            try:
                # Process single subject
                sample = self._process_single_subject(subject_id, mesh_file)
                
                if sample is not None:
                    training_samples.append(sample)
                    
                    # Collect statistics
                    statistics['subject_ids'].append(subject_id)
                    statistics['num_vertices'].append(
                        len(sample['structure_deformations'].get('combined', []))
                    )
                    statistics['num_structures'].append(
                        len(sample['structure_deformations']) - 1  # Exclude 'combined'
                    )
                    
            except Exception as e:
                logger.warning(f"  Error processing {subject_id}: {e}")
                statistics['failed_subjects'].append(subject_id)
        
        # Calculate summary statistics
        summary_stats = self._calculate_statistics(training_samples, statistics)
        
        # Save dataset
        self._save_dataset(training_samples, summary_stats, output_path)
        
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info(f"Total samples: {len(training_samples)}")
        logger.info(f"Failed subjects: {len(statistics.get('failed_subjects', []))}")
        
        return training_samples, summary_stats
    
    def _extract_subject_id(self, filename: str) -> str:
        """Extract subject ID from filename."""
        # Try common patterns
        import re
        
        # Pattern 1: subject_XX or Subject_XX
        match = re.search(r'[Ss]ubject[_-]?(\d+)', filename)
        if match:
            return f"subject_{match.group(1)}"
        
        # Pattern 2: Just numbers
        match = re.search(r'(\d+)', filename)
        if match:
            return f"subject_{match.group(1)}"
        
        # Default: use filename without extension
        return Path(filename).stem
    
    def _process_single_subject(self,
                               subject_id: str,
                               skin_mesh_file: Path) -> Optional[Dict]:
        """
        Process a single subject to create training sample.
        
        Args:
            subject_id: Subject identifier
            skin_mesh_file: Path to skin mesh file
            
        Returns:
            Training sample dictionary or None if failed
        """
        # Load skin GT mesh
        skin_gt_mesh = trimesh.load(str(skin_mesh_file))
        logger.info(f"  Skin mesh: {len(skin_gt_mesh.vertices)} vertices")
        
        # Extract anthropometric features
        anthro_data = self.anthro_extractor.extract_from_mesh(skin_gt_mesh)
        
        # Add subject-specific measurements if available
        if self.subject_measurements is not None and subject_id in self.subject_measurements.index:
            subject_data = self.subject_measurements.loc[subject_id].to_dict()
            anthro_data = self.anthro_extractor.add_subject_data(anthro_data, subject_data)
        else:
            # Use default values
            anthro_data = self.anthro_extractor.add_subject_data(anthro_data, {
                'height': 175.0,  # Default height in cm
                'weight': 70.0,   # Default weight in kg
                'age': 30,        # Default age
                'gender': None,   # Unknown
                'dominant_hand': None  # Unknown
            })
        
        # Convert to feature vector
        anthro_features = self.anthro_extractor.to_feature_vector(anthro_data)
        
        # Compute deformations
        structure_deformations = {}
        
        # Skin deformation
        skin_deformation = self._compute_deformation(
            self.structure_meshes.get('skin', self.unified_mesh),
            skin_gt_mesh, 'skin'
        )
        structure_deformations['skin'] = skin_deformation
        
        # Muscle deformations (if available)
        if self.muscle_gt_data and 'subjects_data' in self.muscle_gt_data:
            subject_muscles = self.muscle_gt_data['subjects_data'].get(subject_id, {})
            muscle_meshes = subject_muscles.get('muscle_meshes', {})
            
            for muscle_name, muscle_data in muscle_meshes.items():
                if muscle_name in self.structure_meshes:
                    muscle_gt_mesh = muscle_data.get('mesh')
                    if muscle_gt_mesh is not None:
                        muscle_deformation = self._compute_deformation(
                            self.structure_meshes[muscle_name],
                            muscle_gt_mesh,
                            muscle_name
                        )
                        structure_deformations[muscle_name] = muscle_deformation
        
        # Combine deformations into unified vector
        combined_deformation = self._combine_deformations(structure_deformations)
        structure_deformations['combined'] = combined_deformation
        
        # Create or load graph representation
        if self.template_graph is not None:
            unified_graph = self.template_graph
        else:
            unified_graph = self.graph_extractor.mesh_to_graph(
                self.unified_mesh,
                self.structure_info
            )
        
        # Create training sample
        sample = {
            'subject_id': subject_id,
            'anthropometric_features': anthro_features,
            'anthropometric_data': anthro_data,
            'unified_template_graph': unified_graph,
            'structure_deformations': structure_deformations,
            'structure_info': self.structure_info,
            'template_mesh_connectivity': {
                'faces': self.template_faces,
                'edges': self.template_edges  # can be None
            },
            'metadata': {
                'skin_vertices': len(skin_gt_mesh.vertices),
                'skin_faces': len(skin_gt_mesh.faces),
                'num_muscles': len(structure_deformations) - 2,  # Exclude skin and combined
            }
        }
        
        logger.info(f"  Sample created with {len(structure_deformations)} structures")
        
        return sample
    
    def _compute_deformation(self,
                           template_mesh: trimesh.Trimesh,
                           target_mesh: trimesh.Trimesh,
                           structure_name: str) -> np.ndarray:
        """
        Compute residual deformation in the TEMPLATE frame.
        Steps:
        1) Align target -> template frame (skin: similarity ICP; muscles: rigid ICP).
        2) Establish correspondences (NN in template frame).
        3) Residual = target_in_template[idx] - template_vertices
        Returns:
        (N,3) float32 residuals, N = template_mesh.vertices.shape[0]
        """
        dst = np.asarray(template_mesh.vertices, dtype=np.float64)  # template (target in ICP)
        src = np.asarray(target_mesh.vertices, dtype=np.float64)    # ground truth

        if structure_name.lower() == 'skin':
            # Similarity ICP (scale allowed)
            R, t, s = similarity_icp(src, dst, max_iters=50, tol=1e-5)
            tgt_T = (s * (src @ R.T)) + t
        else:
            # Rigid ICP (no scale)
            R, t = rigid_icp(src, dst, max_iters=20, tol=1e-4)
            tgt_T = (src @ R.T) + t

        # NN correspondences in template frame
        nn = cKDTree(tgt_T)
        _, idx = nn.query(dst, k=1)

        # Residual in template frame
        residual = (tgt_T[idx] - dst).astype(np.float32)
        return residual
        
    def _combine_deformations(self,
                            structure_deformations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine structure deformations into unified vector.
        
        Args:
            structure_deformations: Dictionary of deformations by structure
            
        Returns:
            Combined deformation vector
        """
        combined = np.zeros((len(self.unified_mesh.vertices), 3), dtype=np.float32)
        for structure_name, deformation in structure_deformations.items():
            if structure_name in self.structure_info:
                v_start, v_end = self.structure_info[structure_name]['vertex_range']
                expected = v_end - v_start
                d = deformation.astype(np.float32, copy=False)
                if d.shape[0] == expected:
                    combined[v_start:v_end] = d
                else:
                    # pad/truncate to expected size
                    if d.shape[0] < expected:
                        pad = np.zeros((expected - d.shape[0], 3), dtype=np.float32)
                        d_fix = np.vstack([d, pad])
                    else:
                        d_fix = d[:expected]
                    combined[v_start:v_end] = d_fix
                    logger.warning(f"    Deformation size mismatch for {structure_name} "
                        f"({d.shape[0]} vs {expected}); padded/truncated.")
        return combined
        
    def _calculate_statistics(self,
                            training_samples: List[Dict],
                            statistics: Dict) -> Dict:
        """
        Calculate dataset statistics.
        
        Args:
            training_samples: List of training samples
            statistics: Raw statistics dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'num_samples': len(training_samples),
            'num_structures': len(self.structure_info),
            'feature_dim': len(training_samples[0]['anthropometric_features']) if training_samples else 0,
            'subjects': statistics['subject_ids'],
            'failed_subjects': statistics.get('failed_subjects', []),
        }
        
        # Calculate deformation statistics
        if training_samples:
            all_deformations = []
            for sample in training_samples:
                combined = sample['structure_deformations'].get('combined')
                if combined is not None:
                    all_deformations.append(combined.flatten())
            
            if all_deformations:
                all_deformations = np.concatenate(all_deformations)
                summary['deformation_stats'] = {
                    'mean': float(np.mean(all_deformations)),
                    'std': float(np.std(all_deformations)),
                    'min': float(np.min(all_deformations)),
                    'max': float(np.max(all_deformations)),
                }
        
        # Structure-specific statistics
        structure_stats = {}
        for structure_name in self.structure_info.keys():
            struct_deforms = []
            for sample in training_samples:
                if structure_name in sample['structure_deformations']:
                    deform = sample['structure_deformations'][structure_name]
                    struct_deforms.append(np.linalg.norm(deform, axis=-1).mean())
            
            if struct_deforms:
                structure_stats[structure_name] = {
                    'mean_magnitude': float(np.mean(struct_deforms)),
                    'std_magnitude': float(np.std(struct_deforms)),
                }
        
        summary['structure_statistics'] = structure_stats
        
        return summary
    
    def _save_dataset(self,
                     training_samples: List[Dict],
                     statistics: Dict,
                     output_path: Path):
        """
        Save training dataset and statistics.
        
        Args:
            training_samples: List of training samples
            statistics: Dataset statistics
            output_path: Output directory
        """
        # Save full dataset
        dataset_path = output_path / 'training_dataset.pkl'
        with open(dataset_path, 'wb') as f:
            pickle.dump(training_samples, f)
        logger.info(f"Dataset saved to: {dataset_path}")
        
        # Save statistics
        stats_path = output_path / 'dataset_statistics.pkl'
        with open(stats_path, 'wb') as f:
            pickle.dump(statistics, f)
        logger.info(f"Statistics saved to: {stats_path}")
        
        # Save human-readable summary
        summary_path = output_path / 'dataset_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("ForearmMeshNet Training Dataset Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Number of samples: {statistics['num_samples']}\n")
            f.write(f"Number of structures: {statistics['num_structures']}\n")
            f.write(f"Feature dimension: {statistics['feature_dim']}\n")
            f.write(f"\nSubjects: {', '.join(statistics['subjects'][:10])}")
            if len(statistics['subjects']) > 10:
                f.write(f"... and {len(statistics['subjects'])-10} more")
            f.write("\n")
            
            if 'deformation_stats' in statistics:
                f.write(f"\nDeformation Statistics:\n")
                for key, value in statistics['deformation_stats'].items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            if 'structure_statistics' in statistics:
                f.write(f"\nStructure-specific Statistics:\n")
                for structure, stats in statistics['structure_statistics'].items():
                    f.write(f"  {structure}:\n")
                    for key, value in stats.items():
                        f.write(f"    {key}: {value:.4f}\n")
        
        logger.info(f"Summary saved to: {summary_path}")
    
    def create_train_val_split(self,
                             training_samples: List[Dict],
                             val_ratio: float = 0.2,
                             random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train/validation split.
        
        Args:
            training_samples: List of all training samples
            val_ratio: Fraction of data for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            train_samples, val_samples
        """
        np.random.seed(random_seed)
        
        n_samples = len(training_samples)
        n_val = int(n_samples * val_ratio)
        
        # Random permutation
        indices = np.random.permutation(n_samples)
        val_indices = set(indices[:n_val])
        
        train_samples = []
        val_samples = []
        
        for i, sample in enumerate(training_samples):
            if i in val_indices:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
        
        logger.info("Train/Val Split:")
        logger.info(f"  Training: {len(train_samples)} samples")
        logger.info(f"  Validation: {len(val_samples)} samples")
        
        return train_samples, val_samples
