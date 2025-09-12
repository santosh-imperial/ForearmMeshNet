# forearm_meshnet/inference/predictor.py
"""
Inference pipeline for ForearmMeshNet
"""

import torch
import numpy as np
import trimesh
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import json

from ..models import ForearmMeshNet
from ..data import DataNormalizer
from ..features import AnthropometricExtractor, GraphFeatureExtractor


class Predictor:
    """
    Inference predictor for ForearmMeshNet.
    
    Handles model loading, feature preparation, and mesh generation
    from anthropometric measurements.
    """
    
    def __init__(self,
                 model_checkpoint_path: str,
                 template_path: str,
                 normalizer_path: str,
                 device: Optional[str] = None):
        """
        Initialize Predictor.
        
        Args:
            model_checkpoint_path: Path to trained model checkpoint
            template_path: Path to unified template
            normalizer_path: Path to data normalizers
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"\nInitializing ForearmMeshNet Predictor")
        print(f"  Device: {self.device}")
        
        # Load model
        self.model, self.config = self._load_model(model_checkpoint_path)
        
        # Load template
        self.template_system = self._load_template_system(template_path)
        self.template_mesh = self.template_system['mesh']
        self.template_graph = self.template_system.get('graph')
        self.structure_info = self.template_system['structure_info']
        
        # Load normalizer
        self.normalizer = self._load_normalizer(normalizer_path)
        
        # Set normalizer in model
        self.model.set_normalizer(self.normalizer)
        
        # Initialize feature extractors
        self.anthro_extractor = AnthropometricExtractor()
        self.graph_extractor = GraphFeatureExtractor()
        
        print(f"\nPredictor ready for inference")
        print(f"  Template vertices: {len(self.template_mesh.vertices)}")
        print(f"  Structures: {len(self.structure_info)}")
    
    def _load_model(self, checkpoint_path: str) -> Tuple[ForearmMeshNet, Dict]:
        """Load trained model from checkpoint."""
        print(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        model = ForearmMeshNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"  Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        best_val = checkpoint.get('best_val_loss', None)
        print(f"  Best val loss: {best_val:.4f}" if isinstance(best_val, (float, int)) else "  Best val loss: N/A") 
        
        return model, config
    
    def _load_template_system(self, template_path: str) -> Dict:
        """Load unified template system."""
        print(f"Loading template from: {template_path}")
        
        path = Path(template_path)
        
        # Try pickle first
        pkl_path = path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                template_data = pickle.load(f)
            
            # Create mesh from data
            mesh = trimesh.Trimesh(
                vertices=template_data['vertices'],
                faces=template_data['faces'],
                process=False
            )
            
            return {
                'mesh': mesh,
                'graph': template_data.get('graph'),
                'structure_info': template_data['structure_info'],
                'vertices': template_data['vertices'],
                'faces': template_data['faces']
            }
        
        # Try mesh file
        mesh_path = path.with_suffix('.ply')
        if not mesh_path.exists():
            mesh_path = path.with_suffix('.obj')
        
        if mesh_path.exists():
            mesh = trimesh.load(str(mesh_path))
            return {
                'mesh': mesh,
                'graph': None,
                'structure_info': {},
                'vertices': mesh.vertices,
                'faces': mesh.faces
            }
        
        raise FileNotFoundError(f"Template not found at {template_path}")
    
    def _load_normalizer(self, normalizer_path: str) -> Dict:
        """Load data normalizer."""
        print(f"Loading normalizer from: {normalizer_path}")
        
        with open(normalizer_path, 'rb') as f:
            norm = pickle.load(f)
        # Normalize to a dict-like interface
        if hasattr(norm, '__dict__'):
            norm = vars(norm)  # object → dict
        return norm
    
    def predict(self,
                anthropometric_measurements: Dict[str, float],
                n_samples: int = 1,
                return_vertices: bool = True) -> Dict[str, Any]:
        """
        Generate mesh predictions from anthropometric measurements.
        
        Args:
            anthropometric_measurements: Dictionary of measurements
            n_samples: Number of samples to generate (for uncertainty)
            return_vertices: Whether to return deformed vertices
            
        Returns:
            Dictionary containing predictions
        """
        print(f"\nGenerating prediction...")
        
        # Prepare features
        features = self._prepare_features(anthropometric_measurements)
        
        # Normalize features
        features_normalized, graph_normalized = self._normalize_inputs(features)
        
        # Generate samples
        with torch.no_grad():
            samples = self.model.sample(
                features_normalized,
                n_samples=n_samples,
                #template_graph=graph_normalized
            )
        
        # Process predictions
        predictions = []
        for sample_idx, sample in enumerate(samples):
            print(f"  Processing sample {sample_idx + 1}/{n_samples}")
            
            # Denormalize deformations
            denorm_deformations = self._denormalize_predictions(sample)
            
            # Apply deformations to template
            if return_vertices:
                deformed_meshes = self._apply_deformations(
                    denorm_deformations,
                    sample.get('affine_params')
                )
            else:
                deformed_meshes = None
            
            predictions.append({
                'deformations': denorm_deformations,
                'meshes': deformed_meshes,
                'affine_params': sample.get('affine_params')
            })
        
        # Prepare output
        result = {
            'predictions': predictions,
            'anthropometric_measurements': anthropometric_measurements,
            'template_info': {
                'num_vertices': len(self.template_mesh.vertices),
                'num_faces': len(self.template_mesh.faces),
                'structures': list(self.structure_info.keys())
            }
        }
        
        print(f"Prediction complete!")
        
        return result
    
    def _prepare_features(self, measurements: Dict[str, float]) -> np.ndarray:
        """Prepare anthropometric features from measurements."""
        # Ensure all required features are present
        required_features = self.anthro_extractor.feature_order
        
        for feature in required_features:
            if feature not in measurements:
                # Use default values
                if 'circumference' in feature:
                    measurements[feature] = 200.0  # mm
                elif 'length' in feature:
                    measurements[feature] = 250.0  # mm
                elif 'ratio' in feature:
                    measurements[feature] = 1.0
                elif 'area' in feature:
                    measurements[feature] = 3000.0  # mm²
                else:
                    measurements[feature] = 0.0
        
        # Add subject data if not present
        subject_data = {
            'height': measurements.get('subject_height', 175.0),
            'weight': measurements.get('subject_weight', 70.0),
            'age': measurements.get('subject_age', 30),
            'gender': measurements.get('subject_gender', None),
            'dominant_hand': measurements.get('dominant_hand', None)
        }
        
        # Add subject data to measurements
        measurements = self.anthro_extractor.add_subject_data(
            measurements, subject_data
        )
        
        # Convert to feature vector
        features = self.anthro_extractor.to_feature_vector(measurements)
        
        return features
    
    def _normalize_inputs(self,
                         features: np.ndarray) -> Tuple[torch.Tensor, Any]:
        """Normalize input features."""
        # Get normalizer
        if hasattr(self.normalizer, 'anthropometric_scaler'):
            # New-style normalizer object
            anthro_scaler = self.normalizer['anthropometric_scaler']
            graph_scaler = self.normalizer.get('graph_feature_scaler')
        else:
            # Direct normalizer
            anthro_scaler = self.normalizer.get('anthropometric_scaler')
            graph_scaler = self.normalizer.get('graph_feature_scaler')
        
        # Normalize anthropometric features
        features_normalized = anthro_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.tensor(
            features_normalized,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        # Normalize graph if available
        graph_normalized = None
        if self.template_graph is not None and graph_scaler is not None:
            graph = self.template_graph.clone()
            node_features = graph.x.detach().cpu().numpy()
            node_features_norm = graph_scaler.transform(node_features)
            graph.x = torch.tensor(node_features_norm, dtype=torch.float32, device=self.device)
            graph_normalized = graph.to(self.device)
        
        return features_tensor, graph_normalized
    
    def _denormalize_predictions(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Denormalize predicted deformations."""
        struct_defs = sample.get('structure_deformations', sample)
        scalers = self.normalizer.get('structure_deformation_scalers', {})
        out = {}
        for name, t in struct_defs.items():
            if name == 'affine_params': 
                continue
            x = t[0] if t.dim() == 3 else t  # strip batch if present
            sc = scalers.get(name)
            if sc is None:
                out[name] = x
                continue
            V = x.shape[0]
            if hasattr(sc, 'mean_') and len(sc.mean_) == V*3:
                flat = x.detach().cpu().numpy().reshape(1, -1)
                inv  = sc.inverse_transform(flat).reshape(V, 3)
                out[name] = torch.tensor(inv, dtype=x.dtype)
            elif hasattr(sc, 'mean_') and len(sc.mean_) == 3:
                mean  = torch.as_tensor(sc.mean_,  dtype=x.dtype)
                scale = torch.as_tensor(sc.scale_, dtype=x.dtype)
                out[name] = x * scale + mean
            else:
                out[name] = x
        return out

    
    def _apply_deformations(self,
                          deformations: Dict[str, torch.Tensor],
                          affine_params: Optional[Dict] = None) -> Dict[str, trimesh.Trimesh]:
        """Apply deformations to template mesh."""
        deformed_meshes = {}
        
        # Apply to unified mesh
        deformed_vertices = self.template_mesh.vertices.copy()
        
        # Apply affine transformation if provided
        if affine_params is not None:
            scale = affine_params['scale'][0].cpu().numpy()
            translation = affine_params['translation'][0].cpu().numpy()
            deformed_vertices = deformed_vertices * scale + translation
        
        # Apply structure deformations
        for struct_name, deformation in deformations.items():
            if struct_name in self.structure_info:
                v_start, v_end = self.structure_info[struct_name]['vertex_range']
                
                deform_np = deformation.cpu().numpy()
                if deform_np.shape[0] == v_end - v_start:
                    deformed_vertices[v_start:v_end] += deform_np
        
        # Create unified deformed mesh
        unified_mesh = trimesh.Trimesh(
            vertices=deformed_vertices,
            faces=self.template_mesh.faces,
            process=False
        )
        deformed_meshes['unified'] = unified_mesh
        
        # Extract individual structure meshes
        for struct_name, info in self.structure_info.items():
            v_start, v_end = info['vertex_range']
            f_start, f_end = info.get('face_range', (0, 0))
            
            if f_end > f_start:
                # Extract vertices and faces
                struct_vertices = deformed_vertices[v_start:v_end]
                struct_faces = self.template_mesh.faces[f_start:f_end] - v_start
                
                struct_mesh = trimesh.Trimesh(
                    vertices=struct_vertices,
                    faces=struct_faces,
                    process=False
                )
                
                deformed_meshes[struct_name] = struct_mesh
        
        return deformed_meshes
    
    def save_prediction(self,
                       prediction: Dict,
                       output_path: str,
                       format: str = 'ply'):
        """
        Save prediction results.
        
        Args:
            prediction: Prediction dictionary from predict()
            output_path: Output directory path
            format: Mesh file format ('ply', 'obj', 'stl')
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save meshes
        for pred_idx, pred in enumerate(prediction['predictions']):
            if pred['meshes'] is None:
                continue
            
            sample_dir = output_dir / f"sample_{pred_idx}"
            sample_dir.mkdir(exist_ok=True)
            
            for mesh_name, mesh in pred['meshes'].items():
                mesh_path = sample_dir / f"{mesh_name}.{format}"
                mesh.export(str(mesh_path))
        
        # Save metadata
        metadata = {
            'anthropometric_measurements': prediction['anthropometric_measurements'],
            'template_info': prediction['template_info'],
            'num_samples': len(prediction['predictions'])
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Prediction saved to: {output_dir}")


class InferencePipeline:
    """
    Complete inference pipeline for batch processing.
    """
    
    def __init__(self,
                 model_checkpoint: str,
                 template_path: str,
                 normalizer_path: str,
                 output_dir: str = "./inference_output"):
        """
        Initialize InferencePipeline.
        
        Args:
            model_checkpoint: Path to model checkpoint
            template_path: Path to template
            normalizer_path: Path to normalizer
            output_dir: Output directory
        """
        self.predictor = Predictor(
            model_checkpoint,
            template_path,
            normalizer_path
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_batch(self,
                     measurements_file: str,
                     n_samples: int = 1) -> Dict[str, Any]:
        """
        Process batch of subjects from CSV file.
        
        Args:
            measurements_file: CSV file with anthropometric measurements
            n_samples: Number of samples per subject
            
        Returns:
            Dictionary with all predictions
        """
        import pandas as pd
        
        # Load measurements
        df = pd.read_csv(measurements_file)
        
        print(f"\nProcessing {len(df)} subjects")
        
        all_predictions = {}
        
        for idx, row in df.iterrows():
            subject_id = row.get('subject_id', f'subject_{idx}')
            print(f"\nProcessing {subject_id}...")
            
            # Convert row to measurements dict
            measurements = row.to_dict()
            
            # Generate prediction
            prediction = self.predictor.predict(
                measurements,
                n_samples=n_samples
            )
            
            # Save prediction
            subject_dir = self.output_dir / subject_id
            self.predictor.save_prediction(
                prediction,
                str(subject_dir)
            )
            
            all_predictions[subject_id] = prediction
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to: {self.output_dir}")
        
        return all_predictions
    
    def interactive_predict(self) -> Dict[str, Any]:
        """
        Interactive prediction with user input.
        
        Returns:
            Prediction results
        """
        print("\n" + "="*60)
        print("INTERACTIVE FOREARM MESH PREDICTION")
        print("="*60)
        
        measurements = {}
        
        # Basic measurements
        print("\nEnter anthropometric measurements (in mm):")
        measurements['forearm_length'] = float(input("  Forearm length [250]: ") or 250)
        measurements['wrist_circumference'] = float(input("  Wrist circumference [170]: ") or 170)
        measurements['mid_forearm_circumference'] = float(input("  Mid-forearm circumference [210]: ") or 210)
        measurements['proximal_circumference'] = float(input("  Proximal circumference [240]: ") or 240)
        
        # Subject data
        print("\nEnter subject information:")
        measurements['subject_height'] = float(input("  Height (cm) [175]: ") or 175)
        measurements['subject_weight'] = float(input("  Weight (kg) [70]: ") or 70)
        measurements['subject_age'] = int(input("  Age [30]: ") or 30)
        
        gender = input("  Gender (M/F) [None]: ").upper()
        measurements['subject_gender'] = gender if gender in ['M', 'F'] else None
        
        hand = input("  Dominant hand (L/R) [None]: ").upper()
        measurements['dominant_hand'] = hand if hand in ['L', 'R'] else None
        
        # Number of samples
        n_samples = int(input("\nNumber of samples to generate [1]: ") or 1)
        
        # Generate prediction
        print("\nGenerating prediction...")
        prediction = self.predictor.predict(measurements, n_samples=n_samples)
        
        # Save results
        output_name = input("\nEnter output name [interactive_result]: ") or "interactive_result"
        output_path = self.output_dir / output_name
        
        self.predictor.save_prediction(prediction, str(output_path))
        
        print(f"\nPrediction complete!")
        print(f"Results saved to: {output_path}")
        
        return prediction