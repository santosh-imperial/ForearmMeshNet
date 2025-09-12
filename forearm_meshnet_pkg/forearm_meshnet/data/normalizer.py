"""
Data normalization module for ForearmMeshNet
"""

import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path


def _to_numpy(x) -> np.ndarray:
    """Safely convert torch/numpy to numpy on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class DataNormalizer:
    """
    Normalize training data for ForearmMeshNet.
    
    Handles normalization of:
    - Anthropometric features
    - Structure-specific deformations
    - Graph node features
    """
    
    def __init__(self, normalization_method: str = 'standard', graph_fit_mode: str = "samplemean"):
        """
        Initialize the DataNormalizer.
        
        Args:
            normalization_method: 'standard' (z-score) or 'minmax' (0-1 scaling)
            graph_fit_mode: 'samplemean' or 'nodewise'
        """
        self.normalization_method = normalization_method
        self.anthropometric_scaler: Optional[Any] = None
        self.deformation_scaler: Optional[Any] = None   # Overall (combined) deformation scaler
        self.graph_feature_scaler: Optional[Any] = None
        self.structure_deformation_scalers: Dict[str, Any] = {}  # Structure-specific scalers
        self.fitted = False
        self.max_deformation_size = 0
        self.structure_expected_sizes: Dict[str, int] = {}
        self.normalize_combined = False 

        
        # Statistics for monitoring
        self.statistics = {
            'anthro_mean': None,
            'anthro_std': None,
            'deform_stats': {},
            'graph_mean': None,
            'graph_std': None
        }
        self._graph_feat_dim_fallback = 7  # vertices(3) + normals(3) + dist(1)
    
    def fit_and_transform(self,
                          training_samples: List[Dict],
                          save_path: Optional[str] = None) -> List[Dict]:
        """
        Fit normalizers on training data and transform all samples.
        
        Args:
            training_samples: List of training samples
            save_path: Optional path to save normalizers
            
        Returns:
            List of normalized training samples
        """
        print("\n" + "="*60)
        print("DATA NORMALIZATION")
        print("="*60)
        
        # Step 1: Collect all data for fitting
        anthro_data, deformation_data, graph_data = self._collect_training_data(training_samples)
        
        # Step 2: Fit normalizers
        self._fit_normalizers(anthro_data, deformation_data, graph_data)
        
        # Step 3: Fit structure-specific normalizers
        self._fit_structure_specific_normalizers(training_samples)
        
        # Step 4: Transform all samples
        normalized_samples = self._transform_samples(training_samples)
        
        # Step 5: Validate normalization
        self._validate_normalization(normalized_samples)
        
        # Step 6: Save normalizers if requested
        if save_path:
            self.save(save_path)
        
        print(f"\n NORMALIZATION COMPLETE")
        print(f"  Normalized {len(normalized_samples)} samples")
        print(f"  Method: {self.normalization_method}")
        
        return normalized_samples
    
    def _collect_training_data(self, training_samples):
        print("\n1. Collecting data for normalization...")

        # 1) Anthropometrics
        anthro_data = []
        for s in training_samples:
            a = s['anthropometric_features']
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            anthro_data.append(a)
        anthro_data = np.asarray(anthro_data)

        # 2) Deformations 
        combined_list = []
        for s in training_samples:
            comb = s['structure_deformations'].get('combined', None)
            if comb is None:
                continue
            if isinstance(comb, torch.Tensor):
                comb = comb.detach().cpu().numpy()
            combined_list.append(comb.reshape(-1))  # flatten

        if not combined_list:
            deformation_data = np.empty((0, 0))
            self.max_deformation_size = 0
        else:
            lengths = [v.size for v in combined_list]
            # assert all equal (pipeline guarantees this)
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent combined deformation length across samples: {set(lengths)}")
            self.max_deformation_size = lengths[0]
            deformation_data = np.stack(combined_list, axis=0)  # (N, D)

        # 3) Graph features
        #    Use the first available graph
        graph_rows = []
        for s in training_samples:
            g = s.get('unified_template_graph', None)
            if g is not None and getattr(g, 'x', None) is not None:
                X = _to_numpy(g.x)
                if self.graph_fit_mode == "samplemean":
                    graph_rows.append(X.mean(axis=0, keepdims=True))
                else:  # "nodewise"
                    graph_rows.append(X)  # beware memory!
        if not graph_rows:
            graph_data = np.empty((0, 0))
        else:
            graph_data = np.vstack(graph_rows)

        print(f"  Anthropometric data: {anthro_data.shape}")
        print(f"  Deformation data (combined): {deformation_data.shape}")
        print(f"  Graph node feature rows: {graph_data.shape}")

        return anthro_data, deformation_data, graph_data
    
    def _fit_normalizers(self,
                        anthro_data: np.ndarray,
                        deformation_data: np.ndarray,
                        graph_data: np.ndarray):
        """
        Fit normalizers on collected data.
        """
        print("\n2. Fitting normalizers...")
        # choose scalers
        Scaler = StandardScaler if self.normalization_method == 'standard' else MinMaxScaler
        self.anthropometric_scaler = Scaler()
        self.graph_feature_scaler = Scaler()
        self.deformation_scaler = Scaler() if deformation_data.size else None

        # fit
        self.anthropometric_scaler.fit(anthro_data)

        if deformation_data.size:
            self.deformation_scaler.fit(deformation_data)

        if graph_data.size:
            self.graph_feature_scaler.fit(graph_data)
        else:
            self.graph_feature_scaler = None  # no graph normalization

        # stats (guard empties)
        self.statistics['anthro_mean'] = np.mean(anthro_data, axis=0)
        self.statistics['anthro_std']  = np.std(anthro_data, axis=0)
        if deformation_data.size:
            print(f"  Deformation: mean={deformation_data.mean():.3f}, std={deformation_data.std():.3f}")
        else:
            print("  Deformation: (none)")
        if graph_data.size:
            self.statistics['graph_mean'] = graph_data.mean(axis=0)
            self.statistics['graph_std']  = graph_data.std(axis=0)
            print(f"  Graph features: mean={graph_data.mean():.3f}, std={graph_data.std():.3f}")
        else:
            print("  Graph features: (none)")

        self.fitted = True
    
    def _fit_structure_specific_normalizers(self, training_samples: List[Dict]):
        """
        Fit separate normalizers for each structure.
        """
        print("\n3. Fitting structure-specific normalizers...")
        
        # Collect deformations by structure (excluding 'combined')
        structure_deformations: Dict[str, List[np.ndarray]] = {}
        
        for sample in training_samples:
            for structure_name, deformation in sample.get('structure_deformations', {}).items():
                if structure_name == 'combined':
                    continue
                structure_deformations.setdefault(structure_name, []).append(_to_numpy(deformation).flatten())
        
        # Fit scaler for each structure with data
        for structure_name, deform_list in structure_deformations.items():
            if not deform_list:
                continue
            deform_array = np.vstack([d.reshape(1, -1) for d in deform_list])
            self.structure_expected_sizes[structure_name] = deform_array.shape[1]
            scaler = StandardScaler() if self.normalization_method == 'standard' else MinMaxScaler()
            scaler.fit(deform_array)
            self.structure_deformation_scalers[structure_name] = scaler
            
            # Store statistics
            self.statistics['deform_stats'][structure_name] = {
                'mean': float(np.mean(deform_array)),
                'std': float(np.std(deform_array)),
                'shape': deform_array.shape
            }
            
            print(f"  {structure_name}: shape={deform_array.shape}, "
                  f"mean={np.mean(deform_array):.3f}, std={np.std(deform_array):.3f}")
    
    def _transform_samples(self, training_samples: List[Dict]) -> List[Dict]:
        """
        Transform all samples using fitted normalizers.
        """
        print("\n4. Transforming samples...")
        normalized = []

        for i, s in enumerate(training_samples):
            out = s.copy()

            # anthropometrics
            a = s['anthropometric_features']
            a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
            a_n = self.anthropometric_scaler.transform(a.reshape(1, -1)).reshape(-1)
            out['anthropometric_features'] = torch.tensor(a_n, dtype=torch.float32)

            # deformations
            nd = {}
            for name, d in s['structure_deformations'].items():
                arr = d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d)
                if name == 'combined':
                    if self.normalize_combined and (self.deformation_scaler is not None):
                        flat = arr.reshape(1, -1)
                        # enforce same length (pipeline guarantee)
                        if flat.shape[1] != self.max_deformation_size:
                            raise ValueError(f"Combined deformation length mismatch ({flat.shape[1]} vs {self.max_deformation_size})")
                        dn = self.deformation_scaler.transform(flat).reshape(arr.shape)
                    else:
                        dn = arr
                    nd[name] = torch.tensor(dn, dtype=torch.float32)
                else:
                    # per-structure scaler (if available)
                    if name in self.structure_deformation_scalers:
                        scaler = self.structure_deformation_scalers[name]
                        exp = self.structure_expected_sizes.get(name)
                        flat = arr.reshape(-1)
                        if exp is not None:
                            if flat.size < exp:
                                flat = np.pad(flat, (0, exp - flat.size), mode='constant')
                            elif flat.size > exp:
                                flat = flat[:exp]
                        dn = scaler.transform(flat.reshape(1, -1)).reshape(-1)[:arr.size]  # trim back
                        nd[name] = torch.tensor(dn.reshape(arr.shape), dtype=torch.float32)
                    else:
                        nd[name] = torch.tensor(arr, dtype=torch.float32)
            out['structure_deformations'] = nd

            # graph
            g = s.get('unified_template_graph', None)
            if g is not None and getattr(g, 'x', None) is not None and self.graph_feature_scaler is not None:
                X = g.x.detach().cpu().numpy() if isinstance(g.x, torch.Tensor) else g.x
                Xn = self._safe_graph_transform(_to_numpy(g.x))
                g2 = g.clone()
                g2.x = torch.tensor(Xn, dtype=torch.float32)
                out['unified_template_graph'] = g2

            normalized.append(out)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(training_samples)} samples")

        return normalized
    
    def _validate_normalization(self, normalized_samples: List[Dict]):
        """
        Validate that normalization was successful.
        """
        print("\n5. Validating normalization...")
        
        # Check anthropometric features
        anthro_values = np.vstack([_to_numpy(s['anthropometric_features']).reshape(1, -1)
                                   for s in normalized_samples])
        a_mean = float(np.mean(anthro_values)) if anthro_values.size else float('nan')
        a_std  = float(np.std(anthro_values)) if anthro_values.size else float('nan')
        print(f"  Normalized anthropometric: mean={a_mean:.3f}, std={a_std:.3f}")
        
        if self.normalization_method == 'standard' and anthro_values.size:
            ok = (abs(a_mean) < 0.1 and abs(a_std - 1.0) < 0.1)
            print(" Anthropometric normalization successful" if ok
                  else " Anthropometric normalization may have issues")
        
        # Check structure deformations (only where per-structure scalers exist)
        for structure_name, scaler in self.structure_deformation_scalers.items():
            struct_values = []
            for sample in normalized_samples:
                d = sample.get('structure_deformations', {}).get(structure_name)
                if d is not None:
                    struct_values.append(_to_numpy(d).flatten())
            if struct_values:
                arr = np.concatenate(struct_values)
                print(f"  {structure_name}: mean={np.mean(arr):.3f}, std={np.std(arr):.3f}")
    
    def normalize_for_inference(self,
                               anthropometric_features: np.ndarray,
                               template_graph: Optional[Any] = None) -> Tuple[torch.Tensor, Any]:
        """
        Normalize data for inference.
        
        Args:
            anthropometric_features: Input features
            template_graph: Template graph (optional)
            
        Returns:
            Normalized features and graph
        """
        if not self.fitted:
            raise ValueError("Normalizers not fitted yet!")
        
        # Normalize anthropometric features
        anthro_normalized = self.anthropometric_scaler.transform(
            _to_numpy(anthropometric_features).reshape(1, -1)
        )
        anthro_tensor = torch.tensor(anthro_normalized.flatten(), dtype=torch.float32)
        
        # Normalize graph if provided
        normalized_graph = None
        if (template_graph is not None and getattr(template_graph, 'x', None) is not None
                and self.graph_feature_scaler is not None):
            X = _to_numpy(template_graph.x)
            Xn = self._safe_graph_transform(X)
            normalized_graph = template_graph.clone()
            normalized_graph.x = torch.tensor(Xn, dtype=torch.float32)
        else:
            normalized_graph = template_graph  # pass-through
        return anthro_tensor, normalized_graph
    
    def denormalize_predictions(self,
                              normalized_deformations: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Denormalize predicted deformations.
        
        Args:
            normalized_deformations: Dictionary of normalized deformations by structure
            
        Returns:
            Dictionary of denormalized deformations
        """
        if not self.fitted:
            raise ValueError("Normalizers not fitted yet!")
        out = {}
        for name, d in normalized_deformations.items():
            arr = _to_numpy(d)
            if name == 'combined' and self.deformation_scaler is not None:
                exp = self.structure_expected_sizes.get(name)
                flat = arr.reshape(-1)
                if exp is not None:
                    if flat.size < exp:
                        flat_pad = np.pad(flat, (0, exp - flat.size), mode='constant')
                    else:
                        flat_pad = flat[:exp]
                inv = scaler.inverse_transform(flat_pad.reshape(1, -1)).reshape(-1)[:flat.size]
                out[name] = inv.reshape(arr.shape)
                continue
            scaler = self.structure_deformation_scalers.get(name)
            if scaler is not None:
                flat = arr.reshape(1, -1)
                inv  = scaler.inverse_transform(flat).reshape(arr.shape)
                out[name] = inv
            else:
                out[name] = arr
        return out
    
    def _safe_graph_transform(self, X: np.ndarray) -> np.ndarray:
        if self.graph_feature_scaler is None:
            return X
        if self.normalization_method == 'standard':
            mean  = self.graph_feature_scaler.mean_
            scale = np.where(self.graph_feature_scaler.scale_ == 0.0, 1.0, self.graph_feature_scaler.scale_)
            return (X - mean) / scale
        else:  # 'minmax'
            dmin = self.graph_feature_scaler.data_min_
            drng = np.where(self.graph_feature_scaler.data_range_ == 0.0, 1.0, self.graph_feature_scaler.data_range_)
            return (X - dmin) / drng
        
    def save(self, save_path: str):
        """
        Save fitted normalizers.
        
        Args:
            save_path: Path to save normalizers
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        normalizer_data = {
            'anthropometric_scaler': self.anthropometric_scaler,
            'deformation_scaler': self.deformation_scaler,
            'graph_feature_scaler': self.graph_feature_scaler,
            'structure_deformation_scalers': self.structure_deformation_scalers,
            'normalization_method': self.normalization_method,
            'fitted': self.fitted,
            'max_deformation_size': self.max_deformation_size,
            'statistics': self.statistics
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(normalizer_data, f)
        
        print(f"  Normalizers saved to: {save_path}")
    
    def load(self, load_path: str):
        """
        Load fitted normalizers.
        
        Args:
            load_path: Path to load normalizers from
        """
        with open(load_path, 'rb') as f:
            normalizer_data = pickle.load(f)
        
        self.anthropometric_scaler = normalizer_data['anthropometric_scaler']
        self.deformation_scaler = normalizer_data['deformation_scaler']
        self.graph_feature_scaler = normalizer_data['graph_feature_scaler']
        self.structure_deformation_scalers = normalizer_data.get('structure_deformation_scalers', {})
        self.normalization_method = normalizer_data['normalization_method']
        self.fitted = normalizer_data['fitted']
        self.max_deformation_size = normalizer_data.get('max_deformation_size', 0)
        self.statistics = normalizer_data.get('statistics', {})
        
        print(f"  Normalizers loaded from: {load_path}")
    
    def transform(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply fitted normalizers to samples (no fitting). Use for val/test splits.
        """
        if not self.fitted:
            raise ValueError("Normalizers not fitted yet!")
        return self._transform_samples(samples)
    def export(self) -> Dict[str, Any]:
        return {
            'anthropometric_scaler': self.anthropometric_scaler,
            'deformation_scaler': self.deformation_scaler,
            'graph_feature_scaler': self.graph_feature_scaler,
            'structure_deformation_scalers': self.structure_deformation_scalers,
            'normalization_method': self.normalization_method,
        }
