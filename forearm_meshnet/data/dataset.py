# forearm_meshnet/data/dataset.py

"""
PyTorch Dataset class for ForearmMeshNet
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import copy
import torch.nn.functional as F

def _deep_clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # shallow copy outer dict
    out = sample.copy()
    # Clone tensors
    out['anthropometric_features'] = (
        sample['anthropometric_features'].clone()
        if isinstance(sample['anthropometric_features'], torch.Tensor)
        else torch.tensor(sample['anthropometric_features'], dtype=torch.float32).clone()
    )
    out['structure_deformations'] = {
        k: (v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32).clone())
        for k, v in sample['structure_deformations'].items()
    }

    # clone graph (it may be mutated by transforms/augs)
    g = sample.get('unified_template_graph')
    if g is not None:
        out['unified_template_graph'] = g.clone()
    return out

def _pad_to_shape(x: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    # Assumes x.dim() == len(target_shape); pads only at the end of each dim.
    if x.shape == target_shape:
        return x
    # Build F.pad tuple (last dim first)
    pads = []
    for cur, tgt in reversed(list(zip(x.shape, target_shape))):
        pads.extend([0, max(0, tgt - cur)])
    return F.pad(x, tuple(pads))



class ForearmDataset(Dataset):
    """
    PyTorch Dataset for ForearmMeshNet training.
    """
    
    def __init__(self,
                samples: List[Dict],
                transform: Optional[Any] = None,
                augment: bool = False,include_combined: bool = False):
        """
        Initialize the ForearmDataset.
        
        Args:
            samples: List of training samples
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        self.samples = samples
        self.transform = transform
        self.augment = augment
        self.include_combined = include_combined
        
        # Extract structure information from first sample
        if samples:
            self.structure_info = samples[0].get('structure_info', {})
            self.num_structures = len(self.structure_info)
        else:
            self.structure_info = {}
            self.num_structures = 0
        
        print(f"Dataset initialized with {len(samples)} samples")
        print(f"Structures: {list(self.structure_info.keys())}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        sample = _deep_clone_sample(self.samples[idx])
        
        # Apply augmentation if enabled
        if self.augment:
            sample = self._apply_augmentation(sample)
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        # Ensure all tensors are float32
        sample = self._ensure_tensor_types(sample)
        
        return sample
    
    def _apply_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply data augmentation to sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Augmented sample
        """
        # Random scaling of anthropometric features (±5%)
        if torch.rand(()) < 0.5:
            scale_factor = float(0.95 + torch.rand(()).item() * 0.10)
            sample['anthropometric_features'] = sample['anthropometric_features'] * scale_factor

        # Random small noise on per-structure deformations (not on 'combined')
        if torch.rand(()) < 0.3:
            for name, d in sample['structure_deformations'].items():
                if name != 'combined':
                    sample['structure_deformations'][name] = d + 0.01 * torch.randn_like(d)
        return sample
    
    def _ensure_tensor_types(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all data is in correct tensor format.
        
        Args:
            sample: Input sample
            
        Returns:
            Sample with correct tensor types
        """
        # Ensure anthropometric features are tensor
        if not isinstance(sample['anthropometric_features'], torch.Tensor):
            sample['anthropometric_features'] = torch.tensor(
                sample['anthropometric_features'], dtype=torch.float32
            )
        
        # Ensure deformations are tensors
        for structure_name in sample['structure_deformations']:
            deform = sample['structure_deformations'][structure_name]
            if not isinstance(deform, torch.Tensor):
                sample['structure_deformations'][structure_name] = torch.tensor(
                    deform, dtype=torch.float32
                )
        
        return sample
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for batching.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # 1) Anthropometric features
        anthro_features = torch.stack([s['anthropometric_features'] for s in batch])

        # 2) Graphs
        graphs = [s.get('unified_template_graph') for s in batch if s.get('unified_template_graph') is not None]
        batched_graph = Batch.from_data_list(graphs) if graphs else None

        # 3) Structures: use UNION of keys across the batch
        all_names = set()
        for s in batch:
            all_names.update(s.get('structure_deformations', {}).keys())
        # Keep a stable order: put 'combined' first if present
        structure_names = ['combined'] + sorted(n for n in all_names if n != 'combined')
        if not any(getattr(b.__class__, "include_combined", False) for b in [ForearmDataset]):  
            structure_names = [n for n in structure_names if n != 'combined']
        batched_deformations, batched_masks = {}, {}
        vertex_masks = {}
        presence_masks = {}

        # First pass: compute per-structure max shapes
        max_shapes = {}
        for name in structure_names:
            present = [s['structure_deformations'][name] for s in batch if name in s['structure_deformations']]
            if not present:
                continue
            ref_dim = present[0].dim()
            max_shape = tuple(max(t.shape[i] for t in present) for i in range(ref_dim))
            max_shapes[name] = max_shape

        # Second pass: pad (or synthesize zeros for missing) + masks
        for name in structure_names:
            if name not in max_shapes:
                continue
            max_shape = max_shapes[name]

            padded_list, vmask_list, pmask_list = [], [], []
            for s in batch:
                if name in s['structure_deformations']:
                    d = s['structure_deformations'][name]
                    # mask: ones on real rows (assume vector field -> last dim is channels)
                    base_mask = torch.ones_like(d[..., 0], dtype=torch.bool)
                    d_pad = _pad_to_shape(d, max_shape)
                    m_pad = _pad_to_shape(base_mask.unsqueeze(-1), max_shape[:-1] + (1,)).squeeze(-1)
                    present = True
                else:
                    # synthesize zeros of target shape for absent structure
                    d_pad = torch.zeros(*max_shape, dtype=torch.float32)
                    m_pad = torch.zeros(max_shape[:-1], dtype=torch.bool)
                    present = False
                padded_list.append(d_pad)
                vmask_list.append(m_pad)
                pmask_list.append(present)

            batched_deformations[name] = torch.stack(padded_list, dim=0)  # (B, *max_shape)
            vertex_masks[name] = torch.stack(vmask_list, dim=0)              # (B, V[, ...])
            presence_masks[name] = torch.tensor(pmask_list, dtype=torch.bool)  # [B]

        # connectivity (from first sample if available)
        tmpl_conn = batch[0].get('template_mesh_connectivity') if batch else None
   
        out = {
            'anthropometric_features': anthro_features,
            'unified_template_graph': batched_graph,
            'structure_deformations': batched_deformations,
            'structure_masks': presence_masks,
            'vertex_masks': vertex_masks, 
            'batch_size': len(batch),
            'structure_info': batch[0].get('structure_info', {}) if batch else {},
            'template_mesh_connectivity': tmpl_conn
        }
        if batch and 'subject_id' in batch[0]:
            out['subject_ids'] = [s.get('subject_id') for s in batch]
        return out


class ForearmDataLoader:
    """
    DataLoader wrapper for ForearmDataset with custom batching.
    """
    
    def __init__(self,
                dataset: ForearmDataset,
                batch_size: int = 8,
                shuffle: bool = True,
                num_workers: int = 0,
                pin_memory: bool = True):
        """
        Initialize the ForearmDataLoader.
        
        Args:
            dataset: ForearmDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=ForearmDataset.collate_fn
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.loader)

