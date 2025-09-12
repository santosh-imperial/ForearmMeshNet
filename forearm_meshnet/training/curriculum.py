# forearm_meshnet/training/curriculum.py
"""
Curriculum learning module for ForearmMeshNet
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import defaultdict


class CurriculumManager:
    """
    Manages curriculum learning for progressive training.
    
    Gradually introduces more difficult samples and structures
    throughout training for better convergence.
    """
    def __init__(self, training_samples: List[Dict], config: Optional[Dict] = None):
        self.all_samples = training_samples
        self.cfg = config or {}
        self.cv_epochs = int(self.cfg.get('cv_epochs', 200))     # total curriculum horizon
        self.rng = np.random.default_rng(self.cfg.get('seed', 42))
        self.easy, self.medium, self.hard = self._bucketize()

        #  default i
        self.epoch_sample_size = int(self.cfg.get('epoch_sample_size', len(self.all_samples)))

        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _bucketize(self):
        easy, medium, hard = [], [], []
        for s in self.all_samples:
            # deformation magnitude (avg L2 over all structure vertices)
            deform_mag = 0.0
            for name, d in s.get('structure_deformations', {}).items():
                if name == 'combined':
                    continue
                arr = d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d
                deform_mag += np.linalg.norm(arr, axis=-1).mean()

            num_structs = max(1, len(s.get('structure_deformations', {})) - 1)
            score = (deform_mag / 10.0) + (20 - num_structs) / 20.0  # 

            if score < 0.7:
                easy.append(s)
            elif score < 1.3:
                medium.append(s)
            else:
                hard.append(s)
        return easy, medium, hard

    def get_epoch_samples(self) -> List[Dict]:
        # progress 0 to 1 across the curriculum window
        p = min(1.0, self.current_epoch / max(1, self.cv_epochs))

        if p < 0.3:
            pool = self.easy or (self.easy + self.medium) or self.all_samples  # fallback if empty
        elif p < 0.6:
            pool = (self.easy + self.medium) or self.all_samples
        else:
            pool = self.all_samples

        n = max(1, self.epoch_sample_size)
        if len(pool) == 0:
            pool = self.all_samples

        # sample with replacement 
        idx = self.rng.integers(0, len(pool), size=n)
        return [pool[i] for i in idx]

    def get_phase_config(self) -> Dict[str, Any]:
        # keep the API the trainer expects
        return {
            'name': 'pipeline_curriculum',
            'epochs': self.cv_epochs,
            'description': 'Easy→Mixed→All pool schedule'
        }

    def get_loss_weights(self) -> Dict[str, float]:
        return {'reconstruction': 1.0, 'geometric': 1.0, 'kl': 1.0}

    def should_evaluate_structure(self, structure_name: str) -> bool:
        return True
    
