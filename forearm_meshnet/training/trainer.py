
# forearm_meshnet/training/trainer.py

"""
Main training class for ForearmMeshNet
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..models import ForearmMeshNet, CombinedLoss
from ..data import ForearmDataset, ForearmDataLoader
from .curriculum import CurriculumManager
from .metrics import MeshEvaluationMetrics

logger = logging.getLogger(__name__)


def _build_edges_from_faces(faces_np):
    """Build unique undirected edges from triangle faces (numpy array F x 3)."""
    if faces_np is None or len(faces_np) == 0:
        return None
    edges = set()
    for (a, b, c) in faces_np:
        e = [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]
        for u, v in e:
            i, j = (u, v) if u < v else (v, u)
            edges.add((i, j))
    if not edges:
        return None
    edges = torch.tensor(sorted(list(edges)), dtype=torch.long)  # [E,2]
    return edges

def _uniform_graph_laplacian(num_verts: int, edges: torch.Tensor, device=None):
    """
    Unweighted graph Laplacian L = D - A as a torch.sparse_coo_tensor of shape [N,N].
    Edges are undirected pairs [E,2] with 0 <= idx < N.
    """
    if edges is None or edges.numel() == 0:
        return None
    if device is None:
        device = edges.device

    E = edges.shape[0]
    i = edges[:, 0]
    j = edges[:, 1]

    # adjacency indices (both directions)
    idx = torch.cat([torch.stack([i, j], dim=0), torch.stack([j, i], dim=0)], dim=1)  # [2, 2E]
    vals = torch.ones(idx.shape[1], device=device)

    A = torch.sparse_coo_tensor(idx, vals, (num_verts, num_verts), device=device)

    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    D_idx = torch.arange(num_verts, device=device)
    D = torch.sparse_coo_tensor(
        torch.stack([D_idx, D_idx], dim=0),
        deg,
        (num_verts, num_verts),
        device=device,
    )
    L = D - A
    return L.coalesce()

def _load_unified_template_assets_local(pkl_path: str):
    """
    Load unified template (.pkl) and split per-structure assets:
    returns dict: {name: {'vertices': Tensor[N,3], 'edges': Long[E,2] or None,
                          'faces': Long[F,3] or None, 'laplacian': sparse or None}}
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"unified_template_pickle not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        tpl = pickle.load(f)

    verts_np = tpl['vertices']            # (N,3) float
    faces_np = tpl.get('faces', None)     # (F,3) int or None
    s_info   = tpl['structure_info']      # dict with 'vertex_range' and optional 'face_range'

    assets = {}
    for name, info in s_info.items():
        v0, v1 = info['vertex_range']
        verts = torch.tensor(verts_np[v0:v1], dtype=torch.float32)  # [n_i, 3]

        # faces (local) -> shift by -v0 to be local indices
        faces_local = None
        if 'face_range' in info:
            f0, f1 = info['face_range']
            if faces_np is not None and f1 > f0:
                faces_block = faces_np[f0:f1].copy()
                faces_block = faces_block - v0
                # keep only faces whose all indices fall into [0, v1-v0)
                n_i = v1 - v0
                mask = ((faces_block >= 0) & (faces_block < n_i)).all(axis=1)
                faces_block = faces_block[mask]
                if len(faces_block) > 0:
                    faces_local = torch.tensor(faces_block, dtype=torch.long)

        # edges (from faces)
        edges = _build_edges_from_faces(faces_local.cpu().numpy()) if faces_local is not None else None
        # laplacian
        lap = _uniform_graph_laplacian(verts.shape[0], edges) if edges is not None else None

        assets[name] = {
            'vertices': verts,         # [n_i,3] float32 (CPU; move later)
            'faces': faces_local,      # [F_i,3] long or None
            'edges': edges,            # [E_i,2] long or None
            'laplacian': lap,          # sparse [n_i,n_i] or None
        }
    return assets


class Trainer:
    """
    Training infrastructure for ForearmMeshNet.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self,
                 model: ForearmMeshNet,
                 train_dataset: ForearmDataset,
                 val_dataset: ForearmDataset,
                 config: Dict[str, Any],
                 output_dir: str = "./output"):
        """
        Initialize Trainer.
        
        Args:
            model: ForearmMeshNet model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Extract structure information
        self.structure_info = train_dataset.samples[0].get('structure_info', {})

        # Load real template assets (verts/edges/faces/laplacian) for losses
        tpl_pkl = self.config.get('unified_template_pickle', None)
        if tpl_pkl is not None:
            self.template_assets = _load_unified_template_assets_local(tpl_pkl)
            logger.info(f"Loaded template assets from {tpl_pkl} (structures: {list(self.template_assets.keys())})")
        else:
            self.template_assets = None
            logger.warning("'unified_template_pickle' not provided; geometric losses will use dummy assets.")
        
        # Data loaders
        self.train_loader = ForearmDataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=True
        )
        
        self.val_loader = ForearmDataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True
        )

        loss_cfg = dict(self.config.get('loss_config', {}))
        norm = getattr(self.model, 'normalizer', None)
        scalers = None
        if isinstance(norm, dict):
            scalers = norm.get('structure_deformation_scalers')
        elif hasattr(norm, 'structure_deformation_scalers'):
            scalers = norm.structure_deformation_scalers
        if scalers is not None:
            loss_cfg['deformation_scalers'] = scalers
                
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss function
        self.criterion = CombinedLoss(
            structure_info=self.structure_info,
            config=loss_cfg
        )
        
        # Curriculum manager
        self.curriculum_manager = CurriculumManager(
            train_dataset.samples,
            config.get('curriculum_config', {})
        )
        
        # Metrics evaluator
        self.metrics_evaluator = MeshEvaluationMetrics(
            device=self.device,
            structure_info=self.structure_info
        )
        
        # Logging
        #self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        self.best_model_path = None
        
        logger.info("Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Batch size: {config.get('batch_size', 8)}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'Adam')
        
        if optimizer_type == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        elif optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 1e-2)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 50),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 200),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        logger.info("TRAINING FOREARM MESHNET")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")
        
        # Training phases for curriculum learning
        phases = self._get_training_phases(num_epochs)
        current_phase_idx = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update training phase
            current_phase = self._get_current_phase(epoch, phases)
            if current_phase != phases[current_phase_idx]:
                current_phase_idx += 1
                logger.info(f"PHASE: {current_phase['name']}")
                logger.info(f"Description: {current_phase['description']}")
            
            # Update curriculum
            self.curriculum_manager.update_epoch(epoch)
            
            # Training step
            train_losses = self._train_epoch(epoch)
            
            # Validation step
            val_losses = self._validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Logging
            self._log_epoch(epoch, train_losses, val_losses, current_phase)
            
            # Checkpointing
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self._save_checkpoint(epoch, is_best=True)
            elif epoch % self.config.get('checkpoint_freq', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self._check_early_stopping(epoch, val_losses['total']):
                logger.info("Early stopping triggered!")
                break
        
        # Save final model
        epoch = getattr(self, 'current_epoch', num_epochs - 1)
        self._save_checkpoint(epoch, is_best=False, is_final=True)
        
        # Save training summary
        self._save_training_summary()
        
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return dict(self.training_history)
    
    def _get_training_phases(self, num_epochs: int) -> List[Dict]:
        """Get training phases for curriculum learning."""
        phases = [
            {
                'name': 'Basic',
                'epochs': min(50, num_epochs // 4),
                'description': 'Skin focus, easy samples'
            },
            {
                'name': 'Multi-Structure',
                'epochs': min(50, num_epochs // 4),
                'description': 'Add muscles, medium samples'
            },
            {
                'name': 'Refinement',
                'epochs': min(50, num_epochs // 4),
                'description': 'All structures, all samples'
            },
            {
                'name': 'Fine-tuning',
                'epochs': num_epochs - min(150, 3 * num_epochs // 4),
                'description': 'Final optimization'
            }
        ]
        
        return phases
    
    def _get_current_phase(self, epoch: int, phases: List[Dict]) -> Dict:
        """Get current training phase."""
        phase_sum = 0
        for phase in phases:
            phase_sum += phase['epochs']
            if epoch < phase_sum:
                return phase
        return phases[-1]
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = defaultdict(float)
        num_batches = 0
        
        # Get samples for this epoch (curriculum learning)
        epoch_samples = self.curriculum_manager.get_epoch_samples()
        epoch_dataset = ForearmDataset(epoch_samples, augment=True)
        epoch_loader = ForearmDataLoader(
            epoch_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=0
        )
        
        progress_bar = tqdm(epoch_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                batch['unified_template_graph'],
                batch['anthropometric_features']
            )
            
            # Compute loss
            loss, loss_dict = self.criterion(
                pred_deformations=outputs['structure_deformations'],
                target_deformations=batch['structure_deformations'],
                template_meshes=self._get_template_meshes(batch),
                affine_meshes=None,  
                mu=outputs.get('mu'),
                logvar=outputs.get('logvar'),
                prior_mu=outputs.get('prior_mu'),
                prior_logvar=outputs.get('prior_logvar'),
                epoch=epoch,
               #structure_masks=batch.get('structure_masks')
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    batch['unified_template_graph'],
                    batch['anthropometric_features']
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    pred_deformations=outputs['structure_deformations'],
                    target_deformations=batch['structure_deformations'],
                    template_meshes=self._get_template_meshes(batch),
                    affine_meshes=None,
                    mu=outputs.get('mu'),
                    logvar=outputs.get('logvar'),
                    prior_mu=outputs.get('prior_mu'),
                    prior_logvar=outputs.get('prior_logvar'),
                    epoch=epoch
                )
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        batch['anthropometric_features'] = batch['anthropometric_features'].to(self.device)
        
        if batch['unified_template_graph'] is not None:
            batch['unified_template_graph'] = batch['unified_template_graph'].to(self.device)
        
        for key in batch['structure_deformations']:
            batch['structure_deformations'][key] = batch['structure_deformations'][key].to(self.device)
        
        return batch
    
    def _get_template_meshes(self, batch: Dict) -> Dict:
        """Broadcast per-structure template assets to batch size and move to device."""
        B = batch['batch_size']
        if B is None:
            try:
                any_tensor = next(iter(batch['structure_deformations'].values()))
                B = any_tensor.shape[0]
            except Exception:
                raise RuntimeError("Unable to infer batch size; please ensure batch['batch_size'] is set.")

        out = {}
        if not self.template_assets:
            # Fallback: zeros (keeps old behavior, but geometric terms won’t be meaningful)
            for name, info in batch['structure_info'].items():
                n_i = info['vertex_range'][1] - info['vertex_range'][0]
                out[name] = {
                    'vertices': torch.zeros(B, n_i, 3, device=self.device),
                    'edges': None,
                    'faces': None,
                    'laplacian': None,
                }
            return out

        for name, d in self.template_assets.items():
            verts = d['vertices'].to(self.device)                 # [n_i,3]
            if verts.dim() == 2:
                verts = verts.unsqueeze(0)                        # [1,n_i,3]
            verts = verts.expand(B, -1, -1).contiguous()          # [B,n_i,3]
            faces = d['faces'].to(self.device) if d['faces'] is not None else None
            edges = d['edges'].to(self.device) if d['edges'] is not None else None
            lap   = d['laplacian']
            if lap is not None:
                lap = torch.sparse_coo_tensor(
                    lap.indices().to(self.device),
                    lap.values().to(self.device),
                    lap.size(),
                    device=self.device
                ).coalesce()
            out[name] = {'vertices': verts, 'edges': edges, 'faces': faces, 'laplacian': lap}
        return out
    
    
    def _log_epoch(self,
                  epoch: int,
                  train_losses: Dict[str, float],
                  val_losses: Dict[str, float],
                  phase: Dict):
        """Log epoch results."""
        # Console logging
        logger.info(f"Epoch {epoch} - Phase: {phase['name']}")
        logger.info(f"  Train Loss: {train_losses['total']:.4f}")
        logger.info(f"  Val Loss: {val_losses['total']:.4f}")
        logger.info(f"  Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        """ # TensorBoard logging
        for key, value in train_losses.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch) """
        
        # History tracking
        self.training_history['epoch'].append(epoch)
        self.training_history['phase'].append(phase['name'])
        
        for key, value in train_losses.items():
            self.training_history[f'train/{key}'].append(value)
        
        for key, value in val_losses.items():
            self.training_history[f'val/{key}'].append(value)
    
    def _save_checkpoint(self,
                        epoch: int,
                        is_best: bool = False,
                        is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': dict(self.training_history)
        }
        
        if is_final:
            path = self.checkpoint_dir / 'final_model.pt'
        elif is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            self.best_model_path = path
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"  Checkpoint saved to {path}")
    
    def _check_early_stopping(self,
                             epoch: int,
                             val_loss: float) -> bool:
        """Check early stopping condition."""
        early_stop_config = self.config.get('early_stopping', {})
        
        if not early_stop_config.get('enabled', False):
            return False
        
        patience = early_stop_config.get('patience', 50)
        min_epochs = early_stop_config.get('min_epochs', 100)
        
        if epoch < min_epochs:
            return False
        
        # Check if loss has improved
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
        else:
            if not hasattr(self, 'patience_counter'):
                self.patience_counter = 0
            self.patience_counter += 1
        
        return self.patience_counter >= patience
    
    def _save_training_summary(self):
        """Save training summary."""
        summary = {
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'training_history': dict(self.training_history),
            'model_params': self.model.get_num_parameters()
        }
        
        # Save as JSON
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create plots
        self._create_loss_plots()
    
    def _create_loss_plots(self):
        """Create and save loss plots."""
        history = self.training_history
        
        if not history['epoch']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(history['epoch'], history['train/total'], label='Train')
        axes[0, 0].plot(history['epoch'], history['val/total'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        if 'train/reconstruction' in history:
            axes[0, 1].plot(history['epoch'], history['train/reconstruction'], label='Train')
            axes[0, 1].plot(history['epoch'], history['val/reconstruction'], label='Val')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Reconstruction Loss')
            axes[0, 1].legend()
            axes[0, 1].set_title('Reconstruction Loss')
            axes[0, 1].grid(True)
        
        # KL loss
        if 'train/kl' in history:
            axes[1, 0].plot(history['epoch'], history['train/kl'], label='Train')
            axes[1, 0].plot(history['epoch'], history['val/kl'], label='Val')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('KL Loss')
            axes[1, 0].legend()
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[1, 1].plot(history['epoch'], [
                self.optimizer.param_groups[0]['lr'] for _ in history['epoch']
            ])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"  Epoch: {self.current_epoch}")
        logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
