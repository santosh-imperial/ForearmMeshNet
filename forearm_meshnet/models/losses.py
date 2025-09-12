# forearm_meshnet/models/losses.py
"""
Loss functions for ForearmMeshNet training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List, Tuple
import numpy as np


class ChamferDistance(nn.Module):
    """
    Chamfer distance loss for point cloud comparison.
    """
    
    def __init__(self):
        """Initialize ChamferDistance."""
        super().__init__()
    
    def forward(self,
                pred_points: torch.Tensor,
                target_points: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional Chamfer distance.
        
        Args:
            pred_points: Predicted points [batch_size, N, 3]
            target_points: Target points [batch_size, M, 3]
            
        Returns:
            Chamfer distance (scalar)
        """
        batch_size = pred_points.shape[0]
        
        # Compute pairwise distances
        distances = torch.cdist(pred_points, target_points)
        
        # Forward distance (pred to target)
        forward_dist = torch.min(distances, dim=2)[0].mean(dim=1)
        
        # Backward distance (target to pred)
        backward_dist = torch.min(distances, dim=1)[0].mean(dim=1)
        
        # Bidirectional Chamfer distance
        chamfer = (forward_dist + backward_dist).mean()
        
        return chamfer


class EdgeLengthLoss(nn.Module):
    """
    Edge length preservation loss for mesh deformation.
    """
    
    def __init__(self):
        """Initialize EdgeLengthLoss."""
        super().__init__()
    
    def forward(self,
                pred_vertices: torch.Tensor,
                target_vertices: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """
        Compute edge length preservation loss.
        
        Args:
            pred_vertices: Predicted vertices [batch_size, N, 3]
            target_vertices: Target vertices [batch_size, N, 3]
            edges: Edge connectivity [E, 2]
            
        Returns:
            Edge length loss (scalar)
        """
        batch_size = pred_vertices.shape[0]
        
        # Compute edge lengths for predicted mesh
        pred_v1 = pred_vertices[:, edges[:, 0]]
        pred_v2 = pred_vertices[:, edges[:, 1]]
        pred_lengths = torch.norm(pred_v2 - pred_v1, dim=-1)
        
        # Compute edge lengths for target mesh
        target_v1 = target_vertices[:, edges[:, 0]]
        target_v2 = target_vertices[:, edges[:, 1]]
        target_lengths = torch.norm(target_v2 - target_v1, dim=-1)
        
        # L2 loss on edge lengths
        edge_loss = F.mse_loss(pred_lengths, target_lengths)
        
        return edge_loss


class NormalConsistencyLoss(nn.Module):
    """
    Normal consistency loss for smooth surface deformation.
    """
    
    def __init__(self):
        """Initialize NormalConsistencyLoss."""
        super().__init__()
    
    def forward(self,
                pred_vertices: torch.Tensor,
                target_vertices: torch.Tensor,
                faces: torch.Tensor) -> torch.Tensor:
        """
        Compute normal consistency loss.
        
        Args:
            pred_vertices: Predicted vertices [batch_size, N, 3]
            target_vertices: Target vertices [batch_size, N, 3]
            faces: Face connectivity [F, 3]
            
        Returns:
            Normal consistency loss (scalar)
        """
        # Compute face normals for predicted mesh
        pred_normals = self._compute_face_normals(pred_vertices, faces)
        
        # Compute face normals for target mesh
        target_normals = self._compute_face_normals(target_vertices, faces)
        
        dot = (pred_normals * target_normals).sum(dim=-1).clamp(-0.999, 0.999)
        one_minus_cos = 1.0 - dot
        normal_loss = F.smooth_l1_loss(one_minus_cos, torch.zeros_like(one_minus_cos), beta=0.1)
            
        return normal_loss
    
    def _compute_face_normals(self,
                              vertices: torch.Tensor,
                              faces: torch.Tensor) -> torch.Tensor:
        """
        Compute face normals.
        
        Args:
            vertices: Vertices [batch_size, N, 3]
            faces: Faces [F, 3]
            
        Returns:
            Face normals [batch_size, F, 3]
        """
        batch_size = vertices.shape[0]
        
        # Get face vertices
        v0 = vertices[:, faces[:, 0]]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]
        
        # Compute edges
        e1 = v1 - v0
        e2 = v2 - v0
        
        # Cross product for normal
        normals = torch.cross(e1, e2, dim=-1)
        
        # Normalize
        normals = F.normalize(normals, dim=-1)
        
        return normals


class LaplacianSmoothingLoss(nn.Module):
    """
    Laplacian smoothing loss for mesh regularization.
    """
    
    def __init__(self):
        """Initialize LaplacianSmoothingLoss."""
        super().__init__()
    
    def forward(self,
                vertices: torch.Tensor,
                laplacian: torch.sparse.Tensor) -> torch.Tensor:
        """
        Compute Laplacian smoothing loss.
        
        Args:
            vertices: Vertices [batch_size, N, 3]
            laplacian: Laplacian matrix [N, N] (sparse)
            
        Returns:
            Laplacian smoothing loss (scalar)
        """
        batch_size = vertices.shape[0]
        loss = 0
        
        for b in range(batch_size):
            # Apply Laplacian
            Lv = torch.sparse.mm(laplacian, vertices[b]) 
            
            # L2 norm of Laplacian coordinates
            loss = loss + (Lv.pow(2).sum(dim=-1).mean())   # mean per-vertex squared laplacian
        
        return loss / batch_size


class VolumeLoss(nn.Module):
    """
    Volume preservation loss for mesh deformation.
    
    """ 
    def __init__(self, clamp_tau: float = 3.0, eps: float = 1e-8):
        super().__init__()
        self.clamp_tau = clamp_tau
        self.eps = eps
    def forward(self, X: torch.Tensor, Y: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        u = Y - X                                 # [V,3] displacement
        vi, vj = edges[0], edges[1]              # [E]
        Xe = X[vj] - X[vi]                        # [E,3]
        Ue = u[vj] - u[vi]                        # [E,3]
        num = (Ue * Xe).sum(dim=-1)               # [E]
        den = (Xe * Xe).sum(dim=-1) + self.eps    # [E]
        dir_deriv = num / den                     # [E]
        V = X.shape[0]
        contrib = torch.zeros(V, device=X.device)
        deg    = torch.zeros(V, device=X.device)
        contrib.index_add_(0, vi, dir_deriv)
        contrib.index_add_(0, vj, dir_deriv)
        ones = torch.ones_like(dir_deriv)
        deg.index_add_(0, vi, ones)
        deg.index_add_(0, vj, ones)
        div_v = contrib / deg.clamp_min(1.0)      # [V]
        return torch.exp(torch.clamp(div_v, max=self.clamp_tau)).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for ForearmMeshNet training.
    
    Combines reconstruction, geometric, and regularization losses
    with adaptive weighting and curriculum learning.
    """
    
    def __init__(self,
                 structure_info: Dict[str, Any],
                 config: Optional[Dict] = None):
        """
        Initialize CombinedLoss.
        
        Args:
            structure_info: Information about anatomical structures
            config: Loss configuration
        """
        super().__init__()
        
        self.structure_info = structure_info
        self.config = config or {}
        
        # Individual loss components
        self.chamfer_loss = ChamferDistance()
        self.edge_loss = EdgeLengthLoss()
        self.normal_loss = NormalConsistencyLoss()
        self.laplacian_loss = LaplacianSmoothingLoss()
        self.volume_loss = VolumeLoss()
        self.deformation_scalers = (config or {}).get('deformation_scalers', {})

        
        # Loss weights
        self.lambda_weights = {
            'reconstruction': 0.35,
            'chamfer': 0.25,
            'normal': 0.20,
            'edge': 0.10,
            'laplacian': 0.05,
            'kl': 0.05,
            'volume': 0.05,
        }
        
        # Update with config if provided
        if 'lambda_weights' in config:
            self.lambda_weights.update(config['lambda_weights'])
        
        # Reconstruction sub-weights
        self.reconstruction_weights = {
            'mse': 0.55,
            'l1': 0.25,
            'l1_mm': 0.15,
            'cos_dir': 0.05,
        }
        
        # Structure-specific weights
        self.structure_weights = config.get('structure_weights', {
            'skin': 1.0,  # Most important
            'FCR': 0.8,   # Clinically significant muscles
            'FCU': 0.8,
            'FDS': 0.7,
            'FDP': 0.7,
        })
        
        # Default weight for unspecified structures
        self.default_structure_weight = 0.5
        
        # Loss scheduling for curriculum learning
        self.loss_schedules = {
            'reconstruction': lambda e: min(1.0, e / 20),
            'chamfer': lambda e: min(1.0, e / 30),
            'normal': lambda e: min(1.0, (e - 20) / 30) if e > 20 else 0,
            'edge': lambda e: min(1.0, (e - 30) / 30) if e > 30 else 0,
            'laplacian': lambda e: min(1.0, (e - 10) / 20) if e > 10 else 0,
            'volume': lambda e: min(1.0, (e - 10) / 20) if e > 10 else 0,
        }
        
        self.eps = 1e-8

    @staticmethod
    def _as_long_on(x, device):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device=device, dtype=torch.long)

    @staticmethod
    def _as_sparse_on(x, device):
        if x is None:
            return None
        return x.to(device)
    
    def forward(self,
                pred_deformations: Dict[str, torch.Tensor],
                target_deformations: Dict[str, torch.Tensor],
                template_meshes: Dict[str, Any],
                affine_meshes: Optional[Dict[str, torch.Tensor]] = None,
                mu: Optional[torch.Tensor] = None,
                logvar: Optional[torch.Tensor] = None,
                prior_mu: Optional[torch.Tensor] = None,
                prior_logvar: Optional[torch.Tensor] = None,
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred_deformations: Predicted deformations per structure
            target_deformations: Target deformations per structure
            template_meshes: Template mesh information
            affine_meshes: Affine-transformed template vertices
            mu: Posterior mean
            logvar: Posterior log variance
            prior_mu: Prior mean
            prior_logvar: Prior log variance
            epoch: Current training epoch
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        device = next(iter(pred_deformations.values())).device

        pred_mm   = self._denorm_batch(pred_deformations)
        target_mm = self._denorm_batch(target_deformations)

        #Get weights/schedules
        w = self.lambda_weights
        S = self.loss_schedules
        
        # 1) Reconstruction
        recon = self._compute_reconstruction_loss(pred_deformations, target_deformations)
        losses['reconstruction'] = w['reconstruction'] * S['reconstruction'](epoch) * recon
        
        # 2) Geometric losses
    
        
        if epoch >= 0:
            chamfer = self._compute_structure_chamfer_loss(pred_mm, target_mm,
                                                           template_meshes, affine_meshes)
            losses['chamfer'] = w['chamfer'] * S['chamfer'](epoch) * chamfer

        if epoch >= 20 and 'edges' in template_meshes.get('skin', {}):
            edge = self._compute_structure_edge_loss(pred_mm, target_mm,
                                                     template_meshes, affine_meshes)
            losses['edge'] = w['edge'] * S['edge'](epoch) * edge

        if epoch >= 20 and 'faces' in template_meshes.get('skin', {}):
            normal = self._compute_structure_normal_loss(pred_mm, target_mm,
                                                         template_meshes, affine_meshes)
            losses['normal'] = w['normal'] * S['normal'](epoch) * normal

        if epoch >= 10 and 'laplacian' in template_meshes.get('skin', {}):
            lap = self._compute_structure_laplacian_loss(pred_mm, template_meshes, affine_meshes)
            losses['laplacian'] = w['laplacian'] * S['laplacian'](epoch) * lap

        if epoch >= 10:
            vol = self._compute_structure_volume_loss(pred_mm, template_meshes, affine_meshes)
            losses['volume'] = w['volume'] * S['volume'](epoch) * vol

        
        # 3. KL divergence loss
        if mu is not None and logvar is not None:
            kl_loss = self._compute_kl_loss(mu, logvar, prior_mu, prior_logvar)
            # Warm-up KL loss
            kl_weight = min(1.0, epoch / 50) * self.lambda_weights['kl']
            losses['kl'] = kl_loss * kl_weight
        
        # Fill in zeros for inactive losses
        for loss_name in ['chamfer', 'normal', 'edge', 'laplacian', 'volume', 'kl']:
            if loss_name not in losses:
                losses[loss_name] = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = sum(losses.values())
        
        # Convert to float dict for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_reconstruction_loss(self,
                                    pred: Dict[str, torch.Tensor],
                                    target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-component reconstruction loss."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in pred.keys():
            if struct_name not in target:
                continue
            
            
            # Get structure weight
            weight = self.structure_weights.get(struct_name, self.default_structure_weight)
            total_loss += weight * F.smooth_l1_loss(pred[struct_name], target[struct_name], reduction='mean')
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_structure_chamfer_loss(self,
                                       pred: Dict[str, torch.Tensor],
                                       target: Dict[str, torch.Tensor],
                                       template_meshes: Dict,
                                       affine_meshes: Optional[Dict]) -> torch.Tensor:
        """Compute Chamfer distance for all structures."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in pred.keys():
            if struct_name not in target or struct_name not in template_meshes:
                continue
            
            # Get vertices
            if affine_meshes and struct_name in affine_meshes:
                template_verts = affine_meshes[struct_name]
            else:
                template_verts = template_meshes[struct_name].get('vertices')
            
            if template_verts is None:
                continue
            
            # Apply deformations
            pred_verts = template_verts + pred[struct_name]
            target_verts = template_verts + target[struct_name]
            
            # Compute Chamfer distance
            chamfer = self.chamfer_loss(pred_verts, target_verts)
            
            # Apply structure weight
            weight = self.structure_weights.get(struct_name, self.default_structure_weight)
            total_loss = total_loss + weight * chamfer
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_structure_edge_loss(self,
                                    pred: Dict[str, torch.Tensor],
                                    target: Dict[str, torch.Tensor],
                                    template_meshes: Dict,
                                    affine_meshes: Optional[Dict]) -> torch.Tensor:
        """Compute edge length loss for all structures."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in ['skin']:  # Only compute for skin to save computation
            if struct_name not in pred or struct_name not in template_meshes:
                continue
            
            device = pred[struct_name].device
            edges = self._as_long_on(template_meshes[struct_name].get('edges'), device)
            if edges is None:
                continue
            
            # Get vertices
            if affine_meshes and struct_name in affine_meshes:
                template_verts = affine_meshes[struct_name]
            else:
                template_verts = template_meshes[struct_name].get('vertices')
            
            # Apply deformations
            pred_verts = template_verts + pred[struct_name]
            target_verts = template_verts + target[struct_name]
            
            # Compute edge loss
            edge_loss = self.edge_loss(pred_verts, target_verts, edges)
            total_loss = total_loss + edge_loss
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_structure_normal_loss(self,
                                      pred: Dict[str, torch.Tensor],
                                      target: Dict[str, torch.Tensor],
                                      template_meshes: Dict,
                                      affine_meshes: Optional[Dict]) -> torch.Tensor:
        """Compute normal consistency loss."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in ['skin']:  # Only for skin
            if struct_name not in pred or struct_name not in template_meshes:
                continue
            device = pred[struct_name].device
            faces = self._as_long_on(template_meshes[struct_name].get('faces'), device)
            if faces is None:
                continue
            
            # Get vertices
            if affine_meshes and struct_name in affine_meshes:
                template_verts = affine_meshes[struct_name]
            else:
                template_verts = template_meshes[struct_name].get('vertices')
            
            # Apply deformations
            pred_verts = template_verts + pred[struct_name]
            target_verts = template_verts + target[struct_name]
            
            # Compute normal loss
            normal_loss = self.normal_loss(pred_verts, target_verts, faces)
            total_loss = total_loss + normal_loss
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_structure_laplacian_loss(self,
                                         pred: Dict[str, torch.Tensor],
                                         template_meshes: Dict,
                                         affine_meshes: Optional[Dict]) -> torch.Tensor:
        """Compute Laplacian smoothing loss."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in ['skin']:
            if struct_name not in pred or struct_name not in template_meshes:
                continue
            
            # Get vertices
            if affine_meshes and struct_name in affine_meshes:
                template_verts = affine_meshes[struct_name].to(device)
            else:
                tv = template_meshes[struct_name].get('vertices')
                template_verts = tv if torch.is_tensor(tv) else torch.tensor(tv, dtype=torch.float32, device=device)

            
            # Apply deformations
            deformed_verts = template_verts + pred[struct_name]

            device = pred[struct_name].device
            laplacian = template_meshes[struct_name].get('laplacian')
            if laplacian is None:
                continue
            laplacian = self._as_sparse_on(laplacian, device)
            
            # Compute Laplacian loss
            lap_loss = self.laplacian_loss(deformed_verts, laplacian)
            total_loss = total_loss + lap_loss
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_structure_volume_loss(self,
                                      pred: Dict[str, torch.Tensor],
                                      template_meshes: Dict,
                                      affine_meshes: Optional[Dict]) -> torch.Tensor:
        """Compute volume preservation loss."""
        total_loss = 0
        num_structures = 0
        
        for struct_name in pred.keys():
            if struct_name not in template_meshes:
                continue
            
            edges = template_meshes[struct_name].get('edges')
            if edges is None:
                continue
            
            # Get vertices
            if affine_meshes and struct_name in affine_meshes:
                source_verts = affine_meshes[struct_name]
            else:
                source_verts = template_meshes[struct_name].get('vertices')
            
            if source_verts is None:
                continue
            
            device = source_verts.device
            edges = self._as_long_on(edges, device)
            # Apply deformations
            X = (affine_meshes[struct_name] if affine_meshes and struct_name in affine_meshes else template_meshes[struct_name]['vertices']).to(device)
            Y = X + pred[struct_name]
            edges = self._as_long_on(template_meshes[struct_name]['edges'], device)
            
            # Compute volume loss for each sample in batch
            batch_size = source_verts.shape[0]
            batch_loss = 0
            
            for b in range(batch_size):
                vol_loss = self.volume_loss(X[b], Y[b], edges)
                batch_loss = batch_loss + vol_loss
            
            # Apply structure weight
            weight = self.structure_weights.get(struct_name, self.default_structure_weight)
            total_loss = total_loss + weight * (batch_loss / batch_size)
            num_structures += 1
        
        return total_loss / max(1, num_structures)
    
    def _compute_kl_loss(self,
                        mu: torch.Tensor,
                        logvar: torch.Tensor,
                        prior_mu: Optional[torch.Tensor],
                        prior_logvar: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute KL divergence loss."""
        if prior_mu is None or prior_logvar is None:
            # Standard normal prior
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        else:
            # Conditional prior
            kl = 0.5 * (
                prior_logvar - logvar +
                torch.exp(logvar) / torch.exp(prior_logvar) +
                ((mu - prior_mu) ** 2) / torch.exp(prior_logvar) - 1
            ).sum(dim=-1)
        
        return kl.mean()
    
    def _denorm_batch(self, struct_dict):
        if not getattr(self, 'deformation_scalers', None):
            return struct_dict
        out = {}
        for name, tensor in struct_dict.items():
            sc = self.deformation_scalers.get(name)
            if sc is None or not hasattr(sc, 'mean_') or not hasattr(sc, 'scale_'):
                out[name] = tensor
                continue
            B, V = tensor.shape[0], tensor.shape[1]
            scaler_size = len(sc.mean_)
            if scaler_size == V * 3:
                den = []
                for b in range(B):
                    flat = tensor[b].reshape(1, -1).detach().cpu().numpy()
                    inv = sc.inverse_transform(flat).reshape(V, 3)
                    den.append(torch.tensor(inv, device=tensor.device, dtype=tensor.dtype))
                out[name] = torch.stack(den, dim=0)
            elif scaler_size == 3:
                flat = tensor.reshape(-1, 3)
                mean  = torch.as_tensor(sc.mean_,  dtype=flat.dtype, device=flat.device)
                scale = torch.as_tensor(sc.scale_, dtype=flat.dtype, device=flat.device)
                out[name] = (flat * scale + mean).reshape(tensor.shape)
            else:
                out[name] = tensor
        return out