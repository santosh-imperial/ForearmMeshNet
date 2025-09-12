# forearm_meshnet/models/decoder.py
"""
Structure-Aware Decoder for ForearmMeshNet
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FiLMParams(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) parameters.
    
    Generates affine transformation parameters for conditioning
    layers based on anthropometric features.
    """
    
    def __init__(self, c_dim: int, channels_per_layer: List[int], hidden: int = 128):
        """
        Initialize FiLM parameter generator.
        
        Args:
            c_dim: Dimension of conditioning features
            channels_per_layer: List of channel dimensions for each layer
            hidden: Hidden layer dimension
        """
        super().__init__()
        
        self.channels_per_layer = list(channels_per_layer)
        out_dim = 2 * sum(self.channels_per_layer)  # gamma and beta for each channel
        
        self.network = nn.Sequential(
            nn.Linear(c_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
        
        # Initialize near identity so FiLM is neutral initially
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, c: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate FiLM parameters.
        
        Args:
            c: Conditioning features [batch_size, c_dim]
            
        Returns:
            gammas: List of scale parameters for each layer
            betas: List of shift parameters for each layer
        """
        params = self.network(c)  # [B, 2*sum(channels)]
        
        sum_channels = sum(self.channels_per_layer)
        gammas_flat, betas_flat = params.split(sum_channels, dim=-1)
        
        # Split into per-layer parameters
        gammas = torch.split(gammas_flat, self.channels_per_layer, dim=-1)
        betas = torch.split(betas_flat, self.channels_per_layer, dim=-1)
        
        return gammas, betas


class AnthroAffine(nn.Module):
    """
    Anthropometric-conditioned affine transformation.
    
    Predicts global scale and translation based on anthropometric features
    for initial coarse alignment.
    """
    
    def __init__(self, c_dim: int, hidden: int = 64):
        """
        Initialize AnthroAffine.
        
        Args:
            c_dim: Dimension of anthropometric features
            hidden: Hidden layer dimension
        """
        super().__init__()
        
        # Scale predictor (3D anisotropic scaling)
        self.scale_net = nn.Sequential(
            nn.Linear(c_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3)
        )
        
        # Translation predictor
        self.trans_net = nn.Sequential(
            nn.Linear(c_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3)
        )
        
        # Initialize to identity transformation
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.ones_(self.scale_net[-1].bias)  # Start with scale = 1
        
        nn.init.zeros_(self.trans_net[-1].weight)
        nn.init.zeros_(self.trans_net[-1].bias)  # Start with translation = 0
    
    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict affine transformation parameters.
        
        Args:
            c: Anthropometric features [batch_size, c_dim]
            
        Returns:
            scale: Scale factors [batch_size, 3]
            translation: Translation vectors [batch_size, 3]
        """
        scale = self.scale_net(c)
        translation = self.trans_net(c)
        
        # Ensure positive scale
        scale = F.softplus(scale - 5.0) + 0.5  # Range approximately [0.5, inf)
        
        return scale, translation
    
    def apply_transform(self,
                       vertices: torch.Tensor,
                       scale: torch.Tensor,
                       translation: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation to vertices.
        
        Args:
            vertices: Input vertices [batch_size, num_vertices, 3]
            scale: Scale factors [batch_size, 3]
            translation: Translation vectors [batch_size, 3]
            
        Returns:
            Transformed vertices [batch_size, num_vertices, 3]
        """
        # Apply scale
        scaled = vertices * scale.unsqueeze(1)
        
        # Apply translation
        transformed = scaled + translation.unsqueeze(1)
        
        return transformed


class StructureAwareDecoder(nn.Module):
    """
    Decoder that generates structure-specific deformations.
    
    Produces separate deformation predictions for each anatomical structure
    (skin and muscles) with FiLM conditioning.
    """
    
    def __init__(self,
                 latent_dim: int,
                 anthro_dim: int,
                 hidden_dims: List[int],
                 num_vertices_per_structure: Dict[str, int],
                 dropout_rate: float = 0.3):
        """
        Initialize StructureAwareDecoder.
        
        Args:
            latent_dim: Dimension of latent code
            anthro_dim: Dimension of anthropometric features
            hidden_dims: List of hidden layer dimensions
            num_vertices_per_structure: Dictionary mapping structure names to vertex counts
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.num_vertices_per_structure = num_vertices_per_structure
        self.latent_dim = latent_dim
        self.anthro_dim = anthro_dim
        
        # FiLM will condition layers; decoder input is latent only
        input_dim = latent_dim
        
        # Shared decoder backbone
        self.shared_layers = nn.ModuleList()
        self.shared_norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_dims])
        self.film = FiLMParams(anthro_dim, hidden_dims)

        
        # Residual connections
        self.residual_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.shared_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i > 0 and i % 2 == 0:
                # Linear(dims[i-1] -> dims[i+1])
                self.residual_layers.append(nn.Linear(dims[i - 1], dims[i + 1]))
        
        # Structure-specific deformation heads
        self.structure_heads = nn.ModuleDict()
        for structure_name, num_vertices in num_vertices_per_structure.items():
            self.structure_heads[structure_name] = nn.Sequential(
                nn.Linear(hidden_dims[-1], 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # the loop uses F.dropout p=0.2; this head keeps your arg
                nn.Linear(256, num_vertices * 3),
            )

        # Per-structure scale from anthropometry
        self.scale_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(anthro_dim, 64), nn.SiLU(),
                nn.Linear(64, 1)
            ) for name in num_vertices_per_structure.keys()
        })
        for head in self.scale_heads.values():
            nn.init.zeros_(head[-1].weight)
            with torch.no_grad():
                # bias so softplus(bias) ≈ 1.0
                head[-1].bias.fill_(math.log(math.exp(1.0) - 1.0))
    
    def forward(self,
                z: torch.Tensor,
                anthropometric_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate structure-specific deformations.
        
        Args:
            z: Latent code [batch_size, latent_dim]
            anthropometric_features: Anthropometric features [batch_size, anthro_dim]
            
        Returns:
            Dictionary mapping structure names to deformation tensors
            Each tensor has shape [batch_size, num_vertices, 3]
        """
        
        # Get FiLM parameters
        gammas, betas = self.film(anthropometric_features)
        
        # Pass through shared layers with FiLM modulation
        x = z
        residual = None
        
        for i, layer in enumerate(self.shared_layers):
            x_prev = x
            
            # Linear transformation
            x = layer(x)
            
            # Layer normalization
            x = self.shared_norms[i](x)
            
            # FiLM modulation
            x = (1.0 + gammas[i]) * x + betas[i]
            
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            
            # Residual connection
            if i > 0 and i % 2 == 0 and len(self.residual_layers) > i // 2 - 1:
                if residual is not None:
                    x = x + self.residual_layers[i // 2 - 1](residual)
                residual = x_prev
            else:
                residual = x_prev
        
        # Generate structure-specific deformations
        structure_deformations = {}
        for structure_name, head in self.structure_heads.items():
            num_vertices = self.num_vertices_per_structure[structure_name]
            deformation = head(x).view(-1, num_vertices, 3)

            s_logit = self.scale_heads[structure_name](anthropometric_features)
            s = F.softplus(s_logit) + 1e-3  # positive scale
            structure_deformations[structure_name] = s.view(-1, 1, 1) * deformation

        return structure_deformations
    
    def decode_deterministic(self,
                            z: torch.Tensor,
                            c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Deterministic decoding (no dropout).
        
        Args:
            z: Latent code
            c: Anthropometric features
            
        Returns:
            Structure deformations
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            deformations = self.forward(z, c)
        
        if was_training:
            self.train()
        
        return deformations


class TemplateAugmentor(nn.Module):
    """
    Template augmentation for improved generalization.
    
    Applies random geometric transformations to template during training
    to make the model more robust to template variations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TemplateAugmentor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.augment_scale = config.get('augment_scale', 0.1)
        self.augment_rotate = config.get('augment_rotate', 0.1)
        self.augment_translate = config.get('augment_translate', 5.0)
        self.augment_noise = config.get('augment_noise', 1.0)
    
    def forward(self,
                vertices: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        """
        Apply augmentation to template vertices.
        
        Args:
            vertices: Template vertices [batch_size, num_vertices, 3]
            training: Whether in training mode
            
        Returns:
            Augmented vertices
        """
        if not training or not self.training:
            return vertices
        
        batch_size = vertices.shape[0]
        device = vertices.device
        
        # Random scaling
        if self.augment_scale > 0:
            scale = 1.0 + (torch.rand(batch_size, 3, device=device) - 0.5) * self.augment_scale
            vertices = vertices * scale.unsqueeze(1)
        
        # Random rotation (small angles)
        if self.augment_rotate > 0:
            angles = (torch.rand(batch_size, 3, device=device) - 0.5) * self.augment_rotate
            vertices = self._rotate_vertices(vertices, angles)
        
        # Random translation
        if self.augment_translate > 0:
            trans = (torch.rand(batch_size, 3, device=device) - 0.5) * self.augment_translate
            vertices = vertices + trans.unsqueeze(1)
        
        # Random noise
        if self.augment_noise > 0:
            noise = torch.randn_like(vertices) * self.augment_noise
            vertices = vertices + noise
        
        return vertices
    
    def _rotate_vertices(self,
                        vertices: torch.Tensor,
                        angles: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to vertices.
        
        Args:
            vertices: Input vertices [batch_size, num_vertices, 3]
            angles: Rotation angles in radians [batch_size, 3]
            
        Returns:
            Rotated vertices
        """
        batch_size = vertices.shape[0]
        
        # Create rotation matrices for each sample
        cos_x = torch.cos(angles[:, 0])
        sin_x = torch.sin(angles[:, 0])
        cos_y = torch.cos(angles[:, 1])
        sin_y = torch.sin(angles[:, 1])
        cos_z = torch.cos(angles[:, 2])
        sin_z = torch.sin(angles[:, 2])
        
        # Rotation matrix around X axis
        Rx = torch.eye(3, device=vertices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Rx[:, 1, 1] = cos_x
        Rx[:, 1, 2] = -sin_x
        Rx[:, 2, 1] = sin_x
        Rx[:, 2, 2] = cos_x
        
        # Rotation matrix around Y axis
        Ry = torch.eye(3, device=vertices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Ry[:, 0, 0] = cos_y
        Ry[:, 0, 2] = sin_y
        Ry[:, 2, 0] = -sin_y
        Ry[:, 2, 2] = cos_y
        
        # Rotation matrix around Z axis
        Rz = torch.eye(3, device=vertices.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Rz[:, 0, 0] = cos_z
        Rz[:, 0, 1] = -sin_z
        Rz[:, 1, 0] = sin_z
        Rz[:, 1, 1] = cos_z
        
        # Combined rotation matrix
        R = torch.bmm(torch.bmm(Rz, Ry), Rx)
        
        # Apply rotation
        rotated = torch.bmm(vertices, R.transpose(1, 2))
        
        return rotated