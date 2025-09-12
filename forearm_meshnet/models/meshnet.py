# forearm_meshnet/models/meshnet.py
"""
Complete ForearmMeshNet model combining encoder and decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from .encoder import VariationalGraphEncoder
from .decoder import StructureAwareDecoder, AnthroAffine, TemplateAugmentor


class ForearmMeshNet(nn.Module):
    """
    Complete ForearmMeshNet with VAE architecture and multi-structure support.
    
    This model learns to predict forearm mesh deformations from anthropometric
    measurements using a variational autoencoder framework with graph convolutions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ForearmMeshNet.
        
        Args:
            config: Configuration dictionary containing:
                - node_feature_dim: Dimension of graph node features
                - anthro_feature_dim: Dimension of anthropometric features
                - encoder_hidden_dims: List of encoder hidden dimensions
                - decoder_hidden_dims: List of decoder hidden dimensions
                - latent_dim: Dimension of latent space
                - num_structures: Number of anatomical structures
                - structure_vertex_counts: Dict mapping structure names to vertex counts
                - dropout_rate: Dropout probability
                - conv_type: Type of graph convolution
                - use_template_augmentation: Whether to use template augmentation
                - use_affine: Whether to use affine transformation
        """
        super().__init__()
        
        self.config = config
        self.structure_info = None
        self.normalizer = None  # Will be set by trainer
        
        # Extract configuration
        self.num_structures = config['num_structures']
        self.structure_vertex_counts = config['structure_vertex_counts']
        self.latent_dim = config['latent_dim']
        self.use_affine = config.get('use_affine', True)
        
        # Template augmentation
        self.use_template_augmentation = config.get('use_template_augmentation', True)
        if self.use_template_augmentation:
            self.template_augmentor = TemplateAugmentor(config)
        
        # Anthropometric feature processor
        self.anthro_processor = nn.Sequential(
            nn.Linear(config['anthro_feature_dim'], 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, config['anthro_feature_dim'])
        )
        
        # Encoder
        self.encoder = VariationalGraphEncoder(
            input_dim=config['node_feature_dim'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            num_structures=self.num_structures,
            dropout_rate=config['dropout_rate'],
            conv_type=config['conv_type'],
            cond_dim=config['anthro_feature_dim']
        )
        
        # Decoder
        self.decoder = StructureAwareDecoder(
            latent_dim=config['latent_dim'],
            anthro_dim=config['anthro_feature_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            num_vertices_per_structure=self.structure_vertex_counts,
            dropout_rate=config['dropout_rate']
        )
        
        # Affine transformation module
        if self.use_affine:
            self.affine = AnthroAffine(
                c_dim=config['anthro_feature_dim'],
                hidden=64
            )
        
        # Conditional prior p(z|c)
        self.prior_net = nn.Sequential(
            nn.Linear(config['anthro_feature_dim'], 128), nn.SiLU(),
            nn.Linear(128, 2 * config['latent_dim'])
        )

        # latent dropout 
        self.latent_dropout_p = float(config.get('latent_dropout_p', 0.07))
        self.latent_dropout_mode = str(config.get('latent_dropout_mode', 'prior_mean'))
        
        # Initialize weights
        #self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if not hasattr(m, '_initialized'):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    m._initialized = True
    
    def set_normalizer(self, normalizer: Any):
        """
        Set data normalizer for inference.
        
        Args:
            normalizer: Data normalizer object
        """
        self.normalizer = normalizer
    
    def forward(self,
                graph_batch: Any,
                anthro_features: torch.Tensor,
                template_vertices: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            graph_batch: Batched graph data (PyTorch Geometric)
            anthro_features: Anthropometric features [batch_size, anthro_dim]
            template_vertices: Optional template vertices for affine transform
            
        Returns:
            Dictionary containing:
                - structure_deformations: Dict of deformations per structure
                - z: Latent code
                - mu: Mean of posterior
                - logvar: Log variance of posterior
                - prior_mu: Mean of prior
                - prior_logvar: Log variance of prior
                - affine_params: Optional affine transformation parameters
        """
        batch_size = anthro_features.shape[0]
        
        # Process anthropometric features
        anthro_processed = self.anthro_processor(anthro_features)

        # Template augmentation 
        if self.training and self.use_template_augmentation:
             graph_batch = self.template_augmentor.augment_template_graph(graph_batch, anthro_processed)

        
        # Encode graph to latent space
        mu, logvar = self.encoder(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            batch=getattr(graph_batch, 'batch', None),
            edge_attr=getattr(graph_batch, 'edge_attr', None),
            pos=getattr(graph_batch, 'pos', None),
            cond=anthro_processed,
        )

        #conditional prior p(z|c)
        prior_params = self.prior_net(anthro_processed)
        prior_mu, prior_logvar = prior_params.split(self.latent_dim, dim=-1)

        if self.training:
            # reparameterize with posterior
            logvar_clamped = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar_clamped)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # eval: use prior mean
            z = prior_mu

        

        # latent dropout 
        if self.training and self.latent_dropout_p > 0.0:
            drop_mask = (torch.rand(z.size(0), 1, device=z.device) < self.latent_dropout_p).float()
            if self.latent_dropout_mode == 'zero':
                z = z * (1.0 - drop_mask)
            elif self.latent_dropout_mode == 'prior_mean':
                z = z * (1.0 - drop_mask) + prior_mu * drop_mask
            else:
                raise ValueError(f"Unknown latent_dropout_mode: {self.latent_dropout_mode}")
        
        # Decode to structure deformations
        structure_deformations = self.decoder(z, anthro_processed)
        
        # Apply affine transformation if enabled
        affine_params = None
        if self.use_affine and template_vertices is not None:
            scale, translation = self.affine(anthro_processed)
            affine_params = {'scale': scale, 'translation': translation}
            
            # Apply affine to template and adjust deformations
            for struct_name in structure_deformations.keys():
                if struct_name in template_vertices:
                    # Apply affine to template
                    affine_template = self.affine.apply_transform(
                        template_vertices[struct_name],
                        scale,
                        translation
                    )
                    # Deformations are relative to affine-transformed template
                            
        
        # Return all outputs
        outputs = {
            'structure_deformations': structure_deformations,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
        }
        
        if affine_params is not None:
            outputs['affine_params'] = affine_params
        
        return outputs
    
    def sample(self,
               anthro_features: torch.Tensor,
               n_samples: int = 1,
               template_graph: Optional[Any] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Sample new meshes given anthropometric features.
        
        Args:
            anthro_features: Anthropometric features [1, anthro_dim] or [batch_size, anthro_dim]
            n_samples: Number of samples to generate
            template_graph: Template graph for encoding (optional)
            
        Returns:
            List of dictionaries containing structure deformations
        """
        self.eval()
        
        with torch.no_grad():
            # Ensure batch dimension
            if anthro_features.dim() == 1:
                anthro_features = anthro_features.unsqueeze(0)
            
            batch_size = anthro_features.shape[0]
            
            # Process anthropometric features
            anthro_processed = self.anthro_processor(anthro_features)
            
            # Get conditional prior
            prior_params = self.prior_net(anthro_processed)
            prior_mu, prior_logvar = prior_params.split(self.latent_dim, dim=-1)
            
            samples = []
            for _ in range(n_samples):
                # Sample from prior
                std = torch.exp(0.5 * prior_logvar)
                eps = torch.randn_like(std)
                z = prior_mu + eps * std
                
                # Decode to deformations
                structure_deformations = self.decoder(z, anthro_processed)
                
                # Apply affine if enabled
                if self.use_affine:
                    scale, translation = self.affine(anthro_processed)
                    affine_params = {'scale': scale, 'translation': translation}
                    
                    # Store affine parameters with deformations
                    structure_deformations['affine_params'] = affine_params
                
                samples.append(structure_deformations)
        
        return samples
    
    def compute_kl_divergence(self,
                             mu: torch.Tensor,
                             logvar: torch.Tensor,
                             prior_mu: torch.Tensor,
                             prior_logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        KL(q(z|x,c) || p(z|c))
        
        Args:
            mu: Mean of posterior [batch_size, latent_dim]
            logvar: Log variance of posterior [batch_size, latent_dim]
            prior_mu: Mean of prior [batch_size, latent_dim]
            prior_logvar: Log variance of prior [batch_size, latent_dim]
            
        Returns:
            KL divergence [batch_size]
        """
        # KL divergence between two Gaussians
        kl = 0.5 * (
            prior_logvar - logvar + 
            torch.exp(logvar) / torch.exp(prior_logvar) +
            ((mu - prior_mu) ** 2) / torch.exp(prior_logvar) - 1
        )
        
        # Sum over latent dimensions, mean over batch
        kl = kl.sum(dim=-1)
        
        return kl
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute additional regularization losses.
        
        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        
        # L2 regularization on decoder weights
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + 0.0001 * torch.norm(param, p=2)
        
        return reg_loss
    
    def denormalize_predictions(self,
                              structure_deformations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Denormalize predicted deformations using stored normalizer.
        
        Args:
            structure_deformations: Normalized deformations
            
        Returns:
            Denormalized deformations in mm
        """
        if self.normalizer is None:
            return structure_deformations
        
        denorm_dict = {}
        structure_scalers = self.normalizer.get('structure_deformation_scalers', {})
        
        for struct_name, deform_tensor in structure_deformations.items():
            if struct_name in structure_scalers:
                scaler = structure_scalers[struct_name]
                
                # Handle batch dimension
                batch_size = deform_tensor.shape[0]
                denorm_list = []
                
                for b in range(batch_size):
                    deform_np = deform_tensor[b].cpu().numpy()
                    original_shape = deform_np.shape
                    deform_flat = deform_np.reshape(1, -1)
                    denorm_flat = scaler.inverse_transform(deform_flat)
                    denorm_reshaped = denorm_flat.reshape(original_shape)
                    denorm_list.append(torch.tensor(denorm_reshaped))
                
                denorm_dict[struct_name] = torch.stack(denorm_list)
            else:
                denorm_dict[struct_name] = deform_tensor
        
        return denorm_dict
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'decoder': decoder_params,
            'affine': sum(p.numel() for p in self.affine.parameters()) if self.use_affine else 0,
            'prior': sum(p.numel() for p in self.prior_net.parameters())
        }