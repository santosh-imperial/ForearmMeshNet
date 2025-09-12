# forearm_meshnet/models/encoder.py
"""
Variational Graph Encoder for ForearmMeshNet 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINEConv, global_mean_pool
from typing import Optional, Tuple, List


class VariationalGraphEncoder(nn.Module):
    """
    Variational encoder for graph-structured forearm data.
    Matches the pipeline:
      - LayerNorm on inputs and per layer
      - Skip every 2 layers: Linear(dims[i-1] -> dims[i+1])
      - Global mean pool + Linear
      - fc_mu / fc_logvar as small MLPs
      - forward() returns (mu, logvar)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 num_structures: int,
                 dropout_rate: float = 0.3,
                 conv_type: str = "gcn",
                 cond_dim: int = 0):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_structures = num_structures
        # pipeline uses "gcn" | "graphsage" | "gine"; accept "sage" too
        ct = conv_type.lower()
        self._conv_type = "graphsage" if ct in ("sage", "graphsage") else ct
        self.cond_dim = cond_dim

        dims = [input_dim] + hidden_dims

        # input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # conv stack
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_connections: List[Optional[nn.Module]] = []

        for i in range(len(dims) - 1):
            in_c, out_c = dims[i], dims[i + 1]
            if self._conv_type == "gcn":
                conv = GCNConv(in_c, out_c)
            elif self._conv_type == "graphsage":
                conv = SAGEConv(in_c, out_c)
            elif self._conv_type == "gine":
                # edge_dim fixed; we auto-compute edge_attr in forward
                # Use edge_dim=in_c so x[col]-x[row] works without extra projection.
                conv = GINEConv(nn.Linear(in_c, out_c), edge_dim=in_c)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.conv_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(out_c))

            # skip every 2 layers: Linear(dims[i-1] -> dims[i+1])
            if i >= 2 and i % 2 == 0:
                self.skip_connections.append(nn.Linear(dims[i - 1], out_c))
            else:
                self.skip_connections.append(None)

        self.dropout = nn.Dropout(dropout_rate)

        # global pooling head
        self.global_pool_fc = nn.Linear(hidden_dims[-1], hidden_dims[-1])

        last = hidden_dims[-1] + (cond_dim if cond_dim > 0 else 0)

        # fc_mu / fc_logvar are small MLPs with LayerNorm, ReLU, Dropout -> Linear
        self.fc_mu = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.LayerNorm(last // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(last // 2, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.LayerNorm(last // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(last // 2, latent_dim),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, Fin]
            edge_index: [2, E]
            batch: [N] graph Ids (optional; defaults to zeros)
            edge_attr: [E, ?] (optional)
            pos: [N, 3] (optional; used to build edge_attr for GINE if provided)
            cond: [B, cond_dim] anthropometric conditioning (concatenated after pooling)
        Returns:
            (mu, logvar): both [B, latent_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # auto-compute edge_attr for GINE if not provided
        if self._conv_type == "gine" and edge_attr is None:
            row, col = edge_index
            if pos is not None:
                # NOTE: if you use this path, GINEConv was built with edge_dim=in_c,
                # so prefer x-based edge_attr below to avoid dim mismatch.
                # Keep exact pipeline behavior: fall back to x if needed.
                edge_attr = pos[col] - pos[row]
                if edge_attr.size(-1) != x.size(-1):
                    # match pipeline's typical choice: use x-difference
                    edge_attr = x[col] - x[row]
            else:
                edge_attr = x[col] - x[row]

        # normalize inputs
        x = self.input_norm(x)
        row, col = edge_index  # reuse across layers

        layer_outputs: List[torch.Tensor] = []
        for i, (conv, norm, skip) in enumerate(zip(self.conv_layers, self.layer_norms, self.skip_connections)):
            h_in = x
            if self._conv_type == "gine":
                if edge_attr is not None and edge_attr.size(-1) == h_in.size(-1):
                    ea = edge_attr
                else:
                    # prefer x-difference; fall back to pos if it matches
                    if pos is not None and pos.size(-1) == h_in.size(-1):
                        ea = pos[col] - pos[row]
                    else:
                        ea = h_in[col] - h_in[row]
                x = conv(h_in, edge_index, ea)
            else:
                x = conv(x, edge_index)

            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

            layer_outputs.append(x)

            # skip from two layers back
            if skip is not None and i >= 2:
                x = x + skip(layer_outputs[i - 2])

        # global pooling → graph embedding
        x = global_mean_pool(x, batch)
        x = self.global_pool_fc(x)

        # concat conditioning after pooling
        if cond is not None and self.cond_dim > 0:
            x = torch.cat([x, cond], dim=-1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=2)  # exact pipeline clamp

        return mu, logvar

    def encode(self,
               x: torch.Tensor,
               edge_index: torch.Tensor,
               batch: Optional[torch.Tensor] = None,
               edge_attr: Optional[torch.Tensor] = None,
               pos: Optional[torch.Tensor] = None,
               cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic encoding (returns mu)."""
        mu, _ = self.forward(x, edge_index, batch=batch, edge_attr=edge_attr, pos=pos, cond=cond)
        return mu


class ConditionalPrior(nn.Module):
    """
    Conditional prior network p(z|c) for VAE.
    Mirrors your pipeline: small MLP → (mu, logvar) with clamped logvar.
    """
    def __init__(self, cond_dim: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.fc_mu.weight); nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(cond)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=2)
        return mu, logvar
