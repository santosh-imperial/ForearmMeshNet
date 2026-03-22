"""
test_encoder.py — Tests for VariationalGraphEncoder and ConditionalPrior.
"""

import pytest
import torch
from tests.conftest import BATCH_SIZE, NODE_FEAT_DIM, ANTHRO_DIM, LATENT_DIM


@pytest.fixture
def gcn_encoder():
    from forearm_meshnet.models.encoder import VariationalGraphEncoder
    return VariationalGraphEncoder(
        input_dim=NODE_FEAT_DIM,
        hidden_dims=[32, 64],
        latent_dim=LATENT_DIM,
        num_structures=2,
        dropout_rate=0.0,
        conv_type="gcn",
        cond_dim=ANTHRO_DIM,
    )


class TestVariationalGraphEncoderInstantiation:
    def test_gcn(self):
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        enc = VariationalGraphEncoder(
            input_dim=NODE_FEAT_DIM, hidden_dims=[32, 64],
            latent_dim=LATENT_DIM, num_structures=2,
            dropout_rate=0.0, conv_type="gcn", cond_dim=0,
        )
        assert enc is not None

    def test_graphsage(self):
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        enc = VariationalGraphEncoder(
            input_dim=NODE_FEAT_DIM, hidden_dims=[32, 64],
            latent_dim=LATENT_DIM, num_structures=2,
            dropout_rate=0.0, conv_type="graphsage", cond_dim=0,
        )
        assert enc is not None

    def test_invalid_conv_type_raises(self):
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        with pytest.raises(ValueError, match="Unknown conv_type"):
            VariationalGraphEncoder(
                input_dim=NODE_FEAT_DIM, hidden_dims=[32],
                latent_dim=LATENT_DIM, num_structures=2,
                dropout_rate=0.0, conv_type="bad_type", cond_dim=0,
            )


class TestVariationalGraphEncoderForward:
    def test_output_shapes(self, gcn_encoder, toy_batch_graph, anthro_features):
        gcn_encoder.eval()
        mu, logvar = gcn_encoder(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        assert mu.shape    == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_logvar_clamped_to_range(self, gcn_encoder, toy_batch_graph, anthro_features):
        gcn_encoder.eval()
        _, logvar = gcn_encoder(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        assert logvar.min().item() >= -10.0 - 1e-5
        assert logvar.max().item() <=   2.0 + 1e-5

    def test_no_nan_in_output(self, gcn_encoder, toy_batch_graph, anthro_features):
        gcn_encoder.eval()
        mu, logvar = gcn_encoder(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        assert not torch.isnan(mu).any(),     "NaN in mu"
        assert not torch.isnan(logvar).any(), "NaN in logvar"

    def test_no_batch_tensor(self, toy_graph):
        """batch=None should default to a single-graph batch of zeros."""
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        enc = VariationalGraphEncoder(
            input_dim=NODE_FEAT_DIM, hidden_dims=[32, 64],
            latent_dim=LATENT_DIM, num_structures=2,
            dropout_rate=0.0, conv_type="gcn", cond_dim=0,
        )
        enc.eval()
        mu, logvar = enc(x=toy_graph.x, edge_index=toy_graph.edge_index)
        assert mu.shape == (1, LATENT_DIM)

    def test_without_conditioning(self, toy_batch_graph):
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        enc = VariationalGraphEncoder(
            input_dim=NODE_FEAT_DIM, hidden_dims=[32, 64],
            latent_dim=LATENT_DIM, num_structures=2,
            dropout_rate=0.0, conv_type="gcn", cond_dim=0,
        )
        enc.eval()
        mu, logvar = enc(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
        )
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)

    def test_graphsage_forward(self, toy_batch_graph, anthro_features):
        from forearm_meshnet.models.encoder import VariationalGraphEncoder
        enc = VariationalGraphEncoder(
            input_dim=NODE_FEAT_DIM, hidden_dims=[32, 64],
            latent_dim=LATENT_DIM, num_structures=2,
            dropout_rate=0.0, conv_type="graphsage", cond_dim=ANTHRO_DIM,
        )
        enc.eval()
        mu, _ = enc(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)

    def test_encode_returns_only_mu(self, gcn_encoder, toy_batch_graph, anthro_features):
        gcn_encoder.eval()
        mu = gcn_encoder.encode(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)

    def test_deterministic_in_eval_mode(self, gcn_encoder, toy_batch_graph, anthro_features):
        """Two eval-mode forward passes with the same input should give the same output."""
        gcn_encoder.eval()
        kwargs = dict(
            x=toy_batch_graph.x,
            edge_index=toy_batch_graph.edge_index,
            batch=toy_batch_graph.batch,
            cond=anthro_features,
        )
        mu1, _ = gcn_encoder(**kwargs)
        mu2, _ = gcn_encoder(**kwargs)
        assert torch.allclose(mu1, mu2)


class TestConditionalPrior:
    def test_forward_shapes(self):
        from forearm_meshnet.models.encoder import ConditionalPrior
        prior = ConditionalPrior(cond_dim=ANTHRO_DIM, latent_dim=LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        mu, logvar = prior(c)
        assert mu.shape    == (BATCH_SIZE, LATENT_DIM)
        assert logvar.shape == (BATCH_SIZE, LATENT_DIM)

    def test_logvar_clamped(self):
        from forearm_meshnet.models.encoder import ConditionalPrior
        prior = ConditionalPrior(cond_dim=ANTHRO_DIM, latent_dim=LATENT_DIM)
        c = torch.randn(BATCH_SIZE, ANTHRO_DIM)
        _, logvar = prior(c)
        assert logvar.min().item() >= -10.0 - 1e-5
        assert logvar.max().item() <=   2.0 + 1e-5
