"""
test_dataset.py — Tests for ForearmDataset and collate_fn.

Additional issue found:
  dataset.py:183: collate_fn checks `getattr(ForearmDataset, "include_combined", False)`
  on the class object (not an instance), so it is always False regardless of
  include_combined=True on the dataset instance — 'combined' is always stripped.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from tests.conftest import ANTHRO_DIM, V_SKIN, V_FCR


def _make_sample(n_verts=V_SKIN, anthro_dim=ANTHRO_DIM,
                 include_combined=False, subject_id="sub_001"):
    """Build a minimal training sample."""
    n_half = n_verts // 2
    row = torch.arange(n_verts)
    col = (row + 1) % n_verts
    deforms = {
        "skin": torch.randn(n_verts, 3),
        "FCR":  torch.randn(n_half, 3),
    }
    if include_combined:
        deforms["combined"] = torch.randn(n_verts * 3 + n_half * 3)
    return {
        "anthropometric_features": torch.randn(anthro_dim),
        "structure_deformations": deforms,
        "structure_info": {
            "skin": {"vertex_range": (0, n_verts), "num_vertices": n_verts},
            "FCR":  {"vertex_range": (n_verts, n_verts + n_half)},
        },
        "unified_template_graph": Data(
            x=torch.randn(n_verts, 7),
            edge_index=torch.stack(
                [torch.cat([row, col]), torch.cat([col, row])], dim=0
            ),
            pos=torch.randn(n_verts, 3),
        ),
        "subject_id": subject_id,
    }


# ── ForearmDataset ────────────────────────────────────────────────────────────

class TestForearmDataset:
    @pytest.fixture
    def dataset(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        return ForearmDataset([_make_sample(subject_id=f"sub_{i}") for i in range(6)])

    def test_length(self, dataset):
        assert len(dataset) == 6

    def test_getitem_returns_dict(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_getitem_required_keys(self, dataset):
        sample = dataset[0]
        assert "anthropometric_features"  in sample
        assert "structure_deformations"   in sample

    def test_anthro_features_float32(self, dataset):
        sample = dataset[0]
        assert sample["anthropometric_features"].dtype == torch.float32

    def test_deformations_float32(self, dataset):
        sample = dataset[0]
        for name, d in sample["structure_deformations"].items():
            assert isinstance(d, torch.Tensor), f"{name} must be a tensor"
            assert d.dtype == torch.float32, f"{name} must be float32"

    def test_anthro_shape(self, dataset):
        sample = dataset[0]
        assert sample["anthropometric_features"].shape == (ANTHRO_DIM,)

    def test_deformation_shapes(self, dataset):
        sample = dataset[0]
        assert sample["structure_deformations"]["skin"].shape == (V_SKIN, 3)
        assert sample["structure_deformations"]["FCR"].shape  == (V_SKIN // 2, 3)

    def test_augmentation_introduces_variation(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        torch.manual_seed(0)
        samples = [_make_sample()]
        ds = ForearmDataset(samples, augment=True)
        # Draw 20 times; augmentation fires with 50% prob so at least some differ
        vals = [ds[0]["anthropometric_features"].clone() for _ in range(20)]
        n_unique = len({v.tolist().__repr__() for v in vals})
        assert n_unique > 1, "Augmented samples should not all be identical"

    def test_no_augmentation_is_deterministic(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        samples = [_make_sample()]
        ds = ForearmDataset(samples, augment=False)
        v0 = ds[0]["anthropometric_features"]
        for _ in range(5):
            assert torch.equal(ds[0]["anthropometric_features"], v0)


# ── collate_fn ────────────────────────────────────────────────────────────────

class TestCollateFn:
    def test_basic_output_keys(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(3)])
        assert "anthropometric_features"  in batch
        assert "structure_deformations"   in batch
        assert "unified_template_graph"   in batch
        assert "batch_size"               in batch

    def test_anthro_batch_shape(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        B = 4
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(B)])
        assert batch["anthropometric_features"].shape == (B, ANTHRO_DIM)

    def test_deformation_batch_dim(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        B = 3
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(B)])
        assert batch["structure_deformations"]["skin"].shape[0] == B
        assert batch["structure_deformations"]["FCR"].shape[0]  == B

    def test_graph_is_batched(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        B = 3
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(B)])
        g = batch["unified_template_graph"]
        assert g is not None
        assert g.batch.max().item() == B - 1

    def test_subject_ids_included(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        samples = [_make_sample(subject_id=f"sub_{i}") for i in range(3)]
        batch = ForearmDataset.collate_fn(samples)
        assert "subject_ids" in batch
        assert len(batch["subject_ids"]) == 3

    def test_missing_structure_filled_with_zeros(self):
        """A sample missing 'FCR' should produce zeros for that structure."""
        from forearm_meshnet.data.dataset import ForearmDataset
        s1 = _make_sample()
        s2 = _make_sample()
        del s2["structure_deformations"]["FCR"]
        batch = ForearmDataset.collate_fn([s1, s2])
        fcr = batch["structure_deformations"]["FCR"]
        # s2's FCR should be all-zero
        assert torch.all(fcr[1] == 0.0), "Missing structure should be zero-padded"

    def test_presence_mask_reflects_missing_structure(self):
        """structure_masks['FCR'][1] should be False when FCR is absent from sample 1."""
        from forearm_meshnet.data.dataset import ForearmDataset
        s1 = _make_sample()
        s2 = _make_sample()
        del s2["structure_deformations"]["FCR"]
        batch = ForearmDataset.collate_fn([s1, s2])
        mask = batch["structure_masks"]["FCR"]
        assert mask[0].item() is True
        assert mask[1].item() is False

    def test_combined_excluded_by_default(self):
        """
        Additional issue: collate_fn checks `ForearmDataset.include_combined`
        on the class (always False) rather than the instance flag, so 'combined'
        is always dropped from the batch regardless of include_combined=True.
        This test documents the current (broken) behaviour and will flip to
        pass once the bug is fixed.
        """
        from forearm_meshnet.data.dataset import ForearmDataset
        samples = [_make_sample(include_combined=True) for _ in range(2)]
        # Even with include_combined=True on the dataset, collate_fn ignores it
        batch = ForearmDataset.collate_fn(samples)
        # After fix: 'combined' should be present when include_combined=True
        assert "combined" in batch["structure_deformations"], (
            "collate_fn should respect include_combined=True (currently broken)"
        )

    def test_batch_size_field_correct(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        B = 5
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(B)])
        assert batch["batch_size"] == B

    def test_structure_info_propagated(self):
        from forearm_meshnet.data.dataset import ForearmDataset
        batch = ForearmDataset.collate_fn([_make_sample() for _ in range(2)])
        assert isinstance(batch["structure_info"], dict)
