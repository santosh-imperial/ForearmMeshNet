"""
test_imports.py — Verify every module can be imported without errors.

Catches:
  Bug 2 (skin_mesh.py:88): duplicate `class SkinMeshGenerator` inside the
      first class causes an IndentationError, making the entire module
      unimportable.
  Bug 1 (metrics.py:18): nested `MeshEvaluationMetrics` class is an import-
      time structural error that only surfaces at instantiation; the import
      itself succeeds but is tested here for completeness.
"""


def test_import_encoder():
    from forearm_meshnet.models.encoder import VariationalGraphEncoder, ConditionalPrior


def test_import_decoder():
    from forearm_meshnet.models.decoder import (
        StructureAwareDecoder,
        FiLMParams,
        AnthroAffine,
        TemplateAugmentor,
    )


def test_import_losses():
    from forearm_meshnet.models.losses import (
        CombinedLoss,
        ChamferDistance,
        EdgeLengthLoss,
        NormalConsistencyLoss,
        LaplacianSmoothingLoss,
        VolumeLoss,
    )


def test_import_meshnet():
    from forearm_meshnet.models.meshnet import ForearmMeshNet


def test_import_metrics():
    # Bug 1: outer class wraps inner class of the same name
    from forearm_meshnet.training.metrics import MeshEvaluationMetrics


def test_import_curriculum():
    from forearm_meshnet.training.curriculum import CurriculumManager


def test_import_skin_mesh():
    # Bug 2: duplicate class definition → IndentationError before fix
    from forearm_meshnet.preprocessing.skin_mesh import SkinMeshGenerator


def test_import_muscle_mesh():
    from forearm_meshnet.preprocessing.muscle_mesh import MuscleMeshGenerator


def test_import_skin_mask():
    from forearm_meshnet.preprocessing.skin_mask import SkinMaskGenerator


def test_import_mesh_utils():
    from forearm_meshnet.preprocessing.mesh_utils import (
        cap_then_polish,
        remove_spurious_triangles,
        detect_isolated_components,
    )


def test_import_dataset():
    from forearm_meshnet.data.dataset import ForearmDataset, ForearmDataLoader


def test_import_normalizer():
    from forearm_meshnet.data.normalizer import DataNormalizer


def test_import_anthropometric():
    from forearm_meshnet.features.anthropometric import AnthropometricExtractor


def test_import_graph_features():
    from forearm_meshnet.features.graph_features import GraphFeatureExtractor


def test_import_predictor():
    from forearm_meshnet.inference.predictor import Predictor


def test_import_skin_template():
    from forearm_meshnet.template.skin_template import SkinTemplateGenerator


def test_import_muscle_template():
    from forearm_meshnet.template.muscle_template import MuscleTemplateGenerator


def test_import_unified_template():
    from forearm_meshnet.template.unified_template import UnifiedTemplateGenerator


def test_import_mesh_operations():
    from forearm_meshnet.utils.mesh_operations import (
        kabsch_align,
        umeyama_align,
    )


def test_import_io_utils():
    from forearm_meshnet.utils.io_utils import load_mesh, save_mesh
