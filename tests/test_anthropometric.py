"""
test_anthropometric.py — Tests for AnthropometricExtractor.
"""

import pytest
import torch
import numpy as np
import trimesh


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_extractor():
    from forearm_meshnet.features.anthropometric import AnthropometricExtractor
    return AnthropometricExtractor(debug_plots=False)


def _make_cylinder_mesh(radius=20.0, height=250.0, sections=32):
    """Return a watertight cylinder trimesh with vertices at intermediate heights.

    trimesh.creation.cylinder only places vertices at the top and bottom caps.
    Two rounds of subdivision add mid-height vertices so circumference extraction
    (which cuts at 25 / 50 / 75 % of height) can find enough points.
    """
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    return m.subdivide().subdivide()  # 2× subdivision → vertices at mid-heights


def _minimal_measurements():
    return {
        "forearm_length": 250.0,
        "wrist_circumference": 170.0,
        "mid_forearm_circumference": 210.0,
        "proximal_circumference": 240.0,
        "taper_ratio": 1.41,
        "length_width_ratio": 6.25,
        "width_depth_ratio": 1.0,
        "wrist_cross_sectional_area": 2300.0,
        "mid_cross_sectional_area": 3500.0,
        "proximal_cross_sectional_area": 4580.0,
        "max_dimension": 250.0,
        "min_dimension": 38.0,
        "bounding_box_volume": 362000.0,
        "surface_area": 17593.0,
        "volume": 314159.0,
    }


# ── Feature dimension / names ──────────────────────────────────────────────────

class TestFeatureDimAndNames:
    def test_feature_dim_with_categorical(self):
        ex = _make_extractor()
        # 15 mesh features + 4 subject features + 6 categorical (3 gender + 3 hand)
        assert ex.get_feature_dim(include_categorical=True) == 25

    def test_feature_dim_without_categorical(self):
        ex = _make_extractor()
        assert ex.get_feature_dim(include_categorical=False) == 19

    def test_feature_names_length_with_categorical(self):
        ex = _make_extractor()
        names = ex.get_feature_names(include_categorical=True)
        assert len(names) == 25

    def test_feature_names_length_without_categorical(self):
        ex = _make_extractor()
        names = ex.get_feature_names(include_categorical=False)
        assert len(names) == 19

    def test_feature_names_are_strings(self):
        ex = _make_extractor()
        for name in ex.get_feature_names():
            assert isinstance(name, str)

    def test_forearm_length_in_names(self):
        ex = _make_extractor()
        assert "forearm_length" in ex.get_feature_names()


# ── to_feature_vector ──────────────────────────────────────────────────────────

class TestToFeatureVector:
    def test_output_is_tensor(self):
        ex = _make_extractor()
        v = ex.to_feature_vector(_minimal_measurements())
        assert isinstance(v, torch.Tensor)

    def test_output_dtype_float32(self):
        ex = _make_extractor()
        v = ex.to_feature_vector(_minimal_measurements())
        assert v.dtype == torch.float32

    def test_output_dim_with_categorical(self):
        ex = _make_extractor()
        v = ex.to_feature_vector(_minimal_measurements(), include_categorical=True)
        assert v.shape == (25,)

    def test_output_dim_without_categorical(self):
        ex = _make_extractor()
        v = ex.to_feature_vector(_minimal_measurements(), include_categorical=False)
        assert v.shape == (19,)

    def test_missing_key_defaults_to_zero(self):
        ex = _make_extractor()
        v = ex.to_feature_vector({})  # all keys missing
        assert v.shape[0] == 25
        # numerical part should be zero (categorical gets unknown = [0,0,1])
        assert v[:19].sum().item() == 0.0

    def test_gender_male_encoding(self):
        ex = _make_extractor()
        m = dict(_minimal_measurements(), subject_gender="M")
        v = ex.to_feature_vector(m)
        gender_bits = v[19:22].tolist()
        assert gender_bits[0] == 1.0  # is_male
        assert gender_bits[1] == 0.0
        assert gender_bits[2] == 0.0

    def test_gender_female_encoding(self):
        ex = _make_extractor()
        m = dict(_minimal_measurements(), subject_gender="F")
        v = ex.to_feature_vector(m)
        gender_bits = v[19:22].tolist()
        assert gender_bits[0] == 0.0
        assert gender_bits[1] == 1.0

    def test_gender_unknown_encoding(self):
        ex = _make_extractor()
        m = dict(_minimal_measurements(), subject_gender=None)
        v = ex.to_feature_vector(m)
        gender_bits = v[19:22].tolist()
        assert gender_bits[2] == 1.0  # gender_unknown

    def test_hand_right_encoding(self):
        ex = _make_extractor()
        m = dict(_minimal_measurements(), dominant_hand="R")
        v = ex.to_feature_vector(m)
        hand_bits = v[22:25].tolist()
        assert hand_bits[1] == 1.0  # is_right_handed

    def test_hand_left_encoding(self):
        ex = _make_extractor()
        m = dict(_minimal_measurements(), dominant_hand="L")
        v = ex.to_feature_vector(m)
        hand_bits = v[22:25].tolist()
        assert hand_bits[0] == 1.0


# ── from_feature_vector ────────────────────────────────────────────────────────

class TestFromFeatureVector:
    def test_roundtrip_numerical_values(self):
        ex = _make_extractor()
        orig = _minimal_measurements()
        vec = ex.to_feature_vector(orig, include_categorical=False)
        back = ex.from_feature_vector(vec, include_categorical=False)
        for key in ex.feature_order:
            assert abs(back[key] - orig[key]) < 1e-3, f"Mismatch for '{key}'"

    def test_roundtrip_gender_male(self):
        ex = _make_extractor()
        orig = dict(_minimal_measurements(), subject_gender="M")
        vec = ex.to_feature_vector(orig)
        back = ex.from_feature_vector(vec)
        assert back.get("subject_gender") == "M"

    def test_roundtrip_gender_female(self):
        ex = _make_extractor()
        orig = dict(_minimal_measurements(), subject_gender="F")
        vec = ex.to_feature_vector(orig)
        back = ex.from_feature_vector(vec)
        assert back.get("subject_gender") == "F"

    def test_roundtrip_hand_right(self):
        ex = _make_extractor()
        orig = dict(_minimal_measurements(), dominant_hand="R")
        vec = ex.to_feature_vector(orig)
        back = ex.from_feature_vector(vec)
        assert back.get("dominant_hand") == "R"

    def test_accepts_numpy_array(self):
        ex = _make_extractor()
        arr = np.zeros(25, dtype=np.float32)
        result = ex.from_feature_vector(arr)
        assert isinstance(result, dict)

    def test_accepts_list(self):
        ex = _make_extractor()
        result = ex.from_feature_vector([0.0] * 25)
        assert isinstance(result, dict)


# ── add_subject_data ──────────────────────────────────────────────────────────

class TestAddSubjectData:
    def test_bmi_calculated(self):
        ex = _make_extractor()
        data = {}
        subject = {"height": 175.0, "weight": 70.0, "age": 30}
        result = ex.add_subject_data(data, subject)
        expected_bmi = 70.0 / (1.75 ** 2)
        assert abs(result["bmi"] - expected_bmi) < 1e-4

    def test_relative_length_calculated(self):
        ex = _make_extractor()
        data = {"forearm_length": 250.0}  # 25 cm
        subject = {"height": 175.0, "weight": 70.0}
        result = ex.add_subject_data(data, subject)
        expected_rel = (250.0 / 10) / 175.0
        assert abs(result["forearm_length_relative"] - expected_rel) < 1e-6

    def test_no_bmi_when_height_missing(self):
        ex = _make_extractor()
        data = {}
        subject = {"weight": 70.0}  # no height
        result = ex.add_subject_data(data, subject)
        assert "bmi" not in result

    def test_gender_stored(self):
        ex = _make_extractor()
        data = {}
        result = ex.add_subject_data(data, {"gender": "M"})
        assert result["subject_gender"] == "M"

    def test_missing_subject_fields_default_none(self):
        ex = _make_extractor()
        data = {}
        result = ex.add_subject_data(data, {})
        assert result["subject_height"] is None
        assert result["subject_weight"] is None


# ── extract_from_mesh ─────────────────────────────────────────────────────────

class TestExtractFromMesh:
    def test_returns_dict(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh()
        result = ex.extract_from_mesh(mesh)
        assert isinstance(result, dict)

    def test_forearm_length_positive(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh(height=250.0)
        result = ex.extract_from_mesh(mesh)
        assert result["forearm_length"] > 0.0

    def test_forearm_length_roughly_correct(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh(radius=20.0, height=250.0)
        result = ex.extract_from_mesh(mesh)
        # The tallest dimension should be ~250 mm
        assert abs(result["forearm_length"] - 250.0) < 2.0

    def test_circumferences_positive(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh()
        result = ex.extract_from_mesh(mesh)
        for key in ("wrist_circumference", "mid_forearm_circumference", "proximal_circumference"):
            assert result[key] > 0.0

    def test_watertight_mesh_has_volume(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh()
        assert mesh.is_watertight
        result = ex.extract_from_mesh(mesh)
        assert "volume" in result
        assert result["volume"] > 0.0

    def test_bounding_box_volume_positive(self):
        ex = _make_extractor()
        mesh = _make_cylinder_mesh()
        result = ex.extract_from_mesh(mesh)
        assert result["bounding_box_volume"] > 0.0
