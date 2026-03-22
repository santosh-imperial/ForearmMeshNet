"""
quickstart.py — Predict a personalised forearm mesh from body measurements.

This is the shortest path from measurements to meshes.

Prerequisites
-------------
You need three files produced by running the full pipeline (pipeline_demo2.py):

  output/model/checkpoints/best_model.pt
  output/templates/unified_template        (saves as .pkl + .ply)
  output/training_data/normalizers.pkl

Update the three path constants below to match your output directory.
"""

from forearm_meshnet.inference import Predictor

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH      = "output/model/checkpoints/best_model.pt"
TEMPLATE_PATH   = "output/templates/unified_template"   # no file extension
NORMALIZER_PATH = "output/training_data/normalizers.pkl"

# ── 1. Load the trained model ─────────────────────────────────────────────────
predictor = Predictor(
    model_checkpoint_path=MODEL_PATH,
    template_path=TEMPLATE_PATH,
    normalizer_path=NORMALIZER_PATH,
)

# ── 2. Provide measurements (all distances in mm, mass in kg, height in cm) ───
measurements = {
    "forearm_length":            260.0,   # wrist crease to elbow crease
    "wrist_circumference":       170.0,
    "mid_forearm_circumference": 210.0,
    "proximal_circumference":    240.0,   # just below the elbow
    # Optional — improves accuracy but safe to omit; defaults are used
    "subject_height": 175.0,   # cm
    "subject_weight":  70.0,   # kg
    "subject_age":     30,
    "subject_gender":  "M",    # "M" or "F"
    "dominant_hand":   "R",    # "L" or "R"
}

# ── 3. Generate mesh ──────────────────────────────────────────────────────────
result = predictor.predict(measurements)

# ── 4. Save to disk ───────────────────────────────────────────────────────────
predictor.save_prediction(result, output_path="my_prediction")

# Output layout:
#   my_prediction/sample_0/unified.ply      ← complete forearm surface
#   my_prediction/sample_0/skin.ply         ← skin mesh only
#   my_prediction/sample_0/<muscle>.ply     ← one file per predicted muscle
#   my_prediction/metadata.json             ← measurements + structure info

print("Meshes saved to my_prediction/")

# ── (Optional) Work with meshes directly in Python ───────────────────────────
meshes = result["predictions"][0]["meshes"]   # dict of trimesh.Trimesh objects

skin_mesh    = meshes["skin"]
unified_mesh = meshes["unified"]

print(f"Skin mesh:    {len(skin_mesh.vertices):,} vertices, "
      f"{len(skin_mesh.faces):,} faces")
print(f"Unified mesh: {len(unified_mesh.vertices):,} vertices")

# To generate multiple samples (useful for uncertainty estimation):
#   result = predictor.predict(measurements, n_samples=5)
#   for i, pred in enumerate(result["predictions"]):
#       predictor.save_prediction({"predictions": [pred], ...}, f"sample_{i}/")
