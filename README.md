# ForearmMeshNet

**ForearmMeshNet** is a deep learning framework for reconstructing anatomically accurate 3D forearm meshes from anthropometric measurements. The package uses a Variational Autoencoder (VAE) with Graph Convolutional Networks to generate subject-specific forearm models including skin surface and 17 individual muscle structures.

This is an End-to-end pipeline which includes preprocessing, unified template generation, training data preparation, model training (with curriculum + geometric losses), and inference.



##  Key Features

- **Multi-Structure Reconstruction**: Generates skin surface and 17 individual forearm muscles
-  **Anthropometric conditioning**: Uses only simple measurements (circumferences, length) - no MRI needed for inference
-  **VAE Architecture**: Variational autoencoder with graph convolutions for robust mesh generation
-  **Uncertainty Quantification**: Generates multiple samples to assess prediction uncertainty
-  **Anatomically Accurate**: Trained on real MRI data with expert segmentations
-  **Modular Design**: Easy to extend and adapt for other anatomical structures
-  **Comprehensive Metrics**: Built-in evaluation tools for mesh quality assessment


---

## Table of Contents

* [Installation](#installation)
* [Data Layout](#data-layout)
* [Quickstart](#quickstart)
* [End-to-End Example](#end-to-end-example)
* [Training Only](#training-only)
* [Inference Only](#inference-only)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Troubleshooting](#troubleshooting)

---

## Installation

### 1) Core dependencies

* Python ≥ 3.9
* PyTorch (CUDA recommended)
* NumPy, SciPy, scikit-learn
* trimesh
* matplotlib, tqdm, pandas
* **Optional:** TensorBoard (for logging)

```bash
# Example (adjust for your CUDA/PyTorch build):
pip install torch torchvision torchaudio

pip install numpy scipy scikit-learn trimesh matplotlib tqdm pandas
pip install tensorboard  # optional
```

### 2) Graph dependencies

If your unified template uses a torch geometric graph (it does, via `graph.x`):

```bash
pip install torch-geometric
# and the appropriate torch-scatter/torch-sparse wheels for your CUDA/PyTorch
# (see PyG installation instructions for your environment)
```

### 3) Medical imaging (if you run preprocessing)

```bash
pip install pydicom nibabel
```

> If you only train/infer from already prepared training samples + templates, you don’t need the DICOM stack.

---

## Data Layout

Expected dataset structure per subject:

```
DATA_ROOT/
  Subject_001/
    mri_files/            # DICOM or NIfTI (as supported by your preprocessing)
    roi_files/            # ROI segmentations (muscles/skin)
    subject_info.json     # optional demographics (height/weight/age/handedness)
  Subject_002/
    ...
```

The pipeline will write outputs under your chosen `OUTPUT_ROOT/`, including:

```
OUTPUT_ROOT/
  prepared_data/
    prepared_data.pkl
    Subject_XXX_skin.ply
    Subject_XXX_muscles/...
  templates/
    skin_template.*              # saved by SkinTemplateGenerator
    muscle_templates.pkl         # saved by MuscleTemplateGenerator
    unified_template.pkl         # base file used by training & inference
    unified_template.*           # mesh/graph sidecar files if produced
  training_data/
    raw/                         # raw exports (optional)
    normalizers.pkl
    train_samples.pkl
    val_samples.pkl
  model/
    checkpoints/
      best_model.pt
      checkpoint_epoch_XX.pt
      final_model.pt
    logs/                        # if TensorBoard enabled
  inference/
    test_prediction/
      sample_0/*.ply
      metadata.json
```

---

## Quickstart

The **complete pipeline** class orchestrates every step:

```python
from forearm_meshnet_complete import ForearmMeshNetComplete  # or your script path

pipeline = ForearmMeshNetComplete()  # uses sensible defaults
pipeline.run_complete_pipeline(
    data_root="/path/to/MRI_Data",
    output_root="./forearm_meshnet_output"
)
```

This will:

1. Prepare data → 2) Build templates → 3) Create & normalize training data
2. Train model → 5) Run inference on a test subject and export meshes

---

## End-to-End Example

We provide a ready-to-run example script (your current version):

* **`ForearmMeshNet - Complete End-to-End Example`**
  It implements:

  * data preparation
  * template generation
  * training data creation + normalization + split
  * training (with geometric losses, curriculum, KL warmup, volume loss)
  * inference (batch/interactive) and mesh export

> Make sure `templates/unified_template.pkl`, `training_data/train_samples.pkl`, `training_data/val_samples.pkl`, and `training_data/normalizers.pkl` are produced before training/inference.

---

## Training Only

If you already have prepared training data:

```python
import pickle
from pathlib import Path
from forearm_meshnet.data import ForearmDataset
from forearm_meshnet.models import ForearmMeshNet
from forearm_meshnet.training import Trainer

data_path = Path("./forearm_meshnet_output/training_data")
model_path = Path("./forearm_meshnet_output/model")

with open(data_path / "train_samples.pkl", "rb") as f:
    train_samples = pickle.load(f)
with open(data_path / "val_samples.pkl", "rb") as f:
    val_samples = pickle.load(f)

# Build datasets
train_dataset = ForearmDataset(train_samples, augment=True)
val_dataset   = ForearmDataset(val_samples, augment=False)

# Derive model config from a sample
sample = train_samples[0]
model_cfg = {
    "node_feature_dim": sample["unified_template_graph"].x.shape[1],
    "anthro_feature_dim": len(sample["anthropometric_features"]),
    "latent_dim": 256,
    "encoder_hidden_dims": [128, 256, 512],
    "decoder_hidden_dims": [512, 256, 128],
    "num_structures": sum(k != "combined" for k in sample["structure_deformations"]),
    "structure_vertex_counts": {k: len(v) for k, v in sample["structure_deformations"].items() if k != "combined"},
    "dropout_rate": 0.1,
    "conv_type": "gcn",
    "use_affine": True,
}

model = ForearmMeshNet(model_cfg)

# Load normalizers and set into the model so losses can denormalize to mm
with open(data_path / "normalizers.pkl", "rb") as f:
    normalizers = pickle.load(f)
if hasattr(normalizers, "__dict__"):
    normalizers = vars(normalizers)
model.set_normalizer(normalizers)

train_cfg = {
    "batch_size": 8,
    "num_epochs": 200,
    "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-2},
    "scheduler": {"type": "CosineAnnealingLR", "T_max": 200, "eta_min": 1e-6},
    "early_stopping": {"enabled": True, "patience": 50, "min_epochs": 100},
    "checkpoint_freq": 10,
    "eval_freq": 5,
    # IMPORTANT: enable geometric losses by pointing to the unified template file
    "unified_template_pickle": str((model_path.parent / "templates" / "unified_template.pkl")),
}

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=train_cfg,
    output_dir=str(model_path)
)
trainer.train(num_epochs=train_cfg["num_epochs"])
```

### Notes on geometric losses

`Trainer` will load per-structure **verts/edges/faces/laplacian** from
`training['unified_template_pickle']` using a local loader.
This enables Chamfer/edge/normal/Laplacian/volume terms in `CombinedLoss`.

---

## Inference Only

If you already have a trained checkpoint and normalizers:

```python
from forearm_meshnet.inference import Predictor

predictor = Predictor(
    model_checkpoint_path="./forearm_meshnet_output/model/checkpoints/best_model.pt",
    template_path="./forearm_meshnet_output/templates/unified_template",
    normalizer_path="./forearm_meshnet_output/training_data/normalizers.pkl"
)

measurements = {
    "forearm_length": 260.0,
    "wrist_circumference": 170.0,
    "mid_forearm_circumference": 210.0,
    "proximal_circumference": 240.0,
    "subject_height": 175.0,
    "subject_weight": 70.0,
    "subject_age": 30,
    "subject_gender": "M",
    "dominant_hand": "R",
}

result = predictor.predict(measurements, n_samples=3)
predictor.save_prediction(result, "./forearm_meshnet_output/inference/test_prediction")
```

---

## Configuration

Key knobs you may want to tweak:

* **Preprocessing**

  * `skin_mask.end_slice_fraction`, ghosting fixes
  * `skin_mesh.target_faces`, `smooth_iterations`
  * `muscle_mesh.target_vertices`, `min_muscle_volume`

* **Template**

  * `skin_vertices`, `muscle_vertices`
  * `min_muscle_availability`

* **Model**

  * `latent_dim`, encoder/decoder sizes, `conv_type`
  * `use_affine`, `use_template_augmentation`

* **Training**

  * `batch_size`, `num_epochs`, optimizer/scheduler
  * `unified_template_pickle` → **required** to enable geometric losses
  * early stopping, checkpoint & eval frequency

* **Losses** (`CombinedLoss`)

  * `lambda_weights`: `reconstruction`, `chamfer`, `normal`, `edge`, `laplacian`, `kl`, `volume`
  * Per-structure weights via `structure_weights`
  * Automatic denormalization to **mm** if your model has `normalizer.structure_deformation_scalers` set.

---

## Project Structure

```
forearm_meshnet/
  preprocessing/           # MRI to skin/muscle meshes
  template/                # skin/muscle templates + unified template builder
  features/                # anthropometric & graph feature extractors
  data/                    # TrainingDataPreparation, DataNormalizer, datasets
  models/                  # ForearmMeshNet, losses (CombinedLoss, etc.)
  training/
    trainer.py             # Trainer (+ local unified template loader)
    curriculum.py          # CurriculumManager
    metrics.py             # MeshEvaluationMetrics
  inference/
    predictor.py           # Predictor & InferencePipeline
```

---

## Troubleshooting

**Q: Training crashes on logging (`self.writer.add_scalar`)?**
Either enable TensorBoard writer or guard the calls. In `trainer.py` we expect:

```python
from torch.utils.tensorboard import SummaryWriter
self.writer = SummaryWriter(self.log_dir)  # enable
```

If you prefer no TB, wrap `add_scalar` calls in `if hasattr(self, "writer"):`.

**Q: Geometric losses are no-ops / zeros?**
Ensure `training['unified_template_pickle']` points to `.../templates/unified_template.pkl`.
The trainer’s local loader populates verts/edges/faces/laplacian used by the losses.

**Q: Shape mismatch inside losses or inference?**
Confirm your `structure_info[struct]['vertex_range']` and (if present) `face_range`
align with the unified template. Deformation tensors must match the per-structure vertex counts.

**Q: Normalizer errors (missing keys)?**
`normalizers.pkl` should contain:

* `anthropometric_scaler`
* `graph_feature_scaler` (optional)
* `structure_deformation_scalers` (dict per structure)
  If you serialized a custom class, convert to dict: `vars(obj)` before use.

**Q: PyG errors about wheels (torch-scatter/torch-sparse)?**
Install the correct CUDA/PyTorch-matching wheels as per PyG’s installation guide.

**Q: Slow Chamfer / memory spikes?**
Chamfer uses `torch.cdist`. You can subsample vertices for that term or reduce `target_faces/skin_vertices` at template gen time for lighter training runs.

---



## Roadmap

- [x] Initial release with core functionality
- [] Pre-trained models for forearm reconstruction
- [ ] Web-based demo application
- [ ] Support for additional anatomical structures
- [ ] Real-time visualization tools
- [ ] Integration with clinical software

## Related Projects

- [MeshDeformNet](https://github.com/fkong7/MeshDeformNet)
- [MeshHeart](https://github.com/MengyunQ/MeshHeart) 
- [LinFlo-Net](https://github.com/ArjunNarayanan/LinFlo-Net)



