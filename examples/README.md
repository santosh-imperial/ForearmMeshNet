# Examples

These scripts cover the full ForearmMeshNet workflow, from a first prediction to a complete training run.

## Where to start

| Script | What it does | Needs data? |
|--------|-------------|:-----------:|
| **[quickstart.py](quickstart.py)** | Predict a forearm mesh from body measurements — the shortest path to a result | Trained model only |
| **[generate_muscle_mesh.py](generate_muscle_mesh.py)** | Minimal preprocessing: load DICOM data and generate individual muscle meshes | Yes |
| **[groundtruth_pipeline_demo.py](groundtruth_pipeline_demo.py)** | Full preprocessing for one or more subjects: skin mask → skin mesh → muscle meshes | Yes |
| **[pipeline_demo1.py](pipeline_demo1.py)** | Process a dataset of subjects, extract features, and build skin/muscle/unified templates | Yes |
| **[pipeline_demo2.py](pipeline_demo2.py)** | Complete end-to-end pipeline: preprocessing → templates → training → inference | Partial* |

\* Running `python pipeline_demo2.py` directly executes a quick sanity check — it instantiates
the model and runs feature extraction without needing any data files. The full pipeline is in
`main()` at the bottom of the file (uncomment to run).

## Recommended reading order

1. **New to the library?**
   Run `python pipeline_demo2.py` first. It requires no data and immediately shows model
   instantiation and feature extraction working.

2. **Have a trained model and just want meshes?**
   Go straight to `quickstart.py`. Update the three path constants at the top and run it.

3. **Processing your own MRI data?**
   Start with `groundtruth_pipeline_demo.py` (single-subject focus), then move to
   `pipeline_demo1.py` for batch processing and template generation.

4. **Training from scratch?**
   Follow `pipeline_demo2.py` end-to-end. It covers all five stages in order.

## Expected data layout

```
MRI_Data/
  Subject_01/
    mri_files/      ← DICOM series
    roi_files/      ← ROI label files (one per muscle label)
    subject_info.json   ← optional: height, weight, age, gender
  Subject_02/
    ...
```

## Configuration

All examples use `DEFAULT_CONFIG` from `forearm_meshnet.config` as their starting point.
To customise, deep-copy it and override individual keys:

```python
import copy
from forearm_meshnet.config import DEFAULT_CONFIG

config = copy.deepcopy(DEFAULT_CONFIG)
config["training"]["num_epochs"] = 300
config["model"]["latent_dim"] = 512
```

See [`forearm_meshnet/config/default_config.py`](../forearm_meshnet/config/default_config.py)
for all available keys and their defaults.
