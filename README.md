# IR-Based Stress Detection

Repository: [github.com/neha191091/IR-based-stress-detection](https://github.com/neha191091/IR-based-stress-detection)

Unsupervised rPPG extraction from near-infrared (NIR) video using [Contrast-Phys+](https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B), with time-domain HRV stress indicators (Baevsky SI, SDNN, RMSSD, pNN50) from the recovered pulse signal. A classical [IR_iHR](https://github.com/natalialmg/IR_iHR) path (optimal SVD + synchrosqueezing) is also incoiorporated without evaluation. 

**The code in this repository has been adapted from the following repositories**
| Component | Source | Use in this project |
|-----------|--------|---------------------|
| Contrast-Phys+ (PhysNet, ST-rPPG head, contrastive loss) | [zhaodongsun/contrast-phys](https://github.com/zhaodongsun/contrast-phys) | Primary neural rPPG path; trained and evaluated on MR-NIRP |
| IR_iHR (optimal SVD, synchrosqueezing iHR) | [natalialmg/IR_iHR](https://github.com/natalialmg/IR_iHR) | Ported as optional classical extractor; **not tested or validated** in this report |

For architecture, evaluation results, and figures, see the **[technical report on GitHub Pages](https://neha191091.github.io/IR-based-stress-detection/REPORT/)** ([source](docs/REPORT.md)).

## Overview

Two signal-extraction paths share the same downstream stress metrics:

```
Neural (default)                    Classical (IR_iHR)
─────────────────                   ──────────────────
NIR video → PhysNet → ST-rPPG head   NIR video → face-region grid → optimal SVD
         → rPPG waveform                      → synchrosqueezing iHR
         ↘                                    ↘
              IBI peak detection → Baevsky SI, SDNN, RMSSD, pNN50
                    ↑
         MR-NIRP pulse oximeter (PPG ground truth, eval / plots only)
```

| Pipeline | Module | Metrics |
|----------|--------|---------|
| Download | `ir_stress.dataset.mr_nirp_download` | — |
| Preprocess | `ir_stress.dataset.mr_nirp_driving` | — |
| Train | `ir_stress.training` | MLflow: loss, IPR |
| Evaluate | `ir_stress.evaluation` | Pearson r, MSE |
| Inference (neural) | `ir_stress.inference` | rPPG, Baevsky SI, HRV |
| Inference (IR_iHR) | `ir_stress.signals.ihr_pipeline` | rPPG, instantaneous HR, Baevsky SI |
| Data exploration | `scripts/data_exploration.py` | Windowed stress plots |
| WESAD visualization | `scripts/visualize_WESAD_stress.py` | Baseline vs stress BVP/ECG plots |

### End-to-end workflow

```
Download MR-NIRP (Google Drive) → download.py → preprocess.py → train.py → evaluate.py
                                      ↘ inference.py extraction_method=neural (H5, or raw PGM + landmarks_csv)
                                      ↘ inference.py extraction_method=ihr (H5 or raw PGM; dlib landmarks optional)
                                      ↘ data_exploration.py (NIR montage + pulse-ox stress plots)
                                      ↘ visualize_WESAD_stress.py (WESAD baseline/stress BVP + ECG plots)
```

Smoke test (`scripts/smoke_test.py`) uses short **synthetic** clips — not MR-NIRP — to verify the install without the full dataset.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd IR-based-stress-detection
uv sync
```

Optional extras:

| Extra | Install | Purpose |
|-------|---------|---------|
| `notebook` | `uv sync --extra notebook` | Matplotlib plots (`data_exploration.py`, `visualize_WESAD_stress.py`, inference comparison PNGs) |
| `docs` | `uv sync --extra docs` | MkDocs site for the technical report (`mkdocs serve`) |
| `ihr` | `uv sync --extra ihr` | dlib landmarks for IR_iHR raw-PGM inference |

For IR_iHR on raw PGM without OpenFace CSVs, also download the dlib shape predictor:

```bash
uv sync --extra ihr
# Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract to data/models/shape_predictor_68_face_landmarks.dat
```

Verify the install:

```bash
uv run scripts/smoke_test.py
```

Uses inline defaults in `scripts/smoke_test.py` (short clips, `num_workers=0`) to keep memory usage low.

All pipeline scripts use [Hydra](https://hydra.cc/) for configuration. Defaults live in each script as dataclasses; override on the command line with `key=value`:

```bash
uv run scripts/train.py epochs=10 'val_subjects=[1]'
uv run scripts/download.py subjects=1 scenes=driving motion_types=still
```

## Dataset

The full **MR-NIRP Car** driving release is on [Google Drive](https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK?usp=sharing) (Subject1–Subject19). You can also request access via the [Rice Computational Imaging Group](https://computationalimaging.rice.edu/mr-nirp-dataset/).

### Selective download

Fetch only the participants and modalities you need. **PulseOX is always included**; NIR and RGB are optional.

By default, `scripts/download.py` downloads selected clips from Google Drive directly into `data/raw/mr-nirp/` and extracts them in place:

```bash
# Defaults: all subjects, NIR only, 940 nm
uv run scripts/download.py

# One participant only (good for a first test)
uv run scripts/download.py subjects=1

# Subjects 1-3, NIR + RGB, both wavelengths, still clips only
uv run scripts/download.py \
  subjects=1-3 download_rgb=true wavelengths=[940,975] motion_types=still

# Driving still clips only at 940 nm
uv run scripts/download.py subjects=1 scenes=driving motion_types=still

# Import from an external local copy instead of Google Drive
uv run scripts/download.py \
  source_dir=/path/to/full/mr-nirp-release subjects=2 download_rgb=false
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subjects` | `all` | `all`, `1,2,3`, or `1-5` |
| `download_nir` | `true` | NIR PGM frames |
| `download_rgb` | `false` | RGB frames |
| `wavelengths` | `[940]` | Clip filter: `[940]`, `[975]`, or `[940,975]` |
| `motion_types` | `all` | `still`, `small_motion`, `large_motion`, or comma-separated |
| `scenes` | `all` | `driving`, `garage`, or comma-separated |
| `out_dir` | `data/raw/mr-nirp` | Dataset output directory |
| `force` | `false` | Re-download clips even if they already exist |
| `source_dir` | — | Import from external local Subject*/ layout |
| `base_url` | — | Optional remote root with `Subject*/clip/*.zip` |

Expected layout after download:

```
data/raw/mr-nirp/
  Subject1/
    subject1_driving_still_940/
      NIR/Frame00001.pgm ...
      PulseOX/pulseOx.mat
    subject1_driving_small_motion_940/
      ...
  Subject2/
    ...
```

v1 defaults to **940 nm NIR** (`in_ch=1`, 30 fps). Use `wavelengths=[975]` or `wavelengths=[940,975]` at download time for other bands.

**Note:** `large_motion` clips exist only at **975 nm** in MR-NIRP Car. Filtering `motion_types=large_motion` with `wavelengths=[940]` yields no clips.

**Note:** Subject 2 and Subject 16 are the same person (day vs night recordings). Do not put both in train and test splits.

### WESAD (wearable baseline / stress)

The [WESAD](https://doi.org/10.1145/3242969.3242985) dataset provides synchronised wrist BVP (Empatica E4, 64 Hz), chest ECG (RespiBAN, 700 Hz), and protocol labels in `data/WESAD/S<subject>/S<subject>.pkl`. Subjects S1 and S12 are missing due to sensor malfunction; 15 subjects remain (S2–S11, S13–S17).

Place the extracted dataset at:

```
data/WESAD/
  S2/S2.pkl
  S3/S3.pkl
  ...
  wesad_readme.pdf
```

Each pickle is a dict with keys `subject`, `signal` (`chest`, `wrist`), and `label` (protocol condition sampled at 700 Hz: 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation).

## Preprocessing

Face crops follow the [Contrast-Phys preprocessing](https://github.com/zhaodongsun/contrast-phys/blob/master/preprocessing.py) layout: detect/crop the face per frame, resize to `face_size`, and write aligned pulse-ox PPG into H5.

**Default: YuNet** (`face_crop_mode=yunet`). Uses the bundled OpenCV YuNet ONNX model (`face_detection_yunet_2023mar.onnx` in the package). No OpenFace or landmark files required.

```bash
# Defaults: raw_dir=data/raw/mr-nirp, face_crop_mode=yunet → data/h5_yunet/
uv run scripts/preprocess.py

uv run scripts/preprocess.py raw_dir=data/raw/mr-nirp face_size=128
```

Output directory is always `data/h5_{face_crop_mode}/` (e.g. `data/h5_yunet`). `preprocess.py` does not accept `h5_dir`; set `h5_dir` on `train.py` / `evaluate.py` only if you need a non-default path.

| `face_crop_mode` | Landmarks | Notes |
|------------------|-----------|-------|
| `yunet` (default) | Not required | On-device face detection per frame |
| `openface` | OpenFace CSV per clip in `landmarks_dir` | Paper-faithful landmark crops |
| `center` | Not required | Fixed center crop (debug / fallback) |

**Optional: OpenFace landmarks** for `face_crop_mode=openface`:

```bash
# Convert PGM sequence to video first, then:
./FeatureExtraction -f <video> -out_dir data/landmarks -2Dfp

uv run scripts/preprocess.py face_crop_mode=openface landmarks_dir=data/landmarks
# → data/h5_openface/
```

Each output file contains:

- `imgs` — `[N, face_size, face_size, 1]` float32 NIR face crops (default `face_size=128`)
- `ppg` — `[N]` pulse-oximeter PPG waveform resampled to video fps

## Training

Training reads **preprocessed H5 clips**. By default `h5_dir` resolves to `data/h5_{face_crop_mode}` (`data/h5_yunet` with default `face_crop_mode=yunet`). You need **at least two H5 files** (different subjects) before `train.py` will run.

### 1. Preprocess first

```bash
uv run scripts/preprocess.py raw_dir=data/raw/mr-nirp
```

### 2. Run training

Leave-one-subject-out (Contrast-Phys+ protocol): hold out one subject for test, train on the rest.

```bash
# Default: hold out subject 1, train on all other preprocessed clips
uv run scripts/train.py

# Hold out subject 11 (quote list args in zsh — see below)
uv run scripts/train.py 'val_subjects=[11]' epochs=30
```

**zsh:** Bracket lists like `[11]` are glob patterns. Quote Hydra overrides:

```bash
uv run scripts/train.py 'val_subjects=[11]' 'wavelengths=[940]'
```

On bash, quoting is optional but still fine.

### 3. GPU training

Training uses CUDA automatically when PyTorch sees a GPU (`device=null`, the default). Mixed precision (AMP) is enabled by default on CUDA (`use_amp=true`).

Verify your environment:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda` is `False` but you have an NVIDIA GPU, reinstall PyTorch with a CUDA wheel (pick the index that matches your driver/CUDA toolkit from [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Example GPU run with a larger batch and DataLoader workers:

```bash
uv run scripts/train.py \
  'val_subjects=[11]' \
  device=cuda \
  num_workers=4 \
  batch_size=2 \
  epochs=30
```

Force CPU (e.g. for debugging): `device=cpu use_amp=false`.

Select a specific GPU: `device=cuda:1`.

### 4. Fast dev run (CPU / low memory)

Training is CPU-heavy and memory-intensive with full paper settings (~128×128, 10 s clips). For quick iteration, lower resolution and shorter clips — no re-preprocess required (`face_size` resizes on load):

```bash
uv run scripts/train.py \
  'val_subjects=[11]' \
  face_size=64 \
  clip_seconds=5 \
  video_duration_sec=20 \
  epochs=5
```

Run long jobs in an **external terminal** (not the IDE integrated terminal) to avoid OOM kills when RAM is tight.

### 5. Paper-faithful settings (Contrast-Phys+ MR-NIRP)

Unsupervised (0% labels), matching the Contrast-Phys+ paper defaults:

```bash
uv run scripts/train.py \
  'val_subjects=[11]' \
  model=physnet \
  in_ch=1 \
  fs=30 \
  clip_seconds=10 \
  spatial_dim=2 \
  epochs=30 \
  lr=1e-5 \
  label_ratio=0.0 \
  batch_size=2 \
  face_size=128 \
  video_duration_sec=60 \
  eval_window_sec=30
```

Weakly-supervised variants from the paper: `label_ratio=0.2`, `0.6`, or `1.0`.

### 6. Monitor and evaluate

Checkpoints (`epoch0.pt`, …), `config.json`, and `split.json` are saved to `checkpoint_dir/` (default `checkpoints/`). Metrics (`loss`, `p_loss`, `n_loss`, `ipr`, …) are logged to `mlflow.db` (SQLite).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000 → experiment "ir-stress-rppg"
```

After training:

```bash
uv run scripts/evaluate.py checkpoint=checkpoints/epoch29.pt
```

`split.json` stores `val_subjects` so the train/test H5 split can be reconstructed at evaluation time.

### Key hyperparameters

Defaults live in `ir_stress.config.TrainConfig` (overridden via Hydra in `scripts/train.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `physnet` | Backbone: `physnet`, `physnet_lite`, or `lejepa` (stub) |
| `in_ch` | `1` | NIR input channels |
| `fs` | `30` | Video frame rate (fps) |
| `clip_seconds` | `5` | Temporal window per batch item (150 frames at 30 fps) |
| `spatial_dim` | `2` | ST-rPPG 2×2 spatial grid (+ mean channel) |
| `epochs` | `30` | Training epochs |
| `lr` | `1e-5` | AdamW learning rate |
| `batch_size` | `2` | **Must be 2** — Contrast-Phys+ contrastive loss needs two clips |
| `face_size` | `128` | Face crop side length; downscale on load if H5 is larger |
| `face_crop_mode` | `yunet` | Must match preprocessing (`yunet`, `openface`, `center`) |
| `label_ratio` | `0.0` | Fraction of clips using GT PPG in loss (0 = unsupervised) |
| `video_duration_sec` | `200` | Effective clip length for steps/epoch (`iters ≈ video_duration_sec / clip_seconds`) |
| `eval_window_sec` | `30` | Evaluation/inference window length |
| `val_subjects` | `[1]` | Held-out subject IDs (leave-one-out) |
| `wavelengths` | `[940]` | Clip bands to include |
| `num_workers` | `0` | DataLoader workers (use 2–4 on GPU) |
| `device` | `null` (auto) | `cuda`, `cuda:N`, `cpu`, or `mps` |
| `use_amp` | `true` | Mixed precision on CUDA (set `false` on CPU) |
| `grad_checkpoint` | `false` | Activation checkpointing in PhysNet / PhysNetLite (saves VRAM) |
| `micro_batch` | `true` | Forward one clip at a time when `batch_size=2` (saves VRAM) |
| `h5_dir` | `data/h5_yunet` | Resolved from `face_crop_mode` when unset |
| `checkpoint_dir` | `checkpoints` | Output directory for weights and run metadata |
| `mlflow_experiment` | `ir-stress-rppg` | MLflow experiment name |

Use `model=physnet_lite` for a smaller backbone when GPU memory is limited.

Optional: re-preprocess at lower resolution for faster I/O and smaller H5 files:

```bash
uv run scripts/preprocess.py face_size=64
uv run scripts/train.py face_size=64 'val_subjects=[11]'
```

Use the same `face_size` at train and eval. Checkpoints trained at `face_size=64` are not interchangeable with `128`.

### Memory notes

- `batch_size` cannot be reduced below 2 without changing the loss.
- Preprocessing writes one frame at a time to H5. Training loads random temporal windows from disk.
- Evaluation and inference read video windows rather than loading full clips.
- On machines without a GPU, expect hours for a full 30-epoch run at paper settings. With a CUDA GPU, training is typically much faster; `use_amp=true` reduces VRAM use.

## Evaluation

```bash
uv run scripts/evaluate.py checkpoint=checkpoints/epoch29.pt
```

Reports **Pearson correlation** and **MSE** on bandpass-filtered, z-score-normalized rPPG vs ground-truth PPG.

## Inference

Set `extraction_method` to choose the signal path. Both methods write stress metrics and can produce a comparison plot against pulse-ox ground truth when an H5 clip or MR-NIRP raw layout is available.

| Method | Config | Checkpoint | Landmarks (raw PGM) |
|--------|--------|------------|---------------------|
| `neural` (default) | `extraction_method=neural` | Required | `landmarks_csv` required by CLI; used only when `face_crop_mode=openface` (YuNet crops ignore it) |
| `ihr` | `extraction_method=ihr` | Not needed | dlib (default) or OpenFace CSV |

Shared inference parameters (`InferenceConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `extraction_method` | `neural` | `neural` or `ihr` |
| `input_h5` | — | Preprocessed H5 clip (mutually exclusive with `input_dir`) |
| `input_dir` | — | Directory of raw NIR PGM frames |
| `face_crop_mode` | `yunet` | `yunet`, `openface`, or `center` (read from checkpoint `config.json` for neural H5) |
| `landmarks_csv` | — | Required for neural raw mode (OpenFace CSV when `face_crop_mode=openface`); optional for IR_iHR |
| `output_dir` | `results/inference` | Output directory |
| `plot` | `true` | Write `{stem}_comparison.png` when ground truth is available |
| `raw_dir` | `data/raw/mr-nirp` | MR-NIRP root used to locate pulse-ox for comparison plots |

Outputs (both methods): `{stem}_results.json`, `{stem}_rppg.npy`, and optionally `{stem}_comparison.png`. IR_iHR also writes `{stem}_ihr_bpm.npy` and `{stem}_ihr_time.npy`.

Set `plot=false` to skip figures. Plots require `uv sync --extra notebook`.

### Neural rPPG (Contrast-Phys+)

From a preprocessed H5 clip:

```bash
uv run scripts/inference.py \
  checkpoint=checkpoints/epoch29.pt \
  input_h5=data/h5_yunet/subject2_some_clip.h5 \
  output_dir=results/inference
```

Model settings (`model`, `face_size`, `face_crop_mode`, …) are read from `config.json` in the checkpoint directory. If the path contains `=` (common for run folder names), quote the override for Hydra and the shell:

```bash
uv run scripts/inference.py \
  'checkpoint="checkpoints/facecrop=center_clipsec=5_epochs=30_lr=1e-05_imgsz=128_backbone=physnet_lite/epoch15.pt"' \
  input_h5=data/h5_yunet/subject1_subject1_driving_still_940.h5 \
  output_dir=results/inference
```

Or directly from raw NIR PGM frames (no H5 required):

```bash
# YuNet face crops (landmarks_csv still required by CLI but ignored for cropping)
uv run scripts/inference.py \
  checkpoint=checkpoints/epoch29.pt \
  input_dir=data/raw/mr-nirp/Subject1/subject1_driving_still_940/NIR \
  landmarks_csv=data/landmarks/subject1_driving_still_940.csv \
  face_crop_mode=yunet \
  output_dir=results/inference

# OpenFace landmark crops (landmarks required)
uv run scripts/inference.py \
  checkpoint=checkpoints/epoch29.pt \
  input_dir=data/raw/mr-nirp/Subject1/subject1_driving_still_940/NIR \
  landmarks_csv=data/landmarks/subject1_driving_still_940.csv \
  face_crop_mode=openface \
  output_dir=results/inference
```

Raw mode crops faces on the fly (one window at a time) using the same `face_crop_mode` logic as preprocessing.

### IR_iHR classical extraction

Classical contactless PPG and instantaneous heart rate from spatial grid signals, following Martinez et al. (ICIP 2019) and the [IR_iHR](https://github.com/natalialmg/IR_iHR) reference implementation:

```
raw PGM / H5 face crops → spatial grid (channels × frames)
  → highpass → optimal SVD shrinkage → quality-ranked eigenvectors
  → best cumulative component → bandpass PPG
  → synchrosqueezing → instantaneous HR (bpm)
  → IBI → Baevsky SI
```

**Minimum length:** IR_iHR needs at least **301 frames** (~10 s at 30 fps) for the synchrosqueezing window.

From a preprocessed H5 clip (uniform grid on 128×128 face crops; no landmarks):

```bash
uv run scripts/inference.py \
  extraction_method=ihr \
  input_h5=data/h5_yunet/subject2_some_clip.h5 \
  output_dir=results/ihr
```

From raw NIR PGM frames with **automatic dlib landmarks** (default for IR_iHR raw mode):

```bash
uv run scripts/inference.py \
  extraction_method=ihr \
  input_dir=data/raw/mr-nirp/Subject1/subject1_driving_still_940/NIR \
  output_dir=results/ihr
```

Or with precomputed OpenFace landmarks instead of dlib:

```bash
uv run scripts/inference.py \
  extraction_method=ihr \
  input_dir=data/raw/mr-nirp/Subject1/subject1_driving_still_940/NIR \
  landmarks_csv=data/landmarks/subject1_driving_still_940.csv \
  output_dir=results/ihr
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_bpm` | `70.0` | Prior heart rate for IR_iHR eigenvector quality ranking |
| `ihr_grid_size` | `null` | Grid cell size in pixels (`8` for H5, `5` for raw PGM when unset) |
| `dlib_predictor` | `data/models/shape_predictor_68_face_landmarks.dat` | dlib 68-point shape model (used when `landmarks_csv` is omitted) |

## Data exploration

Plot NIR frame previews and windowed stress indicators from pulse-ox PPG (requires `uv sync --extra notebook` for matplotlib):

```bash
uv run scripts/data_exploration.py
uv run scripts/data_exploration.py subject_number=4 start_time=9.967 window_sec=100
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subject_number` | `4` | MR-NIRP subject ID |
| `start_time` | `9.97` | Segment start (s from recording start). Video frame at origin: `int(start_time×30)+1` |
| `window_sec` | `100.0` | Seconds of PPG/NIR examined from `start_time` |
| `bin_width_seconds` | `0.05` | Baevsky AMo histogram bin width (50 ms) |
| `raw_root` | `data/raw/mr-nirp` | Raw dataset root |
| `output_dir` | `results` | Plot output directory |

Saves `results/subject{N}_exploration.png`: top row shows one differential NIR frame per 10 s stress band; below are windowed stress metrics and PPG.

### WESAD baseline / stress plots

Compare wrist BVP and chest ECG stress indicators side-by-side for baseline and stress protocol windows (requires `uv sync --extra notebook`):

```bash
# Default: 120 s from each condition, starting at t=0 within the block; all subjects
uv run scripts/visualize_WESAD_stress.py

# Custom window offset and length
uv run scripts/visualize_WESAD_stress.py start_time=60 window_sec=120
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_time` | `0.0` | Offset (s) into each baseline and stress block; 0 = block start |
| `window_sec` | `120.0` | Duration (s) extracted from each condition |

For every available subject, the script:

1. Crops `[start_time, start_time + window_sec)` from the first baseline block and the first stress block
2. Plots them side-by-side (baseline left, stress right), each with its own 0–`window_sec` time axis
3. Saves `results/wesad_S{id}_baseline_stress_{start}s_{window}s.png`

Each plot contains two columns (baseline | stress):

- Wrist BVP waveform with detected peaks marked
- Chest ECG waveform with R-peaks marked
- Windowed stress metrics (10 s sliding windows, 0.5 s hop) from BVP (blue) and ECG (black): mean IBI, SDNN, RMSSD, pNN50, Baevsky SI

IBIs are filtered to 0.35–1.50 s (~40–171 bpm) before metric computation. Subjects where the requested window exceeds a block length are skipped with a message.

Stress metrics are computed in `ir_stress.signals.stress_indicators` from inter-beat intervals (IBIs) extracted via peak detection on the PPG waveform (`extract_ibi`).

## Stress indicators

Time-domain metrics live in `src/ir_stress/signals/stress_indicators.py`:

| Metric | Higher values | Lower values |
|--------|---------------|--------------|
| **Baevsky SI** | More sympathetic tone / stress | Calmer autonomic balance |
| **SDNN** | Greater overall HRV | Reduced variability (stress, fatigue) |
| **RMSSD** | Stronger vagal tone | Sympathetic dominance |
| **pNN50** | More beat-to-beat flexibility | Rigid rhythm / arousal |
| **Mean HR** | Greater arousal (context-dependent) | Rest / recovery |

`extract_ibi()` peaks a PPG/rPPG waveform; `stress_indicators()` computes all metrics from an IBI array in seconds.

## Notebooks

Explore downloaded clips interactively:

```bash
uv sync --extra notebook
uv run jupyter notebook notebooks/visualize_nir_pulseox.ipynb
```

The notebook shows a montage of NIR frames and the aligned pulse-ox PPG waveform for a configurable time window. For scripted plots with full stress-indicator panels, prefer `scripts/data_exploration.py`.

## Project structure

```
scripts/
  download.py            # selective MR-NIRP fetch (Hydra)
  preprocess.py          # raw PGM → H5
  train.py
  evaluate.py
  inference.py           # neural or IR_iHR extraction (H5 or raw PGM)
  data_exploration.py    # NIR montage + windowed stress plots (Hydra)
  visualize_WESAD_stress.py  # WESAD baseline/stress BVP + ECG plots (Hydra)
  visualize_h5.py        # H5 clip preview
  smoke_test.py          # synthetic end-to-end check

mkdocs.yml                 # MkDocs config (technical report site)

.github/workflows/
  docs.yml                 # GitHub Pages deploy on push to main

notebooks/
  visualize_nir_pulseox.ipynb  # NIR frame montage + PulseOX waveform

docs/
  index.md                     # MkDocs landing page
  REPORT.md                    # Technical report (architecture, evaluation, figures)
  images/                      # Figures referenced by the report
  javascripts/mathjax.js       # MathJax config for MkDocs

data/models/
  shape_predictor_68_face_landmarks.dat  # dlib model (not bundled; see Installation)

src/ir_stress/
  models/       Backbone, STRppgHead, PhysNet, PhysNetLite, RppgModel
  dataset/      adapters, face_crop (YuNet/OpenFace/center), mr_nirp_download, H5 clips, synthetic
  training/     ContrastLoss, trainer
  evaluation/   Pearson/MSE evaluator
  inference/    rPPG + stress index pipeline, comparison plots
  signals/      filtering, metrics, stress_indicators; IR_iHR (ihr_core, ihr_regions, ihr_dlib, ihr_pipeline)
```

## Extending

**New dataset** — implement `DatasetAdapter` in `src/ir_stress/dataset/` (see `physdrive.py` stub).

**New backbone** — subclass `Backbone`, implement `encode()`, register in `build_model()` in `models/model.py` (see `lejepa.py` stub). The shared `STRppgHead` attaches to the backbone output.

**New extraction method** — implement grid signal extraction in `signals/ihr_regions.py` and wire into `signals/ihr_pipeline.py`, or add a new branch in `inference/pipeline.py`.

## Acknowledgements

Neural rPPG training and inference follow the [Contrast-Phys / Contrast-Phys+](https://github.com/zhaodongsun/contrast-phys) reference implementation (Sun & Li, TPAMI 2024). This repository adapts their PhysNet backbone, ST-rPPG head, contrastive loss, and preprocessing layout for NIR in-cabin video.

The classical IR_iHR extraction path is ported from the [IR_iHR](https://github.com/natalialmg/IR_iHR) reference code (Martinez et al., ICIP 2019). See `src/ir_stress/signals/ihr_*.py` and `scripts/inference.py` (`extraction_method=ihr`).

## References

- Sun & Li, [Contrast-Phys+](https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B) — reference code for neural rPPG (TPAMI 2024)
- Sun & Li, [Contrast-Phys](https://github.com/zhaodongsun/contrast-phys) (ECCV 2022)
- Martinez et al., [IR_iHR](https://github.com/natalialmg/IR_iHR) — reference code for classical IR iHR extraction (ICIP 2019)
- Nowara et al., [Near-Infrared Imaging Photoplethysmography During Driving](https://doi.org/10.1109/TITS.2020.3038317), IEEE TITS 2020
- [MR-NIRP Dataset](https://computationalimaging.rice.edu/mr-nirp-dataset/)
- [MR-NIRP Car (Google Drive)](https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK?usp=sharing)
- Schmidt et al., [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://doi.org/10.1145/3242969.3242985), ICMI 2018
- Baevsky, [Stress Index](https://www.kubios.com/blog/hrv-analysis-methods/) via HRV inter-beat intervals
- Martinez et al., [Non-Contact Photoplethysmogram and Instantaneous Heart Rate Estimation from Infrared Face Video](https://doi.org/10.1109/ICIP.2019.8802932), ICIP 2019
