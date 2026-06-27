# IR-Based Stress Detection

Unsupervised rPPG extraction from near-infrared (NIR) video using [Contrast-Phys+](https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B), with time-domain HRV stress indicators (Baevsky SI, SDNN, RMSSD, pNN50) from the recovered pulse signal.

## Overview

```
NIR video  →  Backbone (PhysNet)  →  ST-rPPG head  →  rPPG signal  →  IBI  →  stress indicators
                    ↑
         MR-NIRP pulse oximeter (PPG ground truth, eval only)
```

| Pipeline | Module | Metrics |
|----------|--------|---------|
| Download | `ir_stress.dataset.mr_nirp_download` | — |
| Preprocess | `ir_stress.dataset.mr_nirp_driving` | — |
| Train | `ir_stress.training` | MLflow: loss, IPR |
| Evaluate | `ir_stress.evaluation` | Pearson r, MSE |
| Inference | `ir_stress.inference` | rPPG, Baevsky SI, HRV |
| Data exploration | `scripts/data_exploration.py` | Windowed stress plots |

### End-to-end workflow

```
Download MR-NIRP (Google Drive) → download.py → preprocess.py → train.py → evaluate.py
                                      ↘ inference.py (H5 or raw PGM + landmarks)
                                      ↘ data_exploration.py (NIR montage + pulse-ox stress plots)
```

Smoke test (`scripts/smoke_test.py`) uses short **synthetic** clips — not MR-NIRP — to verify the install without the full dataset.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd IR-based-stress-detection
uv sync
```

For plotting scripts and notebooks, install the optional matplotlib extra:

```bash
uv sync --extra notebook
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

## Preprocessing

Face crops follow the [Contrast-Phys preprocessing](https://github.com/zhaodongsun/contrast-phys/blob/master/preprocessing.py) pipeline. Run [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) on each clip to obtain landmark CSVs:

```bash
# Convert PGM sequence to video first, then:
./FeatureExtraction -f <video> -out_dir data/landmarks -2Dfp
```

Preprocess to H5:

```bash
uv run scripts/preprocess.py
uv run scripts/preprocess.py raw_dir=data/raw/mr-nirp h5_dir=data/h5 landmarks_dir=data/landmarks
```

Each output file contains:

- `imgs` — `[N, 128, 128, 1]` float32 NIR face crops
- `ppg` — `[N]` pulse-oximeter PPG waveform resampled to video fps

## Training

Training reads **preprocessed H5 clips** from `h5_dir/`. You need **at least two H5 files** (different subjects) before `train.py` will run.

### 1. Preprocess first

```bash
uv run scripts/preprocess.py raw_dir=data/raw/mr-nirp h5_dir=data/h5 face_crop_mode=yunet
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

Defaults live in `ir_stress.config.Config` (overridden via Hydra in `scripts/train.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `physnet` | Backbone name |
| `in_ch` | `1` | NIR input channels |
| `fs` | `30` | Video frame rate (fps) |
| `clip_seconds` | `10` | Temporal window per batch item (300 frames) |
| `spatial_dim` | `2` | ST-rPPG 2×2 spatial grid (+ mean channel) |
| `epochs` | `30` | Training epochs |
| `lr` | `1e-5` | AdamW learning rate |
| `batch_size` | `2` | **Must be 2** — Contrast-Phys+ contrastive loss needs two clips |
| `face_size` | `128` | Face crop side length; downscale on load if H5 is larger |
| `label_ratio` | `0.0` | Fraction of clips using GT PPG in loss (0 = unsupervised) |
| `video_duration_sec` | `60` | Effective clip length for steps/epoch (`iters ≈ video_duration_sec / clip_seconds`) |
| `eval_window_sec` | `30` | Evaluation/inference window length |
| `val_subjects` | `[1]` | Held-out subject IDs (leave-one-out) |
| `wavelengths` | `[940]` | Clip bands to include |
| `num_workers` | `0` | DataLoader workers (use 2–4 on GPU) |
| `device` | `null` (auto) | `cuda`, `cuda:N`, `cpu`, or `mps` |
| `use_amp` | `true` | Mixed precision on CUDA (set `false` on CPU) |
| `h5_dir` | `data/h5` | Preprocessed H5 directory |
| `checkpoint_dir` | `checkpoints` | Output directory for weights and run metadata |
| `mlflow_experiment` | `ir-stress-rppg` | MLflow experiment name |

Optional: re-preprocess at lower resolution for faster I/O and smaller H5 files:

```bash
uv run scripts/preprocess.py face_size=64 h5_dir=data/h5_64
uv run scripts/train.py h5_dir=data/h5_64 face_size=64 'val_subjects=[11]'
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

From a preprocessed H5 clip:

```bash
uv run scripts/inference.py \
  checkpoint=checkpoints/epoch29.pt \
  input_h5=data/h5/subject2_some_clip.h5 \
  output_dir=results/inference
```

Or directly from raw NIR PGM frames + OpenFace landmarks (no H5 required):

```bash
uv run scripts/inference.py \
  checkpoint=checkpoints/epoch29.pt \
  input_dir=data/raw/mr-nirp/Subject1/subject1_driving_still_940/NIR \
  landmarks_csv=data/landmarks/subject1_driving_still_940.csv \
  output_dir=results/inference
```

Raw mode crops faces on the fly (one window at a time) using the same logic as preprocessing.

Outputs `{stem}_results.json` (summary + Baevsky SI) and `{stem}_rppg.npy` (full waveform).

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

The notebook shows a montage of NIR frames and the aligned pulse-ox PPG waveform for a configurable time window. For scripted plots with full stress-indicator panels, use `scripts/data_exploration.py` instead.

## Project structure

```
scripts/
  download.py            # selective MR-NIRP fetch (Hydra)
  preprocess.py          # raw PGM → H5
  train.py
  evaluate.py
  inference.py           # H5 or raw PGM + landmarks
  data_exploration.py    # NIR montage + windowed stress plots (Hydra)
  smoke_test.py          # synthetic end-to-end check

notebooks/
  visualize_nir_pulseox.ipynb  # NIR frame montage + PulseOX waveform

src/ir_stress/
  models/       Backbone, STRppgHead, PhysNet, RppgModel
  dataset/    adapters, face_crop, mr_nirp_download, H5 clips, synthetic
  training/     ContrastLoss, trainer
  evaluation/   Pearson/MSE evaluator
  inference/    rPPG + stress index pipeline
  signals/      filtering, metrics, stress_indicators (IBI, Baevsky SI, HRV)
```

## Extending

**New dataset** — implement `DatasetAdapter` in `src/ir_stress/dataset/` (see `physdrive.py` stub).

**New backbone** — subclass `Backbone`, implement `encode()`, register in `build_model()` in `models/model.py` (see `lejepa.py` stub). The shared `STRppgHead` attaches to the backbone output.

## References

- Sun & Li, [Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast](https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B), TPAMI 2024
- Sun & Li, [Contrast-Phys](https://github.com/zhaodongsun/contrast-phys), ECCV 2022
- Nowara et al., [Near-Infrared Imaging Photoplethysmography During Driving](https://doi.org/10.1109/TITS.2020.3038317), IEEE TITS 2020
- [MR-NIRP Dataset](https://computationalimaging.rice.edu/mr-nirp-dataset/)
- [MR-NIRP Car (Google Drive)](https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK?usp=sharing)
- Baevsky, [Stress Index](https://www.kubios.com/blog/hrv-analysis-methods/) via HRV inter-beat intervals
