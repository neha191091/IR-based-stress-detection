"""Reproducible train/test H5 splits from split metadata."""

import json
from dataclasses import fields
from pathlib import Path

from ir_stress.config import TrainConfig, resolve_h5_dir
from ir_stress.dataset.base import list_h5_paths
from ir_stress.dataset.mr_nirp_driving import MRNIRPDrivingAdapter

SPLIT_FILENAME = "split.json"


def resolve_h5_splits(cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """Build train/test H5 path lists from config and split settings."""
    h5_path = Path(resolve_h5_dir(cfg))
    adapter = MRNIRPDrivingAdapter()
    sessions = adapter.discover_sessions(Path(cfg.raw_dir), wavelengths=cfg.wavelengths)
    train_sessions, test_sessions = adapter.train_test_split(sessions, cfg.val_subjects)

    all_h5 = {Path(p).name: p for p in list_h5_paths(h5_path)}
    train_list = [
        all_h5[f"subject{s.subject_id}_{s.condition}.h5"]
        for s in train_sessions
        if f"subject{s.subject_id}_{s.condition}.h5" in all_h5
    ]
    test_list = [
        all_h5[f"subject{s.subject_id}_{s.condition}.h5"]
        for s in test_sessions
        if f"subject{s.subject_id}_{s.condition}.h5" in all_h5
    ]

    if not train_list:
        raise ValueError(
            f"No training H5 files matched val_subjects={cfg.val_subjects} in {h5_path}. "
            "Preprocess more subjects or adjust val_subjects."
        )
    return train_list, test_list


def split_metadata(cfg: TrainConfig) -> dict[str, list[int]]:
    """Return the split parameters to persist for a training run."""
    return {"val_subjects": cfg.val_subjects}


def save_split_metadata(run_dir: Path, cfg: TrainConfig) -> Path:
    """Write hold-out subjects next to checkpoints."""
    path = run_dir / SPLIT_FILENAME
    with open(path, "w") as f:
        json.dump(split_metadata(cfg), f, indent=2)
    return path


def load_config_from_run_dir(run_dir: Path) -> TrainConfig:
    """Load the training config saved alongside checkpoints."""
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    valid = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in raw.items() if k in valid})


def load_split_metadata(run_dir: Path) -> dict[str, list[int]]:
    """Load hold-out subjects from a training run directory."""
    with open(run_dir / SPLIT_FILENAME) as f:
        return json.load(f)


def _config_with_split_meta(run_dir: Path) -> TrainConfig:
    """Rebuild config using saved split metadata."""
    cfg = load_config_from_run_dir(run_dir)
    meta = load_split_metadata(run_dir)
    cfg.val_subjects = list(meta["val_subjects"])
    return cfg


def resolve_test_paths(run_dir: Path) -> list[str]:
    """Reconstruct the test H5 paths from saved split metadata."""
    _, test_list = resolve_h5_splits(_config_with_split_meta(run_dir))
    return test_list


def resolve_train_paths(run_dir: Path) -> list[str]:
    """Reconstruct the train H5 paths from saved split metadata."""
    train_list, _ = resolve_h5_splits(_config_with_split_meta(run_dir))
    return train_list
