"""Synthetic H5 clips for smoke testing."""

from pathlib import Path

import numpy as np

from ir_stress.dataset.base import write_h5


def create_synthetic_h5(
    out_dir: Path,
    name: str,
    num_frames: int = 1800,
    fps: int = 30,
    hr_bpm: float = 72.0,
) -> Path:
    """
    Write a synthetic H5 clip with a sinusoidal PPG trace and random NIR frames.

    Returns the path to the created file.
    """
    t = np.arange(num_frames) / fps
    ppg = np.sin(2 * np.pi * (hr_bpm / 60.0) * t).astype(np.float32)
    rng = np.random.default_rng(42)
    imgs = rng.random((num_frames, 128, 128, 1), dtype=np.float32) * 0.1
    imgs += 0.5 + 0.05 * ppg[:, None, None, None]

    out_path = out_dir / f"{name}.h5"
    write_h5(out_path, imgs, ppg)
    return out_path


def create_smoke_dataset(
    out_dir: Path, n_clips: int = 3, num_frames: int = 1800
) -> list[str]:
    """Create multiple synthetic clips and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_clips):
        paths.append(
            str(
                create_synthetic_h5(
                    out_dir, f"synthetic_{i}", num_frames=num_frames, hr_bpm=70.0 + i * 2
                )
            )
        )
    return paths
