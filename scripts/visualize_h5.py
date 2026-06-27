"""Visualize preprocessed H5 clips (face crops + aligned PPG)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore

from ir_stress.config import resolve_h5_dir
from ir_stress.data.base import h5_num_frames, list_h5_paths, read_h5_window
from ir_stress.signals.stress_indicators import extract_ibi_with_beat_times

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunConfig:
    h5_path: str | None = None
    h5_dir: str | None = None
    h5_index: int = 0
    face_crop_mode: str = "yunet"
    start_time: float = 0.0
    window_sec: float = 30.0
    n_preview_frames: int = 10
    fs: int = 30
    output_dir: str = "results"


def resolve_h5_path(cfg: RunConfig) -> tuple[Path, str]:
    """Return the H5 file to visualize and the resolved h5 directory."""
    if cfg.h5_path:
        path = Path(cfg.h5_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"H5 file not found: {path}")
        return path, str(path.parent)

    h5_dir = (PROJECT_ROOT / resolve_h5_dir(cfg)).resolve()
    paths = list_h5_paths(h5_dir)
    if not paths:
        raise FileNotFoundError(f"No .h5 files in {h5_dir}")
    if cfg.h5_index < 0 or cfg.h5_index >= len(paths):
        raise IndexError(
            f"h5_index={cfg.h5_index} out of range for {len(paths)} files in {h5_dir}"
        )
    return Path(paths[cfg.h5_index]), str(h5_dir)


def _preview_frame_indices(start: int, end: int, n_preview: int) -> np.ndarray:
    """Evenly sample frame indices in [start, end)."""
    n_frames = end - start
    if n_frames <= 0:
        return np.array([], dtype=int)
    n_preview = min(n_preview, n_frames)
    return np.linspace(start, end - 1, n_preview, dtype=int)


def plot_h5_clip(
    h5_path: Path,
    imgs: np.ndarray,
    ppg: np.ndarray,
    start_frame: int,
    fs: int,
    n_preview_frames: int,
    output_path: Path,
    face_crop_mode: str,
) -> None:
    """Save a montage of face crops and the aligned PPG segment."""
    n_frames = imgs.shape[0]
    duration = n_frames / fs
    time_sec = np.arange(n_frames) / fs
    segment_start_sec = start_frame / fs

    preview_idx = _preview_frame_indices(0, n_frames, n_preview_frames)
    n_cols = len(preview_idx)
    fig_h = 5.0 if n_cols else 3.0
    fig, axes = plt.subplots(
        2, 1, figsize=(max(10.0, n_cols * 1.2), fig_h), gridspec_kw={"height_ratios": [1.2, 1]}
    )
    ax_montage, ax_ppg = axes

    if n_cols:
        montage = np.hstack([imgs[i, :, :, 0] for i in preview_idx])
        times = [f"{idx / fs:.1f}s" for idx in preview_idx]
        ax_montage.imshow(montage, cmap="gray", aspect="auto")
        ax_montage.set_title(
            f"{n_cols} frames evenly spaced over {duration:.1f} s "
            f"(frames {preview_idx[0] + start_frame}–{preview_idx[-1] + start_frame}; "
            f"times {times[0]}–{times[-1]})"
        )
    ax_montage.axis("off")

    ax_ppg.plot(time_sec, ppg, color="crimson", linewidth=0.9)
    dt = 1.0 / fs
    _, _, peaks = extract_ibi_with_beat_times(ppg, fs, time_sec)
    if len(peaks):
        ax_ppg.plot(
            time_sec[peaks],
            ppg[peaks],
            "o",
            color="crimson",
            markersize=4,
            markerfacecolor="none",
            markeredgewidth=1.0,
            linestyle="none",
        )
    ax_ppg.set_xlabel("Time in segment (s)")
    ax_ppg.set_ylabel("PPG")
    ax_ppg.set_xlim(0, duration)
    ax_ppg.grid(True, alpha=0.3)
    ax_ppg.set_title(f"Aligned PPG ({n_frames} samples @ {fs} Hz, dt={dt:.4f} s)")

    fig.suptitle(
        f"{h5_path.name} | face_crop={face_crop_mode} | "
        f"segment start={segment_start_sec:.2f} s | duration={duration:.1f} s",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run(cfg: RunConfig) -> Path:
    h5_path, h5_dir = resolve_h5_path(cfg)
    total_frames = h5_num_frames(h5_path)
    total_duration = total_frames / cfg.fs

    start_frame = int(cfg.start_time * cfg.fs)
    end_frame = min(start_frame + int(cfg.window_sec * cfg.fs), total_frames)
    if start_frame >= total_frames:
        raise ValueError(
            f"start_time={cfg.start_time}s (frame {start_frame}) is beyond clip "
            f"length ({total_duration:.1f} s, {total_frames} frames)"
        )
    if end_frame <= start_frame:
        raise ValueError("window_sec must be positive and fit within the clip")

    imgs, ppg = read_h5_window(h5_path, start_frame, end_frame)
    output_path = PROJECT_ROOT / cfg.output_dir / f"{h5_path.stem}_h5_preview.png"
    plot_h5_clip(
        h5_path,
        imgs,
        ppg,
        start_frame,
        cfg.fs,
        cfg.n_preview_frames,
        output_path,
        cfg.face_crop_mode,
    )
    print(
        f"Clip: {h5_path.name} | face_crop={cfg.face_crop_mode} | h5_dir={h5_dir} | "
        f"frames {start_frame}–{end_frame - 1} "
        f"({(end_frame - start_frame) / cfg.fs:.1f} s of {total_duration:.1f} s total)"
    )
    return output_path


@hydra.main(version_base=None, config_path=None, config_name="run_config")
def main(cfg: RunConfig) -> None:
    output_path = run(cfg)
    print(f"Saved H5 preview to {output_path}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="run_config", node=RunConfig)
    main()
