"""Explore MR-NIRP NIR frames alongside pulse-ox-derived stress indicators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat

from ir_stress.dataset.face_crop import annotate_bbox, face_bbox_corners, load_landmarks, read_nir_pair
from ir_stress.dataset.mr_nirp_driving import MRNIRPDrivingAdapter
from ir_stress.signals.stress_indicators import extract_ibi_with_beat_times, stress_indicators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIR_FPS = 30
HOP_SEC = 0.5 #1.0
STRESS_WINDOW_SEC = 10.0 #10.0

STRESS_SERIES: tuple[tuple[str, str], ...] = (
    ("mean_ibi_seconds", "Mean IBI (s)"),
    ("sdnn_ms", "SDNN (ms)"),
    ("rmssd_ms", "RMSSD (ms)"),
    ("pnn50", "pNN50 (%)"),
    # ("mean_hr_bpm", "Mean HR (BPM)"),
    ("baevsky_si", "Baevsky SI"),
    # ("baevsky_si_sqrt", "√Baevsky SI"),
)


@dataclass
class RunConfig:
    subject_number: int = 4
    start_time: float = 299 / 30  # recording origin; video frame 300 at t=(300-1)/30
    window_sec: float = 100.0  # seconds of PPG/NIR segment from start_time
    bin_width_seconds: float = 0.05 #0.05
    raw_root: str = "data/raw/mr-nirp"
    output_dir: str = "results"
    face_crop_mode: str = "center" #"yunet"
    landmarks_dir: str = "data/landmarks"


def recording_time_to_frame_num(t_sec: float) -> int:
    """1-based NIR video frame at recording time t (frame 1 captured at t=0)."""
    return int(t_sec * NIR_FPS) + 1


def frame_num_to_recording_time(frame_num: int) -> float:
    """Recording time (s) when a 1-based video frame was captured."""
    return (frame_num - 1) / NIR_FPS


def load_pulseox(pulseox_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pulse-ox PPG waveform and relative time (seconds)."""
    mat = loadmat(pulseox_path)
    if "pulseOxRecord" in mat:
        ppg = mat["pulseOxRecord"].squeeze().astype(np.float64)
        t = mat["pulseOxTime"].squeeze().astype(np.float64)
        t = t - t[0]
        return ppg, t
    if "data" in mat:
        ppg = mat["data"].squeeze().astype(np.float64)
        t = np.arange(len(ppg)) / float(mat.get("fs", [[400]])[0, 0])
        return ppg, t
    if "val" in mat:
        ppg = mat["val"].squeeze().astype(np.float64)
        t = np.arange(len(ppg))
        return ppg, t
    raise KeyError(f"Unknown pulseOx.mat layout: {list(mat.keys())}")


def compute_windowed_stress(
    ppg: np.ndarray,
    time_sec: np.ndarray,
    bin_width_seconds: float,
    hop_sec: float,
    anchor_time: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute stress indicators over STRESS_WINDOW_SEC windows stepped by hop_sec."""
    keys = [key for key, _ in STRESS_SERIES]
    empty = np.array([]), {key: np.array([]) for key in keys}
    if len(ppg) < 2:
        return empty

    dt = np.median(np.diff(time_sec))
    fs = 1.0 / dt if dt > 0 else 1.0
    ibi, beat_times, _ = extract_ibi_with_beat_times(ppg, fs, time_sec)
    if len(ibi) < 2:
        return empty

    series = {key: [] for key in keys}
    centers: list[float] = []
    window_start = float(anchor_time if anchor_time is not None else time_sec[0])
    t_end = float(time_sec[-1])

    if hop_sec >= STRESS_WINDOW_SEC:
        while window_start < t_end - 1e-9:
            window_end = min(window_start + STRESS_WINDOW_SEC, t_end)
            in_window = (beat_times >= window_start) & (beat_times < window_end)
            window_ibi = ibi[in_window]
            if len(window_ibi) >= 2:
                metrics = stress_indicators(window_ibi, bin_width_seconds=bin_width_seconds)
                for key in keys:
                    series[key].append(float(metrics[key]))
                centers.append((window_start + window_end) / 2.0)
            window_start += hop_sec
    else:
        while window_start + STRESS_WINDOW_SEC <= t_end:
            window_end = window_start + STRESS_WINDOW_SEC
            in_window = (beat_times >= window_start) & (beat_times < window_end)
            window_ibi = ibi[in_window]
            if len(window_ibi) >= 2:
                metrics = stress_indicators(window_ibi, bin_width_seconds=bin_width_seconds)
                for key in keys:
                    series[key].append(float(metrics[key]))
                centers.append((window_start + window_end) / 2.0)
            window_start += hop_sec

    return np.asarray(centers), {key: np.asarray(values) for key, values in series.items()}


def find_subject_session(raw_root: Path, subject_number: int):
    adapter = MRNIRPDrivingAdapter()
    sessions = adapter.discover_sessions(raw_root)
    matches = [s for s in sessions if s.subject_id == subject_number]
    if not matches:
        subjects = sorted({s.subject_id for s in sessions})
        raise ValueError(
            f"No session found for subject {subject_number}. "
            f"Available subjects: {subjects}"
        )
    return matches[0]


def _shade_stress_windows(ax: plt.Axes, duration: float) -> None:
    """Alternate light gray / white vertical bands per STRESS_WINDOW_SEC."""
    t = 0.0
    band = 0
    while t < duration:
        t1 = min(t + STRESS_WINDOW_SEC, duration)
        if band % 2:
            ax.axvspan(t, t1, facecolor="0.9", edgecolor="none", zorder=0)
        band += 1
        t += STRESS_WINDOW_SEC


def _stress_band_windows(segment_duration: float) -> list[tuple[float, float]]:
    """10 s bands on the segment grid; final band may be shorter (matches gray shading)."""
    windows: list[tuple[float, float]] = []
    t0 = 0.0
    while t0 < segment_duration - 1e-9:
        t1 = min(t0 + STRESS_WINDOW_SEC, segment_duration)
        windows.append((t0, t1))
        t0 += STRESS_WINDOW_SEC
    return windows


def _load_preview_frames(
    nir_frames: list[Path],
    segment_start: float,
    band_windows: list[tuple[float, float]],
    face_crop_mode: str,
    landmark=None,
) -> list[tuple[float, float, float, int, str, str, np.ndarray]]:
    """Load one differential NIR frame per band at the window center (matches stress circles)."""
    frame_images: list[tuple[float, float, float, int, str, str, np.ndarray]] = []
    for t0, t1 in band_windows:
        t_center = (t0 + t1) / 2.0
        t_recording = segment_start + t_center
        frame_num = recording_time_to_frame_num(t_recording)
        video_idx = frame_num - 1
        on_idx = 2 * video_idx
        off_idx = on_idx + 1
        if video_idx < 0 or off_idx >= len(nir_frames):
            continue
        on_name = nir_frames[on_idx].name
        off_name = nir_frames[off_idx].name
        img = read_nir_pair(nir_frames, video_idx)
        corners = face_bbox_corners(
            img, face_crop_mode, landmark=landmark, video_frame_idx=video_idx
        )
        if corners is not None:
            img = annotate_bbox(img, corners)
        frame_images.append(
            (t0, t1, t_center, frame_num, on_name, off_name, img)
        )
    return frame_images


def _log_frame_mapping(
    segment_start: float,
    frame_images: list[tuple[float, float, float, int, str, str, np.ndarray]],
) -> None:
    """Print which video frame and PGM pair is shown for each band."""
    print("NIR thumbnails (1 per 10 s band; snapshot at window center, same as stress circles):")
    print(f"  segment origin: recording t={segment_start:.3f}s "
          f"(video frame {recording_time_to_frame_num(segment_start)})")
    for t0, t1, t_center, frame_num, on_name, off_name, _ in frame_images:
        t_rec = segment_start + t_center
        print(
            f"  band [{t0:6.1f}, {t1:6.1f})s rel | center {t_center:6.1f}s | "
            f"recording t={t_rec:7.3f}s | video frame {frame_num:4d} | "
            f"PGM {on_name} − {off_name}"
        )


def _draw_thumbnails(
    ax: plt.Axes,
    frame_images: list[tuple[float, float, float, int, str, str, np.ndarray]],
    segment_start: float,
) -> None:
    """Draw one square thumbnail per 10 s band; x-axis units match time (s) below."""
    band_h = STRESS_WINDOW_SEC
    for t0, t1, t_center, frame_num, on_name, off_name, img in frame_images:
        ax.imshow(
            img,
            extent=[t0, t1, 0, band_h],
            cmap="gray",
            aspect="equal",
            origin="upper",
            zorder=2,
        )
        t_rec = segment_start + t_center
        ax.text(
            t_center,
            band_h * 1.05,
            f"fr {frame_num} @ {t_rec:.2f}s\n{on_name}−\n{off_name}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_exploration(
    cfg: RunConfig,
    segment_duration: float,
    frame_images: list[tuple[float, float, float, int, str, str, np.ndarray]],
    segment_start: float,
    line_time: np.ndarray,
    line_series: dict[str, np.ndarray],
    band_time: np.ndarray,
    band_series: dict[str, np.ndarray],
    ppg_time: np.ndarray,
    ppg_values: np.ndarray,
    peak_time: np.ndarray,
    peak_values: np.ndarray,
    output_path: Path,
) -> None:
    """Save a figure with NIR previews on top and stress metrics below."""
    n_metrics = len(STRESS_SERIES)
    plot_duration = max(segment_duration, cfg.window_sec)
    fig_w = 14.0 * max(plot_duration / 100.0, 1.0)
    fig = plt.figure(figsize=(fig_w, 2.8 + 1.8 * n_metrics))
    height_ratios = [1.4] + [1.0] * n_metrics
    gs = GridSpec(
        1 + n_metrics, 1, figure=fig, height_ratios=height_ratios, hspace=0.5
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_xlim(0, plot_duration)
    ax_img.set_ylim(0, STRESS_WINDOW_SEC * 1.2)
    ax_img.set_aspect("equal")
    _shade_stress_windows(ax_img, plot_duration)
    _draw_thumbnails(ax_img, frame_images, segment_start)
    ax_img.axis("off")

    for row, (key, label) in enumerate(STRESS_SERIES, start=1):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_xlim(0, plot_duration)
        _shade_stress_windows(ax, plot_duration)
        line_values = line_series[key]
        band_values = band_series[key]
        if len(line_time) and len(line_values):
            ax.plot(line_time, line_values, linewidth=1.2, color="tab:blue", zorder=2)
        if len(band_time) and len(band_values):
            ax.plot(
                band_time,
                band_values,
                "o",
                color="tab:blue",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
        ax.set_ylabel(label, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.3)

        ax_ppg = ax.twinx()
        if len(ppg_time) and len(ppg_values):
            ax_ppg.plot(ppg_time, ppg_values, linewidth=0.8, color="crimson", alpha=0.7)
        if len(peak_time) and len(peak_values):
            ax_ppg.plot(
                peak_time,
                peak_values,
                "o",
                color="crimson",
                markersize=4,
                markerfacecolor="none",
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
            )
        ax_ppg.set_ylabel("PPG", color="crimson")
        ax_ppg.tick_params(axis="y", labelcolor="crimson")
        if row < n_metrics:
            ax_ppg.set_xticklabels([])

        if row == n_metrics:
            ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Subject {cfg.subject_number} | start={cfg.start_time}s | "
        f"segment={cfg.window_sec}s | face_crop={cfg.face_crop_mode} | "
        f"x-axis = seconds from start",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run(cfg: RunConfig) -> Path:
    raw_root = (PROJECT_ROOT / cfg.raw_root).resolve()
    session = find_subject_session(raw_root, cfg.subject_number)
    nir_frames = list(session.nir_dir.glob("Frame*.pgm"))
    nir_frames.sort(key=lambda x: int(x.stem.replace("Frame", "")))

    ppg, pulse_time = load_pulseox(session.pulseox_path)
    segment_start = cfg.start_time
    segment_end = segment_start + cfg.window_sec
    mask = (pulse_time >= segment_start) & (pulse_time < segment_end)
    if not np.any(mask):
        raise ValueError(
            f"No pulse-ox samples in [{segment_start:.1f}, {segment_end:.1f}) s"
        )

    segment_ppg = ppg[mask]
    segment_time = pulse_time[mask]
    dt = float(np.median(np.diff(segment_time)))
    fs = 1.0 / dt if dt > 0 else 1.0
    _, _, peaks = extract_ibi_with_beat_times(segment_ppg, fs, segment_time)
    peak_time = segment_time[peaks] - segment_start
    peak_values = segment_ppg[peaks]

    line_time, line_series = compute_windowed_stress(
        segment_ppg, segment_time, cfg.bin_width_seconds, hop_sec=HOP_SEC,
        anchor_time=segment_start,
    )
    band_time, band_series = compute_windowed_stress(
        segment_ppg, segment_time, cfg.bin_width_seconds, hop_sec=STRESS_WINDOW_SEC,
        anchor_time=segment_start,
    )
    line_time = line_time - segment_start
    band_time = band_time - segment_start
    ppg_plot_time = segment_time - segment_start
    segment_duration = float(ppg_plot_time[-1])

    band_windows = _stress_band_windows(segment_duration)
    landmark = None
    if cfg.face_crop_mode == "openface":
        csv_path = (PROJECT_ROOT / cfg.landmarks_dir / f"{session.condition}.csv").resolve()
        landmark = load_landmarks(csv_path)
    frame_images = _load_preview_frames(
        nir_frames, segment_start, band_windows, cfg.face_crop_mode, landmark
    )
    _log_frame_mapping(segment_start, frame_images)

    output_path = PROJECT_ROOT / cfg.output_dir / f"subject{cfg.subject_number}_exploration.png"
    plot_exploration(
        cfg,
        segment_duration,
        frame_images,
        segment_start,
        line_time,
        line_series,
        band_time,
        band_series,
        ppg_plot_time,
        segment_ppg,
        peak_time,
        peak_values,
        output_path,
    )
    return output_path


@hydra.main(version_base=None, config_path=None, config_name="run_config")
def main(cfg: RunConfig) -> None:
    output_path = run(cfg)
    print(f"Saved exploration plot to {output_path}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="run_config", node=RunConfig)
    main()
