"""Inference comparison plots: ground truth (blue) vs inferred (black)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ir_stress.dataset.base import SessionMeta, _ppg_dataset
from ir_stress.dataset.mr_nirp_driving import MRNIRPDrivingAdapter, _resample_ppg
from ir_stress.signals.filtering import butter_bandpass
from ir_stress.signals.stress_indicators import extract_ibi_with_beat_times, stress_indicators

HOP_SEC = 0.5
STRESS_WINDOW_SEC = 10.0

STRESS_SERIES: tuple[tuple[str, str], ...] = (
    ("mean_ibi_seconds", "Mean IBI (s)"),
    ("sdnn_ms", "SDNN (ms)"),
    ("rmssd_ms", "RMSSD (ms)"),
    ("pnn50", "pNN50 (%)"),
    ("baevsky_si", "Baevsky SI"),
)

GT_COLOR = "tab:blue"
INF_COLOR = "black"
PPG_BAND_HZ = (0.6, 4.0)
PSD_DISPLAY_HZ = (0.0, 5.0)


def _scale_to_match_range(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Linearly scale pred so its min/max matches ref's min/max."""
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    p_min, p_max = float(pred.min()), float(pred.max())
    r_min, r_max = float(ref.min()), float(ref.max())
    if p_max - p_min < 1e-12:
        return np.full_like(pred, (r_min + r_max) / 2.0)
    return (pred - p_min) / (p_max - p_min) * (r_max - r_min) + r_min


def _compute_psd(
    sig: np.ndarray,
    fs: float,
    *,
    fmin: float = PSD_DISPLAY_HZ[0],
    fmax: float = PSD_DISPLAY_HZ[1],
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of a demeaned signal, normalized to unit area in [fmin, fmax]."""
    from scipy.signal import welch

    sig = np.reshape(sig, -1).astype(np.float64)
    sig = sig - np.mean(sig)
    nperseg = min(256, max(8, len(sig) // 4))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs, psd = freqs[mask], psd[mask]
    total = float(np.sum(psd))
    if total > 0:
        psd = psd / total
    return freqs, psd


def _parse_session_from_stem(stem: str) -> tuple[int, str] | None:
    match = re.match(r"subject(\d+)_(.+)", stem, re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def _find_session(raw_root: Path, subject_id: int, condition: str) -> SessionMeta | None:
    adapter = MRNIRPDrivingAdapter()
    for session in adapter.discover_sessions(raw_root):
        if session.subject_id == subject_id and session.condition == condition:
            return session
    return None


def load_ground_truth_ppg(
    *,
    stem: str,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    raw_root: Path | str = "data/raw/mr-nirp",
    num_frames: int | None = None,
    fs: int = 30,
) -> np.ndarray | None:
    """Load pulse-ox ground truth aligned to the clip, if available."""
    if input_h5 is not None:
        import h5py

        try:
            with h5py.File(input_h5, "r") as f:
                return _ppg_dataset(f)[:].astype(np.float64)
        except KeyError:
            pass

    parsed = _parse_session_from_stem(stem)
    if parsed is None and input_dir is not None:
        condition = input_dir.parent.name
        match = re.match(r"subject(\d+)", condition, re.IGNORECASE)
        if match:
            parsed = int(match.group(1)), condition

    if parsed is None:
        return None

    subject_id, condition = parsed
    session = _find_session(Path(raw_root), subject_id, condition)
    if session is None or num_frames is None:
        return None
    return _resample_ppg(session.pulseox_path, num_frames, fs).astype(np.float64)


def compute_windowed_stress(
    ppg: np.ndarray,
    time_sec: np.ndarray,
    bin_width_seconds: float,
    hop_sec: float,
    stress_window_sec: float = STRESS_WINDOW_SEC,
    anchor_time: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute stress indicators over fixed windows stepped by hop_sec."""
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

    if hop_sec >= stress_window_sec:
        while window_start < t_end - 1e-9:
            window_end = min(window_start + stress_window_sec, t_end)
            in_window = (beat_times >= window_start) & (beat_times < window_end)
            window_ibi = ibi[in_window]
            if len(window_ibi) >= 2:
                metrics = stress_indicators(window_ibi, bin_width_seconds=bin_width_seconds)
                for key in keys:
                    series[key].append(float(metrics[key]))
                centers.append((window_start + window_end) / 2.0)
            window_start += hop_sec
    else:
        while window_start + stress_window_sec <= t_end:
            window_end = window_start + stress_window_sec
            in_window = (beat_times >= window_start) & (beat_times < window_end)
            window_ibi = ibi[in_window]
            if len(window_ibi) >= 2:
                metrics = stress_indicators(window_ibi, bin_width_seconds=bin_width_seconds)
                for key in keys:
                    series[key].append(float(metrics[key]))
                centers.append((window_start + window_end) / 2.0)
            window_start += hop_sec

    return np.asarray(centers), {key: np.asarray(values) for key, values in series.items()}


def _shade_stress_windows(ax, duration: float, stress_window_sec: float) -> None:
    t = 0.0
    band = 0
    while t < duration:
        t1 = min(t + stress_window_sec, duration)
        if band % 2:
            ax.axvspan(t, t1, facecolor="0.9", edgecolor="none", zorder=0)
        band += 1
        t += stress_window_sec


def save_inference_comparison_plot(
    output_path: Path,
    gt_ppg: np.ndarray,
    inferred_ppg: np.ndarray,
    fs: int,
    *,
    title: str = "",
    bin_width_seconds: float = 0.05,
    hop_sec: float = HOP_SEC,
    stress_window_sec: float = STRESS_WINDOW_SEC,
) -> Path:
    """Save stress-metric rows with ground truth (blue) and inferred (black) overlays."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = min(len(gt_ppg), len(inferred_ppg))
    if n < 2:
        raise ValueError("Need at least two samples to plot inference comparison.")

    gt = butter_bandpass(gt_ppg[:n].astype(np.float64), *PPG_BAND_HZ, fs)
    inf = butter_bandpass(inferred_ppg[:n].astype(np.float64), *PPG_BAND_HZ, fs)
    inf_scaled = _scale_to_match_range(inf, gt)
    time_sec = np.arange(n) / fs
    segment_start = 0.0

    gt_line_t, gt_line = compute_windowed_stress(
        gt, time_sec, bin_width_seconds, hop_sec, stress_window_sec, anchor_time=segment_start
    )
    gt_band_t, gt_band = compute_windowed_stress(
        gt,
        time_sec,
        bin_width_seconds,
        stress_window_sec,
        stress_window_sec,
        anchor_time=segment_start,
    )
    inf_line_t, inf_line = compute_windowed_stress(
        inf, time_sec, bin_width_seconds, hop_sec, stress_window_sec, anchor_time=segment_start
    )
    inf_band_t, inf_band = compute_windowed_stress(
        inf,
        time_sec,
        bin_width_seconds,
        stress_window_sec,
        stress_window_sec,
        anchor_time=segment_start,
    )

    gt_line_t = gt_line_t - segment_start
    gt_band_t = gt_band_t - segment_start
    inf_line_t = inf_line_t - segment_start
    inf_band_t = inf_band_t - segment_start
    plot_time = time_sec - segment_start
    duration = float(plot_time[-1])

    _, _, gt_peaks = extract_ibi_with_beat_times(gt, fs, time_sec)
    _, _, inf_peaks = extract_ibi_with_beat_times(inf_scaled, fs, time_sec)
    gt_peak_t = time_sec[gt_peaks] - segment_start
    gt_peak_v = gt[gt_peaks]
    inf_peak_t = time_sec[inf_peaks] - segment_start
    inf_peak_v = inf_scaled[inf_peaks]

    gt_freqs, gt_psd = _compute_psd(gt, fs)
    inf_freqs, inf_psd = _compute_psd(inf_scaled, fs)

    n_metrics = len(STRESS_SERIES)
    n_rows = n_metrics + 1
    fig_w = 14.0 * max(duration / 100.0, 1.0)
    fig = plt.figure(figsize=(fig_w, 1.8 * n_rows))
    gs = GridSpec(n_rows, 1, figure=fig, hspace=0.5)

    ax_psd = fig.add_subplot(gs[0, 0])
    ax_psd.plot(gt_freqs, gt_psd, linewidth=1.2, color=GT_COLOR, label="GT")
    ax_psd.plot(inf_freqs, inf_psd, linewidth=1.2, color=INF_COLOR, label="Inferred")
    ax_psd.axvspan(*PPG_BAND_HZ, facecolor="0.92", edgecolor="none", zorder=0)
    ax_psd.set_xlim(*PSD_DISPLAY_HZ)
    ax_psd.set_ylabel("Norm. PSD")
    ax_psd.set_title("Power spectral density", fontsize=9, loc="left")
    ax_psd.legend(loc="upper right", fontsize=8)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_xticklabels([])

    for row, (key, label) in enumerate(STRESS_SERIES):
        ax = fig.add_subplot(gs[row + 1, 0])
        ax.set_xlim(0, duration)
        _shade_stress_windows(ax, duration, stress_window_sec)

        if len(gt_line_t) and len(gt_line[key]):
            ax.plot(gt_line_t, gt_line[key], linewidth=1.2, color=GT_COLOR, zorder=2)
        if len(gt_band_t) and len(gt_band[key]):
            ax.plot(
                gt_band_t,
                gt_band[key],
                "o",
                color=GT_COLOR,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )
        if len(inf_line_t) and len(inf_line[key]):
            ax.plot(
                inf_line_t,
                inf_line[key],
                linewidth=1.2,
                color=INF_COLOR,
                zorder=2,
            )
        if len(inf_band_t) and len(inf_band[key]):
            ax.plot(
                inf_band_t,
                inf_band[key],
                "o",
                color=INF_COLOR,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

        ax_ppg = ax.twinx()
        ax_ppg.plot(plot_time, gt, linewidth=0.8, color=GT_COLOR, alpha=0.55)
        ax_ppg.plot(plot_time, inf_scaled, linewidth=0.8, color=INF_COLOR, alpha=0.55)
        if len(gt_peak_t):
            ax_ppg.plot(
                gt_peak_t,
                gt_peak_v,
                "o",
                color=GT_COLOR,
                markersize=4,
                markerfacecolor="none",
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
            )
        if len(inf_peak_t):
            ax_ppg.plot(
                inf_peak_t,
                inf_peak_v,
                "o",
                color=INF_COLOR,
                markersize=4,
                markerfacecolor="none",
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
            )
        ax_ppg.set_ylabel("PPG / rPPG")
        if row < n_metrics - 1:
            ax_ppg.set_xticklabels([])
        if row == n_metrics - 1:
            ax.set_xlabel("Time (s)")

    if title:
        fig.suptitle(title, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(hspace=0.5, top=0.94 if title else 0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def maybe_save_inference_plot(
    output_dir: Path,
    stem: str,
    inferred_ppg: np.ndarray,
    fs: int,
    num_frames: int,
    *,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    raw_root: str = "data/raw/mr-nirp",
    title: str = "",
    enabled: bool = True,
) -> str | None:
    """Write a comparison PNG when ground truth is available."""
    if not enabled:
        return None
    try:
        gt_ppg = load_ground_truth_ppg(
            stem=stem,
            input_h5=input_h5,
            input_dir=input_dir,
            raw_root=raw_root,
            num_frames=num_frames,
            fs=fs,
        )
        if gt_ppg is None:
            return None
        plot_path = output_dir / f"{stem}_comparison.png"
        save_inference_comparison_plot(
            plot_path,
            gt_ppg,
            inferred_ppg,
            fs,
            title=title,
        )
        return str(plot_path)
    except ImportError:
        return None
