"""Plot WESAD baseline/stress windows: BVP, ECG, labels, and stress indicators."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from hydra.core.config_store import ConfigStore
from matplotlib.gridspec import GridSpec

from ir_stress.signals.stress_indicators import extract_ibi_with_beat_times, stress_indicators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data/WESAD"
OUTPUT_DIR = PROJECT_ROOT / "results"

LABEL_FS = 700
BVP_FS = 64
METRIC_WINDOW_SEC = 10.0
METRIC_HOP_SEC = 0.5
MIN_IBI_SEC = 0.35
MAX_IBI_SEC = 1.50
MIN_BEATS = 5

CONDITIONS: tuple[tuple[int, str], ...] = ((1, "baseline"), (2, "stress"))
LABEL_COLORS = {1: "#4C72B0", 2: "#C44E52"}

STRESS_SERIES: tuple[tuple[str, str], ...] = (
    ("mean_ibi_seconds", "Mean IBI (s)"),
    ("sdnn_ms", "SDNN (ms)"),
    ("rmssd_ms", "RMSSD (ms)"),
    ("pnn50", "pNN50 (%)"),
    ("baevsky_si", "Baevsky SI"),
)


@dataclass
class RunConfig:
    start_time: float = 0.0
    window_sec: float = 120.0


@dataclass
class ConditionWindow:
    label_id: int
    name: str
    bvp: np.ndarray
    ecg: np.ndarray
    bvp_peaks: np.ndarray
    ecg_peaks: np.ndarray
    bvp_metrics: tuple[np.ndarray, dict[str, np.ndarray]]
    ecg_metrics: tuple[np.ndarray, dict[str, np.ndarray]]


def list_subjects() -> list[int]:
    subjects: list[int] = []
    for folder in sorted(DATA_ROOT.glob("S*")):
        if folder.is_dir() and (folder / f"{folder.name}.pkl").exists():
            subjects.append(int(folder.name[1:]))
    return subjects


def load_subject(subject_id: int) -> dict:
    with (DATA_ROOT / f"S{subject_id}" / f"S{subject_id}.pkl").open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def label_spans(labels: np.ndarray, fs: float, label_id: int) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if len(labels) == 0:
        return spans
    current = int(labels[0])
    start_idx = 0
    for idx in range(1, len(labels)):
        value = int(labels[idx])
        if value == current:
            continue
        if current == label_id:
            spans.append((start_idx / fs, idx / fs))
        current = value
        start_idx = idx
    if current == label_id:
        spans.append((start_idx / fs, len(labels) / fs))
    return spans


def extract_condition_window(
    signal: np.ndarray,
    labels: np.ndarray,
    fs: float,
    label_id: int,
    name: str,
    start_time: float,
    window_sec: float,
) -> np.ndarray:
    spans = label_spans(labels, fs, label_id)
    if not spans:
        raise ValueError(f"no {name} block found")
    block_start, block_end = spans[0]
    win_start = block_start + start_time
    win_end = win_start + window_sec
    if win_start < block_start or win_end > block_end:
        raise ValueError(
            f"{name} window [{win_start:.1f}, {win_end:.1f}) s "
            f"outside block [{block_start:.1f}, {block_end:.1f}) s"
        )
    start_idx = int(round(win_start * fs))
    end_idx = int(round(win_end * fs))
    return signal[start_idx:end_idx]


def filter_ibis(ibi: np.ndarray, beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (ibi >= MIN_IBI_SEC) & (ibi <= MAX_IBI_SEC)
    return ibi[mask], beat_times[mask]


def detect_bvp_peaks(bvp: np.ndarray, time_sec: np.ndarray) -> np.ndarray:
    _, _, peaks = extract_ibi_with_beat_times(bvp, BVP_FS, time_sec)
    return peaks


def detect_ecg_peaks(ecg: np.ndarray) -> np.ndarray:
    cleaned = nk.ecg_clean(ecg.copy(), sampling_rate=LABEL_FS)
    return np.asarray(nk.ecg_findpeaks(cleaned, sampling_rate=LABEL_FS)["ECG_R_Peaks"], dtype=int)


def ibi_from_bvp(bvp: np.ndarray, time_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ibi, beat_times, _ = extract_ibi_with_beat_times(bvp, BVP_FS, time_sec)
    return filter_ibis(ibi, beat_times)


def ibi_from_ecg(ecg: np.ndarray, time_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peaks = detect_ecg_peaks(ecg)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    ibi = (peaks[1:] - peaks[:-1]) / LABEL_FS
    beat_times = time_sec[peaks[1:]]
    return filter_ibis(ibi, beat_times)


def windowed_stress(
    signal: np.ndarray,
    time_sec: np.ndarray,
    fs: float,
    ibi_fn,
    duration: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    keys = [key for key, _ in STRESS_SERIES]
    series = {key: [] for key in keys}
    centers: list[float] = []

    ibi, beat_times = ibi_fn(signal, time_sec)
    if len(ibi) < MIN_BEATS:
        return np.asarray(centers), {key: np.asarray(vals) for key, vals in series.items()}

    window_start = 0.0
    while window_start + METRIC_WINDOW_SEC <= duration + 1e-9:
        window_end = window_start + METRIC_WINDOW_SEC
        in_window = (beat_times >= window_start) & (beat_times < window_end)
        window_ibi = ibi[in_window]
        if len(window_ibi) >= MIN_BEATS:
            metrics = stress_indicators(window_ibi)
            for key in keys:
                series[key].append(float(metrics[key]))
            centers.append((window_start + window_end) / 2.0)
        window_start += METRIC_HOP_SEC

    return np.asarray(centers), {key: np.asarray(vals) for key, vals in series.items()}


def prepare_condition_windows(
    bvp: np.ndarray,
    ecg: np.ndarray,
    chest_labels: np.ndarray,
    start_time: float,
    window_sec: float,
) -> list[ConditionWindow]:
    bvp_labels = chest_labels[
        np.clip((np.arange(len(bvp)) * LABEL_FS / BVP_FS).astype(int), 0, len(chest_labels) - 1)
    ]
    windows: list[ConditionWindow] = []
    for label_id, name in CONDITIONS:
        seg_bvp = extract_condition_window(
            bvp, bvp_labels, BVP_FS, label_id, name, start_time, window_sec
        )
        seg_ecg = extract_condition_window(
            ecg, chest_labels, LABEL_FS, label_id, name, start_time, window_sec
        )
        bvp_time = np.arange(len(seg_bvp)) / BVP_FS
        ecg_time = np.arange(len(seg_ecg)) / LABEL_FS
        windows.append(
            ConditionWindow(
                label_id=label_id,
                name=name,
                bvp=seg_bvp,
                ecg=seg_ecg,
                bvp_peaks=detect_bvp_peaks(seg_bvp, bvp_time),
                ecg_peaks=detect_ecg_peaks(seg_ecg),
                bvp_metrics=windowed_stress(seg_bvp, bvp_time, BVP_FS, ibi_from_bvp, window_sec),
                ecg_metrics=windowed_stress(seg_ecg, ecg_time, LABEL_FS, ibi_from_ecg, window_sec),
            )
        )
    return windows


def plot_subject(
    subject_id: int,
    windows: list[ConditionWindow],
    window_sec: float,
    start_time: float,
    output_path: Path,
) -> None:
    n_metrics = len(STRESS_SERIES)
    n_rows = 2 + n_metrics
    fig = plt.figure(figsize=(14, 2.0 + 1.1 * n_rows))
    gs = GridSpec(
        n_rows,
        2,
        figure=fig,
        height_ratios=[2.0, 1.2] + [1.0] * n_metrics,
        hspace=0.45,
        wspace=0.25,
    )
    fig.suptitle(
        f"WESAD S{subject_id} | {window_sec:.0f} s from t={start_time:.0f} s in each block",
        y=0.995,
        fontsize=11,
    )

    row_axes: list[list[plt.Axes | None]] = [[None, None] for _ in range(n_rows)]

    for col, cond in enumerate(windows):
        bvp_time = np.arange(len(cond.bvp)) / BVP_FS
        ecg_time = np.arange(len(cond.ecg)) / LABEL_FS
        bvp_centers, bvp_series = cond.bvp_metrics
        ecg_centers, ecg_series = cond.ecg_metrics
        tint = mcolors.to_rgba(LABEL_COLORS[cond.label_id], alpha=0.08)
        tint_light = mcolors.to_rgba(LABEL_COLORS[cond.label_id], alpha=0.05)

        ax_bvp = fig.add_subplot(gs[0, col], sharey=row_axes[0][0])
        row_axes[0][col] = ax_bvp
        ax_bvp.set_facecolor(tint)
        ax_bvp.plot(bvp_time, cond.bvp, color="crimson", linewidth=0.7, zorder=2)
        if len(cond.bvp_peaks):
            ax_bvp.plot(
                bvp_time[cond.bvp_peaks],
                cond.bvp[cond.bvp_peaks],
                "o",
                color="crimson",
                markersize=4,
                markerfacecolor="none",
                markeredgewidth=1.0,
                linestyle="none",
                zorder=3,
            )
        ax_bvp.set_ylabel("BVP" if col == 0 else "")
        ax_bvp.set_title(cond.name.capitalize(), color=LABEL_COLORS[cond.label_id], fontweight="bold")
        ax_bvp.grid(True, alpha=0.3)
        ax_bvp.set_xlim(0, window_sec)
        plt.setp(ax_bvp.get_xticklabels(), visible=False)
        if col == 1:
            plt.setp(ax_bvp.get_yticklabels(), visible=False)

        ax_ecg = fig.add_subplot(gs[1, col], sharex=ax_bvp, sharey=row_axes[1][0])
        row_axes[1][col] = ax_ecg
        ax_ecg.set_facecolor(tint)
        step = max(1, len(cond.ecg) // max(len(cond.bvp), 1))
        ax_ecg.plot(ecg_time[::step], cond.ecg[::step], color="0.15", linewidth=0.5, zorder=2)
        if len(cond.ecg_peaks):
            ax_ecg.plot(
                ecg_time[cond.ecg_peaks],
                cond.ecg[cond.ecg_peaks],
                "o",
                color="0.15",
                markersize=3,
                markerfacecolor="none",
                markeredgewidth=0.9,
                linestyle="none",
                zorder=3,
            )
        ax_ecg.set_ylabel("Chest ECG" if col == 0 else "")
        ax_ecg.grid(True, alpha=0.3)
        plt.setp(ax_ecg.get_xticklabels(), visible=False)
        if col == 1:
            plt.setp(ax_ecg.get_yticklabels(), visible=False)

        for row, (key, ylabel) in enumerate(STRESS_SERIES, start=2):
            ax = fig.add_subplot(gs[row, col], sharex=ax_bvp, sharey=row_axes[row][0])
            row_axes[row][col] = ax
            ax.set_facecolor(tint_light)
            if len(bvp_centers):
                ax.plot(bvp_centers, bvp_series[key], color="tab:blue", linewidth=1.2, label="BVP")
            if len(ecg_centers):
                ax.plot(
                    ecg_centers,
                    ecg_series[key],
                    color="black",
                    linewidth=1.0,
                    alpha=0.85,
                    label="ECG",
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if row == 2 and col == 1:
                ax.legend(loc="upper right", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Time in condition (s)")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if col == 1:
                plt.setp(ax.get_yticklabels(), visible=False)

    for axes in row_axes:
        valid = [ax for ax in axes if ax is not None]
        if len(valid) < 2:
            continue
        y_min = min(ax.get_ylim()[0] for ax in valid)
        y_max = max(ax.get_ylim()[1] for ax in valid)
        for ax in valid:
            ax.set_ylim(y_min, y_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def run_subject(subject_id: int, start_time: float, window_sec: float) -> Path:
    data = load_subject(subject_id)
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float64).squeeze()
    ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=np.float64).squeeze()
    chest_labels = np.asarray(data["label"], dtype=np.int32).squeeze()

    windows = prepare_condition_windows(bvp, ecg, chest_labels, start_time, window_sec)
    output_path = OUTPUT_DIR / f"wesad_S{subject_id}_baseline_stress_{int(start_time)}s_{int(window_sec)}s.png"
    plot_subject(subject_id, windows, window_sec, start_time, output_path)
    return output_path


@hydra.main(version_base=None, config_path=None, config_name="run_config")
def main(cfg: RunConfig) -> None:
    if cfg.window_sec <= 0:
        raise ValueError("window_sec must be positive")

    saved: list[Path] = []
    skipped: list[str] = []
    for subject_id in list_subjects():
        try:
            path = run_subject(subject_id, cfg.start_time, cfg.window_sec)
            saved.append(path)
            print(f"S{subject_id}: saved {path}")
        except (ValueError, FileNotFoundError) as exc:
            skipped.append(f"S{subject_id}: {exc}")
            print(f"S{subject_id}: skipped ({exc})")

    print(f"\nDone: {len(saved)} plots saved to {OUTPUT_DIR}, {len(skipped)} skipped")
    for line in skipped:
        print(f"  {line}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="run_config", node=RunConfig)
    main()
