"""Stress indicators from PPG/rPPG signals."""

import numpy as np
import neurokit2 as nk


def extract_ibi_with_beat_times(
    sig: np.ndarray, fs: float, time_sec: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract IBIs (seconds), ending-beat timestamps, and peak sample indices."""
    cleaned = nk.ppg_clean(sig.copy(), sampling_rate=fs, method="elgendi")
    peaks = np.asarray(nk.ppg_findpeaks(cleaned, sampling_rate=fs)["PPG_Peaks"], dtype=int)
    if len(peaks) < 2:
        return np.array([]), np.array([]), np.array([], dtype=int)
    ibi = (peaks[1:] - peaks[:-1]) / fs
    if time_sec is not None:
        beat_times = time_sec[peaks[1:]]
    else:
        beat_times = peaks[1:] / fs
    return ibi, beat_times, peaks


def extract_ibi(sig: np.ndarray, fs: float) -> np.ndarray:
    """Extract inter-beat intervals (seconds) from a PPG/rPPG signal."""
    ibi, _, _ = extract_ibi_with_beat_times(sig, fs)
    return ibi


def baevsky_stress_index(ibi: np.ndarray, bin_width_seconds: float = 0.05) -> dict[str, float]:
    """
    Compute Baevsky Stress Index from inter-beat intervals.

    Measures sympathetic nervous system (SNS) activation from how concentrated
    beat intervals are around the mode versus overall spread.

    Stress correlation: **higher SI → more sympathetic tone / stress** (often
    >500 under strong load); **lower SI → calmer autonomic balance** (often
    ~50–150 at rest).

    SI = AMo / (2 * Mo * MxDMn)
    where Mo is the mode (histogram bin center with the most IBIs), AMo is the
    mode amplitude (% of IBIs in that bin), and MxDMn is the variation range.
    """
    if len(ibi) < 2:
        return {"baevsky_si": float("nan"), "baevsky_si_sqrt": float("nan")}

    mn = float(ibi.min())
    mx = float(ibi.max())
    mxdmn = mx - mn
    if mxdmn == 0:
        return {"baevsky_si": float("nan"), "baevsky_si_sqrt": float("nan")}

    edges = np.arange(mn, mx + bin_width_seconds, bin_width_seconds)
    counts, edges = np.histogram(ibi, bins=edges)
    mode_idx = int(np.argmax(counts))
    mo = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)
    if mo == 0:
        return {"baevsky_si": float("nan"), "baevsky_si_sqrt": float("nan")}

    amo = 100.0 * counts[mode_idx] / len(ibi)

    si = amo / (2.0 * mo * mxdmn)
    return {"baevsky_si": si, "baevsky_si_sqrt": float(np.sqrt(si))}


def sdnn(ibi: np.ndarray) -> float:
    """
    Standard deviation of inter-beat intervals (SDNN).

    Measures overall heart rate variability from all IBIs in the window,
    reflecting combined sympathetic and parasympathetic influence.

    Stress correlation: **higher SDNN → greater variability, generally
    associated with healthier autonomic flexibility and lower stress at rest**;
    **lower SDNN → reduced HRV, often seen with chronic stress, fatigue, or
    illness**.

    Returns SDNN in milliseconds.
    """
    if len(ibi) < 2:
        return float("nan")
    return float(np.std(ibi, ddof=1) * 1000.0)


def rmssd(ibi: np.ndarray) -> float:
    """
    Root mean square of successive IBI differences (RMSSD).

    Measures short-term, beat-to-beat variability driven mainly by
    parasympathetic (vagal) modulation.

    Stress correlation: **higher RMSSD → stronger vagal tone and recovery**;
    **lower RMSSD → sympathetic dominance and acute or chronic stress**.

    Returns RMSSD in milliseconds.
    """
    if len(ibi) < 2:
        return float("nan")
    diff = np.diff(ibi)
    return float(np.sqrt(np.mean(diff**2)) * 1000.0)


def pnn50(ibi: np.ndarray, threshold_seconds: float = 0.05) -> float:
    """
    Percentage of successive IBIs differing by more than 50 ms (pNN50).

    Measures how often consecutive beats change substantially, another
    parasympathetic-sensitive HRV index.

    Stress correlation: **higher pNN50 → more beat-to-beat flexibility and
    lower stress**; **lower pNN50 → rigid rhythm and higher stress/arousal**.

    Returns pNN50 as a percentage in [0, 100].
    """
    if len(ibi) < 2:
        return float("nan")
    diff_ms = np.abs(np.diff(ibi)) * 1000.0
    return float(100.0 * np.sum(diff_ms > threshold_seconds * 1000.0) / len(diff_ms))


def mean_hr(ibi: np.ndarray) -> float:
    """
    Mean heart rate from inter-beat intervals.

    Measures average cardiac rate over the IBI window.

    Stress correlation: **higher mean HR → greater arousal and sympathetic
    drive** (context-dependent); **lower mean HR → rest/recovery**, though very
    low values can also reflect poor signal quality or bradycardia.

    Returns heart rate in beats per minute (BPM).
    """
    if len(ibi) == 0:
        return float("nan")
    return float(60.0 / np.mean(ibi))


def time_domain_hrv(ibi: np.ndarray) -> dict[str, float]:
    """
    Compute common time-domain HRV metrics from inter-beat intervals.

    Returns SDNN, RMSSD, pNN50, and mean HR. IBIs must be in seconds.
    """
    return {
        "sdnn_ms": sdnn(ibi),
        "rmssd_ms": rmssd(ibi),
        "pnn50": pnn50(ibi),
        "mean_hr_bpm": mean_hr(ibi),
    }


def stress_indicators(ibi: np.ndarray, bin_width_seconds: float = 0.05) -> dict[str, float]:
    """
    Compute Baevsky SI and time-domain HRV metrics from inter-beat intervals.

    IBIs must be in seconds.
    """
    if len(ibi) == 0:
        mean_ibi = float("nan")
    else:
        mean_ibi = float(np.mean(ibi))
    return {
        "mean_ibi_seconds": mean_ibi,
        **time_domain_hrv(ibi),
        **baevsky_stress_index(ibi, bin_width_seconds=bin_width_seconds),
    }
