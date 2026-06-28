"""Core IR_iHR signal extraction (optimal SVD + synchrosqueezing iHR).

Based on N. Martinez et al., ICIP 2019 and the reference implementation:
https://github.com/natalialmg/IR_iHR
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal
from scipy.signal import butter

from ir_stress.signals import sq_stft_utils as sq


def butter_bandpass_coeffs(lowcut: float, highcut: float, fs: float, order: int = 5):
    """Return Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")


def butter_highpass_coeffs(highcut: float, fs: float, order: int):
    """Return Butterworth highpass filter coefficients."""
    nyq = 0.5 * fs
    high = highcut / nyq
    return butter(order, high, btype="highpass")


def optimal_svd(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal shrinkage SVD (Gavish & Donoho, 2017)."""
    U, s, V = np.linalg.svd(Y, full_matrices=False)
    m, n = Y.shape
    beta = m / n

    y_med = np.median(s)
    beta_m = (1 - np.sqrt(beta)) ** 2
    beta_p = (1 + np.sqrt(beta)) ** 2

    t_array = np.linspace(beta_m, beta_p, 100_000)
    dt = np.diff(t_array)[0]

    def f(t):
        return np.sqrt((beta_p - t) * (t - beta_m)) / (2 * np.pi * t * beta)

    F = lambda t: np.cumsum(f(t) * dt)
    mu_beta = t_array[np.argmin((F(t_array) - 0.5) ** 2)]
    sigma_hat = y_med / np.sqrt(n * mu_beta)

    def eta(y, beta_):
        mask = y >= (1 + np.sqrt(beta_))
        aux_sqrt = np.sqrt((y[mask] ** 2 - beta_ - 1) ** 2 - 4 * beta_)
        aux = np.zeros(y.shape)
        aux[mask] = aux_sqrt / y[mask]
        return mask * aux

    def eta_sigma(y, beta_, sigma):
        return sigma * eta(y / sigma, beta_)

    s_eta = eta_sigma(s, beta, sigma_hat * np.sqrt(n))
    keep = s_eta > 0
    return U[:, keep], s_eta[keep], V[keep, :]


def curve_extractor(
    spectrogram: np.ndarray, lamb: float, guess: int | None = None
) -> np.ndarray:
    """Track dominant frequency index over time (Cicone & Wu, 2017)."""
    eps = 1e-8
    E = np.abs(np.asarray(spectrogram)).T
    E /= np.sum(E)
    E = np.log(E + eps)

    m, n = E.shape
    curve = np.zeros(m)
    curve[0] = np.argmax(E[0, :]) if guess is None else guess

    freq_indexes = np.arange(n)
    for i in np.arange(m)[1:]:
        penalty = (curve[i - 1] - freq_indexes) ** 2
        curve[i] = np.argmax(E[i, :] - lamb * penalty)
    return curve


def quality_process(
    signals: np.ndarray,
    prior_bpm: float,
    fs: float,
    *,
    window: int = 301,
    f_low: float = 0.4,
    f_high: float = 5.0,
    fp_display_low: float = 0.7,
    fp_display_high: float = 5.0,
) -> np.ndarray:
    """Per-channel quality index from synchrosqueezed spectral power."""
    nv = signals.shape[0]
    quality = np.zeros(nv)
    b, a = butter_bandpass_coeffs(f_low, f_high, fs, order=5)

    for i in range(nv):
        filtered = scipy.signal.filtfilt(b, a, signals[i, :])
        stft_v, f_v, _, _ = sq.SST_helper(
            filtered, fs, f_high / fs, f_low / fs, windowLength=window
        )
        f_v = f_v * fs

        prior_f = prior_bpm / 60.0
        idx_prior = (f_v > prior_f * 0.75) & (f_v < prior_f * 1.25)
        idx_band = (f_v > 0.25 * prior_f) & (f_v < 2.0 * prior_f)

        power = np.abs(stft_v).sum(1)
        band_power = power[idx_band].sum()
        quality[i] = power[idx_prior].sum() / band_power if band_power > 0 else 0.0

    return quality


@dataclass
class IHRExtractionResult:
    """Output of the IR_iHR classical extraction pipeline."""

    ppg: np.ndarray
    ihr_bpm: np.ndarray
    ihr_time: np.ndarray
    quality: float
    raw_component: np.ndarray


def extract_ihr_from_grid(
    Y: np.ndarray,
    fs: float,
    prior_bpm: float = 70.0,
    *,
    window: int = 301,
    f_low: float = 0.4,
    f_high: float = 5.0,
    max_eigenvectors: int = 40,
    curve_lambda: float = 0.05,
) -> IHRExtractionResult:
    """
    Extract a contactless PPG waveform and instantaneous HR from a grid signal matrix.

    Parameters
    ----------
    Y
        Spatial grid signals with shape ``(channels, frames)``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y with shape (channels, frames), got {Y.shape}")
    if Y.shape[1] < window:
        raise ValueError(
            f"Need at least {window} frames for IR_iHR (got {Y.shape[1]} at fs={fs})"
        )

    b_hp, a_hp = butter_highpass_coeffs(f_low, fs, order=5)
    Y_hat = scipy.signal.filtfilt(b_hp, a_hp, Y, axis=1)

    _, _, V = optimal_svd(Y_hat)
    nv = min(max_eigenvectors, V.shape[0])
    quality = quality_process(
        V[:nv],
        prior_bpm,
        fs,
        window=window,
        f_low=f_low,
        f_high=f_high,
    )

    V_sorted = V[np.argsort(quality)[::-1]]
    V_cum = np.cumsum(V_sorted, axis=0)
    quality_cum = quality_process(
        V_cum,
        prior_bpm,
        fs,
        window=window,
        f_low=f_low,
        f_high=f_high,
    )
    best_idx = int(np.argmax(quality_cum))
    component = V_cum[best_idx]

    b_bp, a_bp = butter_bandpass_coeffs(f_low, f_high, fs, order=5)
    filtered = scipy.signal.filtfilt(b_bp, a_bp, component)

    stft_v, f_v, _, _ = sq.SST_helper(
        filtered, fs, f_high / fs, f_low / fs, windowLength=window
    )
    f_hz = f_v * fs
    guess = int(np.argmin(np.abs(f_hz * 60.0 - prior_bpm)))
    curve_idx = curve_extractor(np.abs(stft_v), curve_lambda, guess)
    ihr_bpm = f_hz[curve_idx.astype(int)] * 60.0
    ihr_time = np.arange(ihr_bpm.size) / fs

    return IHRExtractionResult(
        ppg=filtered.astype(np.float64),
        ihr_bpm=ihr_bpm.astype(np.float64),
        ihr_time=ihr_time.astype(np.float64),
        quality=float(quality_cum[best_idx]),
        raw_component=component.astype(np.float64),
    )
