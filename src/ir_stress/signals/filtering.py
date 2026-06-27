"""Bandpass filtering for physiological signals."""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(
    sig: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1D signal."""
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1D signal."""
    sig = np.reshape(sig, -1).astype(np.float64)
    std = sig.std()
    if std == 0:
        return sig - sig.mean()
    return (sig - sig.mean()) / std
