"""Evaluation metrics for rPPG signals."""

import numpy as np
from scipy.stats import pearsonr

from ir_stress.signals.filtering import butter_bandpass, normalize


def pearson_r(pred: np.ndarray, gt: np.ndarray, fs: float) -> float:
    """Pearson correlation after bandpass filtering and z-score normalization."""
    pred_f = normalize(butter_bandpass(pred, 0.6, 4.0, fs))
    gt_f = normalize(butter_bandpass(gt, 0.6, 4.0, fs))
    n = min(len(pred_f), len(gt_f))
    if n < 2:
        return float("nan")
    r, _ = pearsonr(pred_f[:n], gt_f[:n])
    return float(r)


def mse(pred: np.ndarray, gt: np.ndarray, fs: float) -> float:
    """Mean squared error after bandpass filtering and z-score normalization."""
    pred_f = normalize(butter_bandpass(pred, 0.6, 4.0, fs))
    gt_f = normalize(butter_bandpass(gt, 0.6, 4.0, fs))
    n = min(len(pred_f), len(gt_f))
    if n < 1:
        return float("nan")
    return float(np.mean((pred_f[:n] - gt_f[:n]) ** 2))
