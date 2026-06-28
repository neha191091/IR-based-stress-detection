"""Face-region grid signal extraction for IR_iHR.

Adapted from the IR_iHR reference implementation:
  https://github.com/natalialmg/IR_iHR
  (upstream: IR_iHR/video_utils.py)

Original work: Martinez et al., ICIP 2019.

Included in this codebase but not yet evaluated in our proof of concept
(see docs/REPORT.md).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ir_stress.dataset.face_crop import load_landmarks, read_nir_pair
from ir_stress.signals.ihr_dlib import extract_dlib_landmarks

# 68-point landmark regions (1-based indices in reference code, converted to 0-based).
_REGIONS: dict[str, np.ndarray] = {
    "LFH1": np.array([20, 21, 22, 78, 79, 72, 71, 70]) - 1,
    "RFH1": np.array([78, 23, 24, 25, 75, 74, 73, 79]) - 1,
    "LFH2": np.array([1, 18, 19, 20, 70, 69]) - 1,
    "RFH2": np.array([17, 77, 76, 75, 25, 26, 27]) - 1,
    "LE1": np.array([1, 37, 38, 20, 19, 18]) - 1,
    "RE1": np.array([17, 27, 26, 25, 45, 46]) - 1,
    "LE2": np.array([38, 39, 40, 28, 78, 22, 21, 20]) - 1,
    "RE2": np.array([28, 43, 44, 45, 25, 24, 23, 78]) - 1,
    "LN1": np.array([32, 33, 34, 31, 30, 29, 28, 40]) - 1,
    "RN1": np.array([34, 35, 36, 43, 28, 29, 30, 31]) - 1,
    "LCU1": np.array([81, 32, 40, 41, 42]) - 1,
    "LCU2": np.array([81, 42, 37, 1, 2, 3]) - 1,
    "RCU1": np.array([36, 83, 47, 48, 43]) - 1,
    "RCU2": np.array([83, 15, 16, 17, 46, 47]) - 1,
    "LCD1": np.array([81, 32, 49, 80]) - 1,
    "LCD2": np.array([80, 81, 3, 4]) - 1,
    "RCD1": np.array([36, 83, 82, 55]) - 1,
    "RCD2": np.array([83, 82, 14, 15]) - 1,
    "LM": np.array([49, 50, 51, 52, 34, 33, 32]) - 1,
    "RM": np.array([52, 53, 54, 55, 36, 35, 34]) - 1,
    "LC1": np.array([4, 5, 6, 60, 49]) - 1,
    "RC1": np.array([55, 14, 13, 12, 56]) - 1,
    "LC2": np.array([6, 7, 8, 59, 60]) - 1,
    "RC2": np.array([12, 56, 57, 10, 11]) - 1,
    "CC": np.array([10, 57, 58, 59, 8, 9]) - 1,
}


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(np.float32)


def _boost_landmarks(landmarks_xy: np.ndarray) -> np.ndarray:
    """Add synthetic forehead and cheek landmarks (IR_iHR convention).

    Ported from IR_iHR (github.com/natalialmg/IR_iHR).
    """
    boosted = np.array(landmarks_xy, dtype=np.float64)
    face_length = boosted[:, 1].max() - boosted[:, 1].min()
    forehead = boosted[18:27].copy()
    forehead[:, 1] -= face_length * 0.25

    new_points = np.zeros((6, 2), dtype=np.float64)
    new_points[0] = (boosted[21] + boosted[22]) / 2
    new_points[1] = (forehead[3] + forehead[4]) / 2
    new_points[2] = (boosted[48] + boosted[3]) / 2
    new_points[3] = (boosted[31] + boosted[1]) / 2
    new_points[4] = (boosted[54] + boosted[13]) / 2
    new_points[5] = (boosted[35] + boosted[15]) / 2
    return np.concatenate([boosted, forehead, new_points], axis=0)


def region_mask_from_landmarks(
    frame_shape: tuple[int, int], landmarks_xy: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Build a labelled region mask from 68 OpenFace/dlib landmarks."""
    boosted = _boost_landmarks(landmarks_xy)
    mask = np.zeros(frame_shape, dtype=np.float32)
    labels: list[str] = []
    for label_id, (name, indices) in enumerate(_REGIONS.items(), start=1):
        region = _polygon_mask(frame_shape, boosted[indices])
        mask = np.maximum(label_id * region, mask)
        labels.append(name)
    return mask, labels


def grid_signals_from_video(
    video: np.ndarray,
    landmarks: np.ndarray,
    *,
    grid_size: int = 5,
    min_mask_fraction: float = 0.2,
) -> np.ndarray:
    """
    Extract per-grid-cell mean intensity traces from a face video block.

    Ported from IR_iHR (github.com/natalialmg/IR_iHR).

    Parameters
    ----------
    video
        Grayscale frames with shape ``(height, width, frames)``.
    landmarks
        Landmark coordinates with shape ``(68, 2, frames)``.
    """
    if video.ndim != 3:
        raise ValueError(f"video must be (H, W, T), got {video.shape}")
    if landmarks.shape[:2] != (68, 2):
        raise ValueError(f"landmarks must be (68, 2, T), got {landmarks.shape}")

    mean_frame = video.mean(axis=2)
    mean_landmarks = landmarks.mean(axis=2)
    masks, _ = region_mask_from_landmarks(mean_frame.shape, mean_landmarks.T)

    min_x = int(np.floor(landmarks[:, 0, :].min() - 15))
    max_x = int(np.ceil(landmarks[:, 0, :].max() + 15))
    min_y = int(np.floor(landmarks[:, 1, :].min() - 15))
    max_y = int(np.ceil(landmarks[:, 1, :].max() + 15))

    video_box = video[min_y:max_y, min_x:max_x, :]
    masks_box = masks[min_y:max_y, min_x:max_x]
    roi_mask = (masks_box > 0).astype(np.float32)

    rows, cols, num_frames = video_box.shape
    traces: list[np.ndarray] = []
    for j in range(0, rows, grid_size):
        for i in range(0, cols, grid_size):
            cell_mask = np.zeros((rows, cols), dtype=np.float32)
            cell_mask[j : j + grid_size, i : i + grid_size] = 1.0
            combined = cell_mask * roi_mask
            if combined.sum() < min_mask_fraction * grid_size**2:
                continue
            weighted = video_box * combined[:, :, np.newaxis]
            traces.append(weighted.sum(axis=(0, 1)) / combined.sum())

    if not traces:
        raise ValueError("No valid grid cells found inside the face mask")
    return np.stack(traces, axis=0)


def grid_signals_from_face_clip(
    imgs: np.ndarray, *, grid_size: int = 8
) -> np.ndarray:
    """
    Uniform grid on preprocessed face crops (fallback when landmarks are unavailable).

    Parameters
    ----------
    imgs
        Face crops with shape ``(frames, height, width, channels)``.
    """
    if imgs.ndim != 4:
        raise ValueError(f"imgs must be (T, H, W, C), got {imgs.shape}")
    _, height, width, _ = imgs.shape
    traces: list[np.ndarray] = []
    for row in range(0, height, grid_size):
        for col in range(0, width, grid_size):
            cell = imgs[:, row : row + grid_size, col : col + grid_size, 0]
            traces.append(cell.mean(axis=(1, 2)))
    return np.stack(traces, axis=0)


def load_video_landmarks(
    frame_dir: Path, landmark_csv: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load differential NIR frames and OpenFace landmarks for IR_iHR preprocessing.

    Returns ``video`` with shape ``(H, W, T)`` and ``landmarks`` with shape ``(68, 2, T)``.
    """
    frames = sorted(frame_dir.glob("*.pgm"))
    landmark_df = load_landmarks(landmark_csv)
    num_frames = min(len(frames) // 2, len(landmark_df))

    video_frames: list[np.ndarray] = []
    landmark_frames: list[np.ndarray] = []
    for t in range(num_frames):
        if not landmark_df[" success"][2 * t]:
            continue
        frame = read_nir_pair(frames, t)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lm_x = np.array([landmark_df[f" x_{i}"][2 * t] for i in range(68)])
        lm_y = np.array([landmark_df[f" y_{i}"][2 * t] for i in range(68)])
        video_frames.append(frame)
        landmark_frames.append(np.stack([lm_x, lm_y], axis=0))

    if not video_frames:
        raise ValueError(f"No successful landmark frames in {landmark_csv}")

    video = np.stack(video_frames, axis=2).astype(np.float64)
    landmarks = np.stack(landmark_frames, axis=2).astype(np.float64)
    return video, landmarks


def grid_signals_from_pgm(
    frame_dir: Path,
    landmark_csv: Path | None = None,
    *,
    grid_size: int = 5,
    dlib_predictor: str | Path | None = None,
) -> np.ndarray:
    """
    Build the features × time matrix from raw PGM frames.

    Ported from IR_iHR (github.com/natalialmg/IR_iHR).

    Uses OpenFace landmarks when ``landmark_csv`` is provided; otherwise detects
    landmarks with dlib (IR_iHR reference behaviour).
    """
    if landmark_csv is not None:
        video, landmarks = load_video_landmarks(frame_dir, landmark_csv)
    else:
        video, landmarks = extract_dlib_landmarks(frame_dir, dlib_predictor)
    return grid_signals_from_video(video, landmarks, grid_size=grid_size)
