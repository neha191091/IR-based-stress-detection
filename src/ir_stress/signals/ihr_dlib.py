"""dlib 68-point face landmarks for IR_iHR grid extraction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ir_stress.dataset.face_crop import read_nir_pair

DEFAULT_DLIB_PREDICTOR = Path("data/models/shape_predictor_68_face_landmarks.dat")


def resolve_dlib_predictor(path: str | Path | None) -> Path:
    """Return an existing dlib shape predictor path or raise with install hints."""
    predictor = Path(path) if path is not None else DEFAULT_DLIB_PREDICTOR
    if predictor.is_file():
        return predictor
    raise FileNotFoundError(
        "dlib shape predictor not found at "
        f"{predictor.resolve()}. Download shape_predictor_68_face_landmarks.dat from "
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
        "and extract it, or pass dlib_predictor=/path/to/shape_predictor_68_face_landmarks.dat"
    )


@lru_cache(maxsize=4)
def _load_dlib(predictor_path: str):
    try:
        import dlib
    except ImportError as exc:
        raise ImportError(
            "dlib is required for automatic IR_iHR landmarks. "
            "Install with: uv sync --extra ihr"
        ) from exc

    predictor = Path(predictor_path)
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(predictor))


def _shape_to_np(shape) -> np.ndarray:
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)


def _frame_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a float NIR frame to uint8 grayscale for dlib."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clipped = np.clip(frame, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def _detect_landmarks(
    frame_u8: np.ndarray,
    detector,
    predictor,
    *,
    last_shape: np.ndarray | None,
) -> np.ndarray | None:
    """Detect 68 landmarks on one frame; return None when detection fails."""
    dets = detector(frame_u8, 1)
    if not dets:
        return last_shape
    # Use the largest face (IR_iHR uses the first detection; largest is safer).
    rect = max(dets, key=lambda d: d.width() * d.height())
    return _shape_to_np(predictor(frame_u8, rect))


def extract_dlib_landmarks(
    frame_dir: Path,
    predictor_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load differential NIR frames and detect dlib landmarks per frame.

    Returns ``video`` with shape ``(H, W, T)`` as float64 in [0, 1] and
    ``landmarks`` with shape ``(68, 2, T)``.
    """
    predictor_file = resolve_dlib_predictor(predictor_path)
    detector, predictor = _load_dlib(str(predictor_file.resolve()))

    frames = sorted(frame_dir.glob("*.pgm"))
    num_frames = len(frames) // 2
    if num_frames == 0:
        raise ValueError(f"No PGM frame pairs found in {frame_dir}")

    video_frames: list[np.ndarray] = []
    landmark_frames: list[np.ndarray] = []
    last_shape: np.ndarray | None = None

    for t in range(num_frames):
        frame = read_nir_pair(frames, t)
        frame_u8 = _frame_uint8(frame)
        shape = _detect_landmarks(frame_u8, detector, predictor, last_shape=last_shape)
        if shape is None:
            continue
        last_shape = shape
        video_frames.append(frame.astype(np.float64))
        landmark_frames.append(shape.T)

    if not video_frames:
        raise ValueError(f"dlib could not detect landmarks in any frame under {frame_dir}")

    video = np.stack(video_frames, axis=2)
    landmarks = np.stack(landmark_frames, axis=2)
    return video, landmarks
