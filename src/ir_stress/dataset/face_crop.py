"""Face cropping for NIR PGM frames (OpenFace, YuNet, or center crop)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

FaceCropMode = Literal["openface", "yunet", "center"]

FACE_SIZE = 128  # default; override via FaceCropStream(face_size=...)
BBOX_SCALE = 1.5
Y_PAD_FRAC = 0.2
LANDMARK_SMOOTH = 0.1
CENTER_FRAC = 0.6
_YUNET_MODEL = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")


def read_pgm(path: Path) -> np.ndarray:
    """Read a PGM image as float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read PGM: {path}")
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32) / 255.0


def read_nir_pair(frames: list[Path], video_frame_idx: int) -> np.ndarray:
    """
    Load differential NIR for a 0-based video frame (30 fps).

    Raw PGMs are stored in pairs: even index = LED on, odd = LED off.
    Returns normalized (on - off) clipped to [0, 1].
    """
    on_idx = 2 * video_frame_idx
    off_idx = on_idx + 1
    if off_idx >= len(frames):
        raise IndexError(f"No LED-off pair for video frame {video_frame_idx}")
    diff = read_pgm(frames[on_idx]) - read_pgm(frames[off_idx])
    diff = np.clip(diff, 0.0, None)
    peak = float(diff.max())
    if peak > 0:
        diff = diff / peak
    return diff.astype(np.float32)


def load_landmarks(csv_path: Path) -> pd.DataFrame:
    """Load an OpenFace landmark CSV."""
    return pd.read_csv(csv_path)


def bbox_corners(
    cnt_x: int, cnt_y: int, bbox: int, height: int, width: int
) -> tuple[int, int, int, int]:
    """Return axis-aligned crop corners (x0, y0, x1, y1) clipped to the frame."""
    half = bbox // 2
    x0 = max(cnt_x - half, 0)
    y0 = max(cnt_y - half, 0)
    x1 = min(x0 + bbox, width)
    y1 = min(y0 + bbox, height)
    return x0, y0, x1, y1


def annotate_bbox(
    frame: np.ndarray,
    corners: tuple[int, int, int, int],
    color: float = 1.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bbox on a float grayscale frame for visualization."""
    out = frame.copy()
    vis = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    x0, y0, x1, y1 = corners
    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), int(round(color * 255)), thickness)
    return vis.astype(np.float32) / 255.0


def _landmark_bbox(lm_x: np.ndarray, lm_y: np.ndarray) -> tuple[int, int, int]:
    """Compute face center and bbox size from 68 landmarks."""
    minx, maxx = lm_x.min(), lm_x.max()
    miny, maxy = lm_y.min(), lm_y.max()
    miny = miny - (maxy - miny) * Y_PAD_FRAC
    cnt_x = int(round((minx + maxx) / 2))
    cnt_y = int(round((maxy + miny) / 2))
    bbox = int(round(BBOX_SCALE * (maxy - miny)))
    return cnt_x, cnt_y, max(bbox, 1)


def _box_bbox(x: float, y: float, w: float, h: float) -> tuple[int, int, int]:
    """Convert a detector box to center + square size (Contrast-Phys style)."""
    minx, maxx = x, x + w
    miny, maxy = y, y + h
    miny = miny - (maxy - miny) * Y_PAD_FRAC
    cnt_x = int(round((minx + maxx) / 2))
    cnt_y = int(round((maxy + miny) / 2))
    bbox = int(round(BBOX_SCALE * (maxy - y)))
    return cnt_x, cnt_y, max(bbox, 1)


def _center_bbox(height: int, width: int) -> tuple[int, int, int]:
    bbox = int(round(min(height, width) * CENTER_FRAC))
    return width // 2, height // 2, max(bbox, 1)


def _crop_face(
    frame: np.ndarray, cnt_x: int, cnt_y: int, bbox: int, face_size: int
) -> np.ndarray:
    """Crop and resize a square face region to face_size."""
    half = bbox // 2
    face = frame[
        max(cnt_y - half, 0) : cnt_y - half + bbox,
        max(cnt_x - half, 0) : cnt_x - half + bbox,
    ]
    if face.size == 0:
        return np.zeros((face_size, face_size), dtype=np.float32)
    return cv2.resize(face, (face_size, face_size))


def _frame_landmarks(
    landmark: pd.DataFrame, frame_num: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract 68 landmark x/y arrays for one frame, or None if detection failed."""
    if not landmark[" success"][frame_num]:
        return None
    lm_x = np.array([landmark[f" x_{i}"][frame_num] for i in range(68)])
    lm_y = np.array([landmark[f" y_{i}"][frame_num] for i in range(68)])
    return lm_x, lm_y


def _smooth_bbox(
    state: _CropState, cnt_x: int, cnt_y: int, bbox: int
) -> None:
    if state.bbox is None:
        state.cnt_x, state.cnt_y, state.bbox = cnt_x, cnt_y, bbox
        return
    state.cnt_x = int(round((1 - LANDMARK_SMOOTH) * state.cnt_x + LANDMARK_SMOOTH * cnt_x))
    state.cnt_y = int(round((1 - LANDMARK_SMOOTH) * state.cnt_y + LANDMARK_SMOOTH * cnt_y))
    state.bbox = int(round((1 - LANDMARK_SMOOTH) * state.bbox + LANDMARK_SMOOTH * bbox))


def _yunet_bbox(frame: np.ndarray, detector: cv2.FaceDetectorYN) -> tuple[int, int, int] | None:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    gray = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
    return _box_bbox(float(x), float(y), float(fw), float(fh))


def _detect_bbox_single(
    frame: np.ndarray,
    mode: FaceCropMode,
    *,
    landmark: pd.DataFrame | None = None,
    video_frame_idx: int | None = None,
    detector: cv2.FaceDetectorYN | None = None,
) -> tuple[int, int, int] | None:
    """Return face center and square size for one frame (no temporal smoothing)."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    if mode == "center":
        return _center_bbox(h, w)
    if mode == "openface":
        if landmark is None or video_frame_idx is None:
            raise ValueError("landmark and video_frame_idx required for openface mode")
        lms = _frame_landmarks(landmark, 2 * video_frame_idx)
        return _landmark_bbox(*lms) if lms is not None else None
    if detector is None:
        detector = _yunet_detector()
    return _yunet_bbox(frame, detector)


def face_bbox_corners(
    frame: np.ndarray,
    mode: FaceCropMode,
    *,
    landmark: pd.DataFrame | None = None,
    video_frame_idx: int | None = None,
) -> tuple[int, int, int, int] | None:
    """Return crop corners (x0, y0, x1, y1) for one frame (no temporal smoothing)."""
    detected = _detect_bbox_single(
        frame, mode, landmark=landmark, video_frame_idx=video_frame_idx
    )
    if detected is None:
        return None
    h, w = frame.shape[:2]
    return bbox_corners(*detected, h, w)


def _yunet_detector() -> cv2.FaceDetectorYN:
    if not _YUNET_MODEL.exists():
        raise FileNotFoundError(
            f"YuNet model not found at {_YUNET_MODEL}. "
            "Download face_detection_yunet_2023mar.onnx from opencv_zoo."
        )
    return cv2.FaceDetectorYN.create(str(_YUNET_MODEL), "", (320, 320))


@dataclass
class _CropState:
    """Tracked face bbox state across frames."""

    lm_x: np.ndarray | None = None
    lm_y: np.ndarray | None = None
    bbox: int | None = None
    cnt_x: int = 0
    cnt_y: int = 0


class FaceCropStream:
    """Stream face crops from a directory of PGM frames."""

    def __init__(
        self,
        frame_dir: Path,
        mode: FaceCropMode = "yunet",
        landmark_csv: Path | None = None,
        face_size: int = FACE_SIZE,
    ):
        if mode not in ("openface", "yunet", "center"):
            raise ValueError(f"Unknown face crop mode: {mode}")
        self.face_size = face_size
        self.frames = sorted(frame_dir.glob("*.pgm"))
        self.mode = mode
        self.num_frames = len(self.frames) // 2
        self._state = _CropState()
        self._last_idx = -1
        self._last_crop = np.zeros((face_size, face_size), dtype=np.float32)
        self._frame_shape = (0, 0)

        if mode == "openface":
            if landmark_csv is None:
                raise ValueError("landmark_csv required for openface mode")
            self.landmark = load_landmarks(landmark_csv)
            self.num_frames = min(self.num_frames, len(self.landmark))
        elif mode == "yunet":
            self._detector = _yunet_detector()
        else:
            self.landmark = None
            self._detector = None

    def __len__(self) -> int:
        return self.num_frames

    def _update_bbox(self, frame: np.ndarray, index: int) -> None:
        if self.mode == "center":
            h, w = frame.shape[:2]
            self._frame_shape = (h, w)
            self._state.cnt_x, self._state.cnt_y, self._state.bbox = _center_bbox(h, w)
            return

        if self.mode == "openface":
            lms = _frame_landmarks(self.landmark, 2 * index)
            if lms is None:
                return
            lx, ly = lms
            if self._state.lm_x is None:
                self._state.lm_x, self._state.lm_y = lx, ly
            else:
                self._state.lm_x = (1 - LANDMARK_SMOOTH) * self._state.lm_x + LANDMARK_SMOOTH * lx
                self._state.lm_y = (1 - LANDMARK_SMOOTH) * self._state.lm_y + LANDMARK_SMOOTH * ly
            self._state.cnt_x, self._state.cnt_y, self._state.bbox = _landmark_bbox(
                self._state.lm_x, self._state.lm_y
            )
            self._frame_shape = frame.shape[:2]
            return

        h, w = frame.shape[:2]
        self._frame_shape = (h, w)
        detected = _yunet_bbox(frame, self._detector)
        if detected is not None:
            _smooth_bbox(self._state, *detected)

    def _ensure_index(self, index: int) -> None:
        if index < self._last_idx:
            raise ValueError(
                f"FaceCropStream requires non-decreasing indices; got {index} after {self._last_idx}"
            )
        while self._last_idx < index:
            self._last_idx += 1
            frame = read_nir_pair(self.frames, self._last_idx)
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._update_bbox(frame, self._last_idx)
            if self._state.bbox is None:
                self._last_crop = np.zeros((self.face_size, self.face_size), dtype=np.float32)
            else:
                self._last_crop = _crop_face(
                    frame,
                    self._state.cnt_x,
                    self._state.cnt_y,
                    self._state.bbox,
                    self.face_size,
                )

    def crop(self, index: int) -> np.ndarray:
        """Return a face crop for 0-based video frame index."""
        self._ensure_index(index)
        return self._last_crop

    def bbox_corners(self, index: int) -> tuple[int, int, int, int] | None:
        """Return crop corners (x0, y0, x1, y1) for a frame after processing up to index."""
        self._ensure_index(index)
        if self._state.bbox is None:
            return None
        h, w = self._frame_shape
        return bbox_corners(self._state.cnt_x, self._state.cnt_y, self._state.bbox, h, w)


def stack_face_window(faces: list[np.ndarray]) -> np.ndarray:
    """Stack face crops into model input shape [T, H, W, 1]."""
    return np.stack(faces, axis=0)[..., np.newaxis].astype(np.float32)
