"""Dataset adapter interface and H5 helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import cv2


@dataclass
class SessionMeta:
    """Metadata for one recording session."""

    subject_id: int
    condition: str
    wavelength: int
    nir_dir: Path
    pulseox_path: Path
    landmark_csv: Path | None
    fps: int = 30


class DatasetAdapter(Protocol):
    """Interface for raw-dataset ingestion pipelines."""

    def discover_sessions(self, raw_root: Path) -> list[SessionMeta]:
        """Find all recordable sessions under raw_root."""

    def train_test_split(
        self, sessions: list[SessionMeta], val_subjects: list[int]
    ) -> tuple[list[SessionMeta], list[SessionMeta]]:
        """Split sessions by subject id."""

    def preprocess_session(self, session: SessionMeta, out_dir: Path) -> Path:
        """Preprocess one session and write an H5 file; returns output path."""


def resize_face_clip(imgs: np.ndarray, face_size: int) -> np.ndarray:
    """Resize a [T, H, W, C] clip to square face_size (no-op when H and W already match)."""
    h, w = imgs.shape[1], imgs.shape[2]
    if h == face_size and w == face_size:
        return imgs
    out = np.empty((imgs.shape[0], face_size, face_size, imgs.shape[3]), dtype=np.float32)
    for t in range(imgs.shape[0]):
        out[t, :, :, 0] = cv2.resize(imgs[t, :, :, 0], (face_size, face_size))
    return out


def _ppg_dataset(h5_file: h5py.File) -> h5py.Dataset:
    """Return the per-frame pulse-ox PPG dataset (supports legacy ``bpm`` key)."""
    if "ppg" in h5_file:
        return h5_file["ppg"]
    if "bpm" in h5_file:
        return h5_file["bpm"]
    raise KeyError('H5 file must contain dataset "ppg" (legacy: "bpm")')


def write_h5(out_path: Path, imgs: np.ndarray, ppg: np.ndarray) -> None:
    """Write imgs [N,H,W,C] and ppg [N] to an H5 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("imgs", data=imgs, dtype="float32", compression="gzip")
        f.create_dataset("ppg", data=ppg.astype("float32"))


def open_h5_writer(
    out_path: Path, num_frames: int, face_size: int = 128, channels: int = 1
) -> tuple[h5py.File, h5py.Dataset]:
    """
    Open an H5 file for frame-by-frame writes.

    Caller must close the returned file after writing all frames and the ppg dataset.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(out_path, "w")
    imgs = f.create_dataset(
        "imgs",
        shape=(num_frames, face_size, face_size, channels),
        dtype="float32",
        chunks=(1, face_size, face_size, channels),
        compression="gzip",
    )
    return f, imgs


def h5_num_frames(path: str | Path) -> int:
    """Return the number of aligned frames in an H5 clip."""
    with h5py.File(path, "r") as f:
        return min(f["imgs"].shape[0], _ppg_dataset(f).shape[0])


def iter_h5_imgs_windows(
    path: str | Path, window_frames: int, face_size: int | None = None
) -> tuple[int, np.ndarray]:
    """Yield (start_index, imgs_window) without loading the full clip."""
    with h5py.File(path, "r") as f:
        n = f["imgs"].shape[0]
        for start in range(0, n, window_frames):
            end = min(start + window_frames, n)
            imgs = f["imgs"][start:end]
            if face_size is not None:
                imgs = resize_face_clip(imgs.astype(np.float32), face_size)
            yield start, imgs


def read_h5_window(
    path: str | Path, start: int, end: int, face_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read aligned imgs and ppg slices [start:end] from an H5 clip."""
    with h5py.File(path, "r") as f:
        imgs = f["imgs"][start:end]
        ppg = _ppg_dataset(f)[start:end]
    if face_size is not None:
        imgs = resize_face_clip(imgs.astype(np.float32), face_size)
    return imgs, ppg


def list_h5_paths(h5_dir: Path) -> list[str]:
    """Return sorted paths to all .h5 files in a directory."""
    return sorted(str(p) for p in h5_dir.glob("*.h5"))
