"""MR-NIRP driving dataset adapter."""

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

from ir_stress.dataset.base import SessionMeta, open_h5_writer
from ir_stress.dataset.face_crop import FaceCropMode, FaceCropStream, FACE_SIZE


def _resample_ppg(pulseox_path: Path, num_frames: int, fps: int) -> np.ndarray:
    """Load and resample pulse-ox PPG waveform to match video frame count."""
    mat = loadmat(pulseox_path)
    if "pulseOxRecord" in mat:
        ppg = mat["pulseOxRecord"].squeeze().astype(np.float64)
    elif "data" in mat:
        ppg = mat["data"].squeeze().astype(np.float64)
    elif "val" in mat:
        ppg = mat["val"].squeeze().astype(np.float64)
    else:
        raise KeyError(f"No PPG field in {pulseox_path}")
    return resample(ppg, num_frames).astype(np.float32)


class MRNIRPDrivingAdapter:
    """Ingestion pipeline for the MR-NIRP driving dataset."""

    def discover_sessions(
        self, raw_root: Path, wavelengths: list[int] | None = None
    ) -> list[SessionMeta]:
        """Find NIR sessions under subject directories."""
        if wavelengths is None:
            wavelengths = [940]
        sessions = []
        for subject_dir in sorted(raw_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            match = re.search(r"(\d+)", subject_dir.name, re.IGNORECASE)
            if not match:
                continue
            subject_id = int(match.group(1))
            for clip_dir in sorted(subject_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                if not any(str(w) in clip_dir.name for w in wavelengths):
                    continue
                pulseox_candidates = list(clip_dir.glob("**/pulseOx.mat"))
                if not pulseox_candidates:
                    continue
                pulseox = pulseox_candidates[0]
                pgms = list(clip_dir.glob("**/*.pgm"))
                if not pgms:
                    continue
                nir_dir = pgms[0].parent
                wavelength = 940 if "940" in clip_dir.name else 975
                sessions.append(
                    SessionMeta(
                        subject_id=subject_id,
                        condition=clip_dir.name,
                        wavelength=wavelength,
                        nir_dir=nir_dir,
                        pulseox_path=pulseox,
                        landmark_csv=None,
                    )
                )
        return sessions

    def train_test_split(
        self, sessions: list[SessionMeta], val_subjects: list[int]
    ) -> tuple[list[SessionMeta], list[SessionMeta]]:
        """Hold out subjects for validation/test."""
        train, test = [], []
        for s in sessions:
            (test if s.subject_id in val_subjects else train).append(s)
        return train, test

    def preprocess_session(
        self,
        session: SessionMeta,
        out_dir: Path,
        landmarks_dir: Path | None = None,
        face_crop_mode: FaceCropMode = "yunet",
        face_size: int = FACE_SIZE,
    ) -> Path:
        """
        Crop faces from NIR PGM frames and write an H5 file.

        Frames are written one at a time to avoid loading the full clip into RAM.
        OpenFace mode requires landmarks_dir/<condition>.csv.
        """
        landmark_csv = None
        if face_crop_mode == "openface":
            csv_path = session.landmark_csv
            if csv_path is None and landmarks_dir is not None:
                csv_path = landmarks_dir / f"{session.condition}.csv"
            if csv_path is None or not Path(csv_path).exists():
                raise FileNotFoundError(
                    f"Landmark CSV required for {session.condition}. "
                    "Run OpenFace and pass --landmarks-dir."
                )
            landmark_csv = Path(csv_path)

        stream = FaceCropStream(
            session.nir_dir, mode=face_crop_mode, landmark_csv=landmark_csv, face_size=face_size
        )
        out_path = out_dir / f"subject{session.subject_id}_{session.condition}.h5"
        h5_file, imgs_ds = open_h5_writer(out_path, len(stream), face_size=face_size)

        for i in range(len(stream)):
            imgs_ds[i, :, :, 0] = stream.crop(i)

        ppg = _resample_ppg(session.pulseox_path, len(stream), session.fps)
        h5_file.create_dataset("ppg", data=ppg)
        h5_file.close()
        return out_path
