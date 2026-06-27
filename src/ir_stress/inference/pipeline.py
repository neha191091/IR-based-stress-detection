"""End-to-end inference: rPPG extraction and Baevsky Stress Index."""

import json
from pathlib import Path

import numpy as np
import torch

from ir_stress.config import InferenceConfig, TrainConfig
from ir_stress.data.base import h5_num_frames, iter_h5_imgs_windows
from ir_stress.data.face_crop import FaceCropStream, annotate_bbox, stack_face_window
from ir_stress.models.model import build_model, RppgModel
from ir_stress.models.predict import predict_window
from ir_stress.signals.filtering import butter_bandpass
from ir_stress.signals.stress_indicators import baevsky_stress_index, extract_ibi


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(
    config: InferenceConfig | TrainConfig, checkpoint: Path, device: torch.device
) -> RppgModel:
    """Load a trained RppgModel from checkpoint."""
    model = build_model(config.model, spatial_dim=config.spatial_dim, in_ch=config.in_ch)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def _predict_h5_clip(
    model: RppgModel,
    path: Path,
    device: torch.device,
    fs: int,
    window_sec: int,
    face_size: int = 128,
) -> np.ndarray:
    """Predict rPPG window-by-window from a preprocessed H5 clip."""
    window_frames = fs * window_sec
    chunks = []
    for _, imgs in iter_h5_imgs_windows(path, window_frames, face_size=face_size):
        chunks.append(predict_window(model, imgs, device))
    return np.concatenate(chunks) if chunks else np.array([])


@torch.no_grad()
def _predict_raw_clip(
    model: RppgModel,
    frame_dir: Path,
    landmark_csv: Path | None,
    device: torch.device,
    fs: int,
    window_sec: int,
    face_crop_mode: str = "yunet",
) -> tuple[np.ndarray, int]:
    """Predict rPPG window-by-window from raw PGM frames."""
    stream = FaceCropStream(frame_dir, mode=face_crop_mode, landmark_csv=landmark_csv)
    window_frames = fs * window_sec
    batch: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    for i in range(len(stream)):
        batch.append(stream.crop(i))
        if len(batch) == window_frames:
            chunks.append(predict_window(model, stack_face_window(batch), device))
            batch = []
    if batch:
        chunks.append(predict_window(model, stack_face_window(batch), device))
    return (np.concatenate(chunks) if chunks else np.array([]), len(stream))


def _write_results(
    output_dir: Path,
    stem: str,
    source: str,
    fs: int,
    num_frames: int,
    rppg: np.ndarray,
) -> dict:
    """Write rPPG array and summary JSON; return result dict."""
    filtered = butter_bandpass(rppg, 0.6, 4.0, fs)
    ibi = extract_ibi(filtered, fs)
    stress = baevsky_stress_index(ibi)

    output_dir.mkdir(parents=True, exist_ok=True)
    rppg_path = output_dir / f"{stem}_rppg.npy"
    np.save(rppg_path, rppg)

    result = {
        "input": source,
        "fs": fs,
        "num_frames": num_frames,
        "rppg_file": str(rppg_path),
        "ibi": ibi.tolist(),
        **stress,
    }
    out_path = output_dir / f"{stem}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    return result


def run_inference(
    config: InferenceConfig | TrainConfig,
    checkpoint: Path,
    output_dir: Path,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    landmarks_csv: Path | None = None,
) -> dict:
    """
    Run rPPG + Baevsky SI on a preprocessed H5 clip or raw PGM frames.

    Provide either ``input_h5`` or both ``input_dir`` and ``landmarks_csv``.
    """
    has_h5 = input_h5 is not None
    has_raw = input_dir is not None and landmarks_csv is not None
    if has_h5 == has_raw:
        raise ValueError(
            "Provide exactly one input mode: --input-h5 OR (--input-dir AND --landmarks-csv)."
        )

    device = _device()
    model = _load_model(config, checkpoint, device)

    if input_h5 is not None:
        rppg = _predict_h5_clip(
            model,
            input_h5,
            device,
            config.fs,
            config.eval_window_sec,
            face_size=config.face_size,
        )
        return _write_results(
            output_dir,
            input_h5.stem,
            str(input_h5),
            config.fs,
            h5_num_frames(input_h5),
            rppg,
        )

    face_crop_mode = getattr(config, "face_crop_mode", "yunet")
    rppg, num_frames = _predict_raw_clip(
        model,
        input_dir,
        landmarks_csv,
        device,
        config.fs,
        config.eval_window_sec,
        face_crop_mode=face_crop_mode,
    )
    return _write_results(
        output_dir,
        input_dir.name,
        f"{input_dir} + {landmarks_csv}",
        config.fs,
        num_frames,
        rppg,
    )
