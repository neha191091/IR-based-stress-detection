"""End-to-end inference: rPPG extraction and Baevsky Stress Index."""

import json
from pathlib import Path

import numpy as np
import torch

from ir_stress.config import InferenceConfig, TrainConfig
from ir_stress.device import resolve_device
from ir_stress.dataset.base import h5_num_frames, iter_h5_imgs_windows
from ir_stress.dataset.face_crop import FaceCropStream, annotate_bbox, stack_face_window
from ir_stress.models.model import build_model, RppgModel
from ir_stress.models.predict import predict_window
from ir_stress.inference.plot import maybe_save_inference_plot
from ir_stress.signals.filtering import butter_bandpass
from ir_stress.signals.ihr_pipeline import run_ihr_inference
from ir_stress.signals.stress_indicators import baevsky_stress_index, extract_ibi


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
    *,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    raw_root: str = "data/raw/mr-nirp",
    plot: bool = True,
    plot_title: str = "",
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

    if plot:
        plot_file = maybe_save_inference_plot(
            output_dir,
            stem,
            rppg,
            fs,
            num_frames,
            input_h5=input_h5,
            input_dir=input_dir,
            raw_root=raw_root,
            title=plot_title,
            enabled=plot,
        )
        if plot_file is not None:
            result["plot_file"] = plot_file

    return result


def run_inference(
    config: InferenceConfig | TrainConfig,
    checkpoint: Path | None,
    output_dir: Path,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    landmarks_csv: Path | None = None,
) -> dict:
    """
    Run rPPG + Baevsky SI on a preprocessed H5 clip or raw PGM frames.

    Provide either ``input_h5`` or ``input_dir`` (with ``landmarks_csv`` for neural
    openface mode). IR_iHR raw mode can omit ``landmarks_csv`` to use dlib instead.

    Set ``extraction_method=ihr`` on :class:`InferenceConfig` to use the classical
    IR_iHR pipeline (no checkpoint required).
    """
    method = getattr(config, "extraction_method", "neural")
    if method == "ihr":
        return run_ihr_inference(
            config,
            output_dir,
            input_h5=input_h5,
            input_dir=input_dir,
            landmarks_csv=landmarks_csv,
            prior_bpm=getattr(config, "prior_bpm", 70.0),
            grid_size=getattr(config, "ihr_grid_size", None),
            dlib_predictor=getattr(config, "dlib_predictor", None),
        )

    if checkpoint is None:
        raise ValueError("checkpoint is required when extraction_method='neural'")

    has_h5 = input_h5 is not None
    has_raw = input_dir is not None and landmarks_csv is not None
    if has_h5 == has_raw:
        raise ValueError(
            "Provide exactly one input mode: --input-h5 OR (--input-dir AND --landmarks-csv)."
        )

    device = resolve_device(getattr(config, "device", None))
    model = _load_model(config, checkpoint, device)
    plot = getattr(config, "plot", True)
    raw_root = getattr(config, "raw_dir", "data/raw/mr-nirp")
    plot_title = f"Inference vs pulse-ox | {checkpoint.parent.name}"

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
            input_h5=input_h5,
            raw_root=raw_root,
            plot=plot,
            plot_title=plot_title,
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
        input_dir=input_dir,
        raw_root=raw_root,
        plot=plot,
        plot_title=plot_title,
    )
