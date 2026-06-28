"""End-to-end IR_iHR classical signal extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ir_stress.config import InferenceConfig, TrainConfig
from ir_stress.dataset.base import resize_face_clip
from ir_stress.signals.ihr_core import IHRExtractionResult, extract_ihr_from_grid
from ir_stress.signals.ihr_regions import (
    grid_signals_from_face_clip,
    grid_signals_from_pgm,
)
from ir_stress.signals.stress_indicators import baevsky_stress_index, extract_ibi


def extract_from_grid_matrix(
    Y: np.ndarray,
    fs: int,
    prior_bpm: float = 70.0,
    *,
    window: int = 301,
) -> IHRExtractionResult:
    """Run the IR_iHR extraction on a precomputed grid signal matrix."""
    return extract_ihr_from_grid(Y, fs, prior_bpm=prior_bpm, window=window)


def _grid_from_h5(path: Path, face_size: int, grid_size: int) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as f:
        imgs = resize_face_clip(f["imgs"][:], face_size)
    Y = grid_signals_from_face_clip(imgs, grid_size=grid_size)
    return Y, imgs.shape[0]


def _write_ihr_results(
    output_dir: Path,
    stem: str,
    source: str,
    fs: int,
    num_frames: int,
    result: IHRExtractionResult,
    *,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    raw_root: str = "data/raw/mr-nirp",
    plot: bool = True,
) -> dict:
    """Write PPG, iHR arrays, and summary JSON."""
    ibi = extract_ibi(result.ppg, fs)
    stress = baevsky_stress_index(ibi)

    output_dir.mkdir(parents=True, exist_ok=True)
    ppg_path = output_dir / f"{stem}_rppg.npy"
    ihr_path = output_dir / f"{stem}_ihr_bpm.npy"
    ihr_time_path = output_dir / f"{stem}_ihr_time.npy"
    np.save(ppg_path, result.ppg)
    np.save(ihr_path, result.ihr_bpm)
    np.save(ihr_time_path, result.ihr_time)

    summary = {
        "input": source,
        "extraction_method": "ihr",
        "fs": fs,
        "num_frames": num_frames,
        "rppg_file": str(ppg_path),
        "ihr_bpm_file": str(ihr_path),
        "ihr_time_file": str(ihr_time_path),
        "ihr_quality": result.quality,
        "mean_ihr_bpm": float(np.nanmean(result.ihr_bpm)),
        "ibi": ibi.tolist(),
        **stress,
    }
    out_path = output_dir / f"{stem}_results.json"
    with out_path.open("w") as f:
        json.dump(summary, f)

    from ir_stress.inference.plot import maybe_save_inference_plot

    plot_file = maybe_save_inference_plot(
        output_dir,
        stem,
        result.ppg,
        fs,
        num_frames,
        input_h5=input_h5,
        input_dir=input_dir,
        raw_root=raw_root,
        title="IR_iHR vs pulse-ox",
        enabled=plot,
    )
    if plot_file is not None:
        summary["plot_file"] = plot_file

    return summary


def run_ihr_inference(
    config: InferenceConfig | TrainConfig,
    output_dir: Path,
    *,
    input_h5: Path | None = None,
    input_dir: Path | None = None,
    landmarks_csv: Path | None = None,
    prior_bpm: float = 70.0,
    grid_size: int | None = None,
    dlib_predictor: str | Path | None = None,
) -> dict:
    """
    Extract contactless PPG and instantaneous HR using the IR_iHR method.

    Provide either ``input_h5`` (uniform face-crop grid) or ``input_dir`` (raw PGM).
    For raw PGM, ``landmarks_csv`` is optional: when omitted, dlib detects 68-point
    landmarks automatically.
    """
    has_h5 = input_h5 is not None
    has_raw = input_dir is not None
    if has_h5 == has_raw:
        raise ValueError(
            "Provide exactly one input mode: --input-h5 OR --input-dir."
        )

    fs = config.fs
    window = 301
    plot = getattr(config, "plot", True)
    raw_root = getattr(config, "raw_dir", "data/raw/mr-nirp")
    if dlib_predictor is None:
        dlib_predictor = getattr(config, "dlib_predictor", None)
    if grid_size is None:
        grid_size = 8 if has_h5 else 5

    if input_h5 is not None:
        Y, num_frames = _grid_from_h5(input_h5, config.face_size, grid_size)
        result = extract_from_grid_matrix(Y, fs, prior_bpm=prior_bpm, window=window)
        return _write_ihr_results(
            output_dir,
            input_h5.stem,
            str(input_h5),
            fs,
            num_frames,
            result,
            input_h5=input_h5,
            raw_root=raw_root,
            plot=plot,
        )

    landmark_source = str(landmarks_csv) if landmarks_csv is not None else "dlib"
    Y = grid_signals_from_pgm(
        input_dir,
        landmarks_csv,
        grid_size=grid_size,
        dlib_predictor=dlib_predictor,
    )
    num_frames = Y.shape[1]
    result = extract_from_grid_matrix(Y, fs, prior_bpm=prior_bpm, window=window)
    return _write_ihr_results(
        output_dir,
        input_dir.name,
        f"{input_dir} + {landmark_source}",
        fs,
        num_frames,
        result,
        input_dir=input_dir,
        raw_root=raw_root,
        plot=plot,
    )
