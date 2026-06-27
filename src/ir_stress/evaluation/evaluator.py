"""Test-set rPPG evaluation with Pearson correlation and MSE."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from ir_stress.config import TrainConfig
from ir_stress.device import resolve_device
from ir_stress.dataset.base import h5_num_frames, read_h5_window
from ir_stress.models.model import build_model
from ir_stress.models.predict import predict_window
from ir_stress.signals.metrics import mse, pearson_r
from ir_stress.tracking import ensure_mlflow


def evaluate(config: TrainConfig, checkpoint: Path, test_list: list[str]) -> dict[str, float]:
    """
    Evaluate a trained model on test H5 clips.

    Reads each eval window from disk individually to limit memory use.
    """
    device = resolve_device(config.device)
    model = build_model(config.model, spatial_dim=config.spatial_dim, in_ch=config.in_ch)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    window_frames = config.eval_window_sec * config.fs
    rows = []

    with torch.no_grad():
        for h5_path in test_list:
            n = h5_num_frames(h5_path)
            num_blocks = n // window_frames
            for b in range(num_blocks):
                start = b * window_frames
                end = start + window_frames
                imgs, gt = read_h5_window(h5_path, start, end, face_size=config.face_size)
                pred = predict_window(model, imgs, device)
                rows.append(
                    {
                        "file": Path(h5_path).name,
                        "block": b,
                        "pearson_r": pearson_r(pred, gt, config.fs),
                        "mse": mse(pred, gt, config.fs),
                    }
                )

    df = pd.DataFrame(rows)
    metrics = {
        "pearson_mean": float(df["pearson_r"].mean()),
        "pearson_std": float(df["pearson_r"].std()),
        "mse_mean": float(df["mse"].mean()),
        "mse_std": float(df["mse"].std()),
    }

    ensure_mlflow()
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_params({"checkpoint": str(checkpoint), "model": config.model})
        mlflow.log_metrics(metrics)
        csv_path = checkpoint.parent / "eval_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

    return metrics
