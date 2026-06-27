"""Lightweight end-to-end smoke test on synthetic data."""

from pathlib import Path

import numpy as np
import torch

from ir_stress.config import TrainConfig
from ir_stress.data.synthetic import create_smoke_dataset
from ir_stress.evaluation.evaluator import evaluate
from ir_stress.inference.pipeline import run_inference
from ir_stress.models.model import build_model
from ir_stress.signals.stress_indicators import baevsky_stress_index
from ir_stress.signals.metrics import mse, pearson_r
from ir_stress.training.trainer import train

SMOKE_FACE_SIZE = 64
SMOKE_FRAMES = 300  # 10 s at 30 fps — enough for train / eval / infer
SMOKE_CLIPS = 2


def smoke_config() -> TrainConfig:
    """Fast settings for the end-to-end smoke test."""
    return TrainConfig(
        clip_seconds=5,
        epochs=1,
        video_duration_sec=10,
        eval_window_sec=10,
        face_size=SMOKE_FACE_SIZE,
        num_workers=0,
        h5_dir="data/smoke_h5",
        checkpoint_dir="checkpoints/smoke",
        mlflow_experiment="ir-stress-smoke",
    )


def test_model_forward() -> None:
    """Verify PhysNet + STRppgHead output shapes."""
    t = 30 * 5  # one 5 s clip
    model = build_model("physnet", spatial_dim=2, in_ch=1)
    model.eval()
    x = torch.randn(1, 1, t, SMOKE_FACE_SIZE, SMOKE_FACE_SIZE)
    with torch.no_grad():
        out = model(x)
        rppg = model.predict_rppg(x)
    assert out.shape == (1, 5, t), f"unexpected shape {out.shape}"
    assert rppg.shape == (1, t)


def test_signals() -> None:
    """Verify metric and Baevsky SI helpers."""
    fs = 30
    t = np.arange(300) / fs
    pred = np.sin(2 * np.pi * 1.2 * t)
    gt = np.sin(2 * np.pi * 1.2 * t + 0.1)
    assert pearson_r(pred, gt, fs) > 0.9
    assert mse(pred, gt, fs) < 0.5
    ibi = np.full(20, 0.8) + np.random.default_rng(0).normal(0, 0.02, 20)
    si = baevsky_stress_index(ibi)
    assert not np.isnan(si["baevsky_si"])


def test_train_eval_infer(cfg: TrainConfig) -> None:
    """Run a minimal train → evaluate → infer loop."""
    h5_dir = Path(cfg.h5_dir)
    paths = create_smoke_dataset(h5_dir, n_clips=SMOKE_CLIPS, num_frames=SMOKE_FRAMES)

    run_dir = train(cfg, paths, paths[:1])
    ckpt = run_dir / "epoch0.pt"
    assert ckpt.exists(), f"missing checkpoint {ckpt}"

    metrics = evaluate(cfg, ckpt, paths[:1])
    assert "pearson_mean" in metrics

    result = run_inference(cfg, ckpt, Path("results/smoke"), input_h5=Path(paths[0]))
    assert len(result["ibi"]) > 0
    assert Path(result["rppg_file"]).exists()


def main() -> None:
    cfg = smoke_config()
    print("1/3 model forward...", flush=True)
    test_model_forward()
    print("2/3 signals...", flush=True)
    test_signals()
    print("3/3 train / eval / infer...", flush=True)
    test_train_eval_infer(cfg)
    print("All smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
