"""Run rPPG + Baevsky SI inference on an H5 clip or raw PGM frames."""

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import InferenceConfig, TrainConfig
from ir_stress.dataset.splits import load_config_from_run_dir
from ir_stress.inference.pipeline import run_inference


def _resolve_config(cfg: InferenceConfig, checkpoint: Path | None) -> InferenceConfig | TrainConfig:
    """Use training config saved next to the checkpoint (model arch, face crop, etc.)."""
    if checkpoint is None or cfg.extraction_method == "ihr":
        return cfg
    run_cfg = load_config_from_run_dir(checkpoint.parent)
    if cfg.device is not None:
        run_cfg.device = cfg.device
    return run_cfg


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: InferenceConfig) -> None:
    """Extract rPPG and Baevsky Stress Index from an H5 clip or raw PGM frames."""
    checkpoint = Path(cfg.checkpoint) if cfg.checkpoint else None
    config = _resolve_config(cfg, checkpoint)
    result = run_inference(
        config,
        checkpoint,
        Path(cfg.output_dir),
        input_h5=Path(cfg.input_h5) if cfg.input_h5 else None,
        input_dir=Path(cfg.input_dir) if cfg.input_dir else None,
        landmarks_csv=Path(cfg.landmarks_csv) if cfg.landmarks_csv else None,
    )
    method = result.get("extraction_method", cfg.extraction_method)
    if method == "ihr":
        print(
            f"IR_iHR quality: {result['ihr_quality']:.3f}  "
            f"mean HR: {result['mean_ihr_bpm']:.1f} bpm  "
            f"Baevsky SI: {result['baevsky_si']:.2f}"
        )
    else:
        print(
            f"Baevsky SI: {result['baevsky_si']:.2f} "
            f"(sqrt: {result['baevsky_si_sqrt']:.2f})"
        )
    if plot_file := result.get("plot_file"):
        print(f"Saved comparison plot to {plot_file}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=InferenceConfig)
    main()
