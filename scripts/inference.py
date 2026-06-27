"""Run rPPG + Baevsky SI inference on an H5 clip or raw PGM frames."""

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import InferenceConfig
from ir_stress.inference.pipeline import run_inference


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: InferenceConfig) -> None:
    """Extract rPPG and Baevsky Stress Index from an H5 clip or raw PGM frames."""
    result = run_inference(
        cfg,
        Path(cfg.checkpoint),
        Path(cfg.output_dir),
        input_h5=Path(cfg.input_h5) if cfg.input_h5 else None,
        input_dir=Path(cfg.input_dir) if cfg.input_dir else None,
        landmarks_csv=Path(cfg.landmarks_csv) if cfg.landmarks_csv else None,
    )
    print(
        f"Baevsky SI: {result['baevsky_si']:.2f} "
        f"(sqrt: {result['baevsky_si_sqrt']:.2f})"
    )


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=InferenceConfig)
    main()
