"""Evaluate a trained rPPG model."""

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import EvaluateConfig
from ir_stress.dataset.splits import load_config_from_run_dir, resolve_test_paths
from ir_stress.evaluation.evaluator import evaluate


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: EvaluateConfig) -> None:
    """Evaluate Pearson r and MSE on test clips."""
    checkpoint = Path(cfg.checkpoint)
    run_dir = checkpoint.parent
    run_cfg = load_config_from_run_dir(run_dir)
    test_paths = resolve_test_paths(run_dir)
    metrics = evaluate(run_cfg, checkpoint, test_paths)
    print(metrics)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=EvaluateConfig)
    main()
