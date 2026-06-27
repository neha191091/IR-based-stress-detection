"""Train an rPPG model with Contrast-Phys+."""

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import TrainConfig, resolve_config
from ir_stress.data.splits import resolve_h5_splits
from ir_stress.training.trainer import train


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: TrainConfig) -> None:
    """Train RppgModel on MR-NIRP driving H5 clips."""
    cfg = resolve_config(cfg)
    train_list, test_list = resolve_h5_splits(cfg)
    print(f"Training on {len(train_list)} clips, test list: {len(test_list)}")
    run_dir = train(cfg, train_list, test_list)
    print(f"Checkpoints saved to {run_dir}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=TrainConfig)
    main()
