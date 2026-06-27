"""Preprocess raw datasets into H5 format."""

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import PreprocessConfig, h5_dir_for_face_crop_mode
from ir_stress.dataset.mr_nirp_driving import MRNIRPDrivingAdapter


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: PreprocessConfig) -> None:
    """Discover sessions and write preprocessed H5 clips."""
    if cfg.dataset != "mr_nirp_driving":
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    adapter = MRNIRPDrivingAdapter()
    raw = Path(cfg.raw_dir)
    out = Path(h5_dir_for_face_crop_mode(cfg.face_crop_mode))
    landmarks = Path(cfg.landmarks_dir)

    sessions = adapter.discover_sessions(raw, wavelengths=cfg.wavelengths)
    print(f"Found {len(sessions)} sessions")
    for session in sessions:
        path = adapter.preprocess_session(
            session,
            out,
            landmarks_dir=landmarks,
            face_crop_mode=cfg.face_crop_mode,
            face_size=cfg.face_size,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=PreprocessConfig)
    main()
