"""Download MR-NIRP driving dataset with selective subjects and modalities."""

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from ir_stress.config import DownloadConfig
from ir_stress.dataset.mr_nirp_download import (
    MR_NIRP_GDRIVE_URL,
    DownloadOptions,
    download_mr_nirp,
    parse_motion_types,
    parse_scenes,
    parse_subjects,
)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DownloadConfig) -> None:
    """
    Fetch MR-NIRP driving data into data/raw/mr-nirp/.

    PulseOX (pulseOx.mat) is always downloaded. NIR and RGB are optional.
    """
    opts = DownloadOptions(
        subjects=parse_subjects(cfg.subjects),
        download_nir=cfg.download_nir,
        download_rgb=cfg.download_rgb,
        wavelengths=cfg.wavelengths,
        motion_types=parse_motion_types(cfg.motion_types),
        scenes=parse_scenes(cfg.scenes),
        extract=cfg.extract,
    )

    source_dir = Path(cfg.source_dir) if cfg.source_dir else None
    use_gdrive = source_dir is None and cfg.base_url is None
    use_local = source_dir is not None
    use_url = cfg.base_url is not None

    if sum((use_gdrive, use_local, use_url)) != 1:
        raise ValueError(
            "Provide exactly one source: Google Drive (default), source_dir, or base_url."
        )

    if use_local and not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    out_dir = Path(cfg.out_dir)
    print(
        f"Output: {out_dir}\n"
        f"Subjects: {opts.subjects}\n"
        f"Modalities: PulseOX (always), "
        f"NIR={opts.download_nir}, RGB={opts.download_rgb}\n"
        f"Wavelengths: {opts.wavelengths}\n"
        f"Motion types: {opts.motion_types}\n"
        f"Scenes: {opts.scenes}"
    )
    if use_gdrive:
        print(f"Source: Google Drive ({MR_NIRP_GDRIVE_URL})")
    elif use_local:
        print(f"Source: local {source_dir}")

    log = download_mr_nirp(
        out_dir,
        opts,
        source_dir=source_dir if use_local else None,
        base_url=cfg.base_url if use_url else None,
        gdrive_folder_id=cfg.gdrive_folder_id if use_gdrive else None,
        force=cfg.force,
    )
    print(f"Done ({len(log)} clips) -> {out_dir}")
    for line in log:
        print(line)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config", node=DownloadConfig)
    main()
