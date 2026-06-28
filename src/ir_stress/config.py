"""Pipeline configuration (Hydra structured configs per script)."""

from dataclasses import dataclass, field
from typing import Any, TypeVar

from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import MISSING

from ir_stress.dataset.mr_nirp_download import MR_NIRP_DATASET_NAME, MR_NIRP_GDRIVE_FOLDER_ID

T = TypeVar("T")


def h5_dir_for_face_crop_mode(face_crop_mode: str, data_dir: str = "data") -> str:
    """Directory for preprocessed H5 clips for a given face-crop backend."""
    return f"{data_dir}/h5_{face_crop_mode}"


def resolve_h5_dir(cfg: Any) -> str:
    """Return the H5 directory from an explicit override or face_crop_mode."""
    explicit = getattr(cfg, "h5_dir", None)
    if explicit is not None:
        return explicit
    face_crop_mode = getattr(cfg, "face_crop_mode", "yunet")
    return h5_dir_for_face_crop_mode(face_crop_mode)


def _hydra_conf() -> HydraConf:
    return HydraConf(run=RunDir(dir="."), output_subdir=None, job=JobConf(chdir=False))


@dataclass
class HydraConfig:
    hydra: HydraConf = field(default_factory=_hydra_conf)


@dataclass
class ModelConfig:
    """Shared model and signal-processing defaults."""

    model: str = "physnet"
    in_ch: int = 1
    fs: int = 30
    spatial_dim: int = 2
    face_size: int = 128
    eval_window_sec: int = 30
    device: str | None = None


@dataclass
class OldConfig:
    """Monolithic config kept for manual comparison (not used by scripts)."""

    model: str = "physnet"
    in_ch: int = 1
    fs: int = 30
    clip_seconds: int = 5
    spatial_dim: int = 2
    epochs: int = 30
    lr: float = 1e-5
    label_ratio: float = 0.0
    val_subjects: list[int] = field(default_factory=lambda: [1])
    wavelengths: list[int] = field(default_factory=lambda: [940])
    video_duration_sec: int = 200
    eval_window_sec: int = 30
    batch_size: int = 2
    face_size: int = 128
    num_workers: int = 0
    h5_dir: str = "data/h5"
    raw_dir: str = "data/raw/mr-nirp"
    landmarks_dir: str = "data/landmarks"
    face_crop_mode: str = "yunet"
    mlflow_experiment: str = "ir-stress-rppg"
    checkpoint_dir: str = "checkpoints"
    hydra: HydraConf = field(default_factory=_hydra_conf)

    @property
    def T(self) -> int:
        """Temporal clip length in frames."""
        return self.fs * self.clip_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint config.json (excludes hydra)."""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key != "hydra"
        }


@dataclass
class TrainConfig(ModelConfig, HydraConfig):
    """scripts/train.py — Contrast-Phys+ training."""

    clip_seconds: int = 5
    epochs: int = 30
    lr: float = 1e-5
    label_ratio: float = 0.0
    val_subjects: list[int] = field(default_factory=lambda: [1])
    wavelengths: list[int] = field(default_factory=lambda: [940])
    video_duration_sec: int = 200
    batch_size: int = 2
    num_workers: int = 0
    h5_dir: str | None = None
    raw_dir: str = "data/raw/mr-nirp"
    face_crop_mode: str = "yunet"
    checkpoint_dir: str = "checkpoints"
    mlflow_experiment: str = "ir-stress-rppg"
    use_amp: bool = True
    grad_checkpoint: bool = False
    micro_batch: bool = True

    @property
    def T(self) -> int:
        """Temporal clip length in frames."""
        return self.fs * self.clip_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint config.json (excludes hydra)."""
        data = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key != "hydra"
        }
        data["h5_dir"] = resolve_h5_dir(self)
        return data


@dataclass
class PreprocessConfig(HydraConfig):
    """scripts/preprocess.py — raw PGM to H5."""

    raw_dir: str = "data/raw/mr-nirp"
    landmarks_dir: str = "data/landmarks"
    wavelengths: list[int] = field(default_factory=lambda: [940])
    face_crop_mode: str = "yunet"
    face_size: int = 128
    dataset: str = "mr_nirp_driving"


@dataclass
class EvaluateConfig(HydraConfig):
    """scripts/evaluate.py — test-set metrics for a checkpoint."""

    checkpoint: str = MISSING


@dataclass
class InferenceConfig(ModelConfig, HydraConfig):
    """scripts/inference.py — rPPG + Baevsky SI on H5 or raw frames."""

    face_crop_mode: str = "yunet"
    extraction_method: str = "neural"  # "neural" (Contrast-Phys+) or "ihr" (IR_iHR)
    prior_bpm: float = 70.0
    ihr_grid_size: int | None = None  # default: 8 for H5, 5 for raw PGM
    dlib_predictor: str = "data/models/shape_predictor_68_face_landmarks.dat"
    checkpoint: str | None = None
    input_h5: str | None = None
    input_dir: str | None = None
    landmarks_csv: str | None = None
    output_dir: str = "results/inference"
    plot: bool = True
    raw_dir: str = "data/raw/mr-nirp"


@dataclass
class DownloadConfig(HydraConfig):
    """scripts/download.py — fetch MR-NIRP driving data."""

    source_dir: str | None = None
    base_url: str | None = None
    gdrive_folder_id: str = MR_NIRP_GDRIVE_FOLDER_ID
    out_dir: str = f"data/raw/{MR_NIRP_DATASET_NAME}"
    subjects: str = "all"
    download_nir: bool = True
    download_rgb: bool = False
    wavelengths: list[int] = field(default_factory=lambda: [940])
    motion_types: str = "all"
    scenes: str = "all"
    extract: bool = True
    force: bool = False


def resolve_config(cfg: T) -> T:
    """Return a dataclass (unwrap Hydra's OmegaConf node)."""
    from omegaconf import DictConfig, OmegaConf

    if isinstance(cfg, DictConfig):
        return OmegaConf.to_object(cfg)
    return cfg
