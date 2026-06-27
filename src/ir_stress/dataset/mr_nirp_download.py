"""MR-NIRP driving dataset download and selective fetch."""

import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import gdown

ALL_SUBJECTS = list(range(1, 20))
MR_NIRP_DATASET_NAME = "mr-nirp"
MR_NIRP_GDRIVE_FOLDER_ID = "1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK"
MR_NIRP_GDRIVE_URL = (
    f"https://drive.google.com/drive/folders/{MR_NIRP_GDRIVE_FOLDER_ID}"
)
CLIP_CONDITIONS = [
    "driving_still_940",
    "driving_still_975",
    "driving_small_motion_940",
    "driving_small_motion_975",
    "driving_large_motion_975",
    "garage_still_940",
    "garage_still_975",
    "garage_small_motion_940",
    "garage_small_motion_975",
    "garage_large_motion_975",
]
PULSEOX_MODALITY = "PulseOX"
ALL_MOTION_TYPES = ["still", "small_motion", "large_motion"]
ALL_SCENES = ["driving", "garage"]


@dataclass
class DownloadOptions:
    """Controls which MR-NIRP driving data to fetch."""

    subjects: list[int] = field(default_factory=lambda: list(ALL_SUBJECTS))
    download_nir: bool = True
    download_rgb: bool = False
    wavelengths: list[int] = field(default_factory=lambda: [940])
    motion_types: list[str] = field(default_factory=lambda: list(ALL_MOTION_TYPES))
    scenes: list[str] = field(default_factory=lambda: list(ALL_SCENES))
    extract: bool = True


def parse_subjects(spec: str | None) -> list[int]:
    """Parse subject ids from 'all', '1,2,3', or '1-5'."""
    if spec is None or spec.strip().lower() == "all":
        return list(ALL_SUBJECTS)
    subjects: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def parse_motion_types(spec: str | None) -> list[str]:
    """Parse motion types from 'all', 'still', or 'still,small_motion'."""
    if spec is None or spec.strip().lower() == "all":
        return list(ALL_MOTION_TYPES)
    motion_types: list[str] = []
    for part in spec.split(","):
        name = part.strip().lower().replace("-", "_")
        if name not in ALL_MOTION_TYPES:
            raise ValueError(
                f"Unknown motion type {part!r}. "
                f"Use still, small_motion, large_motion, or all."
            )
        motion_types.append(name)
    return motion_types


def parse_scenes(spec: str | None) -> list[str]:
    """Parse scenes from 'all', 'driving', or 'driving,garage'."""
    if spec is None or spec.strip().lower() == "all":
        return list(ALL_SCENES)
    scenes: list[str] = []
    for part in spec.split(","):
        name = part.strip().lower()
        if name not in ALL_SCENES:
            raise ValueError(f"Unknown scene {part!r}. Use driving, garage, or all.")
        scenes.append(name)
    return scenes


def clip_name(subject_id: int, condition: str) -> str:
    """Return the MR-NIRP clip directory name for a subject and condition."""
    return f"subject{subject_id}_{condition}"


def list_target_clips(options: DownloadOptions) -> list[tuple[int, str]]:
    """Return (subject_id, clip_dir_name) pairs selected by options."""
    clips: list[tuple[int, str]] = []
    for subject_id in options.subjects:
        for condition in CLIP_CONDITIONS:
            if not any(str(w) in condition for w in options.wavelengths):
                continue
            if not any(motion in condition for motion in options.motion_types):
                continue
            if not any(condition.startswith(f"{scene}_") for scene in options.scenes):
                continue
            clips.append((subject_id, clip_name(subject_id, condition)))
    return clips


def _subject_dir(root: Path, subject_id: int) -> Path | None:
    """Resolve SubjectN directory under root (case variants)."""
    for name in (f"Subject{subject_id}", f"subject{subject_id}"):
        path = root / name
        if path.is_dir():
            return path
    return None


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a modality zip into the clip directory and remove the zip file."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def _find_modality_source(clip_dir: Path, modality: str) -> tuple[Path, str] | None:
    """Return (path, kind) for a modality folder or zip under clip_dir."""
    if modality == PULSEOX_MODALITY:
        names = ("PulseOX", "PulseOx", "pulseox")
    else:
        names = (modality,)
    for name in names:
        folder = clip_dir / name
        if folder.is_dir():
            return folder, "folder"
        zip_path = clip_dir / f"{name}.zip"
        if zip_path.is_file():
            return zip_path, "zip"
    return None


def _modalities_to_fetch(options: DownloadOptions) -> list[str]:
    """Modalities required for this download (PulseOX always included)."""
    modalities = [PULSEOX_MODALITY]
    if options.download_nir:
        modalities.append("NIR")
    if options.download_rgb:
        modalities.append("RGB")
    return modalities


def _clip_complete(clip_dir: Path, options: DownloadOptions) -> bool:
    """True when clip_dir already has every requested modality."""
    if options.extract:
        for modality in _modalities_to_fetch(options):
            found = _find_modality_source(clip_dir, modality)
            if found is None or found[1] != "folder":
                return False
        return True
    return all(
        _find_modality_source(clip_dir, modality) is not None
        for modality in _modalities_to_fetch(options)
    )


def _extract_modality_zips(clip_dir: Path, options: DownloadOptions) -> None:
    """Extract any modality zips present under clip_dir."""
    for modality in _modalities_to_fetch(options):
        found = _find_modality_source(clip_dir, modality)
        if found is not None and found[1] == "zip":
            _extract_zip(found[0], clip_dir)


def _copy_or_extract_modality(
    src_clip: Path, dst_clip: Path, modality: str, extract: bool
) -> bool:
    """Copy one modality folder or zip from src_clip to dst_clip."""
    found = _find_modality_source(src_clip, modality)
    if found is None:
        return False

    src_path, kind = found
    dst_folder = dst_clip / modality
    dst_zip = dst_clip / f"{modality}.zip"

    if kind == "folder":
        if dst_folder.exists():
            shutil.rmtree(dst_folder)
        shutil.copytree(src_path, dst_folder)
        return True

    shutil.copy2(src_path, dst_zip)
    if extract:
        _extract_zip(dst_zip, dst_clip)
    return True


class _GDriveCatalog:
    """Lazy Google Drive folder listings for MR-NIRP Car."""

    def __init__(self, folder_id: str) -> None:
        self.folder_id = folder_id
        self._subjects: dict[int, str] | None = None
        self._clip_folders: dict[int, dict[str, str]] = {}
        self._zip_files: dict[str, dict[str, str]] = {}

    def _gdrive_session(self):
        from gdown.download import _get_session

        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.102 Safari/537.36"
        )
        return _get_session(proxy=None, use_cookies=True, user_agent=user_agent)[0]

    def _list_children(self, folder_id: str) -> list[tuple[str, str, str]]:
        from gdown.download_folder import _parse_embedded_folder_view

        sess = self._gdrive_session()
        _, children = _parse_embedded_folder_view(sess, folder_id)
        return children

    def subjects(self) -> dict[int, str]:
        if self._subjects is None:
            from gdown.download_folder import _GoogleDriveFile

            self._subjects = {}
            for child_id, child_name, child_type in self._list_children(self.folder_id):
                if child_type != _GoogleDriveFile.TYPE_FOLDER:
                    continue
                match = re.match(r"Subject(\d+)", child_name, re.IGNORECASE)
                if match:
                    self._subjects[int(match.group(1))] = child_id
        return self._subjects

    def clip_folders(self, subject_id: int) -> dict[str, str]:
        if subject_id not in self._clip_folders:
            from gdown.download_folder import _GoogleDriveFile

            subject_folder_id = self.subjects()[subject_id]
            self._clip_folders[subject_id] = {
                name: child_id
                for child_id, name, child_type in self._list_children(subject_folder_id)
                if child_type == _GoogleDriveFile.TYPE_FOLDER
            }
        return self._clip_folders[subject_id]

    def zip_files(self, clip_folder_id: str) -> dict[str, str]:
        if clip_folder_id not in self._zip_files:
            from gdown.download_folder import _GoogleDriveFile

            self._zip_files[clip_folder_id] = {
                name: child_id
                for child_id, name, child_type in self._list_children(clip_folder_id)
                if child_type != _GoogleDriveFile.TYPE_FOLDER
            }
        return self._zip_files[clip_folder_id]


def _resolve_drive_zip(modality: str, zip_files: dict[str, str]) -> str | None:
    """Match a modality to a zip filename listed on Google Drive."""
    for name in zip_files:
        lower = name.lower()
        if modality == PULSEOX_MODALITY and "pulse" in lower and lower.endswith(".zip"):
            return name
        if lower == f"{modality.lower()}.zip":
            return name
    return None


def fetch_clip_from_gdrive(
    catalog: _GDriveCatalog,
    out_root: Path,
    subject_id: int,
    clip_dir_name: str,
    options: DownloadOptions,
    force: bool = False,
) -> list[str]:
    """Download selected modalities for one clip from Google Drive into out_root."""
    dest_clip = out_root / f"Subject{subject_id}" / clip_dir_name
    if dest_clip.is_dir() and _clip_complete(dest_clip, options) and not force:
        return _modalities_to_fetch(options)

    if subject_id not in catalog.subjects():
        raise FileNotFoundError(f"Subject{subject_id} not found on Google Drive")

    clip_folders = catalog.clip_folders(subject_id)
    if clip_dir_name not in clip_folders:
        return []

    if force and dest_clip.exists():
        shutil.rmtree(dest_clip)
    dest_clip.mkdir(parents=True, exist_ok=True)

    zip_files = catalog.zip_files(clip_folders[clip_dir_name])
    for modality in _modalities_to_fetch(options):
        zip_name = _resolve_drive_zip(modality, zip_files)
        if zip_name is None:
            raise FileNotFoundError(
                f"{modality} not found for {clip_dir_name} on Google Drive"
            )
        gdown.download(
            id=zip_files[zip_name],
            output=str(dest_clip / zip_name),
            quiet=False,
            resume=True,
        )

    if options.extract:
        _extract_modality_zips(dest_clip, options)
    return _modalities_to_fetch(options)


def fetch_clip_from_local(
    source_root: Path,
    out_root: Path,
    subject_id: int,
    clip_dir_name: str,
    options: DownloadOptions,
) -> list[str]:
    """Import selected modalities for one clip from an external local dataset."""
    src_subject = _subject_dir(source_root, subject_id)
    if src_subject is None:
        raise FileNotFoundError(f"Subject {subject_id} not found under {source_root}")

    src_clip = src_subject / clip_dir_name
    if not src_clip.is_dir():
        return []

    dst_clip = out_root / f"Subject{subject_id}" / clip_dir_name
    dst_clip.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []
    if _copy_or_extract_modality(src_clip, dst_clip, PULSEOX_MODALITY, options.extract):
        fetched.append(PULSEOX_MODALITY)
    if options.download_nir and _copy_or_extract_modality(src_clip, dst_clip, "NIR", options.extract):
        fetched.append("NIR")
    if options.download_rgb and _copy_or_extract_modality(src_clip, dst_clip, "RGB", options.extract):
        fetched.append("RGB")
    return fetched


def _download_url(url: str, dest: Path) -> None:
    """Download a single file from a URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def fetch_clip_from_url(
    base_url: str,
    out_root: Path,
    subject_id: int,
    clip_dir_name: str,
    options: DownloadOptions,
) -> list[str]:
    """Download selected modalities for one clip from a remote URL base."""
    dst_clip = out_root / f"Subject{subject_id}" / clip_dir_name
    dst_clip.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []
    for modality in _modalities_to_fetch(options):
        url = (
            f"{base_url.rstrip('/')}/Subject{subject_id}/"
            f"{clip_dir_name}/{modality}.zip"
        )
        dst_zip = dst_clip / f"{modality}.zip"
        _download_url(url, dst_zip)
        if options.extract:
            _extract_zip(dst_zip, dst_clip)
        fetched.append(modality)
    return fetched


def download_mr_nirp(
    out_dir: Path,
    options: DownloadOptions,
    source_dir: Path | None = None,
    base_url: str | None = None,
    gdrive_folder_id: str | None = None,
    force: bool = False,
) -> list[str]:
    """
    Fetch MR-NIRP driving clips into out_dir (e.g. data/raw/mr-nirp).

    Provide one of:
    - gdrive_folder_id: download selected clips/modalities from Google Drive
    - source_dir: import from an external local Subject*/ layout
    - base_url: HTTP/FTP root that hosts Subject*/clip/*.zip files
    """
    sources = sum(x is not None for x in (source_dir, base_url, gdrive_folder_id))
    if sources != 1:
        raise ValueError(
            "Provide exactly one of gdrive_folder_id, source_dir, or base_url."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = _GDriveCatalog(gdrive_folder_id) if gdrive_folder_id else None
    log: list[str] = []

    for subject_id, clip_dir_name in list_target_clips(options):
        if catalog is not None:
            fetched = fetch_clip_from_gdrive(
                catalog, out_dir, subject_id, clip_dir_name, options, force=force
            )
        elif source_dir is not None:
            fetched = fetch_clip_from_local(
                source_dir, out_dir, subject_id, clip_dir_name, options
            )
        else:
            fetched = fetch_clip_from_url(
                base_url, out_dir, subject_id, clip_dir_name, options
            )
        if fetched:
            log.append(f"Subject{subject_id}/{clip_dir_name}: {', '.join(fetched)}")
    return log
