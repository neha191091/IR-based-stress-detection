"""PyTorch dataset for H5 face video clips.

Adapted from the Contrast-Phys+ reference implementation:
  https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B
  (upstream: contrast-phys+/dataset.py)

Original work: Sun & Li, TPAMI 2024.
"""

import h5py
import numpy as np
from torch.utils.data import Dataset

from ir_stress.dataset.base import _ppg_dataset, resize_face_clip


class H5ClipDataset(Dataset):
    """Random temporal crops from preprocessed H5 clips for Contrast-Phys+ training."""

    def __init__(
        self,
        file_list: list[str],
        clip_length: int,
        label_ratio: float,
        face_size: int = 128,
    ):
        self.file_list = list(np.random.permutation(file_list))
        self.clip_length = clip_length
        self.label_count = int(len(self.file_list) * label_ratio)
        self.face_size = face_size

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        label_flag = np.float32(1.0 if idx < self.label_count else 0.0)
        with h5py.File(self.file_list[idx], "r") as f:
            ppg_ds = _ppg_dataset(f)
            length = min(f["imgs"].shape[0], ppg_ds.shape[0])
            start = np.random.randint(0, length - self.clip_length)
            end = start + self.clip_length
            ppg = ppg_ds[start:end].astype("float32")
            imgs = resize_face_clip(f["imgs"][start:end].astype(np.float32), self.face_size)
            imgs = np.transpose(imgs, (3, 0, 1, 2)).astype("float32")
        return imgs, ppg, label_flag
