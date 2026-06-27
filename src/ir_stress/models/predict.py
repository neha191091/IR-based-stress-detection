"""Shared model inference helpers."""

import numpy as np
import torch

from ir_stress.models.model import RppgModel


@torch.no_grad()
def predict_window(model: RppgModel, imgs: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model on one video window; imgs shape [T, H, W, C]."""
    batch = imgs.transpose(3, 0, 1, 2)[np.newaxis].astype("float32")
    tensor = torch.tensor(batch, device=device)
    return model.predict_rppg(tensor)[0].cpu().numpy()
