"""Base class for spatiotemporal video encoders."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Backbone(nn.Module, ABC):
    """Spatiotemporal video encoder. Subclasses implement encode()."""

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the encode() output tensor."""

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-clip z-score over temporal and spatial dimensions."""
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        return (x - means) / stds

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, C, T, H, W) to spatiotemporal features before the ST-rPPG head."""
