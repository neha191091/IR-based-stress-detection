"""PhysNet backbone — encoder/decoder 3D convolutions from Contrast-Phys+.

Adapted from the Contrast-Phys+ reference implementation:
  https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B
  (upstream: contrast-phys+/model.py)

Original work: Sun & Li, TPAMI 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ir_stress.models.backbone import Backbone


class PhysNet(Backbone):
    """3D CNN encoder/decoder producing (B, 64, T, 8, 8) feature maps (Contrast-Phys+)."""

    def __init__(self, in_ch: int = 3, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.start = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(),
        )
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

    @property
    def out_channels(self) -> int:
        return 64

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder/decoder path; input is already normalized."""
        parity = []
        x = self.start(x)
        x = self._run(self.loop1, x)
        parity.append(x.size(2) % 2)
        x = self._run(self.encoder1, x)
        parity.append(x.size(2) % 2)
        x = self._run(self.encoder2, x)
        x = self._run(self.loop4, x)

        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self._run(self.decoder1, x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode="replicate")
        x = F.interpolate(x, scale_factor=(2, 1, 1))
        x = self._run(self.decoder2, x)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode="replicate")
        return x
