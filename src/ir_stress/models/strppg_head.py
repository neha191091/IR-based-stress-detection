"""ST-rPPG head that attaches to a backbone's last feature layer."""

import torch
import torch.nn as nn


class STRppgHead(nn.Module):
    """
    Adaptive spatial pooling and 1x1 conv producing an ST-rPPG block.

    Output shape: (B, spatial_dim^2 + 1, T) — spatial samples plus spatial mean.
    """

    def __init__(self, in_channels: int, spatial_dim: int = 2):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, spatial_dim, spatial_dim)),
            nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Convert backbone features to an ST-rPPG block."""
        x = self.pool_conv(features)  # (B, 1, T, S, S)
        s = self.spatial_dim
        spatial_signals = [x[:, :, :, a, b] for a in range(s) for b in range(s)]
        mean_signal = sum(spatial_signals) / (s * s)
        return torch.cat(spatial_signals + [mean_signal], dim=1)
