"""Irrelevant power ratio metric from Contrast-Phys+."""

import torch
import torch.nn as nn


class IrrelevantPowerRatio(nn.Module):
    """Ratio of energy outside the physiological band in the predicted rPPG PSD."""

    def __init__(self, fs: float, high_pass: float, low_pass: float):
        super().__init__()
        self.fs = fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        x_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm="forward"))
        fn = self.fs / 2
        freqs = torch.linspace(0, fn, x_real.shape[-2], device=preds.device)
        use_freqs = (freqs >= self.high_pass / 60) & (freqs <= self.low_pass / 60)
        zero_freqs = ~use_freqs
        use_energy = torch.sum(torch.linalg.norm(x_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(x_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)
        for i in range(len(denom)):
            if denom[i] > 0:
                energy_ratio[i] = zero_energy[i] / denom[i]
        return energy_ratio
