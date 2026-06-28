"""Contrast-Phys+ contrastive loss.

Adapted from the Contrast-Phys+ reference implementation:
  https://github.com/zhaodongsun/contrast-phys/tree/master/contrast-phys%2B
  (upstream: contrast-phys+/loss.py)

Original work: Sun & Li, "Contrast-Phys+: Unsupervised and Weakly-supervised
Video-based Remote Physiological Measurement via Spatiotemporal Contrast",
TPAMI 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalculateNormPSD(nn.Module):
    """Normalized power spectral density in the physiological band (Contrast-Phys+)."""

    def __init__(self, fs: float, high_pass: float, low_pass: float):
        super().__init__()
        self.fs = fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x: torch.Tensor, zero_pad: int = 0) -> torch.Tensor:
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            length = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * length), int(zero_pad / 2 * length)), "constant", 0)
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="forward"))
        x = x[:, 0] ** 2 + x[:, 1] ** 2
        fn = self.fs / 2
        freqs = torch.linspace(0, fn, x.shape[0], device=x.device)
        use_freqs = (freqs >= self.high_pass / 60) & (freqs <= self.low_pass / 60)
        x = x[use_freqs]
        return x / torch.sum(x, dim=-1, keepdim=True)


class ST_sampling(nn.Module):
    """Spatiotemporal sampling on an ST-rPPG block (Contrast-Phys+)."""

    def __init__(self, delta_t: int, k: int, fs: float, high_pass: float, low_pass: float):
        super().__init__()
        self.delta_t = delta_t
        self.k = k
        self.norm_psd = CalculateNormPSD(fs, high_pass, low_pass)

    def forward(self, block: torch.Tensor) -> list[list[torch.Tensor]]:
        samples = []
        for b in range(block.shape[0]):
            per_video = []
            for c in range(block.shape[1]):
                for _ in range(self.k):
                    offset = torch.randint(
                        0, block.shape[-1] - self.delta_t + 1, (1,), device=block.device
                    )
                    per_video.append(self.norm_psd(block[b, c, offset : offset + self.delta_t]))
            samples.append(per_video)
        return samples


class T_sampling(nn.Module):
    """Temporal sampling on ground-truth PPG signals (Contrast-Phys+)."""

    def __init__(self, delta_t: int, k: int, fs: float, high_pass: float, low_pass: float):
        super().__init__()
        self.delta_t = delta_t
        self.k = k
        self.norm_psd = CalculateNormPSD(fs, high_pass, low_pass)

    def forward(
        self, gt: torch.Tensor, block_shape: torch.Size
    ) -> list[list[torch.Tensor]]:
        samples = []
        for b in range(gt.shape[0]):
            per_sig = []
            for _ in range(block_shape[1]):
                for _ in range(self.k):
                    offset = torch.randint(
                        0, gt.shape[-1] - self.delta_t + 1, (1,), device=gt.device
                    )
                    per_sig.append(self.norm_psd(gt[b, offset : offset + self.delta_t]))
            samples.append(per_sig)
        return samples


class ContrastLoss(nn.Module):
    """Spatiotemporal contrastive loss (Contrast-Phys+)."""

    def __init__(self, delta_t: int, k: int, fs: float, high_pass: float, low_pass: float):
        super().__init__()
        self.st_sampling = ST_sampling(delta_t, k, fs, high_pass, low_pass)
        self.t_sampling = T_sampling(delta_t, k, fs, high_pass, low_pass)
        self.distance = nn.MSELoss(reduction="mean")

    def _compare(
        self, list_a: list[torch.Tensor], list_b: list[torch.Tensor], exclude_same: bool = False
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=list_a[0].device)
        count = 0
        for i, a in enumerate(list_a):
            for j, b in enumerate(list_b):
                if exclude_same and i == j:
                    continue
                total = total + self.distance(a, b)
                count += 1
        return total / max(count, 1)

    def forward(
        self,
        model_output: torch.Tensor,
        gt_sig: torch.Tensor,
        label_flag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = self.st_sampling(model_output)
        samples_gt = self.t_sampling(gt_sig, model_output.shape)

        pos_loss = (
            self._compare(samples[0], samples[0], exclude_same=True)
            + self._compare(samples[1], samples[1], exclude_same=True)
        ) / 2
        neg_loss = -self._compare(samples[0], samples[1])

        if torch.sum(label_flag) == 0:
            pos_loss_gt = torch.zeros_like(pos_loss)
            neg_loss_gt = torch.zeros_like(neg_loss)
        else:
            pos_loss_gt = (
                label_flag[0] * self._compare(samples[0], samples_gt[0])
                + label_flag[1] * self._compare(samples[1], samples_gt[1])
            ) / torch.sum(label_flag)
            neg_loss_gt = -(
                label_flag[0] * self._compare(samples[1], samples_gt[0])
                + label_flag[1] * self._compare(samples[0], samples_gt[1])
            ) / torch.sum(label_flag)

        loss = pos_loss + neg_loss + pos_loss_gt + neg_loss_gt
        return loss, pos_loss, neg_loss, pos_loss_gt, neg_loss_gt
