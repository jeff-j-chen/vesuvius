"""self-supervised blind-spot denoiser (Noise2Void, block-masked for correlated CT noise)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BlindSpotDenoiser(nn.Module):
    """small 3D conv net mapping a normalized (B, 1, D, H, W) window to its denoised estimate."""

    def __init__(self, channels: int = 32, layers: int = 6):
        super().__init__()
        blocks = [nn.Conv3d(1, channels, 3, padding=1), nn.LeakyReLU(0.1)]
        for _ in range(layers - 2):
            blocks += [nn.Conv3d(channels, channels, 3, padding=1), nn.LeakyReLU(0.1)]
        blocks.append(nn.Conv3d(channels, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.net(x), 0.0, 1.0)


def block_blind_spot_mask(x: torch.Tensor, centers_per_sample: int, radius: int, rng: np.random.Generator):
    """hide a (2r+1)^2 in-plane block around random centers, refilled with voxels drawn from random
    positions of the same slice, so the net cannot copy noise correlated with the centre's
    immediate neighbours; returns (masked input, boolean map of the centres to score)."""
    batch, _, depth, height, width = x.shape
    masked = x.clone()
    centers = torch.zeros_like(x, dtype=torch.bool)
    for b in range(batch):
        zs = rng.integers(0, depth, centers_per_sample)
        ys = rng.integers(radius, height - radius, centers_per_sample)
        xs = rng.integers(radius, width - radius, centers_per_sample)
        for z, y, xx in zip(zs, ys, xs):
            span = 2 * radius + 1
            fill_y = rng.integers(0, height, span * span)
            fill_x = rng.integers(0, width, span * span)
            fill = x[b, 0, z, fill_y, fill_x].view(span, span)
            masked[b, 0, z, y - radius:y + radius + 1, xx - radius:xx + radius + 1] = fill
            centers[b, 0, z, y, xx] = True
    return masked, centers
