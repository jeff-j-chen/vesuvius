"""learned input upsampler: trilinear interpolation plus a residual learned from co-registered ~2.4 um scans."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedUpsampler(nn.Module):
    """(B, 1, D, H, W) 9.36 um window -> (B, 1, D * depth_factor, H * xy_scale, W * xy_scale).

    the residual head is zero-initialised, so an untrained upsampler is exactly the trilinear
    interpolation v8-in uses; training on paired scans can only add detail on top of it.
    """

    def __init__(self, xy_scale: int = 2, depth_factor: int = 4, channels: int = 48, layers: int = 8):
        super().__init__()
        self.xy_scale, self.depth_factor = int(xy_scale), int(depth_factor)
        blocks = [nn.Conv3d(1, channels, 3, padding=1), nn.GELU()]
        for _ in range(layers - 2):
            blocks += [nn.Conv3d(channels, channels, 3, padding=1), nn.GELU()]
        blocks.append(nn.Conv3d(channels, self.depth_factor * self.xy_scale ** 2, 1))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f, s = self.depth_factor, self.xy_scale
        base = F.interpolate(x, scale_factor=(f, s, s), mode="trilinear", align_corners=False)
        residual = self.net(x)
        batch, _, depth, height, width = residual.shape
        # sub-voxel shuffle: channel (i, j, k) of input voxel (d, h, w) -> output (d*f+i, h*s+j, w*s+k)
        residual = residual.view(batch, f, s, s, depth, height, width).permute(0, 4, 1, 5, 2, 6, 3)
        return base + residual.reshape(batch, 1, depth * f, height * s, width * s)

    def config(self) -> dict:
        convs = [m for m in self.net if isinstance(m, nn.Conv3d)]
        return {"xy_scale": self.xy_scale, "depth_factor": self.depth_factor,
                "channels": convs[0].out_channels, "layers": len(convs)}


def load_upsampler(path: str, device="cpu") -> LearnedUpsampler:
    payload = torch.load(path, map_location=device, weights_only=False)
    module = LearnedUpsampler(**payload["config"])
    module.load_state_dict(payload["state_dict"])
    return module.to(device).eval().requires_grad_(False)
