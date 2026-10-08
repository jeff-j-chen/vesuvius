"""input upsampler for plan U (crossres/PLAN.md section 0.6).

LearnedUpsampler = v8-in's fixed trilinear upsampling plus a learned residual. the residual net runs on
the native 9.36 um grid (cheap) and emits depth_factor * scale^2 sub-voxels per input voxel, which a 3D
sub-voxel shuffle places on the fine grid. its last conv is zero-initialised, so an untrained module is
exactly trilinear interpolation.

network_input() is what the ink model / MAE sees: the upsampled volume with its depth pooled by
depth_pool (x4 depth then /2 = 16 slices from 8, x2 in x/y).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def subvoxel_shuffle(x: torch.Tensor, depth_factor: int, scale: int) -> torch.Tensor:
    """(B, F*S*S, D, H, W) -> (B, 1, D*F, H*S, W*S)."""
    b, _, d, h, w = x.shape
    x = x.view(b, depth_factor, scale, scale, d, h, w)
    x = x.permute(0, 4, 1, 5, 2, 6, 3)
    return x.reshape(b, 1, d * depth_factor, h * scale, w * scale)


class LearnedUpsampler(nn.Module):
    def __init__(self, depth_factor: int = 4, scale: int = 2, width: int = 48, layers: int = 8,
                 depth_pool: int = 2):
        super().__init__()
        self.depth_factor, self.scale, self.depth_pool = int(depth_factor), int(scale), int(depth_pool)
        self.width, self.layers = int(width), int(layers)
        blocks = [nn.Conv3d(1, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        for _ in range(layers - 2):
            blocks += [nn.Conv3d(width, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.out = nn.Conv3d(width, depth_factor * scale * scale, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def trilinear(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=(self.depth_factor, self.scale, self.scale),
                             mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) normalized input -> (B, 1, D*F, H*S, W*S)."""
        residual = subvoxel_shuffle(self.out(self.body(x)), self.depth_factor, self.scale)
        return self.trilinear(x) + residual

    def network_input(self, x: torch.Tensor) -> torch.Tensor:
        up = self(x)
        if self.depth_pool > 1:
            up = F.avg_pool3d(up, kernel_size=(self.depth_pool, 1, 1), stride=(self.depth_pool, 1, 1))
        return up

    def config(self) -> dict:
        return {"depth_factor": self.depth_factor, "scale": self.scale, "width": self.width,
                "layers": self.layers, "depth_pool": self.depth_pool}


def load_upsampler(spec: str, **defaults) -> LearnedUpsampler:
    """'trilinear' -> the untrained (exactly trilinear) module; otherwise a checkpoint from train_upsampler.py."""
    if spec == "trilinear":
        return LearnedUpsampler(**defaults)
    checkpoint = torch.load(spec, map_location="cpu", weights_only=False)
    module = LearnedUpsampler(**checkpoint["config"])
    module.load_state_dict(checkpoint["state_dict"])
    return module
