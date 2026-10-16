"""fixed image filters ported from nonml_ink.py, run on the max over slices of the normalized (B, 1, D, H, W) input.

asf: alternating open/close with growing octagons (radii in native px, as nonml_ink.asf), removing bright and dark
structures thinner than the largest one. clahe: contrast-limited adaptive histogram equalization on a tile grid
with bilinear blending of the tile LUTs, computed from each crop alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ASF_RADII = (2, 4, 6, 8, 10, 12)


def _dilate(x: torch.Tensor, radius: int) -> torch.Tensor:
    """octagon of inradius `radius`: alternating 3x3 square and 3x3 cross steps."""
    for step in range(radius):
        if step % 2 == 0:
            x = F.max_pool2d(x, 3, 1, 1)
        else:
            x = torch.maximum(F.max_pool2d(x, (1, 3), 1, (0, 1)), F.max_pool2d(x, (3, 1), 1, (1, 0)))
    return x


def _erode(x: torch.Tensor, radius: int) -> torch.Tensor:
    return -_dilate(-x, radius)


def asf(x: torch.Tensor, radii=ASF_RADII) -> torch.Tensor:
    for radius in radii:
        x = _dilate(_erode(x, radius), radius)
        x = _erode(_dilate(x, radius), radius)
    return x


def clahe(x: torch.Tensor, tiles: int = 3, bins: int = 64, clip: float = 2.5) -> torch.Tensor:
    """x: (N, 1, H, W) in [0, 1]."""
    n, _, height, width = x.shape
    th, tw = math.ceil(height / tiles), math.ceil(width / tiles)
    padded = F.pad(x, (0, tw * tiles - width, 0, th * tiles - height), mode="reflect")
    q = (padded.clamp(0, 1) * (bins - 1)).round().long()
    blocks = q.view(n, tiles, th, tiles, tw).permute(0, 1, 3, 2, 4).reshape(n, tiles * tiles, th * tw)
    hist = torch.zeros(n, tiles * tiles, bins, device=x.device).scatter_add_(
        2, blocks, torch.ones_like(blocks, dtype=torch.float32))
    limit = clip * th * tw / bins
    excess = (hist - limit).clamp(min=0).sum(-1, keepdim=True)
    hist = hist.clamp(max=limit) + excess / bins
    lut = hist.cumsum(-1) / hist.sum(-1, keepdim=True)
    mapped = torch.gather(lut, 2, q.view(n, 1, -1).expand(-1, tiles * tiles, -1))
    mapped = mapped.view(n, tiles, tiles, tiles * th, tiles * tw)

    def hat(size, tile):
        centres = (torch.arange(tiles, device=x.device) + 0.5) * tile
        return (1 - (torch.arange(size, device=x.device)[:, None] + 0.5 - centres[None]).abs() / tile).clamp(min=0)

    wy, wx = hat(tiles * th, th), hat(tiles * tw, tw)
    wy, wx = wy / wy.sum(1, keepdim=True), wx / wx.sum(1, keepdim=True)
    out = torch.einsum("nabyx,ya,xb->nyx", mapped, wy, wx)
    return out[:, None, :height, :width]


def apply_prefilter(x: torch.Tensor, kind: str, mix: float = 1.0) -> torch.Tensor:
    """filter the max over slices of x (B, 1, D, H, W) and add mix * (filtered - max) to every slice, which keeps the
    depth structure of x while the in-plane map takes on the filter's change."""
    batch, _, depth, height, width = x.shape
    peak = x.float().amax(dim=2)
    if kind == "asf":
        filtered = asf(peak)
    elif kind == "clahe":
        filtered = clahe(peak)
    else:
        raise ValueError(f"unknown input_prefilter {kind!r}")
    return (x.float() + mix * (filtered - peak).unsqueeze(2)).clamp(0.0, 1.0).to(x.dtype)


def peak_map(x: torch.Tensor, kind: str) -> torch.Tensor:
    """asf or clahe of the max over slices of x (B, 1, D, H, W), as (B, 1, 1, H, W) in [0, 1]."""
    peak = x.float().amax(dim=2)
    if kind == "asf":
        out = asf(peak)
    elif kind == "clahe":
        out = clahe(peak)
    else:
        raise ValueError(f"unknown input_cue {kind!r}")
    return out.unsqueeze(2)
