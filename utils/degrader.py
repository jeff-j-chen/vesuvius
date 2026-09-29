"""plan R (crossres/PLAN.md section 0.7): a pooled high-resolution volume -> a synthetic native 9.36 um scan.

LearnedDegrader is the identity plus a zero-initialised 3D conv residual on the 28-layer training grid, so an
untrained module returns its input (today's pooled volume). the input is the pooled volume mapped by a robust
affine (median / MAD over the footprint) onto the native volumes' normalised intensity distribution, whose
median / MAD are stored with the checkpoint. the output is in native normalised units; translate_zarr maps it
back to raw values with a fixed reference native scroll's norm_cache.json entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn

REFERENCE_NATIVE_ID = "20260115000000"  # w044: the typical, cleanest native norm
TRANSLATED_SUFFIX = ".translated"


class LearnedDegrader(nn.Module):
    def __init__(self, width: int = 48, layers: int = 8, ref_median: float = 0.5, ref_mad: float = 0.1):
        super().__init__()
        self.width, self.layers = int(width), int(layers)
        self.ref_median, self.ref_mad = float(ref_median), float(ref_mad)
        blocks = [nn.Conv3d(1, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        for _ in range(layers - 2):
            blocks += [nn.Conv3d(width, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        blocks.append(nn.Conv3d(width, 1, 3, padding=1, padding_mode="replicate"))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def halo(self) -> int:
        """receptive-field radius: one voxel per 3x3x3 conv."""
        return self.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) reference-mapped pooled volume -> native normalised units."""
        return x + self.net(x)

    def config(self) -> dict:
        return {"width": self.width, "layers": self.layers, "ref_median": self.ref_median, "ref_mad": self.ref_mad}


def load_degrader(path: str, device="cpu") -> LearnedDegrader:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = LearnedDegrader(**checkpoint["config"])
    module.load_state_dict(checkpoint["state_dict"])
    return module.to(device).eval().requires_grad_(False)


def checkpoint_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """(median, MAD scaled to a normal std) of the nonzero values."""
    values = np.asarray(values, np.float64)
    values = values[values > 0]
    if values.size == 0:
        raise ValueError("no nonzero voxels for robust statistics")
    median = float(np.median(values))
    return median, float(np.median(np.abs(values - median)) * 1.4826 + 1e-6)


def to_reference(x, stats: tuple[float, float], module: LearnedDegrader):
    """pooled raw intensities -> the native normalised distribution; zeros (outside the footprint) stay 0."""
    median, mad = stats
    mapped = (x - median) / mad * module.ref_mad + module.ref_median
    return mapped * (x > 0)


def is_current(dst: str, checkpoint: str) -> bool:
    try:
        with open(os.path.join(dst, ".zattrs"), encoding="utf-8") as handle:
            return json.load(handle).get("degrader") == checkpoint_digest(checkpoint)
    except (OSError, ValueError):
        return False


def _volume_stats(volume, mask: np.ndarray, band: int = 128, every: int = 8, step: int = 4) -> tuple[float, float]:
    samples = []
    height = int(volume.shape[1])
    for y0 in range(0, height, band * every):
        block = np.asarray(volume[:, y0:y0 + band, ::step], np.float32)
        samples.append(block[:, mask[y0:y0 + band, ::step]].ravel())
    return robust_stats(np.concatenate(samples))


def translate_zarr(src: str, dst: str, checkpoint: str, mask: np.ndarray, reference_norm: tuple,
                   device: str | None = None, tile: int = 256) -> dict:
    """stream src in (all layers, tile, tile) blocks with a halo, translate, write dst (same shape, chunks, dtype).

    reference_norm = (mean, std, min, max) of the reference native scroll (norm_cache.json): the translated
    normalised output is mapped back to raw values with it and clipped to 0-255.
    """
    import zarr

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = load_degrader(checkpoint, device)
    source = zarr.open(src, mode="r")
    depth, height, width = map(int, source.shape)
    footprint_mask = np.zeros((height, width), bool)
    mh, mw = min(height, mask.shape[0]), min(width, mask.shape[1])
    footprint_mask[:mh, :mw] = np.asarray(mask[:mh, :mw]) > 0
    mask = footprint_mask
    stats = _volume_stats(source, mask)
    mean, std, g_min, g_max = (float(v) for v in reference_norm)
    halo = module.halo
    partial = dst.rstrip("/") + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    output = zarr.open(partial, mode="w", shape=source.shape, chunks=source.chunks, dtype=source.dtype,
                       compressor=None, zarr_format=2)
    print(f"[degrader] {src} -> {dst}: input median/MAD {stats[0]:.2f}/{stats[1]:.2f}, "
          f"{module.config()}, device {device}", flush=True)
    for y0 in range(0, height, tile):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, tile):
            x1 = min(x0 + tile, width)
            footprint = mask[y0:y1, x0:x1]
            if not footprint.any():
                continue
            ya, yb, xa, xb = max(0, y0 - halo), min(height, y1 + halo), max(0, x0 - halo), min(width, x1 + halo)
            block = np.asarray(source[:, ya:yb, xa:xb], np.float32)
            with torch.no_grad():
                x = torch.from_numpy(to_reference(block, stats, module)).to(device)[None, None]
                y = module(x)[0, 0].cpu().numpy()
            y = y[:, y0 - ya:y0 - ya + (y1 - y0), x0 - xa:x0 - xa + (x1 - x0)]
            raw = (y * (g_max - g_min) + g_min) * std + mean
            present = block[:, y0 - ya:y0 - ya + (y1 - y0), x0 - xa:x0 - xa + (x1 - x0)] > 0
            raw = np.where(present & footprint[None], np.clip(np.rint(raw), 1, 255), 0)
            output[:, y0:y1, x0:x1] = raw.astype(source.dtype)
        print(f"[degrader] rows {y1}/{height}", flush=True)
    attrs = {"degrader": checkpoint_digest(checkpoint), "degrader_checkpoint": os.path.basename(checkpoint),
             "source": os.path.basename(src.rstrip("/")), "input_median_mad": list(stats),
             "reference_norm": list(reference_norm)}
    with open(os.path.join(partial, ".zattrs"), "w", encoding="utf-8") as handle:
        json.dump(attrs, handle, indent=1)
    del output
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(partial, dst)
    return attrs
