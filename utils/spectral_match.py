"""deterministic spectral matching of the downscaled fine scans to the native 9.36 um / 113 keV regime.

the learned degrader's .translated copies carry 7-15x the high-frequency power of the native volumes and of the
(trilinear) test renders. this fits one gaussian point-spread (sigma_xy in-plane, sigma_z in depth) per fine
scroll so its in-plane spectrum shape and depth correlation match the native reference, then writes a filtered
<id>.matched sibling. noise is not synthesised here; the native noise bank still adds real scan noise per crop.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from scipy.ndimage import gaussian_filter

MATCHED_SUFFIX = ".matched"
# natively scanned 9.36 um / 113 keV training volumes (published trilinear renders); the trilinear test renders
# measure inside the same band
REFERENCE_NATIVE_IDS = ("20260115000000", "20260317000000", "20250108000005", "20260112000000",
                        "20260221022814", "20250628074500", "20260226000000")
SIGMA_XY_GRID = (0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.6, 1.8)
SIGMA_Z_GRID = (0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)


def load_crops(path: str, count: int = 8, size: int = 256, seed: int = 0) -> np.ndarray:
    """(count, depth, size, size) float32 crops whose middle slice lies fully inside the data."""
    import zarr

    volume = zarr.open(path, mode="r")
    depth, height, width = map(int, volume.shape)
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(800):
        if len(crops) >= count:
            break
        y, x = int(rng.integers(0, height - size)), int(rng.integers(0, width - size))
        crop = np.asarray(volume[:, y:y + size, x:x + size], np.float32)
        if (crop[depth // 2 - 1:depth // 2 + 2] > 0).mean() > 0.99:
            crops.append(crop)
    if not crops:
        raise RuntimeError(f"{path}: no fully valid {size}px crops")
    return np.stack(crops)


def crop_stats(crops: np.ndarray, sigma_xy: float = 0.0, sigma_z: float = 0.0) -> np.ndarray:
    """[log hi/mid, log midhi/mid, depth lag-1 correlation] of the middle slice, averaged over crops."""
    out = []
    for crop in crops:
        if sigma_xy or sigma_z:
            crop = gaussian_filter(crop, (sigma_z, sigma_xy, sigma_xy), mode="nearest")
        middle = crop.shape[0] // 2
        s = crop[middle]
        s = (s - s.mean()) / (s.std() + 1e-6)
        power = np.abs(np.fft.fftshift(np.fft.fft2(s))) ** 2
        c = s.shape[0] // 2
        yy, xx = np.indices(power.shape)
        r = np.hypot(yy - c, xx - c) / c
        mid = power[(r > 0.125) & (r < 0.25)].mean()
        midhi = power[(r > 0.25) & (r < 0.5)].mean()
        hi = power[(r > 0.5) & (r < 1.0)].mean()
        a, b = crop[middle - 1].ravel(), crop[middle].ravel()
        out.append([np.log(hi / mid), np.log(midhi / mid), np.corrcoef(a, b)[0, 1]])
    return np.mean(out, axis=0)


def reference_stats(zarr_dir: str) -> np.ndarray:
    return np.mean([crop_stats(load_crops(os.path.join(zarr_dir, f"{zid}.zarr"))) for zid in REFERENCE_NATIVE_IDS],
                   axis=0)


def _distance(stats: np.ndarray, reference: np.ndarray) -> float:
    # log band ratios plus depth correlation (x5: it spans ~0.1 where the log ratios span ~1)
    return float(abs(stats[0] - reference[0]) + abs(stats[1] - reference[1]) + 5.0 * abs(stats[2] - reference[2]))


def fit_sigmas(crops: np.ndarray, reference: np.ndarray) -> tuple[float, float, np.ndarray]:
    best = None
    for sigma_xy in SIGMA_XY_GRID:
        for sigma_z in SIGMA_Z_GRID:
            stats = crop_stats(crops, sigma_xy, sigma_z)
            score = _distance(stats, reference)
            if best is None or score < best[0]:
                best = (score, sigma_xy, sigma_z, stats)
    return best[1], best[2], best[3]


def match_zarr(src: str, dst: str, sigma_xy: float, sigma_z: float, band: int = 256) -> None:
    """masked (normalised-convolution) gaussian over all layers, streamed in row bands with a halo."""
    import zarr

    source = zarr.open(src, mode="r")
    depth, height, width = map(int, source.shape)
    halo = int(np.ceil(4 * max(sigma_xy, 0.5)))
    partial = dst.rstrip("/") + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    output = zarr.open(partial, mode="w", shape=source.shape, chunks=source.chunks, dtype=source.dtype,
                       compressor=None, zarr_format=2)
    sigma = (sigma_z, sigma_xy, sigma_xy)
    for y0 in range(0, height, band):
        y1 = min(y0 + band, height)
        ya, yb = max(0, y0 - halo), min(height, y1 + halo)
        block = np.asarray(source[:, ya:yb, :], np.float32)
        valid = (block > 0).astype(np.float32)
        blurred = gaussian_filter(block * valid, sigma, mode="nearest")
        weight = gaussian_filter(valid, sigma, mode="nearest")
        result = np.where(valid > 0, np.clip(np.rint(blurred / np.maximum(weight, 1e-3)), 1, None), 0)
        output[:, y0:y1, :] = result[:, y0 - ya:y0 - ya + (y1 - y0)].astype(source.dtype)
        print(f"[match] {os.path.basename(dst)} rows {y1}/{height}", flush=True)
    with open(os.path.join(partial, ".zattrs"), "w", encoding="utf-8") as handle:
        json.dump({"spectral_match": {"sigma_xy": sigma_xy, "sigma_z": sigma_z,
                                      "source": os.path.basename(src.rstrip("/")),
                                      "reference_ids": list(REFERENCE_NATIVE_IDS)}}, handle, indent=1)
    del output
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(partial, dst)


def is_matched(path: str) -> bool:
    try:
        with open(os.path.join(path, ".zattrs"), encoding="utf-8") as handle:
            return "spectral_match" in json.load(handle)
    except (OSError, ValueError):
        return False
