"""norm.py -- fast chunk-aligned normalization for zarr surface volumes.

reads the volume exactly once in chunk-aligned z/y bands so each zarr chunk is
touched once (not once per z-slice like the old inline loop). writes stats into
norm_cache.json under the segment id matching the pipeline schema.

used by DataManager._get_or_compute_norm() instead of the old per-slice tqdm loop.
also callable standalone via precompute_norm.py at the repo root.
"""
from __future__ import annotations
import json
import os
import numpy as np
import zarr


UNIFIED_CACHE_PATH = "./norm_cache.json"


def _imread_gray_pil(path: str):
    """PIL-based grayscale loader that survives >1Gpx images."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(path).convert("L"))


def compute_norm(
    scroll_id: str | int,
    zarr_path: str,
    cache_path: str = UNIFIED_CACHE_PATH,
    y_block: int = 512,
    mask_dir: str = "./masks",
) -> tuple[float, float, float, float]:
    """compute normalization stats for one scroll and write to cache.

    returns (mean, std, norm_min, norm_max) consistent with the pipeline schema.
    reads the zarr once in chunk-aligned z/y bands for speed.
    """
    sid = str(scroll_id)
    z_path = os.path.join(zarr_path, f"{sid}.zarr")
    vol = zarr.open(z_path, mode="r")
    D, H, W = map(int, vol.shape)
    zc = int(vol.chunks[0])  # chunk depth -- read whole z-bands so each chunk is hit once

    mask_file = os.path.join(mask_dir, f"{sid}.png")
    try:
        import cv2
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("cv2 returned None")
    except Exception:
        mask = _imread_gray_pil(mask_file)
    mbin = mask > 0
    print(f"[norm] {sid} vol=({D},{H},{W}) chunk_z={zc} mask_valid={float(mbin.mean()):.3f}", flush=True)

    total_sum = 0.0
    total_sq = 0.0
    total_n = 0
    raw_min = float("inf")
    raw_max = float("-inf")

    yb = ((y_block + 31) // 32) * 32
    for z0 in range(0, D, zc):
        z1 = min(z0 + zc, D)
        for y0 in range(0, H, yb):
            y1 = min(y0 + yb, H)
            block = np.asarray(vol[z0:z1, y0:y1, :])
            m = mbin[y0:y1, :]
            m3 = np.broadcast_to(m[None], block.shape)
            valid = block[m3]
            if valid.size == 0:
                continue
            v64 = valid.astype(np.float64)
            total_sum += float(v64.sum())
            total_sq += float(np.square(v64).sum())
            total_n += int(v64.size)
            raw_min = min(raw_min, float(v64.min()))
            raw_max = max(raw_max, float(v64.max()))
        print(f"[norm]   z {z0}:{z1} done  n={total_n}", flush=True)

    if total_n == 0:
        raise ValueError(f"[norm] no valid pixels under mask for {sid}")

    mean = total_sum / total_n
    std = float(np.sqrt(max(total_sq / total_n - mean * mean, 1e-12)))
    norm_min = (raw_min - mean) / std
    norm_max = (raw_max - mean) / std

    stats = {"mean": mean, "std": std, "min": norm_min, "max": norm_max}

    cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    entry = cache.get(sid, {})
    if not isinstance(entry, dict):
        entry = {}
    entry.update(stats)
    cache[sid] = entry
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=4)

    print(
        f"[norm] {sid} mean={mean:.4f} std={std:.4f} "
        f"norm_min={norm_min:.4f} norm_max={norm_max:.4f}"
    )
    return mean, std, norm_min, norm_max


def load_cached_norm(scroll_id: str | int, cache_path: str = UNIFIED_CACHE_PATH):
    """load norm stats from cache if present; return None if missing."""
    sid = str(scroll_id)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except Exception:
        return None
    entry = cache.get(sid)
    if isinstance(entry, dict) and all(k in entry for k in ("mean", "std", "min", "max")):
        return entry["mean"], entry["std"], entry["min"], entry["max"]
    return None
