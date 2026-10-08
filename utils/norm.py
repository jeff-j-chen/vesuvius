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
    mask_id: str | int | None = None,
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

    mask_file = os.path.join(mask_dir, f"{sid if mask_id is None else mask_id}.png")
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


def load_cached_norm(scroll_id: str | int, cache_path: str = UNIFIED_CACHE_PATH, mode: str = "global"):
    """load norm stats from cache if present; return None if missing.

    mode "surface_anchor" maps the scroll's gap level to 0.1 and its surface papyrus level to 0.5.
    """
    if mode == "raw255":
        return 0.0, 1.0, 0.0, 255.0
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
        stats = entry["mean"], entry["std"], entry["min"], entry["max"]
    else:
        stats = None
    if mode == "global":
        return stats
    if mode != "surface_anchor":
        raise ValueError(f"unknown norm mode {mode!r}")
    anchors = _read_json(SURFACE_ANCHOR_CACHE_PATH).get(sid)
    if stats is None or not isinstance(anchors, dict):
        raise RuntimeError(f"surface-anchored normalization needs global and anchor stats for {sid}")
    mean, std = stats[:2]
    gap = (anchors["gap"] - mean) / std
    papyrus = (anchors["papyrus"] - mean) / std
    span = (papyrus - gap) / (ANCHOR_PAPYRUS_LEVEL - ANCHOR_GAP_LEVEL)
    lower = gap - ANCHOR_GAP_LEVEL * span
    return mean, std, lower, lower + span


SURFACE_ANCHOR_CACHE_PATH = "./surface_anchor_cache.json"
# normalized levels the gap and the surface papyrus are pinned to
ANCHOR_GAP_LEVEL = 0.1
ANCHOR_PAPYRUS_LEVEL = 0.5
SUPPORT_THRESHOLD = 0.5 * (ANCHOR_GAP_LEVEL + ANCHOR_PAPYRUS_LEVEL)


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def compute_surface_anchors(
    scroll_id: str | int,
    zarr_path: str,
    surface_dir: str = "./surface_labels",
    mask_dir: str = "./masks",
    tiles: int = 384,
    tile: int = 64,
    seed: int = 0,
) -> dict:
    """raw gap level (1st percentile) and papyrus level (median at the fitted surface) from sampled tiles."""
    import cv2
    sid = str(scroll_id)
    # sibling volumes (<id>.translated) share the original's mask and fitted surface
    base = sid.split(".", 1)[0]
    vol = zarr.open(os.path.join(zarr_path, f"{sid}.zarr"), mode="r")
    _, height, width = map(int, vol.shape)
    mask = cv2.imread(os.path.join(mask_dir, f"{base}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = _imread_gray_pil(os.path.join(mask_dir, f"{base}.png"))
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = np.load(os.path.join(surface_dir, base, "depth.npy"), mmap_mode="r")
    confidence = np.load(os.path.join(surface_dir, base, "confidence.npy"), mmap_mode="r")
    rows, cols = height // tile, width // tile
    valid = ((mask > 0) & (np.asarray(depth) != 255) & (np.asarray(confidence) > 0))
    coverage = valid[:rows * tile, :cols * tile].reshape(rows, tile, cols, tile).mean(axis=(1, 3))
    candidates = np.argwhere(coverage >= 0.9)
    if len(candidates) == 0:
        raise RuntimeError(f"[anchor] {sid} has no tiles with a fitted surface under the mask")
    rng = np.random.default_rng(seed)
    picks = candidates[rng.permutation(len(candidates))[:tiles]]
    all_values, surface_values, noise_values = [], [], []
    for row, col in picks:
        y, x = int(row) * tile, int(col) * tile
        block = np.asarray(vol[:, y:y + tile, x:x + tile], dtype=np.float32)
        keep = valid[y:y + tile, x:x + tile]
        index = np.clip(np.asarray(depth[y:y + tile, x:x + tile], dtype=np.int64), 0, block.shape[0] - 1)
        surface = np.take_along_axis(block, index[None], axis=0)[0]
        values = block[:, keep].ravel()
        # exact zeros are padding outside the scanned data, not air
        all_values.append(values[values > 0])
        pairs = keep[:, 1:] & keep[:, :-1] & (surface[:, 1:] > 0) & (surface[:, :-1] > 0)
        noise_values.append(np.abs(np.diff(surface, axis=1))[pairs])
        surface = surface[keep]
        surface_values.append(surface[surface > 0])
    all_values = np.concatenate(all_values)
    anchors = {
        "gap": float(np.percentile(all_values, 1.0)),
        "papyrus": float(np.median(np.concatenate(surface_values))),
        # robust pixel-to-pixel spread at the surface (texture plus noise), raw units
        "surface_mad": float(1.4826 * np.median(np.concatenate(noise_values)) / np.sqrt(2.0)),
        "tiles": int(len(picks)),
    }
    if anchors["papyrus"] <= anchors["gap"]:
        raise RuntimeError(f"[anchor] {sid} papyrus level is not above the gap level: {anchors}")
    return anchors


def ensure_surface_anchors(scroll_ids, zarr_path: str, surface_dir: str = "./surface_labels") -> dict:
    """compute and cache anchors for any scroll that lacks them; returns the full anchor cache."""
    cache = _read_json(SURFACE_ANCHOR_CACHE_PATH)
    for scroll_id in scroll_ids:
        sid = str(scroll_id)
        if sid in cache:
            continue
        cache[sid] = compute_surface_anchors(sid, zarr_path, surface_dir=surface_dir)
        print(f"[anchor] {sid} {cache[sid]}", flush=True)
        temporary = f"{SURFACE_ANCHOR_CACHE_PATH}.tmp"
        with open(temporary, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(temporary, SURFACE_ANCHOR_CACHE_PATH)
    return cache
