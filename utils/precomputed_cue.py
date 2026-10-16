"""precomputed whole-patch CLAHE cue: the fourth gated input head, built once per volume instead of per crop.

for every pixel the cue is the max over the 8-slice window the model reads there (surface-relative start plus the scroll's
window offset), in the surface-anchor/global normalized units the model sees, equalized with OpenCV CLAHE over the whole
patch (tiles of ~400 px, clip 2.5, papyrus-free pixels filled with the papyrus median, as nonml_ink.py does). stored as
uint8 (0-255 = 0..1) in cue_clahe/<volume id>.npy, on the volume's pixel grid.

python3 -m utils.precomputed_cue [--only 20260927134931,...] [--workers 12] [--force]   (default: every scroll of the default run)
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CUE_DIR = "cue_clahe"
WINDOW = 8
TILE_PX = 400
CLIP = 2.5
BAND = 256


def cue_file(volume_id, cue_dir=CUE_DIR) -> str:
    return os.path.join(cue_dir, f"{volume_id}.npy")


def volume_id_of(config, scroll_id) -> str:
    return f"{scroll_id}{(config.data.zarr_suffix or {}).get(str(scroll_id), '')}"


def load_cue(config, scroll_id):
    """uint8 cue map of a scroll (memory-mapped), or None when the model has no precomputed cue."""
    if str(getattr(config.model, "input_cue", "") or "") != "pre_clahe":
        return None
    path = cue_file(volume_id_of(config, scroll_id), getattr(config.data, "cue_dir", CUE_DIR))
    if not os.path.isfile(path):
        # build it on the spot: the asf_morph map for an asf cue dir (slow, minutes per patch), CLAHE otherwise
        if "asf" in os.path.basename(os.path.normpath(str(config.data.cue_dir))):
            from . import precomputed_asf
            print(f"[cue] building {path} ...", flush=True)
            precomputed_asf.build(scroll_id, config=config)
        elif os.path.dirname(path).rstrip("/") in (CUE_DIR, "./" + CUE_DIR):
            print(f"[cue] building {path} ...", flush=True)
            build(scroll_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing precomputed cue {path}; run python3 -m utils.precomputed_cue / precomputed_asf")
    return np.load(path, mmap_mode="r")


def crop_cue(cue, y0, x0, height, width):
    """float32 (1, height, width) in [0, 1]; zero outside the map (matches the zero-padded raw reads)."""
    out = np.zeros((1, height, width), dtype=np.float32)
    h, w = cue.shape
    ys, ye, xs, xe = max(0, y0), min(h, y0 + height), max(0, x0), min(w, x0 + width)
    if ys < ye and xs < xe:
        out[0, ys - y0:ye - y0, xs - x0:xe - x0] = np.asarray(cue[ys:ye, xs:xe], dtype=np.float32) / 255.0
    return out


def build(scroll_id: int, force: bool = False) -> str:
    import zarr
    from PIL import Image

    from utils.config import Config
    from utils.norm import load_cached_norm, scale_to_unit

    Image.MAX_IMAGE_PIXELS = None
    config = Config()
    volume_id = volume_id_of(config, scroll_id)
    out_path = cue_file(volume_id)
    if os.path.isfile(out_path) and not force:
        return f"{volume_id}: exists"
    mode = str(config.data.norm_mode)
    mean, std, g_min, g_max = load_cached_norm(volume_id, "./norm_cache.json", mode=mode)
    volume = zarr.open(os.path.join(config.data.zarr_path, f"{volume_id}.zarr"), mode="r")
    depth_count, height, width = (int(v) for v in volume.shape)
    mask = np.array(Image.open(f"masks/{scroll_id}.png").convert("L")) > 0
    frame = np.zeros((height, width), dtype=bool)
    h, w = min(height, mask.shape[0]), min(width, mask.shape[1])
    frame[:h, :w] = mask[:h, :w]
    surface_dir = os.path.join(config.data.surface_label_dir, str(scroll_id))
    depth = np.load(os.path.join(surface_dir, "depth.npy"), mmap_mode="r")
    confidence = np.load(os.path.join(surface_dir, "confidence.npy"), mmap_mode="r")
    offset = int(config.data.surface_window_offset_by_scroll.get(int(scroll_id), config.data.surface_window_offset))
    valid_depth = np.asarray(depth[::16, ::16]).astype(np.int64)
    valid_depth = valid_depth[valid_depth != 255]
    fallback = int(np.median(valid_depth)) if valid_depth.size else config.data.train_d_start
    peak = np.zeros((height, width), dtype=np.float32)
    for y0 in range(0, height, BAND):
        y1 = min(height, y0 + BAND)
        block = np.asarray(volume[:, y0:y1, :], dtype=np.float32)
        d = np.asarray(depth[y0:y1, :width]).astype(np.int64)
        c = np.asarray(confidence[y0:y1, :width])
        centre = np.where((d != 255) & (c > 0), d, fallback)
        start = np.clip(centre - (WINDOW - 1) // 2 + offset, 0, max(depth_count - WINDOW, 0))
        best = np.full((y1 - y0, width), -np.inf, dtype=np.float32)
        for k in range(WINDOW):
            index = np.clip(start + k, 0, depth_count - 1)
            best = np.maximum(best, np.take_along_axis(block, index[None], axis=0)[0])
        peak[y0:y1] = scale_to_unit((best - mean) / std, g_min, g_max, mode)
    inside = frame & np.isfinite(peak)
    u8 = np.clip(peak * 255.0, 0, 255).astype(np.uint8)
    u8[~inside] = int(np.median(u8[inside])) if inside.any() else 0
    tiles = (max(2, width // TILE_PX), max(2, height // TILE_PX))
    cue = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=tiles).apply(u8)
    cue[~inside] = 0
    os.makedirs(CUE_DIR, exist_ok=True)
    temporary = out_path + ".tmp.npy"
    np.save(temporary, cue)
    os.replace(temporary, out_path)
    return f"{volume_id}: {height}x{width} tiles={tiles}"


def _build(args):
    return build(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated scroll ids (default: every default-run scroll and vis scroll)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    from utils.config import Config

    config = Config()
    if args.only:
        ids = [int(s) for s in args.only.split(",") if s.strip()]
    else:
        ids = list(dict.fromkeys([int(s.scroll_id) for s in config.data.scrolls] + [int(s) for s in config.data.vis_scroll_ids]))
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(ids)))) as pool:
        for line in pool.map(_build, [(sid, args.force) for sid in ids]):
            print(f"[cue] {line}", flush=True)


if __name__ == "__main__":
    main()
