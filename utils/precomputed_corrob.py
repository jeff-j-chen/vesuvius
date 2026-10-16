"""precomputed whole-patch "corroborated" cue (nonml_ink.py: ten 2D priors, robust z, blurred 15 px, top-4 mean lifted where
the priors agree), a drop-in for the CLAHE map of utils/precomputed_cue.py (same model.input_cue = "pre_clahe" mechanism,
data.cue_dir = "./cue_corrob_s15").

the priors run on the 8-slice window the model reads at every pixel (surface-relative start plus the scroll's window offset,
raw volume units; every prior is scale-free), on the 2x area-downsampled grid as in nonml_ink.py, then the map is upsampled
to the volume's pixel grid and stored as uint8 (0-255 = 0..1, fixed scale, 0 outside the papyrus mask).

python3 -m utils.precomputed_corrob [--only 20260927134931,...] [--workers 5] [--force]   (default: the five fine-tune scrolls)
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .precomputed_cue import WINDOW, volume_id_of

CUE_DIR = "cue_corrob_s15"
BAND = 256
BLUR_NATIVE = 15
FINE_TUNE_IDS = (20260927134931, 20261012073246, 20261012065811, 20261012065813, 20250919125754)


def build(scroll_id: int, force: bool = False) -> str:
    import zarr
    from PIL import Image

    import nonml_ink as nm
    from utils.config import Config

    Image.MAX_IMAGE_PIXELS = None
    cv2.setNumThreads(6)
    config = Config()
    volume_id = volume_id_of(config, scroll_id)
    out_path = os.path.join(CUE_DIR, f"{volume_id}.npy")
    if os.path.isfile(out_path) and not force:
        return f"{volume_id}: exists"
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
    f = nm.F
    h2, w2 = height // f, width // f
    stack = np.zeros((WINDOW, height, width), dtype=np.float32)
    for y0 in range(0, height, BAND):
        y1 = min(height, y0 + BAND)
        block = np.asarray(volume[:, y0:y1, :], dtype=np.float32)
        d = np.asarray(depth[y0:y1, :width]).astype(np.int64)
        c = np.asarray(confidence[y0:y1, :width])
        centre = np.where((d != 255) & (c > 0), d, fallback)
        start = np.clip(centre - (WINDOW - 1) // 2 + offset, 0, max(depth_count - WINDOW, 0))
        for k in range(WINDOW):
            index = np.clip(start + k, 0, depth_count - 1)
            stack[k, y0:y1] = np.take_along_axis(block, index[None], axis=0)[0]
    v = stack[:, :h2 * f, :w2 * f].reshape(WINDOW, h2, f, w2, f).mean((2, 4)).astype(np.float32)
    del stack
    m = frame[:h2 * f, :w2 * f].reshape(h2, f, w2, f).mean((1, 3)) > 0.999
    m = cv2.erode(m.astype(np.uint8), nm.disk(24)).astype(bool)
    nm.MIP_RANGE = list(range(WINDOW))
    z = np.stack([nm.robust_z(x, m, nm.Z_CLIP) for x in nm.priors(v, m).values()]).astype(np.float32)
    zs = np.stack([nm.robust_z(nm.gblur(layer, BLUR_NATIVE), m, nm.Z_CLIP) for layer in z])
    small = nm.corroborated(zs, m)
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[:h2 * f, :w2 * f] = cv2.resize(small, (w2 * f, h2 * f), interpolation=cv2.INTER_LINEAR)
    cue = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    cue[~frame] = 0
    os.makedirs(CUE_DIR, exist_ok=True)
    temporary = out_path + ".tmp.npy"
    np.save(temporary, cue)
    os.replace(temporary, out_path)
    return f"{volume_id}: {height}x{width}"


def _build(args):
    return build(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated scroll ids (default: the five fine-tune scrolls)")
    parser.add_argument("--all", action="store_true", help="every scroll and vis scroll of the default run")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    ids = [int(s) for s in args.only.split(",") if s.strip()] or list(FINE_TUNE_IDS)
    if args.all:
        from utils.config import Config

        config = Config()
        ids = list(dict.fromkeys([int(s.scroll_id) for s in config.data.scrolls] + [int(s) for s in config.data.vis_scroll_ids]))
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(ids)))) as pool:
        for line in pool.map(_build, [(sid, args.force) for sid in ids]):
            print(f"[corrob] {line}", flush=True)


if __name__ == "__main__":
    main()
