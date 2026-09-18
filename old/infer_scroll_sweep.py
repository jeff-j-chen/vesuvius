"""infer_scroll_sweep.py — run a trained ink detector across EVERY depth of a rendered
surface volume and dump one prediction figure per depth window + a max-over-depth composite.

this is the "test_int visualization" applied to an arbitrary (label-free) scroll: it reuses
the exact per-tile inference used for the scroll2/3/4 transfer figures (utils.visualizer.
predict_tiles), but sweeps the depth axis at z_step=1 so you see the model's response at
every single starting layer, not just the coarse depth//2 blocks.

default target = PHerc0191 w044 render (id 20260715114436, 96 layers) with the dense_unet
detector trained on PHerc0139 9.3um (preserved at models/ink_p0139_9um_dense_unet_final.pth).

usage:
  python infer_scroll_sweep.py                          # defaults (PHerc0191 + dense_unet)
  python infer_scroll_sweep.py --scroll-id <id> --weights <pth> --arch dense_unet --z-step 1
outputs (--out-dir, default runs_infer/<id>/):
  depth_<dstart>.png     per-depth prediction heatmap (inferno)
  composite_max.png      max ink-prob across all depth windows (the "is there ink anywhere" map)
  surface_ref.png        the raw mid-layer surface for visual reference
  pred_stack.npy         (n_depths, h_small, w_small) raw prob maps for later analysis
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import zarr
from utils.config import Config
from utils.model import create_model
from utils.training_utils import load_model
from utils.visualizer import predict_tiles
from utils.dataloader import imread_gray


def load_norm(seg_id):
    """read mean/std/min/max from norm_cache.json for the given id (raises if absent)."""
    with open("norm_cache.json") as f:
        cache = json.load(f)
    if seg_id not in cache:
        raise SystemExit(f"norm not cached for {seg_id}; run: python precompute_norm.py --scroll-id {seg_id}")
    s = cache[seg_id]
    return s["mean"], s["std"], s["min"], s["max"]


def gen_coords(mask, y_range, x_range, tile, d_start):
    """all tile-aligned (d_start, y_off, x_off) covering mask>0 in the region."""
    y0, y1 = y_range
    x0, x1 = x_range
    coords = []
    for y_off in range(0, (y1 - y0) - tile + 1, tile):
        for x_off in range(0, (x1 - x0) - tile + 1, tile):
            blk = mask[y0 + y_off:y0 + y_off + tile, x0 + x_off:x0 + x_off + tile]
            if np.any(blk > 0):
                coords.append((d_start, y_off, x_off))
    return coords


def main():
    ap = argparse.ArgumentParser(description="sweep a trained detector over every depth of a surface volume")
    ap.add_argument("--scroll-id", default="20260715114436", help="rendered surface-volume zarr id")
    ap.add_argument("--weights", default="models/ink_p0139_9um_dense_unet_final.pth")
    ap.add_argument("--arch", default="dense_unet")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4, help="depth window fed to the model (match training)")
    ap.add_argument("--z-step", type=int, default=1, help="1 = every single depth (default)")
    ap.add_argument("--smooth-sigma", type=float, default=0.0)
    ap.add_argument("--zarr-path", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    c = Config()
    if args.zarr_path:
        c.data.zarr_path = args.zarr_path
    c.data.tile_size = args.tile_size
    c.data.depth = args.depth
    c.data.input_mode = "single"
    c.data.smooth_sigma = args.smooth_sigma
    c.model.arch = args.arch
    # dropout off for inference-shaped model build (matches trained dense_unet)
    c.model.conv1_drop = c.model.conv2_drop = c.model.fc1_drop = c.model.fc2_drop = 0.0

    sid = str(args.scroll_id)
    out_dir = args.out_dir or os.path.join("runs_infer", sid)
    os.makedirs(out_dir, exist_ok=True)

    # data
    zpath = os.path.join(c.data.zarr_path, f"{sid}.zarr")
    vol = zarr.open(zpath, mode="r")
    D, H, W = map(int, vol.shape)
    mask = imread_gray(f"./masks/{sid}.png")
    if mask is None:
        raise SystemExit(f"mask not found: ./masks/{sid}.png")
    mask = (mask / 255.0)
    g_mean, g_std, g_min, g_max = load_norm(sid)
    print(f"[infer] {sid} vol=({D},{H},{W}) norm mean={g_mean:.2f} std={g_std:.2f} "
          f"min={g_min:.3f} max={g_max:.3f}")

    # model
    model, _ = create_model(c)
    load_model(model, args.weights)
    model.eval()
    print(f"[infer] loaded {args.arch} from {args.weights}")

    y_range = (0, H)
    x_range = (0, W)
    depth = args.depth
    depth_starts = list(range(0, D - depth + 1, args.z_step))
    print(f"[infer] sweeping {len(depth_starts)} depth windows (depth={depth}, z_step={args.z_step})")

    h_small, w_small = H // args.tile_size, W // args.tile_size
    composite = np.full((h_small, w_small), np.nan, dtype=np.float32)
    stack = np.full((len(depth_starts), h_small, w_small), np.nan, dtype=np.float32)

    for i, d in enumerate(depth_starts):
        coords = gen_coords(mask, y_range, x_range, args.tile_size, d)
        pmap = predict_tiles(c, model, vol, mask, coords, y_range, x_range, d,
                             f"{sid}_d{d}", g_mean, g_std, g_min, g_max)
        stack[i] = pmap
        composite = np.fmax(composite, pmap)
        # per-depth heatmap
        fig, ax = plt.subplots(figsize=(10, 10 * h_small / max(w_small, 1)))
        ax.imshow(pmap, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{sid}  depth {d}-{d+depth}  (max p={np.nanmax(pmap):.3f})", fontsize=9)
        ax.axis("off")
        fig.savefig(os.path.join(out_dir, f"depth_{d:03d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[infer] depth {d:3d}/{D}  tiles={len(coords)}  max_p={np.nanmax(pmap):.3f}")

    # composite + raw surface reference
    np.save(os.path.join(out_dir, "pred_stack.npy"), stack)
    fig, ax = plt.subplots(figsize=(12, 12 * h_small / max(w_small, 1)))
    ax.imshow(composite, cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f"{sid}  MAX ink-prob over all {len(depth_starts)} depths", fontsize=11)
    ax.axis("off")
    fig.savefig(os.path.join(out_dir, "composite_max.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    mid = np.asarray(vol[D // 2]).astype(np.float32)
    mid = (mid - mid[mask > 0].min()) / (mid[mask > 0].ptp() + 1e-6)
    Image.fromarray((np.clip(mid, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, "surface_ref.png"))

    print(f"[infer] DONE. wrote {len(depth_starts)} depth figs + composite_max.png to {out_dir}")
    print(f"[infer] global max ink-prob anywhere = {np.nanmax(composite):.3f}")


if __name__ == "__main__":
    main()
