"""evaluate every training scroll of a run: flat (surface = depth 13) and surface-detection inference, with PR-AUC.

each scroll is read exactly as the run trained on it (config.json beside the checkpoint: zarr suffix, window offset,
normalisation, regular ink of inklabel_dir). per scroll two variants are rendered and scored:
  flat     a flat surface at FLAT_DEPTH (the notebook's "flat depth-13" window, per-scroll window offset)
  surface  the stored surface_labels/<id> depth map
the PHerc1447 scrolls (depth_reversed.json: stored depth-flipped to our orientation) get two more:
  flat_reversed_input / surface_reversed_input   the zarr is fed upside down (layer k <-> D-1-k, surface depth d ->
  D-1-d), i.e. the flip undone in the input itself instead of in the surface maps
PR-AUC is computed on the 8 px prediction cells against the ink label (whole scroll, and train/val cells when the run
has a manual train mask). figures: prediction | prediction + red inklabel. outputs go to
output/eval_train_scrolls/<run>/<variant>/ with results.json and results.csv.

  python3 eval_train_scrolls.py [--checkpoint PATH] [--only 20260226000000,...] [--variants flat,surface]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _path in (REPO, REPO / "old"):
    sys.path.insert(0, str(_path))
os.chdir(REPO)

import cv2
import numpy as np
import torch
import zarr
from sklearn.metrics import average_precision_score

from utils.config import Config, ScrollConfig, resolve_run_config
from utils.dataloader import DataManager, split_extra_labels
from utils.fast_figure import colorize, orient_display, red_overlay, save_panel_grid
from utils.model import create_model
from utils.norm import UNIFIED_CACHE_PATH, compute_norm, ensure_surface_anchors, load_cached_norm
from utils.platform import get_zarr_dir
from utils.visualizer import TensorboardVisualizer as TBV
from utils.visualizer import _configured_surface_window_offset, group_by_depth, imread_gray, predict_tiles

FLAT_DEPTH = 13
EVAL_STRIDE = 32
DEFAULT_CHECKPOINT = "/vesuvius/models/runs_iter2/iter2_old_plus_new_2/model_epoch_10.pth"
VARIANTS = ("flat", "surface", "flat_reversed_input", "surface_reversed_input")
AUXILIARY_PREFIXES = ("supcon_head.", "domain_head.")


def build_config(run_config):
    c = Config()
    for section in ("data", "model"):
        for name, value in run_config[section].items():
            if section == "data" and name == "scrolls":
                c.data.scrolls = [ScrollConfig(**item) for item in value]
            elif name != "probe_rois" and hasattr(getattr(c, section), name):
                setattr(getattr(c, section), name, value)
    c.data.eval_stride = EVAL_STRIDE
    c.data.zarr_path = get_zarr_dir()
    c.model.compile_model = False
    c.tra.supcon = False
    c.tra.dann = False
    c.device = "cuda" if torch.cuda.is_available() else "cpu"
    return c


def load_model(c, checkpoint):
    model, _ = create_model(c)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith(AUXILIARY_PREFIXES)}
    model.load_state_dict(state, strict=True)
    return model.eval()


def get_norm(c, sid):
    mode = str(getattr(c.data, "norm_mode", "global"))
    if mode == "surface_anchor":
        ensure_surface_anchors([sid], c.data.zarr_path, surface_dir=c.data.surface_label_dir)
    stats = load_cached_norm(sid, UNIFIED_CACHE_PATH, mode=mode)
    if stats is None:
        compute_norm(sid, c.data.zarr_path, UNIFIED_CACHE_PATH, mask_dir=getattr(c.data, "mask_dir", "./masks"))
        stats = load_cached_norm(sid, UNIFIED_CACHE_PATH, mode=mode)
    if stats is None:
        raise RuntimeError(f"normalization missing for {sid}")
    return stats


def first_existing(*paths):
    return next((p for p in paths if os.path.exists(p)), None)


def manual_train_grid(c, sid, grid_shape):
    """train cells of the run's manual split (train_masks/<id>.png); None when the run splits otherwise."""
    if bool(getattr(c.data, "simple_split", True)):
        return None
    train_mask = imread_gray(os.path.join(c.data.train_mask_dir, f"{sid}.png"))
    if train_mask is None:
        return None
    unit = int(c.model.multitile_subtile)
    aligned = DataManager._align_manual_mask(train_mask, unit)
    height, width = grid_shape
    region = aligned[:height * unit, :width * unit]
    region = np.pad(region, ((0, height * unit - region.shape[0]), (0, width * unit - region.shape[1])))
    return region.reshape(height, unit, width, unit).all(axis=(1, 3))


def pr_auc(prediction, label, cells):
    keep = cells & np.isfinite(prediction)
    target = label[keep] > 0.5
    if target.sum() == 0 or target.all():
        return None
    return float(average_precision_score(target, prediction[keep]))


def predict(c, model, sid, volume, mask, depth_map, confidence_map, norm):
    fake = types.SimpleNamespace(c=c)
    height, width = mask.shape
    coords = TBV._gen_tile_coords(fake, (c.data.d_start, c.data.d_end), (0, height), (0, width), mask,
                                  z_step=c.data.depth)
    grouped = group_by_depth(coords)
    if len(grouped) != 1:
        raise RuntimeError(f"{sid}: surface-relative inference expected one depth pass, got {len(grouped)}")
    offset = next(iter(grouped))
    return predict_tiles(
        c, model, volume, mask, grouped[offset], (0, height), (0, width), c.data.d_start + offset, f"eval_{sid}",
        *norm, surface_depth_map=depth_map, surface_confidence_map=confidence_map,
        surface_depth_offset=_configured_surface_window_offset(c, sid), cue_map=None,
    )


def save_figure(path, title, prediction, label, sid):
    shown = orient_display(prediction, sid)
    ink = orient_display(label, sid)
    height, width = ink.shape
    cell = (max(1, height // 2), max(1, width // 2))
    rgb = cv2.resize(colorize(shown, "gray_r", (0.78, 0.86, 0.95)), (cell[1], cell[0]), interpolation=cv2.INTER_NEAREST)
    overlay = red_overlay(rgb, ink)
    save_panel_grid([[(title, rgb, cv2.INTER_AREA, None), (f"{title} + inklabel (red)", overlay, cv2.INTER_AREA, None)]],
                    cell, str(path), gap_px=24)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--only", default="", help="comma-separated scroll ids")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    run_config, _ = resolve_run_config(checkpoint)
    c = build_config(run_config)
    run_name = f"{checkpoint.parent.name}_{checkpoint.stem}"
    out_root = REPO / (args.out or f"output/eval_train_scrolls/{run_name}")
    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    domains = {int(sid): name for name, ids in (c.data.train_scroll_dict or {}).items() for sid in ids}
    suffixes = {str(k): v for k, v in (c.data.zarr_suffix or {}).items()}
    reversed_scrolls = {int(sid) for sid in c.data.train_scroll_dict.get("pherc1447", [])}
    model = load_model(c, checkpoint)
    print(f"[eval] {checkpoint} | {len(c.data.scrolls)} scrolls | out {out_root}", flush=True)

    results_path = out_root / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else []
    done = {(r["scroll_id"], r["variant"]) for r in results}

    for scroll in c.data.scrolls:
        sid = int(scroll.scroll_id)
        if only and str(sid) not in only:
            continue
        variants = [v for v in VARIANTS if v in wanted and (sid in reversed_scrolls or not v.endswith("reversed_input"))
                    and (sid, v) not in done]
        if not variants:
            continue
        name = domains.get(sid, "scroll")
        try:
            source = zarr.open(os.path.join(c.data.zarr_path, f"{sid}{suffixes.get(str(sid), '')}.zarr"), mode="r")
            mask_image = imread_gray(first_existing(f"masks/{sid}.png", f"masks/extras/{sid}.png"))
            label_image = imread_gray(os.path.join(c.data.inklabel_dir, f"{sid}.png"))
            surface_dir = first_existing(os.path.join(c.data.surface_label_dir, str(sid)),
                                         os.path.join(c.data.surface_label_dir, "extras", str(sid)))
            if mask_image is None or label_image is None or surface_dir is None:
                raise FileNotFoundError("mask, label or surface labels missing")
            height = min(source.shape[1], mask_image.shape[0], label_image.shape[0])
            width = min(source.shape[2], mask_image.shape[1], label_image.shape[1])
            mask = (mask_image[:height, :width] > 127).astype(np.uint8)
            regular, _ = split_extra_labels(label_image[:height, :width])
            labels = regular.astype(np.uint8)
            volume = np.asarray(source[:, :height, :width])
            depth_map = np.asarray(np.load(os.path.join(surface_dir, "depth.npy"), mmap_mode="r")[:height, :width])
            confidence = np.asarray(np.load(os.path.join(surface_dir, "confidence.npy"), mmap_mode="r")[:height, :width])
            norm = get_norm(c, sid)
            fake = types.SimpleNamespace(c=c)
            label_cells, _, _ = TBV._compute_tile_maps(fake, labels, mask, (0, height), (0, width))
        except Exception as error:
            print(f"[skip] {name} {sid}: {error}", flush=True)
            continue
        depth_count = volume.shape[0]
        flat_depth = np.where(mask > 0, FLAT_DEPTH, 255).astype(np.uint8)
        flat_confidence = np.where(mask > 0, 255, 0).astype(np.uint8)
        for variant in variants:
            flipped = variant.endswith("reversed_input")
            surface_variant = variant.startswith("surface")
            d_map, conf = (depth_map, confidence) if surface_variant else (flat_depth, flat_confidence)
            vol = volume
            if flipped:
                vol = np.ascontiguousarray(volume[::-1])
                d_map = np.where(d_map != 255, depth_count - 1 - d_map.astype(np.int64), 255).astype(np.uint8)
            prediction = predict(c, model, sid, vol, mask, d_map, conf, norm)
            grid = manual_train_grid(c, sid, prediction.shape)
            everything = np.ones(prediction.shape, dtype=bool)
            row = {
                "scroll_id": sid, "domain": name, "variant": variant,
                "pr_auc": pr_auc(prediction, label_cells, everything),
                "pr_auc_train_cells": pr_auc(prediction, label_cells, grid) if grid is not None else None,
                "pr_auc_val_cells": pr_auc(prediction, label_cells, ~grid) if grid is not None else None,
            }
            folder = out_root / variant
            folder.mkdir(parents=True, exist_ok=True)
            score = row["pr_auc"]
            title = f"{name} {sid} | {variant} | PR-AUC {'n/a' if score is None else f'{score:.3f}'}"
            save_figure(folder / f"{name}_{sid}.png", title, prediction, labels, sid)
            results.append(row)
            out_root.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(results, indent=2))
            print(f"[eval] {title} (val cells {row['pr_auc_val_cells']})", flush=True)
            del prediction, vol
            gc.collect()
            torch.cuda.empty_cache()
        del volume, labels, mask
        gc.collect()

    with open(out_root / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scroll_id", "domain", "variant", "pr_auc",
                                                    "pr_auc_train_cells", "pr_auc_val_cells"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[eval] {len(results)} results -> {out_root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
