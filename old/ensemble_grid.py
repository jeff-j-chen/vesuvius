"""3 models x 3 depths on straightened2 (the held-out sheet), cached, combined nine ways into one 3x3 figure.

python3 ensemble_grid.py [--patch straightened2_abf] [--models a,b,c] [--offsets -1,0,1]

per-(model, depth) probability maps are cached in output/ensemble_cache/<run>/<patch>_offset<+d>.npy, so a rerun only
recombines. depths are absolute surface offsets (the iter2 runs trained at -1, i.e. slices 9-16).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

PATCHES = {
    "straightened2_abf": ("auto_grown_20261012065803387_straightened2_abf", "20261012065814", 20261012065815),
}
MODELS = ("iter2_asf_cue", "iter2_idea6_layout_prior", "iter2_ring0_posfalse")
CACHE = REPO / "output" / "ensemble_cache"
FLAT_DEPTH = 13
EVAL_STRIDE = 32
TOPK = 3
CORROB_K, CORROB_GAMMA, CORROB_SIGMA, CORROB_RAMP = 3, 0.5, 1.0, (0.15, 0.45)


def cache_path(run: str, patch: str, offset: int) -> Path:
    return CACHE / run / f"{patch}_offset{offset:+d}.npy"


def predict_run(run: str, patch: str, offsets: list[int]) -> None:
    """fill the cache for one model; skips depths that are already there."""
    missing = [o for o in offsets if not cache_path(run, patch, o).is_file()]
    if not missing:
        print(f"[ensemble] {run}: cached", flush=True)
        return
    import torch
    import zarr
    from PIL import Image

    from utils.config import Config, ScrollConfig, resolve_run_config
    from utils.model import create_model
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm
    from utils.platform import get_zarr_dir
    from utils.visualizer import TensorboardVisualizer, predict_tiles

    Image.MAX_IMAGE_PIXELS = None
    name, zid, norm_id = PATCHES[patch]
    checkpoint = REPO / "models" / "runs_iter2" / run / "final.pth"
    run_config, _ = resolve_run_config(checkpoint)
    C = Config()
    for section in ("data", "model"):
        for key, value in run_config[section].items():
            if section == "data" and key == "scrolls":
                C.data.scrolls = [ScrollConfig(**item) for item in value]
            elif key != "probe_rois" and hasattr(getattr(C, section), key):
                setattr(getattr(C, section), key, value)
    C.model.compile_model = False
    C.tra.supcon = False
    C.tra.dann = False
    C.device = "cuda" if torch.cuda.is_available() else "cpu"
    C.data.eval_stride = EVAL_STRIDE
    C.data.surface_window_offset = -1
    C.data.surface_window_offset_by_scroll = {}
    zarr_dir = Path(get_zarr_dir())
    C.data.zarr_path = str(zarr_dir)
    depth = int(C.data.depth)

    volume = np.ascontiguousarray(zarr.open(str(zarr_dir / f"{zid}.zarr"), mode="r")[:])
    volume.setflags(write=False)
    with Image.open(REPO / "masks" / f"{zid}.png") as image:
        mask = np.asarray(image.convert("L")) > 0
    height, width = mask.shape
    surface_depth = np.where(mask, FLAT_DEPTH, 255).astype(np.uint8)
    surface_confidence = np.where(mask, 255, 0).astype(np.uint8)
    coords = TensorboardVisualizer._gen_tile_coords(
        types.SimpleNamespace(c=C), (C.data.d_start, C.data.d_end), (0, height), (0, width), mask, z_step=depth)
    norm = load_cached_norm(norm_id, UNIFIED_CACHE_PATH, mode=str(C.data.norm_mode))
    cue = None
    if str(C.model.input_cue) == "pre_clahe":
        cue = np.load(Path(C.data.cue_dir) / f"{norm_id}.npy", mmap_mode="r")

    model, _ = create_model(C)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state.get("state_dict", state))
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()
             if not k.startswith(("supcon_head.", "domain_head."))}
    model.load_state_dict(state, strict=True)
    model.eval()
    for offset in missing:
        prediction = predict_tiles(
            C, model, volume, mask, coords, (0, height), (0, width), C.data.d_start, name, *norm,
            surface_depth_map=surface_depth, surface_confidence_map=surface_confidence,
            surface_depth_offset=offset, cue_map=cue,
        )
        out = cache_path(run, patch, offset)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.asarray(prediction, dtype=np.float32))
        print(f"[ensemble] {run} offset {offset:+d} -> {out}", flush=True)
    del model, volume
    gc.collect()
    torch.cuda.empty_cache()


# ---- combination helpers; every map is (H, W) with NaN outside the papyrus -------------------------------------------------

def _valid(stack: np.ndarray) -> np.ndarray:
    return np.isfinite(stack).all(axis=0)


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """robust z of a probability map's logits over the valid cells."""
    p = np.clip(np.nan_to_num(x, nan=0.5), 1e-4, 1 - 1e-4)
    logit = np.log(p) - np.log1p(-p)
    values = logit[valid]
    median = np.median(values)
    return (logit - median) / (1.4826 * np.median(np.abs(values - median)) + 1e-6)


def zscore_values(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """robust z of an already-real-valued map."""
    values = x[valid]
    median = np.median(values)
    return (x - median) / (1.4826 * np.median(np.abs(values - median)) + 1e-6)


def mid_boost(stack: np.ndarray, valid: np.ndarray, k: int = CORROB_K) -> np.ndarray:
    """the notebook's corroborated mid-boost over the first axis of a probability stack."""
    p = np.clip(np.nan_to_num(stack, nan=0.5), 1e-4, 1 - 1e-4)
    ranked = np.sort(p, axis=0)[::-1]
    k = min(k, len(ranked))
    weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), CORROB_SIGMA)
    support = cv2.GaussianBlur((ranked[k - 1] * valid).astype(np.float32), (0, 0), CORROB_SIGMA) / np.maximum(weight, 1e-6)
    agree = np.clip((support - CORROB_RAMP[0]) / (CORROB_RAMP[1] - CORROB_RAMP[0]), 0, 1)
    agree = agree * agree * (3 - 2 * agree)
    return ranked[:k].mean(axis=0) ** (1 - (1 - CORROB_GAMMA) * agree)


def combinations(maps: dict[str, list[np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """nine ways to merge maps[model][depth]; returns (title, map)."""
    models = list(maps)
    flat = np.stack([m for run in models for m in maps[run]])
    valid = _valid(flat)
    per_model = {run: np.stack(maps[run]) for run in models}
    flat_z = np.stack([zscore(m, valid) for m in flat])
    out: list[tuple[str, np.ndarray]] = []
    out.append(("1 plain mean of all 9", flat.mean(0)))
    out.append(("2 max of all 9", flat.max(0)))
    out.append((f"3 top-{TOPK} mean of all 9", np.sort(flat, axis=0)[::-1][:TOPK].mean(0)))
    out.append(("4 z-score each of 9, mean", flat_z.mean(0)))
    depth_z = [np.stack([zscore(m, valid) for m in per_model[run]]).mean(0) for run in models]
    out.append(("5 z depths -> mean -> z models -> mean", np.mean([zscore_values(d, valid) for d in depth_z], axis=0)))
    boosted = [mid_boost(per_model[run], valid) for run in models]
    out.append(("6 mid-boost depths -> z models -> mean", np.mean([zscore(b, valid) for b in boosted], axis=0)))
    out.append(("7 mid-boost depths -> mean models", np.mean(boosted, axis=0)))
    out.append(("8 mid-boost jointly over all 9 (k=4)", mid_boost(flat, valid, k=4)))
    out.append(("9 z depths -> mean -> z models -> median", np.median([zscore_values(d, valid) for d in depth_z], axis=0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", default="straightened2_abf", choices=list(PATCHES))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--offsets", default="-1,0,1")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    offsets = [int(o) for o in args.offsets.split(",")]
    if len(models) * len(offsets) != 9:
        raise SystemExit("the 3x3 grid needs exactly 3 models x 3 depths")
    for run in models:
        predict_run(run, args.patch, offsets)

    from utils.fast_figure import colorize, orient_display, save_panel_grid

    name, zid, _ = PATCHES[args.patch]
    mask = cv2.imread(str(REPO / "masks" / f"{zid}.png"), cv2.IMREAD_GRAYSCALE) > 0
    maps = {run: [np.load(cache_path(run, args.patch, o)) for o in offsets] for run in models}
    shape = maps[models[0]][0].shape
    for run in models:
        for m in maps[run]:
            if m.shape != shape:
                raise SystemExit(f"{run}: prediction maps differ in shape")
    cell_hw = orient_display(mask, name).shape
    merged = combinations(maps)
    panels, row = [], []
    for title, array in merged:
        valid = np.isfinite(array)
        vmax = float(np.percentile(array[valid], 99.5))
        shown = colorize(orient_display(np.where(valid, array, np.nan), name), "gray_r", (0.78, 0.86, 0.95),
                         vmin=0.0, vmax=max(vmax, 1e-6))
        row.append((f"{title} | 0..p99.5 = {vmax:.2f}", shown, cv2.INTER_AREA, None))
        if len(row) == 3:
            panels.append(row)
            row = []
    out_dir = REPO / "output" / "ensemble_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.patch}_3models_3depths.png"
    save_panel_grid(panels, cell_hw, str(path), gap_px=24)
    print(f"[ensemble] saved {path}", flush=True)
    image = cv2.imread(str(path))
    scale = 3000 / image.shape[1]
    cv2.imwrite(str(out_dir / f"{args.patch}_3models_3depths_preview.png"),
                cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
