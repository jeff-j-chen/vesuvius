"""rebuild the 0846A training-fragment figures as 2x2 (pred | composite / pred + red inklabel | asf_morph cue)
from the already-rendered one-row figures, without re-predicting.

usage: python old/retrofit_train_figs.py [--source DIR ...] [--out DIR]
a figure missing from the first --source is taken from the next one.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.depth_ensemble import describe, resolve
from utils.fast_figure import crop_panel_pair, gray_rgb, orient_display, red_overlay, save_panel_grid

Image.MAX_IMAGE_PIXELS = None
# name -> (render zarr/mask id, alias id for label and cue, window offset)
TRAIN = {
    "auto_grown_20260927134928530_abf": (20260927134928, 20260927134931, -1),
    "auto_grown_20261012073244544_abf": (20261012073244, 20261012073246, -1),
    "auto_grown_20261012065803387_straight_abf_m28": (20261012065812, 20261012065813, -1),
    "auto_grown_20261012065803387_straightened2_abf_in_1": (20261012065816, 20261012065817, -1),
    "auto_grown_20261012065803387_straightened2_abf_out_1": (20261012065818, 20261012065819, -1),
    "auto_grown_20261012065803387_straightened2_abf_out_2": (20261012065820, 20261012065821, -1),
    "auto_grown_20261012065803387_straightened2_abf_out_3": (20261012065822, 20261012065823, -1),
    "PHerc0009B_20250919125754": (20250919125754, 20250919125754, 0),
}

parser = argparse.ArgumentParser()
parser.add_argument("--source", nargs="+", default=[
    "output/0846A/runs_iter2_asf_cue_final/offset_-1", "output/0846A/runs_iter2_iter2_asf_cue_final/offset_-1"])
parser.add_argument("--out", default="output/0846A/runs_iter2_asf_cue_final/offset_-1")
parser.add_argument("--ensemble", nargs=2, type=int, default=[-1, 1])
parser.add_argument("--method", default="z-score")
args = parser.parse_args()
out_dir = REPO / args.out
offsets, method = resolve(args.ensemble, args.method)

for name, (zid, alias, offset) in TRAIN.items():
    source = next((REPO / d / f"{name}.png" for d in args.source if (REPO / d / f"{name}.png").is_file()), None)
    if source is None:
        print(f"SKIP {name}: no existing figure")
        continue
    with Image.open(REPO / "masks" / f"{zid}.png") as image:
        mask = np.asarray(image.convert("L")) > 0
    cell_hw = orient_display(mask, name).shape
    pred, composite = crop_panel_pair(source, cell_hw)
    with Image.open(REPO / "dilated_inklabels" / f"{alias}.png") as image:
        label = np.asarray(image.convert("L")) > 0
    cue = np.load(REPO / "cue_asf_morph" / f"{alias}.npy")
    if label.shape != mask.shape or cue.shape != mask.shape:
        raise ValueError(f"{name}: mask {mask.shape}, label {label.shape}, cue {cue.shape} differ")
    overlay = red_overlay(pred, orient_display(label, name))
    panel_hw = pred.shape[:2]
    title = f"{name} | {describe([offset + o for o in offsets], method)}"
    depth_path = REPO / "surface_labels" / str(alias) / "depth.npy"
    centre = 13
    if name.startswith("PHerc0009B"):
        depth = np.load(depth_path)
        centre = int(np.median(depth[(depth != 255) & mask]))
    start = centre - 3 + offset
    panels = [
        [(title, pred, cv2.INTER_NEAREST, None), (f"composite {start}-{start + 7}", composite, cv2.INTER_AREA, None)],
        [(f"{name} | inklabel overlay (red)", overlay, cv2.INTER_NEAREST, None),
         ("asf_morph cue", gray_rgb(orient_display(cue, name)), cv2.INTER_AREA, None)],
    ]
    save_panel_grid(panels, panel_hw, str(out_dir / f"{name}.png"), gap_px=24)
    print(f"Saved {out_dir / f'{name}.png'} (from {source.parent.name})")
