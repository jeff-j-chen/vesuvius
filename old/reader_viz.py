"""Figure for a reader-v2 prediction TIFF on the straightened2 holdout: prediction | composite of the 17-layer window."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.fast_figure import colorize, gray_rgb, orient_display, red_overlay, save_panel_grid

tif, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
start, depth = (int(sys.argv[4]), 17) if len(sys.argv) > 4 else (5, 17)
NAME = sys.argv[5] if len(sys.argv) > 5 else "auto_grown_20261012065803387_straightened2_abf"
ZID = sys.argv[6] if len(sys.argv) > 6 else "20261012065814"

prediction = tifffile.imread(tif).astype(np.float32) / 255.0
mask = cv2.imread(f"/vesuvius/masks/{ZID}.png", cv2.IMREAD_GRAYSCALE) > 0
volume = zarr.open(f"/vesuvius/ves_zarrs2/{ZID}.zarr", mode="r")
h, w = min(prediction.shape[0], mask.shape[0]), min(prediction.shape[1], mask.shape[1])
prediction, mask = prediction[:h, :w], mask[:h, :w]
composite = np.clip(np.asarray(volume[start:start + depth, :h, :w]).max(axis=0), 0, 255).astype(np.uint8) * mask
shown = orient_display(np.where(mask, prediction, np.nan), NAME)
composite = orient_display(composite, NAME)
lo, hi = (np.percentile(prediction[mask], [1, 99.5]) if os.environ.get("RR_STRETCH") == "1" else (0.0, 1.0))
panels = [[(title, colorize(shown, "gray_r", (0.78, 0.86, 0.95), vmin=float(lo), vmax=float(hi)), cv2.INTER_AREA, None),
           (f"composite of layers {start}-{start + depth - 1}", gray_rgb(composite), cv2.INTER_AREA, 9.362)]]
if len(sys.argv) > 7:  # trained patch: second row with its inklabel (red) over the prediction and the composite
    label = cv2.imread(f"/vesuvius/dilated_inklabels/{sys.argv[7]}.png", cv2.IMREAD_GRAYSCALE)[:h, :w] > 0
    label = orient_display(label, NAME)
    pred_rgb = cv2.resize(panels[0][0][1], (composite.shape[1], composite.shape[0]), interpolation=cv2.INTER_NEAREST)
    panels.append([(f"{NAME} | inklabel overlay (red)", red_overlay(pred_rgb, label), cv2.INTER_AREA, None),
                   ("composite + inklabel (red)", red_overlay(gray_rgb(composite), label), cv2.INTER_AREA, 9.362)])
save_panel_grid(panels, composite.shape, out, gap_px=24)
print("saved", out)
