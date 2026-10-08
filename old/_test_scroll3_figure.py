"""_test_scroll3_figure.py — garbage-data smoke test of the scroll3 test figure.

does NOT run inference or touch any model/zarr. it fabricates prediction maps shaped exactly
like scroll3's tile grid (H//32 x W//32 = 85 x 790, long-and-skinny) with realistic NaN
"outside-mask" borders, then drives the REAL TensorboardVisualizer._create_combined_test_figure
to confirm the separate 'Scroll3' figure lays out correctly (aspect, panels, no distortion)
before we ever enable the test interval.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.visualizer as V  # importing registers the 'inferno_nan' colormap
from utils.visualizer import TensorboardVisualizer

# scroll3 frame (13303 not applicable): H=2739, W=25309, tile=32
H, W, TILE = 2739, 25309, 32
h_small, w_small = H // TILE, W // TILE   # 85 x 790
print(f"scroll3 tile grid: {h_small} x {w_small}  aspect {w_small/h_small:.2f}")

rng = np.random.default_rng(0)

def fake_pred_map():
    """random [0,1] scores inside a wavy papyrus band, NaN outside (like a real segment)."""
    pmap = np.full((h_small, w_small), np.nan, dtype=np.float32)
    # a wavy horizontal band occupying ~60% of the height, varying across x
    cy = h_small / 2.0
    band = h_small * 0.30
    for x in range(w_small):
        off = np.sin(x / 40.0) * (h_small * 0.12)
        y0 = int(max(0, cy + off - band))
        y1 = int(min(h_small, cy + off + band))
        pmap[y0:y1, x] = rng.random(y1 - y0).astype(np.float32) ** 2  # skew toward 0
    return pmap

# a few depth blocks, like _add_single_test_figure would produce
all_data = [(fake_pred_map(), d, d + 8) for d in (0, 8, 16, 24)]

# call the REAL method on a bare instance (skip __init__ / no model/zarr needed)
vis = TensorboardVisualizer.__new__(TensorboardVisualizer)
fig = vis._create_combined_test_figure(all_data, len(all_data), "Scroll3")

out = r"C:\Users\ChenJeff\Documents\_ves_tmp\scroll3_figure_test.png"
fig.savefig(out, dpi=110)
plt.close(fig)
print(f"[ok] wrote {out}  ({len(all_data)} depth blocks)")
