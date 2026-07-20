"""_native_bbox.py — compute the NATIVE 2.4 rectangle our 7.9 split maps to, and verify the
native ink tif aligns with the 2.4 volume. No warp is applied to any data — we only use the
warp to LOCATE which native 2.4 rectangle corresponds to the 7.9 training crop.
"""
import json, os, numpy as np, tifffile
from build_teacher_zarr import fit_warp, CHUNK

ZARR_PATH = r"C:\Users\ChenJeff\Documents\ves_zarrs2"
REF_ID = "20240304161941"
SRC_H, SRC_W = 41860, 102360
MARGIN = 64

# native ink tif shape check
tif = r"C:\Users\ChenJeff\Documents\_ves_tmp\w023_ink_full.tif"
ink_shape = tifffile.TiffFile(tif).series[0].shape
print(f"native ink tif shape: {ink_shape}  (volume H,W = {SRC_H},{SRC_W})")

fx, fy = fit_warp("warp_MARK_2p4_source_dots.png", "warp_MARK_7p9_target_dots.png")
za = json.load(open(os.path.join(ZARR_PATH, f"{REF_ID}.zarr", ".zarray")))
H79, W79 = int(za["shape"][1]), int(za["shape"][2])

# sample the 7.9 crop corners+grid, map to native 2.4, take bbox
x0f, x1f, y0f, y1f = 0.60, 1.00, 0.00, 0.75
gu = np.linspace(x0f, x1f, 200)
gv = np.linspace(y0f, y1f, 200)
uu, vv = np.meshgrid(gu, gv)
U = np.column_stack([uu.ravel(), vv.ravel()])
su = fx(U) * SRC_W
sv = fy(U) * SRC_H
sx0 = max(0, int(np.floor(su.min())) - MARGIN); sx1 = min(SRC_W, int(np.ceil(su.max())) + MARGIN)
sy0 = max(0, int(np.floor(sv.min())) - MARGIN); sy1 = min(SRC_H, int(np.ceil(sv.max())) + MARGIN)
# snap to chunk boundaries so the native rectangle == whole cached chunks
sx0 = (sx0 // CHUNK) * CHUNK; sx1 = ((sx1 + CHUNK - 1) // CHUNK) * CHUNK
sy0 = (sy0 // CHUNK) * CHUNK; sy1 = ((sy1 + CHUNK - 1) // CHUNK) * CHUNK
nx = (sx1 - sx0) // CHUNK; ny = (sy1 - sy0) // CHUNK
dl_gb = nx * ny * 109 * CHUNK * CHUNK / 1e9
zarr_gb = 109 * (sy1 - sy0) * (sx1 - sx0) * 2 / 1e9
print(f"NATIVE 2.4 bbox  y[{sy0}:{sy1}] x[{sx0}:{sx1}]  ({sx1-sx0} x {sy1-sy0})")
print(f"chunks {nx} x {ny} = {nx*ny}  |  download ~{dl_gb:.0f} GB  |  zarr(109-deep) ~{zarr_gb:.0f} GB")
