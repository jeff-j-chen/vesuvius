"""offset a tifxyz mesh along the renderer's surface normal (assemble_test_segments.compute_normals convention).

    python3 _ves_tmp/offset_mesh.py <src mesh dir> <dst mesh dir> <offset voxels>
"""
import json
import os
import sys

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "/vesuvius")
from assemble_test_segments import compute_normals, load_mesh

src, dst, offset = sys.argv[1], sys.argv[2], float(sys.argv[3])
X, Y, Z, valid = load_mesh(src)
# fill invalid cells from the nearest valid one so edge normals stay finite and the grid frame is unchanged
_, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
normals = compute_normals(X[iy, ix], Y[iy, ix], Z[iy, ix])
# degenerate cells (zero gradient) get a zero normal: borrow the nearest proper one
zero = np.linalg.norm(normals, axis=-1) < 0.5
if zero.any():
    _, (zy, zx) = distance_transform_edt(zero, return_indices=True)
    normals = normals[zy, zx]
    print(f"[offset] {int((zero & valid).sum())} valid cells had a zero normal; filled from neighbours")
out = [np.where(valid, grid + normals[..., axis] * offset, -1).astype(np.float32)
       for axis, grid in enumerate((X, Y, Z))]
os.makedirs(dst, exist_ok=True)
for name, grid in zip(("x", "y", "z"), out):
    tifffile.imwrite(os.path.join(dst, f"{name}.tif"), grid)
meta = json.load(open(os.path.join(src, "meta.json")))
points = np.stack([grid[valid] for grid in out], axis=1)
meta["bbox"] = [points.min(0).tolist(), points.max(0).tolist()]
meta["uuid"] = os.path.basename(os.path.normpath(dst))
meta["normal_offset_from"] = {"mesh": os.path.basename(os.path.normpath(src)), "offset_voxels": offset}
json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=4)
shift = np.sqrt(sum((o[valid] - g[valid]) ** 2 for o, g in zip(out, (X, Y, Z))))
print(f"[offset] {src} -> {dst}: {int(valid.sum())} points moved {shift.min():.2f}-{shift.max():.2f} voxels "
      f"bbox {np.round(meta['bbox'], 1).tolist()}")
