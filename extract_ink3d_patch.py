"""extract_ink3d_patch.py -- sample a whole-scroll 3D ink prediction onto one surface patch.

Only the ink-prediction chunks intersected by the patch's sampling points are downloaded.
Each output voxel (k, y, x) of the local 28-layer zarr is resampled from the 3D ink volume at
the same physical position used by the surface renderer:

    p = tifxyz(y, x) - normal(y, x) * (bin_center_k - 54)

The tifxyz grid convention (half-pixel offset, normal sign, center slice) was verified by
re-rendering a surface-volume chunk from the raw CT (corr 0.9997). Ink level 2 (9.6um) is used
so trilinear samples match the level-2 XY grid and the ~3.9-slice depth pooling.

usage:
  python extract_ink3d_patch.py --dry-run     # count required chunks only
  python extract_ink3d_patch.py               # download chunks + write ink3d_labels/<id>.zarr
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import map_coordinates

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
DEFAULTS = {
    "scroll_id": "20231210121321",
    "tifxyz_url": f"{BUCKET}/PHercParis4/segments/20231210121321/mesh/"
                  "20231210121321-on-20260411134726-2.4um.tifxyz",
    "ink_url": f"{BUCKET}/PHercParis4/representations/predictions/ink-3d/"
               "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr",
}
SOURCE_SLICES = 109
SURFACE_LEVEL = 2
TARGET_DEPTH = 28


def _bin_offsets(source_slices=SOURCE_SLICES, layers_out=TARGET_DEPTH):
    """normal offsets (level-0 voxels) at the center of each assembled depth-pooling bin."""
    center = (source_slices - 1) / 2.0
    offsets = []
    for index in range(layers_out):
        # same bin edges as assemble_training_segments._pool_w013_depth
        start = int(round(index * source_slices / layers_out))
        end = int(round((index + 1) * source_slices / layers_out))
        offsets.append((start + end - 1) / 2.0 - center)
    return np.asarray(offsets, dtype=np.float32)


def _download(url, path, tries=4):
    """fetch url -> path. returns bytes written, 0 for 404 (fill value)."""
    if os.path.exists(path):
        return os.path.getsize(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            tmp = path + ".part"
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
            return len(data)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 0
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(1 + attempt)
    return 0


class PatchGeometry:
    """maps assembled-zarr pixels to level-0 scroll coordinates via the tifxyz grid."""

    def __init__(self, tifxyz_dir: Path):
        meta = json.loads((tifxyz_dir / "meta.json").read_text())
        self.scale = float(meta["scale"][0])
        self.grid = [tifxyz.astype(np.float64) for tifxyz in
                     (tifffile.imread(tifxyz_dir / f"{c}.tif") for c in "xyz")]
        self.valid = (self.grid[2] > 0).astype(np.float32)
        self.factor = 2 ** SURFACE_LEVEL

    def _sample(self, rows, cols):
        return np.stack([map_coordinates(g, [rows, cols], order=1, mode="nearest")
                         for g in self.grid], axis=-1)

    def tile(self, y0, y1, x0, x1):
        """return level-0 points (h,w,3 xyz), unit normals, and validity for a tile."""
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        # level-2 pixel center -> level-0 center -> tifxyz grid (+0.5 px, verified)
        rows = (self.factor * yy + self.factor / 2.0) * self.scale
        cols = (self.factor * xx + self.factor / 2.0) * self.scale
        points = self._sample(rows, cols)
        step = 0.5
        du = self._sample(rows, cols + step) - self._sample(rows, cols - step)
        dv = self._sample(rows + step, cols) - self._sample(rows - step, cols)
        normal = np.cross(du, dv)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        # every bilinear and normal-stencil neighbour must be a real mesh vertex
        valid = map_coordinates(self.valid, [rows, cols], order=1, mode="constant") > 0.999
        for dr, dc in ((0, step), (0, -step), (step, 0), (-step, 0)):
            valid &= map_coordinates(self.valid, [rows + dr, cols + dc], order=1,
                                     mode="constant") > 0.999
        return points.astype(np.float32), normal.astype(np.float32), valid


def _ink_coords(points, normal, offsets, ink_scale):
    """(28,n,3) zyx coordinates in the ink level, using mean-pool voxel centers."""
    level0 = points[None] - normal[None] * offsets[:, None, None]
    xyz = (level0 - (ink_scale - 1) / 2.0) / ink_scale
    return xyz[..., ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scroll-id", default=DEFAULTS["scroll_id"])
    ap.add_argument("--tifxyz-url", default=DEFAULTS["tifxyz_url"])
    ap.add_argument("--ink-url", default=DEFAULTS["ink_url"])
    ap.add_argument("--ink-level", type=int, default=2)
    ap.add_argument("--zarr-dir", default=os.getenv("VESUVIUS_ZARR_PATH", "./ves_zarrs2"))
    ap.add_argument("--output-dir", default="./ink3d_labels")
    ap.add_argument("--cache-dir", default=None,
                    help="sparse chunk cache (default /data/extra/tmp/ink3d_<id> or _ves_tmp/ink3d_<id>)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep downloaded ink chunks after writing the output")
    args = ap.parse_args()

    sid = str(args.scroll_id)
    scratch = "/data/extra/tmp" if os.path.isdir("/data/extra/tmp") else "_ves_tmp"
    cache = Path(args.cache_dir or os.path.join(scratch, f"ink3d_{sid}"))
    tif_dir = cache / "tifxyz"
    for name in ("meta.json", "x.tif", "y.tif", "z.tif"):
        _download(f"{args.tifxyz_url.rstrip('/')}/{name}", str(tif_dir / name))

    local = zarr.open(os.path.join(args.zarr_dir, f"{sid}.zarr"), mode="r")
    depth, height, width = map(int, local.shape)
    if depth != TARGET_DEPTH:
        raise ValueError(f"{sid}: expected {TARGET_DEPTH} layers, found {depth}")
    geometry = PatchGeometry(tif_dir)
    expected = (np.array(geometry.grid[0].shape) / geometry.scale / geometry.factor).round()
    if tuple(expected.astype(int)) != (height, width):
        raise ValueError(f"tifxyz implies level-{SURFACE_LEVEL} {tuple(expected)}, zarr is {(height, width)}")

    ink_base = args.ink_url.rstrip("/")
    level = int(args.ink_level)
    zarray = json.loads(urllib.request.urlopen(f"{ink_base}/{level}/.zarray", timeout=60).read())
    chunk = np.asarray(zarray["chunks"])
    ink_scale = 2.0 ** level
    offsets = _bin_offsets()
    tiles = [(y0, min(y0 + args.tile, height), x0, min(x0 + args.tile, width))
             for y0 in range(0, height, args.tile) for x0 in range(0, width, args.tile)]

    shape = np.asarray(zarray["shape"])
    grid = -(-shape // chunk)

    # pass 1: exact set of chunks touched by any trilinear sample
    def tile_chunks(spec):
        occupied = np.zeros(int(grid.prod()), dtype=bool)
        points, normal, valid = geometry.tile(*spec)
        if not valid.any():
            return occupied
        zyx = _ink_coords(points[valid], normal[valid], offsets, ink_scale).reshape(-1, 3)
        base = np.floor(zyx).astype(np.int64)
        lo = np.clip(base // chunk, 0, grid - 1)
        hi = np.clip((base + 1) // chunk, 0, grid - 1)
        for corner in np.ndindex(2, 2, 2):
            index = np.where(np.asarray(corner, dtype=bool), hi, lo)
            occupied[(index[:, 0] * grid[1] + index[:, 1]) * grid[2] + index[:, 2]] = True
        return occupied

    started = time.time()
    occupied = np.zeros(int(grid.prod()), dtype=bool)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, found in enumerate(pool.map(tile_chunks, tiles), 1):
            occupied |= found
            if done % 200 == 0 or done == len(tiles):
                print(f"[ink3d] scanned {done}/{len(tiles)} tiles, {int(occupied.sum())} chunks", flush=True)
    needed = [tuple(int(v) for v in np.unravel_index(flat, grid)) for flat in np.flatnonzero(occupied)]
    print(f"[ink3d] {len(needed)} level-{level} chunks of {chunk.tolist()} "
          f"({len(needed) * chunk.prod() / 1e9:.1f} GB uncompressed max) in {time.time() - started:.0f}s")
    if args.dry_run:
        return

    # pass 2: sparse local copy of only those chunks (same zarr layout)
    array_dir = cache / str(level)
    array_dir.mkdir(parents=True, exist_ok=True)
    (array_dir / ".zarray").write_text(json.dumps(zarray))
    sep = zarray.get("dimension_separator", ".")

    def fetch(index):
        key = sep.join(map(str, index))
        return _download(f"{ink_base}/{level}/{key}", str(array_dir / key))

    started = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, size in enumerate(pool.map(fetch, needed), 1):
            total_bytes += size
            if done % 200 == 0 or done == len(needed):
                print(f"[ink3d] fetched {done}/{len(needed)} chunks "
                      f"({total_bytes / 1e9:.2f} GB compressed)", flush=True)
    print(f"[ink3d] download {time.time() - started:.0f}s")

    # pass 3: trilinear resample onto the assembled 28-layer grid
    ink = zarr.open(str(array_dir), mode="r")
    out_path = Path(args.output_dir) / f"{sid}.zarr"
    partial = Path(str(out_path) + ".partial")
    output = zarr.open(
        str(partial), mode="w", shape=(depth, height, width),
        chunks=(depth, args.tile, args.tile), dtype="|u1",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
        zarr_format=2,
    )

    def render(spec):
        y0, y1, x0, x1 = spec
        points, normal, valid = geometry.tile(*spec)
        block = np.zeros((depth, y1 - y0, x1 - x0), dtype=np.uint8)
        if valid.any():
            zyx = _ink_coords(points[valid], normal[valid], offsets, ink_scale)
            lo = np.maximum(np.floor(zyx.reshape(-1, 3).min(0)).astype(int), 0)
            hi = np.minimum(np.floor(zyx.reshape(-1, 3).max(0)).astype(int) + 2, shape)
            cube = np.asarray(ink[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            rel = (zyx - lo).reshape(-1, 3).T
            values = map_coordinates(cube, rel, order=1, mode="constant", cval=0.0, output=np.float32)
            block[:, valid] = np.clip(np.rint(values), 0, 255).astype(np.uint8).reshape(depth, -1)
        output[:, y0:y1, x0:x1] = block
        return int(valid.sum())

    started = time.time()
    with ThreadPoolExecutor(max_workers=min(args.workers, 16)) as pool:
        for done, _ in enumerate(pool.map(render, tiles), 1):
            if done % 100 == 0 or done == len(tiles):
                print(f"[ink3d] rendered {done}/{len(tiles)} tiles", flush=True)
    output.attrs.update({
        "source": args.ink_url, "ink_level": level, "tifxyz": args.tifxyz_url,
        "surface_level": SURFACE_LEVEL, "bin_offsets_level0": offsets.tolist(),
        "note": "p = tifxyz - normal * offset; normal = cross(d/dcol, d/drow)",
    })
    del output
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    os.replace(partial, out_path)
    print(f"[ink3d] wrote {out_path} in {time.time() - started:.0f}s")
    if not args.keep_cache:
        import shutil
        shutil.rmtree(array_dir)
        print(f"[ink3d] removed chunk cache {array_dir}")


if __name__ == "__main__":
    main()
