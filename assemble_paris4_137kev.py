"""assemble_paris4_137kev.py -- render PHercParis4 segment 20231210121321 from the 137 keV scan.

The segment's tifxyz mesh lives in the 2.4 um 78 keV scan. Both 2.4 um scans ship an affine
transform (transform.json) into the same reference scan (PHercParis4-20230205180739), so a
mesh point maps into the 137 keV scan as inv(A137) @ A78 @ p. Each output pixel samples the
level-2 pyramid (9.6 um) of the target scan with trilinear interpolation at the level-2 pixel
centre of the surface volume, offset along the surface normal to the centre of each of the 28
depth bins the training assembly pools from the 109-layer surface volume. The output has the
exact frame of ves_zarrs2/20231210121321.zarr (28, 12750, 9995); only the part of the segment
inside the 137 keV region of interest carries data.

A residual transform error is removed afterwards: per tile, the depth and in-plane shift that
best aligns the 137 keV render with the existing 78 keV zarr is estimated, smoothed, and used
for a final render.

usage:
  python assemble_paris4_137kev.py --validate-78        # render the 78 keV scan on a crop and compare
  python assemble_paris4_137kev.py                      # full 137 keV render + registration + checks
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4"
SEGMENT = "20231210121321"
MESH_URL = f"{BUCKET}/segments/{SEGMENT}/mesh/{SEGMENT}-on-20260411134726-2.4um.tifxyz"
VOLUMES = {
    "78": f"{BUCKET}/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr",
    "137": f"{BUCKET}/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr",
}
OUT_ID = "20260323153942"  # the 137 keV scan id; the 78 keV zarr keeps the segment id
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2")
WORK = "_ves_tmp/p4_137"
OUT_SHAPE = (28, 12750, 9995)
SURFACE_LAYERS = 109
MESH_SCALE = 20          # level-0 surface pixels per tifxyz grid step
LEVEL = 2                # 4x in every axis -> 9.6 um
CHUNK = 128
TILE = 128               # output rows rendered per pass


def depth_offsets():
    """level-0 normal offsets of the 28 bin centres the training assembly pools 109 layers into."""
    centre = (SURFACE_LAYERS - 1) / 2.0
    return np.array([
        np.mean(np.arange(int(round(i * SURFACE_LAYERS / 28)), int(round((i + 1) * SURFACE_LAYERS / 28)))) - centre
        for i in range(28)
    ], dtype=np.float32)


def _affine(volume_key):
    matrix = np.array(json.load(open(os.path.join(WORK, f"t{volume_key}.json")))["transformation_matrix"])
    return np.vstack([matrix, [0, 0, 0, 1]])


def transform_to(volume_key):
    """map 78 keV level-0 (x, y, z) into the target scan's level-0 (x, y, z)."""
    if volume_key == "78":
        return np.eye(4)
    return np.linalg.inv(_affine(volume_key)) @ _affine("78")


def fetch_inputs():
    os.makedirs(os.path.join(WORK, "mesh"), exist_ok=True)
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        path = os.path.join(WORK, "mesh", name)
        if not os.path.exists(path):
            subprocess.run(["curl", "-s", "--fail", "-o", path, f"{MESH_URL}/{name}"], check=True)
    for key, base in VOLUMES.items():
        path = os.path.join(WORK, f"t{key}.json")
        if not os.path.exists(path):
            subprocess.run(["curl", "-s", "--fail", "-o", path, f"{base}/transform.json"], check=True)


class Mesh:
    def __init__(self):
        grids = [np.array(Image.open(os.path.join(WORK, "mesh", f"{a}.tif"))).astype(np.float64) for a in "xyz"]
        self.valid = np.all([g != -1 for g in grids], axis=0)
        self.xyz = np.stack(grids, axis=-1)
        dv = np.gradient(self.xyz, axis=0)
        du = np.gradient(self.xyz, axis=1)
        normal = np.cross(dv, du)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        self.normal = normal
        self.maps = [np.ascontiguousarray(self.xyz[..., i], dtype=np.float32) for i in range(3)] + \
                    [np.ascontiguousarray(normal[..., i], dtype=np.float32) for i in range(3)]
        self.valid_f = self.valid.astype(np.float32)

    def sample(self, rows, cols):
        """surface point, unit normal and validity at level-2 output pixel centres."""
        gy = ((4 * rows.astype(np.float32) + 1.5) / MESH_SCALE)[:, None] * np.ones((1, len(cols)), np.float32)
        gx = ((4 * cols.astype(np.float32) + 1.5) / MESH_SCALE)[None, :] * np.ones((len(rows), 1), np.float32)
        values = [cv2.remap(m, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1) for m in self.maps]
        valid = cv2.remap(self.valid_f, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
        point = np.stack(values[:3], axis=-1).astype(np.float64)
        normal = np.stack(values[3:], axis=-1).astype(np.float64)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        return point, normal, valid


class ChunkStore:
    """level-2 chunks of one remote scan, cached on disk; absent chunks are air."""

    def __init__(self, key, workers=32):
        self.key = key
        self.base = f"{VOLUMES[key]}/{LEVEL}"
        meta = json.loads(subprocess.run(["curl", "-s", "--fail", f"{self.base}/.zarray"],
                                         capture_output=True, check=True).stdout)
        self.shape = tuple(meta["shape"])
        self.cache = os.path.join(WORK, f"cache{key}")
        os.makedirs(self.cache, exist_ok=True)
        self.workers = workers
        self.loaded = {}

    def _path(self, chunk):
        return os.path.join(self.cache, "{}_{}_{}.raw".format(*chunk))

    def _fetch(self, chunk):
        path = self._path(chunk)
        if os.path.exists(path):
            return
        url = "{}/{}/{}/{}".format(self.base, *chunk)
        for _ in range(4):
            result = subprocess.run(["curl", "-s", "--connect-timeout", "20", "--max-time", "120",
                                     "-o", path, "-w", "%{http_code}", url], capture_output=True)
            code = result.stdout.decode()[-3:]
            if code == "200" and os.path.getsize(path) == CHUNK ** 3:
                return
            if os.path.exists(path):
                os.remove(path)
            if code in ("404", "403"):
                open(path, "wb").close()
                return
        raise RuntimeError(f"chunk {chunk} failed to download from {url}")

    def ensure(self, chunks):
        todo = [c for c in chunks if not os.path.exists(self._path(c))]
        if todo:
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(self._fetch, todo))

    def chunk(self, key):
        if key not in self.loaded:
            path = self._path(key)
            self.loaded[key] = (np.fromfile(path, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
                                if os.path.getsize(path) else None)
            # the training run shares this machine's RAM; keep at most ~800 MB of chunks
            if len(self.loaded) > 400:
                self.loaded.pop(next(iter(self.loaded)))
        return self.loaded[key]

    @staticmethod
    def chunks_for(zyx):
        base = np.floor(zyx).astype(np.int64)
        keys = np.concatenate([(base + np.array(o)) // CHUNK
                               for o in np.ndindex(2, 2, 2)], axis=0)
        return {tuple(k) for k in np.unique(keys, axis=0)}

    def trilinear(self, zyx):
        base = np.floor(zyx).astype(np.int64)
        frac = (zyx - base).astype(np.float32)
        shape = np.array(self.shape)
        out = np.zeros(len(zyx), np.float32)
        for offset in np.ndindex(2, 2, 2):
            corner = base + np.array(offset)
            weight = np.prod(np.where(np.array(offset)[None, :] == 1, frac, 1.0 - frac), axis=1)
            inside = np.all((corner >= 0) & (corner < shape), axis=1)
            idx = np.flatnonzero(inside)
            if not len(idx):
                continue
            c = corner[idx]
            keys = c // CHUNK
            code = (keys[:, 0] * 4096 + keys[:, 1]) * 4096 + keys[:, 2]
            order = np.argsort(code, kind="stable")
            code_sorted = code[order]
            starts = np.flatnonzero(np.r_[True, code_sorted[1:] != code_sorted[:-1]])
            bounds = np.r_[starts, len(order)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                sel = order[s:e]
                key = tuple(int(v) for v in keys[sel[0]])
                data = self.chunk(key)
                if data is None:
                    continue
                local = c[sel] % CHUNK
                out[idx[sel]] += weight[idx[sel]] * data[local[:, 0], local[:, 1], local[:, 2]]
        return out


def to_level2_zyx(points_xyz):
    """level-0 (x, y, z) -> level-2 continuous (z, y, x) voxel index."""
    return ((points_xyz[..., ::-1] - 1.5) / 4.0).reshape(-1, 3)


def render(store, transform, mesh, rows, cols, sign, offsets, shift=None):
    """render (len(offsets), len(rows), len(cols)); shift = optional per-pixel (dz_normal, dy, dx)
    correction in output pixels / level-0 normal voxels, applied in 78 keV space."""
    point, normal, valid = mesh.sample(rows, cols)
    if shift is not None:
        du = np.gradient(point, axis=1)
        dv = np.gradient(point, axis=0)
        point = point + shift[..., 1:2] * dv + shift[..., 2:3] * du
    out = np.zeros((len(offsets), len(rows), len(cols)), np.float32)
    homog = lambda p: (transform[:3, :3] @ p.reshape(-1, 3).T + transform[:3, 3:]).T.reshape(p.shape)
    for li, off in enumerate(offsets):
        total = off if shift is None else off + shift[..., 0:1]
        sample = homog(point + sign * total * normal)
        zyx = to_level2_zyx(sample)
        out[li] = store.trilinear(zyx).reshape(len(rows), len(cols))
    out[:, ~valid] = 0
    return out, valid


def needed_chunks(store, transform, mesh, rows, cols, sign, offsets, margin=4.0):
    point, normal, valid = mesh.sample(rows, cols)
    point, normal = point[valid], normal[valid]
    keys = set()
    for off in (offsets.min() - margin, 0.0, offsets.max() + margin):
        p = (transform[:3, :3] @ (point + sign * off * normal).T + transform[:3, 3:]).T
        zyx = to_level2_zyx(p)
        inside = np.all((zyx >= -1) & (zyx < np.array(store.shape) + 1), axis=1)
        keys |= ChunkStore.chunks_for(zyx[inside])
    shape_chunks = np.ceil(np.array(store.shape) / CHUNK).astype(int)
    return {k for k in keys if all(0 <= k[i] < shape_chunks[i] for i in range(3))}


def ncc(a, b, mask):
    a, b = a[mask], b[mask]
    if a.size < 100:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def highpass(img, sigma=4.0):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def validate_78(args):
    """render the 78 keV scan on a crop and compare with the existing training zarr."""
    import zarr

    mesh = Mesh()
    store = ChunkStore("78", args.workers)
    old = zarr.open(os.path.join(ZARR_DIR, f"{SEGMENT}.zarr"), mode="r")
    r0, c0, size = args.crop
    rows, cols = np.arange(r0, r0 + size), np.arange(c0, c0 + size)
    reference = np.asarray(old[:, r0:r0 + size, c0:c0 + size], np.float32)
    offsets = depth_offsets()
    for sign in (1.0, -1.0):
        store.ensure(needed_chunks(store, np.eye(4), mesh, rows, cols, sign, offsets))
        rendered, valid = render(store, np.eye(4), mesh, rows, cols, sign, offsets)
        mask = valid & (reference[14] > 0)
        per_layer = [ncc(rendered[i], reference[i], mask) for i in range(28)]
        shifts = {d: np.nanmean([ncc(rendered[i], reference[i + d], mask) for i in range(max(0, -d), min(28, 28 - d))])
                  for d in range(-3, 4)}
        print(f"[validate-78] sign={sign:+.0f} mean layer NCC={np.nanmean(per_layer):.3f} "
              f"mid={per_layer[14]:.3f} | depth-shift NCC " + " ".join(f"{d:+d}:{v:.3f}" for d, v in shifts.items()),
              flush=True)


def estimate_shift(rendered, reference, valid, depth_search=4, xy_search=4):
    """best (dz, dy, dx) aligning rendered (extra depth margin) with the reference 28 layers."""
    best = (np.nan, 0, 0, 0)
    margin = (rendered.shape[0] - 28) // 2
    ref_hp = np.stack([highpass(layer) for layer in reference])
    mask0 = valid & (reference[14] > 0)
    for dz in range(-depth_search, depth_search + 1):
        stack = rendered[margin + dz: margin + dz + 28]
        stack_hp = np.stack([highpass(layer) for layer in stack])
        for dy in range(-xy_search, xy_search + 1):
            for dx in range(-xy_search, xy_search + 1):
                shifted = np.roll(np.roll(stack_hp, dy, axis=1), dx, axis=2)
                m = np.roll(np.roll(mask0, dy, axis=0), dx, axis=1) & mask0
                m[:xy_search] = m[-xy_search:] = False
                m[:, :xy_search] = m[:, -xy_search:] = False
                score = ncc(shifted[6:22], ref_hp[6:22], np.broadcast_to(m, (16,) + m.shape))
                if not np.isnan(score) and (np.isnan(best[0]) or score > best[0]):
                    best = (score, dz, dy, dx)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-78", action="store_true")
    parser.add_argument("--crop", type=int, nargs=3, default=(7600, 4800, 512), help="row col size for --validate-78")
    parser.add_argument("--sign", type=float, default=None, help="normal sign; set from --validate-78")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--reg-tile", type=int, default=256)
    args = parser.parse_args()
    fetch_inputs()
    if args.validate_78:
        validate_78(args)
        return
    if args.sign is None:
        raise SystemExit("pass --sign from the --validate-78 result")

    import zarr

    mesh = Mesh()
    transform = transform_to("137")
    store = ChunkStore("137", args.workers)
    old = zarr.open(os.path.join(ZARR_DIR, f"{SEGMENT}.zarr"), mode="r")
    offsets = depth_offsets()
    step = float(np.mean(np.diff(offsets)))

    # footprint of the 137 keV region inside the output frame
    all_rows = np.arange(0, OUT_SHAPE[1], 16)
    all_cols = np.arange(0, OUT_SHAPE[2], 16)
    point, normal, valid = mesh.sample(all_rows, all_cols)
    p = (transform[:3, :3] @ point.reshape(-1, 3).T + transform[:3, 3:]).T.reshape(point.shape)
    inside = valid & np.all((p >= 0) & (p < np.array([s * 4 for s in store.shape[::-1]])), axis=-1)
    ys, xs = np.nonzero(inside)
    r0, r1 = max(0, all_rows[ys.min()] - 32), min(OUT_SHAPE[1], all_rows[ys.max()] + 48)
    c0, c1 = max(0, all_cols[xs.min()] - 32), min(OUT_SHAPE[2], all_cols[xs.max()] + 48)
    print(f"[137] ROI in output frame rows {r0}:{r1} cols {c0}:{c1} ({inside.mean():.2%} of sampled mesh)", flush=True)

    # pass 1: coarse registration per tile (render with depth margin, compare to the 78 keV zarr)
    t = args.reg_tile
    margin_layers = 4
    reg_offsets = np.r_[offsets[0] - step * np.arange(margin_layers, 0, -1), offsets,
                        offsets[-1] + step * np.arange(1, margin_layers + 1)].astype(np.float32)
    grid = {}
    for tr in range(r0, r1, t):
        for tc in range(c0, c1, t):
            rows, cols = np.arange(tr, min(tr + t, r1)), np.arange(tc, min(tc + t, c1))
            store.ensure(needed_chunks(store, transform, mesh, rows, cols, args.sign, reg_offsets))
            rendered, v = render(store, transform, mesh, rows, cols, args.sign, reg_offsets)
            if (v & (rendered[margin_layers + 14] > 0)).mean() < 0.3:
                continue
            reference = np.asarray(old[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], np.float32)
            score, dz, dy, dx = estimate_shift(rendered, reference, v & (rendered[margin_layers + 14] > 0))
            grid[(tr, tc)] = (score, dz, dy, dx)
            print(f"[reg] tile {tr},{tc} ncc={score:.3f} dz={dz:+d} dy={dy:+d} dx={dx:+d}", flush=True)
    json.dump({f"{k[0]},{k[1]}": v for k, v in grid.items()}, open(os.path.join(WORK, "registration.json"), "w"))

    # smooth per-tile shifts into a dense field over the ROI (confident tiles only)
    good = {k: v for k, v in grid.items() if v[0] > 0.1}
    if not good:
        raise RuntimeError("registration found no confident tiles")
    centres = np.array([(k[0] + t / 2, k[1] + t / 2) for k in good])
    # rolled[r] = rendered[r - dy] matches reference[r], so sample at r - dy
    values = np.array([[v[1] * step, -v[2], -v[3]] for v in good.values()], np.float64)
    median = np.median(values, axis=0)
    print(f"[reg] {len(good)}/{len(grid)} confident tiles; median shift dz={median[0]:+.2f} (level-0 voxels) "
          f"dy={median[1]:+.1f} dx={median[2]:+.1f} (output px)", flush=True)

    def field(rows, cols):
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        d = (rr[..., None] - centres[:, 0]) ** 2 + (cc[..., None] - centres[:, 1]) ** 2
        w = np.exp(-d / (2 * (1.5 * t) ** 2))
        w_sum = w.sum(-1, keepdims=True)
        est = (w @ values) / np.maximum(w_sum, 1e-9)
        blend = np.clip(w_sum / 0.5, 0, 1)
        return (blend * est + (1 - blend) * median).astype(np.float64)

    # pass 2: final render with the correction into a new zarr in the exact training frame
    out_path = os.path.join(ZARR_DIR, f"{OUT_ID}.zarr")
    if os.path.abspath(out_path) == os.path.abspath(os.path.join(ZARR_DIR, f"{SEGMENT}.zarr")):
        raise RuntimeError("refusing to overwrite the training zarr")
    partial = out_path + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    out = zarr.open(partial, mode="w", shape=OUT_SHAPE, chunks=(8, 64, 64), dtype="<u2",
                    compressor=None, zarr_format=2)
    mask = np.zeros(OUT_SHAPE[1:], np.uint8)
    for tr in range(r0, r1, TILE):
        rows = np.arange(tr, min(tr + TILE, r1))
        cols = np.arange(c0, c1)
        shift = field(rows, cols)
        store.ensure(needed_chunks(store, transform, mesh, rows, cols, args.sign, offsets, margin=8.0))
        rendered, v = render(store, transform, mesh, rows, cols, args.sign, offsets, shift=shift)
        block = np.clip(np.rint(rendered), 0, 255).astype(np.uint16)
        out[:, rows[0]:rows[-1] + 1, c0:c1] = block
        mask[rows[0]:rows[-1] + 1, c0:c1] = (block[14] > 0).astype(np.uint8) * 255
        print(f"[render] rows {rows[-1] + 1 - r0}/{r1 - r0}", flush=True)
    del out
    shutil.rmtree(out_path, ignore_errors=True)
    os.replace(partial, out_path)
    Image.fromarray(mask).save(os.path.join("masks", f"{OUT_ID}.png"))
    json.dump({"segment": SEGMENT, "source": VOLUMES["137"], "level": LEVEL, "roi": [int(r0), int(r1), int(c0), int(c1)],
               "sign": args.sign, "median_shift": median.tolist(), "confident_tiles": len(good)},
              open(os.path.join(WORK, "render_meta.json"), "w"), indent=1)
    print(f"[done] wrote {out_path} and masks/{OUT_ID}.png", flush=True)


if __name__ == "__main__":
    main()
