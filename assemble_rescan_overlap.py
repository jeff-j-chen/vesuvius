"""assemble_rescan_overlap.py -- re-render a scroll segment from a second full-scroll scan.

For each segment, the part of the surface covered by BOTH scans is rendered from each scan into
one shared frame at 9.362 um (28 layers along the mesh normal, 9.362 um apart). Registration
uses only the published per-volume transform.json files:
    target voxel = inv(A_target) @ A_mesh_volume @ mesh voxel   (x, y, z order)
followed by a global residual correction measured on probe patches against the mesh-volume
render. Official researcher ink labels (2.4 um surface grid) are area-resampled into the frame.

outputs per segment <seg> (ids: <seg><keV>, e.g. 20231012184424137 and 2023101218442478):
    ves_zarrs2/<id>.zarr        (28, H, W) uint16, chunks (8, 64, 64), uncompressed
    masks/<id>.png              midslice footprint
    researcher_inklabels/<id>.png
    _ves_tmp/rescan/<seg>/frame.json  crop origin in the full 9.362 um segment frame

usage:
    python3 assemble_rescan_overlap.py 20231012184424 20231007101619
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import map_coordinates

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.abspath(__file__))
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4"
MESH_VOLUME = "20260411134726-2.400um-0.2m-78keV-masked"
VOLUMES = {78: MESH_VOLUME, 137: "20260323153942-2.400um-0.2m-137keV-masked"}
LABEL_DIR = "ink-labels/2.4um-volume-20260411134726/20260918/inklabels.zarr"
WORK = os.path.join(ROOT, "_ves_tmp", "rescan")
CACHE = os.path.join(ROOT, "_ves_tmp", "paris4_137")
SOURCE_UM, TARGET_UM, LAYERS = 2.4, 9.362, 28
STEP = TARGET_UM / SOURCE_UM         # 2.4 um surface px per output px (and per output layer)
MESH_SCALE = 20                      # surface px per tifxyz grid step (scale 0.05)
LEVEL = 2                            # target pyramid level (4x4x4 mean, 9.6 um)
CHUNK = 128
MARGIN = 256                         # level-0 voxels kept inside the target scan
BLOCK = 256
CACHE_CAP = 1500                     # chunks per volume on disk (~3 GB)


def curl_json(url):
    return json.loads(subprocess.run(["curl", "-s", "--fail", "--max-time", "60", url],
                                     capture_output=True, check=True).stdout)


def affine(volume):
    path = os.path.join(CACHE, f"{volume}.transform.json")
    if not os.path.exists(path):
        with open(path, "w") as handle:
            json.dump(curl_json(f"{BUCKET}/volumes/{volume}.zarr/transform.json"), handle)
    matrix = np.eye(4)
    matrix[:3] = np.array(json.load(open(path))["transformation_matrix"])
    return matrix


class Volume:
    """level-LEVEL chunks of one scan, fetched on demand into a capped disk cache."""

    def __init__(self, kev):
        self.kev = kev
        self.base = f"{BUCKET}/volumes/{VOLUMES[kev]}.zarr/{LEVEL}"
        self.shape = tuple(curl_json(f"{self.base}/.zarray")["shape"])
        self.level0_xyz = np.array(curl_json(f"{BUCKET}/volumes/{VOLUMES[kev]}.zarr/0/.zarray")["shape"][::-1], float)
        self.dir = os.path.join(CACHE, f"cache_{kev}kev_l{LEVEL}")
        os.makedirs(self.dir, exist_ok=True)
        self.transform = np.linalg.inv(affine(VOLUMES[kev])) @ affine(MESH_VOLUME)
        self.used = {}
        self.clock = 0

    def _path(self, key):
        return os.path.join(self.dir, "_".join(map(str, key)) + ".raw")

    def _fetch(self, key):
        path = self._path(key)
        if os.path.exists(path):
            return
        tmp = path + ".part"
        code = "000"
        for _ in range(5):
            result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "20", "--max-time", "180", "-o", tmp,
                 "-w", "%{http_code}", f"{self.base}/{key[0]}/{key[1]}/{key[2]}"],
                capture_output=True)
            code = result.stdout.decode()[-3:]
            if code == "200" and os.path.getsize(tmp) == CHUNK ** 3:
                os.replace(tmp, path)
                return
            if code == "404":
                if os.path.exists(tmp):
                    os.remove(tmp)
                open(path, "wb").close()
                return
            time.sleep(2)
        raise RuntimeError(f"{self.kev} keV chunk {key}: http {code}")

    def subvolume(self, lo, hi):
        """dense uint8 array covering level-voxel box [lo, hi) (z, y, x); zeros outside the scan."""
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.shape)
        keys = [(z, y, x)
                for z in range(lo[0] // CHUNK, (hi[0] - 1) // CHUNK + 1)
                for y in range(lo[1] // CHUNK, (hi[1] - 1) // CHUNK + 1)
                for x in range(lo[2] // CHUNK, (hi[2] - 1) // CHUNK + 1)]
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(self._fetch, keys))
        out = np.zeros(tuple(hi - lo), np.uint8)
        for key in keys:
            self.clock += 1
            self.used[key] = self.clock
            path = self._path(key)
            if not os.path.getsize(path):
                continue
            chunk = np.fromfile(path, np.uint8).reshape(CHUNK, CHUNK, CHUNK)
            start = np.array(key) * CHUNK
            a = np.maximum(lo, start)
            b = np.minimum(hi, start + CHUNK)
            out[a[0] - lo[0]:b[0] - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]] = \
                chunk[a[0] - start[0]:b[0] - start[0], a[1] - start[1]:b[1] - start[1], a[2] - start[2]:b[2] - start[2]]
        self._evict()
        return out, lo

    def _evict(self):
        files = [f for f in os.listdir(self.dir) if f.endswith(".raw")]
        if len(files) <= CACHE_CAP:
            return
        keys = [tuple(int(v) for v in f[:-4].split("_")) for f in files]
        keys.sort(key=lambda k: self.used.get(k, 0))
        for key in keys[:len(keys) - CACHE_CAP]:
            os.remove(self._path(key))
            self.used.pop(key, None)

    def sample(self, xyz_mesh):
        """trilinear sample at mesh-volume level-0 xyz points (..., 3); 0 outside the scan."""
        flat = xyz_mesh.reshape(-1, 3)
        q = np.c_[flat, np.ones(len(flat))] @ self.transform.T
        zyx = (q[:, [2, 1, 0]] + 0.5) / 2 ** LEVEL - 0.5
        lo = np.floor(zyx.min(0)).astype(int) - 1
        hi = np.ceil(zyx.max(0)).astype(int) + 2
        dense, origin = self.subvolume(lo, hi)
        values = map_coordinates(dense, (zyx - origin).T, order=1, mode="constant", cval=0.0)
        return values.reshape(xyz_mesh.shape[:-1]).astype(np.float32)

    def inside(self, xyz_mesh):
        flat = xyz_mesh.reshape(-1, 3)
        q = np.c_[flat, np.ones(len(flat))] @ self.transform.T
        ok = np.all((q[:, :3] > MARGIN) & (q[:, :3] < self.level0_xyz - MARGIN), axis=1)
        return ok.reshape(xyz_mesh.shape[:-1])


class Mesh:
    def __init__(self, segment):
        self.dir = os.path.join(WORK, segment, "mesh")
        os.makedirs(self.dir, exist_ok=True)
        base = f"{BUCKET}/segments/{segment}/mesh/{segment}-on-20260411134726-2.4um.tifxyz"
        grids = []
        for axis in "xyz":
            path = os.path.join(self.dir, f"{axis}.tif")
            if not os.path.exists(path):
                subprocess.run(["curl", "-s", "--fail", "--max-time", "600", "-o", path, f"{base}/{axis}.tif"], check=True)
            grids.append(np.array(Image.open(path), np.float32))
        self.grids = grids
        self.valid = ((grids[0] != -1) & (grids[1] != -1) & (grids[2] != -1)).astype(np.float32)
        g = [np.gradient(a) for a in grids]
        normal = np.cross(np.stack([g[0][0], g[1][0], g[2][0]], -1), np.stack([g[0][1], g[1][1], g[2][1]], -1))
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        self.normal = [np.ascontiguousarray(normal[..., i]) for i in range(3)]
        del g, normal
        rows, cols = grids[0].shape
        self.frame_shape = (int(((rows - 1) * MESH_SCALE + 1) / STEP), int(((cols - 1) * MESH_SCALE + 1) / STEP))

    def evaluate(self, rows, cols):
        """output-frame pixel centres -> mesh xyz (..., 3), unit normals, validity."""
        gy = (((rows + 0.5) * STEP - 0.5) / MESH_SCALE).astype(np.float32)
        gx = (((cols + 0.5) * STEP - 0.5) / MESH_SCALE).astype(np.float32)
        remap = lambda a: cv2.remap(a, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        xyz = np.stack([remap(a) for a in self.grids], -1)
        normal = np.stack([remap(a) for a in self.normal], -1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        valid = remap(self.valid) > 0.999
        return xyz, normal, valid


def layer_points(xyz, normal, depth_shift=0.0):
    offsets = (np.arange(LAYERS) - (LAYERS - 1) / 2.0 + depth_shift) * STEP
    return xyz[None] + offsets[:, None, None, None] * normal[None]


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def residual(mesh, low, high, roi_coarse, coarse, n_probes=16, size=160):
    """median depth (layers) and in-plane (px) offset of the high render against the low render."""
    ys, xs = np.nonzero(roi_coarse)
    rng = np.random.default_rng(0)
    found = []
    for index in rng.permutation(len(ys)):
        if len(found) >= n_probes:
            break
        r0, c0 = ys[index] * coarse, xs[index] * coarse
        rr, cc = np.mgrid[r0 - size // 2:r0 + size // 2, c0 - size // 2:c0 + size // 2].astype(np.float32)
        xyz, normal, valid = mesh.evaluate(rr, cc)
        if valid.mean() < 0.95 or not high.inside(xyz[valid]).all():
            continue
        pts = layer_points(xyz, normal)
        a = low.sample(pts)
        b = high.sample(pts)
        if (b[13][valid] == 0).mean() > 0.02:
            continue
        shifts = np.arange(-2, 3)
        scores = np.array([np.mean([corr(b[k][valid], a[k + d][valid]) for k in range(4, 24)]) for d in shifts])
        best = int(np.argmax(scores))
        frac = 0.0
        if 0 < best < 4:
            l, c, r = scores[best - 1:best + 2]
            frac = 0.5 * (l - r) / (l - 2 * c + r)
        (sx, sy), _ = cv2.phaseCorrelate(a[13].astype(np.float64), b[13].astype(np.float64))
        found.append((shifts[best] + frac, sy, sx, scores.max()))
        print(f"  [probe] r{r0} c{c0}: depth {shifts[best] + frac:+.2f} dy {sy:+.2f} dx {sx:+.2f} corr {scores.max():.3f}", flush=True)
    f = np.array(found)
    return float(np.median(f[:, 0])), float(np.median(f[:, 1])), float(np.median(f[:, 2])), f


def label_crop(segment, y0, y1, x0, x1):
    """area-mean of the 2.4 um researcher label over each output pixel of the crop (0..1)."""
    import tensorstore as ts

    store = ts.open({"driver": "zarr3", "kvstore": {"driver": "http",
                     "base_url": f"{BUCKET}/segments/{segment}/{LABEL_DIR}/0"}}).result()
    height, width = store.shape
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    col_edges = np.clip(np.arange(x0, x1 + 1) * STEP, 0, width)
    s0 = int(np.floor(col_edges[0]))
    s1 = int(np.ceil(col_edges[-1]))
    strip = 128
    for r in range(y0, y1, strip):
        r_end = min(r + strip, y1)
        row_edges = np.clip(np.arange(r, r_end + 1) * STEP, 0, height)
        a0, a1 = int(np.floor(row_edges[0])), int(np.ceil(row_edges[-1]))
        block = np.asarray(store[a0:a1, s0:s1].read().result(), np.float32) / 255.0
        integral = cv2.integral(block, sdepth=cv2.CV_64F)

        def interp(ys, xs):
            ys = np.clip(ys, 0, integral.shape[0] - 1)
            xs = np.clip(xs, 0, integral.shape[1] - 1)
            iy, ix = np.floor(ys).astype(int), np.floor(xs).astype(int)
            iy1, ix1 = np.minimum(iy + 1, integral.shape[0] - 1), np.minimum(ix + 1, integral.shape[1] - 1)
            fy, fx = (ys - iy)[:, None], (xs - ix)[None, :]
            return ((1 - fy) * (1 - fx) * integral[iy][:, ix] + (1 - fy) * fx * integral[iy][:, ix1]
                    + fy * (1 - fx) * integral[iy1][:, ix] + fy * fx * integral[iy1][:, ix1])

        grid = interp(row_edges - a0, col_edges - s0)
        area = np.outer(np.diff(row_edges), np.diff(col_edges))
        out[r - y0:r_end - y0] = (grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1]) / np.maximum(area, 1e-6)
    return out


def assemble(segment):
    t0 = time.time()
    work = os.path.join(WORK, segment)
    ids = {kev: f"{segment}{kev}" for kev in VOLUMES}
    outs = {kev: os.path.join(ROOT, "ves_zarrs2", f"{ids[kev]}.zarr") for kev in VOLUMES}
    if any(os.path.exists(p) for p in outs.values()):
        raise SystemExit(f"{segment}: output zarr exists; refusing to overwrite")
    mesh = Mesh(segment)
    low, high = Volume(78), Volume(137)
    height, width = mesh.frame_shape
    coarse = 8
    rr, cc = np.mgrid[0:height:coarse, 0:width:coarse].astype(np.float32)
    xyz, _, valid = mesh.evaluate(rr, cc)
    roi_coarse = valid & high.inside(xyz) & low.inside(xyz)
    ys, xs = np.nonzero(roi_coarse)
    y0, y1 = max(0, ys.min() * coarse - coarse), min(height, (ys.max() + 2) * coarse)
    x0, x1 = max(0, xs.min() * coarse - coarse), min(width, (xs.max() + 2) * coarse)
    print(f"[{segment}] frame {height}x{width} @ {TARGET_UM} um | overlap {roi_coarse.sum() * coarse ** 2 / 1e6:.1f}M px "
          f"= {roi_coarse.sum() / valid.sum():.3f} of segment | crop y {y0}:{y1} x {x0}:{x1}", flush=True)

    depth, dy, dx, probes = residual(mesh, low, high, roi_coarse, coarse)
    print(f"[{segment}] residual (137 vs 78): depth {depth:+.2f} layers, dy {dy:+.2f}, dx {dx:+.2f} px "
          f"(corr {probes[:, 3].min():.2f}-{probes[:, 3].max():.2f})", flush=True)
    with open(os.path.join(work, "frame.json"), "w") as handle:
        json.dump({"segment": segment, "ids": ids, "target_um": TARGET_UM, "layers": LAYERS,
                   "full_frame": [height, width], "crop_yx": [int(y0), int(y1), int(x0), int(x1)],
                   "residual_correction_137": {"depth_layers": -depth, "dy_px": dy, "dx_px": dx},
                   "probes": probes.tolist()}, handle, indent=1)

    shape = (LAYERS, y1 - y0, x1 - x0)
    stores = {kev: zarr.open(outs[kev] + ".partial", mode="w", shape=shape, chunks=(8, 64, 64), dtype="<u2",
                             compressor=None, write_empty_chunks=False) for kev in VOLUMES}
    roi = cv2.resize(roi_coarse.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1] > 0
    blocks = [(r, c) for r in range(0, shape[1], BLOCK) for c in range(0, shape[2], BLOCK)
              if roi[r:r + BLOCK, c:c + BLOCK].any()]
    for index, (r, c) in enumerate(blocks):
        h, w = min(BLOCK, shape[1] - r), min(BLOCK, shape[2] - c)
        br, bc = np.mgrid[y0 + r:y0 + r + h, x0 + c:x0 + c + w].astype(np.float32)
        keep = roi[r:r + h, c:c + w].copy()
        for kev, volume in ((78, low), (137, high)):
            shift = (0.0, 0.0, 0.0) if kev == 78 else (-depth, dy, dx)
            xyz, normal, valid = mesh.evaluate(br + np.float32(shift[1]), bc + np.float32(shift[2]))
            keep &= valid
            values = volume.sample(layer_points(xyz, normal, shift[0]))
            values[:, ~keep] = 0
            stores[kev][:, r:r + h, c:c + w] = np.clip(np.rint(values), 0, 255).astype(np.uint16)
        if index % 20 == 0:
            print(f"[{segment}] block {index + 1}/{len(blocks)} ({time.time() - t0:.0f}s)", flush=True)

    for kev in VOLUMES:
        os.replace(outs[kev] + ".partial", outs[kev])
        mid = np.asarray(zarr.open(outs[kev], mode="r")[LAYERS // 2]) > 0
        cv2.imwrite(os.path.join(ROOT, "masks", f"{ids[kev]}.png"), mid.astype(np.uint8) * 255)
    label = label_crop(segment, y0, y1, x0, x1)
    os.makedirs(os.path.join(ROOT, "researcher_inklabels"), exist_ok=True)
    binary = (label >= 0.5).astype(np.uint8) * 255
    for kev in VOLUMES:
        cv2.imwrite(os.path.join(ROOT, "researcher_inklabels", f"{ids[kev]}.png"), binary)
    print(f"[{segment}] done: {ids} | label positive {float((binary > 0).mean()):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", nargs="+")
    for segment in parser.parse_args().segments:
        assemble(segment)
