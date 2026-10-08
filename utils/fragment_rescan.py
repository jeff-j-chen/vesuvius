"""re-render dl.ash2txt fragments from their high-energy (88 keV) scan into the 28-layer 9.362 um frame.

geometry: the exposed-surface `result.ppm` gives, per 3.24 um surface pixel, (x, y, z) and the unit
normal in the low-energy (53/54 keV) volume. output pixel (r, c) samples source pixel
((r + 0.5) * S - 0.5, (c + 0.5) * S - 0.5), S = 9.362 / 3.24, and output layer k sits (k - 13.5) * S
voxels along the normal -- the same frame as the old surface-tiff assembly, so labels, masks and splits
carry over. a fitted affine (low-energy voxel xyz -> 88 keV voxel xyz; the fragments have no published
registration) maps the points into the 88 keV scan. sampling = 3-voxel box pre-filter + trilinear.
some published 88 keV zarrs are missing chunks under the surface; those are rebuilt from the raw slice
tiffs.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

SOURCE_UM, TARGET_UM, LAYERS = 3.24, 9.362, 28
S = TARGET_UM / SOURCE_UM
CHUNK = 128


def out_shape(src_h, src_w):
    return int(round(src_h * SOURCE_UM / TARGET_UM)), int(round(src_w * SOURCE_UM / TARGET_UM))


def read_ppm_grid(url, cache_path):
    """stream the PPM once; keep xyz + normal bilinearly sampled at output pixel centres."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["xyz"], data["normal"], data["valid"]
    proc = subprocess.Popen(["curl", "-s", "--fail", "--max-time", "3600", url], stdout=subprocess.PIPE, bufsize=1 << 24)
    header = {}
    while True:
        line = proc.stdout.readline().decode().strip()
        if line == "<>":
            break
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    width, height = int(header["width"]), int(header["height"])
    oh, ow = out_shape(height, width)
    src_rows = (np.arange(oh) + 0.5) * S - 0.5
    src_cols = (np.arange(ow) + 0.5) * S - 0.5
    c0 = np.clip(np.floor(src_cols).astype(int), 0, width - 1)
    c1 = np.clip(c0 + 1, 0, width - 1)
    fc = (src_cols - np.floor(src_cols)).astype(np.float32)[:, None]
    grid = np.zeros((oh, ow, 6), np.float32)
    ok = np.zeros((oh, ow), bool)
    row_bytes = width * 6 * 8
    previous = None
    target = 0
    for row in range(height):
        buf = proc.stdout.read(row_bytes)
        if len(buf) != row_bytes:
            raise RuntimeError(f"{url}: truncated at row {row}")
        current = np.frombuffer(buf, "<f8").reshape(width, 6)
        while target < oh and np.floor(src_rows[target]) + 1 <= row:
            r0 = int(np.floor(src_rows[target]))
            fr = src_rows[target] - r0
            lower = current if r0 < 0 or row != r0 + 1 else previous
            a = lower[c0] * (1 - fc) + lower[c1] * fc
            b = current[c0] * (1 - fc) + current[c1] * fc
            grid[target] = (a * (1 - fr) + b * fr).astype(np.float32)
            corners = np.stack([lower[c0], lower[c1], current[c0], current[c1]])
            ok[target] = np.all(np.abs(corners[..., 3:]).sum(-1) > 0.5, axis=0)
            target += 1
        previous = current
    while target < oh:
        grid[target] = grid[target - 1]
        target += 1
    proc.stdout.close()
    proc.wait()
    normal = grid[..., 3:]
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path + ".partial.npz", xyz=grid[..., :3], normal=normal, valid=ok)
    os.replace(cache_path + ".partial.npz", cache_path)
    return grid[..., :3], normal, ok


class RemoteZarr:
    """blosc-compressed uint16 zarr (level 0) fetched chunk by chunk into a capped disk cache."""

    def __init__(self, url, cache_dir, cap=800):
        self.url = url.rstrip("/")
        meta = json.loads(subprocess.run(["curl", "-s", "--fail", f"{self.url}/.zarray"], capture_output=True, check=True).stdout)
        self.shape = tuple(meta["shape"])
        self.codec = numcodecs.get_codec(meta["compressor"])
        self.dir = cache_dir
        self.cap = cap
        self.pinned = frozenset()
        os.makedirs(cache_dir, exist_ok=True)
        self.used, self.clock = {}, 0

    def _path(self, key):
        return os.path.join(self.dir, "_".join(map(str, key)))

    def _fetch(self, key):
        path = self._path(key)
        if os.path.exists(path):
            return
        part = f"{path}.{threading.get_ident()}.part"  # parallel tiles may request the same chunk
        code = "000"
        for _ in range(5):
            result = subprocess.run(["curl", "-s", "--connect-timeout", "20", "--max-time", "300", "-o", part,
                                     "-w", "%{http_code}", f"{self.url}/{key[0]}/{key[1]}/{key[2]}"], capture_output=True)
            code = result.stdout.decode()[-3:]
            if code == "200":
                os.replace(part, path)
                return
            if code == "404":
                if os.path.exists(part):
                    os.remove(part)
                if not os.path.exists(path):
                    open(path, "wb").close()
                return
            time.sleep(2)
        raise RuntimeError(f"{self.url} chunk {key}: http {code}")

    def box(self, lo, hi):
        """dense uint16 (z, y, x) array over [lo, hi); zero outside the volume."""
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        lo_c, hi_c = np.maximum(lo, 0), np.minimum(hi, self.shape)
        out = np.zeros(tuple(np.maximum(hi - lo, 1)), np.uint16)
        if np.any(hi_c <= lo_c):
            return out
        keys = [(z, y, x) for z in range(lo_c[0] // CHUNK, (hi_c[0] - 1) // CHUNK + 1)
                for y in range(lo_c[1] // CHUNK, (hi_c[1] - 1) // CHUNK + 1)
                for x in range(lo_c[2] // CHUNK, (hi_c[2] - 1) // CHUNK + 1)]
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(self._fetch, keys))
        for key in keys:
            self.clock += 1
            self.used[key] = self.clock
            path = self._path(key)
            if not os.path.getsize(path):
                continue
            raw = self.codec.decode(open(path, "rb").read())
            start = np.array(key) * CHUNK
            size = np.minimum(start + CHUNK, self.shape) - start
            chunk = np.frombuffer(raw, "<u2").reshape(CHUNK, CHUNK, CHUNK)[:size[0], :size[1], :size[2]]
            a, b = np.maximum(lo_c, start), np.minimum(hi_c, start + size)
            out[tuple(slice(a[i] - lo[i], b[i] - lo[i]) for i in range(3))] = \
                chunk[tuple(slice(a[i] - start[i], b[i] - start[i]) for i in range(3))]
        self._evict()
        return out

    def _evict(self):
        if self.cap is None:
            return
        files = [f for f in os.listdir(self.dir) if not f.endswith(".part")]
        if len(files) <= self.cap + len(self.pinned):
            return
        keys = sorted((tuple(int(v) for v in f.split("_")) for f in files), key=lambda k: self.used.get(k, 0))
        keys = [k for k in keys if k not in self.pinned]
        for key in keys[:max(len(files) - len(self.pinned) - self.cap, 0)]:
            os.remove(self._path(key))
            self.used.pop(key, None)

    def prefetch(self, keys, workers=32):
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(self._fetch, keys))

    def evict_except(self, keep):
        """drop cached chunks outside `keep` (pinned tif-filled chunks always stay)."""
        keep = set(keep) | set(self.pinned)
        for f in os.listdir(self.dir):
            if f.endswith(".part"):
                continue
            key = tuple(int(v) for v in f.split("_"))
            if key not in keep:
                os.remove(os.path.join(self.dir, f))
                self.used.pop(key, None)

    def missing(self, keys):
        """keys whose published chunk is absent (404), whether or not a tif-filled copy is cached."""
        def status(k):
            for _ in range(5):
                out = subprocess.run(["curl", "-s", "-I", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "60",
                                      f"{self.url}/{k[0]}/{k[1]}/{k[2]}"], capture_output=True).stdout.decode()
                if out in ("200", "404"):
                    return k, out
                time.sleep(2)
            raise RuntimeError(f"{self.url} chunk {k}: HEAD status {out}")
        with ThreadPoolExecutor(32) as pool:
            return sorted(k for k, s in pool.map(status, keys) if s == "404")

    def _tif_data_offset(self, url):
        """byte offset of the pixel data of an uncompressed single-strip uint16 tiff of this volume's XY size."""
        head = subprocess.run(["curl", "-s", "--fail", "-r", "0-7", url], capture_output=True, check=True).stdout
        if head[:4] != b"II*\x00":
            raise RuntimeError(f"{url}: not a little-endian classic tiff")
        start = int.from_bytes(head[4:8], "little")  # the IFD usually trails the pixel data
        head = subprocess.run(["curl", "-s", "--fail", "-r", f"{start}-{start + 4095}", url],
                              capture_output=True, check=True).stdout
        ifd = 0
        tags = {}
        for i in range(int.from_bytes(head[ifd:ifd + 2], "little")):
            e = head[ifd + 2 + 12 * i: ifd + 14 + 12 * i]
            tag, typ, count = int.from_bytes(e[0:2], "little"), int.from_bytes(e[2:4], "little"), int.from_bytes(e[4:8], "little")
            value = int.from_bytes(e[8:10], "little") if typ == 3 else int.from_bytes(e[8:12], "little")
            tags[tag] = (count, value)
        W, H = tags[256][1], tags[257][1]
        if (H, W) != tuple(self.shape[1:]) or tags[258][1] != 16 or tags.get(259, (1, 1))[1] != 1 or tags[273][0] != 1:
            raise RuntimeError(f"{url}: unexpected tiff layout {tags}")
        return tags[273][1]

    def fill_from_tifs(self, tif_dir_url, keys, digits):
        """build chunks missing from the published zarr out of the raw slice tiffs (uncompressed
        single-strip uint16, layout checked), one http range per slice covering only the needed rows.
        filled chunks are pinned in the cache."""
        todo = sorted(k for k in keys if not (os.path.exists(self._path(k)) and os.path.getsize(self._path(k))))
        self.pinned = frozenset(self.pinned | set(keys))
        if not todo:
            return
        H, W = self.shape[1], self.shape[2]
        offset = self._tif_data_offset(f"{tif_dir_url}/{todo[0][0] * CHUNK:0{digits}d}.tif")
        rows = {}
        for k in todo:
            rows.setdefault((k[0], k[1]), []).append(k[2])
        t0 = time.time()
        for n, ((zc, yc), xcs) in enumerate(sorted(rows.items())):
            y0, y1 = yc * CHUNK, min(yc * CHUNK + CHUNK, H)
            z0, z1 = zc * CHUNK, min(zc * CHUNK + CHUNK, self.shape[0])
            block = np.zeros((CHUNK, CHUNK, W), np.uint16)

            def read(z):
                lo = offset + y0 * W * 2
                hi = offset + y1 * W * 2 - 1
                for _ in range(5):
                    raw = subprocess.run(["curl", "-s", "--fail", "--max-time", "300", "-r", f"{lo}-{hi}",
                                          f"{tif_dir_url}/{z:0{digits}d}.tif"], capture_output=True).stdout
                    if len(raw) == hi - lo + 1:
                        block[z - z0, :y1 - y0] = np.frombuffer(raw, "<u2").reshape(y1 - y0, W)
                        return
                    time.sleep(2)
                raise RuntimeError(f"tif {z}: short read")

            with ThreadPoolExecutor(16) as pool:
                list(pool.map(read, range(z0, z1)))
            for xc in xcs:
                chunk = np.zeros((CHUNK, CHUNK, CHUNK), np.uint16)
                part = block[:, :, xc * CHUNK:xc * CHUNK + CHUNK]
                chunk[:, :, :part.shape[2]] = part
                path = self._path((zc, yc, xc))
                with open(path + ".part", "wb") as fh:
                    fh.write(self.codec.encode(chunk))
                os.replace(path + ".part", path)
            print(f"  [rescan] tif fill row {n + 1}/{len(rows)} (z{zc} y{yc}, {len(xcs)} chunks, {time.time() - t0:.0f}s)",
                  flush=True)

    def sample(self, zyx, smooth=1):
        """trilinear sample at float voxel (z, y, x) points (..., 3); optional box pre-filter."""
        flat = zyx.reshape(-1, 3)
        lo = np.floor(flat.min(0)).astype(int) - smooth - 1
        hi = np.ceil(flat.max(0)).astype(int) + smooth + 2
        dense = self.box(lo, hi).astype(np.float32)
        if smooth > 1:
            dense = uniform_filter(dense, size=smooth, mode="nearest")
        values = map_coordinates(dense, (flat - lo).T, order=1, mode="constant", cval=0.0)
        return values.reshape(zyx.shape[:-1])


def layer_points_zyx(xyz, normal, transform=None, sign=1.0, depth_shift=0.0):
    """(28, ..., 3) zyx sample points in the target volume; transform maps source xyz -> target xyz."""
    offsets = sign * (np.arange(LAYERS) - (LAYERS - 1) / 2.0 + depth_shift) * S
    pts = xyz[None] + offsets.reshape((-1,) + (1,) * xyz.ndim) * normal[None]
    if transform is not None:
        flat = pts.reshape(-1, 3)
        pts = (np.c_[flat, np.ones(len(flat))] @ np.asarray(transform)[:3].T).reshape(pts.shape)
    return pts[..., ::-1]


def field_at(field, grid_step, r0, r1, c0, c1):
    """bilinear (r1-r0, c1-c0, 3) xyz residual from a coarse field sampled at output pixels (i*step, j*step)."""
    import cv2

    rows = (np.arange(r0, r1, dtype=np.float32) / grid_step)
    cols = (np.arange(c0, c1, dtype=np.float32) / grid_step)
    gx, gy = np.meshgrid(cols, rows)
    return np.stack([cv2.remap(np.ascontiguousarray(field[..., i]), gx, gy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for i in range(3)], -1)


def needed_keys(xyz, valid, normal, transform=None, shape=None, margin=4, step=4, offset=None):
    """chunk keys touched by the 28-layer sample points of all valid pixels (subsampled)."""
    sub = valid[::step, ::step]
    pts = layer_points_zyx(xyz[::step, ::step], normal[::step, ::step], transform=transform)
    if offset is not None:
        pts = pts + offset[::step, ::step, ::-1][None]
    pts = pts[:, sub].reshape(-1, 3)
    keys = set()
    for dz in (-margin, margin):
        for dy in (-margin, margin):
            for dx in (-margin, margin):
                q = np.floor((pts + [dz, dy, dx]) / CHUNK).astype(int)
                if shape is not None:
                    q = q[np.all((q >= 0) & (q < -(-np.array(shape) // CHUNK)), axis=1)]
                keys |= set(map(tuple, np.unique(q, axis=0).tolist()))
    return sorted(keys)


def extend_grid(xyz, normal, valid, margin):
    """continue the surface up to `margin` px beyond the valid PPM pixels: nearest valid pixel plus its local
    tangent (d xyz / d row, col) times the offset; normals copied. returns xyz, normal, extended-valid."""
    from scipy.ndimage import distance_transform_edt

    def fill(values, have):
        _, (ri, ci) = distance_transform_edt(~have, return_indices=True)
        return values[ri, ci], ri, ci

    jr = np.zeros_like(xyz)
    jc = np.zeros_like(xyz)
    jr[1:-1] = (xyz[2:] - xyz[:-2]) / 2
    jc[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / 2
    ok_r = np.zeros_like(valid)
    ok_c = np.zeros_like(valid)
    ok_r[1:-1] = valid[2:] & valid[:-2]
    ok_c[:, 1:-1] = valid[:, 2:] & valid[:, :-2]
    jr, _, _ = fill(jr, ok_r)
    jc, _, _ = fill(jc, ok_c)
    dist = distance_transform_edt(~valid)
    ext = (~valid) & (dist <= margin)
    _, ri, ci = fill(xyz, valid)
    rows, cols = np.mgrid[0:valid.shape[0], 0:valid.shape[1]]
    dr = (rows - ri)[..., None].astype(np.float32)
    dc = (cols - ci)[..., None].astype(np.float32)
    out_xyz = xyz.copy()
    out_nrm = normal.copy()
    out_xyz[ext] = (xyz[ri, ci] + jr[ri, ci] * dr + jc[ri, ci] * dc)[ext]
    out_nrm[ext] = normal[ri, ci][ext]
    return out_xyz, out_nrm, valid | ext


def render(out_zarr, ppm_url, volume_url, affine, cache_dir, footprint=None, empty_layers=(),
           tif_url=None, tif_digits=4, chunks=(8, 64, 64), offset_field=None, field_step=64,
           margin=48, render_threads=8, fetch_threads=32):
    """render the fragment into a new uint16 (28, H, W) zarr at out_zarr; returns the bool footprint
    (valid PPM & optional footprint & inside the scan & nonzero at every rendered layer).
    offset_field: optional coarse (gh, gw, 3) xyz residual in target voxels added after the affine.
    margin: px of extrapolated surface rendered beyond the mesh (context only, never in the footprint)."""
    import zarr

    xyz, nrm, mesh = read_ppm_grid(ppm_url, os.path.join(cache_dir, "ppm_grid.npz"))
    H, W = mesh.shape
    if footprint is None:
        footprint = np.ones((H, W), bool)
    if footprint.shape != (H, W):
        raise RuntimeError(f"footprint {footprint.shape} != ppm frame {(H, W)}")
    xyz, nrm, ok = extend_grid(xyz, nrm, mesh, margin) if margin else (xyz, nrm, mesh.copy())
    keep = mesh & footprint
    aff = np.asarray(affine, float)
    # eviction is band-based (see below), not LRU
    vol = RemoteZarr(volume_url, os.path.join(cache_dir, "chunks"), cap=None)
    offset = None if offset_field is None else field_at(offset_field, field_step, 0, H, 0, W)
    keys = needed_keys(xyz, ok, nrm, transform=aff, shape=vol.shape, offset=offset)
    missing = vol.missing(keys)
    print(f"  [rescan] {len(keys)} chunks under the surface, {len(missing)} missing from the published zarr", flush=True)
    if missing:
        if not tif_url:
            raise RuntimeError("published zarr is incomplete under the surface and no slice-tiff dir is configured")
        vol.fill_from_tifs(tif_url, missing, tif_digits)
    full_layers = [k for k in range(LAYERS) if k not in set(empty_layers)]
    out = zarr.open(out_zarr, mode="w", shape=(LAYERS, H, W), chunks=chunks, dtype="<u2", compressor=None,
                    fill_value=0, zarr_format=2)
    mask = np.zeros((H, W), bool)
    tile, max_box = 128, 80e6

    def tile_render(r0, r1, c0, c1):
        sub = ok[r0:r1, c0:c1]
        if not sub.any():
            return
        pts = layer_points_zyx(xyz[r0:r1, c0:c1], nrm[r0:r1, c0:c1], transform=aff)
        if offset_field is not None:
            pts = pts + field_at(offset_field, field_step, r0, r1, c0, c1)[None, ..., ::-1]
        valid_pts = pts[:, sub].reshape(-1, 3)
        lo, hi = np.floor(valid_pts.min(0)), np.ceil(valid_pts.max(0))
        if np.prod(hi - lo + 8) > max_box and (r1 - r0) > 8:
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            for a, b, c, d in ((r0, rm, c0, cm), (r0, rm, cm, c1), (rm, r1, c0, cm), (rm, r1, cm, c1)):
                tile_render(a, b, c, d)
            return
        inside = np.all((pts >= 1) & (pts <= np.array(vol.shape) - 2), axis=-1).all(0) & sub
        if not inside.any():
            return
        vals = np.zeros(pts.shape[:3], np.float32)
        vals[:, inside] = vol.sample(pts[:, inside], smooth=3)
        inside &= (vals[full_layers] > 0).all(0)  # outside the scanned field of view the volume is 0
        vals[:, ~inside] = 0
        vals[list(empty_layers)] = 0
        out[:, r0:r1, c0:c1] = np.clip(np.rint(vals), 0, 65535).astype(np.uint16)
        mask[r0:r1, c0:c1] = inside & keep[r0:r1, c0:c1]

    tiles = [(r, c) for r in range(0, H, tile) for c in range(0, W, tile)]
    bands = list(range(0, H, tile))

    def band_keys(r0):
        r1 = min(r0 + tile, H)
        if not ok[r0:r1].any():
            return []
        return needed_keys(xyz[r0:r1], ok[r0:r1], nrm[r0:r1], transform=aff, shape=vol.shape, margin=6,
                           offset=None if offset is None else offset[r0:r1])

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(1) as fetcher, ThreadPoolExecutor(render_threads) as renderer:
        next_keys = band_keys(bands[0])
        pending = fetcher.submit(vol.prefetch, next_keys, fetch_threads)
        for b, r in enumerate(bands):
            pending.result()
            next_keys = band_keys(bands[b + 1]) if b + 1 < len(bands) else []
            pending = fetcher.submit(vol.prefetch, next_keys, fetch_threads)
            row = [(r, min(r + tile, H), c, min(c + tile, W)) for c in range(0, W, tile)]
            list(renderer.map(lambda args: tile_render(*args), row))
            done += len(row)
            pending.result()
            vol.evict_except(set(next_keys))
            if b % 4 == 0 or b == len(bands) - 1:
                print(f"  [rescan] tile {done}/{len(tiles)} ({time.time() - t0:.0f}s)", flush=True)
    return mask
