"""render_9um_surface.py — render the PHerc0139 w044 flattened surface (single slice or a
multi-layer surface volume) from the 9.362um tifxyz mesh + the raw 9.362um zarr on S3.

no C++ VC3D / villa build needed. the mesh gives, for each point on the FLATTENED sheet,
its (x,y,z) voxel in the raw scan. we:
  1. load the tifxyz mesh (x/y/z float32 grids, -1 = invalid)
  2. upsample the coordinate grid to full flattened resolution (1 px per voxel-step)
  3. NEAREST-sample the raw volume at those voxels (NO trilinear interpolation, per plan)
for a multi-layer surface volume we offset along the local surface NORMAL by +/- steps.

the raw 9.362um zarr is uint8, chunks 128^3, NO compressor, dimension_separator '/', so each
chunk file is a raw 128*128*128 = 2,097,152-byte blob. we curl the chunks the surface passes
through (threaded), cache them on disk (reusable across layers + reruns), then sample. air
chunks that don't exist on S3 return 404 -> treated as fill_value 0.

usage (step 3, single mid slice for visual verification):
  python render_9um_surface.py --layers 1 --out-png _ves_tmp/p0139_w044/w044_9um_slice.png

usage (step 5, full surface volume -> our training zarr):
  python render_9um_surface.py --layers 64 --normal-step 1.0 \
      --out-zarr C:/Users/ChenJeff/Documents/ves_zarrs2/<id>.zarr --out-id <id>
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# raw 9.362um masked volume, level 0. axes (z, y, x). uint8, chunks 128^3, no compressor.
# defaults = PHerc0139; override via --vol-base / --vol-shape for other scrolls (e.g. PHerc0191).
S3_BASE = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/"
           "20250728140407-9.362um-1.2m-113keV-masked.zarr/0")
VOL_SHAPE = (20974, 6621, 6621)   # (z, y, x)
CHUNK = 128
CHUNK_BYTES = CHUNK * CHUNK * CHUNK  # uint8


def _fetch_chunk(args):
    """download one raw uint8 128^3 chunk to the cache dir. 404 (air) -> sentinel empty file.
    returns (key, status). hardened curl (timeouts+retry) so one hung request can't freeze us."""
    zc, yc, xc, cache_dir = args
    out = os.path.join(cache_dir, f"{zc}_{yc}_{xc}.raw")
    if os.path.exists(out):
        return (zc, yc, xc), "cached"
    url = f"{S3_BASE}/{zc}/{yc}/{xc}"
    r = subprocess.run(
        ["curl.exe", "-s", "--fail", "--connect-timeout", "20", "--max-time", "120",
         "--retry", "4", "--retry-delay", "2", "--retry-all-errors", url, "-o", out],
        capture_output=True)
    if r.returncode != 0:
        # 404 / missing = all-air chunk -> write a zero-length sentinel so we skip re-fetch
        open(out, "wb").close()
        return (zc, yc, xc), "air"
    return (zc, yc, xc), "ok"


def _load_chunk(cache_dir, zc, yc, xc):
    """read a cached chunk as (128,128,128) uint8; air/empty sentinel -> zeros."""
    p = os.path.join(cache_dir, f"{zc}_{yc}_{xc}.raw")
    try:
        if os.path.getsize(p) == CHUNK_BYTES:
            return np.frombuffer(open(p, "rb").read(), dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
    except Exception:
        pass
    return None  # air / missing -> caller fills 0


def load_mesh(mesh_dir):
    """load tifxyz mesh -> X,Y,Z float32 grids + valid mask. x/y/z.tif hold x/y/z coords."""
    X = np.array(Image.open(os.path.join(mesh_dir, "x.tif"))).astype(np.float32)
    Y = np.array(Image.open(os.path.join(mesh_dir, "y.tif"))).astype(np.float32)
    Z = np.array(Image.open(os.path.join(mesh_dir, "z.tif"))).astype(np.float32)
    valid = (X != -1) & (Y != -1) & (Z != -1)
    return X, Y, Z, valid


def upsample(grid, W, H):
    """bilinear upsample a coordinate grid to (H,W)."""
    return cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)


def compute_normals(Xu, Yu, Zu):
    """per-pixel unit surface normal from gradients of the upsampled coord maps.
    P(u,v) = (x,y,z); normal = normalize(dP/dv x dP/du)."""
    dxv, dxu = np.gradient(Xu)
    dyv, dyu = np.gradient(Yu)
    dzv, dzu = np.gradient(Zu)
    # tangent along u (axis1) and v (axis0)
    tu = np.stack([dxu, dyu, dzu], axis=-1)
    tv = np.stack([dxv, dyv, dzv], axis=-1)
    n = np.cross(tv, tu)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    nn[nn == 0] = 1.0
    return (n / nn).astype(np.float32)   # (H,W,3) in (x,y,z) order


def render(mesh_dir, cache_dir, layers, normal_step, upsample_factor, workers,
           out_png=None, out_zarr=None, out_id=None, crop_valid=False, crop_margin=8):
    os.makedirs(cache_dir, exist_ok=True)
    X, Y, Z, valid = load_mesh(mesh_dir)
    # crop to the valid bounding box (+margin) FIRST. some meshes store a small compact
    # sheet inside a huge mostly-empty padded grid (e.g. a 344x455 blob in a 6203x6203 grid);
    # upsampling the full grid would make a ~124k x 124k canvas. cropping renders only the
    # real surface. harmless for meshes whose valid region already fills the grid.
    if crop_valid and valid.any():
        ys, xs = np.where(valid)
        y0 = max(0, int(ys.min()) - crop_margin); y1 = min(valid.shape[0], int(ys.max()) + 1 + crop_margin)
        x0 = max(0, int(xs.min()) - crop_margin); x1 = min(valid.shape[1], int(xs.max()) + 1 + crop_margin)
        print(f"[mesh] crop valid bbox grid[{y0}:{y1}, {x0}:{x1}] from {X.shape}", flush=True)
        X = X[y0:y1, x0:x1]; Y = Y[y0:y1, x0:x1]; Z = Z[y0:y1, x0:x1]; valid = valid[y0:y1, x0:x1]
    gh, gw = X.shape
    H = int(round((gh - 1) * upsample_factor)) + 1
    W = int(round((gw - 1) * upsample_factor)) + 1
    print(f"[mesh] grid {gh}x{gw} -> flattened {H}x{W}  valid={valid.mean():.3f}", flush=True)

    Xu = upsample(X, W, H); Yu = upsample(Y, W, H); Zu = upsample(Z, W, H)
    validu = upsample(valid.astype(np.float32), W, H) > 0.999   # conservative: drop edges

    # depth layer offsets (centered): e.g. layers=64 -> offsets -31.5..+31.5 * normal_step
    if layers > 1:
        normals = compute_normals(Xu, Yu, Zu)
        offsets = (np.arange(layers) - (layers - 1) / 2.0) * normal_step
    else:
        normals = None
        offsets = np.array([0.0])

    # ---- phase A: figure out every chunk any layer touches, download once (threaded) ----
    # encode each chunk coord as a single int64 (zc,yc,xc) and take ONE unique over all
    # offsets concatenated — far faster than np.unique(axis=0) structured row-sort per offset.
    GYC = VOL_SHAPE[1] // CHUNK + 2   # y-chunk stride for encoding
    GXC = VOL_SHAPE[2] // CHUNK + 2   # x-chunk stride
    vm = validu
    codes_all = []
    for off in offsets:
        if normals is not None:
            xs = Xu[vm] + normals[..., 0][vm] * off
            ys = Yu[vm] + normals[..., 1][vm] * off
            zs = Zu[vm] + normals[..., 2][vm] * off
        else:
            xs, ys, zs = Xu[vm], Yu[vm], Zu[vm]
        zc = (np.clip(np.rint(zs), 0, VOL_SHAPE[0] - 1).astype(np.int64)) // CHUNK
        yc = (np.clip(np.rint(ys), 0, VOL_SHAPE[1] - 1).astype(np.int64)) // CHUNK
        xc = (np.clip(np.rint(xs), 0, VOL_SHAPE[2] - 1).astype(np.int64)) // CHUNK
        codes_all.append((zc * GYC + yc) * GXC + xc)
    codes = np.unique(np.concatenate(codes_all))
    zc = codes // (GYC * GXC)
    rem = codes % (GYC * GXC)
    yc = rem // GXC
    xc = rem % GXC
    need = set(zip(zc.tolist(), yc.tolist(), xc.tolist()))
    print(f"[chunks] {len(need)} unique chunks to ensure (~{len(need)*2/1024:.1f} GB max)", flush=True)

    jobs = [(zc, yc, xc, cache_dir) for (zc, yc, xc) in sorted(need)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _key, _st in pool.map(_fetch_chunk, jobs):
            done += 1
            if done % 500 == 0:
                print(f"[chunks] fetched {done}/{len(jobs)}", flush=True)
    print(f"[chunks] all {len(jobs)} present", flush=True)

    # ---- phase B: sample each layer (nearest neighbor) ----
    def sample_layer(off):
        if normals is not None:
            xs = Xu + normals[..., 0] * off
            ys = Yu + normals[..., 1] * off
            zs = Zu + normals[..., 2] * off
        else:
            xs, ys, zs = Xu, Yu, Zu
        zi = np.clip(np.rint(zs), 0, VOL_SHAPE[0] - 1).astype(np.int32)
        yi = np.clip(np.rint(ys), 0, VOL_SHAPE[1] - 1).astype(np.int32)
        xi = np.clip(np.rint(xs), 0, VOL_SHAPE[2] - 1).astype(np.int32)
        out = np.zeros((H, W), dtype=np.uint8)
        zc = zi // CHUNK; yc = yi // CHUNK; xc = xi // CHUNK
        # group pixels by chunk to sample each cached chunk once
        cid = (zc.astype(np.int64) * 100000 + yc) * 100000 + xc
        flat_cid = cid[validu]
        ys_i = np.where(validu)
        order = np.argsort(flat_cid, kind="stable")
        rows = ys_i[0][order]; cols = ys_i[1][order]
        scid = flat_cid[order]
        uniq, starts = np.unique(scid, return_index=True)
        starts = list(starts) + [len(scid)]
        for gi in range(len(uniq)):
            s, e = starts[gi], starts[gi + 1]
            rr = rows[s:e]; cc2 = cols[s:e]
            zc0 = int(uniq[gi] // 100000 // 100000)
            yc0 = int(uniq[gi] // 100000 % 100000)
            xc0 = int(uniq[gi] % 100000)
            chunk = _load_chunk(cache_dir, zc0, yc0, xc0)
            if chunk is None:
                continue
            out[rr, cc2] = chunk[zi[rr, cc2] % CHUNK, yi[rr, cc2] % CHUNK, xi[rr, cc2] % CHUNK]
        return out

    if out_png is not None:
        mid = offsets[len(offsets) // 2]
        img = sample_layer(mid)
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        Image.fromarray(img).save(out_png)
        # also write the papyrus mask (valid-vertex footprint) for convenience
        mp = os.path.splitext(out_png)[0] + "_mask.png"
        Image.fromarray((validu.astype(np.uint8) * 255)).save(mp)
        print(f"[png] wrote {out_png}  ({H}x{W}) mid-layer  + mask {mp}", flush=True)

    if out_zarr is not None:
        import zarr
        D = len(offsets)
        store = zarr.open(out_zarr, mode="w", shape=(D, H, W), chunks=(min(8, D), 32, 32),
                          dtype="<u2", compressor=None, zarr_format=2)
        for li, off in enumerate(offsets):
            store[li] = sample_layer(off).astype(np.uint16)
            print(f"[zarr] layer {li+1}/{D}", flush=True)
        if out_id is not None:
            os.makedirs("masks", exist_ok=True)
            Image.fromarray((validu.astype(np.uint8) * 255)).save(f"masks/{out_id}.png")
        print(f"[zarr] wrote {out_zarr}  ({D},{H},{W})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="render PHerc0139 w044 9.362um flattened surface")
    ap.add_argument("--mesh-dir", default=r"C:\Users\ChenJeff\Documents\_ves_tmp\w044_9um_mesh")
    ap.add_argument("--cache-dir", default=r"C:\Users\ChenJeff\Documents\_ves_tmp\p0139_9um_chunks")
    ap.add_argument("--vol-base", default=None,
                    help="raw volume level-0 base URL (…/volume.zarr/0). default = PHerc0139")
    ap.add_argument("--vol-shape", default=None,
                    help="raw volume shape 'z,y,x' (level 0). default = PHerc0139 20974,6621,6621")
    ap.add_argument("--layers", type=int, default=1, help="depth layers (1 = mid slice only)")
    ap.add_argument("--normal-step", type=float, default=1.0, help="voxels between depth layers")
    ap.add_argument("--upsample", type=float, default=20.0, help="flattened px per mesh grid step (1/scale); may be fractional")
    ap.add_argument("--crop-valid", action="store_true", help="crop the mesh grid to the valid bbox before upsampling (needed for sparse padded grids)")
    ap.add_argument("--crop-margin", type=int, default=8, help="grid cells of margin kept around the valid bbox when --crop-valid")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--out-zarr", default=None)
    ap.add_argument("--out-id", default=None)
    args = ap.parse_args()
    global S3_BASE, VOL_SHAPE
    if args.vol_base:
        S3_BASE = args.vol_base.rstrip("/")
    if args.vol_shape:
        VOL_SHAPE = tuple(int(v) for v in args.vol_shape.split(","))
    render(args.mesh_dir, args.cache_dir, args.layers, args.normal_step, args.upsample,
           args.workers, args.out_png, args.out_zarr, args.out_id,
           crop_valid=args.crop_valid, crop_margin=args.crop_margin)


if __name__ == "__main__":
    main()
