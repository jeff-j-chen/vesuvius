#!/usr/bin/env python3
"""assemble_test_segments.py -- render the 5 competition test-segment zarrs from their tifxyz meshes
(PHerc0813, PHerc0211 x2, PHerc1203, PHerc1447). the w055 HOLDOUT is a PHerc0139 segment and is
assembled by assemble_training_segments.py (download path), not here.

the tifxyz mesh is stored in:
  tifxyz/auto_grown_20260716083545968/   <- root level = latest state (max_gen=179)
  tifxyz/auto_grown_20260716083545968/N/ <- numbered subdirs = historical autosaves

HOW IT WORKS
The tifxyz does NOT contain intensity values -- it contains the SURFACE COORDINATES.
For each (u,v) cell in the flattened 2D papyrus grid, x.tif/y.tif/z.tif store the
3D (x,y,z) voxel coordinate of that surface point in the raw CT scan. Intensity values
live in the raw CT volume on S3. The render script:
  1. reads x.tif/y.tif/z.tif -> knows WHERE on the raw CT each surface pixel came from
  2. fetches those voxels (+ depth neighbors) from the S3 raw volume
  3. writes them as a local zarr with shape (layers, H, W)
This is why the tifxyz is tiny (~0.6 MB per file) even for a large segment.

requires: python + project venv, curl, ~2 GB disk

usage:
  python assemble_test_segments.py [--workers N] [--out-dir DIR]

output:
  ves_zarrs2/20260716083545.zarr   (28, 4421, 4421) uint16
  masks/20260716083545.png
"""
from __future__ import annotations
import argparse, multiprocessing, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
RAW_CHUNK = 128    # raw volume chunks are 128^3 uint8

# OUTPUT zarr chunk sizes (default optimized for 48x48 context, 16x16 tiles, 8-slice windows)
DEFAULT_CHUNK_DEPTH = 8   # matches 8-slice depth windows in triple mode
DEFAULT_CHUNK_Y = 32      # 2x tile_size, reasonable cache granularity
DEFAULT_CHUNK_X = 32      # 2x tile_size, reasonable cache granularity

# output zarr dir: honor $VESUVIUS_ZARR_PATH (same var config/precompute read); default is
# /vesuvius/ves_zarrs2 on linux, the local documents path on windows.
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH",
                     "/vesuvius/ves_zarrs2" if os.name == "posix"
                     else r"C:\Users\ChenJeff\Documents\ves_zarrs2")


# ---- mesh rendering functions (formerly render_9um_surface.py) ----

def _fetch_raw_chunk(args):
    """download one raw uint8 128^3 chunk to the cache dir. 404 (air) -> sentinel empty file.
    returns (key, status). hardened curl (timeouts+retry) so one hung request can't freeze us."""
    zc, yc, xc, cache_dir, vol_base = args
    out = os.path.join(cache_dir, f"{zc}_{yc}_{xc}.raw")
    if os.path.exists(out):
        return (zc, yc, xc), "cached"
    url = f"{vol_base}/{zc}/{yc}/{xc}"
    # NO --retry-all-errors: a 404 here is an EXPECTED air chunk; retrying it burned ~4x2s each.
    # --retry still covers transient 5xx/timeouts.
    r = subprocess.run(
        ["curl", "-s", "--fail", "--connect-timeout", "20", "--max-time", "120",
         "--retry", "3", "--retry-delay", "1", url, "-o", out],
        capture_output=True)
    if r.returncode != 0:
        # 404 / missing = all-air chunk -> write a zero-length sentinel so we skip re-fetch
        open(out, "wb").close()
        return (zc, yc, xc), "air"
    return (zc, yc, xc), "ok"


def _load_raw_chunk(cache_dir, zc, yc, xc):
    """read a cached chunk as (128,128,128) uint8; air/empty sentinel -> None."""
    p = os.path.join(cache_dir, f"{zc}_{yc}_{xc}.raw")
    chunk_bytes = RAW_CHUNK * RAW_CHUNK * RAW_CHUNK
    try:
        if os.path.getsize(p) == chunk_bytes:
            return np.frombuffer(open(p, "rb").read(), dtype=np.uint8).reshape(RAW_CHUNK, RAW_CHUNK, RAW_CHUNK)
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


def render_surface_volume(mesh_dir, cache_dir, vol_base, vol_shape, layers, normal_step,
                          upsample_factor, workers, out_zarr, out_id, chunk_depth, chunk_y, chunk_x,
                          crop_valid=True, crop_margin=8):
    """render a flattened surface volume from tifxyz mesh + raw volume on S3.
    
    the mesh gives, for each point on the FLATTENED sheet, its (x,y,z) voxel in the raw scan.
    we:
      1. load the tifxyz mesh (x/y/z float32 grids, -1 = invalid)
      2. upsample the coordinate grid to full flattened resolution (1 px per voxel-step)
      3. NEAREST-sample the raw volume at those voxels (NO trilinear interpolation)
    for a multi-layer surface volume we offset along the local surface NORMAL by +/- steps.
    
    the raw volume is uint8, chunks 128^3, NO compressor, dimension_separator '/'. we curl
    the chunks the surface passes through (threaded), cache them on disk (reusable across
    layers + reruns), then sample. air chunks that don't exist on S3 return 404 -> treated
    as fill_value 0."""
    os.makedirs(cache_dir, exist_ok=True)
    X, Y, Z, valid = load_mesh(mesh_dir)
    
    # crop to the valid bounding box (+margin) FIRST. some meshes store a small compact
    # sheet inside a huge mostly-empty padded grid (e.g. a 344x455 blob in a 6203x6203 grid);
    # upsampling the full grid would make a ~124k x 124k canvas. cropping renders only the
    # real surface. harmless for meshes whose valid region already fills the grid.
    if crop_valid and valid.any():
        ys, xs = np.where(valid)
        y0 = max(0, int(ys.min()) - crop_margin)
        y1 = min(valid.shape[0], int(ys.max()) + 1 + crop_margin)
        x0 = max(0, int(xs.min()) - crop_margin)
        x1 = min(valid.shape[1], int(xs.max()) + 1 + crop_margin)
        print(f"[mesh] crop valid bbox grid[{y0}:{y1}, {x0}:{x1}] from {X.shape}", flush=True)
        X = X[y0:y1, x0:x1]
        Y = Y[y0:y1, x0:x1]
        Z = Z[y0:y1, x0:x1]
        valid = valid[y0:y1, x0:x1]
    
    gh, gw = X.shape
    H = int(round((gh - 1) * upsample_factor)) + 1
    W = int(round((gw - 1) * upsample_factor)) + 1
    print(f"[mesh] grid {gh}x{gw} -> flattened {H}x{W}  valid={valid.mean():.3f}", flush=True)

    Xu = upsample(X, W, H)
    Yu = upsample(Y, W, H)
    Zu = upsample(Z, W, H)
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
    GYC = vol_shape[1] // RAW_CHUNK + 2   # y-chunk stride for encoding
    GXC = vol_shape[2] // RAW_CHUNK + 2   # x-chunk stride
    vm = validu
    codes_all = []
    for off in offsets:
        if normals is not None:
            xs = Xu[vm] + normals[..., 0][vm] * off
            ys = Yu[vm] + normals[..., 1][vm] * off
            zs = Zu[vm] + normals[..., 2][vm] * off
        else:
            xs, ys, zs = Xu[vm], Yu[vm], Zu[vm]
        zc = (np.clip(np.rint(zs), 0, vol_shape[0] - 1).astype(np.int64)) // RAW_CHUNK
        yc = (np.clip(np.rint(ys), 0, vol_shape[1] - 1).astype(np.int64)) // RAW_CHUNK
        xc = (np.clip(np.rint(xs), 0, vol_shape[2] - 1).astype(np.int64)) // RAW_CHUNK
        codes_all.append((zc * GYC + yc) * GXC + xc)
    codes = np.unique(np.concatenate(codes_all))
    zc = codes // (GYC * GXC)
    rem = codes % (GYC * GXC)
    yc = rem // GXC
    xc = rem % GXC
    need = set(zip(zc.tolist(), yc.tolist(), xc.tolist()))
    print(f"[chunks] {len(need)} unique chunks to ensure (~{len(need)*2/1024:.1f} GB max)", flush=True)

    jobs = [(zc, yc, xc, cache_dir, vol_base) for (zc, yc, xc) in sorted(need)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _key, _st in pool.map(_fetch_raw_chunk, jobs):
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
        zi = np.clip(np.rint(zs), 0, vol_shape[0] - 1).astype(np.int32)
        yi = np.clip(np.rint(ys), 0, vol_shape[1] - 1).astype(np.int32)
        xi = np.clip(np.rint(xs), 0, vol_shape[2] - 1).astype(np.int32)
        out = np.zeros((H, W), dtype=np.uint8)
        zc = zi // RAW_CHUNK
        yc = yi // RAW_CHUNK
        xc = xi // RAW_CHUNK
        # group pixels by chunk to sample each cached chunk once
        cid = (zc.astype(np.int64) * 100000 + yc) * 100000 + xc
        flat_cid = cid[validu]
        ys_i = np.where(validu)
        order = np.argsort(flat_cid, kind="stable")
        rows = ys_i[0][order]
        cols = ys_i[1][order]
        scid = flat_cid[order]
        uniq, starts = np.unique(scid, return_index=True)
        starts = list(starts) + [len(scid)]
        for gi in range(len(uniq)):
            s, e = starts[gi], starts[gi + 1]
            rr = rows[s:e]
            cc2 = cols[s:e]
            zc0 = int(uniq[gi] // 100000 // 100000)
            yc0 = int(uniq[gi] // 100000 % 100000)
            xc0 = int(uniq[gi] % 100000)
            chunk = _load_raw_chunk(cache_dir, zc0, yc0, xc0)
            if chunk is None:
                continue
            out[rr, cc2] = chunk[zi[rr, cc2] % RAW_CHUNK,
                                 yi[rr, cc2] % RAW_CHUNK,
                                 xi[rr, cc2] % RAW_CHUNK]
        return out

    # write zarr + mask
    import zarr
    D = len(offsets)
    store = zarr.open(out_zarr, mode="w", shape=(D, H, W), 
                      chunks=(min(chunk_depth, D), chunk_y, chunk_x),
                      dtype="<u2", compressor=None, zarr_format=2)
    for li, off in enumerate(offsets):
        store[li] = sample_layer(off).astype(np.uint16)
        print(f"[zarr] layer {li+1}/{D}", flush=True)
    
    os.makedirs("masks", exist_ok=True)
    Image.fromarray((validu.astype(np.uint8) * 255)).save(f"masks/{out_id}.png")
    print(f"[zarr] wrote {out_zarr}  ({D},{H},{W})", flush=True)


# ---- test fragment definitions ----

# each entry: (out_id, tifxyz mesh subdir, raw-volume base url (.zarr/0), vol shape z,y,x)
# the mesh voxel coords live in the listed volume's space, so vol-base + vol-shape MUST match the
# mesh (verified against each mesh bbox). NOTE: 1447's only volume is 8.640um (not 9.362um) -- that
# IS the volume its mesh was built on (vc3d folder 20250521151220_editable).
FRAGMENTS = [
    ("20260716083545", "auto_grown_20260716083545968", 
     f"{BUCKET}/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/0",
     "16993,7947,7947"),
    # PHerc0211 large merged segment (replaces 20260717193517520 and 20260719202304218)
    # combines 5 patches into a significantly larger rectangular area
    ("20260717193517", "auto_grown_20260717193517520_0_1_2_3_4_merged",
     f"{BUCKET}/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/0",
     "19416,7948,7948"),
    ("20260720090842", "auto_grown_20260720090842117",
     f"{BUCKET}/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0",
     "18977,6844,6844"),
    ("20250703034159", "20250703034159",
     f"{BUCKET}/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr/0",
     "24297,8343,8343"),
]


def render_fragment(zid, mesh_sub, vol_base, vol_shape, workers, out_dir, script_dir,
                   chunk_depth, chunk_y, chunk_x):
    """render one test fragment. returns (zid, status) for summary."""
    mesh_dir = os.path.join(script_dir, "tifxyz", mesh_sub)
    out_zarr = os.path.join(out_dir, f"{zid}.zarr")
    mask_path = f"masks/{zid}.png"
    
    # idempotent: skip if this zarr + mask already exist
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print(f"  zarr + mask exist -> skip")
        return (zid, "OK (cached)")
    
    if not os.path.isdir(mesh_dir):
        print(f"  [WARN] mesh dir missing: {mesh_dir} -- skipping")
        return (zid, f"SKIP (no mesh)")
    
    # parse vol_shape string "z,y,x" -> tuple
    vol_shape_tuple = tuple(int(v) for v in vol_shape.split(","))
    
    print(f"  raw vol: {vol_base}  shape={vol_shape}")
    print(f"  (renders on-demand from S3 -- can take 10-30 min per fragment depending on size/speed)")
    
    try:
        render_surface_volume(
            mesh_dir=mesh_dir,
            cache_dir=os.path.join("_ves_tmp", f"render_{zid}"),
            vol_base=vol_base,
            vol_shape=vol_shape_tuple,
            layers=28,
            normal_step=1.0,
            upsample_factor=20.0,
            workers=workers,
            out_zarr=out_zarr,
            out_id=zid,
            chunk_depth=chunk_depth,
            chunk_y=chunk_y,
            chunk_x=chunk_x,
            crop_valid=True,
            crop_margin=8)
        return (zid, "OK")
    except Exception as e:
        import traceback
        print(f"  [WARN] render failed for {zid}: {e}")
        traceback.print_exc()
        return (zid, f"FAIL: {e}")


def main():
    ap = argparse.ArgumentParser(description="assemble test segment zarrs from tifxyz meshes")
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel S3 chunk-download workers PER fragment (default 32 for EPYC 7702)")
    ap.add_argument("--out-dir", type=str, default=ZARR_DIR,
                    help=f"output zarr directory (default: {ZARR_DIR})")
    ap.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH,
                    help=f"zarr depth chunk size (default {DEFAULT_CHUNK_DEPTH}, optimized for 8-slice windows)")
    ap.add_argument("--chunk-y", type=int, default=DEFAULT_CHUNK_Y,
                    help=f"zarr Y chunk size (default {DEFAULT_CHUNK_Y})")
    ap.add_argument("--chunk-x", type=int, default=DEFAULT_CHUNK_X,
                    help=f"zarr X chunk size (default {DEFAULT_CHUNK_X})")
    args = ap.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ensure output directories exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("_ves_tmp", exist_ok=True)
    
    print(f"[assemble] python={sys.executable}  out_dir={args.out_dir}  workers={args.workers}")
    print(f"[assemble] {len(FRAGMENTS)} test fragment(s)  "
          f"chunks=({args.chunk_depth},{args.chunk_y},{args.chunk_x})")
    
    results = []
    for i, (zid, mesh_sub, vol_base, vol_shape) in enumerate(FRAGMENTS, 1):
        print(f"\n{'='*70}\n=== {i}/{len(FRAGMENTS)}  {zid}  (mesh {mesh_sub}) ===\n{'='*70}", flush=True)
        result = render_fragment(zid, mesh_sub, vol_base, vol_shape, 
                                args.workers, args.out_dir, script_dir,
                                args.chunk_depth, args.chunk_y, args.chunk_x)
        results.append(result)
    
    print(f"\n{'='*70}\n[assemble] SUMMARY\n{'='*70}")
    for zid, status in results:
        print(f"  {zid}: {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
