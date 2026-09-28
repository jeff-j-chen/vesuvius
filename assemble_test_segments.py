#!/usr/bin/env python3
"""assemble_test_segments.py -- render the 10 competition test-segment zarrs from their tifxyz meshes
(PHerc0813, PHerc0211, PHerc1203, PHerc1447 x2, PHerc0826, PHerc0846A, PHerc0175A, PHerc0306B, PHerc0800). the w055 HOLDOUT is a PHerc0139 segment and is
assembled by assemble_training_segments.py (download path), not here.

the exact tifxyz mesh directory and source volume for each patch are listed in FRAGMENTS below.

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

EXTRAS (--include-extras, off by default)
  downloads atlas.zip from the R2 bucket root (R2_PUBLIC_URL [+ ACCESS_KEY/SECRET_KEY]) into tifxyz/atlas/,
  renders every atlas segment (uint8, zarr id = folder name, mask in masks/extras/), then builds surface
  labels in surface_labels/extras/. those labels are NOT committed: they are pulled from R2 as
  atlas_surface_labels.zip when present, and re-uploaded whenever new ones are generated. extras are not
  part of DEFAULT_SCROLLS, so MAE pretraining never samples them. rendering stops once free disk falls
  below --min-free-gb.

usage:
  python assemble_test_segments.py [--workers N] [--out-dir DIR] [--include-extras]

output:
    ves_zarrs2/<scroll_id>.zarr
    masks/<scroll_id>.png
"""
from __future__ import annotations
import argparse, json, multiprocessing, os, shutil, subprocess, sys, threading, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
RAW_CHUNK = 128    # raw volume chunks are 128^3 uint8

# OUTPUT zarr chunks optimized for arbitrary 192px contexts and 8-slice windows
DEFAULT_CHUNK_DEPTH = 8   # matches 8-slice depth windows in triple mode
DEFAULT_CHUNK_Y = 64      # balances arbitrary 192px context reads and chunk over-read
DEFAULT_CHUNK_X = 64

# output zarr dir: honor $VESUVIUS_ZARR_PATH (same var config/precompute read); default is
# /vesuvius/ves_zarrs2 on linux, the local documents path on windows.
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH",
                     "/vesuvius/ves_zarrs2" if os.name == "posix"
                     else r"C:\Users\ChenJeff\Documents\ves_zarrs2")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.join(SCRIPT_DIR, "masks")

EXTRAS_NAME = "atlas"
EXTRAS_ZIP = "atlas.zip"
EXTRAS_SURFACE_ZIP = "atlas_surface_labels.zip"
EXTRAS_MESH_ROOT = os.path.join(SCRIPT_DIR, "tifxyz", EXTRAS_NAME)
EXTRAS_MASK_DIR = os.path.join(MASK_DIR, "extras")
SURFACE_LABEL_ROOT = os.path.join(SCRIPT_DIR, "surface_labels")
EXTRAS_SURFACE_DIR = os.path.join(SURFACE_LABEL_ROOT, "extras")
# atlas segment name prefix -> (raw volume the mesh coords live in, vol shape z,y,x); bboxes verified to fit
EXTRA_VOLUMES = {
    "PHerc0125": (f"{BUCKET}/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0", "20840,8387,8387"),
    "PHerc0211": (f"{BUCKET}/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/0", "19416,7948,7948"),
    "PHerc0257": (f"{BUCKET}/PHerc0257/volumes/20250821151750-9.362um-1.2m-113keV-masked.zarr/0", "18872,8388,8388"),
    "PHerc0358": (f"{BUCKET}/PHerc0358/volumes/20250821151737-9.362um-1.2m-113keV-masked.zarr/0", "14744,7783,7783"),
    "PHerc0800": (f"{BUCKET}/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0", "24298,9867,9867"),
    "PHerc0813": (f"{BUCKET}/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/0", "16993,7947,7947"),
    "PHerc0826": (f"{BUCKET}/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr/0", "16920,8169,8169"),
}
# extras are stored as uint8 (the raw data is uint8) to halve their ~150 GB uint16 footprint
EXTRAS_DTYPE = "|u1"

# free-disk floor for downloads/renders (set from --min-free-gb); tripping it stops all further rendering
_MIN_FREE_BYTES = 0
_LOW_DISK = threading.Event()


class LowDiskError(RuntimeError):
    pass


def _disk_ok(path):
    if _LOW_DISK.is_set():
        return False
    if _MIN_FREE_BYTES and shutil.disk_usage(path).free < _MIN_FREE_BYTES:
        _LOW_DISK.set()
        return False
    return True


def _require_disk(path):
    if not _disk_ok(path):
        raise LowDiskError(f"free disk at {path} is below {_MIN_FREE_BYTES / 1e9:.0f} GB")


# ---- mesh rendering functions (formerly render_9um_surface.py) ----

def _fetch_raw_chunk(args):
    """download one raw uint8 128^3 chunk to the cache dir. 404 (air) -> sentinel empty file.
    returns (key, status). hardened curl (timeouts+retry) so one hung request can't freeze us."""
    zc, yc, xc, cache_dir, vol_base = args
    out = os.path.join(cache_dir, f"{zc}_{yc}_{xc}.raw")
    if os.path.exists(out):
        return (zc, yc, xc), "cached"
    if not _disk_ok(cache_dir):
        return (zc, yc, xc), "lowdisk"
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
                          crop_valid=True, crop_margin=8, mask_dir=MASK_DIR, output_dtype="<u2"):
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
    _require_disk(cache_dir)
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

    # write zarr
    import zarr
    D = len(offsets)
    store = zarr.open(out_zarr, mode="w", shape=(D, H, W), 
                      chunks=(min(chunk_depth, D), chunk_y, chunk_x),
                      dtype=output_dtype, compressor=None, zarr_format=2)
    for li, off in enumerate(offsets):
        _require_disk(os.path.dirname(os.path.abspath(out_zarr)))
        store[li] = sample_layer(off).astype(store.dtype)
        print(f"[zarr] layer {li+1}/{D}", flush=True)

    # the rendered zarr is the source of truth for its usable footprint
    # derive the mask from the center layer rather than mesh validity alone
    midslice_mask = (np.asarray(store[D // 2]) > 0).astype(np.uint8) * 255
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"{out_id}.png")
    Image.fromarray(midslice_mask).save(mask_path)
    print(f"[mask] wrote {mask_path} from zarr layer {D // 2}  "
                    f"valid_frac={(midslice_mask > 0).mean():.3f}", flush=True)
    print(f"[zarr] wrote {out_zarr}  ({D},{H},{W})", flush=True)


# ---- test fragment definitions ----

# each entry: (out_id, tifxyz mesh subdir, raw-volume base url (.zarr/0), vol shape z,y,x)
# the mesh voxel coords live in the listed volume's space, so vol-base + vol-shape MUST match the
# mesh (verified against each mesh bbox). NOTE: 1447's only volume is 8.640um (not 9.362um) -- that
# IS the volume its mesh was built on (vc3d folder 20250521151220_editable).
FRAGMENTS = [
    ("20260814140748", "auto_grown_20260814140748456_flatboi",
     f"{BUCKET}/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/0",
     "16993,7947,7947"),
    # PHerc0211 large merged segment (replaces 20260717193517520 and 20260719202304218)
    # combines 5 patches into a significantly larger rectangular area
    ("20260717193517", "auto_grown_20260717193517520_0_1_2_3_4_merged_flatboi",
     f"{BUCKET}/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/0",
     "19416,7948,7948"),
    ("20260720090842", "auto_grown_20260720090842117_flatboi",
     f"{BUCKET}/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0",
     "18977,6844,6844"),
    ("20250703034159", "20250703034159_flatboi",
     f"{BUCKET}/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr/0",
     "24297,8343,8343"),
    # PHerc0826 merged patch (2026-08-08)
    ("20260723112922", "auto_grown_20260723112922652_merged_flatboi",
     f"{BUCKET}/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr/0",
     "16920,8169,8169"),
    ("20260921094413", "auto_grown_20260921094413486_flatboi",
     f"{BUCKET}/PHerc0846A/volumes/20250728152254-9.362um-1.2m-113keV-masked.zarr/0",
     "14019,7726,7726"),
    ("20260918132724", "auto_grown_20260918132724424_flatboi",
     f"{BUCKET}/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr/0",
     "12748,9363,9363"),
    ("20260922073234", "auto_grown_20260922073234974_flatboi",
     f"{BUCKET}/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr/0",
     "15898,8849,8849"),
    ("20260922161631", "auto_grown_20260922161631422_merged_flatboi",
     f"{BUCKET}/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0",
     "24298,9867,9867"),
    # second PHerc1447 surface, same volume as 20250703034159
    ("20260925085345", "auto_grown_20260925085345806_abf",
     f"{BUCKET}/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr/0",
     "24297,8343,8343"),
]


def render_fragment(zid, mesh_sub, vol_base, vol_shape, workers, out_dir, script_dir,
                   chunk_depth, chunk_y, chunk_x, force=False, mask_dir=MASK_DIR,
                   output_dtype="<u2", drop_cache=False):
    """render one test fragment. returns (zid, status) for summary."""
    mesh_dir = os.path.join(script_dir, "tifxyz", mesh_sub)
    out_zarr = os.path.join(out_dir, f"{zid}.zarr")
    mask_path = os.path.join(mask_dir, f"{zid}.png")
    cache_dir = os.path.join("_ves_tmp", f"render_{zid}")

    if force:
        shutil.rmtree(out_zarr, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
        print("  --force: cleared zarr + mask; keeping the raw chunk cache")
    
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
        _require_disk(out_dir)
        render_surface_volume(
            mesh_dir=mesh_dir,
            cache_dir=cache_dir,
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
            crop_margin=8,
            mask_dir=mask_dir,
            output_dtype=output_dtype)
        if drop_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        return (zid, "OK")
    except LowDiskError as e:
        print(f"  [STOP] {zid}: {e} -- removing the partial zarr and stopping renders", flush=True)
        shutil.rmtree(out_zarr, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return (zid, f"STOPPED: {e}")
    except Exception as e:
        import traceback
        print(f"  [WARN] render failed for {zid}: {e}")
        traceback.print_exc()
        return (zid, f"FAIL: {e}")


def _safe_extract(zip_path, dest, top):
    """extract a zip whose entries must all live under top/ (no absolute paths or '..')."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            parts = member.replace("\\", "/").split("/")
            if member.startswith(("/", "\\")) or ".." in parts or parts[0] != top:
                raise RuntimeError(f"{zip_path}: unexpected entry {member!r}")
        archive.extractall(dest)


def fetch_extras_meshes(ats, r2_root):
    """download + extract atlas.zip into tifxyz/atlas/ unless it is already there."""
    if os.path.isdir(EXTRAS_MESH_ROOT) and os.listdir(EXTRAS_MESH_ROOT):
        print(f"[extras] meshes present in {EXTRAS_MESH_ROOT}")
        return
    if r2_root is None:
        raise SystemExit(f"[extras] {EXTRAS_MESH_ROOT} is missing and R2 is not configured "
                         "(export R2_PUBLIC_URL [+ ACCESS_KEY/SECRET_KEY])")
    _require_disk(SCRIPT_DIR)
    local = os.path.join("_ves_tmp", EXTRAS_ZIP)
    print(f"[extras] downloading {EXTRAS_ZIP} from R2 ...", flush=True)
    try:
        ats._r2_fetch(f"{r2_root}/{EXTRAS_ZIP}", output=local)
        _safe_extract(local, os.path.dirname(EXTRAS_MESH_ROOT), EXTRAS_NAME)
    finally:
        if os.path.exists(local):
            os.remove(local)
    print(f"[extras] extracted {len(os.listdir(EXTRAS_MESH_ROOT))} meshes -> {EXTRAS_MESH_ROOT}")


def extra_fragments():
    """(zid, mesh subdir, vol base, vol shape) for every atlas mesh; zid = folder name."""
    if not os.path.isdir(EXTRAS_MESH_ROOT):
        return []
    fragments = []
    for name in sorted(os.listdir(EXTRAS_MESH_ROOT)):
        if not os.path.isfile(os.path.join(EXTRAS_MESH_ROOT, name, "x.tif")):
            continue
        volume = EXTRA_VOLUMES.get(name.split("_")[0])
        if volume is None:
            print(f"[extras] WARN no raw volume mapped for {name} -- skipping")
            continue
        fragments.append((name, f"{EXTRAS_NAME}/{name}", *volume))
    return fragments


def _has_extra_surface(zid):
    # metadata.json is written after depth/confidence are complete
    folder = os.path.join(EXTRAS_SURFACE_DIR, zid)
    return all(os.path.isfile(os.path.join(folder, name))
               for name in ("depth.npy", "confidence.npy", "metadata.json"))


def fetch_extras_surface(ats, r2_root, zids):
    """download + extract atlas_surface_labels.zip from R2 when any selected extra lacks labels."""
    missing = [zid for zid in zids if not _has_extra_surface(zid)]
    if not missing:
        print(f"[extras] surface labels present for all {len(zids)} selected extras")
        return
    if r2_root is None:
        print(f"[extras] R2 not configured -- {len(missing)} extra(s) lack surface labels and will be generated")
        return
    _require_disk(SCRIPT_DIR)
    local = os.path.join("_ves_tmp", EXTRAS_SURFACE_ZIP)
    print(f"[extras] {len(missing)} extra(s) lack surface labels -> downloading {EXTRAS_SURFACE_ZIP} from R2 ...",
          flush=True)
    try:
        ats._r2_fetch(f"{r2_root}/{EXTRAS_SURFACE_ZIP}", output=local)
    except RuntimeError as exc:
        print(f"[extras] {EXTRAS_SURFACE_ZIP} not fetched from R2 ({exc}); missing labels will be generated")
    else:
        _safe_extract(local, SURFACE_LABEL_ROOT, "extras")
        still = sum(not _has_extra_surface(zid) for zid in zids)
        print(f"[extras] extracted {EXTRAS_SURFACE_ZIP} -> {EXTRAS_SURFACE_DIR}; {still} still missing")
    finally:
        if os.path.exists(local):
            os.remove(local)


def sync_extras_surface(ats, r2_root, zids, zarr_dir):
    """generate surface labels for assembled extras still lacking them, and re-upload if new ones were made."""
    ready = [zid for zid in zids
             if os.path.isdir(os.path.join(zarr_dir, f"{zid}.zarr"))
             and os.path.isfile(os.path.join(EXTRAS_MASK_DIR, f"{zid}.png"))]
    missing = [zid for zid in ready if not _has_extra_surface(zid)]
    print(f"[extras] surface labels: {len(ready) - len(missing)}/{len(ready)} present, generating {len(missing)}")
    generated = 0
    for index, zid in enumerate(missing, 1):
        print(f"\n[extras] surface {index}/{len(missing)} {zid}", flush=True)
        result = subprocess.run(
            [
                sys.executable, os.path.join(SCRIPT_DIR, "generate_surface_supervision.py"),
                "--scroll-id", zid,
                "--z-start", "4", "--z-end", "28",
                "--zarr-dir", zarr_dir,
                "--mask-dir", EXTRAS_MASK_DIR,
                "--output-dir", EXTRAS_SURFACE_DIR,
                "--review-dir", os.path.join(SCRIPT_DIR, "output", "surface_review", "extras"),
            ],
            cwd=SCRIPT_DIR,
        )
        if result.returncode == 0 and _has_extra_surface(zid):
            generated += 1
        else:
            print(f"[extras] WARN surface generation failed for {zid} (exit {result.returncode})")
    if generated and r2_root is not None:
        upload_extras_surface(ats, r2_root)
    elif generated:
        print(f"[extras] WARN R2 not configured -- {generated} new surface label(s) were not uploaded")


def upload_extras_surface(ats, r2_root):
    """zip every complete surface_labels/extras/<id>/ and PUT it to R2 as atlas_surface_labels.zip."""
    local = os.path.join("_ves_tmp", EXTRAS_SURFACE_ZIP)
    zids = sorted(name for name in os.listdir(EXTRAS_SURFACE_DIR) if _has_extra_surface(name))
    with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for zid in zids:
            for name in ("depth.npy", "confidence.npy", "metadata.json"):
                archive.write(os.path.join(EXTRAS_SURFACE_DIR, zid, name), f"extras/{zid}/{name}")
    size_mb = os.path.getsize(local) / 1e6
    print(f"[extras] uploading {EXTRAS_SURFACE_ZIP} ({len(zids)} segments, {size_mb:.0f} MB) to R2 ...", flush=True)
    try:
        ats._r2_put(f"{r2_root}/{EXTRAS_SURFACE_ZIP}", local)
        print(f"[extras] uploaded {EXTRAS_SURFACE_ZIP}")
    except RuntimeError as exc:
        print(f"[extras] WARN upload failed: {exc}")
    finally:
        os.remove(local)


def main():
    ap = argparse.ArgumentParser(description="assemble test segment zarrs from tifxyz meshes")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="assemble one output ID; repeat to select multiple fragments",
    )
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel S3 chunk-download workers PER fragment (default 32 for EPYC 7702)")
    ap.add_argument("--out-dir", type=str, default=ZARR_DIR,
                    help=f"output zarr directory (default: {ZARR_DIR})")
    ap.add_argument("--force", action="store_true",
                    help="re-render selected fragments while keeping downloaded raw chunks")
    ap.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH,
                    help=f"zarr depth chunk size (default {DEFAULT_CHUNK_DEPTH}, optimized for 8-slice windows)")
    ap.add_argument("--chunk-y", type=int, default=DEFAULT_CHUNK_Y,
                    help=f"zarr Y chunk size (default {DEFAULT_CHUNK_Y})")
    ap.add_argument("--chunk-x", type=int, default=DEFAULT_CHUNK_X,
                    help=f"zarr X chunk size (default {DEFAULT_CHUNK_X})")
    ap.add_argument("--include-extras", action="store_true",
                    help="also fetch atlas.zip from R2, render every atlas segment, and sync their surface labels")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="stop downloading/rendering once free disk drops below this (default 10)")
    args = ap.parse_args()
    global _MIN_FREE_BYTES
    _MIN_FREE_BYTES = int(args.min_free_gb * 1e9)
    os.makedirs("_ves_tmp", exist_ok=True)

    ats = r2_root = None
    extras = []
    if args.include_extras:
        sys.path.insert(0, SCRIPT_DIR)
        import assemble_training_segments as ats
        try:
            r2_root = ats.configure_r2()
        except ValueError as exc:
            print(f"[extras] R2 unavailable ({exc}); using local extras only")
        fetch_extras_meshes(ats, r2_root)
        extras = extra_fragments()

    selected_ids = {str(value) for value in args.only}
    fragments = [
        fragment for fragment in FRAGMENTS
        if not selected_ids or fragment[0] in selected_ids
    ]
    extras = [fragment for fragment in extras if not selected_ids or fragment[0] in selected_ids]
    missing_ids = selected_ids - {fragment[0] for fragment in fragments + extras}
    if missing_ids:
        ap.error(f"unknown --only IDs: {sorted(missing_ids)}")
    if extras:
        fetch_extras_surface(ats, r2_root, [fragment[0] for fragment in extras])
    
    script_dir = SCRIPT_DIR
    
    # ensure output directories exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    
    print(f"[assemble] python={sys.executable}  out_dir={args.out_dir}  workers={args.workers}")
    print(f"[assemble] {len(fragments)} test fragment(s) + {len(extras)} extra(s)  "
          f"chunks=({args.chunk_depth},{args.chunk_y},{args.chunk_x})  min_free={args.min_free_gb:g} GB")
    
    results = []
    for i, (zid, mesh_sub, vol_base, vol_shape) in enumerate(fragments, 1):
        if _LOW_DISK.is_set():
            break
        print(f"\n{'='*70}\n=== {i}/{len(fragments)}  {zid}  (mesh {mesh_sub}) ===\n{'='*70}", flush=True)
        result = render_fragment(zid, mesh_sub, vol_base, vol_shape, 
                                args.workers, args.out_dir, script_dir,
                                args.chunk_depth, args.chunk_y, args.chunk_x,
                                force=args.force)
        results.append(result)

    for i, (zid, mesh_sub, vol_base, vol_shape) in enumerate(extras, 1):
        if _LOW_DISK.is_set():
            break
        print(f"\n{'='*70}\n=== extra {i}/{len(extras)}  {zid} ===\n{'='*70}", flush=True)
        results.append(render_fragment(
            zid, mesh_sub, vol_base, vol_shape, args.workers, args.out_dir, script_dir,
            args.chunk_depth, args.chunk_y, args.chunk_x, force=args.force,
            mask_dir=EXTRAS_MASK_DIR, output_dtype=EXTRAS_DTYPE, drop_cache=True,
        ))
    if _LOW_DISK.is_set():
        print(f"\n[assemble] free disk fell below {args.min_free_gb:g} GB -- remaining renders skipped")
    if extras:
        sync_extras_surface(ats, r2_root, [fragment[0] for fragment in extras], args.out_dir)
    
    print(f"\n{'='*70}\n[assemble] SUMMARY\n{'='*70}")
    for zid, status in results:
        print(f"  {zid}: {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
