"""assemble_w013_1667.py -- download and assemble PHerc1667 w013 as an isotropic zarr.

SOURCE
  pre-rendered OME-Zarr surface volume at 2.399um / 78keV (level 2 = 4x downsampled):
  shape (109, 10400, 19900), chunk (109, 128, 128), dot-separator, blosc-compressed

ISOTROPIC MATH
  XY: level 2 is already 2.399*4 = 9.596um/pixel (2.5% coarser than 9.362um, acceptable)
  Z:  109 input layers * 2.399um = 261.5um total depth
      mean-pool to 28 output layers: 261.5/28 = 9.34um/layer
      -> all three axes at ~9.4-9.6um, isotropic to within 2.5%

CROP
  right 75% of the segment is blank scroll interior; we keep only the left 25% (W=4975)
  by downloading only the left ceil(4975/128)=39 XY column chunks (out of 156 total)

INKLABELS
  the 2.4um ink TIF (41600x79600) is resized to (10400, 19900) then cropped to (10400, 4975)
  to match the zarr exactly. eroded_inklabels/ is NOT touched -- user manages that folder.

OUTPUT zarr shape: (28, 10400, 4975)  ~9.5um isotropic

usage:
  python assemble_w013_1667.py              # full run
  python assemble_w013_1667.py --skip-norm  # skip norm precompute
  python assemble_w013_1667.py --dry-run    # show params only
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# ---- constants -----------------------------------------------------------

BUCKET   = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SEG_BASE = f"{BUCKET}/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi"

# pre-rendered surface volume at 2.399um, level 2 = 4x XY downsample = 9.596um XY
SURF_BASE  = f"{SEG_BASE}/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"
SURF_LEVEL = 2
SURF_SHAPE = (109, 10400, 19900)   # (D, H, W) at level 2

INK_URL = (
    f"{SEG_BASE}/ink-detection/"
    "PHerc1667-20240304141531-2.399um-0.22m-78keV-volume-20251217075048-"
    "20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
)

ZID        = "20240304141531"
LAYERS_IN  = 109      # depth layers in the level-2 source
LAYERS_OUT = 28       # target depth layers (matches all other training zarrs)
LEFT_FRAC  = 0.25     # keep left 25% of W (right side is blank)
CHUNK_XY   = 128      # source chunk size in XY

def _default_zarr_dir():
    if os.name != "posix":
        return r"C:\Users\ChenJeff\Documents\ves_zarrs2"
    if os.path.exists("/media/jeff/Seagate/"):
        return "/media/jeff/Seagate/ves_zarrs2"
    return "/vesuvius/ves_zarrs2"
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH", _default_zarr_dir())
TMP = "_ves_tmp"

# output zarr: W_crop = left 25% of 19900 = 4975
H_OUT  = SURF_SHAPE[1]                         # 10400
W_FULL = SURF_SHAPE[2]                         # 19900
W_OUT  = int(W_FULL * LEFT_FRAC)               # 4975

CHUNK_DEPTH = 8
CHUNK_Y     = 32
CHUNK_X     = 32


# ---- download helpers ----------------------------------------------------

def _curl_code(url, out, tries=3):
    """download url -> out, return http status code. 200=ok, 404=air, else=fail."""
    for _ in range(tries):
        r = subprocess.run(
            ["curl.exe", "-s", "--connect-timeout", "20", "--max-time", "120",
             "-o", out, "-w", "%{http_code}", url], capture_output=True)
        code = (r.stdout.decode("utf-8", "ignore").strip() or "000")[-3:]
        if code == "200":
            return code
        if os.path.exists(out):
            try: os.remove(out)
            except OSError: pass
        if code == "404":
            return code
        time.sleep(1)
    return code


def _fetch_chunk(args):
    """download one surface-volume chunk. uses dot-separator format: level/0.yc.xc"""
    level, yc, xc, cache_dir = args
    out = os.path.join(cache_dir, f"{yc}.{xc}.raw")
    if os.path.exists(out):
        return (yc, xc, "cached")
    url = f"{SURF_BASE}/{level}/0.{yc}.{xc}"
    code = _curl_code(url, out)
    if code == "200":
        return (yc, xc, "ok")
    if code == "404":
        open(out, "wb").close()   # air sentinel
        return (yc, xc, "air")
    return (yc, xc, "fail")


def _load_surface_chunk(cache_dir, yc, xc, D):
    """load cached chunk as (D, CHUNK_XY, CHUNK_XY) uint8, or None if air/missing."""
    p = os.path.join(cache_dir, f"{yc}.{xc}.raw")
    try:
        if os.path.getsize(p) == D * CHUNK_XY * CHUNK_XY:
            return np.frombuffer(open(p, "rb").read(), dtype=np.uint8).reshape(D, CHUNK_XY, CHUNK_XY)
    except Exception:
        pass
    return None


# ---- z mean-pooling ------------------------------------------------------

def _pool_z(vol_dhw, layers_out):
    """area-average a (D,H,W) uint8 array in depth to layers_out. returns float32."""
    D = vol_dhw.shape[0]
    out = np.zeros((layers_out, vol_dhw.shape[1], vol_dhw.shape[2]), dtype=np.float32)
    for i in range(layers_out):
        s = int(round(i * D / layers_out))
        e = int(round((i + 1) * D / layers_out))
        out[i] = vol_dhw[s:e].mean(axis=0)
    return out


# ---- pipeline steps -------------------------------------------------------

def step1_download_zarr(workers, dry_run):
    """stream level-2 surface zarr directly from S3 via HTTP, z-pool 109->28, write output zarr.
    the source chunks are blosc-compressed so we let zarr handle decompression rather than
    caching raw bytes. reads one 128px-tall XY row at a time to keep RAM bounded."""
    import zarr, fsspec

    out_zarr  = os.path.join(ZARR_DIR, f"{ZID}.zarr")
    mask_path = f"masks/{ZID}.png"
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print("  zarr + mask already exist -> skip")
        return

    D, H, W = SURF_SHAPE
    n_rows = (H_OUT + CHUNK_XY - 1) // CHUNK_XY    # 82 row-chunks of 128px each

    print(f"  source: {SURF_BASE} level {SURF_LEVEL}  ({D},{H},{W})")
    print(f"  z-pool {D}->{LAYERS_OUT}  crop W to {W_OUT}  ->  output ({LAYERS_OUT},{H_OUT},{W_OUT})")
    print(f"  reading {n_rows} row-slices of 128px from remote zarr")

    if dry_run:
        print("  [dry] skipping")
        return

    src_url = f"{SURF_BASE}/{SURF_LEVEL}"
    src = zarr.open(src_url, mode="r")

    os.makedirs(ZARR_DIR, exist_ok=True)
    store = zarr.open(out_zarr, mode="w",
                      shape=(LAYERS_OUT, H_OUT, W_OUT),
                      chunks=(CHUNK_DEPTH, CHUNK_Y, CHUNK_X),
                      dtype="<u2", compressor=None, zarr_format=2)
    mask_buf = np.zeros((H_OUT, W_OUT), dtype=np.uint8)

    for ri in range(n_rows):
        y0 = ri * CHUNK_XY
        y1 = min(y0 + CHUNK_XY, H_OUT)
        # read all 109 depth layers for this row-strip, left W_OUT columns only
        row_vol = src[:, y0:y1, :W_OUT].astype(np.float32)   # (109, H_row, W_out)
        row_pooled = _pool_z(row_vol, LAYERS_OUT)              # (28, H_row, W_out)
        store[:, y0:y1, :] = np.clip(row_pooled, 0, 255).astype(np.uint16)
        mask_buf[y0:y1, :] = (row_pooled[LAYERS_OUT // 2] > 0).astype(np.uint8) * 255
        if (ri + 1) % 10 == 0 or ri == n_rows - 1:
            print(f"  rows: {ri+1}/{n_rows}", flush=True)

    os.makedirs("masks", exist_ok=True)
    Image.fromarray(mask_buf).save(mask_path)
    print(f"  zarr written: ({LAYERS_OUT},{H_OUT},{W_OUT})")
    print(f"  mask: {mask_path}  valid_frac={(mask_buf>0).mean():.3f}")


def step2_inklabels(dry_run):
    """download 2.4um ink TIF if needed, resize to (H_OUT, W_OUT), save inklabels/."""
    import cv2

    ink_out = f"inklabels/{ZID}.png"
    if os.path.exists(ink_out):
        print(f"  inklabels exist -> skip")
        return

    zarr_path = os.path.join(ZARR_DIR, f"{ZID}.zarr")
    if not os.path.isdir(zarr_path):
        print(f"  zarr missing, skipping inklabels step")
        return

    ink_tif = os.path.join(TMP, f"{ZID}_ink_2p4.tif")
    if not os.path.exists(ink_tif):
        if dry_run:
            print(f"  [dry] would download ink TIF (~256 MB)")
            return
        print(f"  downloading ink TIF (~256 MB) ...")
        r = subprocess.run(["curl.exe", "--fail", "--max-time", "300",
                            "-o", ink_tif, INK_URL])
        if r.returncode != 0:
            raise RuntimeError("ink TIF download failed")

    print(f"  loading ink TIF ...")
    ink = np.array(Image.open(ink_tif))   # (41600, 79600) uint8
    print(f"  ink TIF shape: {ink.shape}")

    # level-2 zarr is 4x XY downsample: ink (41600,79600) -> zarr (10400,19900)
    # resize ink preserving aspect ratio, then crop to W_OUT
    ink_h, ink_w = ink.shape[:2]
    W_full_resize = int(round(ink_w * H_OUT / ink_h))
    print(f"  resizing ({ink_h},{ink_w}) -> ({H_OUT},{W_full_resize}) -> crop to ({H_OUT},{W_OUT})")
    ink_r = cv2.resize(ink.astype(np.float32), (W_full_resize, H_OUT),
                       interpolation=cv2.INTER_AREA)
    ink_crop = ink_r[:, :W_OUT]

    mask_path = f"masks/{ZID}.png"
    mask = np.array(Image.open(mask_path).convert("L")) > 0 if os.path.exists(mask_path) else None

    # eroded_inklabels/ is user-managed -- only write to inklabels/ (both top-level and 2_4um subdir)
    os.makedirs("inklabels", exist_ok=True)
    os.makedirs("inklabels/2_4um", exist_ok=True)
    ink_u8 = np.clip(ink_crop, 0, 255).astype(np.uint8)
    if mask is not None:
        ink_u8 = (ink_u8 * mask).astype(np.uint8)
    Image.fromarray(ink_u8).save(ink_out)
    Image.fromarray(ink_u8).save(f"inklabels/2_4um/{ZID}.png")
    print(f"  wrote {ink_out}  ink_frac={(ink_u8>0).mean():.4f}")
    print(f"  wrote inklabels/2_4um/{ZID}.png")


def step3_norm(skip):
    zarr_path = os.path.join(ZARR_DIR, f"{ZID}.zarr")
    if skip or not os.path.isdir(zarr_path):
        print("  --skip-norm or zarr missing -> skip")
        return
    if os.path.exists("norm_cache.json"):
        try:
            if ZID in json.load(open("norm_cache.json")):
                print("  norm cached -> skip")
                return
        except Exception:
            pass
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.norm import compute_norm
    compute_norm(ZID, ZARR_DIR)


def step4_overlay(dry_run):
    """write midslice + inklabel overlay to _ves_tmp/{ZID}_overlay.png."""
    import zarr, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zarr_path = os.path.join(ZARR_DIR, f"{ZID}.zarr")
    ink_path  = f"inklabels/{ZID}.png"
    if not os.path.isdir(zarr_path):
        print("  zarr missing, skipping overlay")
        return

    z = zarr.open(zarr_path, mode="r")
    D, H, W = z.shape
    mid = D // 2
    print(f"  loading midslice {mid} from ({D},{H},{W}) ...")
    layer = z[mid].astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"w013 PHerc1667  ZID={ZID}  ~9.6um isotropic  layer {mid}/{D}")

    axes[0].imshow(layer, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"zarr midslice (layer {mid})")
    axes[0].axis("off")

    if os.path.exists(ink_path):
        ink = np.array(Image.open(ink_path).convert("L")).astype(np.float32)
        rgb = np.stack([layer / 255.0] * 3, axis=-1)
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + ink / 255.0 * 0.5, 0, 1)
        axes[1].imshow(rgb)
        axes[1].set_title(f"inklabels overlay  ink_frac={(ink>0).mean():.3f}")
        axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "inklabels missing", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    out_png = os.path.join(TMP, f"{ZID}_overlay.png")
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  overlay -> {out_png}")
    try:
        os.startfile(out_png)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-norm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overlay-only", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print(f"w013 PHerc1667  ZID={ZID}")
    print(f"source: level-2 pre-rendered surface zarr  ({LAYERS_IN},{H_OUT},{W_FULL})")
    print(f"z-pool: {LAYERS_IN} layers @ 2.399um -> {LAYERS_OUT} layers @ ~9.34um/layer")
    print(f"crop:   left {LEFT_FRAC*100:.0f}% of W ({W_FULL}) -> output ({LAYERS_OUT},{H_OUT},{W_OUT})")
    print(f"zarr_dir: {ZARR_DIR}")
    print("=" * 70)

    if args.overlay_only:
        step4_overlay(args.dry_run)
        return

    print("\n[1/4] download + z-pool + write zarr")
    step1_download_zarr(args.workers, args.dry_run)

    print("\n[2/4] inklabels")
    step2_inklabels(args.dry_run)

    print("\n[3/4] norm")
    step3_norm(args.skip_norm)

    print("\n[4/4] overlay")
    step4_overlay(args.dry_run)

    print(f"\n[done]  ({LAYERS_OUT},{H_OUT},{W_OUT})  zarr: {os.path.join(ZARR_DIR,ZID)}.zarr")


if __name__ == "__main__":
    main()
