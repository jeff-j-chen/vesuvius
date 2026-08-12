#!/usr/bin/env python3
"""download_w035_labels.py -- download and process PHerc0139 w035 ink labels.

downloads the 1.129um / 59keV ink detection TIF from S3, resizes it to the
9.362um zarr coordinate system (shape 28x5820x5240), and saves:
  inklabels/20260317000000.png        -- continuous 0-255 probability map
  eroded_inklabels/20260317000000.png -- conservative binary labels for training

NOTE: the zarr volume itself is assembled separately via:
  python assemble_training_segments.py --only w035
the mask (masks/20260317000000.png) is only available after zarr assembly;
without it the labels are saved unmasked (run again after assembly to apply mask).
"""
import os, subprocess, sys, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ZID = "20260317000000"
# 9.362um zarr shape (from .zarray metadata: D,H,W = 28,5820,5240)
ZARR_H, ZARR_W = 5820, 5240

# 1.129um / 59keV ink detection (L1 merged model -- same source as other PHerc0139 fragments)
INK_URL = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
    "PHerc0139/segments/20260317000000-w035_2026031718/ink-detection/"
    "PHerc0139-20260317000000-1.129um-0.22m-59keV-volume-20260413113053-"
    "L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif"
)
TMP_TIF  = f"_ves_tmp/{ZID}_ink.tif"
INKLABEL_PNG = f"inklabels/{ZID}.png"
ERODED_PNG   = f"eroded_inklabels/{ZID}.png"
MASK_PNG     = f"masks/{ZID}.png"

# standard PHerc0139 thresholds (same as all other fragments)
THRESHOLD   = 140   # ink if > 0.55*255
ERODE_ITERS = 12    # 3x3 kernel iterations


def download_tif():
    os.makedirs("_ves_tmp", exist_ok=True)
    if os.path.exists(TMP_TIF) and os.path.getsize(TMP_TIF) > 1_000_000:
        print(f"[ink] TIF already cached ({os.path.getsize(TMP_TIF)/1e6:.0f} MB), skipping")
        return
    print("[ink] downloading inklabels TIF (may take a few minutes)...")
    t0 = time.time()
    r = subprocess.run([
        "curl", "-s", "--fail", "--max-time", "900", "--retry", "3",
        "--retry-all-errors", "-o", TMP_TIF, INK_URL
    ])
    if r.returncode != 0:
        raise RuntimeError(f"curl failed with code {r.returncode}")
    elapsed = time.time() - t0
    size_mb = os.path.getsize(TMP_TIF) / 1e6
    print(f"[ink] downloaded {size_mb:.0f} MB in {elapsed:.0f}s")


def load_and_resize_tif():
    print(f"[ink] loading TIF from {TMP_TIF}...")
    try:
        import tifffile
        arr = tifffile.imread(TMP_TIF)
        print(f"[ink] loaded via tifffile: shape={arr.shape} dtype={arr.dtype}")
    except ImportError:
        print("[ink] tifffile not available, falling back to PIL...")
        arr = np.array(Image.open(TMP_TIF))
        print(f"[ink] loaded via PIL: shape={arr.shape} dtype={arr.dtype}")

    # take max over channels/pages if multi-dimensional
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    elif arr.ndim > 3:
        arr = arr.max(axis=tuple(range(arr.ndim - 2)))

    # normalize to uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # source is ~8.29x larger per dimension (9.362/1.129 scale ratio)
    print(f"[ink] TIF source shape: {arr.shape}  scale ~{arr.shape[0]/ZARR_H:.2f}x")

    img_resized = Image.fromarray(arr).resize((ZARR_W, ZARR_H), Image.LANCZOS)
    out = np.array(img_resized)
    print(f"[ink] resized to zarr dims: ({ZARR_H}, {ZARR_W})")
    return out


def apply_mask(arr):
    """apply the papyrus mask if available (generated during zarr assembly)."""
    if not os.path.exists(MASK_PNG):
        print(f"[ink] mask not found at {MASK_PNG} -- skipping "
              f"(assemble zarr first, then re-run to apply mask)")
        return arr
    mask = np.array(Image.open(MASK_PNG).convert("L"))
    if mask.shape != (ZARR_H, ZARR_W):
        mask = np.array(Image.fromarray(mask).resize((ZARR_W, ZARR_H), Image.NEAREST))
    arr = arr * (mask > 127).astype(arr.dtype)
    print(f"[ink] mask applied: valid_frac={(arr > 0).mean():.4f}")
    return arr


def save_inklabels(arr):
    os.makedirs("inklabels", exist_ok=True)
    Image.fromarray(arr).save(INKLABEL_PNG)
    ink_frac = (arr > THRESHOLD).mean()
    print(f"[ink] saved {INKLABEL_PNG}  ink_frac@{THRESHOLD}={ink_frac:.4f}")


def generate_eroded(arr):
    """threshold + erode to produce conservative binary training labels."""
    import cv2
    binary = (arr > THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=ERODE_ITERS)
    if os.path.exists(MASK_PNG):
        mask = np.array(Image.open(MASK_PNG).convert("L"))
        if mask.shape != eroded.shape:
            mask = np.array(Image.fromarray(mask).resize(
                (eroded.shape[1], eroded.shape[0]), Image.NEAREST))
        eroded = eroded * (mask > 127).astype(eroded.dtype)
    os.makedirs("eroded_inklabels", exist_ok=True)
    Image.fromarray(eroded).save(ERODED_PNG)
    ink_frac = (eroded > 0).mean()
    print(f"[ink] saved {ERODED_PNG}  eroded_ink_frac={ink_frac:.4f}  "
          f"(thr={THRESHOLD}, iters={ERODE_ITERS})")


def main():
    print(f"[ink] processing PHerc0139 w035 inklabels for zarr id {ZID}")
    download_tif()
    arr = load_and_resize_tif()
    arr = apply_mask(arr)
    save_inklabels(arr)
    generate_eroded(arr)
    print(f"[ink] done. review {INKLABEL_PNG} and re-run with --erode-only if you edit it.")


if __name__ == "__main__":
    if "--erode-only" in sys.argv:
        # re-erode from existing inklabels PNG (use after manual edits)
        print(f"[ink] re-eroding from existing {INKLABEL_PNG}...")
        arr = np.array(Image.open(INKLABEL_PNG).convert("L"))
        generate_eroded(arr)
    else:
        main()
