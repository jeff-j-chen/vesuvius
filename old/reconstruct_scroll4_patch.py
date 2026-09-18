"""reconstruct a scroll4 (PHerc1667) w018 surface-volume PATCH into our training zarr
format: shape (64, H, W), chunks (8, 32, 32), dtype uint16, compressor null, zarr_format 2.

SOURCE (verified correct frame): the S3 w018 flatboi segment. its surface-volume zarr AND
its ink-detection prediction share ONE coordinate frame: 42380 x 98100 in-plane, 109 depth
slices, 2.399um isotropic (level 0). so a bbox on the ink label maps 1:1 to the surface
volume (only depth 109->64 resample + uint8->uint16 cast needed).

NB: the dl.ash2txt /paths/<...>_flatboi composite is a DIFFERENT (older, 3.24/7.91um)
flattening with a different aspect ratio -- it does NOT match this ink label. do not use it.

memory-safe: streams one y-chunk-row strip at a time (download -> assemble -> depth-resample
-> write zarr rows), so the whole volume is never held in RAM. works at any patch size.

label pipeline (order matters): otsu binarize -> morphological CLOSE (repair letter gaps)
-> remove small connected components (kill the isolated speckle noise) -> erode (conservative
positives, matches eroded_inklabels convention).

run:
  python reconstruct_scroll4_patch.py            # baked-in clean-text region
"""
import argparse
import os
import subprocess
import numpy as np
import zarr
import cv2
import tifffile

# ---- source (public open-data bucket, anonymous) ---------------------------
S3_SEG = "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi"
S3_BUCKET = "vesuvius-challenge-open-data"
S3_VOL_L0 = f"s3://{S3_BUCKET}/{S3_SEG}/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0"
S3_INK_KEY = (f"{S3_SEG}/ink-detection/PHerc1667-20240304144031-2.399um-0.22m-78keV-"
              "volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-"
              "tile256-stride128.tif")

SRC_Z = 109
CHUNK = 128
CHUNK_BYTES = SRC_Z * CHUNK * CHUNK   # 1,785,856 uint8 per level0 chunk file
DEPTH_OUT = 64
INK_FRAME = (42380, 98100)            # full-res ink tif == surface-vol level0 in-plane


def _aws(*args):
    subprocess.run(["aws", "s3"] + list(args) + ["--no-sign-request"], check=True)


def download_chunks(y0, y1, x0, x1, cache_dir):
    """pull only the level0 chunk files intersecting the bbox (resumable). one aws cp
    per y-chunk-row, include-filtered to the needed x-chunks."""
    yc0, yc1 = y0 // CHUNK, (y1 - 1) // CHUNK
    xc0, xc1 = x0 // CHUNK, (x1 - 1) // CHUNK
    xc_names = [str(x) for x in range(xc0, xc1 + 1)]
    n_y = yc1 - yc0 + 1
    print(f"[chunks] y {yc0}..{yc1} ({n_y}) x {xc0}..{xc1} ({len(xc_names)}) = "
          f"{n_y * len(xc_names)} files (~{n_y * len(xc_names) * CHUNK_BYTES / 1e9:.1f}GB)")
    for i, yc in enumerate(range(yc0, yc1 + 1)):
        dst = os.path.join(cache_dir, str(yc))
        os.makedirs(dst, exist_ok=True)
        if all(os.path.exists(os.path.join(dst, xn)) for xn in xc_names):
            continue
        includes = []
        for xn in xc_names:
            includes += ["--include", xn]
        _aws("cp", "--recursive", f"{S3_VOL_L0}/0/{yc}/", dst + os.sep,
             "--exclude", "*", *includes)
        if (i + 1) % 8 == 0 or i + 1 == n_y:
            print(f"[chunks] {i + 1}/{n_y} y-rows")


def _resample_weights(z_in, z_out):
    """linear depth resample matrix (z_out, z_in): out = W @ in over depth axis"""
    W = np.zeros((z_out, z_in), dtype=np.float32)
    coords = np.linspace(0.0, z_in - 1, z_out)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, z_in - 1)
    frac = (coords - lo).astype(np.float32)
    for i in range(z_out):
        W[i, lo[i]] += 1.0 - frac[i]
        W[i, hi[i]] += frac[i]
    return W


def stream_write_zarr(y0, y1, x0, x1, cache_dir, out_path):
    """stream per y-chunk-row: assemble strip -> depth-resample 109->64 -> write zarr.
    also accumulates the 2D papyrus mask (any-depth signal). returns the mask array."""
    H, W = y1 - y0, x1 - x0
    Wmat = _resample_weights(SRC_Z, DEPTH_OUT)
    store = zarr.open(out_path, mode="w", shape=(DEPTH_OUT, H, W),
                      chunks=(8, 32, 32), dtype="<u2", compressor=None, zarr_format=2)
    mask = np.zeros((H, W), dtype=np.uint8)
    yc0, yc1 = y0 // CHUNK, (y1 - 1) // CHUNK
    xc0, xc1 = x0 // CHUNK, (x1 - 1) // CHUNK
    for yc in range(yc0, yc1 + 1):
        gy = yc * CHUNK
        ys, ye = max(gy, y0), min(gy + CHUNK, y1)
        strip = np.zeros((SRC_Z, ye - ys, W), dtype=np.uint8)   # one chunk-row tall
        for xc in range(xc0, xc1 + 1):
            gx = xc * CHUNK
            xs, xe = max(gx, x0), min(gx + CHUNK, x1)
            fp = os.path.join(cache_dir, str(yc), str(xc))
            # sparse zarr: missing chunk == all fill_value (0), i.e. air/empty region.
            # aws cp silently skips non-existent chunks, so absence is expected, not an error.
            if not os.path.exists(fp):
                continue
            raw = np.fromfile(fp, dtype=np.uint8)
            if raw.size != CHUNK_BYTES:
                raise ValueError(f"bad chunk {yc}/{xc}: {raw.size} != {CHUNK_BYTES}")
            ck = raw.reshape(SRC_Z, CHUNK, CHUNK)
            strip[:, ys - gy:ye - gy, xs - x0:xe - x0] = ck[:, ys - gy:ye - gy, xs - gx:xe - gx]
        flat = strip.astype(np.float32).reshape(SRC_Z, -1)      # depth resample this strip
        res = (Wmat @ flat).reshape(DEPTH_OUT, ye - ys, W)
        np.clip(res, 0, 65535, out=res)
        store[:, ys - y0:ye - y0, :] = res.round().astype(np.uint16)
        mask[ys - y0:ye - y0, :] = (strip.max(axis=0) > 0).astype(np.uint8) * 255
    print(f"[zarr] wrote {out_path} shape ({DEPTH_OUT},{H},{W})")
    return mask


def build_labels(y0, y1, x0, x1, out_id, tmp_dir, mask,
                 close_size=3, min_component=8, erosion_size=3, iterations=12):
    """crop full-res ink prediction to the SAME bbox, binarize, close, de-speck, erode.
    the full-res tif is (42380,98100) == the surface-volume frame, so the crop is 1:1."""
    tif_local = os.path.join(tmp_dir, "ink_full.tif")
    if not os.path.exists(tif_local):
        _aws("cp", f"s3://{S3_BUCKET}/{S3_INK_KEY}", tif_local)
    ink = tifffile.imread(tif_local)
    if ink.ndim == 3:
        ink = ink[..., 0]
    assert ink.shape == INK_FRAME, f"ink tif {ink.shape} != expected {INK_FRAME}"
    ink = ink[y0:y1, x0:x1]
    if ink.dtype != np.uint8:
        ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_frac = float((binary > 0).mean())
    # CLOSE: repair small gaps within letter strokes (dilate then erode)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    # DE-SPECK: drop connected components smaller than min_component px (the bottom dots)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
    keep = np.zeros_like(closed)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] >= min_component:
            keep[lbl == c] = 255
    despeck_frac = float((keep > 0).mean())
    os.makedirs("inklabels", exist_ok=True)
    os.makedirs("eroded_inklabels", exist_ok=True)
    cv2.imwrite(f"inklabels/{out_id}.png", keep)
    eroded = cv2.erode(keep, np.ones((erosion_size, erosion_size), np.uint8), iterations=iterations)
    cv2.imwrite(f"eroded_inklabels/{out_id}.png", eroded)
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(f"masks/{out_id}.png", mask)
    print(f"[label] otsu={raw_frac:.3f} -> close+despeck={despeck_frac:.3f} "
          f"eroded={float((eroded > 0).mean()):.3f}  mask valid={float((mask > 0).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser(description="reconstruct a scroll4 w018 clean-text patch")
    # baked-in default = best clean-text block (chunk-aligned, avoids noisy bottom band)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=9600)
    ap.add_argument("--x0", type=int, default=6144)
    ap.add_argument("--x1", type=int, default=16384)
    ap.add_argument("--out-id", type=str, default="20240304144031")
    ap.add_argument("--zarr-path", type=str,
                    default=os.getenv("VESUVIUS_ZARR_PATH", r"C:\Users\ChenJeff\Documents\ves_zarrs2"))
    ap.add_argument("--cache-dir", type=str, default=r"C:\Users\ChenJeff\Documents\_ves_tmp\s4_chunks")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    tmp_dir = os.path.dirname(args.cache_dir)
    out_zarr = os.path.join(args.zarr_path, f"{args.out_id}.zarr")
    print(f"[patch] y[{args.y0}-{args.y1}] x[{args.x0}-{args.x1}] "
          f"H={args.y1-args.y0} W={args.x1-args.x0} -> {out_zarr}")

    download_chunks(args.y0, args.y1, args.x0, args.x1, args.cache_dir)
    mask = stream_write_zarr(args.y0, args.y1, args.x0, args.x1, args.cache_dir, out_zarr)
    build_labels(args.y0, args.y1, args.x0, args.x1, args.out_id, tmp_dir, mask)
    print("[done] reconstruction complete")


if __name__ == "__main__":
    main()
