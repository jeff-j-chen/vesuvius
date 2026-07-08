"""precompute_norm.py — fast, chunk-aligned normalization stats for a big zarr.

WHY: the in-pipeline norm loop reads full z-slices (vol[z,:,:]). with our tiny (8,32,32)
chunks and a huge (13303x31674) frame, one z-slice touches ~410k chunk files AND, because
chunks are 8 deep, every z index re-reads its whole 8-deep chunk band -> the full 54GB volume
is effectively read ~8x with millions of tiny random I/Os (observed ~330s per z, ~12h total,
and a likely disk-thrash crash trigger).

THIS reads the volume ONCE in chunk-aligned blocks (z step 8 = chunk depth, y in wide bands
= full width is contiguous), so each chunk is touched exactly once. a single monotonic pass
gives sum/sqsum/count + raw min/max; the normalized min/max the pipeline caches are then just
(raw_min-mean)/std and (raw_max-mean)/std (normalization is monotincreasing, so it preserves
extrema). result is written into norm_cache.json under the segment id, matching the schema the
visualizer/dataloader expect, so training then SKIPS the slow loop entirely.
"""
import argparse
import json
import os
import numpy as np
import zarr
import cv2


def main():
    ap = argparse.ArgumentParser(description="fast chunk-aligned norm precompute -> norm_cache.json")
    ap.add_argument("--scroll-id", type=str, required=True)
    ap.add_argument("--zarr-path", type=str,
                    default=os.getenv("VESUVIUS_ZARR_PATH", r"C:\Users\ChenJeff\Documents\ves_zarrs2"))
    ap.add_argument("--cache", type=str, default="./norm_cache.json")
    ap.add_argument("--y-block", type=int, default=512, help="y band height per read (mult of 32)")
    args = ap.parse_args()

    sid = args.scroll_id
    zpath = os.path.join(args.zarr_path, f"{sid}.zarr")
    vol = zarr.open(zpath, mode="r")
    D, H, W = map(int, vol.shape)
    zc = int(vol.chunks[0])  # chunk depth (8) -> read whole z-bands so chunks are hit once

    mask = cv2.imread(f"./masks/{sid}.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask not found: ./masks/{sid}.png")
    mbin = mask > 0
    print(f"[norm] {sid} vol=({D},{H},{W}) chunkz={zc} mask_valid={float(mbin.mean()):.3f}", flush=True)

    total_sum = 0.0
    total_sq = 0.0
    total_n = 0
    raw_min = np.inf
    raw_max = -np.inf

    yb = ((args.y_block + 31) // 32) * 32
    for z0 in range(0, D, zc):
        z1 = min(z0 + zc, D)
        for y0 in range(0, H, yb):
            y1 = min(y0 + yb, H)
            block = np.asarray(vol[z0:z1, y0:y1, :])       # (dz, dy, W) one pass over chunks
            m = mbin[y0:y1, :]                              # (dy, W)
            m3 = np.broadcast_to(m[None, :, :], block.shape)
            valid = block[m3]
            if valid.size == 0:
                continue
            v64 = valid.astype(np.float64)
            total_sum += float(v64.sum())
            total_sq += float(np.square(v64).sum())
            total_n += int(v64.size)
            raw_min = min(raw_min, float(v64.min()))
            raw_max = max(raw_max, float(v64.max()))
        print(f"[norm] z-band {z0}:{z1} done  n={total_n}", flush=True)

    if total_n == 0:
        raise ValueError("no valid pixels under mask")

    mean = total_sum / total_n
    std = float(np.sqrt(max(total_sq / total_n - mean * mean, 1e-12)))
    # normalized extrema (monotonic transform preserves argmin/argmax)
    norm_min = (raw_min - mean) / std
    norm_max = (raw_max - mean) / std

    # merge into cache under the seg id, matching the pipeline schema
    cache = {}
    if os.path.exists(args.cache):
        try:
            with open(args.cache) as f:
                cache = json.load(f)
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    cache[sid] = {"mean": mean, "std": std, "min": norm_min, "max": norm_max}
    with open(args.cache, "w") as f:
        json.dump(cache, f, indent=4)

    print(f"[norm] mean={mean:.4f} std={std:.4f} rawmin={raw_min} rawmax={raw_max} "
          f"normmin={norm_min:.4f} normmax={norm_max:.4f}")
    print(f"[norm] wrote {args.cache}[{sid}]")


if __name__ == "__main__":
    main()
