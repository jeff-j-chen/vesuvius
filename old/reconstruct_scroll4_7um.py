"""reconstruct a scroll4 (PHerc1667) 7.91um/53keV path segment region into our training
zarr format: shape (64, H, W), chunks (8, 32, 32), dtype uint16, compressor null, zarr_format 2.

SOURCE: dl.ash2txt volpkg path 20240304144031/layers/{00..64}.tif -- the "arched" flatboi
flattening on volume 20231117161658 (7.91um, 53keV). 65 depth layers, each a single-strip
uncompressed row-major bigtiff-free TIF of (14377, 30563) uint16.

NB: this is a DIFFERENT flattening from the 2.4um S3 surface (arched vs straight), so its
frame does NOT correspond pixel-wise to the 2.4um patch. ink labels for this region must be
drawn / warped separately -- this script produces ONLY the volume + papyrus mask.

download strategy: each layer is one contiguous row-major strip (pixel data at byte offset
`off`, row y at off + y*W*2). so a y-band [y0:y1] full-width is one contiguous byte range
per layer -> one range request per layer per y-block. x cannot be cheaply cropped (rows are
contiguous), so full width is kept.

memory-safe: streams y-block by y-block (range-request all 65 layers for the block ->
resample depth 65->64 -> write zarr rows). never holds the whole volume in RAM.
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zarr
import cv2

BASE = ("https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/"
        "paths/20240304161941/layers/")
N_LAYERS = 65
DEPTH_OUT = 64
BPP = 2  # uint16

# these layers are single-strip uncompressed row-major TIFs with the IFD at END of file
# (so a header range can't be parsed). structure verified from the local full layer 32 of w023:
#   data offset 8, frame (13303, 31674), size 13303*31674*2 + 8 + IFD == server Content-Length.
# all 65 layers are machine-generated identically, so we use these constants directly.
# NB: we FLIP each layer HORIZONTALLY (x reversed) so the volume matches the horizontally-
# flipped 7.91 frame used for the warp/inklabels.
DATA_OFFSET = 8
FRAME_H = 13303
FRAME_W = 31674


def _curl_range(url, start, end, out):
    """fetch byte range [start,end] (inclusive) of url to file out via curl.

    timeouts + retry are ESSENTIAL: without --max-time a single hung request from the
    public server freezes the whole reconstruction indefinitely (observed repeatedly).
    --retry handles transient drops; --connect-timeout caps the handshake; --max-time caps
    the whole transfer (a full-width 256-row layer band is ~16MB, so 180s is generous)."""
    subprocess.run(
        ["curl", "-s", "--fail",
         "--connect-timeout", "20", "--max-time", "180",
         "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
         "-r", f"{start}-{end}", url, "-o", out],
        check=True)


def _fetch_layer(args):
    """thread worker: download one layer's y-band to its own temp file and return the array"""
    L, url, start, end, out, rows, W = args
    _curl_range(url, start, end, out)
    arr = np.fromfile(out, dtype="<u2")
    if arr.size != rows * W:
        raise ValueError(f"layer {L}: {arr.size} != {rows*W}")
    return L, arr.reshape(rows, W)


def _resample_weights(z_in, z_out):
    """linear depth resample matrix (z_out, z_in): out = W @ in over depth axis"""
    Wm = np.zeros((z_out, z_in), dtype=np.float32)
    coords = np.linspace(0.0, z_in - 1, z_out)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, z_in - 1)
    frac = (coords - lo).astype(np.float32)
    for i in range(z_out):
        Wm[i, lo[i]] += 1.0 - frac[i]
        Wm[i, hi[i]] += frac[i]
    return Wm


def reconstruct(y0, y1, W, out_zarr, out_id, tmp_dir, block=256, flip_h=True, workers=8):
    """stream y-blocks: range-request all 65 layers for the block (in PARALLEL), depth-resample
    65->64, write zarr. accumulates + writes the papyrus mask (any-depth signal). flip_h reverses
    x so the volume matches the horizontally-flipped inklabels.

    RESUMABLE: a sidecar <out_zarr>/.recon_progress records the last completed block and the
    running mask is checkpointed to <tmp>/_mask_<id>.npy. on restart we reopen the zarr r+ and
    skip already-written blocks, so a stall no longer wipes all progress."""
    H = y1 - y0
    Wmat = _resample_weights(N_LAYERS, DEPTH_OUT)
    n_blocks = (H + block - 1) // block

    prog_path = os.path.join(out_zarr, ".recon_progress")
    mask_ckpt = os.path.join(tmp_dir, f"_mask_{out_id}.npy")

    # decide fresh vs resume: resume only if the zarr, progress sidecar, and mask ckpt all
    # exist and the stored geometry matches exactly.
    resume_from = 0
    can_resume = False
    if os.path.exists(prog_path) and os.path.exists(mask_ckpt):
        try:
            import json
            with open(prog_path) as f:
                p = json.load(f)
            if (p.get("H") == H and p.get("W") == W and p.get("block") == block
                    and p.get("flip_h") == flip_h):
                resume_from = int(p.get("next_block", 0))
                can_resume = resume_from > 0
        except Exception:
            can_resume = False

    if can_resume:
        store = zarr.open(out_zarr, mode="r+")
        mask = np.load(mask_ckpt)
        print(f"[resume] continuing from block {resume_from+1}/{n_blocks}", flush=True)
    else:
        store = zarr.open(out_zarr, mode="w", shape=(DEPTH_OUT, H, W),
                          chunks=(8, 32, 32), dtype="<u2", compressor=None, zarr_format=2)
        mask = np.zeros((H, W), dtype=np.uint8)
        resume_from = 0

    block_starts = list(range(0, H, block))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for bi, b0 in enumerate(block_starts):
            if bi < resume_from:
                continue
            b1 = min(b0 + block, H)
            gy0, gy1 = y0 + b0, y0 + b1               # global y range for this block
            rows = b1 - b0
            start = DATA_OFFSET + gy0 * W * BPP
            end = DATA_OFFSET + gy1 * W * BPP - 1
            strip = np.empty((N_LAYERS, rows, W), dtype=np.uint16)
            # fetch all 65 layers concurrently (each to its own temp file)
            jobs = [(L, f"{BASE}{L:02d}.tif", start, end,
                     os.path.join(tmp_dir, f"_band_{L:02d}.raw"), rows, W)
                    for L in range(N_LAYERS)]
            for L, arr in pool.map(_fetch_layer, jobs):
                strip[L] = arr
            res = (Wmat @ strip.astype(np.float32).reshape(N_LAYERS, -1)).reshape(DEPTH_OUT, rows, W)
            np.clip(res, 0, 65535, out=res)
            block_u16 = res.round().astype(np.uint16)
            block_mask = (strip.max(axis=0) > 0).astype(np.uint8) * 255
            if flip_h:                                # x-reverse to match flipped inklabels
                block_u16 = block_u16[:, :, ::-1]
                block_mask = block_mask[:, ::-1]
            store[:, b0:b1, :] = block_u16
            mask[b0:b1, :] = block_mask
            # checkpoint progress + running mask so a stall can resume from here
            np.save(mask_ckpt, mask)
            import json
            with open(prog_path, "w") as f:
                json.dump({"H": H, "W": W, "block": block, "flip_h": flip_h,
                           "next_block": bi + 1}, f)
            print(f"[block] {bi+1}/{n_blocks} rows[{gy0}:{gy1}]", flush=True)
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(f"masks/{out_id}.png", mask)
    # clean up the resume checkpoints on successful completion
    try:
        os.remove(mask_ckpt)
        os.remove(prog_path)
    except Exception:
        pass
    print(f"[zarr] wrote {out_zarr} shape ({DEPTH_OUT},{H},{W}); mask valid={float((mask>0).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser(description="reconstruct scroll4 7.91um path region -> our zarr")
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=FRAME_H)   # full frame by default (7.91 is small enough)
    # integer id (pipeline does int(scroll_id)); use the w023 segment id
    ap.add_argument("--out-id", type=str, default="20240304161941")
    ap.add_argument("--zarr-path", type=str,
                    default=os.getenv("VESUVIUS_ZARR_PATH", r"C:\Users\ChenJeff\Documents\ves_zarrs2"))
    ap.add_argument("--tmp-dir", type=str, default=r"C:\Users\ChenJeff\Documents\_ves_tmp")
    ap.add_argument("--no-flip", action="store_true", help="do NOT horizontally flip (default: flip)")
    ap.add_argument("--workers", type=int, default=8, help="parallel layer downloads per block")
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    W = FRAME_W
    y1 = min(args.y1, FRAME_H)
    out_zarr = os.path.join(args.zarr_path, f"{args.out_id}.zarr")
    flip_h = not args.no_flip
    print(f"[region] y[{args.y0}:{y1}] full width {W} (frame {FRAME_H}x{FRAME_W}) flip_h={flip_h} -> {out_zarr}", flush=True)
    reconstruct(args.y0, y1, W, out_zarr, args.out_id, args.tmp_dir, flip_h=flip_h, workers=args.workers)
    print("[done] 7.91um reconstruction complete")


if __name__ == "__main__":
    main()
