"""build_scroll_zarr.py — unified scroll/segment -> training-zarr builder.

replaces the three one-off scripts (reconstruct_scroll3_7um.py, reconstruct_scroll4_7um.py,
reconstruct_scroll4_patch.py) with one CLI that can assemble ANY scroll fragment into our
training format: shape (64, H, W), chunks (8,32,32), dtype uint16, compressor null, zarr_format 2.

TWO SOURCE TYPES
  volpkg  — dl.ash2txt volpkg surface segment, i.e. paths/<seg>/layers/{00..64}.tif.
            65 single-strip uncompressed row-major TIFF layers (the 7.91um scroll3/scroll4
            source). geometry (W, H, data offset, layer count) is AUTO-DETECTED from the
            layer-0 TIFF header, so any segment works without hardcoding. optional h-flip.
  s3patch — S3 open-data surface-volume zarr (2.399um) + its ink-detection prediction tif.
            downloads only the chunks intersecting a bbox, depth-resamples, and (optionally)
            bakes inklabels from the prediction tif. this is the label-bearing 2.4um path.

COMMON PIPELINE
  - stream by y-blocks / chunk-rows (never hold the whole volume in RAM)
  - linear depth resample source-depth -> 64
  - papyrus mask = any-depth signal > 0
  - hardened download: curl --max-time/--retry, parallel layers, resumable checkpoint
  - writes ves_zarrs2/<id>.zarr + masks/<id>.png (+ inklabels/ eroded_inklabels/ for s3patch)

EXAMPLES
  # named presets (the three we already use)
  python build_scroll_zarr.py preset scroll3
  python build_scroll_zarr.py preset scroll4-79
  python build_scroll_zarr.py preset scroll4-24-patch

  # ANY volpkg segment (geometry auto-detected)
  python build_scroll_zarr.py volpkg \
      --base-url https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/paths/<seg>/layers/ \
      --out-id <seg> [--flip] [--y0 0 --y1 4000] [--workers 8]

  # a bbox patch from an S3 surface volume (+ optional ink labels)
  python build_scroll_zarr.py s3patch \
      --seg PHerc1667/segments/<...>_flatboi \
      --vol-subpath surface-volumes/<vol>.zarr/0 \
      --ink-key <...>/ink-detection/<...>.tif \
      --out-id <id> --y0 0 --y1 9600 --x0 6144 --x1 16384

AFTER BUILDING (big frames): precompute normalization to avoid the slow in-pipeline norm loop:
  python precompute_norm.py --scroll-id <id>
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zarr
import cv2

DEPTH_OUT = 64
BPP = 2  # uint16 output
# portable curl: windows ships curl.exe, linux/mac ship curl. pick whatever is on PATH
CURL = "curl.exe" if shutil.which("curl.exe") else "curl"
# default output/scratch roots. on linux (remote server) the scrolls live on the mounted
# network volume at /workspace; on windows fall back to the old local docs path
_POSIX_DEFAULT_ZARR = "/vesuvius/ves_zarrs2"
_POSIX_DEFAULT_TMP = "/vesuvius/_ves_tmp"
_WIN_DEFAULT_ZARR = r"C:\Users\ChenJeff\Documents\ves_zarrs2"
_WIN_DEFAULT_TMP = r"C:\Users\ChenJeff\Documents\_ves_tmp"
DEFAULT_ZARR = os.getenv("VESUVIUS_ZARR_PATH",
                         _POSIX_DEFAULT_ZARR if os.name == "posix" else _WIN_DEFAULT_ZARR)
DEFAULT_TMP = os.getenv("VESUVIUS_TMP_DIR",
                        _POSIX_DEFAULT_TMP if os.name == "posix" else _WIN_DEFAULT_TMP)


# ============================================================================
# shared helpers
# ============================================================================
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


def _curl_range(url, start, end, out):
    """fetch byte range [start,end] (inclusive) of url to file out via curl.

    timeouts + retry are ESSENTIAL: without --max-time a single hung request from the public
    server freezes the whole reconstruction indefinitely (observed repeatedly)."""
    subprocess.run(
        [CURL, "-s", "--fail",
         "--connect-timeout", "20", "--max-time", "180",
         "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
         "-r", f"{start}-{end}", url, "-o", out],
        check=True)


def _curl_head_len(url):
    """return Content-Length for url via a HEAD request, or None if it doesn't exist."""
    try:
        out = subprocess.run([CURL, "-sI", "--fail", "--connect-timeout", "20", url],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


# ============================================================================
# volpkg layer-TIFF path
# ============================================================================
def detect_layer_format(base_url):
    """find the layer filename format + count. tries 2- then 3-digit naming, then probes
    upward (HEAD) until a layer is missing. returns (fmt, n_layers)."""
    for fmt in ("{:02d}.tif", "{:03d}.tif"):
        if _curl_head_len(base_url + fmt.format(0)) is not None:
            # count: probe upward. start dense then confirm the boundary.
            n = 1
            while _curl_head_len(base_url + fmt.format(n)) is not None:
                n += 1
                if n > 400:  # safety cap
                    break
            return fmt, n
    raise RuntimeError(f"no layer 0 found at {base_url} (tried 00.tif and 000.tif)")


def detect_tiff_geometry(url, content_len):
    """parse a classic (little/big-endian) TIFF header + IFD to get (W, H, data_offset).

    these layers are single-strip uncompressed; the IFD sits at the END of the file, but the
    header still carries the IFD offset in bytes 4-8. we fetch the header, then the IFD."""
    hdr_path = os.path.join(DEFAULT_TMP, "_geom_hdr.bin")
    _curl_range(url, 0, 15, hdr_path)
    b = open(hdr_path, "rb").read()
    order = "<" if b[:2] == b"II" else ">"
    ifd_off = struct.unpack(order + "I", b[4:8])[0]
    ifd_path = os.path.join(DEFAULT_TMP, "_geom_ifd.bin")
    _curl_range(url, ifd_off, content_len - 1, ifd_path)
    d = open(ifd_path, "rb").read()
    ntags = struct.unpack(order + "H", d[:2])[0]
    fields = {}
    for i in range(ntags):
        off = 2 + i * 12
        tag, typ, cnt = struct.unpack(order + "HHI", d[off:off + 8])
        raw = d[off + 8:off + 12]
        val = struct.unpack(order + "I", raw)[0] if typ == 4 else struct.unpack(order + "HH", raw)[0]
        fields[tag] = val
    W, H = fields.get(256), fields.get(257)
    data_off = fields.get(273, 8)   # StripOffsets
    if not W or not H:
        raise RuntimeError(f"could not parse W/H from IFD of {url}")
    return int(W), int(H), int(data_off)


def _fetch_layer(args):
    """thread worker: download one layer's y-band to its own temp file and return the array"""
    L, url, start, end, out, rows, W = args
    _curl_range(url, start, end, out)
    arr = np.fromfile(out, dtype="<u2")
    if arr.size != rows * W:
        raise ValueError(f"layer {L}: {arr.size} != {rows*W}")
    return L, arr.reshape(rows, W)


def build_volpkg(base_url, out_id, zarr_path, tmp_dir, y0=0, y1=None,
                 flip_h=False, block=256, workers=8):
    """assemble a volpkg surface segment (layer TIFFs) into our zarr. geometry auto-detected.

    RESUMABLE: .recon_progress sidecar + _mask_<id>.npy checkpoint; restart reopens r+ and
    skips done blocks (a stall no longer wipes progress)."""
    os.makedirs(tmp_dir, exist_ok=True)
    fmt, n_layers = detect_layer_format(base_url)
    clen = _curl_head_len(base_url + fmt.format(0))
    W, H_full, data_off = detect_tiff_geometry(base_url + fmt.format(0), clen)
    y1 = H_full if y1 is None else min(y1, H_full)
    H = y1 - y0
    out_zarr = os.path.join(zarr_path, f"{out_id}.zarr")
    print(f"[volpkg] {n_layers} layers fmt={fmt} frame={H_full}x{W} data_off={data_off} "
          f"flip_h={flip_h}", flush=True)
    print(f"[region] y[{y0}:{y1}] full width {W} -> {out_zarr}", flush=True)

    Wmat = _resample_weights(n_layers, DEPTH_OUT)
    n_blocks = (H + block - 1) // block
    prog_path = os.path.join(out_zarr, ".recon_progress")
    mask_ckpt = os.path.join(tmp_dir, f"_mask_{out_id}.npy")

    resume_from, can_resume = 0, False
    if os.path.exists(prog_path) and os.path.exists(mask_ckpt):
        try:
            import json
            p = json.load(open(prog_path))
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for bi, b0 in enumerate(range(0, H, block)):
            if bi < resume_from:
                continue
            b1 = min(b0 + block, H)
            gy0, gy1 = y0 + b0, y0 + b1
            rows = b1 - b0
            start = data_off + gy0 * W * BPP
            end = data_off + gy1 * W * BPP - 1
            strip = np.empty((n_layers, rows, W), dtype=np.uint16)
            jobs = [(L, base_url + fmt.format(L), start, end,
                     os.path.join(tmp_dir, f"_band_{out_id}_{L:03d}.raw"), rows, W)
                    for L in range(n_layers)]
            for L, arr in pool.map(_fetch_layer, jobs):
                strip[L] = arr
            res = (Wmat @ strip.astype(np.float32).reshape(n_layers, -1)).reshape(DEPTH_OUT, rows, W)
            np.clip(res, 0, 65535, out=res)
            block_u16 = res.round().astype(np.uint16)
            block_mask = (strip.max(axis=0) > 0).astype(np.uint8) * 255
            if flip_h:
                block_u16 = block_u16[:, :, ::-1]
                block_mask = block_mask[:, ::-1]
            store[:, b0:b1, :] = block_u16
            mask[b0:b1, :] = block_mask
            np.save(mask_ckpt, mask)
            import json
            with open(prog_path, "w") as f:
                json.dump({"H": H, "W": W, "block": block, "flip_h": flip_h,
                           "next_block": bi + 1}, f)
            print(f"[block] {bi+1}/{n_blocks} rows[{gy0}:{gy1}]", flush=True)

    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(f"masks/{out_id}.png", mask)
    for p in (mask_ckpt, prog_path):
        try:
            os.remove(p)
        except Exception:
            pass
    print(f"[zarr] wrote {out_zarr} shape ({DEPTH_OUT},{H},{W}); "
          f"mask valid={float((mask>0).mean()):.3f}", flush=True)
    print("[done] volpkg reconstruction complete")


# ============================================================================
# S3 chunked-zarr patch path
# ============================================================================
S3_BUCKET = "vesuvius-challenge-open-data"


def _aws(*args):
    # call awscli via this venv's python (no aws.exe on PATH here); public bucket
    subprocess.run([sys.executable, "-m", "awscli", "s3"] + list(args) + ["--no-sign-request"], check=True)


def _download_chunks(vol_l0, y0, y1, x0, x1, cache_dir, chunk, src_z):
    """pull only the level0 chunk files intersecting the bbox (resumable)."""
    chunk_bytes = src_z * chunk * chunk
    yc0, yc1 = y0 // chunk, (y1 - 1) // chunk
    xc0, xc1 = x0 // chunk, (x1 - 1) // chunk
    xc_names = [str(x) for x in range(xc0, xc1 + 1)]
    n_y = yc1 - yc0 + 1
    print(f"[chunks] y {yc0}..{yc1} ({n_y}) x {xc0}..{xc1} ({len(xc_names)}) = "
          f"{n_y*len(xc_names)} files (~{n_y*len(xc_names)*chunk_bytes/1e9:.1f}GB)")
    for i, yc in enumerate(range(yc0, yc1 + 1)):
        dst = os.path.join(cache_dir, str(yc))
        os.makedirs(dst, exist_ok=True)
        if all(os.path.exists(os.path.join(dst, xn)) for xn in xc_names):
            continue
        includes = []
        for xn in xc_names:
            includes += ["--include", xn]
        _aws("cp", "--recursive", f"{vol_l0}/0/{yc}/", dst + os.sep, "--exclude", "*", *includes)
        if (i + 1) % 8 == 0 or i + 1 == n_y:
            print(f"[chunks] {i+1}/{n_y} y-rows")


def build_s3patch(seg, vol_subpath, ink_key, out_id, zarr_path, tmp_dir,
                  y0, y1, x0, x1, src_z=109, chunk=128, ink_frame=None,
                  close_size=3, min_component=8, erosion_size=3, iterations=12,
                  depth_out=None, purge_chunks=False):
    """assemble a bbox from an S3 surface-volume zarr, and (if ink_key given) bake labels.

    depth_out: output zarr depth layers. None -> src_z (keep NATIVE depth, no z-resample).
    purge_chunks: delete each y-chunk-row's cache files right after it is consumed, so the
        on-disk peak stays ~= the output zarr size instead of chunks+zarr simultaneously.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    cache_dir = os.path.join(tmp_dir, f"{out_id}_chunks")
    os.makedirs(cache_dir, exist_ok=True)
    vol_l0 = f"s3://{S3_BUCKET}/{seg}/{vol_subpath}"
    out_zarr = os.path.join(zarr_path, f"{out_id}.zarr")
    H, W = y1 - y0, x1 - x0
    chunk_bytes = src_z * chunk * chunk
    z_out = int(depth_out) if depth_out else int(src_z)   # native depth by default
    print(f"[s3patch] y[{y0}-{y1}] x[{x0}-{x1}] H={H} W={W} depth_out={z_out} -> {out_zarr}", flush=True)

    _download_chunks(vol_l0, y0, y1, x0, x1, cache_dir, chunk, src_z)

    Wmat = _resample_weights(src_z, z_out)
    store = zarr.open(out_zarr, mode="w", shape=(z_out, H, W),
                      chunks=(8, 32, 32), dtype="<u2", compressor=None, zarr_format=2)
    mask = np.zeros((H, W), dtype=np.uint8)
    yc0, yc1 = y0 // chunk, (y1 - 1) // chunk
    xc0, xc1 = x0 // chunk, (x1 - 1) // chunk
    for yc in range(yc0, yc1 + 1):
        gy = yc * chunk
        ys, ye = max(gy, y0), min(gy + chunk, y1)
        strip = np.zeros((src_z, ye - ys, W), dtype=np.uint8)
        for xc in range(xc0, xc1 + 1):
            gx = xc * chunk
            xs, xe = max(gx, x0), min(gx + chunk, x1)
            fp = os.path.join(cache_dir, str(yc), str(xc))
            if not os.path.exists(fp):     # sparse zarr: missing chunk == air (zeros)
                continue
            raw = np.fromfile(fp, dtype=np.uint8)
            if raw.size != chunk_bytes:
                raise ValueError(f"bad chunk {yc}/{xc}: {raw.size} != {chunk_bytes}")
            ck = raw.reshape(src_z, chunk, chunk)
            strip[:, ys - gy:ye - gy, xs - x0:xe - x0] = ck[:, ys - gy:ye - gy, xs - gx:xe - gx]
        res = (Wmat @ strip.astype(np.float32).reshape(src_z, -1)).reshape(z_out, ye - ys, W)
        np.clip(res, 0, 65535, out=res)
        store[:, ys - y0:ye - y0, :] = res.round().astype(np.uint16)
        mask[ys - y0:ye - y0, :] = (strip.max(axis=0) > 0).astype(np.uint8) * 255
        if purge_chunks:
            # free this row's raw chunks now that it's baked into the zarr — keeps disk peak
            # near the zarr size instead of chunks+zarr at once.
            import shutil as _sh
            _sh.rmtree(os.path.join(cache_dir, str(yc)), ignore_errors=True)
    print(f"[zarr] wrote {out_zarr} shape ({z_out},{H},{W})", flush=True)

    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(f"masks/{out_id}.png", mask)

    if ink_key:
        import tifffile
        tif_local = os.path.join(tmp_dir, f"{out_id}_ink_full.tif")
        if not os.path.exists(tif_local):
            _aws("cp", f"s3://{S3_BUCKET}/{ink_key}", tif_local)
        ink = tifffile.imread(tif_local)
        if ink.ndim == 3:
            ink = ink[..., 0]
        if ink_frame is not None:
            assert ink.shape == tuple(ink_frame), f"ink {ink.shape} != {ink_frame}"
        ink = ink[y0:y1, x0:x1]
        if ink.dtype != np.uint8:
            ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
        keep = np.zeros_like(closed)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= min_component:
                keep[lbl == c] = 255
        os.makedirs("inklabels", exist_ok=True)
        os.makedirs("eroded_inklabels", exist_ok=True)
        cv2.imwrite(f"inklabels/{out_id}.png", keep)
        eroded = cv2.erode(keep, np.ones((erosion_size, erosion_size), np.uint8), iterations=iterations)
        cv2.imwrite(f"eroded_inklabels/{out_id}.png", eroded)
        print(f"[label] ink frac={float((keep>0).mean()):.3f} "
              f"eroded={float((eroded>0).mean()):.3f}", flush=True)
    print("[done] s3patch reconstruction complete")


# ============================================================================
# presets (the three original scripts, as named configs)
# ============================================================================
PRESETS = {
    "scroll3": dict(
        kind="volpkg", out_id="20240716140050", flip_h=False,
        base_url=("https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/"
                  "paths/20240716140050/layers/"),
    ),
    "scroll4-79": dict(
        kind="volpkg", out_id="20240304161941", flip_h=True,
        base_url=("https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/"
                  "paths/20240304161941/layers/"),
    ),
    "scroll4-24-patch": dict(
        kind="s3patch", out_id="20240304144031",
        seg="PHerc1667/segments/20240304144031-w018_20240304144031_flatboi",
        vol_subpath="surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0",
        ink_key=("PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/"
                 "ink-detection/PHerc1667-20240304144031-2.399um-0.22m-78keV-"
                 "volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-"
                 "tile256-stride128.tif"),
        y0=0, y1=9600, x0=6144, x1=16384, src_z=109, chunk=128, ink_frame=(42380, 98100),
    ),
}


def run_preset(name, zarr_path, tmp_dir, workers):
    if name not in PRESETS:
        raise SystemExit(f"unknown preset '{name}'. options: {list(PRESETS)}")
    p = dict(PRESETS[name])
    kind = p.pop("kind")
    if kind == "volpkg":
        build_volpkg(p["base_url"], p["out_id"], zarr_path, tmp_dir,
                     flip_h=p.get("flip_h", False), workers=workers)
    else:
        build_s3patch(p["seg"], p["vol_subpath"], p.get("ink_key"), p["out_id"],
                      zarr_path, tmp_dir, p["y0"], p["y1"], p["x0"], p["x1"],
                      src_z=p.get("src_z", 109), chunk=p.get("chunk", 128),
                      ink_frame=p.get("ink_frame"))


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="unified scroll -> training-zarr builder")
    ap.add_argument("--zarr-path", default=DEFAULT_ZARR)
    ap.add_argument("--tmp-dir", default=DEFAULT_TMP)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("volpkg", help="build from dl.ash2txt layer TIFFs (auto-geometry)")
    pv.add_argument("--base-url", required=True, help=".../paths/<seg>/layers/ URL")
    pv.add_argument("--out-id", required=True, help="segment id (zarr/mask filename)")
    pv.add_argument("--flip", action="store_true", help="horizontally flip (match flipped labels)")
    pv.add_argument("--y0", type=int, default=0)
    pv.add_argument("--y1", type=int, default=None)
    pv.add_argument("--workers", type=int, default=8)

    ps = sub.add_parser("s3patch", help="build a bbox from an S3 surface volume (+ink labels)")
    ps.add_argument("--seg", required=True)
    ps.add_argument("--vol-subpath", required=True, help="surface-volumes/<vol>.zarr/0")
    ps.add_argument("--ink-key", default=None, help="S3 key of the ink-detection tif (optional)")
    ps.add_argument("--out-id", required=True)
    ps.add_argument("--y0", type=int, required=True)
    ps.add_argument("--y1", type=int, required=True)
    ps.add_argument("--x0", type=int, required=True)
    ps.add_argument("--x1", type=int, required=True)
    ps.add_argument("--src-z", type=int, default=109)
    ps.add_argument("--chunk", type=int, default=128)
    ps.add_argument("--depth-out", type=int, default=None,
                    help="output zarr depth layers; omit to keep NATIVE src-z depth (no z-resample)")
    ps.add_argument("--purge-chunks", action="store_true",
                    help="delete each chunk-row after baking it into the zarr (caps disk peak)")

    pp = sub.add_parser("preset", help="run a named preset")
    pp.add_argument("name", choices=list(PRESETS))
    pp.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "volpkg":
        build_volpkg(args.base_url, args.out_id, args.zarr_path, args.tmp_dir,
                     y0=args.y0, y1=args.y1, flip_h=args.flip, workers=args.workers)
    elif args.cmd == "s3patch":
        build_s3patch(args.seg, args.vol_subpath, args.ink_key, args.out_id,
                      args.zarr_path, args.tmp_dir, args.y0, args.y1, args.x0, args.x1,
                      src_z=args.src_z, chunk=args.chunk,
                      depth_out=args.depth_out, purge_chunks=args.purge_chunks)
    elif args.cmd == "preset":
        run_preset(args.name, args.zarr_path, args.tmp_dir, args.workers)


if __name__ == "__main__":
    main()
