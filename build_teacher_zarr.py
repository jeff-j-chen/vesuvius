"""build_teacher_zarr.py — assemble the high-res 2.4um "teacher" training zarr in the
ALIGNED (3.3x-dense 7.91um-flipped) frame.

why this exists (vs build_scroll_zarr.py):
  the teacher must live in the SAME coordinate space as the 7.91um student so that a
  106x106 teacher tile == a 32x32 student tile physically. we therefore WARP the native
  2.4um data into the 7.91um-flipped frame and up-sample that frame by SCALE=106/32=3.3125,
  which keeps full 2.4um in-plane fidelity (~1:1 with native pixels). labels/mask are the
  hand-cleaned 7.91um ones simply scaled up by the same factor (user preference: cleaner
  than the raw 2.4 ink prediction).

pipeline:
  1. fit a thin-plate-spline from the colored dots: normalized 7.91(u,v) -> normalized 2.4(su,sv)
  2. crop the 7.91 frame to a sub-region (default RIGHT 30% x / TOP 40% y = the ink-rich area)
  3. teacher frame = crop * SCALE. precompute a COARSE 7.91->2.4 pixel map, upsample per block.
  4. download only the 2.4 level-0 chunks intersecting the mapped source bbox (resumable).
  5. stream target y-blocks: load the needed 2.4 source rows, depth-resample 109->64, remap
     each depth layer into the teacher block, write to zarr (64,Ht,Wt) chunks (8,32,32) uint16.
  6. write scaled labels + eroded labels + mask (crop the 7.91 pngs, NEAREST-resize by SCALE).
  7. QA: warp a low-res 2.4 texture into the crop and report mask agreement vs the 7.91 mask.

note: NOTHING here downsamples the 2.4 in-plane data — the teacher grid is ~1:1 with 2.4.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import numpy as np
import cv2
import zarr
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RBFInterpolator

from warp_from_dots import detect_dots, match_by_color

S3_BUCKET = "vesuvius-challenge-open-data"
DEPTH_OUT = 64
SRC_Z = 109
CHUNK = 128


def _aws(*args):
    subprocess.run(["aws", "s3"] + list(args) + ["--no-sign-request"], check=True)


def fit_warp(src_dots, dst_dots, ww=3600):
    """fit normalized 7.91(u,v) -> normalized 2.4(su,sv) thin-plate-spline from colored dots.
    source = 2.4 (unflipped native), target = 7.91 (flipped, == our 7.91 zarr frame).

    CRITICAL: this replicates warp_from_dots.py EXACTLY (pixel-space fit at ww=3600,
    smoothing=1.0, 4 frame-corner anchors) so the warped 2.4 DATA lands in the same place
    the committed 7.91 inklabels were baked from. an approximate (normalized / low-smoothing)
    fit introduces a ~24px (7.91-space) offset that misaligns teacher data vs teacher labels.
    returns fx, fy each taking an (N,2) array of normalized 7.91 (u,v) and returning (N,)
    normalized 2.4 coordinates."""
    sdots, (sh, sw) = detect_dots(src_dots)   # 2.4 source
    ddots, (dh, dw) = detect_dots(dst_dots)   # 7.91 target
    pairs = match_by_color(sdots, ddots)
    if len(pairs) < 3:
        raise SystemExit(f"need >=3 dot pairs, got {len(pairs)}")
    ssc, dsc = ww / sw, ww / dw
    H24h, H79h = int(round(sh * ssc)), int(round(dh * dsc))
    src, dst = [], []
    for xs, ys, xd, yd, _ in pairs:
        src.append([xs * ssc, ys * ssc])       # 2.4 source in the ww-frame
        dst.append([xd * dsc, yd * dsc])       # 7.91 target in the ww-frame
    for cx in (0, ww - 1):                      # frame-corner anchors (exactly as original)
        src += [[cx, 0], [cx, H24h - 1]]
        dst += [[cx, 0], [cx, H79h - 1]]
    src, dst = np.array(src, float), np.array(dst, float)
    fxp = RBFInterpolator(dst, src[:, 0], kernel="thin_plate_spline", smoothing=1.0)
    fyp = RBFInterpolator(dst, src[:, 1], kernel="thin_plate_spline", smoothing=1.0)
    print(f"[warp] fit {len(pairs)} dot pairs (+4 corners), ww={ww} smoothing=1.0 (matches bake)")

    def fx(uv):
        uv = np.atleast_2d(uv)
        return fxp(np.column_stack([uv[:, 0] * ww, uv[:, 1] * H79h])) / ww
    def fy(uv):
        uv = np.atleast_2d(uv)
        return fyp(np.column_stack([uv[:, 0] * ww, uv[:, 1] * H79h])) / H24h
    return fx, fy


def _resample_weights(z_in, z_out):
    Wm = np.zeros((z_out, z_in), dtype=np.float32)
    coords = np.linspace(0.0, z_in - 1, z_out)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, z_in - 1)
    frac = (coords - lo).astype(np.float32)
    for i in range(z_out):
        Wm[i, lo[i]] += 1.0 - frac[i]
        Wm[i, hi[i]] += frac[i]
    return Wm


def download_chunks(vol_l0, sx0, sx1, sy0, sy1, cache_dir, workers=8):
    """pull only the level-0 2.4 chunks intersecting the source bbox (resumable)."""
    os.makedirs(cache_dir, exist_ok=True)
    xc0, xc1 = sx0 // CHUNK, (sx1 - 1) // CHUNK
    yc0, yc1 = sy0 // CHUNK, (sy1 - 1) // CHUNK
    xc_names = [str(x) for x in range(xc0, xc1 + 1)]
    n_y = yc1 - yc0 + 1
    cbytes = SRC_Z * CHUNK * CHUNK
    print(f"[chunks] y {yc0}..{yc1} ({n_y}) x {xc0}..{xc1} ({len(xc_names)}) = "
          f"{n_y*len(xc_names)} files (~{n_y*len(xc_names)*cbytes/1e9:.1f}GB)", flush=True)

    def _one(yc):
        dst = os.path.join(cache_dir, str(yc))
        os.makedirs(dst, exist_ok=True)
        if all(os.path.exists(os.path.join(dst, xn)) for xn in xc_names):
            return
        includes = []
        for xn in xc_names:
            includes += ["--include", xn]
        _aws("cp", "--recursive", f"{vol_l0}/0/{yc}/", dst + os.sep, "--exclude", "*", *includes)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_one, range(yc0, yc1 + 1))):
            if (i + 1) % 8 == 0 or i + 1 == n_y:
                print(f"[chunks] {i+1}/{n_y} y-rows done", flush=True)


def _load_source_window(cache_dir, sy0, sy1, sx0, sx1):
    """assemble native 2.4 (SRC_Z, sy1-sy0, sx1-sx0) uint8 from cached chunk files.
    missing chunks == air (zeros), same convention as build_s3patch."""
    H, W = sy1 - sy0, sx1 - sx0
    strip = np.zeros((SRC_Z, H, W), dtype=np.uint8)
    cbytes = SRC_Z * CHUNK * CHUNK
    yc0, yc1 = sy0 // CHUNK, (sy1 - 1) // CHUNK
    xc0, xc1 = sx0 // CHUNK, (sx1 - 1) // CHUNK
    for yc in range(yc0, yc1 + 1):
        gy = yc * CHUNK
        ys, ye = max(gy, sy0), min(gy + CHUNK, sy1)
        for xc in range(xc0, xc1 + 1):
            gx = xc * CHUNK
            xs, xe = max(gx, sx0), min(gx + CHUNK, sx1)
            fp = os.path.join(cache_dir, str(yc), str(xc))
            if not os.path.exists(fp):
                continue
            raw = np.fromfile(fp, dtype=np.uint8)
            if raw.size != cbytes:
                continue
            ck = raw.reshape(SRC_Z, CHUNK, CHUNK)
            strip[:, ys - sy0:ye - sy0, xs - sx0:xe - sx0] = \
                ck[:, ys - gy:ye - gy, xs - gx:xe - gx]
    return strip


def build(args):
    fx, fy = fit_warp(args.src_dots, args.dst_dots)

    # 7.91 frame geometry from the existing zarr
    import json
    za = json.load(open(os.path.join(args.zarr_path, f"{args.ref_id}.zarr", ".zarray")))
    H79, W79 = int(za["shape"][1]), int(za["shape"][2])
    # 7.91 crop box (pixels)
    cx0, cx1 = int(args.x0f * W79), int(args.x1f * W79)
    cy0, cy1 = int(args.y0f * H79), int(args.y1f * H79)
    cw, ch = cx1 - cx0, cy1 - cy0
    scale = args.scale
    Wt, Ht = int(round(cw * scale)), int(round(ch * scale))
    print(f"[frame] 7.91 {H79}x{W79}  crop x[{cx0}:{cx1}] y[{cy0}:{cy1}] ({cw}x{ch})")
    print(f"[frame] teacher scale={scale} -> ({Ht}x{Wt})")

    # native 2.4 dims
    H24, W24 = args.src_h, args.src_w

    # COARSE 7.91->2.4 pixel map over the crop (upsampled per block later).
    # uniform grid (linspace, endpoints included) so per-block cv2.resize is exact.
    step = args.coarse_step
    ncols = max(2, int(np.ceil((Wt - 1) / step)) + 1)
    nrows = max(2, int(np.ceil((Ht - 1) / step)) + 1)
    gcols = np.linspace(0, Wt - 1, ncols).astype(np.float32)
    grows = np.linspace(0, Ht - 1, nrows).astype(np.float32)
    u_row = (cx0 + gcols / scale) / W79                # normalized 7.91 u per col
    v_col = (cy0 + grows / scale) / H79                # normalized 7.91 v per row
    uu, vv = np.meshgrid(u_row, v_col)                 # (nrows, ncols)
    U = np.column_stack([uu.ravel(), vv.ravel()])
    su = fx(U).reshape(len(grows), len(gcols)) * W24   # native 2.4 x px
    sv = fy(U).reshape(len(grows), len(gcols)) * H24   # native 2.4 y px
    # source bbox (with margin), clamped + chunk-snapped
    m = args.margin
    sx0 = max(0, int(np.floor(su.min())) - m); sx1 = min(W24, int(np.ceil(su.max())) + m)
    sy0 = max(0, int(np.floor(sv.min())) - m); sy1 = min(H24, int(np.ceil(sv.max())) + m)
    print(f"[source] 2.4 bbox x[{sx0}:{sx1}] y[{sy0}:{sy1}] ({sx1-sx0}x{sy1-sy0})")

    # download the needed 2.4 chunks
    cache_dir = os.path.join(args.tmp, f"{args.out_id}_chunks")
    vol_l0 = f"s3://{S3_BUCKET}/{args.seg}/{args.vol_subpath}"
    if not args.no_download:
        download_chunks(vol_l0, sx0, sx1, sy0, sy1, cache_dir, workers=args.workers)

    # full-res coordinate maps (native 2.4 abs px) via one bilinear upsample of the coarse
    # grid (plenty of RAM; ~2GB per map). exact because the coarse grid is uniform.
    MAPX = cv2.resize(su, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    MAPY = cv2.resize(sv, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    Wmat = _resample_weights(SRC_Z, DEPTH_OUT)
    out_zarr = os.path.join(args.zarr_path, f"{args.out_id}.zarr")
    store = zarr.open(out_zarr, mode="w", shape=(DEPTH_OUT, Ht, Wt),
                      chunks=(8, 32, 32), dtype="<u2", compressor=None, zarr_format=2)
    mask = np.zeros((Ht, Wt), dtype=np.uint8)

    block = args.block
    n_blocks = (Ht + block - 1) // block
    for bi, b0 in enumerate(range(0, Ht, block)):
        b1 = min(b0 + block, Ht)
        rows = b1 - b0
        mapx = MAPX[b0:b1]
        mapy = MAPY[b0:b1]

        # source window for this block
        wsx0 = max(0, int(np.floor(mapx.min())))
        wsx1 = min(W24, int(np.ceil(mapx.max())) + 1)
        wsy0 = max(0, int(np.floor(mapy.min())))
        wsy1 = min(H24, int(np.ceil(mapy.max())) + 1)
        strip = _load_source_window(cache_dir, wsy0, wsy1, wsx0, wsx1)  # (109, sh, sw)
        # depth resample 109 -> 64
        res = (Wmat @ strip.astype(np.float32).reshape(SRC_Z, -1)).reshape(
            DEPTH_OUT, wsy1 - wsy0, wsx1 - wsx0)
        # remap each depth layer into the teacher block
        rx = (mapx - wsx0).astype(np.float32)
        ry = (mapy - wsy0).astype(np.float32)
        out_block = np.empty((DEPTH_OUT, rows, Wt), dtype=np.uint16)
        for d in range(DEPTH_OUT):
            w = cv2.remap(res[d], rx, ry, cv2.INTER_LINEAR, borderValue=0)
            out_block[d] = np.clip(w, 0, 65535).astype(np.uint16)
        store[:, b0:b1, :] = out_block
        mask[b0:b1, :] = (out_block.max(axis=0) > 0).astype(np.uint8) * 255
        print(f"[block] {bi+1}/{n_blocks} rows[{b0}:{b1}] "
              f"src y[{wsy0}:{wsy1}] x[{wsx0}:{wsx1}]", flush=True)

    print(f"[zarr] wrote {out_zarr} ({DEPTH_OUT},{Ht},{Wt})  valid={float((mask>0).mean()):.3f}", flush=True)

    # ---- scaled labels + mask (crop the cleaned 7.91 pngs, NEAREST-resize by SCALE) ----
    def _scale_png(src_png, dst_png):
        im = cv2.imread(src_png, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"[warn] missing {src_png}, skipping")
            return None
        crop = im[cy0:cy1, cx0:cx1]
        out = cv2.resize(crop, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        out = (out > 127).astype(np.uint8) * 255
        cv2.imwrite(dst_png, out)
        print(f"[label] {os.path.basename(dst_png)} frac={float((out>0).mean()):.3f}")
        return out

    os.makedirs("inklabels", exist_ok=True)
    os.makedirs("eroded_inklabels", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    _scale_png(f"inklabels/{args.ref_id}.png", f"inklabels/{args.out_id}.png")
    _scale_png(f"eroded_inklabels/{args.ref_id}.png", f"eroded_inklabels/{args.out_id}.png")
    # teacher mask = intersection of the warped-signal papyrus mask and the scaled 7.91 mask
    m79 = _scale_png(f"masks/{args.ref_id}.png", os.path.join(args.tmp, f"_m79_{args.out_id}.png"))
    if m79 is not None:
        teacher_mask = ((mask > 0) & (m79 > 0)).astype(np.uint8) * 255
    else:
        teacher_mask = mask
    cv2.imwrite(f"masks/{args.out_id}.png", teacher_mask)
    print(f"[mask] masks/{args.out_id}.png valid={float((teacher_mask>0).mean()):.3f}")

    # ---- QA: mask agreement warped-2.4 vs 7.91 over the crop ----
    m79_crop = cv2.imread(f"masks/{args.ref_id}.png", cv2.IMREAD_GRAYSCALE)[cy0:cy1, cx0:cx1] > 0
    warp_small = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST) > 0
    agree = float((warp_small == m79_crop).mean())
    inter = float((warp_small & m79_crop).sum())
    union = float((warp_small | m79_crop).sum())
    mask_iou = inter / union if union else 0.0
    print(f"[QA] mask pixel-agreement={agree:.4f}  IoU={mask_iou:.4f}  (want agreement>0.98)")

    def _ncc(a, b, m=None):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        if m is not None:
            m = m.ravel().astype(bool); a = a[m]; b = b[m]
        if a.size < 100 or a.std() < 1e-6 or b.std() < 1e-6:
            return float("nan")
        a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    # ---- QA: structural alignment via depth-MEAN coarse NCC (robust to modality/depth diff) ----
    # read both depth-means in bounded y-blocks (never load the whole 70GB volume into RAM).
    try:
        ref = zarr.open(os.path.join(args.zarr_path, f"{args.ref_id}.zarr"), mode="r")
        ref_m = np.asarray(ref[:, cy0:cy1, cx0:cx1]).astype(np.float32).mean(0)   # (ch,cw) small
        tea_m = np.empty((Ht, Wt), dtype=np.float32)
        qb = 1024
        for yb in range(0, Ht, qb):
            ye = min(yb + qb, Ht)
            tea_m[yb:ye] = np.asarray(store[:, yb:ye, :]).astype(np.float32).mean(0)
        tea_m = cv2.resize(tea_m, (cw, ch), interpolation=cv2.INTER_AREA)
        a8 = cv2.GaussianBlur(tea_m, (0, 0), 8); b8 = cv2.GaussianBlur(ref_m, (0, 0), 8)
        print(f"[QA] structural NCC (depth-mean): full={_ncc(tea_m, ref_m):.3f}  "
              f"coarse(sig8)={_ncc(a8, b8):.3f}  (>0.3 = well aligned across modalities)", flush=True)
    except Exception as e:
        print(f"[QA] structural NCC skipped: {e}", flush=True)

    # ---- QA: DEFINITIVE label/data alignment. warp the 2.4 ink prediction with the SAME maps
    # as the data, compare to the scaled 7.91 labels via IoU. this confirms the teacher's
    # labels sit on the ink actually present in the teacher data (no shift search needed).
    if args.ink_tif and os.path.exists(args.ink_tif):
        try:
            import tifffile
            ink = tifffile.imread(args.ink_tif)
            if ink.ndim == 3:
                ink = ink[..., 0]
            ix0 = max(0, int(np.floor(MAPX.min()))); ix1 = min(W24, int(np.ceil(MAPX.max())) + 1)
            iy0 = max(0, int(np.floor(MAPY.min()))); iy1 = min(H24, int(np.ceil(MAPY.max())) + 1)
            inkw = cv2.remap(ink[iy0:iy1, ix0:ix1],
                             (MAPX - ix0).astype(np.float32), (MAPY - iy0).astype(np.float32),
                             cv2.INTER_LINEAR, borderValue=0) > args.ink_thr
            lab = cv2.imread(f"inklabels/{args.out_id}.png", cv2.IMREAD_GRAYSCALE) > 127
            i = float((inkw & lab).sum()); u = float((inkw | lab).sum())
            print(f"[QA] label/data IoU (warped-2.4-ink vs scaled-7.91-labels) = "
                  f"{(i/u if u else 0):.4f}  (higher = labels sit on the ink in the data)")
        except Exception as e:
            print(f"[QA] label/data IoU skipped: {e}")

    # QA overlay png (downscaled): R=warped-2.4 signal, G=7.91 mask
    ov = np.dstack([(warp_small * 255).astype(np.uint8),
                    (m79_crop * 255).astype(np.uint8),
                    np.zeros_like(warp_small, np.uint8)])
    ov = cv2.resize(ov, (1600, int(1600 * ch / cw)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"teacher_warp_overlay_{args.out_id}.png", ov)
    print(f"[QA] wrote teacher_warp_overlay_{args.out_id}.png")
    print("[done] teacher zarr build complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-id", default="20240304161941_t24", help="teacher zarr/label id")
    ap.add_argument("--ref-id", default="20240304161941", help="7.91 reference id (frame + labels)")
    ap.add_argument("--zarr-path", default=os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2"))
    ap.add_argument("--tmp", default="/vesuvius/_ves_tmp")
    ap.add_argument("--src-dots", default="warp_MARK_2p4_source_dots.png")
    ap.add_argument("--dst-dots", default="warp_MARK_7p9_target_dots.png")
    ap.add_argument("--ink-tif", default="/vesuvius/_ves_tmp/w023_ink_full.tif",
                    help="2.4 ink prediction tif (only for the label/data-alignment QA)")
    ap.add_argument("--ink-thr", type=int, default=99, help="threshold for the QA ink IoU")
    ap.add_argument("--seg", default="PHerc1667/segments/20240304161941-w023_20240304161941_flatboi")
    ap.add_argument("--vol-subpath",
                    default="surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0")
    ap.add_argument("--src-h", type=int, default=41860)
    ap.add_argument("--src-w", type=int, default=102360)
    # 7.91 crop fractions: RIGHT 30% x, TOP 40% y (ink-rich; maps to 2.4 right30/top40)
    ap.add_argument("--x0f", type=float, default=0.70)
    ap.add_argument("--x1f", type=float, default=1.0)
    ap.add_argument("--y0f", type=float, default=0.0)
    ap.add_argument("--y1f", type=float, default=0.40)
    ap.add_argument("--scale", type=float, default=106.0 / 32.0)   # 3.3125
    ap.add_argument("--coarse-step", type=int, default=32, help="teacher px between warp samples")
    ap.add_argument("--margin", type=int, default=64, help="source bbox padding (native 2.4 px)")
    ap.add_argument("--block", type=int, default=256, help="teacher rows per streamed block")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-download", action="store_true", help="reuse cached chunks")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
