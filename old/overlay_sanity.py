"""overlay_2p4_9um.py -- alignment sanity check overlaying a HI-res surface midslice
(RED, half opacity) on top of a LO-res surface midslice (GREEN). overlap -> YELLOW.

both midslices sample the SAME physical surface at different resolutions, so the hi-res
image is resized to the lo-res frame before compositing (same extent -> straight resize
aligns them). prints a normalized cross-correlation over valid pixels as a quantitative
alignment score (higher = better aligned).

generic usage (any hi/lo pair, e.g. w059 1.1um over 9.4um):
  python overlay_2p4_9um.py --hi <hi_midslice.png> --lo <lo_midslice.png> \
      --mask <lo_mask.png> --hi-label 1.1um --lo-label 9.4um \
      --out-prefix overlay_w059_1p1_9um --out-dir C:/Users/ChenJeff/Documents/_ves_tmp

back-compat (original PHerc0139 w044 2.4um vs 9.3um defaults):
  python overlay_2p4_9um.py
  python overlay_2p4_9um.py --img24 <2.4 midslice> --img9 <9.3 midslice> --mask <mask.png>

outputs (into --out-dir):
  <out-prefix>_full.png   full-frame composite (downscaled to --max-view)
  <out-prefix>_crop.png   100% zoom of a central sub-region (per-pixel alignment check)
"""
from __future__ import annotations
import argparse, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEF_DIR  = r"C:\Users\ChenJeff\Documents\_ves_tmp\p0139_w044"
DEF_IMG24 = os.path.join(DEF_DIR, "w044_2p4um_L2_midslice.png")
DEF_IMG9  = os.path.join(DEF_DIR, "w044_9um_midslice.png")
DEF_MASK  = os.path.join(DEF_DIR, "w044_9um_midslice_mask.png")


def _norm(a: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """robust 1-99 percentile stretch to [0,1]; ignores zeros / out-of-mask."""
    a = a.astype(np.float32)
    valid = a > 0 if mask is None else (mask > 0)
    if valid.sum() == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(a[valid], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0, 1)
    out[~valid] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="hi-res(red) over lo-res(green) alignment overlay")
    # generic hi/lo args (preferred); fall back to the legacy 2.4/9.3 defaults
    ap.add_argument("--hi", default=None, help="HI-res middle-slice PNG (rendered RED)")
    ap.add_argument("--lo", default=None, help="LO-res middle-slice PNG (rendered GREEN)")
    ap.add_argument("--hi-label", default="2.4um", help="label for the hi-res (red) layer")
    ap.add_argument("--lo-label", default="9.3um", help="label for the lo-res (green) layer")
    ap.add_argument("--out-prefix", default="overlay_2p4_9um", help="output filename prefix")
    # legacy aliases (kept so old invocations still work)
    ap.add_argument("--img24", default=None, help="[legacy] alias for --hi")
    ap.add_argument("--img9",  default=None, help="[legacy] alias for --lo")
    ap.add_argument("--mask",  default=None, help="valid-region mask in the lo-res frame (optional)")
    ap.add_argument("--alpha", type=float, default=0.5, help="opacity of the red (hi-res) layer")
    ap.add_argument("--out-dir", default=DEF_DIR)
    ap.add_argument("--max-view", type=int, default=2400, help="max long-edge px for the full overview")
    ap.add_argument("--crop-frac", type=float, default=0.18, help="central crop size as a fraction of the frame")
    args = ap.parse_args()

    # resolve hi/lo/mask from generic args, then legacy aliases, then defaults
    hi_path = args.hi or args.img24 or DEF_IMG24
    lo_path = args.lo or args.img9 or DEF_IMG9
    mask_path = args.mask if args.mask is not None else DEF_MASK

    print(f"[overlay] HI ({args.hi_label}, red):  {hi_path}")
    print(f"[overlay] LO ({args.lo_label}, green): {lo_path}")
    img_hi = np.asarray(Image.open(hi_path).convert("L"))
    img_lo = np.asarray(Image.open(lo_path).convert("L"))
    H, W = img_lo.shape
    print(f"[overlay] LO frame {W}x{H}; HI {img_hi.shape[1]}x{img_hi.shape[0]} -> resizing HI to LO frame")

    # resize HI to the LO frame (same surface extent, different sampling density)
    img_hi_r = np.asarray(Image.fromarray(img_hi).resize((W, H), Image.BILINEAR))

    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape != (H, W):
            mask = np.asarray(Image.fromarray(mask).resize((W, H), Image.NEAREST))
        print(f"[overlay] mask valid fraction {(mask > 0).mean():.3f}")

    g = _norm(img_lo,   mask)     # green = lo-res
    r = _norm(img_hi_r, mask)     # red   = hi-res

    # composite: full green base (lo-res), red (hi-res) painted on top at alpha.
    # where both are bright -> red+green = YELLOW, so agreement is visible directly.
    a = float(args.alpha)
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0] = a * r                 # red   = hi-res at alpha opacity
    rgb[..., 1] = g                     # green = lo-res base
    if mask is not None:
        rgb[mask == 0] = 0.0
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)

    # full overview (downscaled)
    view = Image.fromarray(rgb_u8)
    scale = min(1.0, args.max_view / max(H, W))
    if scale < 1.0:
        view = view.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
    full_path = os.path.join(args.out_dir, f"{args.out_prefix}_full.png")
    view.save(full_path)
    print(f"[overlay] wrote {full_path}  ({view.size[0]}x{view.size[1]})")

    # central 100%-zoom crop for per-pixel alignment inspection
    ch, cw = int(H * args.crop_frac), int(W * args.crop_frac)
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    crop = Image.fromarray(rgb_u8[y0:y0 + ch, x0:x0 + cw])
    crop_path = os.path.join(args.out_dir, f"{args.out_prefix}_crop.png")
    crop.save(crop_path)
    print(f"[overlay] wrote {crop_path}  (100% zoom, {cw}x{ch} @ center)")

    # quantitative alignment score: normalized cross-correlation over valid pixels
    v = (mask > 0) if mask is not None else ((img_lo > 0) & (img_hi_r > 0))
    if v.sum() > 0:
        rv, gv = r[v] - r[v].mean(), g[v] - g[v].mean()
        denom = (np.linalg.norm(rv) * np.linalg.norm(gv))
        ncc = float((rv * gv).sum() / denom) if denom > 0 else 0.0
        print(f"[overlay] normalized cross-correlation ({args.hi_label} vs {args.lo_label}, "
              f"valid px): {ncc:+.3f}  (higher = better aligned)")


if __name__ == "__main__":
    main()
