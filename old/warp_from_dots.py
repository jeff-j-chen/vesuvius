"""warp_from_dots.py — align 2.4um <-> 7.91um scroll4 w023 sheets using COLORED-DOT anchors
the user marks on the two exported grayscale PNGs.

workflow:
  1. warp_MARK_2p4_source.png (2.4, SOURCE) and warp_MARK_7p9_target.png (7.91-flipped, TARGET)
     were exported at the SAME width (1600). open each in an image editor.
  2. mark corresponding features with dots. use the SAME COLOR for a feature in BOTH images
     (e.g. a red dot on the top-left wave in both; a green dot on the mid-left artifact in both).
     use distinct, saturated colors per correspondence. a few px wide is plenty.
     ANCHOR TRIPLING: a color may be reused UP TO 3x per image -> 36 anchors from a 12-color
     palette. when a color appears multiple times, dots pair by x-order (leftmost<->leftmost, ...
     rightmost<->rightmost); safe because the warp is mild and preserves x-order.
  3. save as warp_MARK_2p4_source_dots.png and warp_MARK_7p9_target_dots.png (or pass paths).
  4. run this script -> it detects each colored dot, pairs source<->target by color (+x-order),
     fits a thin-plate spline through the pairs (+ corners), warps the 2.4 ink labels into the
     7.91 frame, and writes QA overlays.

detection: a pixel is 'marked' if it is COLORED (HSV saturation & value high) - the underlying
image is grayscale so any saturated pixel is a user dot. dots are clustered by connected
components; each dot's mean color snaps to the nearest palette entry; source and target dots are
matched by palette name, and (when a color is used multiple times) by left-to-right x-order.
"""
import argparse
import os
import numpy as np
import cv2
import tifffile
from scipy.interpolate import RBFInterpolator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

TMP = r'C:\Users\ChenJeff\Documents\_ves_tmp'
DOC = r'C:\Users\ChenJeff\Documents'
W = 1600  # export/working width (must match the exported PNG width)


def detect_dots(path, sat_thr=80, val_thr=60, min_area=6):
    """find colored dots on a mostly-grayscale image. returns list of (hue, x, y, area)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
# reference palette (BGR). 12 well-separated colors -> snap each detected dot to nearest.
# mark a feature with the SAME palette color in BOTH images. distinct color per correspondence.
PALETTE_BGR = {
    "red":      (0, 0, 255),
    "green":    (0, 255, 0),
    "blue":     (255, 0, 0),
    "yellow":   (0, 255, 255),
    "magenta":  (255, 0, 255),
    "cyan":     (255, 255, 0),
    "orange":   (0, 128, 255),
    "purple":   (128, 0, 128),
    "pink":     (128, 128, 255),
    "teal":     (128, 128, 0),
    "brown":    (0, 75, 150),
    "violet":   (255, 0, 128),
}


def _lab(bgr):
    a = np.uint8([[list(bgr)]])
    return cv2.cvtColor(a, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)


_PAL_NAMES = list(PALETTE_BGR)
_PAL_LAB = np.array([_lab(PALETTE_BGR[n]) for n in _PAL_NAMES], np.float32)


def _nearest_palette(mean_bgr):
    """snap a dot's mean color to the nearest palette entry in Lab; return (name, dist)."""
    lab = _lab(tuple(int(round(c)) for c in mean_bgr))
    d = np.linalg.norm(_PAL_LAB - lab, axis=1)
    i = int(np.argmin(d))
    return _PAL_NAMES[i], float(d[i])


def detect_dots(path, sat_thr=80, val_thr=60, min_area=6):
    """find colored dots on a mostly-grayscale image. returns list of
    (name, x, y, area, palette_dist) with each dot snapped to a palette color."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    colored = ((s >= sat_thr) & (v >= val_thr)).astype(np.uint8)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lbl, st, ct = cv2.connectedComponentsWithStats(colored, 8)
    dots = []
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < min_area:
            continue
        mask = lbl == c
        mean_bgr = img[mask].astype(np.float32).mean(axis=0)   # blob mean color
        name, dist = _nearest_palette(mean_bgr)
        dots.append((name, float(ct[c][0]), float(ct[c][1]),
                     int(st[c, cv2.CC_STAT_AREA]), dist))
    return dots, img.shape[:2]


def match_by_color(src_dots, dst_dots, max_dist=40.0):
    """pair source and target dots by palette color, allowing UP TO THREE dots per color.

    tripling the anchor budget: a color may be used up to 3x in each image -> 36 anchors from a
    12-color palette. when a color has multiple dots in both images they are paired by x-order
    (leftmost <-> leftmost, ... rightmost <-> rightmost -- safe because the warp is mild enough
    not to reorder features in x). a color with one dot in each image pairs directly. if the
    per-color counts differ between images, only the largest (count-matched) dots are paired
    left-to-right and a warning is printed.

    returns (src_x, src_y, dst_x, dst_y, label) where label is e.g. 'blue' (single) or, when a
    color is used multiple times, 'blue.1'/'blue.2'/'blue.3' ordered left->right by x.
    """
    MAX_PER_COLOR = 3

    def group(dots):
        by = {}
        for name, x, y, area, dist in dots:
            if dist > max_dist:
                continue
            by.setdefault(name, []).append((x, y, area))
        # keep the N largest blobs per color, then order them left->right by x
        for name in by:
            top = sorted(by[name], key=lambda t: t[2], reverse=True)[:MAX_PER_COLOR]
            by[name] = sorted(top, key=lambda t: t[0])   # left-to-right
        return by

    s_by, d_by = group(src_dots), group(dst_dots)
    pairs = []
    for name in sorted(s_by):
        if name not in d_by:
            continue
        s_list, d_list = s_by[name], d_by[name]
        n = min(len(s_list), len(d_list))
        if len(s_list) != len(d_list):
            print(f"[warn] color '{name}' has {len(s_list)} src vs {len(d_list)} dst dots; "
                  f"pairing the {n} largest by x-order")
        # single dot -> bare name; multiple -> left-to-right numbered suffix (.1/.2/.3)
        suffixes = ("", ) if n == 1 else tuple(f".{i+1}" for i in range(n))
        for i in range(n):
            xs, ys, _ = s_list[i]; xd, yd, _ = d_list[i]
            pairs.append((xs, ys, xd, yd, name + suffixes[i]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="warp 2.4->7.91 via colored-dot correspondences")
    ap.add_argument("--id", default="w023",
                    help="tag for output map files (<id>_dotwarp_map{x,y}.npy in the tmp dir)")
    ap.add_argument("--src-dots", default=os.path.join(DOC, "warp_MARK_2p4_source_dots.png"),
                    help="2.4um SOURCE image with colored dots")
    ap.add_argument("--dst-dots", default=os.path.join(DOC, "warp_MARK_7p9_target_dots.png"),
                    help="7.91um TARGET image with colored dots (same colors on matching features)")
    ap.add_argument("--src-tex", default=os.path.join(DOC, "warp_MARK_2p4_source.png"),
                    help="clean 2.4um SOURCE texture PNG (grayscale, same size as --src-dots)")
    ap.add_argument("--dst-tex", default=os.path.join(DOC, "warp_MARK_7p9_target.png"),
                    help="clean 7.91um TARGET texture PNG (grayscale, same size as --dst-dots)")
    ap.add_argument("--ink-tif", default=os.path.join(TMP, "w023_ink_full.tif"),
                    help="optional 2.4um ink prediction tif for the QA ink-on-target overlay")
    ap.add_argument("--ink-thr", type=int, default=60, help="threshold for the QA ink overlay")
    ap.add_argument("--ww", type=int, default=3600, help="hi-res warp width (target frame)")
    ap.add_argument("--out", default=DOC)
    ap.add_argument("--tmp", default=TMP, help="where to write the *_dotwarp_map{x,y}.npy files")
    args = ap.parse_args()

    sdots, (sh, sw) = detect_dots(args.src_dots)
    ddots, (dh, dw) = detect_dots(args.dst_dots)
    print(f"[dots] source={len(sdots)} ({sw}x{sh})  target={len(ddots)} ({dw}x{dh})")
    pairs = match_by_color(sdots, ddots)
    print(f"[match] {len(pairs)} color-matched pairs")
    for xs, ys, xd, yd, name in pairs:
        print(f"   {name:11s} src({xs:.0f},{ys:.0f}) -> dst({xd:.0f},{yd:.0f})")
    if len(pairs) < 3:
        print("[!] need >=3 matched dots for a stable warp. mark more corresponding colors.")
        return

    # load the grayscale textures (same pixel space as the marked images)
    tex24 = cv2.imread(args.src_tex, cv2.IMREAD_GRAYSCALE)
    tex79 = cv2.imread(args.dst_tex, cv2.IMREAD_GRAYSCALE)
    if tex24 is None or tex79 is None:
        raise FileNotFoundError(f"texture not found: {args.src_tex} / {args.dst_tex}")
    H24, H79 = tex24.shape[0], tex79.shape[0]

    # scale dot coords into the hi-res warp frame. source and target are scaled by their OWN
    # width (they were each exported at their marked-image width), so mixed widths still work.
    ssc, dsc = args.ww / sw, args.ww / dw
    H24h, H79h = int(round(H24 * ssc)), int(round(H79 * dsc))
    src, dst = [], []
    for xs, ys, xd, yd, _ in pairs:
        src.append([xs * ssc, ys * ssc]); dst.append([xd * dsc, yd * dsc])
    for cx in (0, args.ww - 1):            # corner anchors for edge stability
        src += [[cx, 0], [cx, H24h - 1]]; dst += [[cx, 0], [cx, H79h - 1]]
    src, dst = np.array(src, float), np.array(dst, float)

    fx = RBFInterpolator(dst, src[:, 0], kernel="thin_plate_spline", smoothing=1.0)
    fy = RBFInterpolator(dst, src[:, 1], kernel="thin_plate_spline", smoothing=1.0)
    gy, gx = np.mgrid[0:H79h, 0:args.ww]
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    mapx = fx(pts).reshape(H79h, args.ww).astype(np.float32)
    mapy = fy(pts).reshape(H79h, args.ww).astype(np.float32)

    tex24h = cv2.resize(tex24, (args.ww, H24h))
    tex79h = cv2.resize(tex79, (args.ww, H79h))
    tex24_w = cv2.remap(tex24h, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
    Image.fromarray(np.dstack([tex79h, tex24_w, np.zeros_like(tex79h)])).save(
        os.path.join(args.out, f"warp_dots_overlay_{args.id}.png"))

    # optional QA: warp the 2.4 ink prediction the same way and overlay on the target
    if args.ink_tif and os.path.exists(args.ink_tif):
        ink = tifffile.imread(args.ink_tif)
        if ink.ndim == 3:
            ink = ink[..., 0]
        ink = cv2.resize(ink, (args.ww, H24h), interpolation=cv2.INTER_AREA)
        inkb = (ink > args.ink_thr).astype(np.uint8) * 255
        ink_w = cv2.remap(inkb, mapx, mapy, cv2.INTER_NEAREST, borderValue=0)
        over = np.dstack([tex79h, (tex79h * 0.35).astype(np.uint8),
                          np.maximum((tex79h * 0.35).astype(np.uint8), ink_w)])
        Image.fromarray(over).save(os.path.join(args.out, f"warp_dots_ink_on_79_{args.id}.png"))

    os.makedirs(args.tmp, exist_ok=True)
    np.save(os.path.join(args.tmp, f"{args.id}_dotwarp_mapx.npy"), mapx)
    np.save(os.path.join(args.tmp, f"{args.id}_dotwarp_mapy.npy"), mapy)
    print(f"[done] wrote warp_dots_overlay_{args.id}.png (+ maps {args.id}_dotwarp_map{{x,y}}.npy in {args.tmp})")


if __name__ == "__main__":
    main()
