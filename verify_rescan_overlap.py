"""verify_rescan_overlap.py -- low vs high energy midslices, dice, and label placement checks.

per segment writes to output/rescan_verify/<seg>/:
    midslice_overlay.png    red = 137 keV, green = 78 keV (yellow where both agree)
    label_check.png         ink-rich windows: researchers' 2.4 um render | our 78 keV | our 137 keV,
                            each with the researcher label outlined in cyan
usage: python3 verify_rescan_overlap.py 20231012184424 20231007101619
"""
import json
import os
import sys

import cv2
import numpy as np
import zarr

ROOT = os.path.dirname(os.path.abspath(__file__))
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments"
SURFACE = "surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2"
STEP = 9.362 / 2.4


def pearson(a, b):
    return float(np.corrcoef(a.astype(np.float64), b.astype(np.float64))[0, 1])


def norm8(x, m):
    lo, hi = np.percentile(x[m], (1, 99))
    return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * 255 * m).astype(np.uint8)


def surface_window(segment, r0, c0, h, w):
    """researchers' 2.4 um surface render (level 2), layer nearest our midslice, in our frame px."""
    src = zarr.open(f"{BUCKET}/{segment}/{SURFACE}", mode="r")
    rows = ((np.arange(r0, r0 + h) + 0.5) * STEP) / 4 - 0.5
    cols = ((np.arange(c0, c0 + w) + 0.5) * STEP) / 4 - 0.5
    a0, a1 = int(np.floor(rows[0])), int(np.ceil(rows[-1])) + 2
    b0, b1 = int(np.floor(cols[0])), int(np.ceil(cols[-1])) + 2
    mid = src.shape[0] // 2 + 2          # our layer 14 sits ~2 surface voxels above the mesh
    patch = np.asarray(src[mid, a0:a1, b0:b1], np.float32)
    gx, gy = np.meshgrid((cols - b0).astype(np.float32), (rows - a0).astype(np.float32))
    return cv2.remap(patch, gx, gy, cv2.INTER_LINEAR)


def verify(segment):
    frame = json.load(open(os.path.join(ROOT, "_ves_tmp", "rescan", segment, "frame.json")))
    ids = {int(k): v for k, v in frame["ids"].items()}
    y0, _, x0, _ = frame["crop_yx"]
    low = zarr.open(os.path.join(ROOT, "ves_zarrs2", f"{ids[78]}.zarr"), mode="r")
    high = zarr.open(os.path.join(ROOT, "ves_zarrs2", f"{ids[137]}.zarr"), mode="r")
    out_dir = os.path.join(ROOT, "output", "rescan_verify", segment)
    os.makedirs(out_dir, exist_ok=True)
    mid = low.shape[0] // 2
    a = np.asarray(low[mid]).astype(np.float32)
    b = np.asarray(high[mid]).astype(np.float32)
    ma, mb = a > 0, b > 0
    both = ma & mb
    print(f"== {segment}: shapes {low.shape} {high.shape} dtype {low.dtype}")
    print(f"   midslice mask dice {2 * both.sum() / (ma.sum() + mb.sum()):.4f} | "
          f"78-in-137 {both.sum() / ma.sum():.4f} | 137-in-78 {both.sum() / mb.sum():.4f}")
    rs = [pearson(np.asarray(low[z])[both], np.asarray(high[z])[both]) for z in range(low.shape[0])]
    print("   per-layer pearson r (78 vs 137):", " ".join(f"{r:.2f}" for r in rs))
    ta, tb = a[both] > np.median(a[both]), b[both] > np.median(b[both])
    print(f"   midslice dice of above-median papyrus: {2 * (ta & tb).sum() / (ta.sum() + tb.sum()):.4f}")
    overlay = np.zeros(a.shape + (3,), np.uint8)
    overlay[..., 1] = norm8(a, ma)
    overlay[..., 2] = norm8(b, mb)
    scale = min(1.0, 6000 / overlay.shape[1])
    cv2.imwrite(os.path.join(out_dir, "midslice_overlay.png"),
                cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))

    label = cv2.imread(os.path.join(ROOT, "researcher_inklabels", f"{ids[137]}.png"), 0) > 0
    print(f"   label positive fraction in footprint {float((label & both).sum() / both.sum()):.4f}; "
          f"label pixels outside footprint {int((label & ~both).sum())}")
    win = 384
    density = cv2.boxFilter((label & both).astype(np.float32), -1, (win, win), normalize=True, anchor=(0, 0),
                            borderType=cv2.BORDER_CONSTANT)
    density[max(0, density.shape[0] - win):, :] = 0
    density[:, max(0, density.shape[1] - win):] = 0
    rows = []
    for _ in range(3):
        r, c = np.unravel_index(np.argmax(density), density.shape)
        if density[r, c] <= 0:
            break
        density[max(0, r - win):r + win, max(0, c - win):c + win] = 0
        ref = surface_window(segment, y0 + r, x0 + c, win, win)
        sl = (slice(r, r + win), slice(c, c + win))
        m = both[sl]
        print(f"   window r{r} c{c}: pearson(researcher render, our 78 keV) {pearson(ref[m], a[sl][m]):.3f} | "
              f"(researcher render, our 137 keV) {pearson(ref[m], b[sl][m]):.3f}")
        contour = cv2.morphologyEx(label[sl].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        panels = []
        for img in (ref, a[sl], b[sl]):
            g = norm8(img, np.ones_like(m))
            rgb = np.dstack([g, g, g])
            rgb[contour] = (255, 255, 0)
            panels.append(rgb)
        rows.append(np.hstack([np.pad(p, ((4, 4), (4, 4), (0, 0))) for p in panels]))
    cv2.imwrite(os.path.join(out_dir, "label_check.png"), np.vstack(rows))
    print(f"   wrote {out_dir}")


if __name__ == "__main__":
    for seg in sys.argv[1:]:
        verify(seg)
