"""CPU audit: does a scroll's ink contrast follow its detected surface, or a fixed depth band?

For each scroll, random ink-bearing tiles are read at all depths. Ink pixels (inklabel) are compared
with nearby papyrus (24-64 px from ink) by Cohen's d at every absolute depth and at every offset from
the detected surface. The training window is the 8 slices surface-3 .. surface+4 (flat 13 -> 10-17).

    python3 old/surface_window_audit.py 20250628074500 20260930144758 ...
    python3 old/surface_window_audit.py --json _ves_tmp/surface_window_audit.json <ids>
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
WINDOW = (-3, 5)  # slices surface-3 .. surface+4
OFFSETS = np.arange(-8, 9)


def _cohen(a, b):
    pooled = np.sqrt(0.5 * (a.var() + b.var())) + 1e-6
    return float((a.mean() - b.mean()) / pooled)


def audit(sid, tiles=48, size=256, seed=0, label_dir="dilated_inklabels"):
    volume = zarr.open(str(ROOT / "ves_zarrs2" / f"{sid}.zarr"), mode="r")
    depth_n, height, width = (int(v) for v in volume.shape)
    mask = cv2.imread(str(ROOT / "masks" / f"{sid}.png"), 0)[:height, :width] > 0
    label = cv2.imread(str(ROOT / label_dir / f"{sid}.png"), 0)[:height, :width] >= 192
    surface = np.load(ROOT / "surface_labels" / str(sid) / "depth.npy", mmap_mode="r")
    ink = label & mask
    near = cv2.dilate(label.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (129, 129))) > 0
    gap = cv2.dilate(label.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))) > 0
    papyrus = near & ~gap & mask
    ys, xs = np.nonzero(ink[::16, ::16])
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ys), size=min(tiles, len(ys)), replace=False)
    corners = [(max(0, min(height - size, ys[i] * 16 - size // 2)), max(0, min(width - size, xs[i] * 16 - size // 2)))
               for i in picks]

    def read(corner):
        y, x = corner
        return y, x, np.asarray(volume[:, y:y + size, x:x + size], dtype=np.float32)

    with ThreadPoolExecutor(8) as pool:
        blocks = list(pool.map(read, corners))
    ink_cols, pap_cols, ink_surf, pap_surf, all_cols = [], [], [], [], []
    for y, x, block in blocks:
        window = (slice(y, y + size), slice(x, x + size))
        s = np.asarray(surface[window]).astype(np.int16)
        for target, cols, surf in ((ink[window], ink_cols, ink_surf), (papyrus[window], pap_cols, pap_surf)):
            sel = target & (s != 255)
            cols.append(block[:, sel].T)
            surf.append(s[sel])
        all_cols.append(block[:, mask[window]].T[::8])
    ink_cols, pap_cols = np.concatenate(ink_cols), np.concatenate(pap_cols)
    ink_surf, pap_surf = np.concatenate(ink_surf), np.concatenate(pap_surf)
    profile = np.concatenate(all_cols).mean(axis=0)
    absolute = np.array([_cohen(ink_cols[:, d], pap_cols[:, d]) for d in range(depth_n)])

    def relative(cols, surf):
        index = np.clip(surf[:, None] + OFFSETS[None, :], 0, depth_n - 1)
        values = np.take_along_axis(cols, index, axis=1)
        inside = (surf[:, None] + OFFSETS[None, :] >= 0) & (surf[:, None] + OFFSETS[None, :] < depth_n)
        return np.where(inside, values, np.nan)

    ink_rel, pap_rel = relative(ink_cols, ink_surf), relative(pap_cols, pap_surf)
    rel = np.array([_cohen(ink_rel[:, k][np.isfinite(ink_rel[:, k])], pap_rel[:, k][np.isfinite(pap_rel[:, k])])
                    for k in range(len(OFFSETS))])
    full_surface = np.asarray(surface)[:height, :width][mask]
    valid = full_surface[full_surface != 255].astype(np.float32)
    rel_window = [OFFSETS.tolist().index(k) for k in range(*WINDOW)]
    detected_score = float(np.abs(rel[rel_window]).sum())
    flat_scores = {c: float(np.abs(absolute[c + WINDOW[0]:c + WINDOW[1]]).sum())
                   for c in range(-WINDOW[0], depth_n - WINDOW[1] + 1)}
    best_flat = max(flat_scores, key=flat_scores.get)
    # sheet centre: intensity-weighted centre of the densest 8 slices
    dense = int(np.argmax(np.convolve(profile, np.ones(8), mode="valid")))
    centre = float((np.arange(dense, dense + 8) * profile[dense:dense + 8]).sum() / profile[dense:dense + 8].sum())
    return {
        "scroll_id": int(sid), "depth": depth_n, "tiles": len(blocks),
        "ink_px": int(len(ink_cols)), "papyrus_px": int(len(pap_cols)),
        "surface_median": float(np.median(valid)) if valid.size else None,
        "surface_iqr": [float(np.percentile(valid, 25)), float(np.percentile(valid, 75))] if valid.size else None,
        "surface_invalid_frac": float(1 - valid.size / max(full_surface.size, 1)),
        "profile": [round(float(v), 1) for v in profile],
        "absolute_d": [round(float(v), 3) for v in absolute],
        "relative_d": {int(k): round(float(v), 3) for k, v in zip(OFFSETS, rel)},
        "detected_window_score": round(detected_score, 3),
        "best_flat_depth": int(best_flat), "best_flat_score": round(flat_scores[best_flat], 3),
        "flat13_score": round(flat_scores.get(13, float("nan")), 3),
        "sheet_centre": round(centre, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+", type=int)
    parser.add_argument("--json")
    parser.add_argument("--tiles", type=int, default=48)
    args = parser.parse_args()
    os.chdir(ROOT)
    results = []
    for sid in args.ids:
        r = audit(sid, tiles=args.tiles)
        results.append(r)
        print(f"{sid}: surface median {r['surface_median']} IQR {r['surface_iqr']} invalid {r['surface_invalid_frac']:.2f} "
              f"| sheet centre {r['sheet_centre']} | detected-window |d| {r['detected_window_score']} "
              f"| best flat {r['best_flat_depth']} |d| {r['best_flat_score']} (flat13 {r['flat13_score']})", flush=True)
        print(f"    absolute d: {r['absolute_d']}", flush=True)
        print(f"    relative d: {r['relative_d']}", flush=True)
        print(f"    profile:    {r['profile']}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
