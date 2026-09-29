"""step 2: midslice sanity check and registration of each 2.4 um render onto its native low-resolution render.

per pair: the low midslice (level 0) and the high midslice at level 2 (the 2.4 um render downscaled 4x by
the pyramid) are resampled onto the low grid, binarised to papyrus (Otsu inside the rendered footprint)
and compared by Dice and NCC three ways: as rendered (top-left anchored), after a phase-correlation
shift, and after a SIFT + RANSAC similarity fit. the best transform is saved as the map from low-native
pixels to high level-0 pixels, plus a depth offset from the two layer-intensity profiles.

surface-volume chunks span the full depth, so one slice costs streaming the whole level: ~0.1-1 GB for
the low render and ~1-4 GB for the level-2 high render per segment. nothing but the slices and the
report is written to disk.

    python crossres/sanity_midslice.py                     # every pair
    python crossres/sanity_midslice.py --names w035 p841   # a subset
    python crossres/sanity_midslice.py --high-level 3      # quicker, coarser (8x)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import PAIRS, url  # noqa: E402

OUT = Path(__file__).resolve().parent / "sanity"


def _stream_slice(array, z: int, rows: int = 128) -> np.ndarray:
    """one depth slice, read in row strips so only the slice is kept in memory."""
    height, width = int(array.shape[1]), int(array.shape[2])
    out = np.zeros((height, width), dtype=np.float32)
    for y0 in range(0, height, rows):
        out[y0:y0 + rows] = np.asarray(array[z, y0:y0 + rows, :], dtype=np.float32)
    return out


def _to_u8(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = image[valid]
    if values.size == 0:
        return np.zeros(image.shape, np.uint8)
    lo, hi = np.percentile(values, [0.5, 99.5])
    return (np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8) * valid


def _papyrus(image_u8: np.ndarray, valid: np.ndarray) -> np.ndarray:
    threshold, _ = cv2.threshold(image_u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (image_u8 > threshold) & valid


def _metrics(low_u8, low_valid, high_u8, high_valid) -> dict:
    both = low_valid & high_valid
    if both.sum() < 1000:
        return {"dice": 0.0, "ncc": 0.0, "overlap": int(both.sum())}
    a, b = _papyrus(low_u8, both), _papyrus(high_u8, both)
    dice = 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    x, y = low_u8[both].astype(np.float64), high_u8[both].astype(np.float64)
    ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))
    return {"dice": float(dice), "ncc": ncc, "overlap": int(both.sum())}


def _fit(canvas_shape, image, valid, matrix):
    warped = cv2.warpAffine(image, matrix, canvas_shape[::-1], flags=cv2.INTER_LINEAR)
    warped_valid = cv2.warpAffine(valid.astype(np.uint8), matrix, canvas_shape[::-1], flags=cv2.INTER_NEAREST) > 0
    return warped, warped_valid


def _sift_similarity(low_u8, high_u8, max_side: int):
    scale = min(1.0, max_side / max(low_u8.shape))
    small = [cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for image in (high_u8, low_u8)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small = [clahe.apply(image) for image in small]
    sift = cv2.SIFT_create(nfeatures=20000)
    (kp_h, des_h), (kp_l, des_l) = (sift.detectAndCompute(image, None) for image in small)
    if des_h is None or des_l is None or len(kp_h) < 20 or len(kp_l) < 20:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_h, des_l, k=2)
    good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp_h[m.queryIdx].pt for m in good]) / scale
    dst = np.float32([kp_l[m.trainIdx].pt for m in good]) / scale
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return matrix, int(inliers.sum()) if inliers is not None else 0


def _depth_profile(array, box, valid_level0=None) -> np.ndarray:
    y0, y1, x0, x1 = box
    block = np.asarray(array[:, y0:y1, x0:x1], dtype=np.float32)
    mask = block.max(axis=0) > 0 if valid_level0 is None else valid_level0
    return np.array([layer[mask].mean() if mask.any() else 0.0 for layer in block])


def _depth_offset(low_profile, high_profile, low_um, high_um) -> float:
    """shift (in low slices) that best aligns the high profile, pooled onto low slices, with the low one."""
    def pooled(offset):
        centres = (np.arange(len(low_profile)) - (len(low_profile) - 1) / 2 + offset) * low_um
        positions = centres / high_um + (len(high_profile) - 1) / 2
        return np.interp(positions, np.arange(len(high_profile)), high_profile, left=np.nan, right=np.nan)
    best, best_score = 0.0, -np.inf
    for offset in np.arange(-3.0, 3.01, 0.25):
        candidate = pooled(offset)
        keep = np.isfinite(candidate)
        if keep.sum() < len(low_profile) // 2:
            continue
        score = np.corrcoef(candidate[keep], low_profile[keep])[0, 1]
        if score > best_score:
            best, best_score = float(offset), float(score)
    return best if np.isfinite(best_score) else float("nan")


def run_pair(pair: dict, high_level: int, max_side: int) -> dict:
    low_array = zarr.open(url(pair, "low", 0), mode="r")
    high_array = zarr.open(url(pair, "high", high_level), mode="r")
    low_z, high_z = int(low_array.shape[0]) // 2, int(high_array.shape[0]) // 2
    print(f"[{pair['name']}] low {tuple(low_array.shape)} z={low_z}  high L{high_level} "
          f"{tuple(high_array.shape)} z={high_z}", flush=True)
    low = _stream_slice(low_array, low_z)
    high = _stream_slice(high_array, high_z)
    # high pixels at this level, measured in low pixels
    factor = pair["high_um"] * (2 ** high_level) / pair["low_um"]
    high = cv2.resize(high, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    low_valid, high_valid = low > 0, high > 0
    low_u8, high_u8 = _to_u8(low, low_valid), _to_u8(high, high_valid)

    report = {"name": pair["name"], "zid": pair["zid"], "high_level": high_level, "factor": factor}
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    candidates = {"as_rendered": identity}
    window = cv2.createHanningWindow(low.shape[::-1], cv2.CV_32F)
    canvas = np.zeros(low.shape, np.float32)
    canvas[:min(high.shape[0], low.shape[0]), :min(high.shape[1], low.shape[1])] = \
        high_u8[:low.shape[0], :low.shape[1]]
    (dx, dy), response = cv2.phaseCorrelate(canvas, low_u8.astype(np.float32), window)
    candidates["phase_shift"] = np.float32([[1, 0, dx], [0, 1, dy]])
    # opencv's sign convention is easy to get backwards; let the Dice decide
    candidates["phase_shift_neg"] = np.float32([[1, 0, -dx], [0, 1, -dy]])
    matrix, inliers = _sift_similarity(low_u8, high_u8, max_side)
    report["sift_inliers"] = inliers
    if matrix is not None:
        candidates["sift_similarity"] = matrix.astype(np.float32)
    for label, candidate in candidates.items():
        warped, warped_valid = _fit(low.shape, high_u8, high_valid, candidate)
        report[label] = {**_metrics(low_u8, low_valid, warped, warped_valid), "matrix": candidate.tolist()}
    best = max(candidates, key=lambda label: report[label]["dice"])
    report["best"] = best
    matrix = np.vstack([np.float64(candidates[best]), [0, 0, 1]])
    scale_estimate = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    # resized-high px -> low px; invert and rescale to low px -> high level-0 px
    to_high0 = np.linalg.inv(matrix)
    to_high0[:2] *= (2 ** high_level) / factor
    report["low_to_high0"] = to_high0[:2].tolist()
    report["residual_scale"] = scale_estimate

    # depth alignment on a 512 px block at the footprint centre (level 0 low, level `high_level` high)
    ys, xs = np.nonzero(low_valid)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    box = (max(0, cy - 256), cy + 256, max(0, cx - 256), cx + 256)
    corners = np.array([[box[2], box[0], 1], [box[3], box[1], 1]], dtype=np.float64)
    high_box = (corners @ to_high0.T) / (2 ** high_level)
    hy0, hy1 = sorted(int(v) for v in high_box[:, 1])
    hx0, hx1 = sorted(int(v) for v in high_box[:, 0])
    report["depth_offset_low_slices"] = _depth_offset(
        _depth_profile(low_array, box), _depth_profile(high_array, (max(0, hy0), hy1, max(0, hx0), hx1)),
        pair["low_um"], pair["high_um"],
    )

    dice = report[best]["dice"]
    report["verdict"] = "good" if dice >= 0.85 else "check" if dice >= 0.7 else "INVESTIGATE"
    OUT.mkdir(parents=True, exist_ok=True)
    warped, warped_valid = _fit(low.shape, high_u8, high_valid, candidates[best])
    overlay = np.dstack([np.zeros_like(low_u8), warped, low_u8])
    shrink = min(1.0, 2500 / max(low.shape))
    cv2.imwrite(str(OUT / f"{pair['name']}_overlay.jpg"),
                cv2.resize(overlay, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA))
    checker = ((np.indices(low.shape) // 256).sum(axis=0) % 2).astype(bool)
    cv2.imwrite(str(OUT / f"{pair['name']}_checker.jpg"),
                cv2.resize(np.where(checker, low_u8, warped), None, fx=shrink, fy=shrink,
                           interpolation=cv2.INTER_AREA))
    print(f"[{pair['name']}] dice as_rendered={report['as_rendered']['dice']:.3f} "
          f"phase={report['phase_shift']['dice']:.3f} "
          f"sift={report.get('sift_similarity', {}).get('dice', float('nan')):.3f} (inliers {inliers}) "
          f"best={best} residual_scale={scale_estimate:.4f} "
          f"depth_offset={report['depth_offset_low_slices']:+.2f} slices -> {report['verdict']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--high-level", type=int, default=2, help="2 = the 2.4 um render downscaled 4x")
    parser.add_argument("--max-side", type=int, default=3000, help="longest side used for SIFT matching")
    args = parser.parse_args()
    pairs = [pair for pair in PAIRS if not args.names or pair["name"] in args.names]
    reports = []
    for pair in pairs:
        try:
            reports.append(run_pair(pair, args.high_level, args.max_side))
        except Exception as error:  # keep going: one broken pair must not hide the others
            print(f"[{pair['name']}] FAILED: {error}", flush=True)
            reports.append({"name": pair["name"], "zid": pair["zid"], "error": str(error)})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "registration.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update({report["name"]: report for report in reports})
    path.write_text(json.dumps(existing, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
