"""assemble_researcher_inklabels.py -- map official 2026-09-18 researcher ink labels onto our training frames.

Writes researcher_inklabels/<zid>.png (uint8 0/255, same shape as inklabels/<zid>.png).

  same canvas : w044/w035 use the 9.362um label directly; w013/w018/paris4 area-downsample the
                2.4um label by exactly 4x (our volume is level 2 of the same surface render).
  re-rendered : seg46527/p9b_487/p841 labels live on a newer 2.4um render. The S3 level-2 surface
                midslice is registered to our volume midslice (SIFT affine + smoothed tile-wise
                displacement field) and the label is warped with the same transform.

usage: python assemble_researcher_inklabels.py [--only 20260221022814,...]
"""
from __future__ import annotations
import argparse, json, os

import cv2
import numpy as np
import tensorstore as ts
import zarr

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
OUT_DIR = "researcher_inklabels"
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH", "ves_zarrs2")
RELEASE = "20260918"

# zid -> (name, segment prefix, ink-labels volume dir, mode, extra)
SOURCES = {
    "20260115000000": ("w044", "PHerc0139/segments/20260115000000-w044_2026011522",
                       "9.362um-volume-20250728140407", "direct", {}),
    "20260317000000": ("w035", "PHerc0139/segments/20260317000000-w035_2026031718",
                       "9.362um-volume-20250728140407", "direct", {}),
    "20240304141531": ("w013", "PHerc1667/segments/20240304141531-w013_20240304141531_flatboi",
                       "2.399um-volume-20251217075048", "pool4", {"x1_level2": 4975}),
    "20240304144031": ("w018", "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi",
                       "2.399um-volume-20251217075048", "pool4", {}),
    "20231210121321": ("paris4", "PHercParis4/segments/20231210121321",
                       "2.4um-volume-20260411134726", "pool4", {}),
    "20260226000000": ("seg46527", "PHerc0814/segments/20260226000000-46527_2um_try2",
                       "2.399um-volume-20260309142202", "register",
                       {"surface": "2.399um-0.22m-78keV-volume-20260309142202.zarr"}),
    "20250919125754": ("p9b_487", "PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr",
                       "2.401um-volume-20250820154339", "register",
                       {"surface": "2.401um-0.35m-77keV-volume-20250820154339.zarr"}),
    "20260221022814": ("p841", "PHerc0841/segments/20260221022814-auto_grown_20260220174252405",
                       "2.403um-volume-20260319124803", "register",
                       {"surface": "2.403um-0.22m-77keV-volume-20260319124803.zarr"}),
}


def open_remote(url):
    try:
        return zarr.open(url, mode="r")
    except Exception:
        return ts.open({"driver": "zarr3", "kvstore": {"driver": "http", "base_url": url}}).result()


def read(array, *index):
    out = array[index] if index else array[...]
    return np.asarray(out.read().result() if hasattr(out, "read") else out)


def label_url(seg, voldir, level):
    return f"{BUCKET}/{seg}/ink-labels/{voldir}/{RELEASE}/inklabels.zarr/{level}"


def pool4_label(seg, voldir, x1_level2=None):
    """level-0 label -> exact 4x area mean (float 0..1) on the level-2 grid."""
    src = open_remote(label_url(seg, voldir, 0))
    height, width = map(int, src.shape)
    if height % 4 or width % 4:
        raise RuntimeError(f"{seg}: level-0 shape {src.shape} is not divisible by 4")
    x1 = width if x1_level2 is None else 4 * int(x1_level2)
    out = np.zeros((height // 4, x1 // 4), dtype=np.float32)
    strip = 2048
    for y0 in range(0, height, strip):
        y1 = min(y0 + strip, height)
        block = read(src, slice(y0, y1), slice(0, x1)).astype(np.float32) / 255.0
        out[y0 // 4:y1 // 4] = block.reshape((y1 - y0) // 4, 4, x1 // 4, 4).mean(axis=(1, 3))
    return out


def norm8(image):
    image = image.astype(np.float32)
    valid = image[image > 0]
    lo, hi = np.percentile(valid, [1, 99]) if valid.size else (0.0, 1.0)
    return (np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def fit_affine(src_img, dst_img):
    sift = cv2.SIFT_create(nfeatures=40000)
    clahe = cv2.createCLAHE(3.0, (16, 16))
    ks, ds = sift.detectAndCompute(clahe.apply(src_img), (src_img > 0).astype(np.uint8))
    kd, dd = sift.detectAndCompute(clahe.apply(dst_img), (dst_img > 0).astype(np.uint8))
    pairs = cv2.BFMatcher().knnMatch(ds, dd, k=2)
    good = [a for a, b in pairs if a.distance < 0.8 * b.distance]
    ps = np.float32([ks[m.queryIdx].pt for m in good])
    pd = np.float32([kd[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffine2D(ps, pd, ransacReprojThreshold=3.0, maxIters=20000,
                                           confidence=0.999)
    if affine is None or int(inliers.sum()) < 100:
        raise RuntimeError(f"affine registration failed ({0 if inliers is None else int(inliers.sum())} inliers)")
    return affine, int(inliers.sum())


def tile_ncc(a, b, tile=256):
    both = (a > 0) & (b > 0)
    scores = []
    for y in range(0, a.shape[0] - tile + 1, tile):
        for x in range(0, a.shape[1] - tile + 1, tile):
            m = both[y:y + tile, x:x + tile]
            if m.mean() < 0.8:
                continue
            p = a[y:y + tile, x:x + tile][m].astype(np.float32)
            q = b[y:y + tile, x:x + tile][m].astype(np.float32)
            if p.std() > 5 and q.std() > 5:
                scores.append(float(np.corrcoef(p, q)[0, 1]))
    return float(np.median(scores)) if scores else float("nan")


def displacement_field(warped, dst, step=64, patch=160, search=24, min_score=0.35):
    """per-node shift (dx, dy) such that dst(p) ~ warped(p + d); smoothed, dense float32 maps."""
    clahe = cv2.createCLAHE(3.0, (16, 16))
    a = clahe.apply(dst).astype(np.float32)
    b = clahe.apply(warped).astype(np.float32)
    height, width = dst.shape
    ys = np.arange(patch // 2, height - patch // 2, step)
    xs = np.arange(patch // 2, width - patch // 2, step)
    dx = np.full((len(ys), len(xs)), np.nan, np.float32)
    dy = np.full_like(dx, np.nan)
    half, reach = patch // 2, patch // 2 + search
    for i, cy in enumerate(ys):
        for j, cx in enumerate(xs):
            if cy - reach < 0 or cx - reach < 0 or cy + reach > height or cx + reach > width:
                continue
            templ = a[cy - half:cy + half, cx - half:cx + half]
            if (dst[cy - half:cy + half, cx - half:cx + half] == 0).mean() > 0.05 or templ.std() < 8:
                continue
            region = b[cy - reach:cy + reach, cx - reach:cx + reach]
            if (warped[cy - reach:cy + reach, cx - reach:cx + reach] == 0).mean() > 0.05:
                continue
            score = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
            _, best, _, loc = cv2.minMaxLoc(score)
            if best < min_score:
                continue
            px, py = loc
            sub = [0.0, 0.0]
            for axis, (p, n) in enumerate(((px, score.shape[1]), (py, score.shape[0]))):
                if 0 < p < n - 1:
                    l, c, r = ((score[py, p - 1], score[py, p], score[py, p + 1]) if axis == 0
                               else (score[p - 1, px], score[p, px], score[p + 1, px]))
                    den = l - 2 * c + r
                    sub[axis] = 0.5 * (l - r) / den if den < 0 else 0.0
            dx[i, j] = px + sub[0] - search
            dy[i, j] = py + sub[1] - search
    valid = np.isfinite(dx)
    # reject outliers against a local median, then fill holes from nearest valid nodes
    for field in (dx, dy):
        filled = np.where(valid, field, 0).astype(np.float32)
        med = cv2.medianBlur(filled, 5)
        valid &= np.abs(field - med) < 3.0
    if valid.sum() < 20:
        return None, int(valid.sum())
    _, nearest = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    src_idx = np.zeros(int(nearest.max()) + 1, dtype=np.int64)
    src_idx[nearest[valid]] = np.flatnonzero(valid)
    maps = []
    for field in (dx, dy):
        full = field.ravel()[src_idx[nearest]].reshape(field.shape)
        full = cv2.GaussianBlur(cv2.medianBlur(full.astype(np.float32), 3), (0, 0), 1.5)
        dense = cv2.resize(full, (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((height, width), np.float32)
        y0 = int(ys[0]) - step // 2
        x0 = int(xs[0]) - step // 2
        canvas[:] = cv2.copyMakeBorder(dense, y0, max(0, height - y0 - dense.shape[0]), x0,
                                       max(0, width - x0 - dense.shape[1]),
                                       cv2.BORDER_REPLICATE)[:height, :width]
        maps.append(canvas)
    return maps, int(valid.sum())


def register_label(zid, seg, voldir, surface):
    ours = zarr.open(os.path.join(ZARR_DIR, f"{zid}.zarr"), mode="r")
    height, width = map(int, ours.shape[1:])
    dst = norm8(np.asarray(ours[int(ours.shape[0]) // 2]))
    src_vol = open_remote(f"{BUCKET}/{seg}/surface-volumes/{surface}/2")
    src = norm8(read(src_vol, int(src_vol.shape[0]) // 2))
    affine, inliers = fit_affine(src, dst)
    inverse = cv2.invertAffineTransform(affine)
    gx, gy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    def to_src(dx=None, dy=None):
        x = gx if dx is None else gx + dx
        y = gy if dy is None else gy + dy
        return (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2],
                inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2])

    mx, my = to_src()
    warped = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderValue=0)
    ncc_affine = tile_ncc(dst, warped)
    maps, nodes = displacement_field(warped, dst)
    ncc_final = ncc_affine
    if maps is not None:
        rx, ry = to_src(*maps)
        refined = cv2.remap(src, rx, ry, cv2.INTER_LINEAR, borderValue=0)
        ncc_refined = tile_ncc(dst, refined)
        if ncc_refined > ncc_affine:
            mx, my, ncc_final = rx, ry, ncc_refined
    label = pool4_label(seg, voldir)
    if label.shape != src.shape:
        raise RuntimeError(f"{zid}: pooled label {label.shape} != surface level-2 {src.shape}")
    out = cv2.remap(label, mx, my, cv2.INTER_LINEAR, borderValue=0)
    coverage = float(((warped > 0) & (dst > 0)).sum() / max(int((dst > 0).sum()), 1))
    info = {"affine_src_level2_to_ours": np.round(affine, 6).tolist(), "sift_inliers": inliers,
            "flow_nodes": nodes, "tile_ncc_affine": round(ncc_affine, 4),
            "tile_ncc_final": round(ncc_final, 4), "coverage_of_our_surface": round(coverage, 4)}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else set(SOURCES)
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for zid, (name, seg, voldir, mode, extra) in SOURCES.items():
        if zid not in want:
            continue
        ref = cv2.imread(f"inklabels/{zid}.png", cv2.IMREAD_GRAYSCALE)
        print(f"== {name} ({zid}) mode={mode} target={ref.shape}", flush=True)
        info = {}
        if mode == "direct":
            prob = read(open_remote(label_url(seg, voldir, 0))).astype(np.float32) / 255.0
        elif mode == "pool4":
            prob = pool4_label(seg, voldir, extra.get("x1_level2"))
        else:
            prob, info = register_label(zid, seg, voldir, extra["surface"])
        if prob.shape != ref.shape:
            raise RuntimeError(f"{zid}: assembled {prob.shape} != inklabels {ref.shape}")
        label = (prob >= 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(OUT_DIR, f"{zid}.png"), label)
        old, new = ref > 0, label > 0
        dice = 2 * (old & new).sum() / max(int(old.sum() + new.sum()), 1)
        info.update({"name": name, "mode": mode, "source": label_url(seg, voldir, 0),
                     "shape": list(label.shape), "positive_frac": round(float(new.mean()), 5),
                     "dice_vs_inklabels": round(float(dice), 4)})
        manifest[zid] = info
        print("  ", json.dumps(info), flush=True)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    main()
