"""non-ML ink maps for every PHerc0846A and PHerc0175A fragment on disk: one 2x2 figure per fragment.

composite = max over slices 10-17, each slice local-z normalised and blurred 4 px. ten 2D filters of the composite
(the priors) are turned into robust z-maps and combined per pixel. panels: asf_morph | corroborated (priors blurred
15 px first) / consensus7 | mean_rank. lengths are native px; the work grid is 2x area-downsampled (strokes are 50+ px).

fragments = the numbered renders in NAMED plus every rendered tifxyz/quick_patches_from_D/<scroll>/<segment>.

  python3 nonml_ink.py                                  # everything -> output/nonml_ink/<scroll>/<id>_2x2.jpg
  python3 nonml_ink.py --scrolls PHerc0175A             # one scroll
  python3 nonml_ink.py --only 20261012065810,20261012065812
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import zarr
from PIL import Image
from scipy import ndimage as ndi

from utils.fast_figure import orient_display

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.abspath(__file__))
# numbered renders (assemble_test_segments.FRAGMENTS); flat aliases and layer-reversed copies are the same surfaces
NAMED = {
    "20260921094413": ("PHerc0846A", "test surface 6"),
    "20260927134928": ("PHerc0846A", "ABF patch (alias 20260927134931)"),
    "20261012073244": ("PHerc0846A", "073244 patch (alias 20261012073246)"),
    "20261012073249": ("PHerc0846A", "073244 +29 offset, p29 (alias 20261012073250)"),
    "20261012081237": ("PHerc0846A", "081237 patch (alias 20261012081238)"),
    "20261012065810": ("PHerc0846A", "straight re-trace (alias 20261012065811)"),
    "20261014084610": ("PHerc0846A", "second straight sheet, assembled rotated 180 (alias 20261014084611)"),
    "20261012065812": ("PHerc0846A", "straight -28 offset (alias 20261012065813)"),
    "20261012065814": ("PHerc0846A", "straightened2 re-trace, rotated 180 (alias 20261012065815)"),
    "20261012065816": ("PHerc0846A", "straightened2 in_1, rotated 180 (alias 20261012065817)"),
    "20261012065818": ("PHerc0846A", "straightened2 out_1, rotated 180 (alias 20261012065819)"),
    "20261012065820": ("PHerc0846A", "straightened2 out_2, rotated 180 (alias 20261012065821)"),
    "20261012065822": ("PHerc0846A", "straightened2 out_3, rotated 180 (alias 20261012065823)"),
    "20261012065824": ("PHerc0846A", "straightened2 out_4, rotated 180 (alias 20261012065825)"),
    "20261012065826": ("PHerc0846A", "straightened2 out_5, rotated 180 (alias 20261012065827)"),
    "20260918132724": ("PHerc0175A", "test surface 7"),
}
# quick patches already rendered under a numbered id above
QUICK_DUPLICATES = {"auto_grown_20260927134928530_abf"}
SCROLLS = ("PHerc0846A", "PHerc0175A")
F = 2
# local ink contrast per slice peaks at 13-14 on both labeled sheets; an 8-slice max over 10-17 scored best
MIP_RANGE = list(range(10, 18))
Z_CLIP = 5.0
OUT = os.path.join(ROOT, "output", "nonml_ink")


def w(n):
    return n / F


def odd(n):
    return max(3, int(round(n)) | 1)


def gblur(x, sigma_native):
    return cv2.GaussianBlur(x, (0, 0), w(sigma_native), borderType=cv2.BORDER_REFLECT)


def nblur(x, m, sigma_native):
    """mask-normalised gaussian (only papyrus pixels contribute)."""
    mf = m.astype(np.float32)
    return gblur(x * mf, sigma_native) / np.maximum(gblur(mf, sigma_native), 1e-3)


def fill(x, m):
    """push-pull: smooth extrapolation of the papyrus into the background so wide filters see no mask edge."""
    xs, ms = [(x * m).astype(np.float32)], [m.astype(np.float32)]
    while min(xs[-1].shape) > 4:
        xs.append(cv2.resize(xs[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        ms.append(cv2.resize(ms[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    val = np.where(ms[-1] > 1e-6, xs[-1] / np.maximum(ms[-1], 1e-6), float(x[m].mean())).astype(np.float32)
    for a, b in zip(xs[-2::-1], ms[-2::-1]):
        up = cv2.resize(val, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        val = a + (1.0 - b) * up
    return np.where(m, x, val).astype(np.float32)


def flatfield(x, m, sigma_native=250):
    return fill(x - nblur(x, m, sigma_native), m)


def local_z(x, m, sigma_native=250):
    mu = nblur(x, m, sigma_native)
    sd = np.sqrt(np.maximum(nblur((x - mu) ** 2, m, sigma_native), 1e-6))
    return fill((x - mu) / sd, m)


def robust_z(x, m, clip=None):
    med = np.median(x[m])
    z = (x - med) / max(1.4826 * np.median(np.abs(x[m] - med)), 1e-6)
    return z if clip is None else np.clip(z, -clip, clip)


def hann_conv(x, m, size_native):
    n = odd(w(size_native))
    k = np.hanning(n + 2)[1:-1].astype(np.float32)
    k /= k.sum()
    mf = m.astype(np.float32)
    num = cv2.sepFilter2D(x * mf, -1, k, k, borderType=cv2.BORDER_REFLECT)
    den = cv2.sepFilter2D(mf, -1, k, k, borderType=cv2.BORDER_REFLECT)
    return num / np.maximum(den, 1e-3)


def disk(radius_native):
    r = max(1, int(round(w(radius_native))))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def asf(x, radii_native=(2, 4, 6, 8, 10, 12)):
    """alternating open/close: removes bright and dark structures thinner than the largest disk (fibres, specks)."""
    for r in radii_native:
        k = disk(r)
        x = cv2.morphologyEx(cv2.morphologyEx(x, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return x


def to_u8(x, m, lo=1.0, hi=99.5):
    a, b = np.percentile(x[m], [lo, hi])
    return np.where(m, np.clip((x - a) / max(b - a, 1e-6) * 255, 0, 255), 0).astype(np.uint8)


def load(zid, mask_path):
    vol = np.asarray(zarr.open(os.path.join(ROOT, "ves_zarrs2", f"{zid}.zarr"), mode="r")[:], dtype=np.float32)
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    h, wd = vol.shape[1] // F * F, vol.shape[2] // F * F
    v = vol[:, :h, :wd].reshape(vol.shape[0], h // F, F, wd // F, F).mean((2, 4)).astype(np.float32)
    m = mask[:h, :wd].reshape(h // F, F, wd // F, F).mean((1, 3)) > 0.999
    # renders carry a bright rim along the mask edge
    return v, cv2.erode(m.astype(np.uint8), disk(24)).astype(bool)


def priors(v, m):
    vz = np.stack([gblur(local_z(fill(v[d], m), m, 250), 4) for d in range(v.shape[0])])
    raw = fill(np.stack([gblur(fill(v[d], m), 4) for d in MIP_RANGE]).max(0), m)
    base = fill(vz[MIP_RANGE].max(0), m)

    u8 = to_u8(gblur(base, 8), m, 0.5, 99.5)
    u8 = np.where(m, u8, int(np.median(u8[m]))).astype(np.uint8)
    tiles = (max(2, int(u8.shape[1] * F / 400)), max(2, int(u8.shape[0] * F / 400)))
    smooth = gblur(base, 8)
    k, it = disk(10), 12
    local = robust_z(gblur(local_z(raw, m, 250), 15), m)
    return {
        "mip": base,
        "flatfield": gblur(flatfield(raw, m, 250), 15),
        "local_z": np.tanh(local),
        "clahe": gblur(cv2.createCLAHE(clipLimit=2.5, tileGridSize=tiles).apply(u8).astype(np.float32), 6),
        "hann": hann_conv(base, m, 96) - hann_conv(base, m, 480),
        "depth_peak": gblur(base - np.median(vz, 0), 15),
        "asf_morph": gblur(asf(base), 10),
        "tophat": gblur(smooth - cv2.dilate(cv2.erode(smooth, k, iterations=it), k, iterations=it), 8),
        "log": np.max([-(w(s) ** 2) * ndi.gaussian_laplace(base, w(s)) for s in (20, 30, 40)], 0),
        "dog": gblur(base, 15) - gblur(base, 80),
    }


def percentile_rank(x, m):
    out = np.zeros_like(x)
    vals = x[m]
    out[m] = np.argsort(np.argsort(vals, kind="stable"), kind="stable") / max(len(vals) - 1, 1)
    return out


def corroborated(z, m, k=4, gamma=0.5, sigma_native=16, ramp=(0.35, 0.65)):
    """mid-boost over priors (as in 0846A_eval): p = sigmoid(1.5 (z - 1)); where the k-th strongest prior (smoothed)
    is also high, the top-k mean is lifted by a gamma curve; a lone single-prior spike stays at the top-k mean."""
    p = 1.0 / (1.0 + np.exp(-1.5 * (z - 1.0)))
    ranked = np.sort(p, 0)[::-1]
    agree = np.clip((nblur(ranked[k - 1], m, sigma_native) - ramp[0]) / (ramp[1] - ramp[0]), 0, 1)
    agree = agree * agree * (3 - 2 * agree)
    return ranked[:k].mean(0) ** (1 - (1 - gamma) * agree)


def titled(img, text, height=44, scale=1.0):
    bar = np.zeros((height, img.shape[1], 3), np.uint8)
    while scale > 0.4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > img.shape[1] - 20:
        scale *= 0.9
    cv2.putText(bar, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def fragments():
    """[(scroll, zarr id, label, mask path)] for every fragment whose zarr and mask exist."""
    from utils.pherc0175a import alias_of

    out = []
    for zid, (scroll, label) in NAMED.items():
        mask = os.path.join(ROOT, "masks", f"{zid}.png")
        if os.path.isdir(os.path.join(ROOT, "ves_zarrs2", f"{zid}.zarr")) and os.path.isfile(mask):
            out.append((scroll, zid, label, mask))
    for scroll in SCROLLS:
        for name in sorted(os.listdir(os.path.join(ROOT, "tifxyz", "quick_patches_from_D", scroll))):
            mask = os.path.join(ROOT, "masks", "extras", f"{name}.png")
            if name in QUICK_DUPLICATES or not os.path.isfile(mask) \
                    or not os.path.isdir(os.path.join(ROOT, "ves_zarrs2", f"{name}.zarr")):
                continue
            label = f"quick patch (alias {alias_of(name)})" if scroll == "PHerc0175A" else "quick patch"
            out.append((scroll, name, label, mask))
    return out


def figure(item):
    scroll, zid, label, mask_path = item
    t0 = time.time()
    cv2.setNumThreads(6)
    v, m = load(zid, mask_path)
    p = priors(v, m)
    z = np.stack([robust_z(x, m, Z_CLIP) for x in p.values()]).astype(np.float32)
    zs = np.stack([robust_z(gblur(layer, 15), m, Z_CLIP) for layer in z])
    panels = [
        ("asf_morph", p["asf_morph"]),
        ("corroborated (priors blurred 15 px)", corroborated(zs, m)),
        ("consensus7 (7th strongest of 10 priors)", np.sort(z, 0)[len(z) - 7]),
        ("mean_rank (mean percentile rank of 10 priors)", np.mean([percentile_rank(layer, m) for layer in z], 0)),
    ]
    tiles = [titled(cv2.cvtColor(orient_display(to_u8(x, m), zid), cv2.COLOR_GRAY2BGR), name) for name, x in panels]
    pad = lambda t: np.pad(t, ((8, 8), (8, 8), (0, 0)), constant_values=40)  # noqa: E731
    fig = np.vstack([np.hstack([pad(tiles[0]), pad(tiles[1])]), np.hstack([pad(tiles[2]), pad(tiles[3])])])
    head = f"{scroll} {zid}  {label}  |  composite = max of slices {MIP_RANGE[0]}-{MIP_RANGE[-1]}"
    os.makedirs(os.path.join(OUT, scroll), exist_ok=True)
    path = os.path.join(OUT, scroll, f"{zid}_2x2.jpg")
    cv2.imwrite(path, titled(fig, head, 56, 1.3), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[{scroll} {zid}] {path} ({time.time() - t0:.0f}s)", flush=True)
    return path


def safe_figure(item):
    """'' on success, else the error text (one bad fragment must not stop the batch)."""
    try:
        figure(item)
        return ""
    except Exception as exc:
        return repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", default=",".join(SCROLLS), help="comma-separated scrolls")
    ap.add_argument("--only", default="", help="comma-separated zarr ids (render ids or quick-patch names)")
    ap.add_argument("--workers", type=int, default=12, help="fragments processed in parallel")
    args = ap.parse_args()
    scrolls = {s.strip() for s in args.scrolls.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    items = [f for f in fragments() if f[0] in scrolls and (not only or f[1] in only)]
    print(f"{len(items)} fragments: " + ", ".join(f"{s} {sum(f[0] == s for f in items)}" for s in sorted(scrolls)))
    # largest first so the long ones do not straggle at the end
    items.sort(key=lambda f: -np.prod(zarr.open(os.path.join(ROOT, "ves_zarrs2", f"{f[1]}.zarr"), mode="r").shape))
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(items)))) as ex:
        failed = [(f[1], r) for f, r in zip(items, ex.map(safe_figure, items)) if r]
    for zid, error in failed:
        print(f"FAILED {zid}: {error}")
    print(f"done: {len(items) - len(failed)}/{len(items)} figures in {OUT}")


if __name__ == "__main__":
    main()
