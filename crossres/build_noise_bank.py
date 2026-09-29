"""plan R.3: build the native-noise bank from the 113 keV / 1.2 m training volumes (crossres/PLAN.md 0.7).

per patch: a (28, S+2h, S+2h) raw block inside the footprint and at least --label-margin px from any ink
label (the denoiser also removes faint ink, so its residual near text would hold signal), normalised
exactly as training (norm_cache.json, clipped to [0, 1]), denoised with the frozen blind-spot denoiser; the
bank keeps residual = normalised - denoised over the central --depth slices and S x S pixels.

intensity check: residual std per decile of the denoised intensity. if the largest / smallest exceeds
--scale-threshold, patches are stored at unit std (divided by a fitted sigma(intensity)) and the dataloader
multiplies them by sigma(local clean intensity); the fit is in the .json sidecar.

only local files are read. the bank (~130 MB float16) stays out of git; rebuild it on each machine:
    python crossres/build_noise_bank.py --dry-run
    python crossres/build_noise_bank.py          # -> _ves_tmp/native_noise_bank.npy + .json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.denoise import BlindSpotDenoiser  # noqa: E402
from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm  # noqa: E402

HALO = 8  # > the denoiser's receptive-field radius (6 convs of 3x3x3)


def native_scroll_ids() -> list[int]:
    import campaign_archs_33 as campaign33
    import campaign_archs_35 as campaign35
    return [int(s) for domain in campaign35.QUALITY_REFERENCE_DOMAINS for s in campaign33.CAMPAIGN33_SCROLL_DICT[domain]]


def far_from_ink(zid: int, shape: tuple[int, int], margin: int) -> np.ndarray:
    """footprint pixels at least `margin` px from every ink label."""
    mask = cv2.imread(str(ROOT / "masks" / f"{zid}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"masks/{zid}.png")
    allowed = np.zeros(shape, bool)
    h, w = min(shape[0], mask.shape[0]), min(shape[1], mask.shape[1])
    allowed[:h, :w] = mask[:h, :w] > 0
    for folder in ("dilated_inklabels", "inklabels"):
        label = cv2.imread(str(ROOT / folder / f"{zid}.png"), cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        ink = np.zeros(shape, np.uint8)
        lh, lw = min(shape[0], label.shape[0]), min(shape[1], label.shape[1])
        ink[:lh, :lw] = label[:lh, :lw] > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        allowed &= cv2.dilate(ink, kernel) == 0
    return allowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-dir", default=os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    ap.add_argument("--scroll-ids", type=int, nargs="*", default=None,
                    help="default: the 113 keV / 1.2 m training domains (pherc0139, pherc0814, pherc0500p2)")
    ap.add_argument("--denoiser", default=str(ROOT / "models" / "denoiser_n2v_block3_4k.pth"))
    ap.add_argument("--out", default=str(ROOT / "_ves_tmp" / "native_noise_bank.npy"))
    ap.add_argument("--patches", type=int, default=256)
    ap.add_argument("--depth", type=int, default=16, help="central slices kept (the 8-slice window moves inside)")
    ap.add_argument("--size", type=int, default=128, help="patch side, px (>= the 96 px training context)")
    ap.add_argument("--label-margin", type=int, default=64)
    ap.add_argument("--scale-threshold", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="4 patches, print the statistics, write nothing")
    args = ap.parse_args()

    import zarr
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    denoiser = BlindSpotDenoiser()
    denoiser.load_state_dict(torch.load(args.denoiser, map_location="cpu", weights_only=True))
    denoiser = denoiser.to(dev).eval().requires_grad_(False)
    scroll_ids = args.scroll_ids or native_scroll_ids()
    n_patches = 4 if args.dry_run else args.patches
    size, span = args.size, args.size + 2 * HALO

    sources = []
    for zid in scroll_ids:
        volume = zarr.open(os.path.join(args.zarr_dir, f"{zid}.zarr"), mode="r")
        depth, height, width = map(int, volume.shape)
        allowed = far_from_ink(zid, (height, width), args.label_margin)
        # patch origins whose whole (span x span) block is allowed: a box filter over the allowed map
        inside = cv2.boxFilter(allowed.astype(np.float32), -1, (span, span), normalize=True,
                               anchor=(0, 0), borderType=cv2.BORDER_CONSTANT) > 0.999
        inside[height - span + 1:] = False
        inside[:, width - span + 1:] = False
        ys, xs = np.nonzero(inside)
        if len(ys) == 0:
            print(f"[noise] {zid}: no footprint {span}px from ink; skipped", flush=True)
            continue
        norm = load_cached_norm(str(zid), str(ROOT / UNIFIED_CACHE_PATH))
        if norm is None:
            raise RuntimeError(f"{zid} has no norm_cache.json entry")
        sources.append((zid, volume, norm, ys, xs))
        print(f"[noise] {zid}: {len(ys):,} candidate origins", flush=True)
    if not sources:
        raise SystemExit("no usable native volumes")

    z0 = None
    patches, cleans, per_scroll = [], [], {}
    for index in range(n_patches):
        zid, volume, (mean, std, g_min, g_max), ys, xs = sources[index % len(sources)]
        depth = int(volume.shape[0])
        z0 = (depth - args.depth) // 2
        pick = int(rng.integers(len(ys)))
        y, x = int(ys[pick]), int(xs[pick])
        raw = np.asarray(volume[:, y:y + span, x:x + span], np.float32)
        normalised = np.clip(((raw - mean) / std - g_min) / (g_max - g_min + 1e-12), 0, 1)
        with torch.no_grad():
            clean = denoiser(torch.from_numpy(normalised).to(dev)[None, None])[0, 0].cpu().numpy()
        crop = (slice(z0, z0 + args.depth), slice(HALO, HALO + size), slice(HALO, HALO + size))
        patches.append((normalised - clean)[crop])
        cleans.append(clean[crop])
        per_scroll[str(zid)] = per_scroll.get(str(zid), 0) + 1
    patches, cleans = np.stack(patches), np.stack(cleans)

    # residual std per decile of the clean intensity
    sample = rng.choice(patches.size, size=min(patches.size, 2_000_000), replace=False)
    residual, intensity = patches.ravel()[sample], cleans.ravel()[sample]
    edges = np.quantile(intensity, np.linspace(0, 1, 11))
    centres, sigmas = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (intensity >= lo) & (intensity <= hi)
        centres.append(float(intensity[keep].mean()))
        sigmas.append(float(residual[keep].std()))
    spread = max(sigmas) / max(min(sigmas), 1e-9)
    coeffs = np.polyfit(centres, sigmas, 1)
    floor = 0.5 * min(sigmas)
    scaled = spread > args.scale_threshold
    if scaled:
        patches = patches / np.maximum(np.polyval(coeffs, cleans), floor)
    report = {
        "scroll_ids": scroll_ids, "patches_per_scroll": per_scroll, "denoiser": os.path.basename(args.denoiser),
        "slices": [z0, z0 + args.depth], "size": size, "label_margin": args.label_margin,
        "normalisation": "norm_cache.json global, clipped to [0, 1]", "residual_std": float(residual.std()),
        "sigma_by_intensity": [{"intensity": c, "sigma": s} for c, s in zip(centres, sigmas)],
        "sigma_spread": spread, "intensity_scaled": bool(scaled), "sigma_coeffs": [float(c) for c in coeffs],
        "sigma_floor": floor,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "sigma_by_intensity"}, indent=1))
    for row in report["sigma_by_intensity"]:
        print(f"[noise] intensity {row['intensity']:.3f}: sigma {row['sigma']:.4f}")
    if args.dry_run:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, patches.astype(np.float16))
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[noise] wrote {out} {patches.shape} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
