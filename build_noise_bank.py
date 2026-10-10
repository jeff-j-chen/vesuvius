"""build_noise_bank.py -- real 113 keV / 1.2 m scan noise for the degraded fine-scan fragments (plan R.3).

The degrader (utils/degrader.py) predicts the *mean* native 9.36 um scan, so translated fragments come out
too clean. This bank holds residuals raw - denoiser(raw) cut from natively scanned 9.36 um volumes, in the
training normalisation, so utils/native_noise.py can add a fresh real-noise patch to every training crop.

1. a blind-spot denoiser (train_denoiser.py) is trained on the native volumes if it does not exist yet
2. normalised (depth, ctx, ctx) crops are denoised; residual = raw - clean inside the papyrus
3. the residual std is fitted against the clean intensity (quadratic), and patches are stored at unit std
   so NativeNoise rescales them by sigma(local clean intensity)

    python3 build_noise_bank.py
    python3 build_noise_bank.py --patches 512 --out _ves_tmp/native_noise_bank.npy
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mae_pretrain_nnunet import CropSampler  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.denoise import BlindSpotDenoiser  # noqa: E402

# natively scanned at 9.36 um / 113 keV / 1.2 m: pherc0139, pherc0814, pherc0500p2
NATIVE_SCROLL_IDS = (
    20260115000000, 20260317000000, 20250223000000, 20250108000005, 20260112000000,
    20260126000000, 20250831000000, 20260108000000, 20260302000000,
    20260226000000, 20250628074500,
)
DEFAULT_OUT = "_ves_tmp/native_noise_bank.npy"
DEFAULT_DENOISER = "models/denoiser_n2v_native.pth"


def ensure_denoiser(path: Path, scroll_ids, steps: int, depth: int) -> None:
    if path.is_file():
        return
    command = [
        sys.executable, str(ROOT / "train_denoiser.py"),
        "--name", path.stem, "--out-dir", str(path.parent),
        "--scroll-ids", *(str(value) for value in scroll_ids),
        "--steps", str(steps), "--depth", str(depth),
    ]
    print(f"[noise-bank] training denoiser: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="build the native-scan noise bank")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--denoiser", default=DEFAULT_DENOISER)
    ap.add_argument("--denoiser-steps", type=int, default=4000)
    ap.add_argument("--scroll-ids", type=int, nargs="+", default=list(NATIVE_SCROLL_IDS))
    ap.add_argument("--patches", type=int, default=384)
    ap.add_argument("--ctx", type=int, default=128, help="patch side; must be >= the training context")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--norm-mode", default=None, help="defaults to utils.config Config().data.norm_mode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(ROOT)
    denoiser_path = ROOT / args.denoiser
    ensure_denoiser(denoiser_path, args.scroll_ids, args.denoiser_steps, args.depth)

    config = Config()
    norm_mode = str(args.norm_mode or config.data.norm_mode)
    config.data.norm_mode = norm_mode
    config.data.tile_size = 16
    config.data.depth = args.depth
    config.data.context_size = args.ctx
    import zarr

    samplers = []
    for scroll_id in dict.fromkeys(args.scroll_ids):
        volume = zarr.open(os.path.join(config.data.zarr_path, f"{scroll_id}.zarr"), mode="r")
        height, width = int(volume.shape[1]), int(volume.shape[2])
        samplers.append(CropSampler(
            scroll_id, config.data.zarr_path, config, 0, (height // args.ctx) * args.ctx,
            0, (width // args.ctx) * args.ctx, "train", holdout_frac=0.05,
        ))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlindSpotDenoiser().to(device).eval()
    model.load_state_dict(torch.load(denoiser_path, map_location=device, weights_only=True))
    rng = np.random.default_rng(args.seed)

    raws, cleans = [], []
    per_scroll = int(np.ceil(args.patches / len(samplers)))
    with torch.no_grad():
        for sampler in samplers:
            remaining = per_scroll
            while remaining > 0:
                batch = sampler.sample(min(args.batch_size, remaining), rng)
                if batch is None:
                    raise RuntimeError(f"scroll {sampler.scroll_id} returned no crops")
                clean = model(batch.to(device)).float().cpu()
                raws.append(batch[:, 0].numpy())
                cleans.append(clean[:, 0].numpy())
                remaining -= int(batch.shape[0])
            print(f"[noise-bank] scroll {sampler.scroll_id}: {per_scroll} crops", flush=True)
    raw = np.concatenate(raws)[:args.patches]
    clean = np.concatenate(cleans)[:args.patches]
    inside = raw > 0
    residual = np.where(inside, raw - clean, 0.0).astype(np.float32)

    # residual std as a function of local clean intensity, fitted on well-populated bins
    edges = np.linspace(0.0, 1.0, 33)
    centers, sigmas, counts = [], [], []
    values, levels = residual[inside], clean[inside]
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (levels >= low) & (levels < high)
        if selected.sum() >= 2000:
            centers.append(0.5 * (low + high))
            sigmas.append(float(values[selected].std()))
            counts.append(int(selected.sum()))
    if len(centers) < 3:
        raise RuntimeError("too few populated intensity bins to fit the noise level")
    coeffs = np.polyfit(centers, sigmas, 2, w=np.sqrt(counts))
    sigma_floor = 0.25 * float(np.median(sigmas))
    sigma = np.maximum(np.polyval(coeffs, clean), sigma_floor)
    bank = np.where(inside, residual / sigma, 0.0).astype(np.float16)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, bank)
    meta = {
        "norm_mode": norm_mode,
        "intensity_scaled": True,
        "sigma_coeffs": [float(value) for value in coeffs],
        "sigma_floor": sigma_floor,
        "bins": {"centers": centers, "sigma": sigmas, "count": counts},
        "denoiser": str(denoiser_path.relative_to(ROOT)),
        "scroll_ids": list(dict.fromkeys(int(value) for value in args.scroll_ids)),
        "shape": list(bank.shape),
        "residual_std": float(values.std()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[noise-bank] wrote {out} {bank.shape} residual_std={meta['residual_std']:.4f} "
        f"sigma(0.1/0.5)={np.polyval(coeffs, 0.1):.4f}/{np.polyval(coeffs, 0.5):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
