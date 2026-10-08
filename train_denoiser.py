"""train_denoiser.py -- self-supervised blind-spot denoiser for the model's input front-end.

Samples normalized (D, ctx, ctx) crops exactly as the MAE does (same normalization as training),
hides a 3x3 in-plane block around random centres and learns to predict each centre from what is
left. Noise that is independent beyond one voxel cannot be predicted and is removed; structure that
repeats across neighbours (strokes, fibres, the surface) is kept. No labels are used.

    python train_denoiser.py --name denoiser_n2v_4k --scroll-ids 20260115000000 ... --steps 4000
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from mae_pretrain_nnunet import CropSampler, make_physical_sampler
from utils.config import Config
from utils.denoise import BlindSpotDenoiser, block_blind_spot_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="blind-spot denoiser pretraining")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--scroll-ids", type=int, nargs="+", required=True)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d-start", type=int, default=4)
    ap.add_argument("--d-end", type=int, default=28)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--centers", type=int, default=192, help="scored blind-spot centres per crop")
    ap.add_argument("--radius", type=int, default=1, help="in-plane half-width of the hidden block")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = Config()
    config.data.tile_size = 16
    config.data.depth = args.depth
    config.data.train_d_start = args.d_start
    config.data.train_d_end = args.d_end
    config.data.context_size = args.ctx
    zarr_path = config.data.zarr_path
    samplers = []
    import zarr
    for scroll_id in dict.fromkeys(args.scroll_ids):
        volume = zarr.open(os.path.join(zarr_path, f"{scroll_id}.zarr"), mode="r")
        height, width = int(volume.shape[1]), int(volume.shape[2])
        samplers.append(CropSampler(
            scroll_id, zarr_path, config, 0, (height // args.ctx) * args.ctx,
            0, (width // args.ctx) * args.ctx, "train", holdout_frac=0.05,
        ))
    sampler, groups = make_physical_sampler(samplers)
    print(f"[denoiser] {len(samplers)} volumes in {groups} physical groups", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = BlindSpotDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = sampler.sample(args.batch_size, rng).to(device)
        masked, centers = block_blind_spot_mask(batch, args.centers, args.radius, rng)
        prediction = model(masked)
        loss = (prediction[centers] - batch[centers]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"[denoiser] step {step}/{args.steps} loss={loss.item():.6f} lr={lr:.2e}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{args.name}.pth"
    torch.save(model.state_dict(), checkpoint)
    checkpoint.with_suffix(".complete.json").write_text(json.dumps({
        "scroll_ids": list(dict.fromkeys(args.scroll_ids)), "steps": args.steps, "ctx": args.ctx,
        "depth": args.depth, "radius": args.radius, "centers": args.centers,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[denoiser] saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
