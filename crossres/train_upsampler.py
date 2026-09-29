"""train the learned input upsampler (utils/upsampler.py) on the paired tiles from build_pairs.py.

input: normalized 9.36 um crops (8 x 96 x 96); target: the co-registered ~2.4 um data, intensity-matched
onto the input's scale (--plan xyz: 32 x 192 x 192; --plan depth: 32 x 96 x 96). loss: MSE where the
target is valid. the head starts as exact trilinear interpolation (v8-in's fixed upsample), so the
monitor line "vs trilinear" is the whole story: below 1.0 means the learned detail is real.

--steps 0 saves the untrained module: pure trilinear, the control that isolates learned vs fixed.

UNTESTED: written without running. run --dry-run first.

    python crossres/train_upsampler.py --plan xyz --name upsampler_xy2_d4 --exclude-holdout-pairs
    python crossres/train_upsampler.py --plan xyz --name upsampler_trilinear_xy2_d4 --steps 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from mae_pretrain_crossres import PairedGroups, PairedTiles, build_config  # noqa: E402
from pairs import PAIRS  # noqa: E402
from utils.upsampler import LearnedUpsampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", choices=("xyz", "depth"), default="xyz")
    ap.add_argument("--exclude-holdout-pairs", action="store_true", help="drop 0841's pair (the strict holdout)")
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-int", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = build_config(SimpleNamespace(fiber_coordinate_branch=False, data_norm_mode="global",
                                       depth=8, d_start=8, d_end=20, ctx=96))
    dev = cfg.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    wanted = [p for p in PAIRS if (ROOT / "pairs" / args.plan / str(p["zid"]) / "meta.json").exists()
              and not (args.exclude_holdout_pairs and p["role"] == "holdout")]
    if not wanted:
        raise SystemExit(f"no built pairs under crossres/pairs/{args.plan}")
    train, monitor = [], []
    for pair in wanted:
        for role, bucket in (("train", train), ("monitor", monitor)):
            tiles = PairedTiles(pair["zid"], cfg, role, "global", args.plan)
            tiles.scroll = pair["scroll"]
            bucket.append(tiles)
    scale, factor = train[0].scale, train[0].factor
    paired, paired_monitor = PairedGroups(train), PairedGroups(monitor)
    model = LearnedUpsampler(scale, factor, args.channels, args.layers).to(dev)
    print(f"[upsampler] {len(wanted)} pairs, x{scale} x/y, x{factor} depth, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    def loss_of(x, target, valid):
        pred = model(x)
        weight = valid.expand_as(target)
        return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)

    def evaluate(batches=8):
        model.eval()
        learned = baseline = 0.0
        with torch.no_grad():
            for _ in range(batches):
                x, target, valid = (t.to(dev) for t in paired_monitor.sample(args.batch_size, rng))
                learned += float(loss_of(x, target, valid))
                weight = valid.expand_as(target)
                trilinear = torch.nn.functional.interpolate(x, scale_factor=(factor, scale, scale),
                                                            mode="trilinear", align_corners=False)
                baseline += float(((trilinear - target) ** 2 * weight).sum() / (weight.sum() + 1e-8))
        model.train()
        return learned / batches, baseline / batches

    if args.dry_run:
        x, target, valid = paired.sample(2, rng)
        print(f"[upsampler] dry-run: input {tuple(x.shape)} -> output {tuple(model(x.to(dev)).shape)} "
              f"target {tuple(target.shape)}")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    history = []
    best = math.inf
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{args.name}.pth")

    def save():
        torch.save({"state_dict": model.state_dict(), "config": model.config(), "plan": args.plan,
                    "pairs": [p["name"] for p in wanted]}, path)

    save()
    started = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for group in opt.param_groups:
            group["lr"] = lr
        x, target, valid = (t.to(dev) for t in paired.sample(args.batch_size, rng))
        loss = loss_of(x, target, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_int == 0 or step == args.steps:
            learned, baseline = evaluate()
            history.append({"step": step, "train": float(loss), "monitor": learned, "trilinear": baseline})
            print(f"[upsampler] step {step}/{args.steps} train={float(loss):.5f} monitor={learned:.5f} "
                  f"trilinear={baseline:.5f} (x{learned / baseline:.3f}) {time.time() - started:.0f}s", flush=True)
            if learned < best:
                best = learned
                save()
    learned, baseline = evaluate(32)
    report = {"monitor": learned, "trilinear": baseline, "ratio": learned / baseline, "history": history}
    Path(path).with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[upsampler] saved {path}; monitor/trilinear = {learned / baseline:.3f}", flush=True)


if __name__ == "__main__":
    main()
