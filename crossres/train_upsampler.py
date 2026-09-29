"""plan U, step 1: fit utils/upsampler.LearnedUpsampler on plan X's tiles (crossres/PLAN.md section 0.6).

input: 8 x 96 x 96 at 9.36 um (normalized exactly as fine-tuning); target: the co-registered 2.4 um render
as 32 x 192 x 192 (4x depth, 2x x/y), quantile-matched per segment onto the input's intensities
(PairedSegment in mae_pretrain_crossres.py). loss: MSE over valid target voxels.

report (fixed monitor crops from tiles never trained on, one per 10 tiles per segment):
  ratio_vs_trilinear  monitor MSE / trilinear MSE; < 1 means the learned detail is real
  ratio_vs_linear3    monitor MSE / a per-segment least-squares 3x3x3 filter on the trilinear output

holdout pairs (0841, 0009B) are excluded unless --include-holdout-pairs.

    python crossres/train_upsampler.py --dry-run
    python crossres/train_upsampler.py            # -> models/upsampler_learned_xy2_d4.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from mae_pretrain_crossres import PairedGroups, PairedSegment, apply_linear3, fit_linear3, upsample  # noqa: E402
from pairs import PAIRS  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.upsampler import LearnedUpsampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="upsampler_learned_xy2_d4")
    ap.add_argument("--plan", default="xyz")
    ap.add_argument("--include-holdout-pairs", action="store_true")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--log-int", type=int, default=100)
    ap.add_argument("--monitor-crops", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.depth, cfg.data.context_size = 8, 96
    cfg.data.train_d_start, cfg.data.train_d_end = 8, 20
    dev = cfg.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    wanted = [p for p in PAIRS if (ROOT / "pairs" / args.plan / str(p["zid"]) / "meta.json").exists()
              and (args.include_holdout_pairs or p["role"] != "holdout")]
    if not wanted:
        raise SystemExit(f"no built pairs under crossres/pairs/{args.plan}")
    segments = [PairedSegment(p, cfg, "global", args.plan) for p in wanted]
    factor, scale = segments[0].factor, segments[0].scale
    if (factor, scale) != (4, 2):
        raise SystemExit(f"expected plan xyz tiles (x4 depth, x2 xy), got x{factor} depth, x{scale} xy")
    train, monitor = PairedGroups(segments, "train"), PairedGroups(segments, "monitor")
    print(f"[upsampler] {len(segments)} segments {[s.name for s in segments]}; scroll weights {train.describe()}",
          flush=True)
    mon_x, mon_t, mon_v, mon_z = monitor.sample(args.monitor_crops, np.random.default_rng(args.seed + 1))
    linear3 = {s.zid: fit_linear3(s, factor, scale, dev, np.random.default_rng(args.seed + 2)) for s in segments}

    model = LearnedUpsampler(factor, scale, args.width, args.layers).to(dev)
    print(f"[upsampler] params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    def mse(pred, target, valid):
        weight = valid.expand_as(target)
        return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)

    @torch.no_grad()
    def evaluate():
        model.eval()
        sums = {"monitor": [0.0, 0.0], "trilinear": [0.0, 0.0], "linear3": [0.0, 0.0]}
        per_segment = {}
        for i in range(0, len(mon_x), 16):
            x, t, v = (a[i:i + 16].to(dev) for a in (mon_x, mon_t, mon_v))
            zids = mon_z[i:i + 16]
            preds = {"monitor": model(x), "trilinear": upsample(x, factor, scale),
                     "linear3": torch.cat([apply_linear3(x[j:j + 1], *linear3[int(z)], factor, scale)
                                           for j, z in enumerate(zids)])}
            weight = v.expand_as(t)
            for key, pred in preds.items():
                err = (pred - t) ** 2 * weight
                sums[key][0] += float(err.sum())
                sums[key][1] += float(weight.sum())
                for j, z in enumerate(zids):
                    entry = per_segment.setdefault(int(z), {k: [0.0, 0.0] for k in sums})
                    entry[key][0] += float(err[j].sum())
                    entry[key][1] += float(weight[j].sum())
        model.train()
        out = {key: num / max(den, 1e-8) for key, (num, den) in sums.items()}
        out["ratio_vs_trilinear"] = out["monitor"] / out["trilinear"]
        out["ratio_vs_linear3"] = out["monitor"] / out["linear3"]
        names = {s.zid: s.name for s in segments}
        out["per_segment_ratio_vs_trilinear"] = {
            names[z]: (e["monitor"][0] / max(e["monitor"][1], 1e-8)) / (e["trilinear"][0] / max(e["trilinear"][1], 1e-8))
            for z, e in per_segment.items()}
        return out

    if args.dry_run:
        x, t, v, _ = train.sample(args.batch_size, rng)
        loss = mse(model(x.to(dev)), t.to(dev), v.to(dev))
        loss.backward()
        report = evaluate()
        print(f"[upsampler] dry-run batch {tuple(x.shape)} -> {tuple(t.shape)} loss={float(loss):.5f} "
              f"(untrained == trilinear) monitor ratio_vs_trilinear={report['ratio_vs_trilinear']:.4f} "
              f"ratio_vs_linear3={report['ratio_vs_linear3']:.4f}", flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    from torch.utils.tensorboard import SummaryWriter
    run_dir = os.path.join("runs_mae", f"{args.name}_{time.strftime('%m%d_%H-%M-%S')}")
    writer = SummaryWriter(run_dir)
    save_path = Path("models") / f"{args.name}.pth"
    history, best = [], None
    started = time.time()

    def log(step, values):
        scalars = {k: v for k, v in values.items() if not isinstance(v, dict)}
        for tag, value in scalars.items():
            writer.add_scalar(f"Upsampler/{tag}", float(value), step)
        history.append({"step": step, **values})
        (Path(run_dir) / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
        print(f"[upsampler] step {step}/{args.steps} " + " ".join(f"{k}={float(v):.5f}" for k, v in scalars.items())
              + f" {time.time() - started:.0f}s", flush=True)

    log(0, evaluate())
    running = []
    for step in range(1, args.steps + 1):
        x, t, v, _ = train.sample(args.batch_size, rng)
        loss = mse(model(x.to(dev)), t.to(dev), v.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running.append(float(loss))
        if step % args.log_int == 0 or step == args.steps:
            report = evaluate()
            log(step, {"train": float(np.mean(running)), **report})
            running = []
            # keep the checkpoint with the best monitor ratio, so a late overfit cannot ship
            if best is None or report["ratio_vs_trilinear"] < best["ratio_vs_trilinear"]:
                best = {"step": step, **report}
                torch.save({"state_dict": model.state_dict(), "config": model.config(), "report": best,
                            "pairs": [s.name for s in segments], "plan": args.plan}, save_path)
    writer.close()
    print(f"[upsampler] best step {best['step']}: ratio_vs_trilinear={best['ratio_vs_trilinear']:.4f} "
          f"ratio_vs_linear3={best['ratio_vs_linear3']:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
