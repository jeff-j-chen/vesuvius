"""plan R, step 2: fit utils/degrader.LearnedDegrader on plan degrade's tiles (crossres/PLAN.md section 0.7).

per tile: input = the 2.4 um target averaged over its 4 sub-slices (the pooled 9.36 um volume assembly would
make), mapped by the segment's robust affine (median / MAD) onto the native normalised distribution;
target = the real native 9.36 um slices, normalised with their norm_cache.json entry. loss: MSE inside the
valid target and the footprint. only tiles with midslice NCC >= --min-ncc train (a residual shift looks
exactly like blur, so the net would learn it as degradation).

pairs: those in the regime of the fragments to translate (high *-0.22m-*, low *-1.2m-113keV): the nine 0139
segments and 0814. 0841 (built with --include-holdouts --names p841) is the monitor only; without it every
tenth training tile monitors instead.

report (the gate, on the monitor crops):
  ratio            MSE(translated, real) / MSE(pooled, real); must be below 1
  spectrum_*       mean |log radial power - real's| over 8 bands, translated vs pooled; should move to real
  slice_stats_*    mean |per-slice mean / std - real's|, translated vs pooled

    python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan degrade --names w044 w059 w030 w043 w045 \
        w040 w041 w039 w035 seg46527
    python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan degrade --include-holdouts --names p841
    python crossres/train_degrader.py --dry-run
    python crossres/train_degrader.py            # -> models/degrader_pooled_native.pth + .json
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

from pairs import PAIRS  # noqa: E402
from utils.degrader import LearnedDegrader, REFERENCE_NATIVE_ID, robust_stats, to_reference  # noqa: E402
from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm  # noqa: E402


def in_regime(pair: dict) -> bool:
    return "-0.22m-" in pair["high"] and "-1.2m-113keV" in pair["low"]


class DegradeSegment:
    """one segment's degrade tiles in RAM: pooled high (float16), normalised native input, validity."""

    def __init__(self, pair: dict, plan: str, min_ncc: float):
        import zarr
        self.name, self.zid, self.scroll = pair["name"], int(pair["zid"]), pair["scroll"]
        folder = ROOT / "pairs" / plan / str(self.zid)
        meta = json.loads((folder / "meta.json").read_text())
        self.factor = int(meta["depth_factor"])
        if int(meta["xy_scale"]) != 1 or meta["z_range"] != [0, 28]:
            raise SystemExit(f"{self.name}: plan {plan} is not level-2 x1 over slices 0-28")
        group = zarr.open_group(str(folder / "pairs.zarr"), mode="r")
        ncc = np.asarray(meta.get("tile_ncc") or [np.inf] * group["input"].shape[0])
        self.kept = np.flatnonzero(ncc >= min_ncc)
        mean, std, g_min, g_max = load_cached_norm(str(self.zid), UNIFIED_CACHE_PATH)
        natives, pooled, valids = [], [], []
        for tile in self.kept:
            raw = group["input"][tile].astype(np.float32)
            native = np.clip(((raw - mean) / std - g_min) / (g_max - g_min + 1e-12), 0, 1) * (raw > 0)
            target = group["target"][tile]
            depth = target.shape[0] // self.factor
            high = target.reshape(depth, self.factor, *target.shape[1:])
            pool = high.astype(np.float32).mean(axis=1) * (high > 0).all(axis=1)
            natives.append(native.astype(np.float16))
            pooled.append(pool.astype(np.float16))
            valids.append(group["valid"][tile])
        self.native, self.pooled, self.valid = np.stack(natives), np.stack(pooled), np.stack(valids)
        self.stats = robust_stats(self.pooled[self.pooled > 0][::7].astype(np.float32))
        index = np.arange(len(self.kept))
        self.tiles = {"train": index[index % 10 != 0], "monitor": index[index % 10 == 0], "all": index}

    def sample(self, role: str, n: int, crop: int, rng):
        size = self.native.shape[-1]
        xs, ts, ws = [], [], []
        for _ in range(n):
            for _ in range(20):
                tile = int(rng.choice(self.tiles[role]))
                y, x = (int(v) for v in rng.integers(0, size - crop + 1, size=2))
                valid = self.valid[tile, y:y + crop, x:x + crop]
                if valid.mean() >= 0.9:
                    break
            pooled = self.pooled[tile, :, y:y + crop, x:x + crop].astype(np.float32)
            native = self.native[tile, :, y:y + crop, x:x + crop].astype(np.float32)
            xs.append(pooled)
            ts.append(native)
            ws.append((native > 0) & (pooled > 0) & valid[None])
        to = lambda items: torch.from_numpy(np.stack(items).astype(np.float32)).unsqueeze(1)
        return to(xs), to(ts), to(ws), self.stats


def radial_log_power(volume: torch.Tensor, bands: int = 8) -> torch.Tensor:
    """(B, 1, D, H, W) -> (bands,) mean log power per radial frequency band over all slices."""
    power = torch.fft.rfft2(volume.float()).abs().pow(2).mean(dim=(0, 1, 2))
    h, w = power.shape
    fy = torch.fft.fftfreq(h, device=volume.device).abs()[:, None]
    fx = torch.fft.rfftfreq(volume.shape[-1], device=volume.device)[None, :]
    radius = torch.sqrt(fy ** 2 + fx ** 2) / 0.5
    band = (radius.clamp(max=0.999) * bands).long()
    return torch.stack([power[band == b].mean().clamp(min=1e-12).log() for b in range(bands)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="degrader_pooled_native")
    ap.add_argument("--plan", default="degrade")
    ap.add_argument("--train-names", nargs="*", default=None, help="default: every built in-regime training pair")
    ap.add_argument("--monitor-names", nargs="*", default=["p841"])
    ap.add_argument("--min-ncc", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--log-int", type=int, default=100)
    ap.add_argument("--monitor-crops", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    built = lambda p: (ROOT / "pairs" / args.plan / str(p["zid"]) / "meta.json").exists()
    train_pairs = [p for p in PAIRS if built(p) and p["role"] == "train" and in_regime(p)
                   and (not args.train_names or p["name"] in args.train_names)]
    monitor_pairs = [p for p in PAIRS if built(p) and p["name"] in (args.monitor_names or ())]
    if not train_pairs:
        raise SystemExit(f"no in-regime pairs under crossres/pairs/{args.plan}; run build_pairs.py --plan degrade")
    train = [DegradeSegment(p, args.plan, args.min_ncc) for p in train_pairs]
    monitor = [DegradeSegment(p, args.plan, args.min_ncc) for p in monitor_pairs]
    monitor_role = "all" if monitor else "monitor"
    monitor = monitor or train
    sample_native = np.concatenate([s.native[s.tiles["train"]][:, :, ::4, ::4].ravel() for s in train])
    ref_median, ref_mad = robust_stats(sample_native.astype(np.float32))
    model = LearnedDegrader(args.width, args.layers, ref_median, ref_mad).to(dev)
    print(f"[degrader] train {[(s.name, len(s.tiles['train'])) for s in train]} "
          f"monitor {[s.name for s in monitor]} ({monitor_role}); NCC >= {args.min_ncc}; "
          f"reference median/MAD {ref_median:.4f}/{ref_mad:.4f}; params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    def batch(segments, role, n, generator):
        parts = [segments[int(generator.integers(len(segments)))].sample(role, 1, args.crop, generator)
                 for _ in range(n)]
        x = torch.cat([to_reference(p[0], p[3], model) for p in parts])
        return x, torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts])

    mon = batch(monitor, monitor_role, args.monitor_crops, np.random.default_rng(args.seed + 1))

    def mse(pred, target, weight):
        return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)

    @torch.no_grad()
    def evaluate():
        model.eval()
        errors = {"translated": [0.0, 0.0], "pooled": [0.0, 0.0]}
        preds = {"translated": [], "pooled": []}
        for i in range(0, len(mon[0]), 16):
            x, t, w = (a[i:i + 16].to(dev) for a in mon)
            for key, pred in (("translated", model(x)), ("pooled", x)):
                errors[key][0] += float(((pred - t) ** 2 * w).sum())
                errors[key][1] += float(w.sum())
                preds[key].append(pred * (w > 0))
        model.train()
        real = mon[1].to(dev) * (mon[2].to(dev) > 0)
        real_power = radial_log_power(real)
        report = {f"mse_{k}": num / max(den, 1e-8) for k, (num, den) in errors.items()}
        report["ratio"] = report["mse_translated"] / report["mse_pooled"]
        for key, items in preds.items():
            pred = torch.cat(items)
            report[f"spectrum_{key}"] = float((radial_log_power(pred) - real_power).abs().mean())
            mean_gap = (pred.mean(dim=(0, 1, 3, 4)) - real.mean(dim=(0, 1, 3, 4))).abs().mean()
            std_gap = (pred.std(dim=(0, 1, 3, 4)) - real.std(dim=(0, 1, 3, 4))).abs().mean()
            report[f"slice_stats_{key}"] = float(mean_gap + std_gap)
        return report

    if args.dry_run:
        x, t, w = batch(train, "train", args.batch_size, rng)
        loss = mse(model(x.to(dev)), t.to(dev), w.to(dev))
        loss.backward()
        print(f"[degrader] dry-run batch {tuple(x.shape)} loss={float(loss):.5f} (untrained == pooled) "
              f"monitor {json.dumps({k: round(v, 5) for k, v in evaluate().items()})}", flush=True)
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
    save_path.parent.mkdir(exist_ok=True)
    history, best, running = [], None, []
    started = time.time()

    def log(step, values):
        for tag, value in values.items():
            writer.add_scalar(f"Degrader/{tag}", float(value), step)
        history.append({"step": step, **values})
        (Path(run_dir) / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
        print(f"[degrader] step {step}/{args.steps} " + " ".join(f"{k}={float(v):.5f}" for k, v in values.items())
              + f" {time.time() - started:.0f}s", flush=True)

    log(0, evaluate())
    for step in range(1, args.steps + 1):
        x, t, w = batch(train, "train", args.batch_size, rng)
        loss = mse(model(x.to(dev)), t.to(dev), w.to(dev))
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
            if best is None or report["ratio"] < best["ratio"]:
                best = {"step": step, **report}
                torch.save({"state_dict": model.state_dict(), "config": model.config(), "report": best},
                           save_path)
    writer.close()
    sidecar = {
        "pairs": [s.name for s in train], "monitor": [s.name for s in monitor], "monitor_role": monitor_role,
        "min_ncc": args.min_ncc, "tiles": {s.name: int(len(s.kept)) for s in train + monitor},
        "input_normalisation": "pooled 2.4 um: per-segment median/MAD -> reference median/MAD",
        "target_normalisation": "native norm_cache.json entry, clipped to [0, 1]",
        "reference_native_id": REFERENCE_NATIVE_ID, "report": best,
    }
    save_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    print(f"[degrader] best step {best['step']}: ratio={best['ratio']:.4f} -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
