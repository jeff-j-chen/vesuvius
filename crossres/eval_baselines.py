"""fair baselines for the cross-resolution heads: every method scored on the same held-out (monitor) crops.

per plan (depth: 96^2 x 32, xyz: 192^2 x 32):
  trilinear        fixed interpolation of the 9.36 um input
  linear3_global   ONE least-squares 3x3x3 filter + bias on the trilinear output, fitted on training tiles of
                   all segments together (the fair bar: usable on a new scroll)
  linear3_segment  the same fitted per segment (the training-time health check; needs 2.4 um data of that scroll)
  D / X head       the generator checkpoints (models/mae_crossres_{depth,xyz}.generator.pth), unmasked input
  upsampler        models/upsampler_learned_xy2_d4.pth (xyz only)

runs on CPU with lazily opened tiles, so it can run beside a training job.

    python crossres/eval_baselines.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from mae_pretrain_crossres import CrossResMAE, PairedSegment, _neighbourhood, apply_linear3, build_config, upsample  # noqa: E402
from pairs import PAIRS  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.model import create_model  # noqa: E402
from utils.upsampler import load_upsampler  # noqa: E402

MODELS = ROOT.parent / "models"


def data_cfg():
    cfg = Config()
    cfg.data.depth, cfg.data.context_size = 8, 96
    cfg.data.train_d_start, cfg.data.train_d_end = 8, 20
    return cfg


def fit(segments, factor, scale, rng, crops_per_segment, chunk=4, keep_frac=0.25):
    """least-squares 3x3x3 filter + bias; one per segment and one over all segments pooled."""
    per_segment, total_xtx, total_xty = {}, torch.zeros(28, 28, dtype=torch.float64), torch.zeros(28, dtype=torch.float64)
    for seg in segments:
        xtx, xty = torch.zeros(28, 28, dtype=torch.float64), torch.zeros(28, dtype=torch.float64)
        for _ in range(crops_per_segment // chunk):
            x, target, valid = seg.sample("train", chunk, rng)
            feats = _neighbourhood(upsample(x, factor, scale))
            mask = (valid.expand_as(target)[:, 0] > 0) & (torch.rand(target[:, 0].shape) < keep_frac)
            design = feats.permute(0, 2, 3, 4, 1)[mask].double()
            design = torch.cat([design, torch.ones_like(design[:, :1])], dim=1)
            xtx += design.T @ design
            xty += design.T @ target[:, 0][mask].double()
        per_segment[seg.zid] = solve(xtx, xty)
        total_xtx += xtx
        total_xty += xty
    return per_segment, solve(total_xtx, total_xty)


def solve(xtx, xty):
    w = torch.linalg.solve(xtx + 1e-6 * torch.eye(28, dtype=torch.float64), xty)
    return w[:27].float().view(1, 1, 3, 3, 3), w[27].float()


def load_generator(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = types.SimpleNamespace(fiber_coordinate_branch=False, data_norm_mode="global", depth=checkpoint["depth"],
                                 d_start=8, d_end=20, ctx=checkpoint["ctx"])
    backbone, _ = create_model(build_config(args))
    model = CrossResMAE(backbone, checkpoint["depth"], scale=checkpoint["xy_scale"], depth_factor=checkpoint["depth_factor"],
                        head=checkpoint.get("sr_head", "small"))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def load_upsampler_copy(path):
    # the upsampler run may be rewriting its best checkpoint; read a copy
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / path.name
        shutil.copyfile(path, copy)
        return load_upsampler(str(copy)).eval(), torch.load(copy, map_location="cpu", weights_only=False).get("report")


def evaluate(plan, args):
    rng = np.random.default_rng(args.seed)
    pairs = [p for p in PAIRS if p["role"] == "train" and (ROOT / "pairs" / plan / str(p["zid"]) / "meta.json").exists()]
    segments = [PairedSegment(p, data_cfg(), "global", plan, lazy=True) for p in pairs]
    factor, scale = segments[0].factor, segments[0].scale
    print(f"[{plan}] {len(segments)} segments, x{factor} depth, x{scale} xy: fitting filters", flush=True)
    per_segment, global_filter = fit(segments, factor, scale, rng, args.fit_crops)

    methods = {}
    tag = "D" if plan == "depth" else "X"
    for head in sorted(MODELS.glob(f"mae_crossres_{plan}*.generator.pth")):
        if ".step" in head.name:
            continue
        version = head.name.removeprefix(f"mae_crossres_{plan}").removesuffix(".generator.pth").lstrip("_") or "v1"
        methods[f"{tag} head {version}"] = load_generator(head)
    up_report = None
    if plan == "xyz" and (MODELS / "upsampler_learned_xy2_d4.pth").is_file():
        methods["upsampler"], up_report = load_upsampler_copy(MODELS / "upsampler_learned_xy2_d4.pth")

    names = ["trilinear", "linear3_global", "linear3_segment", *methods]
    sums = {name: [0.0, 0.0] for name in names}
    by_segment = {}
    monitor_rng = np.random.default_rng(args.seed + 1)
    for seg in segments:
        seg_sums = {name: [0.0, 0.0] for name in names}
        for _ in range(args.monitor_crops // 4):
            x, t, v = seg.sample("monitor", 4, monitor_rng)
            with torch.no_grad():
                preds = {"trilinear": upsample(x, factor, scale),
                         "linear3_global": apply_linear3(x, *global_filter, factor, scale),
                         "linear3_segment": apply_linear3(x, *per_segment[seg.zid], factor, scale)}
                for name, module in methods.items():
                    preds[name] = module(x) if name == "upsampler" else module(x, x)[1]
            weight = v.expand_as(t)
            for name, pred in preds.items():
                err = float(((pred.float() - t) ** 2 * weight).sum())
                for bucket in (sums, seg_sums):
                    bucket[name][0] += err
                    bucket[name][1] += float(weight.sum())
        by_segment[seg.name] = {n: s[0] / max(s[1], 1) for n, s in seg_sums.items()}
    overall = {n: s[0] / max(s[1], 1) for n, s in sums.items()}
    return {"overall": overall, "by_segment": by_segment, "upsampler_training_report": up_report,
            "monitor_crops_per_segment": args.monitor_crops}


def table(plan, result):
    overall = result["overall"]
    tri, glob = overall["trilinear"], overall["linear3_global"]
    print(f"\n[{plan}] MSE on held-out tiles ({result['monitor_crops_per_segment']} crops per segment)")
    print(f"{'method':18s} {'MSE':>9s} {'vs trilinear':>13s} {'vs global filter':>17s}")
    for name, value in overall.items():
        print(f"{name:18s} {value:9.5f} {100 * (1 - value / tri):+12.1f}% {100 * (1 - value / glob):+16.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="*", default=["depth", "xyz"])
    ap.add_argument("--fit-crops", type=int, default=64, help="training crops per segment for the filters")
    ap.add_argument("--monitor-crops", type=int, default=24, help="held-out crops per segment")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results = {}
    for plan in args.plans:
        results[plan] = evaluate(plan, args)
        table(plan, results[plan])
    out = ROOT / "eval_baselines.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
