"""grad_affinity: per-patch, per-batch gradient cosine against the target sheet, for several checkpoints at once.

for every labelled patch of the whole corpus (plus the new 0846A set and the target) it draws K batches from the patch's own
training stream, takes the gradient of the masked tile BCE (ink cells vs supervised negatives; no nnPU/ranking/aux terms) at each
checkpoint (eval mode, no augmentation), and compares it with the target patch's mean gradient. a positive cosine means a step on
that patch also lowers the target's loss. the same batches feed every checkpoint.

reported per patch and checkpoint, for feature weights (feat), biases/norms/heads (calib) and all parameters:
  cos   cosine of the K-batch mean gradient with the target's;  sd  spread of the single-batch cosines
  ccos  the same after removing the direction shared by all patches (feat only)
  rel   reliability of the K-batch mean (split-half cosine, Spearman-Brown);  corr  cos / sqrt(rel_patch * rel_target)
  r1    mean cosine between two single-batch gradients of the same patch (is one batch enough?)
patches without a cue map are fed a flat cue (128/255), the input the cue-dropout recipe trains on; fine scans read
their .matched volumes and get the native noise bank, as in training.

usage: python3 grad_affinity.py [--batches 12] [--patches id,id] [--target id] [--ckpt name=path ...] [--out output/grad_affinity]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "old"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_iter2 as base  # noqa: E402

DEFAULT_CKPTS = {
    "flip_2": "models/runs_iter2/iter2_flip_2/model_epoch_6.pth",
    "asf_cue_labelsv2": "models/runs_iter2/iter2_asf_cue_labelsv2/model_epoch_10.pth",
    "asf_cue": "models/runs_iter2/iter2_asf_cue/model_epoch_10.pth",
}
FLAT_CUE = 128.0 / 255.0
SUBSETS = ("feat", "calib", "all")


def _flat_cue_block(origin, ctx, shift=None):
    return np.full((1, ctx, ctx), FLAT_CUE, dtype=np.float32)


def build_config(target: int):
    from utils.config import Config, add_0846a_train_patch

    config = Config()
    base._old_plus_new(config)
    add_0846a_train_patch(config, target)
    data = config.data
    data.extra_hard_negative_scroll_ids = sorted({*data.extra_hard_negative_scroll_ids, target})
    data.vis_scroll_ids = []
    data.whole_patch_valid_scroll_ids = [s for s in data.whole_patch_valid_scroll_ids if int(s) != target]
    data.selective_chunk_preload = False
    data.vis_preload_persistent = False
    data.ram_safe_vis = True
    config.model.input_cue = "pre_clahe"
    data.cue_dir = base.CUE_ASF_DIR
    config.model.compile_model = False
    config.init_weights = ""
    config.dl.num_workers = 2
    # patches whose volume or supervision inputs are not on this machine (yet) are left out
    suffix = data.zarr_suffix or {}

    def ready(sid):
        sid = int(sid)
        needed = [
            os.path.join(data.zarr_path, f"{sid}{suffix.get(str(sid), '')}.zarr"),
            os.path.join("masks", f"{sid}.png"),
            os.path.join(data.inklabel_dir, f"{sid}.png"),
            os.path.join(data.train_mask_dir, f"{sid}.png"),
            os.path.join(data.surface_label_dir, str(sid), "depth.npy"),
            os.path.join(data.surface_label_dir, str(sid), "confidence.npy"),
        ]
        return all(os.path.exists(path) for path in needed)

    present = {int(s) for ids in data.train_scroll_dict.values() for s in ids if ready(s)}
    skipped = sorted({int(s) for ids in data.train_scroll_dict.values() for s in ids} - present)
    print(f"[grad] {len(present)} patches ready, {len(skipped)} not: {skipped}", flush=True)
    weights = dict(zip(data.train_scroll_dict, data.train_scroll_weights))
    data.train_scroll_dict = {k: [s for s in ids if int(s) in present] for k, ids in data.train_scroll_dict.items()}
    data.train_scroll_dict = {k: ids for k, ids in data.train_scroll_dict.items() if ids}
    data.train_scroll_weights = [weights[k] for k in data.train_scroll_dict]
    data.scrolls = [s for s in data.scrolls if int(s.scroll_id) in present]
    config.tra.dann_n_domains = len(data.train_scroll_dict)
    return config


def load_models(config, ckpts: dict):
    from utils.model import create_model

    models = {}
    for name, path in ckpts.items():
        model, _ = create_model(config)
        state = torch.load(REPO / path, map_location=config.device, weights_only=True)
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        models[name] = model
    return models


def unpack(batch, config):
    from utils.dataloader import needs_domain_ids, needs_patch_ids

    images, labels, mask = batch[:3]
    i = 3
    domain = None
    if needs_domain_ids(config):
        domain, i = batch[i], i + 1
    if needs_patch_ids(config):
        i += 1
    if bool(config.tra.character_macro_metrics):
        i += 1
    offsets = None
    if bool(config.data.target_aware_ctx_jitter):
        offsets, i = batch[i], i + 1
    depth = conf = None
    if any(bool(getattr(config.model, k, False)) for k in ("new_learned_surface", "better_surface", "surface_teacher_input")):
        depth, conf = batch[i], batch[i + 1]
    if offsets is not None and tuple(offsets.shape[1:]) != (2,):
        raise RuntimeError(f"unexpected batch layout, offsets shape {tuple(offsets.shape)}")
    return images, labels, mask, domain, offsets, depth, conf


def batch_gradient(model, criterion, parts, config):
    """flat fp32 gradient of the masked tile BCE for one batch, plus (positive cells, supervised cells)."""
    device = config.device
    images, labels, mask, domain, offsets, depth, conf = parts
    images = images.to(device)
    B = images.size(0)
    labels = labels.to(device).view(B, -1)
    soft = (labels > 0.5) & (labels < 1.0)
    level = torch.where(soft, labels, torch.ones_like(labels))
    labels = torch.where(soft, torch.ones_like(labels), labels)
    mask = mask.to(device).view(B, -1)
    mask = (mask > 0).float() if mask.shape[1] == labels.shape[1] else (mask.sum(dim=1) > 0).float().unsqueeze(1)
    if mask.sum() <= 0:
        return None
    kwargs = dict(
        target_offsets=offsets.to(device).view(B, 2) if offsets is not None else None,
        teacher_surface_depth=depth.to(device).float() if depth is not None else None,
        teacher_surface_confidence=conf.to(device).float() if conf is not None else None,
    )
    model.zero_grad(set_to_none=True)
    if hasattr(model, "forward_with_extras") and bool(config.model.dual_scale):
        outputs = model.forward_with_extras(
            images, grl_scale=1.0, domain_ids=domain.to(device).view(-1) if domain is not None else None,
            sagnet_grl_scale=0.1, **kwargs,
        )[0]
    else:
        outputs = model(images, **kwargs)
    if outputs.dim() == 4:
        outputs = outputs.flatten(1).max(dim=1, keepdim=True).values
    targets = torch.where(labels < 0, torch.zeros_like(labels), labels).float()
    targets = torch.where(labels > 0.5, targets * level, targets)
    loss = (criterion(outputs.float(), targets) * mask).sum() / mask.sum().clamp(min=1.0)
    loss.backward()
    flat = torch.cat([p.grad.reshape(-1).float() if p.grad is not None else torch.zeros(p.numel(), device=device)
                      for p in model.parameters() if p.requires_grad])
    positives = float(((targets > 0.5) & (mask > 0)).sum())
    return flat, positives, float(mask.sum())


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-20))


def patch_stats(G):
    """G (K, N): single-batch reliability r1, split-half reliability of the mean, and the mean gradient."""
    K = G.size(0)
    unit = G / G.norm(dim=1, keepdim=True).clamp(min=1e-20)
    gram = unit @ unit.T
    r1 = float((gram.sum() - gram.diag().sum()) / max(1, K * (K - 1)))
    half = cosine(G[: K // 2].sum(0), G[K // 2: 2 * (K // 2)].sum(0)) if K >= 2 else float("nan")
    rel = 2 * half / (1 + half) if half > -0.99 else float("nan")
    return G.mean(0), r1, half, rel


def parameter_subsets(model, device):
    """masks over the flat gradient: feat = conv/linear weights (what learns features), calib = biases, norm scales, heads, lse_r."""
    calib = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            is_calib = p.dim() <= 1 or "head" in name or name.startswith("lse_r")
            calib.append(torch.full((p.numel(),), is_calib, dtype=torch.bool))
    calib = torch.cat(calib).to(device)
    return {"feat": ~calib, "calib": calib, "all": torch.ones_like(calib)}


def run(args):
    from utils.norm import ensure_surface_anchors
    from utils.training_utils import create_loss_function
    from train import Trainer

    target = int(args.target)
    config = build_config(target)
    os.chdir(REPO)
    ensure_surface_anchors(dict.fromkeys(config.scroll_ids()), config.data.zarr_path, surface_dir=config.data.surface_label_dir)
    ckpts = {k: v for k, v in (c.split("=", 1) for c in args.ckpt)} if args.ckpt else DEFAULT_CKPTS
    models = load_models(config, ckpts)
    n_params = sum(p.numel() for p in next(iter(models.values())).parameters() if p.requires_grad)
    print(f"[grad] {len(models)} checkpoints, {n_params:,} parameters", flush=True)
    pos_weight = float(config.tra.tile_pos_weight or 0.0)
    criterion = create_loss_function(torch.tensor([pos_weight]) if pos_weight > 0 else None, config)

    warm = Trainer.__new__(Trainer)
    warm.c = config
    warm._setup_data()
    train_sets = warm._scroll_train_sets
    domain_of = {int(s): name for name, ids in config.data.train_scroll_dict.items() for s in ids}
    flat_cue = set()
    for sid, ds in train_sets.items():
        if not os.path.isfile(ds._cue_path):
            ds._cue_block = _flat_cue_block
            flat_cue.add(int(sid))
    order = [target] + [int(s) for s in train_sets if int(s) != target]
    if args.patches:
        wanted = {int(s) for s in args.patches.split(",")}
        order = [s for s in order if s == target or s in wanted]
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    subsets = parameter_subsets(next(iter(models.values())), config.device)
    print("[grad] " + ", ".join(f"{k}={int(v.sum()):,}" for k, v in subsets.items()), flush=True)
    rows, means = [], {name: {} for name in models}
    target_state = {}
    for index, sid in enumerate(order):
        ds = train_sets[sid]
        ds.apply_transforms = False
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, drop_last=True,
                            worker_init_fn=partial(_worker_seed, base=sid % 100000),
                            prefetch_factor=2 if args.workers else None)
        grads = {name: [] for name in models}
        pos = cells = 0.0
        started = time.time()
        for batch in loader:
            parts = unpack(batch, config)
            results = {name: batch_gradient(model, criterion, parts, config) for name, model in models.items()}
            if any(r is None for r in results.values()):
                continue
            for name, (flat, p, c) in results.items():
                grads[name].append(flat)
            pos, cells = pos + p, cells + c
            if len(grads[next(iter(models))]) >= args.batches:
                break
        del loader
        K = len(grads[next(iter(models))])
        if K < 2:
            print(f"[grad] {sid}: only {K} usable batch(es), skipped", flush=True)
            continue
        for name in models:
            G = torch.stack(grads[name])
            means[name][sid] = G.mean(0).cpu()
            row = dict(model=name, scroll_id=sid, domain=domain_of.get(sid, "?"), is_target=int(sid == target),
                       flat_cue=int(sid in flat_cue), batches=K, pos_frac=pos / max(cells, 1.0))
            for subset, mask in subsets.items():
                Gs = G[:, mask]
                mean, r1, half, rel = patch_stats(Gs)
                row.update({f"r1_{subset}": r1, f"rel_{subset}": rel, f"norm_{subset}": float(Gs.norm(dim=1).mean())})
                if sid == target:
                    target_state[(name, subset)] = (mean, rel)
                    row.update({f"cos_{subset}": 1.0, f"sd_{subset}": float("nan"), f"corr_{subset}": 1.0})
                    continue
                gT, relT = target_state[(name, subset)]
                unit = Gs / Gs.norm(dim=1, keepdim=True).clamp(min=1e-20)
                cb = unit @ (gT / gT.norm().clamp(min=1e-20))
                cm = cosine(mean, gT)
                reliable = rel >= 0.3 and relT >= 0.3
                row.update({f"cos_{subset}": cm, f"sd_{subset}": float(cb.std()),
                            f"corr_{subset}": cm / (rel * relT) ** 0.5 if reliable else float("nan")})
            rows.append(row)
        del grads
        torch.cuda.empty_cache()
        print(f"[grad] {index + 1}/{len(order)} {sid} {domain_of.get(sid)} K={K} {time.time() - started:.0f}s "
              + " ".join(f"{r['model']}:{r['cos_feat']:+.3f}" for r in rows[-len(models):]), flush=True)
    add_centered(rows, means, target, subsets["feat"].cpu())
    with open(out / "grad_affinity.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_grads:
        for name in models:
            torch.save(means[name], out / f"mean_grads_{name}.pt")
    summarize(rows, list(models), out)


def add_centered(rows, means, target, feat_mask):
    """ccos_feat: cosine of patch and target mean gradients after removing the mean unit direction over all patches.

    a raw cosine is dominated by the direction every patch shares (e.g. 'predict less ink everywhere'); the centered one
    keeps only what is specific to a patch."""
    for name, per_patch in means.items():
        ids = list(per_patch)
        unit = torch.stack([per_patch[s][feat_mask] for s in ids]).cuda()
        unit = unit / unit.norm(dim=1, keepdim=True).clamp(min=1e-20)
        common = unit.mean(0)
        centered = unit - common
        centered_target = centered[ids.index(target)]
        for row in rows:
            if row["model"] == name:
                row["ccos_feat"] = cosine(centered[ids.index(row["scroll_id"])], centered_target)
                row["common_share"] = float(common.norm() ** 2)
        del unit, centered


def _worker_seed(worker_id, base):
    import random

    from utils.dataloader import limit_cpu_threads

    limit_cpu_threads(1)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def _rank(values):
    order = np.argsort(values)
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def summarize(rows, names, out):
    others = [r for r in rows if not r["is_target"]]
    lines = []
    for name in names:
        mine = [r for r in others if r["model"] == name]
        lines.append(f"\n== {name} ==  (target excluded, {len(mine)} patches; feat = conv/linear weights, calib = biases/norms/heads)")
        lines.append("single-batch r1 (median): " + "  ".join(
            f"{s} {np.median([r[f'r1_{s}'] for r in mine]):+.3f}" for s in SUBSETS)
            + "   K-mean reliability (median): " + "  ".join(
            f"{s} {np.nanmedian([r[f'rel_{s}'] for r in mine]):.2f}" for s in SUBSETS))
        lines.append(f"{'patch':<16}{'domain':<20}{'cue':<5}{'pos%':>6}{'cos_feat':>9}{'ccos':>7}{'sd_b':>7}{'corr':>8}{'rel':>6}{'r1':>7}"
                     f"{'cos_calib':>10}")
        for r in sorted(mine, key=lambda r: -r["ccos_feat"]):
            lines.append(f"{r['scroll_id']:<16}{r['domain']:<20}{'flat' if r['flat_cue'] else 'real':<5}{100 * r['pos_frac']:>6.1f}"
                         f"{r['cos_feat']:>+9.3f}{r['ccos_feat']:>+7.2f}{r['sd_feat']:>7.3f}{r['corr_feat']:>+8.2f}{r['rel_feat']:>6.2f}"
                         f"{r['r1_feat']:>+7.3f}{r['cos_calib']:>+10.3f}")
        lines.append(f"share of the mean-gradient energy common to all patches: {mine[0]['common_share']:.2f}")
    if len(names) > 1:
        for subset in ("cos_feat", "ccos_feat"):
            table = {n: {r["scroll_id"]: r[subset] for r in others if r["model"] == n} for n in names}
            ids = sorted(set.intersection(*(set(t) for t in table.values())))
            lines.append(f"\n== agreement between checkpoints ({subset} cosine over {len(ids)} patches) ==")
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    x, y = np.array([table[a][s] for s in ids]), np.array([table[b][s] for s in ids])
                    rho = np.corrcoef(_rank(x), _rank(y))[0, 1] if len(ids) > 2 else float("nan")
                    lines.append(f"{a} vs {b}: spearman {rho:+.2f}, same sign {np.mean(np.sign(x) == np.sign(y)):.0%}")
            same = [s for s in ids if len({np.sign(table[n][s]) for n in names}) == 1]
            lines.append(f"patches with the same sign in all {len(names)}: {len(same)}/{len(ids)}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target", default=str(base.HOLDOUT))
    parser.add_argument("--patches", default="", help="comma-separated subset (the target is always included)")
    parser.add_argument("--ckpt", action="append", help="name=path, repeatable (default: flip_2, asf_cue_labelsv2, asf_cue)")
    parser.add_argument("--out", default="output/grad_affinity")
    parser.add_argument("--no-save-grads", dest="save_grads", action="store_false")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
