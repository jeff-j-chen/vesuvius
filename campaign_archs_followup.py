"""campaign_archs_followup.py -- post-mean-teacher follow-up sweep for arch regularizers.

focuses on the next questions raised by the first arch sweep:
- does supcon t=0.07 prefer lambda 0.2 under matched linux defaults?
- does gentler dann (0.25/0.30/0.35) with a longer ramp avoid the epoch-5 cliff?
- does attention-mil help on its own, or only when paired with supcon?
- does the promising dann+supcon combo keep improving when run longer?
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign_archs import MAE_CKPT, _base_config, run_test


INTER_RUN_COOLDOWN_SECS = 60


def _mk(tid, tag, **overrides):
    d = {
        "tid": tid,
        "tag": tag,
        "init_weights": MAE_CKPT,
    }
    d.update(overrides)
    return d


TESTS = [
    # matched pure-supcon interpolation point between sc1 and sc2
    _mk("sc5", "ctx48_supcon_t007_lam02", supcon=True, supcon_temp=0.07, supcon_lambda=0.2),

    # rerun isolated attention to answer whether the old failure was just plumbing
    _mk("attn1r", "ctx48_attentionmil_rerun", attn_mil=True),

    # pair attention with the promising supcon setting to test whether attn adds value
    _mk("sc_attn2", "ctx48_sc_t007_lam02_attnmil",
        supcon=True, supcon_temp=0.07, supcon_lambda=0.2, attn_mil=True),

    # re-scan dann around the apparent sweet spot with slower ramp-up
    _mk("dann25", "ctx48_dann_lam025_r8", dann=True, dann_lambda=0.25, dann_ramp_epochs=8),
    _mk("dann30r8", "ctx48_dann_lam03_r8", dann=True, dann_lambda=0.30, dann_ramp_epochs=8),
    _mk("dann35", "ctx48_dann_lam035_r10", dann=True, dann_lambda=0.35, dann_ramp_epochs=10),

    # the best-looking composition so far gets the longer run
    _mk("dann_sc1l", "ctx48_dann03_r8_sc_t007_lam01_e20",
        dann=True, dann_lambda=0.30, dann_ramp_epochs=8,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.1,
        n_epochs=20),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "probe_int": ("tra", "probe_int"),
    "l1": ("tra", "l1_lambda"),
    "lr": ("tra", "lr"),
    "weight_decay": ("tra", "weight_decay"),
    "ranking_lambda": ("tra", "ranking_lambda"),
    "label_smooth": ("tra", "label_smooth"),
    "dann": ("tra", "dann"),
    "dann_lambda": ("tra", "dann_lambda"),
    "dann_ramp_epochs": ("tra", "dann_ramp_epochs"),
    "dann_n_domains": ("tra", "dann_n_domains"),
    "supcon": ("tra", "supcon"),
    "supcon_lambda": ("tra", "supcon_lambda"),
    "supcon_temp": ("tra", "supcon_temp"),
    "mean_teacher": ("tra", "mean_teacher"),
    "mean_teacher_alpha": ("tra", "mean_teacher_alpha"),
    "mean_teacher_lambda": ("tra", "mean_teacher_lambda"),
    "mean_teacher_ramp_epochs": ("tra", "mean_teacher_ramp_epochs"),
    "verified_neg_lambda": ("tra", "verified_neg_lambda"),
    "test_consistency": ("tra", "test_consistency"),
    "test_consistency_lambda": ("tra", "test_consistency_lambda"),
    "context_size": ("data", "context_size"),
    "context_downsample": ("data", "context_downsample"),
    "ring_label_source": ("data", "ring_label_source"),
    "eval_infer_bs": ("data", "eval_infer_bs"),
    "batch_size": ("dl", "batch_size"),
    "num_workers": ("dl", "num_workers"),
}

def _apply_linux_optimizations(c):
    """apply linux-specific optimizations: no cooldowns, higher infer batch"""
    import os
    if os.name == "posix":  # linux
        c.tra.epoch_cooldown_secs = 0
        c.tra.val_cooldown_secs = 0
        c.tra.eval_cooldown_secs = 0
        c.tra.fig_chunk_cooldown_ms = 0
        if c.data.eval_infer_bs <= 32:
            c.data.eval_infer_bs = 256

def build_config(t: dict):
    c = _base_config(f"cmp_archs_{t['tid']}_{t['tag']}")
    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            setattr(getattr(c, sec), attr, t[k])

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw

    c.dl.data_aug = any([
        c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
        c.dl.brightness_prob, c.dl.contrast_prob,
        c.dl.cutout_prob, c.dl.depth_mask_prob,
    ])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs_followup", exist_ok=True)
    c.save_final = f"models/archs_followup/{t['tid']}_{t['tag']}_final.pth"
    _apply_linux_optimizations(c)
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def main():
    ap = argparse.ArgumentParser(description="campaign_archs_followup: post-MT arch follow-up sweep")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [t for t in TESTS if t["tid"] in want]
        missing = want - {t["tid"] for t in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: {[t['tid'] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t["tid"] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs-followup] {len(selected)} test(s) queued")

    results = {}
    for i, t in enumerate(selected):
        c = build_config(t)
        ok = run_test(c, args.dry_run)
        results[t["tid"]] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del c
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {t['tid']}")

    print(f"\n{'='*70}\n[archs-followup] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()