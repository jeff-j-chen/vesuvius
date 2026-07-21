"""campaign_runner_combo.py -- combination sweep for v14c_mil_lcn.

builds on iso campaign findings (2026-07-20):
  - a06 (flip+rot+noise+bright+contrast moderate) had best val_loss (0.7835)
  - d02 (head_drop=0.5) had best dropout val_F1 (0.6021)
  - s01 (cutout 1-patch, no depth mask) regularizes without destroying signal
  these three were tested independently; now we combine them with L1 to see
  if they cooperate or interfere.

TESTS:
  t01 -- strong combo: a06 aug + head_drop=0.5 + cutout-1patch + L1=7e-5
  t02 -- weakened combo: a05 aug + head_drop=0.4 + cutout-1patch + L1=3e-5
         (hedges against the combined regularization being too aggressive)

shared:
  - v14c_mil_lcn, tile=16, depth=8, train 12->24, infer 0->28
  - 4 scrolls, 20 epochs, eval_int=20, probe_int=5
  - log_dir: ./runs_reg (same tensorboard as iso/reg campaigns)
  - no depth masking (iso showed it adds regularization without val gain)

run all:   python campaign_runner_combo.py
dry-run:   python campaign_runner_combo.py --dry-run
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config

INTER_RUN_COOLDOWN_SECS = 120


def _base_config(exp_name: str) -> Config:
    """same base as iso/reg campaigns."""
    c = Config()
    c.exp_name = exp_name
    c.model.arch       = "v14c_mil_lcn"
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    c.model.head_drop  = 0.0
    c.data.tile_size     = 16
    c.data.depth         = 8
    c.data.train_d_start = 12
    c.data.train_d_end   = 24
    c.data.d_start = 0
    c.data.d_end   = 28
    c.tra.n_epochs  = 20
    c.tra.eval_int  = 20
    c.tra.test_int  = 9999
    c.tra.probe_int = 5
    c.tra.log_dir   = "./runs_reg"
    c.tra.deterministic = True   # exact reproducibility for seeded runs
    c.tra.l1_lambda    = 0.0
    c.tra.weight_decay = 0.0
    c.dl.batch_size  = 64
    c.dl.num_workers = 0
    c.dl.data_aug    = False
    c.dl.channel_mixing_prob = 0.0
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [
    # t01: strong combination
    #   a06 aug (all-moderate): flip=0.5, rot=0.25, noise=0.25, bright=0.35, contrast=0.35
    #   d02 head_drop:  0.5
    #   cutout: 1 patch, 35% spatial, 50% prob, no depth masking
    #   L1: 7e-5
    dict(tid="t01",
         flip=0.5, rotation=0.25, noise=0.25, brightness=0.35, contrast=0.35,
         h_drop=0.5, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.5, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         l1=7e-5,
         tag="a06_hdrop05_cutout1_L1=7e-5"),

    # t02: weakened combination
    #   a05 aug (all-very-weak): flip=0.5, rot=0.15, noise=0.10, bright=0.15, contrast=0.15
    #   head_drop: 0.4 (between iso d01=0.3 and d02=0.5)
    #   cutout: 1 patch, 35% spatial, 50% prob, no depth masking
    #   L1: 3e-5
    dict(tid="t02",
         flip=0.5, rotation=0.15, noise=0.10, brightness=0.15, contrast=0.15,
         h_drop=0.4, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.5, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         l1=3e-5,
         tag="a05_hdrop04_cutout1_L1=3e-5"),
]


def build_config(t: dict) -> Config:
    tid = t["tid"]
    tag = t["tag"]
    c = _base_config(f"cmp_combo_2026_07_21_{tid}_{tag}")

    c.tra.l1_lambda    = t["l1"]
    c.model.head_drop  = t["h_drop"]
    c.model.conv1_drop = t["c1_drop"]
    c.model.conv2_drop = t["c2_drop"]

    flip       = t["flip"]
    rotation   = t["rotation"]
    noise      = t["noise"]
    brightness = t["brightness"]
    contrast   = t["contrast"]
    cutout_prob      = t.get("cutout_prob", 0.0)
    cutout_max_frac  = t.get("cutout_max_frac", 0.35)
    cutout_n_patches = t.get("cutout_n_patches", 1)
    depth_mask_prob  = t.get("depth_mask_prob", 0.0)

    c.dl.data_aug = any([flip, rotation, noise, brightness, contrast,
                         cutout_prob, depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    c.dl.flip_prob        = flip
    c.dl.rotation_prob    = rotation
    c.dl.noise_prob       = noise
    c.dl.brightness_prob  = brightness
    c.dl.contrast_prob    = contrast
    c.dl.cutout_prob      = cutout_prob
    c.dl.cutout_max_frac  = cutout_max_frac
    c.dl.cutout_n_patches = cutout_n_patches
    c.dl.depth_mask_prob  = depth_mask_prob

    os.makedirs("models/combo", exist_ok=True)
    c.save_final = f"models/combo/{tid}_{tag}_final.pth"
    return c


def _fmt(t: dict) -> str:
    return (f"L1={t['l1']:.1e}  flip={t['flip']} rot={t['rotation']} "
            f"noise={t['noise']} bright={t['brightness']} contrast={t['contrast']}  "
            f"h_drop={t['h_drop']}  cutout={t['cutout_prob']}/{t['cutout_n_patches']}patch")


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[combo] {c.exp_name}\n{'='*70}", flush=True)
    if dry_run:
        print("  [DRY RUN] skipping")
        return True
    from train import Trainer
    try:
        trainer = Trainer(c)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description="combination regularization sweep")
    ap.add_argument("--only",  type=str, default=None)
    ap.add_argument("--from",  dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [t for t in TESTS if t["tid"] == args.only]
        if not selected:
            print(f"[ABORT] --only '{args.only}' not found; valid: {[t['tid'] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t["tid"] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[combo] {len(selected)} test(s) queued  (v14c_mil_lcn, tile=16, train 12->24)")

    results = {}
    for i, t in enumerate(selected):
        tid = t["tid"]
        c = build_config(t)
        print(f"\n{'='*70}\n[combo] {c.exp_name}\n{'='*70}", flush=True)
        print(f"  {_fmt(t)}")

        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"

        if not args.dry_run:
            del c
            gc.collect()

        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {tid}")

    print(f"\n{'='*70}\n[combo] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
