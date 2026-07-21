"""campaign_runner_twostage.py -- two-stage MIL architecture baseline sweep.

architecture: v15_twostage_lcn
  stage 1: v14c_mil_lcn backbone (tied weights) applied to each of 3
           non-overlapping 8-slice depth windows (abs depth 4-12, 12-20, 20-28)
           with correct absolute depth PE per window.
  stage 2: small 3D CNN fuses the 3 per-voxel logit maps -> final tile logit
           via MIL-LSE. learns cross-window depth consistency patterns.

how this differs from the old dense_unet (commit de9e902):
  dense_unet used HARD depth-max -> 2D U-Net decoder + per-pixel dense labels.
  v15 uses SOFT per-window MIL-LSE -> learned 3D fusion -> tile-label MIL.
  these are fundamentally different: dense_unet required pixel labels and worked
  on 2.4um data; v15 stays in the tile-label MIL framing of the current system.

how this differs from single-window v14c_mil_lcn:
  - sees all 24 depth slices per tile simultaneously (not a random 8-slice window)
  - stage 2 learns cross-window consistency (e.g. window 2 lights up but not 1+3)
  - depth_pe is applied with correct absolute offsets (4, 12, 20) so the backbone
    genuinely distinguishes depth bands (currently v14c always uses PE positions 0-7)

timing note: run AFTER campaign_runner_iso.py finishes. that campaign identifies
the best regularization strategy; apply winning config here as a follow-up.
for now we run two baseline tests to establish the architecture's baseline behavior:
  ts01: no regularization (pure overfitting reference)
  ts02: L1=7e-5 (modest regularizer that showed some effect in reg campaign)

shared config:
  - arch: v15_twostage_lcn
  - tile_size=16, depth=24 (3 windows x 8 slices)
  - train_d_start=4, train_d_end=28, d_start=4, d_end=28
    (fixed single 24-slice block per tile; no random depth sampling since
    only one window fits in [4,28] with depth=24)
  - 4 training scrolls (DEFAULT_SCROLLS)
  - 20 epochs, eval_int=20, probe_int=5
  - log_dir: ./runs_reg (same tensorboard as other campaigns)

run all:   python campaign_runner_twostage.py
dry-run:   python campaign_runner_twostage.py --dry-run
run from:  python campaign_runner_twostage.py --from ts02
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
    """fresh config for the two-stage architecture sweep."""
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "v15_twostage_lcn"
    # depth=24 covers all 3 windows (4-12, 12-20, 20-28)
    c.data.tile_size     = 16
    c.data.depth         = 24
    c.data.train_d_start = 4
    c.data.train_d_end   = 28
    # fixed 24-slice inference block (4->28)
    c.data.d_start = 4
    c.data.d_end   = 28
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    c.model.head_drop  = 0.0
    c.tra.n_epochs     = 20
    c.tra.eval_int     = 20
    c.tra.test_int     = 9999
    c.tra.probe_int    = 5
    c.tra.log_dir      = "./runs_reg"
    c.tra.deterministic = True   # exact reproducibility for seeded runs
    c.tra.l1_lambda    = 0.0
    c.tra.weight_decay = 0.0
    c.dl.batch_size    = 64
    c.dl.num_workers   = 0
    c.dl.data_aug      = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [
    # (tid, l1, tag)
    # baseline: no regularization -- pure overfitting reference for the new architecture
    ("ts01", 0.0,  "baseline"),
    # L1=7e-5: the moderate L1 that showed some effect in the reg campaign
    ("ts02", 7e-5, "L1=7e-5"),
]


def build_config(tid: str, l1: float, tag: str) -> Config:
    c = _base_config(f"cmp_twostage_2026_07_20_{tid}_{tag}")
    c.tra.l1_lambda = l1
    os.makedirs("models/twostage", exist_ok=True)
    c.save_final = f"models/twostage/{tid}_{tag}_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[twostage] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch=v15_twostage_lcn  depth=24 (3x8)  train=4-28  L1={c.tra.l1_lambda:.1e}")
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
    ap = argparse.ArgumentParser(description="two-stage MIL architecture sweep")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [t for t in TESTS if t[0] == args.only]
        if not selected:
            print(f"[ABORT] --only '{args.only}' not found; valid: {[t[0] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t[0] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[twostage] {len(selected)} test(s) queued  (v15_twostage_lcn, depth=24, train 4-28)")

    results = {}
    for i, (tid, l1, tag) in enumerate(selected):
        c = build_config(tid, l1, tag)
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"

        if not args.dry_run:
            del c
            gc.collect()

        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {tid}")

    print(f"\n{'='*70}\n[twostage] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t[2] for t in TESTS if t[0] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
