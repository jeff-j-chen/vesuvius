"""campaign_runner_reg.py -- regularization sweep for v14c_mil_lcn.

context: lcn was the clear winner of the triple sweep, but massively overfits
(train PR-AUC ~0.95, valid PR-AUC <0.5 at 20 epochs). this campaign sweeps
regularization strength aggressively to close that gap.

CHANGES FROM LCN CAMPAIGN:
  - depth range EXPANDED: train 12->24 (was 8->16); windows 12->20 and 16->24
  - inference still runs on FULL 0->28 (d_start/d_end unchanged)
  - l1 and l2 swept over a wide range (need ~10-100x more than 7e-6)
  - data augmentation: two runs (default + stronger probs)
  - dropout: head_drop (before voxel head, like old FC dropout) + intermediate sweep
  - instancnorm: NOT included (risk of erasing weak signals; BatchNorm already present)

DROPOUT LOCATIONS (for reference):
  conv1_drop: Dropout3d between the two 3D conv blocks in depth-mix (channel-wise)
  conv2_drop: Dropout3d at end of depth-mix (channel-wise)
  head_drop:  Dropout3d before the 1x1x1 voxel head  [NEW -- most like FC-head dropout]
  per-slice stem: NO dropout (spatial zeroing hurts localization)

TESTS (15):
  L1 only:
    t01  L1=2e-5              (3x current)
    t02  L1=7e-5              (10x)
    t03  L1=2e-4              (30x)
    t04  L1=7e-4              (100x)
    t05  L1=2e-3              (300x)
  L2 only (via optimizer weight_decay):
    t06  L2=1e-3
    t07  L2=1e-2
    t08  L2=1e-1
  L1+L2 mixed:
    t09  L1=7e-5  L2=1e-3
    t10  L1=2e-4  L2=1e-2
  Augmentation (L1=7e-5 as base):
    t11  aug=True  default probs
    t12  aug=True  stronger probs (all probs +0.25, brightness/contrast +0.3)
  Dropout (L1=7e-5 as base, conv1/conv2 at default 0.05/0.075):
    t13  head_drop=0.3  (late-stage, closest to old FC dropout)
    t14  head_drop=0.5  (strong late-stage)
    t15  conv1_drop=0.2, conv2_drop=0.3  (stronger intermediate)

shared:
  - v14c_mil_lcn, tile=16, depth=8
  - 4 scrolls: w044+w059+w047+w056 (DEFAULT_SCROLLS)
  - train 12->24, inference 0->28
  - 20 epochs, eval at end, probes every 5 epochs
  - log_dir: ./runs_reg

run all:   python campaign_runner_reg.py
run from:  python campaign_runner_reg.py --from t06
dry-run:   python campaign_runner_reg.py --dry-run
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config

INTER_RUN_COOLDOWN_SECS = 120


def _base_config(exp_name: str) -> Config:
    """fresh config with reg-sweep defaults:
    lcn arch, tile=16, depth=8, train 12->24, inference 0->28, 4 scrolls."""
    c = Config()
    c.exp_name = exp_name
    c.model.arch       = "v14c_mil_lcn"
    c.model.conv1_drop = 0.05    # default intermediate dropout (keep stable across sweep)
    c.model.conv2_drop = 0.075
    c.model.head_drop  = 0.0     # overridden in dropout tests
    c.data.tile_size   = 16
    c.data.depth       = 8
    # EXPANDED training range: 12->24 covers windows 12-20 and 16-24
    c.data.train_d_start = 12
    c.data.train_d_end   = 24
    # full inference range (d_start/d_end) stays 0->28 so eval figures show everything
    c.data.d_start = 0
    c.data.d_end   = 28
    c.tra.n_epochs  = 20
    c.tra.eval_int  = 20
    c.tra.test_int  = 9999
    c.tra.probe_int = 5
    c.tra.log_dir   = "./runs_reg"
    c.tra.deterministic = True   # exact reproducibility for seeded runs
    c.tra.l1_lambda   = 7e-6    # overridden per test
    c.tra.weight_decay = 0.0    # overridden per test
    c.dl.batch_size  = 64
    c.dl.num_workers = 0
    c.dl.data_aug    = False    # overridden in aug tests
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [
    # (tid,    l1,    l2,   aug,  aug_strong, aug_weak, h_drop, c1_drop, c2_drop, tag)
    # --- L1 sweep (t01-t05, pure L1) ---
    ("t01", 2e-5,  0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=2e-5"),
    ("t02", 7e-5,  0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=7e-5"),
    ("t03", 2e-4,  0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=2e-4"),
    ("t04", 7e-4,  0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=7e-4"),
    ("t05", 2e-3,  0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=2e-3"),
    # --- L2 sweep (t06-t08) --- NOTE: L2 via AdamW weight_decay is ineffective on BN networks
    # (BatchNorm's scale invariance lets conv weights decay while gamma compensates)
    # confirmed empirically: 1e-3, 1e-2, 1e-1 all showed no effect
    ("t06", 0.0,   1e-3, False, False, False, 0.0,  0.05,  0.075, "L2=1e-3"),
    ("t07", 0.0,   1e-2, False, False, False, 0.0,  0.05,  0.075, "L2=1e-2"),
    ("t08", 0.0,   1e-1, False, False, False, 0.0,  0.05,  0.075, "L2=1e-1"),
    # --- L1=1e-4: critical midpoint between 7e-5 (underreg) and 2e-4 (overreg) ---
    ("t09b", 1e-4, 0.0,  False, False, False, 0.0,  0.05,  0.075, "L1=1e-4"),
    # --- t09/t10 (L1+L2 mixed) REMOVED: L2 inert on BN nets, so these ≈ t02 and t03 ---
    # --- augmentation (t11-t12, base L1=7e-5 as moderate regularizer) ---
    ("t11", 7e-5,  0.0,  True,  False, False, 0.0,  0.05,  0.075, "L1=7e-5_aug_default"),
    ("t12", 7e-5,  0.0,  True,  True,  False, 0.0,  0.05,  0.075, "L1=7e-5_aug_strong"),
    # t12b: default aug too strong for this model; try halved probs + softer noise/brightness
    ("t12b", 5e-5, 0.0,  True,  False, True,  0.0,  0.05,  0.075, "L1=5e-5_aug_weak"),
    # --- dropout (t13-t15, base L1=7e-5) ---
    ("t13", 7e-5,  0.0,  False, False, False, 0.3,  0.05,  0.075, "L1=7e-5_hdrop=0.3"),
    ("t14", 7e-5,  0.0,  False, False, False, 0.5,  0.05,  0.075, "L1=7e-5_hdrop=0.5"),
    ("t15", 7e-5,  0.0,  False, False, False, 0.0,  0.2,   0.3,   "L1=7e-5_convdrop++"),
]


def build_config(tid, l1, l2, aug, aug_strong, aug_weak, h_drop, c1_drop, c2_drop, tag) -> Config:
    campaign = "reg_2026_07_20"
    c = _base_config(f"cmp_{campaign}_{tid}_{tag}")
    c.tra.l1_lambda    = l1
    c.tra.weight_decay = l2
    c.model.conv1_drop = c1_drop
    c.model.conv2_drop = c2_drop
    c.model.head_drop  = h_drop
    c.dl.data_aug = aug or aug_strong or aug_weak
    if aug_strong:
        # all probabilities raised significantly over defaults
        c.dl.channel_mixing_prob = 0.0 # NEVER channel mix; depth is too important
        c.dl.rotation_prob       = 0.5
        c.dl.flip_prob           = 0.5
        c.dl.noise_prob          = 0.6
        c.dl.brightness_prob     = 0.8
        c.dl.contrast_prob       = 0.8
    elif aug_weak:
        # ~half the default probs; gentler than defaults which proved too disruptive
        c.dl.channel_mixing_prob = 0.0 # NEVER channel mix; depth is too important
        c.dl.rotation_prob       = 0.10
        c.dl.flip_prob           = 0.15
        c.dl.noise_prob          = 0.15
        c.dl.brightness_prob     = 0.25
        c.dl.contrast_prob       = 0.25
    c.save_final = f"models/reg/{tid}_{tag}_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[reg] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  tile={c.data.tile_size}  depth={c.data.depth}"
          f"  train={c.data.train_d_start}-{c.data.train_d_end}"
          f"  infer={c.data.d_start}-{c.data.d_end}")
    print(f"  L1={c.tra.l1_lambda:.1e}  L2={c.tra.weight_decay:.1e}"
          f"  aug={c.dl.data_aug}"
          f"  conv1_drop={c.model.conv1_drop}  conv2_drop={c.model.conv2_drop}"
          f"  head_drop={c.model.head_drop}")
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
    ap = argparse.ArgumentParser(description="regularization sweep for v14c_mil_lcn")
    ap.add_argument("--only", type=str, default=None,
                    help="run only this test id (e.g. t03)")
    ap.add_argument("--from", dest="from_id", type=str, default=None,
                    help="start from this test id, skipping earlier (e.g. t06)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print configs without training")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [t for t in TESTS if t[0].startswith(args.only)]
        if not selected:
            print(f"[ABORT] --only '{args.only}' matched nothing; valid: {[t[0] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t[0] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"\n[reg] {len(selected)} test(s) queued  (v14c_mil_lcn, tile=16, train 12->24, infer 0->28)")
    results: List[str] = []

    for i, row in enumerate(selected, 1):
        tid, l1, l2, aug, aug_s, aug_w, hd, c1, c2, tag = row
        c = build_config(tid, l1, l2, aug, aug_s, aug_w, hd, c1, c2, tag)
        ok = run_test(c, args.dry_run)
        status = "OK" if ok else "FAIL"
        results.append(f"  {tid} ({tag}): {status}")
        print(f"[reg] done {tid} -> {status}", flush=True)
        if i < len(selected) and not args.dry_run:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            cooldown(INTER_RUN_COOLDOWN_SECS, "inter-run")

    print(f"\n{'='*70}\n[reg] SUMMARY\n{'='*70}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
