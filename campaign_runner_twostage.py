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
    c.tra.n_epochs     = 15
    c.tra.eval_int     = 15
    c.tra.test_int     = 15
    c.tra.probe_int    = 5
    c.tra.save_int     = 2       # save every 2 epochs so a crash doesn't wipe the run (BSODs ongoing)
    c.tra.log_dir      = "./runs_reg"
    c.tra.deterministic = False   # exact reproducibility for seeded runs
    c.tra.l1_lambda    = 0.0
    c.tra.weight_decay = 0.0
    c.dl.batch_size    = 128
    c.dl.num_workers   = 4
    c.dl.data_aug      = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    # load the 4 default test scrolls; test figures fire once at epoch 30 (test_int).
    # only the primary scroll-vis loads them (see Trainer), so RAM stays bounded.
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [
    # ARCHITECTURE / LOSS VARIANTS (2026-07-21): run these FIRST. all with NO augmentation
    # and NO regularization (drops/L1/cutout off), 20 epochs, so any change is attributable
    # purely to the arch/loss. peak TRAIN PR-AUC had plateaued ~0.66 => underfitting, so
    # these target FIT (capacity, supervision density, matched objective, physical feature).
    # (A) dense per-pixel supervision -- ~64x more gradient signal per tile than a tile scalar
    # dict(tid="tsA", arch="v15_twostage_dense", dense=True,
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="dense"),

    # (C) wider stage-2 fusion CNN (3->32->32->16->1) -- fixes the ~4.8k-param fusion bottleneck
    # dict(tid="tsC", arch="v15_twostage_wide",
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="wide_fusion"),

    # (D) BCE + pairwise ranking (AUC surrogate) -- objective matched to PR-AUC on balanced
    #     ring data (not focal, which targets imbalance we don't have)
    # dict(tid="tsD", arch="v15_twostage_lcn", ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="ranking_auc"),

    # (E) shared backbone also ingests dI/dz ([raw, lcn, dz]) -- explicit ink-interface feature
    # dict(tid="tsE", arch="v15_twostage_zgrad",
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="zgrad"),

    # (F) COMBO C+D: wide stage-2 fusion + pairwise-ranking loss (both proven effective).
    # #     no aug/reg, 20 epochs -- isolates the combined arch+loss effect.
    # dict(tid="tsF", arch="v15_twostage_wide", ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="wide_ranking"),

    # (G) COMBO C+D+E: wide fusion + zgrad backbone + ranking loss (E not yet proven).
    # dict(tid="tsG", arch="v15_twostage_wide_zgrad", ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=20, tag="wide_zgrad_ranking"),

        
    dict(tid="tsJ", arch="v15_twostage_wide_zgrad", ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0, rotation=0, noise=0, brightness=0, contrast=0,
         h_drop=0, c1_drop=0, c2_drop=0,
         cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
         l1=0.0, n_epochs=30, tag="wide_zgrad_ranking_aug_noreg_alldata"),

    dict(tid="tsH", arch="v15_twostage_wide_zgrad", ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0.3, rotation=0.1, noise=0.01, brightness=0.05, contrast=0.05,
         h_drop=0.2, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.1, cutout_max_frac=0.15, cutout_n_patches=1, depth_mask_prob=0.0,
         l1=7e-6, n_epochs=30, tag="wide_zgrad_ranking_aug_weakreg_alldata"),
    
    
    dict(tid="tsI", arch="v15_twostage_wide_zgrad", ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0.4, rotation=0.4, noise=0.05, brightness=0.1, contrast=0.1,
         h_drop=0.3, c1_drop=0.1, c2_drop=0.1,
         cutout_prob=0.2, cutout_max_frac=0.15, cutout_n_patches=2, depth_mask_prob=0.0,
         l1=7e-5, n_epochs=30, tag="wide_zgrad_ranking_aug_strongreg_alldata"),

            # dict(tid="ts02",
             #      flip=0.5, rotation=0.05, noise=0.0, brightness=0.05, contrast=0.05,
             #      h_drop=0.2, c1_drop=0.05, c2_drop=0.075,
             #      cutout_prob=0.15, cutout_max_frac=0.15, cutout_n_patches=1, depth_mask_prob=0.0,
             #      l1=7e-6, n_epochs=20,
             #      tag="veryweak_fullcombo"),

    # ts00: PURE BASELINE -- no augmentation, no regularization (dropout/L1/cutout all off).
    # 30 epochs to see how far the raw two-stage arch fits before overfitting. reference
    # for how much the aug/reg in ts01/ts02 actually helps generalization.
    # dict(tid="ts00",
    #      flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
    #      h_drop=0.0, c1_drop=0.0, c2_drop=0.0,
    #      cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0, depth_mask_prob=0.0,
    #      l1=0.0, n_epochs=30,
    #      tag="baseline_noreg"),

    # FULL WEAKENED COMBO (2026-07-21): apply every regularizer at weak strength on the
    # two-stage arch -- data aug + head/conv dropout + cutout patches + L1. mirrors the
    # winning combo-campaign weak settings. ts_weak (aug-only) was the best performer and
    # first to show validation loss dropping; these add the rest of the combo on top.
    # ts01: weak full combo
    # dict(tid="ts01",
    #      flip=0.5, rotation=0.10, noise=0.05, brightness=0.10, contrast=0.10,
    #      h_drop=0.3, c1_drop=0.05, c2_drop=0.075,
    #      cutout_prob=0.25, cutout_max_frac=0.20, cutout_n_patches=1, depth_mask_prob=0.0,
    #      l1=3e-5, n_epochs=20,
    #      tag="weak_fullcombo"),

    # # ts02: very weak full combo -- same knobs as ts01, lower values across the board
    # dict(tid="ts02",
    #      flip=0.5, rotation=0.05, noise=0.0, brightness=0.05, contrast=0.05,
    #      h_drop=0.2, c1_drop=0.05, c2_drop=0.075,
    #      cutout_prob=0.15, cutout_max_frac=0.15, cutout_n_patches=1, depth_mask_prob=0.0,
    #      l1=7e-6, n_epochs=20,
    #      tag="veryweak_fullcombo"),
]


def build_config(t: dict) -> Config:
    tid = t["tid"]
    tag = t["tag"]
    c = _base_config(f"cmp_twostage_2026_07_21_{tid}_{tag}")

    # optional arch / supervision / loss overrides (arch-variant tests)
    if t.get("arch"):
        c.model.arch = t["arch"]
    if t.get("dense", False):
        c.data.dense_labels = True
    c.tra.ranking_lambda   = float(t.get("ranking_lambda", 0.0))
    c.tra.ranking_neg_frac = float(t.get("ranking_neg_frac", 1.0))

    c.tra.l1_lambda    = t["l1"]
    c.tra.n_epochs     = int(t.get("n_epochs", 20))
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

    os.makedirs("models/twostage", exist_ok=True)
    c.save_final = f"models/twostage/{tid}_{tag}_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[twostage] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  depth=24 (3x8)  train=4-28  n_epochs={c.tra.n_epochs}")
    print(f"  L1={c.tra.l1_lambda:.1e}  rank_lambda={getattr(c.tra,'ranking_lambda',0.0)}  "
          f"dense={getattr(c.data,'dense_labels',False)}")
    print(f"  flip={c.dl.flip_prob} rot={c.dl.rotation_prob} "
          f"noise={c.dl.noise_prob} bright={c.dl.brightness_prob} contrast={c.dl.contrast_prob}  "
          f"h_drop={c.model.head_drop}  cutout={c.dl.cutout_prob}/{c.dl.cutout_n_patches}patch")
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

    print(f"[twostage] {len(selected)} test(s) queued  (v15_twostage_lcn, depth=24, train 4-28)")

    results = {}
    for i, t in enumerate(selected):
        tid = t["tid"]
        c = build_config(t)
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"

        if not args.dry_run:
            del c
            gc.collect()

        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {tid}")

    print(f"\n{'='*70}\n[twostage] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
