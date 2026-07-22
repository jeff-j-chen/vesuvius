"""campaign_runner_iso.py -- isolation sweep for v14c_mil_lcn.

context: campaign_runner_reg.py showed:
  - L1=7e-5: weak effect; L1=2e-4: kills learning; sweet spot near 1e-4
  - L2 (AdamW weight_decay): completely inert on BN networks (scale invariance)
  - weak aug (t12b, L1=5e-5): BEST result -- lifted val the most
  - head_drop=0.5 (t14): showed promise
  both t12b and t14 were confounded with non-zero L1 lambda.

purpose: re-run the promising methods with L1=0 throughout, so that each
regularizer's contribution is cleanly isolated. structured as two sweeps:

  AUG SWEEP: isolate which augmentation types and intensities help
  DROPOUT SWEEP: isolate which dropout configurations help
  FOCAL LOSS: proposed additional technique -- down-weights easy negatives
    via (1-p_t)^gamma modulation; concentrates gradient on hard ink-boundary
    cases where the model is less certain.

channel_mixing_prob is fixed to 0.0 throughout. permuting depth slices
disrupts the depth-ordering signal that the LCN stem is designed to exploit.

TESTS (14):
  baseline:
    b00  no reg, no aug, no dropout  (pure overfitting reference)
  augmentation (channel-mix=0 throughout):
    a01  flip=0.5  (safest: pure spatial symmetry)
    a02  flip=0.5  rotation=0.25  (geometric only)
    a03  flip=0.5  brightness=0.35  contrast=0.35  (photometric only, no noise)
    a04  flip=0.5  noise=0.20  (additive noise only)
    a05  flip=0.5  rotation=0.15  noise=0.10  brightness=0.15  contrast=0.15
         (all-very-weak; lighter than t12b which still had L1=5e-5)
    a06  flip=0.5  rotation=0.25  noise=0.25  brightness=0.35  contrast=0.35
         (moderate all-in; isolates L1 confound from t12b)
  dropout:
    d01  head_drop=0.3  (re-isolate from L1)
    d02  head_drop=0.5  (re-isolate from L1)
    d03  head_drop=0.7  (push further)
    d04  conv1_drop=0.20  conv2_drop=0.30  (convdrop++ -- first clean test)
    d05  head_drop=0.5   conv1_drop=0.20  conv2_drop=0.30  (best estimated combo)
  focal loss:
    f01  focal_gamma=1.0  (mild: partial down-weight of easy cases)
    f02  focal_gamma=2.0  (standard: strong down-weight of easy cases)

shared:
  - v14c_mil_lcn, tile=16, depth=8, train 12->24, infer 0->28
  - L1=0.0 throughout  (explicit isolation)
  - 4 scrolls: w044+w059+w047+w056 (DEFAULT_SCROLLS)
  - 20 epochs, eval at end, probes every 5 epochs
  - log_dir: ./runs_reg  (same as reg campaign -- compare in same tensorboard)
  - models: models/iso/

run all:   python campaign_runner_iso.py
run from:  python campaign_runner_iso.py --from a03
dry-run:   python campaign_runner_iso.py --dry-run
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
    """fresh config: same base as reg campaign (lcn, tile=16, depth=8, 12->24)."""
    c = Config()
    c.exp_name = exp_name
    c.model.arch       = "v14c_mil_lcn"
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    c.model.head_drop  = 0.0
    c.data.tile_size   = 16
    c.data.depth       = 8
    c.data.train_d_start = 12
    c.data.train_d_end   = 24
    c.data.d_start = 0
    c.data.d_end   = 28
    c.tra.n_epochs  = 20
    c.tra.eval_int  = 20
    c.tra.test_int  = 9999
    c.tra.probe_int = 5
    c.tra.log_dir   = "./runs_reg"   # same dir as reg campaign
    c.tra.deterministic = True   # exact reproducibility for seeded runs
    c.tra.l1_lambda    = 0.0   # explicitly OFF for all iso tests
    c.tra.weight_decay = 0.0
    c.dl.batch_size  = 64
    c.dl.num_workers = 0
    c.dl.data_aug    = False
    # channel mixing always off -- depth order is meaningful for LCN stem
    c.dl.channel_mixing_prob = 0.0
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


# each test is a dict with all varied parameters explicit
TESTS = [
    # ------------------------------------------------------------------ BASELINE
    # no regularization of any kind -- pure overfitting reference
    dict(tid="b00", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="baseline"),

    # ------------------------------------------------------------------ AUG SWEEP
    # flip only -- spatial symmetry, zero information loss, true lower bound of aug
    dict(tid="a01", flip=0.5, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_flip_only"),

    # geometric only -- rotation adds 90/180/270 permutations on top of flip
    dict(tid="a02", flip=0.5, rotation=0.25, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_geom_only"),

    # photometric only -- brightness and contrast, no noise, no spatial warp
    # tests whether mild intensity jitter helps without geometric invariance
    dict(tid="a03", flip=0.5, rotation=0.0, noise=0.0, brightness=0.35, contrast=0.35,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_photometric"),

    # noise only -- gaussian additive noise (std 0.005-0.015); tests scanner-noise robustness
    dict(tid="a04", flip=0.5, rotation=0.0, noise=0.20, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_noise_only"),

    # all-very-weak combined -- lighter than t12b (which had L1=5e-5 confound)
    # lowest combined aug load that still exercises all types
    dict(tid="a05", flip=0.5, rotation=0.15, noise=0.10, brightness=0.15, contrast=0.15,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_all_very_weak"),

    # moderate combined -- isolates the L1 confound in t12b
    # (t12b used flip=0.15, rot=0.10, noise=0.15, bright=0.25, contrast=0.25 with L1=5e-5)
    dict(tid="a06", flip=0.5, rotation=0.25, noise=0.25, brightness=0.35, contrast=0.35,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="aug_all_moderate"),

    # ------------------------------------------------------------------ DROPOUT SWEEP
    # head_drop (before voxel head) -- closest analog to FC dropout
    dict(tid="d01", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.3, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="hdrop=0.3"),

    dict(tid="d02", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.5, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="hdrop=0.5"),

    dict(tid="d03", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.7, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="hdrop=0.7"),

    # convdrop++ -- first clean test without L1 confound (t15 had L1=7e-5)
    dict(tid="d04", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.20, c2_drop=0.30,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="convdrop++"),

    # combined: best head_drop + convdrop++ together
    dict(tid="d05", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.5, c1_drop=0.20, c2_drop=0.30,
         cutout_prob=0.0, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="hdrop=0.5_convdrop++"),

    # ------------------------------------------------------------------ SPECAUGMENT MASKING
    # XY cutout: zero a random rectangle across all depth slices simultaneously.
    # forces the model to use distributed spatial evidence rather than
    # memorizing specific tile locations. applied 50% of the time.
    # s01: 1 patch, up to 35% of each spatial dim (up to ~5x5 on 16x16 tile)
    dict(tid="s01", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.5, cutout_max_frac=0.35, cutout_n_patches=1, depth_mask_prob=0.0,
         tag="cutout_1patch"),

    # s02: 2 patches (up to 40%) + per-slice depth masking (20% per slice).
    # combination maximally prevents spatial and depth memorization.
    dict(tid="s02", flip=0.0, rotation=0.0, noise=0.0, brightness=0.0, contrast=0.0,
         h_drop=0.0, c1_drop=0.05, c2_drop=0.075,
         cutout_prob=0.5, cutout_max_frac=0.40, cutout_n_patches=2, depth_mask_prob=0.20,
         tag="cutout_2patch_depthmask"),
]


def build_config(t: dict) -> Config:
    tid = t["tid"]
    tag = t["tag"]
    c = _base_config(f"cmp_iso_2026_07_20_{tid}_{tag}")

    # dropout
    c.model.head_drop  = t["h_drop"]
    c.model.conv1_drop = t["c1_drop"]
    c.model.conv2_drop = t["c2_drop"]

    # augmentation: enable data_aug if any aug prob is non-zero
    flip = t["flip"]; rotation = t["rotation"]
    noise = t["noise"]; brightness = t["brightness"]; contrast = t["contrast"]
    cutout_prob     = t.get("cutout_prob", 0.0)
    cutout_max_frac = t.get("cutout_max_frac", 0.35)
    cutout_n_patches = t.get("cutout_n_patches", 1)
    depth_mask_prob = t.get("depth_mask_prob", 0.0)
    c.dl.data_aug = any([flip, rotation, noise, brightness, contrast,
                         cutout_prob, depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0   # always off -- depth ordering matters
    c.dl.flip_prob        = flip
    c.dl.rotation_prob    = rotation
    c.dl.noise_prob       = noise
    c.dl.brightness_prob  = brightness
    c.dl.contrast_prob    = contrast
    c.dl.cutout_prob      = cutout_prob
    c.dl.cutout_max_frac  = cutout_max_frac
    c.dl.cutout_n_patches = cutout_n_patches
    c.dl.depth_mask_prob  = depth_mask_prob

    os.makedirs("models/iso", exist_ok=True)
    c.save_final = f"models/iso/{tid}_{tag}_final.pth"
    return c


def _fmt(t: dict) -> str:
    aug = (f"flip={t['flip']} rot={t['rotation']} "
           f"noise={t['noise']} bright={t['brightness']} contrast={t['contrast']}")
    drop = f"h_drop={t['h_drop']} c1={t['c1_drop']} c2={t['c2_drop']}"
    mask = (f"cutout_prob={t.get('cutout_prob',0.0)} "
            f"max_frac={t.get('cutout_max_frac',0.35)} "
            f"n_patches={t.get('cutout_n_patches',1)} "
            f"depth_mask_prob={t.get('depth_mask_prob',0.0)}")
    return f"L1=0  {aug}  {drop}  {mask}"


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[iso] {c.exp_name}\n{'='*70}", flush=True)
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
    ap = argparse.ArgumentParser(description="isolation sweep for v14c_mil_lcn")
    ap.add_argument("--only", type=str, default=None,
                    help="run only this test id (e.g. a03)")
    ap.add_argument("--from", dest="from_id", type=str, default=None,
                    help="start from this test id, skipping earlier (e.g. d01)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print configs without training")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [t for t in TESTS if t["tid"] == args.only]
        if not selected:
            valid = [t["tid"] for t in TESTS]
            print(f"[ABORT] --only '{args.only}' not found; valid: {valid}")
            return
    elif args.from_id:
        ids = [t["tid"] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[iso] {len(selected)} test(s) queued  "
          f"(v14c_mil_lcn, tile=16, train 12->24, infer 0->28, L1=0 throughout)")

    results = {}
    for i, t in enumerate(selected):
        tid = t["tid"]
        c = build_config(t)
        print(f"\n{'='*70}\n[iso] {c.exp_name}\n{'='*70}", flush=True)
        print(f"  {_fmt(t)}")

        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"

        if not args.dry_run:
            del c
            gc.collect()

        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {tid}")

    print(f"\n{'='*70}\n[iso] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
