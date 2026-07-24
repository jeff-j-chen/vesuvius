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

# MAE checkpoint every run warm-starts from (produced by run_mae_then_twostage.py -> mae_pretrain_twostage.py).
# stage1.* transfers into the two-stage backbone; stage2 stays fresh. only applied if the file exists.
MAE_CKPT = "models/mae_twostage.pth"


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
    c.tra.n_epochs     = 10
    c.tra.eval_int     = 10
    c.tra.test_int     = 999
    c.tra.probe_int    = 5
    c.tra.save_int     = 2       # save every 2 epochs so a crash doesn't wipe the run (BSODs ongoing)
    c.tra.log_dir      = "./runs_ts_mae"
    
    c.tra.deterministic = False   # exact reproducibility for seeded runs
    c.tra.l1_lambda    = 0.0
    c.tra.weight_decay = 0.0
    c.dl.batch_size    = 96
    c.dl.num_workers   = 4
    c.dl.data_aug      = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "closed"   # closed ring off (hand-cleaned) eroded map
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    # load the 4 default test scrolls; test figures fire once at epoch 30 (test_int).
    # only the primary scroll-vis loads them (see Trainer), so RAM stays bounded.
    c.tra.epoch_cooldown_secs   = 9
    c.tra.val_cooldown_secs     = 12
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [

    # dict(tid="tsJ", arch="v15_twostage_wide_zgrad", context_size=0, init_weights=MAE_CKPT,
    #      ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.55, rotation=0.55, noise=0.2, brightness=0.5, contrast=0.5,
    #      h_drop=0.4, c1_drop=0.15, c2_drop=0.15,
    #      cutout_prob=0.4, cutout_max_frac=0.15, cutout_n_patches=2, depth_mask_prob=0.0,
    #      l1=7e-5, tag="ts_gce_strongreg_mae"),

    # 48x48 context-window duplicates (competition-legal limit): larger input crop centered on
    # each tile, model center-pools MIL over the tile region so ring labels are unchanged.

    # dict(tid="tsJc", arch="v15_twostage_wide_zgrad_ctx", context_size=48, init_weights=MAE_CKPT,
    #      batch_size=24,   # 48px crops are ~9x the activations of a 16px tile -> shrink batch to avoid OOM
    #      ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.55, rotation=0.55, noise=0.2, brightness=0.5, contrast=0.5,
    #      h_drop=0.4, c1_drop=0.15, c2_drop=0.15,
    #      cutout_prob=0.4, cutout_max_frac=0.15, cutout_n_patches=2, depth_mask_prob=0.0,
    #      l1=7e-5, tag="ts_gce_strongreg_ctx48_mae"),


    # tsJd baseline (closed ring, ctx32) -- already trained; kept here as the reference config
    # dict(tid="tsJd", arch="v15_twostage_wide_zgrad_ctx", context_size=32, init_weights=MAE_CKPT,
    #      batch_size=32,
    #      ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
    #      h_drop=0.5, c1_drop=0.15, c2_drop=0.155,
    #      cutout_prob=0.5, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
    #      l1=1e-4, tag="ts_gce_vstrongreg_ctx32d_mae"),

    # NEXT CAMPAIGN: two single-variable changes off the tsJd config (everything else identical).

    # (1) tsJd (ctx32) but ERODED ring labels instead of closed. eroded positives trace the
    #     letter tightly; this tests the eroded-vs-closed tradeoff UNDER the large receptive
    #     field -- an untested combo (the old eroded/closed comparison was scroll1-only, easy
    #     letters + concrete labels; we are now on hard letters + uncertain labels).
    dict(tid="tsJe", arch="v15_twostage_wide_zgrad_ctx", context_size=32, init_weights=MAE_CKPT,
         batch_size=32, ring_label_source="eroded",
         ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
         h_drop=0.5, c1_drop=0.15, c2_drop=0.155,
         cutout_prob=0.5, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
         l1=1e-4, tag="ts_gce_vstrongreg_ctx32_eroded_mae"),

    # (2) tsJd (closed ring) but a COARSER context at the SAME 32px extent: avg-pool the input
    #     2x (context_downsample=2) so the model keeps the full context window but at half
    #     resolution -> ~1/4 the activations (near-plain compute), smaller overfit surface, and
    #     it should stop the big-fragment inference OOM. tradeoff: the center tile is coarsened too.
    dict(tid="tsJf", arch="v15_twostage_wide_zgrad_ctx", context_size=32, context_downsample=2,
         init_weights=MAE_CKPT, batch_size=32,
         ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
         h_drop=0.5, c1_drop=0.15, c2_drop=0.155,
         cutout_prob=0.5, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
         l1=1e-4, tag="ts_gce_vstrongreg_ctx32ds2_closed_mae"),

    # (3) tsJd (closed ring) but FOVEATED context: full-res central tile + coarse full-extent
    #     surround, fused before MIL. keeps the middle at full 10um resolution (where the letter
    #     detail lives; prior models resolved ink at 1-2um) while still giving the convs the wider
    #     context. ~2x plain-tile compute (two tile passes) vs ~4x for full-res ctx32.
    dict(tid="tsJg", arch="v15_twostage_wide_zgrad_fovea", context_size=32,
         init_weights=MAE_CKPT, batch_size=32,
         ranking_lambda=0.5, ranking_neg_frac=1.0,
         flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
         h_drop=0.5, c1_drop=0.15, c2_drop=0.155,
         cutout_prob=0.5, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
         l1=1e-4, tag="ts_gce_vstrongreg_ctx32fovea_closed_mae"),

    # (4) [NOT YET ACTIVE] regularization run: take the RF winner from tsJe/tsJf/tsJg, then turn
    #     ON the two new regularizers -- AdamW weight decay + TTA-consistency. fill in the winning
    #     arch/labels/context knobs, drop l1 (weight_decay replaces it), then uncomment to run.
    # dict(tid="tsJh", arch="<winner arch>", context_size=32, init_weights=MAE_CKPT, batch_size=32,
    #      ranking_lambda=0.5, ranking_neg_frac=1.0,
    #      flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
    #      h_drop=0.5, c1_drop=0.15, c2_drop=0.155,
    #      cutout_prob=0.5, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
    #      l1=0.0, weight_decay=1e-2, tta_consistency=True, tta_cons_lambda=0.5,
    #      tag="ts_gce_vstrongreg_regv2_mae"),

]


# dict-key -> (config-section, attribute). ONLY keys present in a test dict override the
# matching _base_config value; every other field stays exactly as _base_config set it, so the
# base is the single source of truth (not config.py's dataclass defaults).
_OVERRIDES = {
    "arch":             ("model", "arch"),
    "n_epochs":         ("tra", "n_epochs"),
    "eval_int":         ("tra", "eval_int"),
    "test_int":         ("tra", "test_int"),
    "probe_int":        ("tra", "probe_int"),
    "l1":               ("tra", "l1_lambda"),
    "weight_decay":     ("tra", "weight_decay"),
    "tta_consistency":  ("tra", "tta_consistency"),
    "tta_cons_lambda":  ("tra", "tta_consistency_lambda"),
    "ranking_lambda":   ("tra", "ranking_lambda"),
    "ranking_neg_frac": ("tra", "ranking_neg_frac"),
    "ranking_margin":   ("tra", "ranking_margin"),
    "loss_type":        ("tra", "loss_type"),
    "gce_q":            ("tra", "gce_q"),
    "label_smooth":     ("tra", "label_smooth"),
    "pos_weight_enabled": ("tra", "pos_weight_enabled"),
    "focal_gamma":      ("tra", "focal_gamma"),
    "h_drop":           ("model", "head_drop"),
    "c1_drop":          ("model", "conv1_drop"),
    "c2_drop":          ("model", "conv2_drop"),
    "context_size":     ("data", "context_size"),
    "context_downsample": ("data", "context_downsample"),
    "dense":            ("data", "dense_labels"),
    "ring_label_source": ("data", "ring_label_source"),
    "ring_close_r":     ("data", "ring_close_r"),
    "ring_gap_r":       ("data", "ring_gap_r"),
    "ring_shell_r":     ("data", "ring_shell_r"),
    "batch_size":       ("dl", "batch_size"),
    "num_workers":      ("dl", "num_workers"),
    "flip":             ("dl", "flip_prob"),
    "rotation":         ("dl", "rotation_prob"),
    "noise":            ("dl", "noise_prob"),
    "brightness":       ("dl", "brightness_prob"),
    "contrast":         ("dl", "contrast_prob"),
    "cutout_prob":      ("dl", "cutout_prob"),
    "cutout_max_frac":  ("dl", "cutout_max_frac"),
    "cutout_n_patches": ("dl", "cutout_n_patches"),
    "depth_mask_prob":  ("dl", "depth_mask_prob"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_twostage_2026_07_21_{tid}_{tag}")

    # apply ONLY the keys present in this test dict; all else stays as _base_config set it
    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            setattr(getattr(c, sec), attr, t[k])

    # warm-start from an MAE checkpoint if the file exists (skips cleanly if MAE hasn't run)
    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[twostage] init_weights '{iw}' not found -- training {tid} from scratch")

    # data_aug reflects the FINAL aug probabilities (base or overridden), not a hardcoded flag
    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0

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
