"""campaign_ts_ctx.py -- context-architecture sweep for the two-stage MIL detector.

continues from the ISO sanity runs (campaign_runner_twostage.py) that reproduced the original
tsJ / tsJd results on the home PC. here we hold the proven ctx32 baseline fixed and vary ONLY
the context handling / ring label source, so each run is a clean single-variable delta.

SEG46527 IS ON for this ENTIRE campaign (PHerc0814, 20260226000000 -> the full 15-scroll set).
test 1 is the final sanity check: the exact ctx32 ISO config but WITH seg46527, 5 epochs. if it
still trains as expected, the extra fragment is definitively NOT the source of past discrepancies.

tests:
  1. c1sanity : ctx32 closed ring, seg ON, 5 epochs (probes only)   -- seg46527 sanity check
  2. c2eroded : ctx32 ERODED ring labels (vs closed)                 -- tight letter-tracing labels
  3. c3coarse : ctx32 closed + coarse context (context_downsample=2) -- full extent, half res
  4. c4fovea  : foveated context (full-res center + coarse surround) -- keep the middle sharp
  5/6 (commented): pick the winner of 1-4, then add AdamW weight decay / TTA-consistency.

every run auto-saves its full resolved config to config.json in its own TB run dir (train.py).

run all:   python campaign_ts_ctx.py
dry-run:   python campaign_ts_ctx.py --dry-run
run one:   python campaign_ts_ctx.py --only c2eroded
run some:  python campaign_ts_ctx.py --only c2eroded,c7ctx48   (comma-separated, runs in TESTS order)
run from:  python campaign_ts_ctx.py --from c3coarse
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config

INTER_RUN_COOLDOWN_SECS = 120

# MAE checkpoint every run warm-starts from (stage1.* transfers; stage2 stays fresh).
MAE_CKPT = "models/mae_twostage.pth"


def _base_config(exp_name: str) -> Config:
    """fresh config for the context sweep. matches the twostage base EXCEPT seg46527 stays IN
    (full 15-scroll set) and logs to ./runs_ts_ctx."""
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "v15_twostage_lcn"
    c.data.tile_size     = 16
    c.data.depth         = 24
    c.data.train_d_start = 4
    c.data.train_d_end   = 28
    c.data.d_start = 4
    c.data.d_end   = 28
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    c.model.head_drop  = 0.0
    c.tra.n_epochs     = 15
    c.tra.eval_int     = 15
    c.tra.test_int     = 999
    c.tra.probe_int    = 5
    c.tra.save_int     = 2
    c.tra.log_dir      = "./runs_ts_ctx"

    c.tra.deterministic = False
    c.tra.l1_lambda    = 0.0
    c.tra.weight_decay = 0.0
    c.dl.batch_size    = 96
    c.dl.num_workers   = 4
    c.dl.data_aug      = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "closed"
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    c.tra.epoch_cooldown_secs   = 9*5 * 0
    c.tra.val_cooldown_secs     = 12*5 * 0
    c.tra.eval_cooldown_secs    = 60*5 * 0
    c.tra.fig_chunk_cooldown_ms = 60*5 * 0
    # NOTE: seg46527 (20260226000000) is INTENTIONALLY kept in for this whole campaign.
    return c


# common ctx32 baseline (= the proven tsJdISO 'ctx32d_ISO_noseg_lowerreg' aug/reg block).
# tests 2-4 are single-variable deltas off this so the context change is the ONLY difference.
_CTX32 = dict(
    arch="v15_twostage_wide_zgrad_ctx", context_size=32, batch_size=32, init_weights=MAE_CKPT,
    ranking_lambda=0.5, ranking_neg_frac=1.0,
    flip=0.6, rotation=0.6, noise=0.3, brightness=0.6, contrast=0.6,
    h_drop=0.4, c1_drop=0.15, c2_drop=0.15,
    cutout_prob=0.4, cutout_max_frac=0.2, cutout_n_patches=2, depth_mask_prob=0.0,
    l1=7e-5,
)


def _mk(tid, tag, **overrides):
    """build a test dict from the shared _CTX32 baseline + per-test overrides."""
    d = dict(_CTX32); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


# closed ctx48/ds2 baseline for the regularization sweep (= _CTX32 aug/reg block + 48px coarse
# context + 10ep/eval10/probe5, matching c8c48closed). eval_cmap_norm defaults to raw (config).
_CTX48 = dict(context_size=48, context_downsample=2, n_epochs=10, eval_int=10, probe_int=5)


def _mk48(tid, tag, **overrides):
    """closed ctx48/ds2 baseline + per-test overrides (regularization sweep)."""
    d = dict(_CTX48); d.update(overrides)
    return _mk(tid, tag, **d)


TESTS = [
    # # 1) SANITY: exact ctx32 ISO config but seg46527 ON + only 5 epochs. final check that the
    # #    extra PHerc0814 fragment is not the source of past discrepancies.
    # _mk("c1sanity", "ctx32_closed_segON_5ep_sanity"),

    # # 2) ERODED ring labels instead of closed -- tighter positives that trace the letter.
    # _mk("c2eroded", "ctx32_eroded_segON", ring_label_source="eroded"),

    # # 2b) WIDER context: 48px extent at half res (ds2) -- larger surround, same closed ring.
    # _mk("c7ctx48", "ctx48ds2_closed_segON", context_size=48, context_downsample=2),

    # # 3) COARSE context at the same 32px extent (avg-pool input 2x) -- full window, half res.
    # _mk("c3coarse", "ctx32ds2_closed_segON", context_downsample=2),

    # # 4) FOVEATED context -- full-res central tile + coarse full-extent surround, fused pre-MIL.
    # _mk("c4fovea", "ctx32fovea_closed_segON", arch="v15_twostage_wide_zgrad_fovea"),

    # # 8) 48px COARSE (ds2) CLOSED, 10ep -- readability A/B leg 1 (probe@5, eval@10).
    # _mk("c8c48closed", "ctx48ds2_closed_10ep", context_size=48, context_downsample=2,
    #     n_epochs=10, eval_int=10, probe_int=5, eval_cmap_norm="raw"),

    # # 9) 48px COARSE (ds2) ERODED, 10ep -- readability A/B leg 2. eroded reads far better than
    # #    its counting stats suggest; verify that holds at the wider coarse context.
    # _mk("c9c48eroded", "ctx48ds2_eroded_10ep", context_size=48, context_downsample=2,
    #     ring_label_source="eroded", n_epochs=10, eval_int=10, probe_int=5, eval_cmap_norm="raw"),

    # 5) [COMMENTED] winner of 1-4 + AdamW weight decay (drop l1). set arch/context to the winner.
    # _mk("c5wd", "WINNER_wd", weight_decay=1e-2, l1=0.0),

    # 6) [COMMENTED] winner of 1-4 + TTA-consistency. set arch/context to the winner.
    # _mk("c6tta", "WINNER_tta", tta_consistency=True, tta_cons_lambda=0.5),

    # ========================================================================================
    # REGULARIZATION SWEEP  (baseline = closed ctx48/ds2, 10ep, l1=7e-5).  run in two batches:
    #   python campaign_ts_ctx.py --only cons025f,cons05f,cons05d,cons075f,cons075d
    #   python campaign_ts_ctx.py --only wd1e3,wd3e3,wd1e2,wd3e2,wd1e1,l1r_wd3e3,l1r_wd1e2,l1max_wd1e3
    # baseline for comparison is the already-trained c8c48closed (same config, no consistency, wd=0).
    # ========================================================================================

    # --- TTA-consistency sweep (eroded ring labels, ctx48 ds2) ---
    _mk48("cons025f",  "ctx48_cons0p25_flips_eroded",    tta_consistency=True, tta_cons_lambda=0.25, tta_cons_mode="flips", ring_label_source="eroded"),
    _mk48("cons025d",  "ctx48_cons0p25_dihedral_eroded", tta_consistency=True, tta_cons_lambda=0.25, tta_cons_mode="dihedral", ring_label_source="eroded"),
    _mk48("cons01f",   "ctx48_cons0p1_flips_eroded",     tta_consistency=True, tta_cons_lambda=0.1,  tta_cons_mode="flips", ring_label_source="eroded"),
    _mk48("cons05f",   "ctx48_cons0p5_flips_eroded",     tta_consistency=True, tta_cons_lambda=0.5,  tta_cons_mode="flips", ring_label_source="eroded"),
    _mk48("cons05d",   "ctx48_cons0p5_dihedral_eroded",  tta_consistency=True, tta_cons_lambda=0.5,  tta_cons_mode="dihedral", ring_label_source="eroded"),

    # # --- [COMMENTED OUT] tests for different consistency modes/lambdas with closed ring ---
    # _mk48("cons075f",  "ctx48_cons0p75_flips",    tta_consistency=True, tta_cons_lambda=0.75, tta_cons_mode="flips"),
    # _mk48("cons075d",  "ctx48_cons0p75_dihedral", tta_consistency=True, tta_cons_lambda=0.75, tta_cons_mode="dihedral"),

    # # --- [COMMENTED OUT] AdamW weight-decay sweep. l1=7e-5 is the l1 CEILING (>that stops learning),
    # #     so the pure-wd legs DROP l1 (=0) to let weight decay be the sole regularizer. l1 (drives
    # #     weights to exactly 0 -> sparsity) and wd (shrinks all weights -> smoothness) regularize
    # #     DIFFERENTLY, so a mix is worth testing -- but stacking wd on the FULL l1 likely exceeds
    # #     the total-reg ceiling, so combos use a REDUCED l1 (3e-5); l1max_wd1e3 probes that edge. ---
    # _mk48("wd1e3",     "ctx48_wd1e3_nol1",   weight_decay=1e-3, l1=0.0),
    # _mk48("wd3e3",     "ctx48_wd3e3_nol1",   weight_decay=3e-3, l1=0.0),
    # _mk48("wd1e2",     "ctx48_wd1e2_nol1",   weight_decay=1e-2, l1=0.0),
    # _mk48("wd3e2",     "ctx48_wd3e2_nol1",   weight_decay=3e-2, l1=0.0),
    # _mk48("wd1e1",     "ctx48_wd1e1_nol1",   weight_decay=1e-1, l1=0.0),
    # _mk48("l1r_wd3e3", "ctx48_l1_3e5_wd3e3", l1=3e-5, weight_decay=3e-3),   # combo: reduced l1 + light wd
    # _mk48("l1r_wd1e2", "ctx48_l1_3e5_wd1e2", l1=3e-5, weight_decay=1e-2),   # combo: reduced l1 + moderate wd
    # _mk48("l1max_wd1e3", "ctx48_l1_7e5_wd1e3", l1=7e-5, weight_decay=1e-3), # combo: full l1 + a touch of wd
]


# dict-key -> (config-section, attribute). ONLY keys present in a test dict override _base_config.
_OVERRIDES = {
    "arch":             ("model", "arch"),
    "n_epochs":         ("tra", "n_epochs"),
    "eval_int":         ("tra", "eval_int"),
    "eval_int_scrolls": ("tra", "eval_int_scrolls"),
    "test_int":         ("tra", "test_int"),
    "probe_int":        ("tra", "probe_int"),
    "l1":               ("tra", "l1_lambda"),
    "weight_decay":     ("tra", "weight_decay"),
    "tta_consistency":  ("tra", "tta_consistency"),
    "tta_cons_lambda":  ("tra", "tta_consistency_lambda"),
    "tta_cons_mode":    ("tra", "tta_consistency_mode"),
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
    "eval_cmap_norm":   ("data", "eval_cmap_norm"),
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
    "brightness_delta": ("dl", "brightness_delta"),
    "contrast_delta":   ("dl", "contrast_delta"),
    "noise_std_min":    ("dl", "noise_std_min"),
    "noise_std_max":    ("dl", "noise_std_max"),
    "cutout_prob":      ("dl", "cutout_prob"),
    "cutout_max_frac":  ("dl", "cutout_max_frac"),
    "cutout_n_patches": ("dl", "cutout_n_patches"),
    "depth_mask_prob":  ("dl", "depth_mask_prob"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_tsctx_2026_07_27_{tid}_{tag}")

    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            setattr(getattr(c, sec), attr, t[k])

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[tsctx] init_weights '{iw}' not found -- training {tid} from scratch")

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
    print(f"\n{'='*70}\n[tsctx] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}  "
          f"ring={c.data.ring_label_source}  n_epochs={c.tra.n_epochs}")
    print(f"  L1={c.tra.l1_lambda:.1e}  wd={c.tra.weight_decay:.1e}  rank={getattr(c.tra,'ranking_lambda',0.0)}  "
          f"scrolls={len(c.data.scrolls)}")
    print(f"  flip={c.dl.flip_prob} rot={c.dl.rotation_prob} noise={c.dl.noise_prob} "
          f"bright={c.dl.brightness_prob} contrast={c.dl.contrast_prob}  h_drop={c.model.head_drop}  "
          f"cutout={c.dl.cutout_prob}/{c.dl.cutout_n_patches}patch")
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
    ap = argparse.ArgumentParser(description="two-stage context-architecture sweep")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [t for t in TESTS if t["tid"] in want]  # keeps TESTS order
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

    print(f"[tsctx] {len(selected)} test(s) queued  (full 15-scroll set, seg46527 ON, log -> ./runs_ts_ctx)")

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

    print(f"\n{'='*70}\n[tsctx] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
