"""campaign_archs_11.py -- baseline-family follow-up on the 18-fragment default set (2026-08-14).

GOAL: promote the new all-data baseline and probe the next structural follow-ups now that
PHerc1667 w013 is part of the default training pool.

BASELINE GOING FORWARD:
  baseline_softaug = nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_supcon_learnsurf_mae

TRAINING SET:
  all 18 default fragments from utils.config DEFAULT_SCROLLS, including:
    - PHerc0139 fragments
    - PHerc0814 seg46527
    - PHerc0500P2 500P2_front
    - PHerc1667 w013

NOTE ON MAE WARM-STARTS:
  models/mae_nnunet_96.pth is used for BOTH ctx96 and ctx128 tests. this backbone is fully
  convolutional, so the checkpoint is shape-compatible across the larger context size.

PHASE 1 RESULTS (2026-08-14, all 5 ctx96 runs + baseline_ctx128):
  - DANN: mixed per-patch; net positive but kills low-contrast regions (fixed-grl_scale=1.0 is the
    root cause; annealing warmup added below)
  - regular aug: severely damaging -- the dz stem makes this model sensitive to per-slice jitter;
    soft aug is correct; could try even weaker or zero
  - spill reduction (prob-based): beneficial but confidence-capping: equilibrium lands at logit~0
    because the prob-space penalty fights the classification gradient; replaced with logit-variance
  - ctx128: simply better, consistent with prior results

PHASE 2 (this run): ctx128 variants only, probing modified DANN + modified spill
  9: ctx128 + DANN with grl annealing (lambda=0.25, schedule 0->1 over training)
  10: ctx128 + variance-based spill (logit std target, lambda=0.4, min_depth_var=0.5)

CONFIGURATION:
  12 epochs, eval at epoch 12 only, probes every 6 epochs
  test_int=999, fast_eval_figure=False, eval_int_scrolls=4
    selected eval scrolls: 500P2_front, w013, seg46527, w035
  batch=32, eval_bs=64, workers=4
  keep default GCE q=0.7, NO asymmetric label smoothing

  python campaign_archs_11.py --dry-run
  python campaign_archs_11.py --only dann_ctx128_annealed,spill_ctx128_var
  python campaign_archs_11.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS
from utils.platform import get_default_lr, get_zarr_dir

LOG_DIR = "./runs_archs11"
ALL_SCROLLS = list(DEFAULT_SCROLLS)
_MAE_CTX96 = "models/mae_nnunet_96.pth"

# 4-scroll fast subset matching vis_scroll_ids: 500P2_front, w013, seg46527, w035
_FAST_IDS = {20250628074500, 20240304141531, 20260226000000, 20260317000000}
FAST_SCROLLS = [s for s in ALL_SCROLLS if s.scroll_id in _FAST_IDS]

SOFT_AUGS = dict(
    flip_prob=0.25,
    rotation_prob=0.25,
    noise_prob=0.15,
    brightness_prob=0.15,
    contrast_prob=0.15,
    cutout_prob=0.1,
    cutout_max_frac=0.1,
    cutout_n_patches=1,
    depth_mask_prob=0.0,
)

REGULAR_AUGS = dict(
    flip_prob=0.6,
    rotation_prob=0.6,
    noise_prob=0.3,
    brightness_prob=0.6,
    contrast_prob=0.6,
    cutout_prob=0.4,
    cutout_max_frac=0.2,
    cutout_n_patches=2,
    depth_mask_prob=0.0,
)

TTA_FLIPS = dict(
    tta_consistency=True,
    tta_consistency_lambda=0.3,
    tta_consistency_mode="flips",
)

SPATIAL_SUPCON = dict(
    supcon=True,
    supcon_curriculum=True,
    supcon_lambda_start=0.05,
    supcon_lambda_end=0.5,
    supcon_curriculum_epochs=10,
    supcon_temp=0.07,
)

GCE_ASYM = dict(
    gce_q=0.9,
    label_smooth_pos=0.25,
    label_smooth_neg=0.02,
)


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "nnunet3d_lcndz"

    c.data.zarr_path = get_zarr_dir()
    c.data.scrolls = list(FAST_SCROLLS)

    c.data.tile_size = 16
    c.data.depth = 24
    c.data.train_d_start = 4
    c.data.train_d_end = 28
    c.data.d_start = 4
    c.data.d_end = 28
    c.data.context_size = 96
    c.data.context_downsample = 2
    c.model.conv1_drop = 0.15
    c.model.conv2_drop = 0.15
    c.model.head_drop = 0.4
    c.tra.n_epochs = 12
    c.tra.eval_int = 12
    c.tra.test_int = 999
    c.tra.probe_int = 6
    c.tra.save_int = 6
    c.tra.log_dir = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = 64
    c.tra.eval_int_scrolls = 1
    c.data.vis_scroll_ids = [20250628074500, 20240304141531, 20260226000000, 20260317000000]
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
    c.tra.dann_n_domains = len(FAST_SCROLLS)
    c.dl.batch_size = 32
    c.dl.num_workers = 4
    c.dl.data_aug = True
    c.data.mask_memmap = True
    setattr(c.data, "mask_bitpack", True)
    c.data.ring_negatives = True
    c.data.ring_close_r = 3
    c.data.ring_gap_r = 3
    c.data.ring_shell_r = 2
    c.tra.fast_eval_figure = False
    c.dl.flip_prob = 0.0
    c.dl.rotation_prob = 0.0
    c.dl.noise_prob = 0.0
    c.dl.brightness_prob = 0.0
    c.dl.contrast_prob = 0.0
    c.dl.cutout_prob = 0.0
    c.dl.cutout_max_frac = 0.0
    c.dl.cutout_n_patches = 0
    c.dl.depth_mask_prob = 0.0
    c.tra.epoch_cooldown_secs = 0
    c.tra.val_cooldown_secs = 0
    c.tra.eval_cooldown_secs = 0
    c.tra.fig_chunk_cooldown_ms = 0
    return c


_BASE11 = dict(
    arch="nnunet3d_lcndz",
    init_weights=_MAE_CTX96,
    attn_mil=True,
    attn_entropy_weight=0.03,
    learned_surface=True,
    dann=False,
    dann_lambda=0.0,
    spill_reduction=False,
    spill_lambda=0.0,
    spill_depth_threshold=0.35,
    spill_active_depth_tau=0.08,
    spill_max_active_depth_frac=0.35,
    **SOFT_AUGS,
    **TTA_FLIPS,
    **SPATIAL_SUPCON,
)


def _mk11(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE11)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


TESTS = [
    # PHASE 1 (ctx96 + baseline ctx128) -- all completed 2026-08-14; commented out
    # _mk11(
    #     "baseline_softaug",
    #     "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_supcon_learnsurf_mae",
    # ),
    # _mk11(
    #     "regularaug_softaug",
    #     "nnunet3d_ds2_lcndz_ctx96_regularaug_tta_attn_supcon_learnsurf_mae",
    #     **REGULAR_AUGS,
    # ),
    # _mk11(
    #     "dann_softaug",
    #     "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_supcon_learnsurf_dann035_mae",
    #     dann=True,
    #     dann_lambda=0.35,
    # ),
    # _mk11(
    #     "spill_softaug",
    #     "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_supcon_learnsurf_spillred_mae",
    #     spill_reduction=True,
    #     spill_lambda=0.6,
    # ),
    # _mk11(
    #     "baseline_ctx128",
    #     "nnunet3d_ds2_lcndz_ctx128_softaug_tta_attn_supcon_learnsurf_mae96",
    #     context_size=128,
    # ),
    # _mk11(
    #     "regularaug_ctx128",
    #     "nnunet3d_ds2_lcndz_ctx128_regularaug_tta_attn_supcon_learnsurf_mae96",
    #     context_size=128,
    #     **REGULAR_AUGS,
    # ),
    # _mk11(
    #     "dann_ctx128",
    #     "nnunet3d_ds2_lcndz_ctx128_softaug_tta_attn_supcon_learnsurf_dann035_mae96",
    #     context_size=128,
    #     dann=True,
    #     dann_lambda=0.35,
    # ),
    # _mk11(
    #     "spill_ctx128",
    #     "nnunet3d_ds2_lcndz_ctx128_softaug_tta_attn_supcon_learnsurf_spillred_mae96",
    #     context_size=128,
    #     spill_reduction=True,
    #     spill_lambda=0.6,
    # ),

    # PHASE 2a: ctx128 fast explorations -- completed; ctx128_gceasym ran as no-op (GCE_ASYM
    # overrides were missing from _OVERRIDES; fixed -- re-run included below)
    # _mk11("ctx128_fast", "ctx128_fast", context_size=128),
    # _mk11("ctx128_ds4", "ctx128_ds4", context_size=128, context_downsample=4),
    _mk11("ctx128_gceasym", "ctx128_gceasym", context_size=128, **GCE_ASYM),

    # PHASE 2b: ctx128 modifier tests
    _mk11(
        "ctx128_dann",
        "ctx128_dann",
        context_size=128,
        dann=True,
        dann_lambda=0.25,
        dann_grl_anneal=True,
    ),
    _mk11(
        "ctx128_spill",
        "ctx128_spill",
        context_size=128,
        spill_reduction=True,
        spill_lambda=0.4,
        spill_min_depth_var=0.5,
    ),

    # PHASE 2c: ctx192 tests (192x192 context, ds2, 4-scroll fast subset)
    _mk11(
        "ctx192_fast",
        "ctx192_fast",
        context_size=192,
    ),
    _mk11(
        "ctx192_newspill",
        "ctx192_newspill",
        context_size=192,
        spill_entropy=True,
        spill_entropy_lambda=0.3,
        spill_max_depth_entropy=2.1,
    ),
    _mk11(
        "ctx192_fullscroll",
        "ctx192_fullscroll",
        context_size=192,
        scrolls=list(ALL_SCROLLS),
        eval_int_scrolls=4,
        dann_n_domains=len(ALL_SCROLLS),
    ),
    _mk11(
        "ctx192_fullscroll_fullres",
        "ctx192_fullscroll_fullres",
        context_size=192,
        context_downsample=1,
        scrolls=list(ALL_SCROLLS),
        eval_int_scrolls=4,
        dann_n_domains=len(ALL_SCROLLS),
    ),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "learned_surface": ("model", "learned_surface"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "probe_int": ("tra", "probe_int"),
    "tta_consistency": ("tra", "tta_consistency"),
    "tta_consistency_lambda": ("tra", "tta_consistency_lambda"),
    "tta_consistency_mode": ("tra", "tta_consistency_mode"),
    "context_size": ("data", "context_size"),
    "context_downsample": ("data", "context_downsample"),
    "flip_prob": ("dl", "flip_prob"),
    "rotation_prob": ("dl", "rotation_prob"),
    "noise_prob": ("dl", "noise_prob"),
    "brightness_prob": ("dl", "brightness_prob"),
    "contrast_prob": ("dl", "contrast_prob"),
    "cutout_prob": ("dl", "cutout_prob"),
    "cutout_max_frac": ("dl", "cutout_max_frac"),
    "cutout_n_patches": ("dl", "cutout_n_patches"),
    "depth_mask_prob": ("dl", "depth_mask_prob"),
    "supcon": ("tra", "supcon"),
    "supcon_lambda": ("tra", "supcon_lambda"),
    "supcon_temp": ("tra", "supcon_temp"),
    "supcon_curriculum": ("tra", "supcon_curriculum"),
    "supcon_lambda_start": ("tra", "supcon_lambda_start"),
    "supcon_lambda_end": ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    "dann": ("tra", "dann"),
    "dann_lambda": ("tra", "dann_lambda"),
    "dann_n_domains": ("tra", "dann_n_domains"),
    "dann_grl_anneal": ("tra", "dann_grl_anneal"),
    "spill_reduction": ("tra", "spill_reduction"),
    "spill_lambda": ("tra", "spill_lambda"),
    "spill_depth_threshold": ("tra", "spill_depth_threshold"),
    "spill_active_depth_tau": ("tra", "spill_active_depth_tau"),
    "spill_max_active_depth_frac": ("tra", "spill_max_active_depth_frac"),
    "spill_min_depth_var": ("tra", "spill_min_depth_var"),
    "spill_entropy": ("tra", "spill_entropy"),
    "spill_entropy_lambda": ("tra", "spill_entropy_lambda"),
    "spill_max_depth_entropy": ("tra", "spill_max_depth_entropy"),
    "gce_q": ("tra", "gce_q"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
    "scrolls": ("data", "scrolls"),
    "eval_int_scrolls": ("tra", "eval_int_scrolls"),
    "vis_scroll_ids": ("data", "vis_scroll_ids"),
}


def build_config(t: dict) -> Config:
    tid = str(t["tid"])
    tag = str(t["tag"])
    c = _base_config(tag)
    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: {sec}.{attr} does not exist")
    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs11] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([
        c.dl.flip_prob,
        c.dl.rotation_prob,
        c.dl.noise_prob,
        c.dl.brightness_prob,
        c.dl.contrast_prob,
        c.dl.cutout_prob,
        c.dl.depth_mask_prob,
    ])
    os.makedirs("models/archs11", exist_ok=True)
    setattr(c, "save_final", f"models/archs11/{tid}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"\n{'=' * 70}\n[archs11] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  train_scrolls={len(scroll_ids)}"
    )
    print(f"  scroll_ids={scroll_ids}")
    print(
        f"  n_epochs={c.tra.n_epochs}  probe_int={c.tra.probe_int}  eval_int={c.tra.eval_int}"
        f"  test_int={c.tra.test_int}"
    )
    print(
        f"  fast_eval_figure={c.tra.fast_eval_figure}  eval_int_scrolls={c.tra.eval_int_scrolls}"
        f"  eval_bs={c.data.eval_infer_bs}"
    )
    print(f"  eval_scroll_ids={getattr(c.data, 'vis_scroll_ids', None)}")
    print(
        f"  loss={c.tra.loss_type}  gce_q={c.tra.gce_q}  supcon={c.tra.supcon}"
        f"  learned_surface={c.model.learned_surface}"
    )
    print(
        f"  attn_mil={c.model.attn_mil}  tta_consistency={c.tra.tta_consistency}"
        f"  dann={c.tra.dann}  spill_reduction={c.tra.spill_reduction}"
    )
    print(
        f"  dann_lambda={c.tra.dann_lambda}  dann_grl_anneal={c.tra.dann_grl_anneal}"
        f"  spill_lambda={c.tra.spill_lambda}  spill_min_depth_var={getattr(c.tra, 'spill_min_depth_var', 0.5)}"
    )
    print(
        f"  aug: flip={c.dl.flip_prob} rot={c.dl.rotation_prob} noise={c.dl.noise_prob}"
        f" bright={c.dl.brightness_prob} contrast={c.dl.contrast_prob} cutout={c.dl.cutout_prob}"
    )
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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="campaign_archs_11: 18-fragment baseline-family sweep")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [t for t in TESTS if str(t["tid"]) in want]
        missing = want - {str(t["tid"]) for t in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: {[str(t['tid']) for t in TESTS]}")
            return
    elif args.from_id:
        ids = [str(t["tid"]) for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs11] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs11] default training set: {len(ALL_SCROLLS)} fragments from 4 scroll families")
    print("[archs11] baseline = baseline_softaug (ctx96, ds2, softaug, tta, attn, supcon, learned surface, MAE)")

    results = {}
    for t in selected:
        tid = str(t["tid"])
        c = build_config(t)
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del c
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'=' * 70}\n[archs11] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()