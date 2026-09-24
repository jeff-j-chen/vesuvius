"""campaign_archs_12.py -- ctx192/ds2 modifier sweep on the 4-scroll fast subset (2026-08-17).

BASELINE (already tested in archs11, not rerun here):
  ctx192_fast = ctx192/ds2, FAST_SCROLLS, SOFT_AUGS, TTA, attn-MIL, supcon, MAE init

GOAL: probe the key regularization/label modifiers at ctx192/ds2 resolution.

TESTS:
  nonoise     -- flip+rot only; all photometric augmentations off (ablate aug contribution)
  oldspill    -- variance-based spill regularization (lambda=0.4, min_depth_var=0.5)
  dann        -- DANN with GRL annealing (lambda=0.25, anneal 0->1)
  oldspill_dann -- conjugate: both spill and DANN active
  eroded2     -- same baseline but labels from eroded2_inklabels (~36% additional erosion)

MEMORY NOTE (RTX 5090 / 32GB):
  ctx192/ds2 activations ~0.495 GB/sample; batch=48 -> ~26 GB (safe).
  batch=96 would need ~47 GB -- does not fit. eval_infer_bs=96 is fine (no grad storage).

CONFIGURATION:
  12 epochs, eval at epoch 12 only, probes every 6 epochs, fast_eval_figure=True
  eval_int_scrolls=4 (all 4 vis scrolls: 500P2_front, w013, seg46527, w035)
  batch=48, lr=1.8e-4 (sqrt(48/32) scaled from default), eval_bs=96, workers=8

  python campaign_archs_12.py --dry-run
  python campaign_archs_12.py
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

LOG_DIR = "./runs_archs12"
ALL_SCROLLS = list(DEFAULT_SCROLLS)
_MAE_CTX96 = "models/mae_nnunet_96.pth"

_FAST_IDS = {20250628074500, 20240304141531, 20260226000000, 20260317000000}
FAST_SCROLLS = [s for s in ALL_SCROLLS if s.scroll_id in _FAST_IDS]

SOFT_AUGS = dict(
    flip_prob=0.25,
    rotation_prob=0.25,
    noise_prob=0.15,
    brightness_prob=0.15,
    contrast_prob=0.15,
    cutout_prob=0,
    cutout_max_frac=0.1,
    cutout_n_patches=1,
    depth_mask_prob=0.0,
)

PHOTOM = dict(
    flip_prob=0.6,
    rotation_prob=0.6,
    noise_prob=0.3,
    brightness_prob=0.6,
    contrast_prob=0.6,
    cutout_prob=0,
    cutout_max_frac=0,
    cutout_n_patches=0,
    depth_mask_prob=0.0,
)

# flip + rotation only; all photometric noise off
NO_NOISE_AUGS = dict(
    flip_prob=0.25,
    rotation_prob=0.25,
    noise_prob=0.0,
    brightness_prob=0.0,
    contrast_prob=0.0,
    cutout_prob=0.0,
    cutout_max_frac=0.0,
    cutout_n_patches=0,
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

_BATCH = 48
_LR = 1.8e-4   # sqrt(48/32) * 1.5e-4; keeps per-sample effective LR matched to baseline
_EVAL_BS = 96
_WORKERS = 8


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
    c.data.context_size = 192
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
    c.tra.lr = _LR
    c.data.eval_infer_bs = _EVAL_BS
    c.tra.eval_int_scrolls = 4
    c.data.vis_scroll_ids = [20250628074500, 20240304141531, 20260226000000, 20260317000000]
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
    c.tra.dann_n_domains = len(FAST_SCROLLS)
    c.dl.batch_size = _BATCH
    c.dl.num_workers = _WORKERS
    c.dl.data_aug = True
    c.data.mask_memmap = True
    setattr(c.data, "mask_bitpack", True)
    c.data.ring_negatives = True
    c.data.ring_close_r = 3
    c.data.ring_gap_r = 3
    c.data.ring_shell_r = 2
    c.tra.fast_eval_figure = True
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


_BASE12 = dict(
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


def _mk12(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE12)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


TESTS = [
    # ctx192_fast baseline tested in archs11; not rerun
    # _mk12("baseline", "ctx192_ds2_softaug_tta_attn_supcon_mae"),

    # _mk12(
    #     "nonoise",
    #     "192_ds2_nonoise",
    #     **NO_NOISE_AUGS,
    # ),
    _mk12(
        "oldspill",
        "192_ds2_oldspill",
        spill_reduction=True,
        spill_lambda=0.4,
        spill_min_depth_var=0.5,
    ),
    _mk12(
        "dots",
        "192_ds2_dots",
        # inject sparse positive-only tiles from dot labels (no ring negatives from dots)
        dot_inklabel_dir="./dots",
    ),
    _mk12(
        "dann",
        "192_ds2_dann",
        dann=True,
        dann_lambda=0.25,
        dann_grl_anneal=True,
    ),
    _mk12(
        "oldspill_dann",
        "192_ds2_oldspill_dann",
        spill_reduction=True,
        spill_lambda=0.4,
        spill_min_depth_var=0.5,
        dann=True,
        dann_lambda=0.25,
        dann_grl_anneal=True,
    ),
    _mk12(
        "eroded2",
        "192_ds2_eroded2",
        inklabel_dir="./eroded2_inklabels",
    ),
    _mk12(
        "fda",
        "192_ds2_fda",
        # beta=0.05 swaps wavelengths > ~48px (background texture); ink strokes are high-freq and survive
        fda_prob=0.5,
        fda_beta=0.05,
    ),
    _mk12(
        "ibn",
        "192_ds2_ibn",
        # IBN-a in enc1+enc2: IN strips fragment-specific amplitude style, BN preserves content
        use_ibn=True,
    ),
    _mk12(
        "cross_frag_sc",
        "192_ds2_cross_frag_supcon",
        # restrict supcon positives to cross-fragment pairs only -> forces fragment-invariant ink embedding
        supcon_cross_frag=True,
    ),
    _mk12(
        "prototype",
        "192_ds2_prototype",
        # replace bag score (MIL) with online prototype cosine classifier over bottleneck embedding
        use_prototype=True,
    ),
    _mk12(
        "elastic",
        "192_ds2_elastic",
        # elastic deformation: preserves ink signal (no cutout), warps spatial shape only
        # alpha=15px displacement, sigma=5 smoothing -- moderate for 192x192 context
        elastic_prob=0.5,
        elastic_alpha=15.0,
        elastic_sigma=5.0,
    ),
    _mk12(
        "photom",
        "192_ds2_photom",
        **PHOTOM,
    ),
    _mk12(
        "entropy_min",
        "192_ds2_entropy_min",
        # maximize entropy on unlabeled (validation) tiles; attacks confident not-ink fixed point
        entropy_min_lambda=0.1,
    ),
    # COMBO TESTS: stack of confirmed-safe conjugates
    _mk12(
        "combo1",
        "192_ds2_combo1",
        fda_prob=0.5, fda_beta=0.05,
        elastic_prob=0.5, elastic_alpha=15.0, elastic_sigma=5.0,
        inklabel_dir="./eroded2_inklabels",
        dot_inklabel_dir="./dots",
        dann=True, dann_lambda=0.25, dann_grl_anneal=True,
        spill_reduction=True, spill_lambda=0.4, spill_min_depth_var=0.5,
    ),
    _mk12(
        "combo2",
        "192_ds2_combo2",
        fda_prob=0.5, fda_beta=0.05,
        elastic_prob=0.5, elastic_alpha=15.0, elastic_sigma=5.0,
        inklabel_dir="./eroded2_inklabels",
        dot_inklabel_dir="./dots",
        dann=True, dann_lambda=0.25, dann_grl_anneal=True,
        spill_reduction=True, spill_lambda=0.4, spill_min_depth_var=0.5,
        entropy_min_lambda=0.1,
        use_ibn=True,
        use_prototype=True,
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
    "spill_prob": ("tra", "spill_prob"),
    "gce_q": ("tra", "gce_q"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
    "scrolls": ("data", "scrolls"),
    "eval_int_scrolls": ("tra", "eval_int_scrolls"),
    "batch_size": ("dl", "batch_size"),
    "eval_infer_bs": ("data", "eval_infer_bs"),
    "lr": ("tra", "lr"),
    "num_workers": ("dl", "num_workers"),
    "vis_scroll_ids": ("data", "vis_scroll_ids"),
    "inklabel_dir": ("data", "inklabel_dir"),
    "fda_prob": ("dl", "fda_prob"),
    "fda_beta": ("dl", "fda_beta"),
    "use_ibn": ("model", "use_ibn"),
    "supcon_cross_frag": ("tra", "supcon_cross_frag"),
    "use_prototype": ("model", "use_prototype"),
    "prototype_ema": ("model", "prototype_ema"),
    "elastic_prob": ("dl", "elastic_prob"),
    "elastic_alpha": ("dl", "elastic_alpha"),
    "elastic_sigma": ("dl", "elastic_sigma"),
    "entropy_min_lambda": ("tra", "entropy_min_lambda"),
    "entropy_min_batch_size": ("tra", "entropy_min_batch_size"),
    "dot_inklabel_dir": ("data", "dot_inklabel_dir"),
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
        print(f"[archs12] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([
        c.dl.flip_prob,
        c.dl.rotation_prob,
        c.dl.noise_prob,
        c.dl.brightness_prob,
        c.dl.contrast_prob,
        c.dl.cutout_prob,
        c.dl.depth_mask_prob,
        getattr(c.dl, "elastic_prob", 0.0),
    ])
    os.makedirs("models/archs12", exist_ok=True)
    setattr(c, "save_final", f"models/archs12/{tid}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"\n{'=' * 70}\n[archs12] {c.exp_name}\n{'=' * 70}", flush=True)
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
        f"  learned_surface={c.model.learned_surface}  inklabel_dir={c.data.inklabel_dir}"
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
    ap = argparse.ArgumentParser(description="campaign_archs_12: ctx192/ds2 modifier sweep")
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

    print(f"[archs12] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs12] 4-scroll fast subset, ctx192/ds2, batch={_BATCH}, lr={_LR:.1e}")
    print("[archs12] baseline = ctx192_fast (archs11; not rerun)")

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

    print(f"\n{'=' * 70}\n[archs12] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
