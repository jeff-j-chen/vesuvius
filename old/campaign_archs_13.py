"""campaign_archs_13.py -- focused 2-scroll sweep, combo baseline (2026-08-18).

BASELINE: full combo stack from archs12 + PHOTOM + IBN, 2 scrolls only.
  fda + elastic + eroded2 + dots + dann + oldspill + PHOTOM + IBN + aug_start_epoch=0

SCROLLS (2):
  20240304141531  w013 (PHerc1667) -- only w013 in the FAST set from 1667
  20260115000000  w044 (PHerc0139) -- w044, the original anchor fragment

CONFIGURATION:
  8 epochs, eval at epoch 8, probe_int=999 (never), fast_eval=True
  2 eval figures (vis_scroll_ids = TWO_SCROLLS)
  batch=48, lr=1.8e-4, eval_bs=96, workers=8
  aug_start_epoch=0: augmentation fires from epoch 0 (not the legacy epoch>=5)

  python campaign_archs_13.py --dry-run
  python campaign_archs_13.py
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

from utils.config import Config, DEFAULT_SCROLLS, ScrollConfig
from utils.platform import get_default_lr, get_zarr_dir

LOG_DIR = "./runs_archs13"
ALL_SCROLLS = list(DEFAULT_SCROLLS)
_MAE_CTX96 = "models/mae_nnunet_96.pth"

_TWO_IDS = {20240304141531}
TWO_SCROLLS = [s for s in ALL_SCROLLS if s.scroll_id in _TWO_IDS]

# crop to top 40% of each scroll to cut epoch time; left/right splits unchanged
_CAMPAIGN_SCROLLS = []
for _s in TWO_SCROLLS:
    # if _s.scroll_id == 20240304141531:
    #     _CAMPAIGN_SCROLLS.append(ScrollConfig(
    #         _s.scroll_id, split_axis=_s.split_axis,
    #         train_split_frac=_s.train_split_frac, crop_y_frac=(0.0, 0.4)))
    # elif _s.scroll_id == 20250628074500:
    #     # also uses the new default x-split at 60/40
    #     _CAMPAIGN_SCROLLS.append(ScrollConfig(
    #         _s.scroll_id, split_axis="x",
    #         train_split_frac=0.6, crop_y_frac=(0.0, 0.4)))
    # else:
    _CAMPAIGN_SCROLLS.append(_s)

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
    supcon_curriculum_epochs=8,
    supcon_temp=0.07,
)

_BATCH = 32
_LR = 1e-4
_EVAL_BS = 64
_WORKERS = 8


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "nnunet3d_lcndz"

    c.data.zarr_path = get_zarr_dir()
    c.data.scrolls = list(_CAMPAIGN_SCROLLS)

    c.data.tile_size = 16
    c.data.depth = 24
    c.data.train_d_start = 4
    c.data.train_d_end = 28
    c.data.d_start = 4
    c.data.d_end = 28
    c.data.context_size = 192
    c.data.context_downsample = 2
    c.model.conv1_drop = 0.0  # explicitly disabled for baseline; set non-zero in dropout test
    c.model.conv2_drop = 0.0
    c.model.head_drop = 0.0
    c.tra.n_epochs = 10
    c.tra.eval_int = 10
    c.tra.test_int = 999
    c.tra.probe_int = 999
    c.tra.save_int = 10
    c.tra.log_dir = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = _LR
    c.tra.aug_start_epoch = 0
    c.data.eval_infer_bs = _EVAL_BS
    c.tra.eval_int_scrolls = 2
    c.data.vis_scroll_ids = [20240304141531, 20260115000000]
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
    c.tra.dann_n_domains = len(TWO_SCROLLS)
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


# full combo: all confirmed-safe ingredients from archs12 + PHOTOM + IBN
_BASE13 = dict(
    arch="nnunet3d_lcndz",
    init_weights=_MAE_CTX96,
    attn_mil=True,
    attn_entropy_weight=0.03,
    learned_surface=True,
    # domain regularization (2-scroll: dann_n_domains=2)
    dann=True,
    dann_lambda=0.25,
    dann_grl_anneal=True,
    dann_n_domains=len(TWO_SCROLLS),
    # depth spill regularization
    spill_reduction=True,
    spill_lambda=0.4,
    spill_min_depth_var=0.5,
    spill_depth_threshold=0.35,
    spill_active_depth_tau=0.08,
    spill_max_active_depth_frac=0.35,
    # labels
    inklabel_dir="./eroded2_inklabels",
    dot_inklabel_dir="./dots",
    # input augmentation
    fda_prob=0.5,
    fda_beta=0.05,
    elastic_prob=0.5,
    elastic_alpha=15.0,
    elastic_sigma=5.0,
    **PHOTOM,
    # architecture
    use_ibn=True,
    **TTA_FLIPS,
    **SPATIAL_SUPCON,
)


def _mk13(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE13)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


# shared overrides for the full 18-fragment runs ('all' full-capacity, 'all_half' half-width)
_ALL_KW = dict(
    scrolls=list(ALL_SCROLLS),
    dann_n_domains=len(ALL_SCROLLS),
    # dots only from the four fragments whose dot maps are actually processed
    dot_scroll_whitelist=[20250628074500, 20240304141531, 20260226000000, 20260317000000],
    conv1_drop=0.25,
    conv2_drop=0.25,
    head_drop=0.4,
    skip_drop=0.4,
    depth_jitter=4,
    spill_min_depth_var=0.8,
    spill_lambda=0.5,
    n_epochs=20,
    probe_int=5,
    eval_int=20,
    save_int=5,
    eval_int_scrolls=len(ALL_SCROLLS),
    vis_scroll_ids=[s.scroll_id for s in ALL_SCROLLS],
)


TESTS = [
    _mk13("baseline", "13_baseline"),
    _mk13(
        "jitter_large",
        "13_jitter_large",
        # context position jitter: each training step loads the context offset by ±8px,
        # varying the surrounding tiles visible to the model for the same labeled tile
        ctx_jitter=32,
    ),
    _mk13(
        "skip_drop_hard",
        "13_skip_drop_hard",
        # zero each skip connection with 30% probability; forces decoder to use bottleneck
        # rather than scroll-specific spatial encodings flowing through skip paths
        skip_drop=0.6,
    ),
    _mk13(
        "depth_profile",
        "13_depth_profile",
        # classify using only the depth profile of the center crop: zero spatial capacity,
        # cannot memorize tile coordinates, must learn 'is there an elevated depth-layer signal'
        use_depth_profile=True,
    ),
    _mk13(
        "strong_spill",
        "13_strong_spill",
        # raise min_depth_var 0.5->0.8: demands narrower depth focus so the penalty
        # stays active longer and doesn't vanish after the first epoch
        spill_min_depth_var=0.8,
        spill_lambda=0.5,
    ),
    _mk13(
        "dropout",
        "13_dropout",
        # spatial channel dropout on enc1/enc2 + head; matching old v16_arch_ctx values
        conv1_drop=0.25,
        conv2_drop=0.25,
        head_drop=0.5,
    ),
    _mk13(
        "depth_jitter",
        "13_depth_jitter",
        # shift depth window start by ±4 slices: ink peak lands at different relative position
        # directly attacks 'ink at absolute depth 12' memorization
        depth_jitter=4,
    ),
    _mk13(
        "no_dz",
        "13_no_dz",
        # zero the dz input channel; if model still learns: ink is in raw intensity, not depth gradient
        # if learning collapses: depth gradient IS the primary ink discriminator
        no_dz=True,
    ),
    # full 18-fragment runs. 'all' = full capacity; 'all_half' = half channel width (~4x fewer
    # conv FLOPs, ~1.5M params). NB: the MAE warm-start is full-width, so 'all_half' trains from
    # scratch (shape-mismatched tensors are skipped at load).
    _mk13("all", "13_all", **_ALL_KW),
    _mk13("all_half", "13_all_half", channels_mult=0.5, **_ALL_KW),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "learned_surface": ("model", "learned_surface"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "probe_int": ("tra", "probe_int"),
    "aug_start_epoch": ("tra", "aug_start_epoch"),
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
    "dot_inklabel_dir": ("data", "dot_inklabel_dir"),
    "fda_prob": ("dl", "fda_prob"),
    "fda_beta": ("dl", "fda_beta"),
    "use_ibn": ("model", "use_ibn"),
    "use_prototype": ("model", "use_prototype"),
    "prototype_ema": ("model", "prototype_ema"),
    "supcon_cross_frag": ("tra", "supcon_cross_frag"),
    "elastic_prob": ("dl", "elastic_prob"),
    "elastic_alpha": ("dl", "elastic_alpha"),
    "elastic_sigma": ("dl", "elastic_sigma"),
    "entropy_min_lambda": ("tra", "entropy_min_lambda"),
    "entropy_min_batch_size": ("tra", "entropy_min_batch_size"),
    "ctx_jitter": ("data", "ctx_jitter"),
    "skip_drop": ("model", "skip_drop"),
    "use_depth_profile": ("model", "use_depth_profile"),
    "depth_jitter": ("data", "depth_jitter"),
    "no_dz": ("model", "no_dz"),
    "conv1_drop": ("model", "conv1_drop"),
    "conv2_drop": ("model", "conv2_drop"),
    "head_drop": ("model", "head_drop"),
    "save_int": ("tra", "save_int"),
    "dot_scroll_whitelist": ("data", "dot_scroll_whitelist"),
    "channels_mult": ("model", "channels_mult"),
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
        print(f"[archs13] init_weights '{iw}' not found -- {tid} trains from scratch")
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
    os.makedirs("models/archs13", exist_ok=True)
    setattr(c, "save_final", f"models/archs13/{tid}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"\n{'=' * 70}\n[archs13] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  train_scrolls={len(scroll_ids)}"
    )
    print(f"  scroll_ids={scroll_ids}")
    print(
        f"  n_epochs={c.tra.n_epochs}  probe_int={c.tra.probe_int}  eval_int={c.tra.eval_int}"
        f"  test_int={c.tra.test_int}  aug_start_epoch={getattr(c.tra, 'aug_start_epoch', 5)}"
    )
    print(
        f"  fast_eval_figure={c.tra.fast_eval_figure}  eval_int_scrolls={c.tra.eval_int_scrolls}"
        f"  eval_bs={c.data.eval_infer_bs}"
    )
    print(f"  eval_scroll_ids={getattr(c.data, 'vis_scroll_ids', None)}")
    print(
        f"  loss={c.tra.loss_type}  gce_q={c.tra.gce_q}  supcon={c.tra.supcon}"
        f"  learned_surface={c.model.learned_surface}  inklabel_dir={c.data.inklabel_dir}"
        f"  dot_dir={getattr(c.data, 'dot_inklabel_dir', '')}"
    )
    print(
        f"  attn_mil={c.model.attn_mil}  tta_consistency={c.tra.tta_consistency}"
        f"  dann={c.tra.dann}  dann_lambda={c.tra.dann_lambda}"
        f"  spill_reduction={c.tra.spill_reduction}  spill_lambda={c.tra.spill_lambda}"
    )
    print(
        f"  use_ibn={getattr(c.model, 'use_ibn', False)}"
        f"  fda_prob={getattr(c.dl, 'fda_prob', 0)}  elastic_prob={getattr(c.dl, 'elastic_prob', 0)}"
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
    ap = argparse.ArgumentParser(description="campaign_archs_13: 2-scroll combo baseline sweep")
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

    print(f"[archs13] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs13] 2-scroll [{', '.join(str(s.scroll_id) for s in TWO_SCROLLS)}]")
    print(f"[archs13] combo baseline: fda + elastic + eroded2 + dots + dann + oldspill + PHOTOM + IBN")
    print(f"[archs13] aug_start_epoch=0 (augmentation from epoch 1)")

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

    print(f"\n{'=' * 70}\n[archs13] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
