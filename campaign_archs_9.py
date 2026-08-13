"""campaign_archs_9.py -- all-scroll nnunet integration sweep (2026-08-13).

GOAL: keep the current 3-test sparse-label baseline set intact and add only further
integration tests for the remaining baseline-family ideas the current nnUNet path can
now consume directly: spatial SupCon and learned surface attention.

ALL TESTS USE MAE WARM-START via models/mae_nnunet_ctx48.pth (ctx=48 tests) and
models/mae_nnunet_ctx96.pth (ctx=96 tests). Run before campaign:
    python mae_pretrain_nnunet.py --steps 6000 --ctx 48 --ds 2 -n mae_nnunet_ctx48
    python mae_pretrain_nnunet.py --steps 6000 --ctx 96 --ds 2 -n mae_nnunet_ctx96

ORGANIZATION (9 TESTS):
1:     nnunet3d_ds2_lcndz_softaug_tta_attn_mae baseline (ctx=48)
2:     baseline + GCE q=0.9 + asymmetric label smoothing
3:     baseline + spatial SupCon
4:     baseline + GCE/asym + spatial SupCon
5:     ctx=96 + GCE/asym + MAE
6:     ctx=96 + GCE/asym + learned surface + MAE
7:     ctx=96 + GCE/asym + spatial SupCon + MAE
8:     ctx=96 + GCE/asym + spatial SupCon + learned surface + MAE
9:     ctx=96 + GCE/asym + spatial SupCon + learned surface + MAE (same as 8, duplicate run)

BASELINE FOR COMPARISON:
- cmp_archs8_w044_nnunet3d_lcndz_w044_nnunet3d_ds2_lcndz_12_17-14-53 from archs8

CONFIGURATION:
- 15 epochs, eval at epoch 15 only
- probes every 5 epochs
- fast_eval_figure=False, eval_int_scrolls=1, test_int=999
- batch=32, eval_bs=64, workers=4
- trains on all 17 scrolls from utils.config DEFAULT_SCROLLS
- attn entropy weight stays at 0.03 for all attn-MIL runs

    python campaign_archs_9.py --dry-run
    python campaign_archs_9.py --only lcndz_softaug_tta_attn_mae
    python campaign_archs_9.py
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

from utils.config import Config
from utils.platform import get_default_lr, get_zarr_dir

LOG_DIR = "./runs_archs9"

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

    c.data.tile_size = 16
    c.data.depth = 24
    c.data.train_d_start = 4
    c.data.train_d_end = 28
    c.data.d_start = 4
    c.data.d_end = 28
    c.data.context_size = 48
    c.data.context_downsample = 2
    c.model.conv1_drop = 0.15
    c.model.conv2_drop = 0.15
    c.model.head_drop = 0.4
    c.tra.n_epochs = 15
    c.tra.eval_int = 15
    c.tra.test_int = 999
    c.tra.probe_int = 5
    c.tra.save_int = 5
    c.tra.log_dir = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = 64
    c.tra.eval_int_scrolls = 1
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
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


_BASE9 = dict(
    arch="nnunet3d_lcndz",
    init_weights=None,
    supcon=False,
    supcon_temp=0.07,
    supcon_curriculum=False,
    supcon_lambda_start=0.05,
    supcon_lambda_end=0.5,
    supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,
)


def _mk9(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE9)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


_MAE_CTX48 = "models/mae_nnunet_48.pth"
_MAE_CTX96 = "models/mae_nnunet_96.pth"

TESTS = [
    # ctx=48 baseline family (MAE warm-start)
    # _mk9(
    #     "lcndz_softaug_tta_attn_mae",
    #     "nnunet3d_ds2_lcndz_softaug_tta_attn_mae",
    #     arch="nnunet3d_lcndz",
    #     init_weights=_MAE_CTX48,
    #     **SOFT_AUGS,
    #     **TTA_FLIPS,
    # ),
    _mk9(
        "lcndz_softaug_tta_attn_gceasym_mae",
        "nnunet3d_ds2_lcndz_softaug_tta_attn_gceasym_mae",
        arch="nnunet3d_lcndz",
        init_weights=_MAE_CTX48,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
    ),
    _mk9(
        "lcndz_softaug_tta_attn_supcon_mae",
        "nnunet3d_ds2_lcndz_softaug_tta_attn_supcon_mae",
        arch="nnunet3d_lcndz",
        init_weights=_MAE_CTX48,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **SPATIAL_SUPCON,
    ),
    _mk9(
        "lcndz_softaug_tta_attn_gceasym_supcon_mae",
        "nnunet3d_ds2_lcndz_softaug_tta_attn_gceasym_supcon_mae",
        arch="nnunet3d_lcndz",
        init_weights=_MAE_CTX48,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
        **SPATIAL_SUPCON,
    ),
    # ctx=96 family (MAE warm-start)
    _mk9(
        "lcndz_ctx96_softaug_tta_attn_gceasym_mae",
        "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_gceasym_mae",
        arch="nnunet3d_lcndz",
        context_size=96,
        context_downsample=2,
        init_weights=_MAE_CTX96,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
    ),
    _mk9(
        "lcndz_ctx96_softaug_tta_attn_gceasym_learnsurf_mae",
        "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_gceasym_learnsurf_mae",
        arch="nnunet3d_lcndz",
        context_size=96,
        context_downsample=2,
        learned_surface=True,
        init_weights=_MAE_CTX96,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
    ),
    _mk9(
        "lcndz_ctx96_softaug_tta_attn_gceasym_supcon_mae",
        "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_gceasym_supcon_mae",
        arch="nnunet3d_lcndz",
        context_size=96,
        context_downsample=2,
        init_weights=_MAE_CTX96,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
        **SPATIAL_SUPCON,
    ),
    _mk9(
        "lcndz_ctx96_softaug_tta_attn_gceasym_supcon_learnsurf_mae",
        "nnunet3d_ds2_lcndz_ctx96_softaug_tta_attn_gceasym_supcon_learnsurf_mae",
        arch="nnunet3d_lcndz",
        context_size=96,
        context_downsample=2,
        learned_surface=True,
        init_weights=_MAE_CTX96,
        **SOFT_AUGS,
        **TTA_FLIPS,
        **GCE_ASYM,
        **SPATIAL_SUPCON,
    ),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "learned_surface": ("model", "learned_surface"),
    "conv1_drop": ("model", "conv1_drop"),
    "conv2_drop": ("model", "conv2_drop"),
    "head_drop": ("model", "head_drop"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "probe_int": ("tra", "probe_int"),
    "l1": ("tra", "l1_lambda"),
    "weight_decay": ("tra", "weight_decay"),
    "tta_consistency": ("tra", "tta_consistency"),
    "tta_consistency_lambda": ("tra", "tta_consistency_lambda"),
    "tta_consistency_mode": ("tra", "tta_consistency_mode"),
    "gce_q": ("tra", "gce_q"),
    "loss_type": ("tra", "loss_type"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
    "context_size": ("data", "context_size"),
    "context_downsample": ("data", "context_downsample"),
    "ring_label_source": ("data", "ring_label_source"),
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
    "batch_size": ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = str(t["tid"])
    tag = str(t["tag"])
    c = _base_config(f"cmp_archs9_{tid}_{tag}")
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
        print(f"[archs9] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([
        c.dl.flip_prob,
        c.dl.rotation_prob,
        c.dl.noise_prob,
        c.dl.brightness_prob,
        c.dl.contrast_prob,
        c.dl.cutout_prob,
        c.dl.depth_mask_prob,
    ])
    os.makedirs("models/archs9", exist_ok=True)
    setattr(c, "save_final", f"models/archs9/{tid}_{tag}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'=' * 70}\n[archs9] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  entropy={c.model.attn_entropy_weight}"
    )
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"  train_scrolls={len(scroll_ids)}")
    print(f"  scroll_ids={scroll_ids}")
    print(
        f"  n_epochs={c.tra.n_epochs}  probe_int={c.tra.probe_int}  eval_int={c.tra.eval_int}"
        f"  test_int={c.tra.test_int}"
    )
    print(
        f"  fast_eval_figure={c.tra.fast_eval_figure}  eval_int_scrolls={c.tra.eval_int_scrolls}"
        f"  eval_bs={c.data.eval_infer_bs}"
    )
    print(
        f"  loss={c.tra.loss_type}  gce_q={c.tra.gce_q}"
        f"  ls_pos={c.tra.label_smooth_pos}  ls_neg={c.tra.label_smooth_neg}"
    )
    print(f"  supcon={c.tra.supcon}  learned_surface={c.model.learned_surface}")
    print(f"  attn_mil={c.model.attn_mil}  tta_consistency={c.tra.tta_consistency}")
    print(
        f"  tta_lambda={c.tra.tta_consistency_lambda}  tta_mode={c.tra.tta_consistency_mode}"
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
    ap = argparse.ArgumentParser(description="campaign_archs_9: all-scroll nnunet integration sweep")
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

    print(f"[archs9] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs9] Multi-scroll run: all 17 training scrolls from utils.config DEFAULT_SCROLLS")
    print("[archs9] All tests use MAE warm-start (ctx48 or ctx96 backbone)")
    print("[archs9] Attn entropy weight=0.03, probe every 5ep, no test figs mid-run")

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

    print(f"\n{'=' * 70}\n[archs9] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()