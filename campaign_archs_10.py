"""campaign_archs_10.py -- 4-scroll cross-domain generalization test (2026-08-13).

GOAL: establish a cross-domain baseline by training on 4 structurally disparate fragments
from 4 different scrolls (each at 9.362um isotropic), to test whether the model can learn
ink-agnostic features that transfer across scroll domains.

TRAINING SCROLLS (4 only):
  20260115000000  w044   PHerc0139  (original reference scroll)
  20260226000000  seg46527  PHerc0814  (different scroll, horizontal split)
  20250628074500  500P2_front  PHerc0500P2  (high-quality labels, vertical split)
  20240304141531  w013  PHerc1667 3.24um->9.362um  (new, 2.4um source isotropically rendered)

CONFIGURATION:
  15 epochs, eval_int=15 (fires at end only), probe_int=999 (off), test_int=999 (off)
  eval_int_scrolls=4 (all 4 fragments rendered), fast_eval_figure=False
  baseline arch: nnunet3d_ds2_lcndz_softaug_tta_attn_mae (same as archs9 test 1)
  MAE warm-start: models/mae_nnunet_48.pth

NOTE: assemble w013 first:
  python assemble_w013_1667.py --workers 32

    python campaign_archs_10.py --dry-run
    python campaign_archs_10.py
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

from utils.config import Config, ScrollConfig
from utils.platform import get_default_lr, get_zarr_dir

LOG_DIR = "./runs_archs10"
_MAE_CTX48 = "models/mae_nnunet_48.pth"

# 4-scroll cross-domain training set
CROSS_DOMAIN_SCROLLS = [
    ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044 PHerc0139
    ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),    # seg46527 PHerc0814
    ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75),    # 500P2_front PHerc0500P2
    ScrollConfig(20240304141531, split_axis="y", train_split_frac=0.75),    # w013 PHerc1667
]

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

TTA_FLIPS = dict(
    tta_consistency=True,
    tta_consistency_lambda=0.3,
    tta_consistency_mode="flips",
)


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "nnunet3d_lcndz"

    c.data.zarr_path = get_zarr_dir()
    c.data.scrolls = list(CROSS_DOMAIN_SCROLLS)

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
    c.tra.eval_int = 15      # only eval at final epoch
    c.tra.test_int = 999     # no test rendering
    c.tra.probe_int = 999    # no probe figures
    c.tra.save_int = 15
    c.tra.log_dir = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = 64
    c.tra.eval_int_scrolls = 4    # render all 4 fragments in eval figure
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
    c.tra.fast_eval_figure = False   # full eval figure (all tiles)
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


TESTS = [
    # {
    #     "tid": "cross4_lcndz_softaug_tta_attn_mae_3scroll",
    #     "tag": "4scroll_nnunet3d_ds2_lcndz_softaug_tta_attn_mae_3scroll",
    #     "scrolls": [
    #         ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044 PHerc0139
    #         ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),    # seg46527 PHerc0814
    #         ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75),    # 500P2_front PHerc0500P2
    #     ],
    #     "arch": "nnunet3d_lcndz",
    #     "init_weights": _MAE_CTX48,
    #     "attn_mil": True,
    #     "attn_entropy_weight": 0.03,
    #     "supcon": False,
    #     **SOFT_AUGS,
    #     **TTA_FLIPS,
    # },
    {
        "tid": "cross4_lcndz_softaug_tta_attn_mae_4scroll",
        "tag": "4scroll_nnunet3d_ds2_lcndz_softaug_tta_attn_mae_4scroll",
        "scrolls": [
            ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044 PHerc0139
            ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),    # seg46527 PHerc0814
            ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75),    # 500P2_front PHerc0500P2
            ScrollConfig(20240304141531, split_axis="y", train_split_frac=0.75),    # w013 PHerc1667
        ],
        "arch": "nnunet3d_lcndz",
        "init_weights": _MAE_CTX48,
        "attn_mil": True,
        "attn_entropy_weight": 0.03,
        "supcon": False,
        **SOFT_AUGS,
        **TTA_FLIPS,
    },
]

_OVERRIDES = {
    "arch":                    ("model", "arch"),
    "attn_mil":                ("model", "attn_mil"),
    "attn_entropy_weight":     ("model", "attn_entropy_weight"),
    "learned_surface":         ("model", "learned_surface"),
    "tta_consistency":         ("tra", "tta_consistency"),
    "tta_consistency_lambda":  ("tra", "tta_consistency_lambda"),
    "tta_consistency_mode":    ("tra", "tta_consistency_mode"),
    "supcon":                  ("tra", "supcon"),
    "flip_prob":               ("dl", "flip_prob"),
    "rotation_prob":           ("dl", "rotation_prob"),
    "noise_prob":              ("dl", "noise_prob"),
    "brightness_prob":         ("dl", "brightness_prob"),
    "contrast_prob":           ("dl", "contrast_prob"),
    "cutout_prob":             ("dl", "cutout_prob"),
    "cutout_max_frac":         ("dl", "cutout_max_frac"),
    "cutout_n_patches":        ("dl", "cutout_n_patches"),
    "depth_mask_prob":         ("dl", "depth_mask_prob"),
    "scrolls":                 ("data", "scrolls"),
}


def build_config(t: dict) -> Config:
    tid = str(t["tid"])
    tag = str(t["tag"])
    c = _base_config(f"cmp_archs10_{tid}_{tag}")
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
        print(f"[archs10] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([
        c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
        c.dl.brightness_prob, c.dl.contrast_prob,
        c.dl.cutout_prob, c.dl.depth_mask_prob,
    ])
    os.makedirs("models/archs10", exist_ok=True)
    setattr(c, "save_final", f"models/archs10/{tid}_{tag}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"\n{'=' * 70}\n[archs10] {c.exp_name}\n{'=' * 70}", flush=True)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size}  ds={c.data.context_downsample}")
    print(f"  train_scrolls={len(scroll_ids)}  scroll_ids={scroll_ids}")
    print(f"  n_epochs={c.tra.n_epochs}  eval_int={c.tra.eval_int}  eval_int_scrolls={c.tra.eval_int_scrolls}")
    print(f"  probe_int={c.tra.probe_int}  test_int={c.tra.test_int}  fast_eval={c.tra.fast_eval_figure}")
    print(f"  loss={c.tra.loss_type}  supcon={c.tra.supcon}  attn_mil={c.model.attn_mil}")
    print(f"  tta={c.tra.tta_consistency}  batch={c.dl.batch_size}")
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
    ap = argparse.ArgumentParser(description="campaign_archs_10: 4-scroll cross-domain baseline")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [t for t in TESTS if str(t["tid"]) in want]

    print(f"[archs10] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs10] training scrolls: {[s.scroll_id for s in CROSS_DOMAIN_SCROLLS]}")
    print("[archs10] 4-scroll cross-domain generalization test")

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

    print(f"\n{'=' * 70}\n[archs10] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
