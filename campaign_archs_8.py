"""campaign_archs_8.py -- composed late-feature architecture sweep on p500 + w044 (2026-08-12).

GOAL: combine the ideas from campaign 7 that looked genuinely compatible:
- nnU-Net-style dense spatial processing is useful, but needs a binary late head
- structured attention over preserved depth helps more than exotic global attention
- top-down cross-window fusion helps when applied to a stable binary head
- explicitly probe the strongest untested follow-ups: nnunet input channels, nnunet attention-mil,
    bigger ds2 context for lcndz, a longer p500 lcndz retry, and a less aggressive center crop
    on the baseline family

ORGANIZATION (19 TESTS):
1:     PHerc0500P2 nnU-Net 3D overfit baseline (known hard case)
2-5:   w044 nnU-Net probes (raw ds2, attn-mil, raw+lcn+dz, raw+lcn+dz+attn-mil)
6-7:   w044 nnU-Net lcndz larger-context probes (96x96 ds2, 128x128 ds2)
8:     PHerc0500P2 nnU-Net lcndz baseline-context retry (48x48 ds2, 15 epochs, full eval figure)
9:     w044 relaxed-crop baseline
10-11: w044 anchors (latecollapse32, late_unet)
12-15: w044 latecollapse32 + {nonlocal, coord, depthse, fpn}
16-19: w044 late_unet + {nonlocal, coord, depthse, fpn}

BASELINE FOR COMPARISON:
- cmp_archs7_w044_noaug_w044_no_augs_11_21-34-49 from archs7

EXPECTED WINNERS:
- late binary heads should keep the nnU-Net-style spatial richness without the threshold-collapse
  seen in raw nnunet3d
- nonlocal / coord / depth-se / fpn were the most compatible structured refinements from archs7
- raw+lcn+dz should test whether the baseline's useful physics channels transfer into the dense model
- nnunet attention-mil should test whether the main nnunet failure is bagging/calibration rather than capacity
- bigger lcndz context may help now that the dense model has shown it can exploit preserved spatial detail
- relaxed center cropping should test whether the baseline family is discarding too much useful spatial support

CONFIGURATION:
- mostly 12 epochs, eval at epoch 12 only
- one PHerc0500P2 lcndz retry runs 15 epochs with eval at epoch 15 and fast_eval disabled
- no augmentation for clean overfit/isolation behavior
- batch=32, eval_bs=64, workers=4

  python campaign_archs_8.py --dry-run
  python campaign_archs_8.py --only p500_nnunet3d
    python campaign_archs_8.py --only w044_nnunet3d_attn,w044_nnunet3d_lcndz_attn
    python campaign_archs_8.py --only p500_nnunet3d_lcndz_15ep
  python campaign_archs_8.py
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

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs8"

P500_SCROLL = ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75)
W044_SCROLL = ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "v16_arch_ctx"

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
    c.tra.n_epochs = 12
    c.tra.eval_int = 12
    c.tra.test_int = 999
    c.tra.probe_int = 999
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
    c.tra.ranking_lambda = 0.5
    c.tra.ranking_neg_frac = 1.0
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
    c.tra.dann_n_domains = 1
    return c


_BASE8 = dict(
    init_weights=MAE_CKPT,
    scrolls=[W044_SCROLL],
    dann=False,
    supcon=True,
    supcon_temp=0.07,
    supcon_curriculum=True,
    supcon_lambda_start=0.05,
    supcon_lambda_end=0.5,
    supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,
    depth_supcon=True,
    depth_supcon_lambda=0.3,
    mean_teacher=False,
    test_consistency=False,
)


def _mk8(tid, tag, **overrides):
    d = dict(_BASE8)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


TESTS = [
    _mk8(
        "p500_nnunet3d",
        "p500_nnunet3d_overfit_baseline",
        arch="nnunet3d",
        init_weights=None,
        attn_mil=False,
        scrolls=[P500_SCROLL],
    ),
    _mk8(
        "w044_nnunet3d",
        "w044_nnunet3d_ds2_raw",
        arch="nnunet3d",
        init_weights=None,
        attn_mil=False,
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "w044_nnunet3d_attn",
        "w044_nnunet3d_ds2_attn_mil",
        arch="nnunet3d",
        init_weights=None,
        attn_mil=True,
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "w044_nnunet3d_lcndz",
        "w044_nnunet3d_ds2_lcndz",
        arch="nnunet3d_lcndz",
        init_weights=None,
        attn_mil=False,
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "w044_nnunet3d_lcndz_attn",
        "w044_nnunet3d_ds2_lcndz_attn_mil",
        arch="nnunet3d_lcndz",
        init_weights=None,
        attn_mil=True,
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "p500_nnunet3d_lcndz_15ep",
        "p500_nnunet3d_ds2_lcndz_15ep",
        arch="nnunet3d_lcndz",
        init_weights=None,
        attn_mil=False,
        scrolls=[P500_SCROLL],
        context_size=48,
        context_downsample=2,
        n_epochs=15,
        eval_int=15,
        fast_eval_figure=False,
    ),
    _mk8(
        "w044_nnunet3d_lcndz_ctx96",
        "w044_nnunet3d_ds2_lcndz_ctx96",
        arch="nnunet3d_lcndz",
        init_weights=None,
        attn_mil=False,
        scrolls=[W044_SCROLL],
        context_size=96,
        context_downsample=2,
    ),
    _mk8(
        "w044_nnunet3d_lcndz_ctx96_attn_mil",
        "w044_nnunet3d_ds2_lcndz_ctx96_attn_mil",
        arch="nnunet3d_lcndz",
        init_weights=None,
        attn_mil=True,
        scrolls=[W044_SCROLL],
        context_size=96,
        context_downsample=2,
    ),
    _mk8(
        "w044_relaxedcrop",
        "w044_baseline_relaxed_crop",
        arch="v16_arch_ctx_relaxedcrop",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "latecollapse32_base",
        "latecollapse32_anchor",
        arch="v16_latecollapse32",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "late_unet_base",
        "late_unet_anchor",
        arch="v16_late_unet",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lc32_nonlocal",
        "latecollapse32_nonlocal",
        arch="v16_latecollapse32_nonlocal",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lc32_coord",
        "latecollapse32_coord",
        arch="v16_latecollapse32_coord",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lc32_depthse",
        "latecollapse32_depthse",
        arch="v16_latecollapse32_depthse",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lc32_fpn",
        "latecollapse32_fpn",
        arch="v16_latecollapse32_fpn",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lunet_nonlocal",
        "lateunet_nonlocal",
        arch="v16_late_unet_nonlocal",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lunet_coord",
        "lateunet_coord",
        arch="v16_late_unet_coord",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lunet_depthse",
        "lateunet_depthse",
        arch="v16_late_unet_depthse",
        scrolls=[W044_SCROLL],
    ),
    _mk8(
        "lunet_fpn",
        "lateunet_fpn",
        arch="v16_late_unet_fpn",
        scrolls=[W044_SCROLL],
    ),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "physics_stem": ("model", "physics_stem"),
    "physics_stem_depthmax": ("model", "physics_stem_depthmax"),
    "surface_stem": ("model", "surface_stem"),
    "surface_stem_withdog": ("model", "surface_stem_withdog"),
    "learned_surface": ("model", "learned_surface"),
    "n_depth_windows": ("model", "n_depth_windows"),
    "depth_attention_mode": ("model", "depth_attention_mode"),
    "normalization_layer": ("model", "normalization_layer"),
    "activation": ("model", "activation"),
    "conv1_drop": ("model", "conv1_drop"),
    "conv2_drop": ("model", "conv2_drop"),
    "head_drop": ("model", "head_drop"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "fast_eval_figure": ("tra", "fast_eval_figure"),
    "probe_int": ("tra", "probe_int"),
    "l1": ("tra", "l1_lambda"),
    "weight_decay": ("tra", "weight_decay"),
    "ranking_lambda": ("tra", "ranking_lambda"),
    "tv_lambda": ("tra", "tv_lambda"),
    "depth_supcon": ("tra", "depth_supcon"),
    "depth_supcon_lambda": ("tra", "depth_supcon_lambda"),
    "tta_consistency": ("tra", "tta_consistency"),
    "tta_consistency_lambda": ("tra", "tta_consistency_lambda"),
    "gce_q": ("tra", "gce_q"),
    "loss_type": ("tra", "loss_type"),
    "focal_gamma": ("tra", "focal_gamma"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
    "normalization_mode": ("data", "normalization_mode"),
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
    c = _base_config(f"cmp_archs8_{tid}_{tag}")
    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: {sec}.{attr} does not exist")
    if "scrolls" in t:
        c.data.scrolls = t["scrolls"]
    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs8] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([
        c.dl.flip_prob,
        c.dl.rotation_prob,
        c.dl.noise_prob,
        c.dl.brightness_prob,
        c.dl.contrast_prob,
        c.dl.cutout_prob,
        c.dl.depth_mask_prob,
    ])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs8", exist_ok=True)
    setattr(c, "save_final", f"models/archs8/{tid}_{tag}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'=' * 70}\n[archs8] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  entropy={c.model.attn_entropy_weight}"
    )
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"  scrolls={scroll_ids}")
    print(f"  n_epochs={c.tra.n_epochs}  eval_int={c.tra.eval_int}  fast_eval_figure={c.tra.fast_eval_figure}")
    print(f"  depth_supcon={c.tra.depth_supcon}  depth_supcon_lam={c.tra.depth_supcon_lambda}")
    print(f"  gce_q={c.tra.gce_q}  n_depth_windows={c.model.n_depth_windows}")
    print(f"  attn_mil={c.model.attn_mil}")
    print(
        f"  aug: flip={c.dl.flip_prob} rot={c.dl.rotation_prob} noise={c.dl.noise_prob}"
        f" cutout={c.dl.cutout_prob}"
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


def main():
    ap = argparse.ArgumentParser(description="campaign_archs_8: composed late-feature sweep on p500 + w044")
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

    print(f"[archs8] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs8] Test 1 is the PHerc0500P2 nnunet3d overfit baseline; the rest stay on isolated w044")
    print("[archs8] New probes: nnunet attn-mil, nnunet raw+lcn+dz, lcndz with 96/128 ds2 context,")
    print("[archs8]             a 15-epoch PHerc0500P2 lcndz retry, and a relaxed-crop baseline variant")
    print("[archs8] Existing anchors still cover the skip-connected late-fusion idea: latecollapse32 and late_unet")
    print("[archs8] Structured combo tests remain: +nonlocal, +coord, +depthse, +fpn on latecollapse32 and late_unet")
    print("[archs8] Config: mostly 12ep no-aug fast_eval runs, plus one 15ep PHerc0500P2 full-eval lcndz retry")

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

    print(f"\n{'=' * 70}\n[archs8] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()