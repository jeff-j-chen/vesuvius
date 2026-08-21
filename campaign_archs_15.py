"""campaign_archs_15.py -- center-size / sub-tile-density A/B on all 18 scrolls (2026-08-22).

QUESTION: what center size and sub-tile density maximize useful gradient?
  the current head (campaign-14 multi16_pos_tinydrop) uses a 32px center split into a 4x4
  grid of 8px sub-tiles (16 targets/window). we sweep center size (16/32/48/64px) and
  sub-tile size (4/8/16px) to find the gradient-useful density range.

  too fine (4px tiles) -> near the inklabel noise floor (4px ~1 fiber width at 4um scan);
  too many targets per window may hurt when near-boundary labels are uncertain;
  too small a center (16px) -> gradient count approaches single-tile;
  too large a center (64px) -> many sub-tiles straddle the ink/non-ink boundary.
  ink strokes ~100px wide, so:
    16px center = sub-stroke (usually homogeneous: all-ink or all-papyrus)
    32px center = ~1/3 stroke width (sometimes mixed at edges)
    48px center = ~half stroke width (cleanest boundary: ~24px ink, ~24px non-ink)
    64px center = ~2/3 stroke width (wide coverage, some boundary contamination)

BASELINE: campaign-14 multi16_pos_tinydrop (restored to all 18 scrolls):
  archs-13 combo (_BASE13 + _ALL_KW) + multitile(subtile=8, grid=4, step=16, pos_only=True)
  + tinydrop (conv1=.05, conv2=.05, head=.10, skip=.20) + inklabel_dir=eroded_inklabels.
  dann_n_domains=18 (restored; campaign-14 used 1 for the w013-only run).
  attn_mil=False (LSE aggregator; same as campaign-14 multitile arms).
  all archs-13 ingredients held fixed: fda, elastic, IBN, spill, supcon, tta=light.
  note: supcon with bs=8 yields very few positive pairs per batch; it remains but
  contributes little at this batch size. kept for baseline parity, not effectiveness.

HARDWARE (linux-desktop, rtx 3060 mobile):
  6GB vram, 32GB ram. single-scroll campaign-14 used 27GB vram @ bs=48.
  batch=8, lr=6e-5 = 1.5e-4 * sqrt(8/48) (square-root batch scaling), workers=0
  (external seagate HDD; multi-worker contention is slower than single-threaded).
  eval_bs=32. eval_prefetch=0 (no disk prefetch on HDD).

TESTS (8 total, 6 user-specified + 2 additional):
  c32_t8    32px center, 8px tile, 4x4=16 targets  -- CONTROL (campaign-14 baseline)
  c16_t8    16px center, 8px tile, 2x2=4  targets  -- sub-stroke, very few targets
  c16_t4    16px center, 4px tile, 4x4=16 targets  -- fine tile, small center
  c32_t4    32px center, 4px tile, 8x8=64 targets  -- fine tile (near noise floor)
  c32_t16   32px center, 16px tile, 2x2=4 targets  -- coarse tile, same center as control
  c48_t8    48px center, 8px tile, 6x6=36 targets  -- half-stroke [ADDED]
  c64_t8    64px center, 8px tile, 8x8=64 targets  -- full-stroke width
  c64_t16   64px center, 16px tile, 4x4=16 targets -- full-stroke, coarse tile

  WHY c32_t8: running the campaign-14 config with 18 scrolls is a necessary control.
  1->18 scrolls is a large distribution shift; without it we cannot separate scroll-count
  effects from center-size effects when comparing against campaign-14 results.

  WHY c48_t8: 48px ~= half a 100px ink stroke. a window centered on a stroke edge sees
  ~24px of clean ink and ~24px of clean non-ink, giving the sharpest class boundary of
  any configuration. bridges the 32->64px gap and tests whether a sweet spot exists there.
  6x6=36 targets is between the 16 (c32_t8) and 64 (c64_t8) extremes.

  NOT TESTED (kept for focus on center-size axis, one variable at a time):
    multitile_train_step (step=8 vs 16): secondary question, separate campaign
    pos_only ablation (pos_only=False): separately testable but kept fixed here

CONFIG:
  15 epochs, eval at epoch 15, probe_int=999 (never), test_int=999 (off).
  4 eval figures: 20250628074500, 20240304141531, 20260226000000, 20260317000000.
  fast_eval_figure=False (full scroll figures).

  python campaign_archs_15.py --dry-run
  python campaign_archs_15.py --only c32_t8,c16_t8
  python campaign_archs_15.py
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
from utils.platform import get_zarr_dir
from campaign_archs_13 import _BASE13, _ALL_KW
from campaign_archs_14 import _OVERRIDES   # reuse the full override map; avoids drift

LOG_DIR = "./runs_archs15"
ALL_SCROLLS = list(DEFAULT_SCROLLS)    # all 18 fragments

# desktop hardware: 6GB vram, 32GB ram, external HDD.
# lr scaled from runpod (bs=48, lr=1.5e-4) using square-root rule: 1.5e-4 * sqrt(8/48) = 6.1e-5
_BATCH = 8
_LR = 6e-5
_EVAL_BS = 32
_WORKERS = 0    # external seagate HDD; single-threaded beats multi-worker contention

_EVAL_VIS_IDS = [20250628074500, 20240304141531, 20260226000000, 20260317000000]


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "nnunet3d_lcndz"

    c.data.zarr_path = get_zarr_dir()
    c.data.scrolls = list(ALL_SCROLLS)

    c.data.tile_size = 16
    c.data.depth = 24
    c.data.train_d_start = 4
    c.data.train_d_end = 28
    c.data.d_start = 4
    c.data.d_end = 28
    c.data.context_size = 192
    c.data.context_downsample = 2
    c.model.conv1_drop = 0.0
    c.model.conv2_drop = 0.0
    c.model.head_drop = 0.0
    c.tra.n_epochs = 15
    c.tra.eval_int = 15
    c.tra.test_int = 999
    c.tra.probe_int = 999
    c.tra.save_int = 15
    c.tra.log_dir = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = _LR
    c.tra.aug_start_epoch = 0
    c.data.eval_infer_bs = _EVAL_BS
    c.data.eval_prefetch = 0        # no prefetch on external HDD
    c.data.tta_mode = "light"
    c.tra.eval_int_scrolls = 4      # generate figures for only the 4 eval scrolls
    c.data.vis_scroll_ids = list(_EVAL_VIS_IDS)
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
    # auto pos_weight: multitile imbalance varies with grid size (4px tile ~16:1, 8px ~5:1)
    c.tra.loss_type = "bce"
    c.tra.tile_pos_weight_auto = True
    c.dl.batch_size = _BATCH
    c.dl.num_workers = _WORKERS
    c.dl.data_aug = True
    c.data.mask_memmap = True
    setattr(c.data, "mask_bitpack", True)
    c.data.ring_negatives = True
    c.data.ring_close_r = 3
    c.data.ring_gap_r = 3
    c.data.ring_shell_r = 2
    c.tra.fast_eval_figure = False   # full figures for all eval scrolls
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


# campaign-15 baseline: archs-13 combo restored to 18 scrolls + multi16_pos_tinydrop settings.
#
# construction order:
#   _BASE13   -> init_weights(MAE96), attn_mil=True, dann, spill, fda, elastic, IBN, supcon,
#                inklabel_dir=eroded2_inklabels
#   _ALL_KW   -> 18 scrolls, dann_n_domains=18, heavy dropouts, depth_jitter=4,
#                spill_min_depth_var=0.8, spill_lambda=0.5
#   .update() -> override dropouts with tinydrop; add multitile; apply desktop constraints;
#                fix eval schedule; restore dann_n_domains for 18 scrolls; switch inklabel_dir
_BASE15 = dict(_BASE13)
_BASE15.update(_ALL_KW)
_BASE15.update(
    # MULTITILE: subtile/grid overridden per test; these match the campaign-14 control
    multitile=True,
    attn_mil=False,             # LSE aggregator replaces bag-MIL for multitile arms
    multitile_train_step=16,    # window stride (px); fixed across tests to isolate center-size
    multitile_pos_only=True,    # in ink windows, supervise only ink sub-tiles (not boundary-adjacent non-ink)

    # TINYDROP: override the heavy _ALL_KW dropouts (.25/.25/.4/.4) with the lighter recipe
    conv1_drop=0.05,
    conv2_drop=0.05,
    head_drop=0.1,
    skip_drop=0.2,

    # DESKTOP CONSTRAINTS: 6GB vram, external HDD
    batch_size=_BATCH,
    lr=_LR,
    num_workers=_WORKERS,
    eval_infer_bs=_EVAL_BS,
    eval_prefetch=0,

    # EVAL SCHEDULE: fire once at epoch 15, no probe, no test
    n_epochs=15,
    eval_int=15,
    probe_int=999,
    save_int=15,
    eval_int_scrolls=4,
    vis_scroll_ids=list(_EVAL_VIS_IDS),

    # 18 SCROLLS: restore from campaign-14's w013-only run
    scrolls=list(ALL_SCROLLS),
    dann_n_domains=len(ALL_SCROLLS),

    # dot labels: only the 4 fragments whose dot maps are processed (same as _ALL_KW whitelist)
    dot_scroll_whitelist=list(_EVAL_VIS_IDS),

    # eroded_inklabels: carried from campaign-14 (differs from _BASE13's eroded2_inklabels)
    inklabel_dir="./eroded_inklabels",
)


def _mk15(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE15)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


# all tests inherit: pos_only=True, step=16px, tinydrop, 18 scrolls, desktop constraints.
# only multitile_subtile (sub-tile px) and multitile_grid (grid side length) vary.
# center_px = subtile * grid; targets_per_window = grid^2.
TESTS = [
    # CONTROL: exact campaign-14 multi16_pos_tinydrop config, now on 18 scrolls.
    # necessary reference: 1->18 scroll shift is large; without this we cannot separate
    # scroll-count effects from center-size effects when comparing to campaign-14.
    _mk15("c32_t8", "15_c32_t8", multitile_subtile=8, multitile_grid=4),    # center=32px 4x4=16 targets

    # 16px center: sub-stroke coverage; a window centered on ink usually sees homogeneous
    # signal (all-ink or all-papyrus). tests whether small centers give enough gradient.
    _mk15("c16_t8", "15_c16_t8", multitile_subtile=8, multitile_grid=2),    # center=16px 2x2=4  targets
    _mk15("c16_t4", "15_c16_t4", multitile_subtile=4, multitile_grid=4),    # center=16px 4x4=16 targets (finer tile)

    # 32px center with tile sizes above and below the control (8px):
    # c32_t4: 64 targets/window, 4px sub-tiles are ~1 fiber width -- near the label noise floor
    _mk15("c32_t4", "15_c32_t4", multitile_subtile=4, multitile_grid=8),    # center=32px 8x8=64 targets
    # c32_t16: 4 targets/window, same gradient count as c16_t8 but each target covers 16px
    _mk15("c32_t16", "15_c32_t16", multitile_subtile=16, multitile_grid=2), # center=32px 2x2=4  targets

    # ADDED: 48px center, 8px tile. ~half a 100px ink stroke per window when centered on
    # a stroke edge: ~24px of clean ink + ~24px of clean non-ink -> sharpest class boundary.
    # 6x6=36 targets bridges the 16-target (c32_t8) and 64-target (c64_t8) density extremes.
    _mk15("c48_t8", "15_c48_t8", multitile_subtile=8, multitile_grid=6),    # center=48px 6x6=36 targets

    # 64px center: spans ~2/3 of an ink stroke width; many targets but each may straddle boundary
    _mk15("c64_t8", "15_c64_t8", multitile_subtile=8, multitile_grid=8),    # center=64px 8x8=64 targets
    _mk15("c64_t16", "15_c64_t16", multitile_subtile=16, multitile_grid=4), # center=64px 4x4=16 targets
]


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
        print(f"[archs15] init_weights '{iw}' not found -- {tid} trains from scratch")
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
    os.makedirs("models/archs15", exist_ok=True)
    setattr(c, "save_final", f"models/archs15/{tid}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    n_sub = int(getattr(c.model, "multitile_subtile", 8))
    n_grid = int(getattr(c.model, "multitile_grid", 4))
    center_px = n_sub * n_grid
    n_targets = n_grid * n_grid
    print(f"\n{'=' * 70}\n[archs15] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  train_scrolls={len(scroll_ids)}"
    )
    print(
        f"  center={center_px}px  subtile={n_sub}px  grid={n_grid}x{n_grid}={n_targets} targets/window"
        f"  step={getattr(c.data, 'multitile_train_step', 0)}px"
        f"  pos_only={getattr(c.data, 'multitile_pos_only', False)}"
    )
    print(
        f"  n_epochs={c.tra.n_epochs}  eval_int={c.tra.eval_int}  probe_int={c.tra.probe_int}"
        f"  fast_eval_figure={c.tra.fast_eval_figure}"
    )
    print(
        f"  batch={c.dl.batch_size}  lr={c.tra.lr:.2e}  eval_bs={c.data.eval_infer_bs}"
        f"  workers={c.dl.num_workers}  eval_prefetch={getattr(c.data, 'eval_prefetch', 0)}"
        f"  tta_mode={getattr(c.data, 'tta_mode', 'flips')}"
    )
    print(
        f"  dann_n_domains={getattr(c.tra, 'dann_n_domains', 0)}"
        f"  vis_ids={getattr(c.data, 'vis_scroll_ids', [])}"
    )
    print(
        f"  loss={c.tra.loss_type}  supcon={c.tra.supcon}  spill={c.tra.spill_reduction}"
        f"  inklabel_dir={c.data.inklabel_dir}"
    )
    print(
        f"  drop: conv1={c.model.conv1_drop} conv2={c.model.conv2_drop}"
        f" head={c.model.head_drop} skip={getattr(c.model, 'skip_drop', 0.0)}"
    )
    print(
        f"  aug: fda={getattr(c.dl, 'fda_prob', 0)} elastic={getattr(c.dl, 'elastic_prob', 0)}"
        f" flip={c.dl.flip_prob} rot={c.dl.rotation_prob}"
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
    ap = argparse.ArgumentParser(description="campaign_archs_15: center-size A/B on 18 scrolls (desktop)")
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

    print(f"[archs15] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs15] 18-scroll center-size sweep  batch={_BATCH}  lr={_LR:.2e}  eval_bs={_EVAL_BS}")

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

    print(f"\n{'=' * 70}\n[archs15] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        t = next(x for x in TESTS if str(x["tid"]) == tid)
        n_sub = int(t.get("multitile_subtile", 8))
        n_grid = int(t.get("multitile_grid", 4))
        tag = str(t["tag"])
        print(f"  {tid} ({tag})  center={n_sub * n_grid}px tile={n_sub}px grid={n_grid}x{n_grid}: {status}")


if __name__ == "__main__":
    main()
