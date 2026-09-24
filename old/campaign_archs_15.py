"""campaign_archs_15.py -- center-size / sub-tile-density A/B on w013 only (runpod, 2026-09-01).

SEQUENCING: choose the ARCHITECTURE (head geometry) first, THEN tune DANN separately.
  the first (desktop, 18-scroll) attempt of this campaign was catastrophic -- it conflated
  three changes at once (1->18 scrolls, dann no-op -> active-but-unlearnable at bs=4, and
  bs 48->4). the DANN adversary in particular was never actually exercised in campaign-14
  (dann_n_domains=1 => loss=-log(1)=0), so 18-way DANN was untested territory. this rerun
  strips all that back to the CLEAN c14 operating point so center-size is the ONLY variable.

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

BASELINE: campaign-14 multi16_pos_tinydrop -- REPRODUCED, not modified:
  archs-13 combo (_BASE13 + _ALL_KW) + multitile(subtile=8, grid=4, step=16, pos_only=True)
  + tinydrop (conv1=.05, conv2=.05, head=.10, skip=.20) + inklabel_dir=eroded_inklabels.
  w013 ONLY (scroll 20240304141531), dann_n_domains=1 (DANN is a no-op, exactly as in c14).
  attn_mil=True + attn_entropy_weight=0.03 (campaign-14 attn winner beats LSE).
  all archs-13 ingredients held fixed: fda, elastic, IBN, spill, supcon, tta=light.
  the c32_t8 arm SHOULD reproduce the c14 best performer; if it doesn't, something drifted.

HARDWARE (runpod, rtx 5090 / a4500):
  exact c14 operating point where the whole recipe was tuned -- do NOT sqrt-scale.
  batch=48, lr=1.5e-4, workers=8, eval_bs=128, eval_prefetch=3 (SSD).

TESTS (8 total, 6 user-specified + 2 additional):
  c32_t8    32px center, 8px tile, 4x4=16 targets  -- CONTROL (reproduces campaign-14 best)
  c16_t8    16px center, 8px tile, 2x2=4  targets  -- sub-stroke, very few targets
  c16_t4    16px center, 4px tile, 4x4=16 targets  -- fine tile, small center
  c32_t4    32px center, 4px tile, 8x8=64 targets  -- fine tile (near noise floor)
  c32_t16   32px center, 16px tile, 2x2=4 targets  -- coarse tile, same center as control
  c48_t8    48px center, 8px tile, 6x6=36 targets  -- half-stroke [ADDED]
  c64_t8    64px center, 8px tile, 8x8=64 targets  -- full-stroke width
  c64_t16   64px center, 16px tile, 4x4=16 targets -- full-stroke, coarse tile

  WHY c32_t8: it is the c14 best-performer config. reproducing it on w013 at the c14
  operating point is the anchor that makes every other arm interpretable.

  WHY c48_t8: 48px ~= half a 100px ink stroke. a window centered on a stroke edge sees
  ~24px of clean ink and ~24px of clean non-ink, giving the sharpest class boundary of
  any configuration. bridges the 32->64px gap and tests whether a sweet spot exists there.
  6x6=36 targets is between the 16 (c32_t8) and 64 (c64_t8) extremes.

  NOT TESTED (kept for focus on center-size axis, one variable at a time):
    pos_only ablation (pos_only=False): separately testable but kept fixed here
    multi-scroll + DANN: deferred to a follow-up campaign on the WINNING geometry

  STEP POLICY: step = 16 FIXED for all arms (matches c14's multi16 operating point).
  this gives every center size EQUAL (maximal) gradient updates, so geometry is the only
  variable. equal-coverage (step=center) was tried in the failed run and starved large
  centers of gradient steps (c64 got ~5x fewer windows than c16) -- avoided here.

CONFIG:
  15 epochs, eval at epoch 15, probe_int=999 (never), test_int=999 (off).
  1 eval figure (w013). fast_eval_figure=False (full scroll figure).

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

# ARCHITECTURE-SELECTION campaign: isolate center-size/tile-density as the ONLY variable.
# run on w013 ONLY (like campaign-14) so results are directly comparable to the c14 baseline,
# and so scroll-count / multi-domain-DANN are NOT confounded into the center-size question.
# multi-scroll + DANN tuning is a SEPARATE follow-up on the winning geometry.
_W013_ID = 20240304141531
ONE_SCROLL = [s for s in ALL_SCROLLS if s.scroll_id == _W013_ID]

# runpod hardware (5090 / a4500): exact c14 operating point where the whole recipe was tuned.
# bs=48, lr=1.5e-4, workers=8, eval_bs=128. do NOT sqrt-scale -- this IS the reference point.
_BATCH = 32
_LR = 1e-4
_EVAL_BS = 64
_WORKERS = 8

_EVAL_VIS_IDS = [_W013_ID]


def _base_config(exp_name: str) -> Config:
    c = Config()
    c.exp_name = exp_name
    c.model.arch = "nnunet3d_lcndz"

    c.data.zarr_path = get_zarr_dir()
    c.data.scrolls = list(ONE_SCROLL)

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
    c.data.eval_prefetch = 3        # runpod SSD: overlap zarr reads with gpu inference
    c.data.tta_mode = "light"
    c.tra.eval_int_scrolls = 1      # single eval figure (w013)
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


# campaign-15 baseline: archs-13 combo + multi16_pos_tinydrop settings, reduced to the c14
# operating point (w013-only, dann no-op, bs=48/lr=1.5e-4).
#
# construction order:
#   _BASE13   -> init_weights(MAE96), attn_mil=True, dann, spill, fda, elastic, IBN, supcon,
#                inklabel_dir=eroded2_inklabels
#   _ALL_KW   -> 18 scrolls, dann_n_domains=18, heavy dropouts, depth_jitter=4,
#                spill_min_depth_var=0.8, spill_lambda=0.5
#   .update() -> override dropouts with tinydrop; add multitile (fixed step=16); reduce to
#                w013-only + dann_n_domains=1 (no-op); apply runpod hyperparams; fix eval;
#                switch inklabel_dir to eroded_inklabels
_BASE15 = dict(_BASE13)
_BASE15.update(_ALL_KW)
_BASE15.update(
    # MULTITILE: subtile/grid overridden per test; step FIXED at 16 for all arms so every
    # center size gets equal (maximal) gradient updates -- geometry is the only variable.
    # (equal-coverage step=center would starve large centers of updates; see campaign notes.)
    # campaign-14 finding: attn_mil=True + entropy reg beats LSE aggregator for multitile.
    multitile=True,
    attn_mil=True,              # gated attention-MIL (beats LSE per campaign-14 results)
    attn_entropy_weight=0.03,
    multitile_train_step=16,    # FIXED across all arms (matches c14 multi16 operating point)
    multitile_pos_only=True,    # in ink windows, supervise only ink sub-tiles (not boundary-adjacent non-ink)

    # TINYDROP: override the heavy _ALL_KW dropouts (.25/.25/.4/.4) with the lighter recipe
    conv1_drop=0.05,
    conv2_drop=0.05,
    head_drop=0.1,
    skip_drop=0.2,

    # RUNPOD 5090/A4500: exact c14 operating point (bs=48, lr=1.5e-4, workers=8, eval_bs=128)
    batch_size=_BATCH,
    lr=_LR,
    num_workers=_WORKERS,
    eval_infer_bs=_EVAL_BS,
    eval_prefetch=3,

    # EVAL SCHEDULE: fire once at epoch 15, no probe, no test
    n_epochs=15,
    eval_int=15,
    probe_int=999,
    save_int=15,
    eval_int_scrolls=1,
    vis_scroll_ids=list(_EVAL_VIS_IDS),

    # W013 ONLY: single scroll, like c14, so center-size is the only variable.
    # dann_n_domains=1 => DANN loss is -log(1)=0 (a no-op), matching the c14 baseline.
    # multi-scroll + real DANN is a SEPARATE follow-up once the geometry is chosen.
    scrolls=list(ONE_SCROLL),
    dann_n_domains=1,

    # dot labels: only w013 (the single training/eval scroll here)
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


# all tests inherit: pos_only=True, step=16 FIXED (equal gradient updates), tinydrop,
# w013-only, dann no-op, c14 hyperparams. only multitile_subtile / multitile_grid vary.
# center_px = subtile * grid; targets_per_window = grid^2.
TESTS = [
    # from 14: attn and pos are best

    # CONTROL: exact campaign-14 multi16_pos_tinydrop_attn config (w013-only, c14 hyperparams).
    # this SHOULD reproduce the c14 best performer -- if it doesn't, something else drifted.
    _mk15("c32_t8",  "15_c32_t8",  multitile_subtile=8,  multitile_grid=4),  # center=32px 4x4=16 targets

    # 16px center: sub-stroke coverage; a window centered on ink usually sees homogeneous
    # signal (all-ink or all-papyrus). tests whether small centers give enough gradient.
    _mk15("c16_t8",  "15_c16_t8",  multitile_subtile=8,  multitile_grid=2),  # center=16px 2x2=4  targets
    _mk15("c16_t4",  "15_c16_t4",  multitile_subtile=4,  multitile_grid=4),  # center=16px 4x4=16 targets (finer tile)

    _mk15("c16_t4_b32",  "15_c16_t4_b32",  multitile_subtile=4,  multitile_grid=4),  # center=16px 4x4=16 targets (finer tile)

    # 32px center with tile sizes above and below the control (8px):
    # c32_t4: 64 targets/window, 4px sub-tiles are ~1 fiber width -- near the label noise floor
    _mk15("c32_t4",  "15_c32_t4",  multitile_subtile=4,  multitile_grid=8),  # center=32px 8x8=64 targets
    # c32_t16: 4 targets/window, same gradient count as c16_t8 but each target covers 16px
    _mk15("c32_t16", "15_c32_t16", multitile_subtile=16, multitile_grid=2),  # center=32px 2x2=4  targets

    # ADDED: 48px center, 8px tile. ~half a 100px ink stroke per window when centered on
    # a stroke edge: ~24px of clean ink + ~24px of clean non-ink -> sharpest class boundary.
    # 6x6=36 targets bridges the 16-target (c32_t8) and 64-target (c64_t8) density extremes.
    _mk15("c48_t8",  "15_c48_t8",  multitile_subtile=8,  multitile_grid=6),  # center=48px 6x6=36 targets

    # 64px center: spans ~2/3 of an ink stroke width; many targets but each may straddle boundary
    _mk15("c64_t8",  "15_c64_t8",  multitile_subtile=8,  multitile_grid=8),  # center=64px 8x8=64 targets
    _mk15("c64_t16", "15_c64_t16", multitile_subtile=16, multitile_grid=4),  # center=64px 4x4=16 targets
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
    ap = argparse.ArgumentParser(description="campaign_archs_15: center-size A/B on w013 (runpod)")
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
    print(f"[archs15] w013-only center-size sweep  batch={_BATCH}  lr={_LR:.2e}  eval_bs={_EVAL_BS}")

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
