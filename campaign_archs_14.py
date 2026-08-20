"""campaign_archs_14.py -- single-tile vs multitile head A/B on w013 (2026-08-19).

QUESTION: does a denser prediction target give more useful gradient?
  the current head aggregates the 16px center of the context into ONE score. multitile
  instead predicts the 32px center as a 4x4 grid of 8px sub-tiles (16 targets per window),
  each labeled papyrus (0) unless it contains .any() eroded ink (1). more gradient per
  forward without going fully dense (dense was the past killer).

TESTS (side-by-side, same backbone + warm-start, only the head/target differ):
  single   control: current attn_mil head, one 16px score per window
  multi16  multitile head, dataloader window stride 16px (16x labels/forward, same #windows)
  multi32  multitile head, window stride 32px (4x fewer windows -> ~1/4 epoch time)

SCROLL (1): 20240304141531  w013 (PHerc1667) -- train on w013 only, per request

BASELINE: shared by ALL arms == the archs-13 combo baseline (_BASE13 + _ALL_KW), imported
  directly. that includes fda + elastic + eroded2 + PHOTOM + IBN + dann + tta_consistency +
  spill + supcon + dropouts (conv .25/.25, head .4, skip .4) + depth_jitter=4. the ONLY
  intended deviations are: w013-only scroll, dann_n_domains=1 (so dann is a no-op), and the
  eval/epoch schedule (n_epochs=15, eval_int=15, probe_int=999).

CONFIG:
  15 epochs, eval at epoch 15, probe_int=999 (never), fast_eval_figure=False
  1 eval figure (w013), batch=32, lr=1e-4, eval_bs=96, workers=8

  python campaign_archs_14.py --dry-run
  python campaign_archs_14.py --only single,multi16
  python campaign_archs_14.py
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
# the campaign-14 baseline IS the archs-13 combo baseline (_BASE13 + _ALL_KW). import it
# directly so the two stay in lockstep instead of transcribing (and drifting from) the recipe.
from campaign_archs_13 import _BASE13, _ALL_KW

LOG_DIR = "./runs_archs14"
ALL_SCROLLS = list(DEFAULT_SCROLLS)
_MAE_CTX96 = "models/mae_nnunet_96.pth"

_W013_ID = 20240304141531
ONE_SCROLL = [s for s in ALL_SCROLLS if s.scroll_id == _W013_ID]

_BATCH = 32
_LR = 1e-4
_EVAL_BS = 96
_WORKERS = 8


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
    # visualizer speed wins for training-time eval figures: overlap zarr reads with gpu
    # inference (prefetch, byte-identical output) and use the light 2-view tta (id+hflip)
    c.data.eval_prefetch = 3
    c.data.tta_mode = "light"
    c.tra.eval_int_scrolls = 1
    c.data.vis_scroll_ids = [_W013_ID]
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda = 0.0
    # use BCE going forward; auto-derive pos_weight from data (per scroll+mode, cached) so both
    # single-tile (~1.9:1) and multitile (~5:1) arms get the right class weighting, no hardcode.
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
    # keep the full eval figure (NOT fast) so we can eyeball the denser multitile map
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


# the campaign-14 baseline == the archs-13 combo baseline (_BASE13 + _ALL_KW), used by ALL arms.
# only w013, the domain count (1 scroll), and the eval/epoch schedule deviate from it.
_BASE14 = dict(_BASE13)
_BASE14.update(_ALL_KW)
_BASE14.update(
    scrolls=list(ONE_SCROLL),
    dann_n_domains=1,               # w013 only -> 1 domain (dann is a no-op here, kept for parity)
    dot_scroll_whitelist=[_W013_ID],
    eval_int_scrolls=1,
    vis_scroll_ids=[_W013_ID],
    # the ONLY intended deviations from the archs-13 baseline:
    n_epochs=15,
    probe_int=999,
    eval_int=15,
    inklabel_dir="./eroded_inklabels",
)


def _mk14(tid: str, tag: str, **overrides: object) -> dict:
    d = dict(_BASE14)
    d.update(overrides)
    d["tid"] = tid
    d["tag"] = tag
    return d


# multitile arms disable attn_mil (the LSE aggregator replaces the bag scorer); every other
# ingredient is inherited from the shared baseline so only the head/target density differs.
# pos_weight is derived automatically from the data (tile_pos_weight_auto in _base_config) --
# multitile's ~5:1 sub-tile imbalance would otherwise collapse the head to all-negative.
_MT_KW = dict(
    multitile=True,
    multitile_subtile=8,
    multitile_grid=4,
    attn_mil=False,
)


TESTS = [
    _mk14("single_tinydrop", "14_single_tinydrop",
        conv1_drop=0.05,
        conv2_drop=0.05,
        head_drop=0.1,
        skip_drop=0.2,
    ),
    # pos-only labelling: in ink windows only the ink sub-tiles are supervised, non-ink
    # neighbours are masked out (negatives come from ink-free ring windows).
    _mk14("multi16_pos", "14_multi16_pos", multitile_train_step=16, multitile_pos_only=True, **_MT_KW),
    _mk14("multi16", "14_multi16", multitile_train_step=16, **_MT_KW),
    # multitile with per-sub-tile gated attention-MIL (instead of LSE pooling) + attn entropy;
    # step 16 matches multi16 so the ONLY change vs it is the sub-tile aggregator.
    # _mk14(
    #     "multi16_attn",
    #     "14_multi16_attn",
    #     multitile=True,
    #     multitile_subtile=8,
    #     multitile_grid=4,
    #     multitile_train_step=16,
    #     attn_mil=True,
    #     attn_entropy_weight=0.03,
    # ),
    # _mk14("multi32", "14_multi32", multitile_train_step=32, **_MT_KW),
]


_OVERRIDES = {
    "arch": ("model", "arch"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "learned_surface": ("model", "learned_surface"),
    # multitile head knobs
    "multitile": ("model", "multitile"),
    "multitile_subtile": ("model", "multitile_subtile"),
    "multitile_grid": ("model", "multitile_grid"),
    "multitile_train_step": ("data", "multitile_train_step"),
    "multitile_pos_only": ("data", "multitile_pos_only"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "probe_int": ("tra", "probe_int"),
    "aug_start_epoch": ("tra", "aug_start_epoch"),
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
    "spill_reduction": ("tra", "spill_reduction"),
    "spill_lambda": ("tra", "spill_lambda"),
    "spill_depth_threshold": ("tra", "spill_depth_threshold"),
    "spill_active_depth_tau": ("tra", "spill_active_depth_tau"),
    "spill_max_active_depth_frac": ("tra", "spill_max_active_depth_frac"),
    "spill_min_depth_var": ("tra", "spill_min_depth_var"),
    # domain-adversarial + tta-consistency + extra regularizers inherited from the archs-13 baseline
    "dann": ("tra", "dann"),
    "dann_lambda": ("tra", "dann_lambda"),
    "dann_grl_anneal": ("tra", "dann_grl_anneal"),
    "dann_n_domains": ("tra", "dann_n_domains"),
    "tta_consistency": ("tra", "tta_consistency"),
    "tta_consistency_lambda": ("tra", "tta_consistency_lambda"),
    "tta_consistency_mode": ("tra", "tta_consistency_mode"),
    "tta_consistency_prob": ("tra", "tta_consistency_prob"),
    "tile_pos_weight": ("tra", "tile_pos_weight"),
    "tile_pos_weight_auto": ("tra", "tile_pos_weight_auto"),
    "loss_type": ("tra", "loss_type"),
    "skip_drop": ("model", "skip_drop"),
    "depth_jitter": ("data", "depth_jitter"),
    "scrolls": ("data", "scrolls"),
    "eval_int_scrolls": ("tra", "eval_int_scrolls"),
    "batch_size": ("dl", "batch_size"),
    "eval_infer_bs": ("data", "eval_infer_bs"),
    "eval_prefetch": ("data", "eval_prefetch"),
    "tta_mode": ("data", "tta_mode"),
    "lr": ("tra", "lr"),
    "num_workers": ("dl", "num_workers"),
    "vis_scroll_ids": ("data", "vis_scroll_ids"),
    "inklabel_dir": ("data", "inklabel_dir"),
    "dot_inklabel_dir": ("data", "dot_inklabel_dir"),
    "dot_scroll_whitelist": ("data", "dot_scroll_whitelist"),
    "fda_prob": ("dl", "fda_prob"),
    "fda_beta": ("dl", "fda_beta"),
    "use_ibn": ("model", "use_ibn"),
    "elastic_prob": ("dl", "elastic_prob"),
    "elastic_alpha": ("dl", "elastic_alpha"),
    "elastic_sigma": ("dl", "elastic_sigma"),
    "conv1_drop": ("model", "conv1_drop"),
    "conv2_drop": ("model", "conv2_drop"),
    "head_drop": ("model", "head_drop"),
    "save_int": ("tra", "save_int"),
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
        print(f"[archs14] init_weights '{iw}' not found -- {tid} trains from scratch")
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
    os.makedirs("models/archs14", exist_ok=True)
    setattr(c, "save_final", f"models/archs14/{tid}_final.pth")
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    scroll_ids = [getattr(s, "scroll_id", None) for s in c.data.scrolls]
    print(f"\n{'=' * 70}\n[archs14] {c.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
        f"  train_scrolls={len(scroll_ids)}"
    )
    print(f"  scroll_ids={scroll_ids}")
    print(
        f"  multitile={getattr(c.model, 'multitile', False)}"
        f"  subtile={getattr(c.model, 'multitile_subtile', 0)}"
        f"  grid={getattr(c.model, 'multitile_grid', 0)}"
        f"  train_step={getattr(c.data, 'multitile_train_step', 0)}"
        f"  pos_only={getattr(c.data, 'multitile_pos_only', False)}"
        f"  attn_mil={c.model.attn_mil}"
    )
    print(
        f"  n_epochs={c.tra.n_epochs}  probe_int={c.tra.probe_int}  eval_int={c.tra.eval_int}"
        f"  fast_eval_figure={c.tra.fast_eval_figure}  eval_bs={c.data.eval_infer_bs}"
        f"  eval_prefetch={getattr(c.data, 'eval_prefetch', 0)}  tta_mode={getattr(c.data, 'tta_mode', 'flips')}"
    )
    print(
        f"  loss={c.tra.loss_type}  supcon={c.tra.supcon}  spill_reduction={c.tra.spill_reduction}"
        f"  inklabel_dir={c.data.inklabel_dir}"
    )
    print(
        f"  aug: flip={c.dl.flip_prob} rot={c.dl.rotation_prob} fda={getattr(c.dl, 'fda_prob', 0)}"
        f" elastic={getattr(c.dl, 'elastic_prob', 0)}"
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
    ap = argparse.ArgumentParser(description="campaign_archs_14: single vs multitile head A/B (w013)")
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

    print(f"[archs14] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs14] 1-scroll w013 [{_W013_ID}]  single vs multitile head")

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

    print(f"\n{'=' * 70}\n[archs14] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        tag = next(str(t["tag"]) for t in TESTS if str(t["tid"]) == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
