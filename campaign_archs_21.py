"""campaign_archs_21.py -- follow-up SupCon and surface experiments on 500P2

Prior tests are retained in PRIOR_TESTS for reference. TESTS queues:
- SupCon + plain LSE + the original supervised surface head
- SupCon + plain LSE + a physical, broad-context surface head
- SupCon + plain LSE + literal pre-generated depth-map features
- a fixed-weight SupCon sweep on plain LSE

  python campaign_archs_21.py --dry-run
    python campaign_archs_21.py --only baseline
    python campaign_archs_21.py --from old_surface
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

from campaign_archs_20 import base_config as campaign20_base_config
from utils.config import DEFAULT_SCROLLS

LOG_DIR = "./runs_archs21"
_SCROLL_IDS = {
    "500p2": 20250628074500,
    "w044": 20260115000000,
}
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}

_missing = set(_SCROLL_IDS.values()) - set(_SCROLLS_BY_ID)
if _missing:
    raise RuntimeError(f"campaign-21 scroll definitions missing: {sorted(_missing)}")


def _scrolls(*names: str):
    return [_SCROLLS_BY_ID[_SCROLL_IDS[name]] for name in names]


PRIOR_TESTS = [
    {
        "tid": "baseline",
        "tag": "21_baseline",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": False,
        "new_learned_surface": False,
    },
    {
        "tid": "supcon",
        "tag": "21_supcon",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": True,
        "learned_surface": False,
        "new_learned_surface": False,
    },
    {
        "tid": "old_surface",
        "tag": "21_old_surface",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": True,
        "new_learned_surface": False,
    },
    {
        "tid": "new_surface",
        "tag": "21_new_surface",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": False,
        "new_learned_surface": True,
    },
    {
        "tid": "plain_lse",
        "tag": "21_plain_lse",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": False,
        "new_learned_surface": False,
        "feature_attn_mil": False,
        "attn_mil": False,
    },
    {
        "tid": "feature_attn_no_entropy",
        "tag": "21_feature_attn_no_entropy",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": False,
        "new_learned_surface": False,
        "feature_attn_mil": True,
        "attn_mil": False,
        "attn_entropy_weight": 0.0,
    },
    {
        "tid": "depth20_jitter4",
        "tag": "21_depth20_jitter4",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": False,
        "learned_surface": False,
        "new_learned_surface": False,
        "depth": 20,
        "depth_jitter": 4,
        "require_symmetric_depth_jitter": True,
    },
]


def _next_test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"21_{tid}",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "supcon": True,
        "learned_surface": False,
        "new_learned_surface": False,
        "better_surface": False,
        "surface_teacher_input": False,
        "feature_attn_mil": False,
        "attn_mil": False,
    }
    test.update(overrides)
    return test


TESTS = [
    _next_test(
        "supcon_lse_new_surface",
        new_learned_surface=True,
    ),
    _next_test(
        "supcon_lse_better_surface",
        better_surface=True,
        compile_model=False,
    ),
    _next_test(
        "supcon_lse_literal_surface",
        surface_teacher_input=True,
        compile_model=False,
    ),
    _next_test(
        "supcon_lse_fixed_depth8_16",
        depth=8,
        depth_jitter=0,
        train_d_start=8,
        train_d_end=16,
        d_start=8,
        d_end=16,
    ),
    _next_test(
        "supcon_lse_literal_surface_slice8",
        surface_teacher_input=True,
        surface_relative_depth_window=True,
        depth=8,
        depth_jitter=0,
        compile_model=False,
    ),
    _next_test(
        "supcon_lse_literal_surface_slice8_jitter2",
        surface_teacher_input=True,
        surface_relative_depth_window=True,
        depth=8,
        depth_jitter=2,
        compile_model=False,
    ),
    _next_test("supcon_lse_fixed_010", supcon_curriculum=False, supcon_lambda=0.10),
    _next_test("supcon_lse_fixed_020", supcon_curriculum=False, supcon_lambda=0.20),
    _next_test("supcon_lse_fixed_050", supcon_curriculum=False, supcon_lambda=0.50),
]


def build_config(test: dict):
    """clone the campaign-20 soft-BCE baseline and change only data scope and scale."""
    config = campaign20_base_config(str(test["tag"]))
    config.tra.log_dir = LOG_DIR
    config.tra.lr = 1.0e-4
    config.dl.batch_size = 32
    config.data.scrolls = list(test["scrolls"])
    config.data.max_samples_per_epoch = int(test["max_samples_per_epoch"])
    config.data.vis_scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    config.tra.eval_int_scrolls = len(config.data.scrolls)
    config.tra.spill_reduction = False
    config.tra.spill_lambda = 0.0
    config.tra.supcon = bool(test["supcon"])
    config.tra.supcon_curriculum = bool(
        test.get("supcon_curriculum", config.tra.supcon_curriculum)
    )
    if "supcon_lambda" in test:
        config.tra.supcon_lambda = float(test["supcon_lambda"])
    config.model.learned_surface = bool(test["learned_surface"])
    config.model.new_learned_surface = bool(test["new_learned_surface"])
    config.model.better_surface = bool(test.get("better_surface", False))
    config.model.surface_teacher_input = bool(test.get("surface_teacher_input", False))
    if "compile_model" in test:
        config.model.compile_model = bool(test["compile_model"])
    if "surface_canonicalize" in test:
        config.model.surface_canonicalize = bool(test["surface_canonicalize"])
    if "surface_canonical_depth" in test:
        config.model.surface_canonical_depth = int(test["surface_canonical_depth"])
    if "feature_attn_mil" in test:
        config.model.feature_attn_mil = bool(test["feature_attn_mil"])
    if "attn_mil" in test:
        config.model.attn_mil = bool(test["attn_mil"])
    if "attn_entropy_weight" in test:
        config.model.attn_entropy_weight = float(test["attn_entropy_weight"])
    if "depth" in test:
        config.data.depth = int(test["depth"])
    if "depth_jitter" in test:
        config.data.depth_jitter = int(test["depth_jitter"])
    for attr in ("train_d_start", "train_d_end", "d_start", "d_end"):
        if attr in test:
            setattr(config.data, attr, int(test[attr]))
    if "surface_relative_depth_window" in test:
        config.data.surface_relative_depth_window = bool(
            test["surface_relative_depth_window"]
        )
    if "eval_int" in test:
        config.tra.eval_int = int(test["eval_int"])

    if bool(test.get("require_symmetric_depth_jitter", False)):
        nominal_start = int(config.data.train_d_start)
        source_end = int(config.data.train_d_end)
        min_start = nominal_start - int(config.data.depth_jitter)
        max_end = nominal_start + int(config.data.depth_jitter) + int(config.data.depth)
        if min_start < 0 or max_end > source_end:
            raise ValueError(
                f"depth jitter would leave [0, {source_end}): "
                f"depth={config.data.depth}, start={nominal_start}, "
                f"jitter=+/-{config.data.depth_jitter}"
            )

    os.makedirs("models/archs21", exist_ok=True)
    config.save_final = f"models/archs21/{test['tid']}_final.pth"
    return config


def run_test(config, dry_run: bool) -> bool:
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    nominal_start = int(config.data.train_d_start)
    jitter = int(config.data.depth_jitter)
    jitter_low = max(-jitter, -nominal_start)
    jitter_high = min(
        jitter,
        int(config.data.train_d_end) - nominal_start - int(config.data.depth),
    )
    print(f"\n{'=' * 70}\n[archs21] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  scrolls={scroll_ids} samples/scroll={config.data.max_samples_per_epoch}"
        f" total_samples~={config.data.max_samples_per_epoch * len(scroll_ids)}"
    )
    print(
        f"  batch={config.dl.batch_size} lr={config.tra.lr:.2e}"
        f" context={config.data.context_size}/ds{config.data.context_downsample}"
        f" center={config.model.multitile_subtile * config.model.multitile_grid}"
        f" depth={config.data.depth} jitter=[{jitter_low:+d},{jitter_high:+d}]"
        f" loss={config.tra.loss_type}"
    )
    print(
        f"  supcon={config.tra.supcon} old_surface={config.model.learned_surface}"
        f" new_surface={config.model.new_learned_surface}"
        f" better_surface={config.model.better_surface}"
        f" teacher_surface={config.model.surface_teacher_input}"
        f" spill={config.tra.spill_reduction}"
    )
    print(
        f"  feature_attn={config.model.feature_attn_mil}"
        f" voxel_attn={config.model.attn_mil}"
        f" attn_entropy={config.model.attn_entropy_weight}"
    )
    print(
        f"  canonicalize={config.model.surface_canonicalize}"
        f" canonical_depth={config.model.surface_canonical_depth}"
    )
    print(
        f"  supcon_curriculum={config.tra.supcon_curriculum}"
        f" supcon_lambda={config.tra.supcon_lambda}"
    )
    if dry_run:
        print("  [DRY RUN] skipping")
        return True

    from train import Trainer

    try:
        trainer = Trainer(config)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign_archs_21: per-scroll baseline isolation")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in TESTS if str(test["tid"]) in wanted]
        missing = wanted - {str(test["tid"]) for test in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found")
            return
    elif args.from_id:
        ids = [str(test["tid"]) for test in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs21] {len(selected)} run(s) queued (log -> {LOG_DIR})")
    results = {}
    for test in selected:
        tid = str(test["tid"])
        config = build_config(test)
        results[tid] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'=' * 70}\n[archs21] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
