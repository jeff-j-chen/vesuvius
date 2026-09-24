"""campaign 23: focused triple-scroll generalization tests.

Every run uses literal surface-relative eight-slice input, plain LSE, and the
15-epoch campaign-21 operating point on w013, 500P2, and w044.

Usage:
    python3 campaign_archs_23.py --dry-run
    python3 campaign_archs_23.py --only baseline
    python3 campaign_archs_23.py --from xfrag_curriculum_0p8
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

LOG_DIR = "./runs_archs23"
MODEL_DIR = "models/archs23"
_SCROLL_IDS = (20240304141531, 20250628074500, 20260115000000)
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}
_missing = set(_SCROLL_IDS) - set(_SCROLLS_BY_ID)
if _missing:
    raise RuntimeError(f"campaign-23 scroll definitions missing: {sorted(_missing)}")
TRIPLE_SCROLLS = [_SCROLLS_BY_ID[scroll_id] for scroll_id in _SCROLL_IDS]


def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"23_{tid}",
        "scrolls": TRIPLE_SCROLLS,
        "max_samples_per_epoch": 6_667,
    }
    test.update(overrides)
    return test


_XFRAG = {
    "supcon_cross_frag": True,
    "supcon_curriculum": True,
    "supcon_lambda_start": 0.05,
    "supcon_lambda_end": 0.8,
    "supcon_curriculum_epochs": 8,
}
_CTX = {
    "context_replace_prob": 0.35,
    "context_replace_margin": 8,
    "context_replace_feather": 40,
}
_CUTOUT = {
    "cutout_prob": 0.50,
    "cutout_max_frac": 0.16,
    "cutout_n_patches": 3,
}

TESTS = [
    _test("baseline"),
    _test("baseline_depth4", depth=4, allow_depth4=True),
    _test("baseline_depth12", depth=12),
    _test("xfrag_curriculum_0p8", **_XFRAG),
    _test(
        "replace_p035_feather40",
        context_replace_prob=0.35,
        context_replace_margin=20,
        context_replace_feather=40,
    ),
    _test("replace_p035_feather40_margin8", **_CTX),
    _test("cutout_p050_f160_n3", **_CUTOUT),
    _test("jitter1", depth_jitter=1),
    _test("destructive_combo", **_CTX, **_CUTOUT, depth_jitter=1),
    _test("full_strength_combo", **_XFRAG, **_CTX, **_CUTOUT, depth_jitter=1),
    _test("feature_depth_fusion", feature_depth_fusion=True),
    _test("mae_preserve", encoder_freeze_epochs=2, encoder_lr_scale=0.1),
    _test(
        "per_column_canonical_d12_to8",
        depth=12,
        surface_canonicalize=True,
        surface_canonical_depth=8,
    ),
    _test(
        "depth_view_consistency_p050_o2",
        depth_view_consistency=True,
        depth_view_consistency_prob=0.5,
        depth_view_consistency_lambda=0.2,
        depth_view_consistency_offset=2,
    ),
]

_OVERRIDE_MAP = {
    "depth": ("data", "depth"),
    "depth_jitter": ("data", "depth_jitter"),
    "allow_depth4": ("model", "allow_depth4"),
    "feature_depth_fusion": ("model", "feature_depth_fusion"),
    "surface_canonicalize": ("model", "surface_canonicalize"),
    "surface_canonical_depth": ("model", "surface_canonical_depth"),
    "cutout_prob": ("dl", "cutout_prob"),
    "cutout_max_frac": ("dl", "cutout_max_frac"),
    "cutout_n_patches": ("dl", "cutout_n_patches"),
    "context_replace_prob": ("dl", "context_replace_prob"),
    "context_replace_margin": ("dl", "context_replace_margin"),
    "context_replace_feather": ("dl", "context_replace_feather"),
    "supcon_cross_frag": ("tra", "supcon_cross_frag"),
    "supcon_curriculum": ("tra", "supcon_curriculum"),
    "supcon_lambda_start": ("tra", "supcon_lambda_start"),
    "supcon_lambda_end": ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    "encoder_lr_scale": ("tra", "encoder_lr_scale"),
    "encoder_freeze_epochs": ("tra", "encoder_freeze_epochs"),
    "depth_view_consistency": ("tra", "depth_view_consistency"),
    "depth_view_consistency_prob": ("tra", "depth_view_consistency_prob"),
    "depth_view_consistency_lambda": ("tra", "depth_view_consistency_lambda"),
    "depth_view_consistency_offset": ("tra", "depth_view_consistency_offset"),
}


def build_config(test: dict):
    """build the campaign-21 literal-surface baseline plus one focused override."""
    config = campaign20_base_config(str(test["tag"]))
    config.init_weights = "models/mae_nnunet_192_ibn_depth8_22scroll_2k.pth"
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 15
    config.tra.eval_int = 15
    config.tra.fast_eval_figure = True
    config.tra.lr = 1.5e-4
    config.dl.batch_size = 96
    config.dl.num_workers = 8
    config.data.eval_infer_bs = 192
    config.data.eval_prefetch = 4
    config.data.eval_chunk_gb = 4.0
    config.data.scrolls = list(test["scrolls"])
    config.data.max_samples_per_epoch = int(test["max_samples_per_epoch"])
    config.data.vis_scroll_ids = list(_SCROLL_IDS)
    config.tra.eval_int_scrolls = 3

    config.data.depth = 8
    config.data.depth_jitter = 0
    config.data.surface_relative_depth_window = True
    config.model.learned_surface = False
    config.model.new_learned_surface = False
    config.model.better_surface = False
    config.model.surface_teacher_input = True
    config.model.surface_canonicalize = False
    config.model.feature_attn_mil = False
    config.model.feature_depth_fusion = False
    config.model.attn_mil = False
    config.tra.spill_reduction = False
    config.tra.spill_lambda = 0.0

    # Ordinary SupCon remains part of the baseline; only designated arms make it cross-fragment.
    config.tra.supcon = True
    config.tra.supcon_cross_frag = False
    config.tra.supcon_curriculum = True
    config.tra.supcon_lambda_start = 0.05
    config.tra.supcon_lambda_end = 0.5
    config.tra.supcon_curriculum_epochs = 8
    config.tra.encoder_lr_scale = 1.0
    config.tra.encoder_freeze_epochs = 0
    config.tra.depth_view_consistency = False
    config.tra.depth_view_consistency_prob = 0.5
    config.tra.depth_view_consistency_lambda = 0.2
    config.tra.depth_view_consistency_offset = 2
    # DANN stays disabled in campaign 23. Fixed 0.05 is retained as the future reference.
    config.tra.dann = False
    config.tra.dann_lambda = 0.0
    config.tra.dann_grl_anneal = False
    config.tra.dann_n_domains = 3
    config.dl.cutout_prob = 0.0
    config.dl.context_replace_prob = 0.0

    for key, (section, attr) in _OVERRIDE_MAP.items():
        if key in test:
            setattr(getattr(config, section), attr, test[key])

    os.makedirs(MODEL_DIR, exist_ok=True)
    config.save_final = os.path.join(MODEL_DIR, f"{test['tid']}.pth")
    return config


def run_test(config, dry_run: bool) -> bool:
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    print(f"\n{'=' * 78}\n[campaign23] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  scrolls={scroll_ids} total~={config.data.max_samples_per_epoch * len(scroll_ids)}"
        f" batch={config.dl.batch_size} lr={config.tra.lr:.2e}"
    )
    print(
        f"  depth={config.data.depth} literal_surface={config.model.surface_teacher_input}"
        f" canonical_depth={config.model.surface_canonical_depth if config.model.surface_canonicalize else 0}"
        f" feature_depth_fusion={config.model.feature_depth_fusion}"
        f" jitter=+/-{config.data.depth_jitter}"
        f" xfrag={config.tra.supcon_cross_frag}"
        f" supcon={config.tra.supcon_lambda_start}->{config.tra.supcon_lambda_end}"
    )
    print(
        f"  replace=(p={config.dl.context_replace_prob}, margin={config.dl.context_replace_margin},"
        f" feather={config.dl.context_replace_feather})"
        f" cutout=(p={config.dl.cutout_prob}, frac={config.dl.cutout_max_frac},"
        f" n={config.dl.cutout_n_patches})"
    )
    print(
        f"  mae_preserve=(freeze_epochs={config.tra.encoder_freeze_epochs},"
        f" encoder_lr_scale={config.tra.encoder_lr_scale})"
    )
    print(
        f"  depth_view_consistency=(enabled={config.tra.depth_view_consistency},"
        f" p={config.tra.depth_view_consistency_prob},"
        f" lambda={config.tra.depth_view_consistency_lambda},"
        f" offset=+/-{config.tra.depth_view_consistency_offset})"
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
    parser = argparse.ArgumentParser(description="campaign 23: triple-scroll focused tests")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in TESTS if test["tid"] in wanted]
        missing = wanted - {test["tid"] for test in selected}
        if missing:
            raise ValueError(f"unknown test ids: {sorted(missing)}")
    elif args.from_id:
        ids = [test["tid"] for test in TESTS]
        if args.from_id not in ids:
            raise ValueError(f"unknown --from {args.from_id!r}; valid={ids}")
        selected = TESTS[ids.index(args.from_id):]

    print(f"[campaign23] {len(selected)} run(s) queued (log -> {LOG_DIR})")
    results = {}
    for test in selected:
        config = build_config(test)
        results[test["tid"]] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'=' * 78}\n[campaign23] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
