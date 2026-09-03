"""campaign_archs_17.py -- surface, attention, and geometry follow-ups on w013.

All arms use campaign 16's hand-mask split and synchronized multitile flips/rotations.
Elastic and context jitter stay disabled until their dense target warps are implemented.

Tests:
    baseline                 c16_t8 control with corrected augmentation + cell SupCon
    surface_softmax_fixedaug new surface feature without forced aggregation
    surface_guided           entropy-gated surface-band/global aggregation
    feature_attn_c16_t8      feature-level attention-MIL at the winning geometry
    feature_attn_c32_t8      32px center, 4x4 8px targets
    feature_attn_c16_t4      16px center, 4x4 4px targets

  python campaign_archs_17.py --dry-run
    python campaign_archs_17.py --only surface_softmax_fixedaug
  python campaign_archs_17.py
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

from campaign_archs_16 import TESTS as _TESTS16
from campaign_archs_16 import _OVERRIDES as _OVERRIDES16
from campaign_archs_15 import _base_config as _base_config15
from utils.config import Config

LOG_DIR = "./runs_archs17"

_C16_MANUAL = next(test for test in _TESTS16 if str(test["tid"]) == "manual_split")
_BASE17 = {key: value for key, value in _C16_MANUAL.items() if key not in {"tid", "tag"}}
_BASE17.update(elastic_prob=0.0, ctx_jitter=0)
_OVERRIDES = dict(_OVERRIDES16)
_OVERRIDES.update({
    "new_learned_surface": ("model", "new_learned_surface"),
    "new_surface_lambda": ("tra", "new_surface_lambda"),
    "new_surface_smooth_lambda": ("tra", "new_surface_smooth_lambda"),
    "surface_guided_mil": ("model", "surface_guided_mil"),
    "surface_guided_mix": ("model", "surface_guided_mix"),
    "surface_band_sigma": ("model", "surface_band_sigma"),
    "feature_attn_mil": ("model", "feature_attn_mil"),
    "ctx_jitter": ("data", "ctx_jitter"),
})


def _mk17(tid: str, tag: str, **overrides: object) -> dict:
    test = dict(_BASE17)
    test.update(overrides)
    test["tid"] = tid
    test["tag"] = tag
    return test


TESTS = [
    _mk17(
        "baseline",
        "17_baseline",
        simple_split=False,
        new_learned_surface=False,
    ),
    _mk17(
        "surface_softmax_fixedaug",
        "17_surface_softmax_fixedaug",
        simple_split=False,
        new_learned_surface=True,
        new_surface_lambda=0.1,
        new_surface_smooth_lambda=0.02,
    ),
    _mk17(
        "surface_guided",
        "17_surface_guided",
        simple_split=False,
        new_learned_surface=True,
        surface_guided_mil=True,
        surface_guided_mix=0.5,
        surface_band_sigma=1.5,
        new_surface_lambda=0.1,
        new_surface_smooth_lambda=0.02,
    ),
    _mk17(
        "feature_attn_c16_t8",
        "17_feature_attn_c16_t8",
        simple_split=False,
        new_learned_surface=False,
        attn_mil=False,
        feature_attn_mil=True,
        multitile_subtile=8,
        multitile_grid=2,
    ),
    _mk17(
        "feature_attn_c32_t8",
        "17_feature_attn_c32_t8",
        simple_split=False,
        new_learned_surface=False,
        attn_mil=False,
        feature_attn_mil=True,
        multitile_subtile=8,
        multitile_grid=4,
    ),
    _mk17(
        "feature_attn_c16_t4",
        "17_feature_attn_c16_t4",
        simple_split=False,
        new_learned_surface=False,
        attn_mil=False,
        feature_attn_mil=True,
        multitile_subtile=4,
        multitile_grid=4,
    ),
]


def build_config(test: dict) -> Config:
    tid = str(test["tid"])
    config = _base_config15(str(test["tag"]))
    config.tra.log_dir = LOG_DIR
    for key, (section, attr) in _OVERRIDES.items():
        if key not in test:
            continue
        try:
            setattr(getattr(config, section), attr, test[key])
        except AttributeError:
            print(f"[WARNING] {tid}: {section}.{attr} does not exist")

    init_weights = test.get("init_weights")
    if init_weights and os.path.exists(init_weights):
        config.init_weights = str(init_weights)
    elif init_weights:
        print(f"[archs17] init_weights '{init_weights}' not found -- {tid} trains from scratch")

    config.dl.data_aug = any([
        config.dl.flip_prob,
        config.dl.rotation_prob,
        config.dl.noise_prob,
        config.dl.brightness_prob,
        config.dl.contrast_prob,
        config.dl.cutout_prob,
        config.dl.depth_mask_prob,
        getattr(config.dl, "elastic_prob", 0.0),
    ])
    os.makedirs("models/archs17", exist_ok=True)
    setattr(config, "save_final", f"models/archs17/{tid}_final.pth")
    return config


def run_test(config: Config, dry_run: bool) -> bool:
    subtile = int(config.model.multitile_subtile)
    grid = int(config.model.multitile_grid)
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]

    print(f"\n{'=' * 70}\n[archs17] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={config.model.arch}  scrolls={scroll_ids}  simple_split={config.data.simple_split}"
        f"  new_surface={config.model.new_learned_surface}"
        f"  surface_guided={config.model.surface_guided_mil}"
        f"  feature_attn={config.model.feature_attn_mil}"
    )
    print(
        f"  context={config.data.context_size}px  center={subtile * grid}px"
        f"  subtile={subtile}px  grid={grid}x{grid}={grid * grid} targets"
    )
    print(
        f"  epochs={config.tra.n_epochs}  batch={config.dl.batch_size}  lr={config.tra.lr:.2e}"
        f"  surface_lambda={config.tra.new_surface_lambda}"
        f"  smooth_lambda={config.tra.new_surface_smooth_lambda}"
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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign_archs_17: depth-softmax surface A/B on w013")
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
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: "
                  f"{[str(test['tid']) for test in TESTS]}")
            return
    elif args.from_id:
        ids = [str(test["tid"]) for test in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs17] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs17] fixed augmentation + surface/attention/geometry tests on the hand-mask split")

    results = {}
    for test in selected:
        tid = str(test["tid"])
        config = build_config(test)
        results[tid] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'=' * 70}\n[archs17] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        test = next(item for item in TESTS if str(item["tid"]) == tid)
        print(f"  {tid} ({test['tag']}) new_surface={test['new_learned_surface']}: {status}")


if __name__ == "__main__":
    main()
