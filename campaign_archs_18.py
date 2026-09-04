"""campaign_archs_18.py -- character-balanced sampling on the c17 baseline.

All arms use connected-component character metrics. Metrics do not affect the
training objective, so the only experimental variable is character-balanced
sampling:

  baseline            ordinary window sampling
  character_balanced  uniform characters, one positive and one local-ring window

Additional arms independently add one hard augmentation to ordinary sampling:

    depth_warp           smooth spatially varying +/-2-slice depth displacement
    surface_attenuation  weaken local contrast only near the estimated surface
    acquisition_blur     mild in-plane point-spread blur
    correlated_noise     low-frequency noise shared across neighboring voxels
    context_cutout       cut out context while protecting the labeled center
    context_jitter       keep the global target fixed while moving it within context

All use w013, the manual train/validation mask, corrected multitile geometric
augmentation, and the campaign-17 c16_t8 architecture baseline.

  python campaign_archs_18.py --dry-run
  python campaign_archs_18.py --only baseline
  python campaign_archs_18.py --only character_balanced
  python campaign_archs_18.py
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

from campaign_archs_17 import TESTS as _TESTS17
from campaign_archs_17 import _OVERRIDES as _OVERRIDES17
from campaign_archs_15 import _base_config as _base_config15
from utils.config import Config

LOG_DIR = "./runs_archs18"

_C17_BASELINE = next(test for test in _TESTS17 if str(test["tid"]) == "baseline")
_BASE18 = {key: value for key, value in _C17_BASELINE.items() if key not in {"tid", "tag"}}
_OVERRIDES = dict(_OVERRIDES17)
_OVERRIDES.update({
    "character_balanced_sampling": ("data", "character_balanced_sampling"),
    "character_min_pixels": ("data", "character_min_pixels"),
    "character_macro_metrics": ("tra", "character_macro_metrics"),
    "character_score_threshold": ("tra", "character_score_threshold"),
    "character_recall_target": ("tra", "character_recall_target"),
    "character_max_ring_fpr": ("tra", "character_max_ring_fpr"),
    "depth_warp_prob": ("dl", "depth_warp_prob"),
    "depth_warp_max": ("dl", "depth_warp_max"),
    "depth_warp_sigma": ("dl", "depth_warp_sigma"),
    "surface_atten_prob": ("dl", "surface_atten_prob"),
    "surface_atten_min": ("dl", "surface_atten_min"),
    "surface_atten_max": ("dl", "surface_atten_max"),
    "surface_atten_sigma": ("dl", "surface_atten_sigma"),
    "acquisition_blur_prob": ("dl", "acquisition_blur_prob"),
    "acquisition_blur_min": ("dl", "acquisition_blur_min"),
    "acquisition_blur_max": ("dl", "acquisition_blur_max"),
    "correlated_noise_prob": ("dl", "correlated_noise_prob"),
    "correlated_noise_min": ("dl", "correlated_noise_min"),
    "correlated_noise_max": ("dl", "correlated_noise_max"),
    "correlated_noise_sigma": ("dl", "correlated_noise_sigma"),
    "cutout_protect_center": ("dl", "cutout_protect_center"),
    "target_aware_ctx_jitter": ("data", "target_aware_ctx_jitter"),
})


def _mk18(tid: str, tag: str, **overrides: object) -> dict:
    test = dict(_BASE18)
    test.update(overrides)
    test["tid"] = tid
    test["tag"] = tag
    return test


_CHARACTER_METRICS = dict(
    character_macro_metrics=True,
    character_min_pixels=8,
    character_score_threshold=0.5,
    character_recall_target=0.5,
    character_max_ring_fpr=0.1,
)

TESTS = [
    _mk18(
        "baseline",
        "18_baseline",
        character_balanced_sampling=False,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "character_balanced",
        "18_character_balanced",
        character_balanced_sampling=True,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "depth_warp",
        "18_depth_warp",
        character_balanced_sampling=False,
        depth_warp_prob=0.5,
        depth_warp_max=2.0,
        depth_warp_sigma=24.0,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "surface_attenuation",
        "18_surface_attenuation",
        character_balanced_sampling=False,
        surface_atten_prob=0.5,
        surface_atten_min=0.1,
        surface_atten_max=0.3,
        surface_atten_sigma=2.0,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "acquisition_blur",
        "18_acquisition_blur",
        character_balanced_sampling=False,
        acquisition_blur_prob=0.5,
        acquisition_blur_min=0.4,
        acquisition_blur_max=0.8,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "correlated_noise",
        "18_correlated_noise",
        character_balanced_sampling=False,
        correlated_noise_prob=0.5,
        correlated_noise_min=0.003,
        correlated_noise_max=0.01,
        correlated_noise_sigma=8.0,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "context_cutout",
        "18_context_cutout",
        character_balanced_sampling=False,
        cutout_prob=0.5,
        cutout_max_frac=0.12,
        cutout_n_patches=2,
        cutout_protect_center=True,
        **_CHARACTER_METRICS,
    ),
    _mk18(
        "context_jitter",
        "18_context_jitter",
        character_balanced_sampling=False,
        target_aware_ctx_jitter=True,
        ctx_jitter=32,
        **_CHARACTER_METRICS,
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
        print(f"[archs18] init_weights '{init_weights}' not found -- {tid} trains from scratch")

    config.dl.data_aug = any([
        config.dl.flip_prob,
        config.dl.rotation_prob,
        config.dl.noise_prob,
        config.dl.brightness_prob,
        config.dl.contrast_prob,
        config.dl.cutout_prob,
        config.dl.depth_mask_prob,
        getattr(config.dl, "elastic_prob", 0.0),
        getattr(config.dl, "depth_warp_prob", 0.0),
        getattr(config.dl, "surface_atten_prob", 0.0),
        getattr(config.dl, "acquisition_blur_prob", 0.0),
        getattr(config.dl, "correlated_noise_prob", 0.0),
    ])
    os.makedirs("models/archs18", exist_ok=True)
    setattr(config, "save_final", f"models/archs18/{tid}_final.pth")
    return config


def run_test(config: Config, dry_run: bool) -> bool:
    subtile = int(config.model.multitile_subtile)
    grid = int(config.model.multitile_grid)
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]

    print(f"\n{'=' * 70}\n[archs18] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={config.model.arch}  scrolls={scroll_ids}  simple_split={config.data.simple_split}"
        f"  character_balanced={config.data.character_balanced_sampling}"
    )
    print(
        f"  context={config.data.context_size}px  center={subtile * grid}px"
        f"  subtile={subtile}px  grid={grid}x{grid}={grid * grid} targets"
    )
    print(
        f"  character_metric: score>={config.tra.character_score_threshold}"
        f" recall>={config.tra.character_recall_target}"
        f" ring_fpr<={config.tra.character_max_ring_fpr}"
    )
    print(
        f"  epochs={config.tra.n_epochs}  batch={config.dl.batch_size}  lr={config.tra.lr:.2e}"
        f"  eval_int={config.tra.eval_int}  eval_bs={config.data.eval_infer_bs}"
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
    parser = argparse.ArgumentParser(description="campaign_archs_18: character-balanced sampling on w013")
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

    print(f"[archs18] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs18] ordinary versus character-balanced sampling; character metrics on both")

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

    print(f"\n{'=' * 70}\n[archs18] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        test = next(item for item in TESTS if str(item["tid"]) == tid)
        print(
            f"  {tid} ({test['tag']}) "
            f"character_balanced={test['character_balanced_sampling']}: {status}"
        )


if __name__ == "__main__":
    main()
