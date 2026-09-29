"""campaign 39: cross-resolution pretrains (crossres/PLAN.md) and v8-in's input handling

Base: campaign 38's holdout_n96_combined (bag rank + regime weights + RSC + dropout/augs, native 96 px
field, ring c2g2s4, sigma-8 soft edges, early-gated patch-GroupDRO, seed 41); pherc0841 and pherc0009b
held out and rendered at full extent after the final epoch. The campaign-37 holdout_n96 and
holdout_n96_seed42 runs are copied into ./runs_archs39 for comparison.

The cross-resolution MAEs were continued from the production (non-fiber) native-96 MAE on another machine,
so they have no fibre-coordinate branch: every arm that uses them runs the base without it, against a
production-MAE control with the same config.

| arm                             | init / input                                                   | batch / lr  |
|---------------------------------|----------------------------------------------------------------|-------------|
| holdout_n96_combined_v8norm     | the base (fiber MAE), v8-in input: raw clipped to [0, 200] /   | 96 / 1.5e-4 |
|                                 | 255, z-scored per crop                                         |             |
| holdout_n96_nofiber             | control: the base without fiber, production native-96 MAE      | 96 / 1.5e-4 |
| holdout_n96_crossres_depth (D)  | MAE + head predicting 32 x 96^2 (4x depth) of the 2.4 um scan  | 96 / 1.5e-4 |
| holdout_n96_crossres_xyz (X)    | MAE + head predicting 32 x 192^2 (4x depth, 2x x/y)            | 96 / 1.5e-4 |
| holdout_n96_slab (S)            | the same continued MAE with slab masking, no 2.4 um pairs      | 96 / 1.5e-4 |
| holdout_n96_slab_b32            | S at the upsampled arms' batch: their native baseline          | 32 / 1e-4   |
| holdout_n96_upsampled_trilinear | (U-1) crop normalised, then trilinear x2 x/y, x4 depth pooled  | 32 / 1e-4   |
|                                 | 2 -> 16 x 192^2 into the network; MAE continued on that        |             |
| holdout_n96_upsampled_learned   | (U-2) the same with trilinear + a residual learned from the    | 32 / 1e-4   |
|                                 | 2.4 um pairs (models/upsampler_learned_xy2_d4.pth)             |             |

D - S and X - S are what the real high-resolution pairs add; S - nofiber is slab masking alone. The
upsampled arms run at the native192_depth16 recipe (16 x 192^2 per crop, 8x the voxels of the base), so
they are compared with holdout_n96_slab_b32 under one fine-tune config and seed.

Upsampled arms (model.input_upsampler): the data pipeline is unchanged (96 px context, 8-slice
surface-relative window, 16 px tiles, labels and masks on the native grid). The model normalises, then
applies the frozen upsampler in _prepare_input, and is built at the network grid (context 192, depth 16,
tile and multitile sub-tile 32 network px = 16 native px), so each output cell covers the base's papyrus.

The pretrained checkpoints are produced elsewhere; the campaign refuses to start an arm whose checkpoint
(or upsampler) is missing. A JSON sidecar next to an upsampled MAE (models/<name>.json with
"input_upsampler" and optionally "upsampler_depth_pool") overrides the default upsampler.

Usage:
    python3 campaign_archs_39.py --dry-run
    python3 campaign_archs_39.py --smoke
    python3 campaign_archs_39.py --only holdout_n96_crossres_depth,holdout_n96_slab
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_33 as campaign33
import campaign_archs_34 as campaign34
import campaign_archs_35 as campaign35
import campaign_archs_36 as campaign36
import campaign_archs_37 as campaign37
import campaign_archs_38 as campaign38
from utils.config import startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs39"
MODEL_DIR = "models/archs39"
COMBINED = campaign38.COMBINED
NOFIBER = {**COMBINED, "model.fiber_coordinate_branch": False}
V8_NORM = {"data.norm_mode": "raw255", "model.input_tile_norm": "clip200"}
PRODUCTION_MAE = "early_gated_native96"
# native192_depth16 ran at this batch; the upsampled network grid is the same 16 x 192^2
UPSAMPLED_TRAINING = {"batch_size": 32, "lr": 1e-4}
LEARNED_UPSAMPLER = "models/upsampler_learned_xy2_d4.pth"


def _test(tid: str, changes: dict, arch: dict | None = None, init_weights: str | None = None,
          upsampler: str | None = None) -> dict:
    """the campaign-38 combined base (via campaign 37) plus this arm's changes."""
    test = campaign37._test(tid, changes, arch=arch)
    test["tag"] = f"39_{tid}"
    test["init_weights"] = init_weights
    test["upsampler"] = upsampler
    return test


TESTS = [
    # changes the prepared-dataset cache key (norm_mode), so it loads its own data first
    _test("holdout_n96_combined_v8norm", {**COMBINED, **V8_NORM},
          arch={"pretrain_key": "early_gated_fiber_native96"}),
    _test("holdout_n96_nofiber", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE}),
    _test("holdout_n96_crossres_depth", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_crossres_depth.pth"),
    _test("holdout_n96_crossres_xyz", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_crossres_xyz.pth"),
    _test("holdout_n96_slab", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_slab_native96.pth"),
    _test("holdout_n96_slab_b32", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE, **UPSAMPLED_TRAINING},
          init_weights="models/mae_slab_native96.pth"),
    _test("holdout_n96_upsampled_trilinear", NOFIBER,
          arch={"pretrain_key": PRODUCTION_MAE, **UPSAMPLED_TRAINING},
          init_weights="models/mae_upsampled_trilinear_native96.pth", upsampler="trilinear"),
    _test("holdout_n96_upsampled_learned", NOFIBER,
          arch={"pretrain_key": PRODUCTION_MAE, **UPSAMPLED_TRAINING},
          init_weights="models/mae_upsampled_learned_native96.pth", upsampler=LEARNED_UPSAMPLER),
]


def _upsampler_settings(test: dict) -> tuple[str, int]:
    """(upsampler, depth pool): the MAE's sidecar if it names one, else the arm's default."""
    upsampler, pool = str(test["upsampler"]), 2
    sidecar = (ROOT / test["init_weights"]).with_suffix(".json")
    if sidecar.is_file():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        upsampler = str(info.get("input_upsampler") or upsampler)
        pool = int(info.get("upsampler_depth_pool", pool))
    return upsampler, pool


def _external_files(test: dict) -> list[str]:
    files = [test["init_weights"]] if test["init_weights"] else []
    if test["upsampler"]:
        upsampler, _ = _upsampler_settings(test)
        if upsampler != "trilinear":
            files.append(upsampler)
    return files


def preflight_external(selected: list[dict], dry_run: bool) -> None:
    """the cross-resolution checkpoints come from another machine; stop before any arm if one is missing."""
    missing = sorted({
        path for test in selected for path in _external_files(test) if not (ROOT / path).is_file()
    })
    if not missing:
        return
    message = f"campaign 39 needs the externally pretrained files: missing={missing}"
    if not dry_run:
        raise FileNotFoundError(message)
    print(f"[campaign39] WARNING {message}", flush=True)


@contextlib.contextmanager
def _campaign35_paths():
    saved = campaign35.LOG_DIR, campaign35.MODEL_DIR
    campaign35.LOG_DIR, campaign35.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign35.LOG_DIR, campaign35.MODEL_DIR = saved


def build_config(test: dict):
    with _campaign35_paths():
        config = campaign35.build_config(test)
    if test["init_weights"]:
        config.init_weights = test["init_weights"]
    if test["upsampler"]:
        config.model.input_upsampler, config.model.input_upsampler_depth_pool = _upsampler_settings(test)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 39: cross-resolution pretrains and v8-in input")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm on pherc0814 alone from random init, no figures, "
                             "separate log/model dirs")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs39_smoke", "models/archs39_smoke"
        # never re-measure quality filters from volumes that may be mid-rebuild
        campaign35._quality_sources_newer_than_cache = lambda: False

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

    with startup_output():
        campaign29.preflight_train_masks(
            campaign33.CAMPAIGN33_SCROLLS,
            inklabel_dir=Path(campaign33.ROOT) / campaign33.INKLABEL_DIR,
            strict=not (args.dry_run or args.smoke),
        )
        if not args.smoke:
            campaign34.preflight_pretraining(selected, args.dry_run)
            preflight_external(selected, args.dry_run)
        print(f"[campaign39] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        dropped = campaign36._smoke_config(config) if args.smoke else []
        if args.smoke and config.model.input_upsampler and not (ROOT / config.model.input_upsampler).is_file():
            config.model.input_upsampler = "trilinear"
        with startup_output():
            print(f"[campaign39] {test['tid']}: ctx={config.data.context_size} depth={config.data.depth} "
                  f"norm={config.data.norm_mode}/{config.model.input_tile_norm or 'none'} "
                  f"upsampler={config.model.input_upsampler or 'none'}"
                  f"/pool{config.model.input_upsampler_depth_pool} "
                  f"fiber={config.model.fiber_coordinate_branch} batch={config.dl.batch_size} lr={config.tra.lr} "
                  f"seed={config.tra.seed} init={config.init_weights} dropped={len(dropped)} "
                  f"overrides={test['config']}", flush=True)
        if args.dry_run:
            success = campaign31.run_test(config, True)
        elif args.smoke:
            success = campaign31.run_test_isolated(config)
        else:
            campaign29.prewarm_data_cache(config)
            success = campaign31.run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        del config
        gc.collect()

    print(f"\n{'=' * 78}\n[campaign39] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
