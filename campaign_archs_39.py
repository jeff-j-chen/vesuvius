"""campaign 39: cross-resolution pretrains (crossres/PLAN.md) on a surface-normalised baseline

Base: campaign 38's holdout_n96_combined (bag rank + fiber + regime weights + RSC + dropout/augs, native
96 px field, ring c2g2s4, sigma-8 soft edges, early-gated patch-GroupDRO, seed 41) plus surface-anchored
normalisation (gap -> 0.1, surface papyrus -> 0.5); pherc0841 and pherc0009b held out and rendered at full
extent after the final epoch. Every arm trains at batch 32 / lr 1e-4. The campaign-37 holdout_n96 and
holdout_n96_seed42 runs are copied into ./runs_archs39 for comparison.

The cross-resolution MAEs were continued from the production (non-fiber) native-96 MAE on another machine,
so they have no fibre-coordinate branch: the arms that use them run the base without it. They, the learned
upsampler and the degrader were all pretrained on globally normalised data; only fine-tuning is surface-
normalised (as holdout_n96_combined_surface_norm does with the global-norm fiber MAE).

| arm                               | init / input                                                     |
|-----------------------------------|------------------------------------------------------------------|
| holdout_n96_combined_surface_norm | the baseline: c38 combined (fiber MAE) + surface-anchored norm   |
| holdout_n96_upsampled_learned     | (U-2) crop normalised, then trilinear + a residual learned from  |
|                                   | the 2.4 um pairs, x2 x/y, x4 depth pooled 2 -> 16 x 192^2; MAE   |
|                                   | continued on that (models/upsampler_learned_xy2_d4.pth)          |
| holdout_n96_upsampled_trilinear   | (U-1) the same with plain trilinear upsampling                   |
| holdout_n96_crossres_depth (D)    | MAE + head predicting 32 x 96^2 (4x depth) of the 2.4 um scan    |
| holdout_n96_crossres_xyz (X)      | MAE + head predicting 32 x 192^2 (4x depth, 2x x/y)              |
| holdout_n96_slab (S)              | the same continued MAE with slab masking, no 2.4 um pairs        |
| holdout_n96_downsampled           | (R-1) the baseline, with every fine-scan fragment (FINE_NATIVE_  |
|                                   | SCROLL_IDS: pherc1667, paris4, paris2 fr143/fr47, 51cr4 fr8,     |
|                                   | paris1 fr34) read from its degrader-translated sibling zarr      |
| holdout_n96_downsampled_noise     | R-1 plus real 113 keV / 1.2 m scan noise on every training crop  |
|                                   | of the translated fragments (crossres/build_noise_bank.py)       |
| ..._downsampled_upsampled_trilinear | (R-2) R-1's data through U-1's input head and MAE              |
| ..._downsampled_upsampled_learned   | (R-3) R-1's data through U-2's input head and MAE              |

U-2 - U-1 is learned vs fixed upsampling; D - S and X - S are what the real high-resolution pairs add.
R-1 is judged against the baseline, R-2 against U-1 and R-3 against U-2.

Upsampled arms (model.input_upsampler): the data pipeline is unchanged (96 px context, 8-slice
surface-relative window, 16 px tiles, labels and masks on the native grid). The model normalises, then
applies the frozen upsampler in _prepare_input, and is built at the network grid (context 192, depth 16,
tile and multitile sub-tile 32 network px = 16 native px), so each output cell covers the base's papyrus.

The pretrained checkpoints are produced elsewhere; the campaign refuses to start an arm whose checkpoint
(or upsampler) is missing. The JSON sidecar next to an upsampled MAE (models/<name>.json, written by
crossres/mae_pretrain_crossres.py: "upsampler") names the upsampler the MAE was trained behind.

Downsampled arms (plan R, crossres/PLAN.md 0.7) need the translated siblings first; the campaign never
writes them and refuses to start if any is missing, stale for models/degrader_pooled_native.pth, or has no
norm or surface-anchor entry:
    python assemble_training_segments.py --degrader models/degrader_pooled_native.pth
Masks, labels, train masks and surface maps are shared with the original zarrs. The noise arm also needs
the local bank, built in the same normalisation: python crossres/build_noise_bank.py

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
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs39"
MODEL_DIR = "models/archs39"
COMBINED = campaign38.COMBINED
BASE = {**COMBINED, **campaign38.SURFACE_NORM}
NOFIBER = {**BASE, "model.fiber_coordinate_branch": False}
FIBER_MAE = "early_gated_fiber_native96"
PRODUCTION_MAE = "early_gated_native96"
# every arm, whether or not a larger batch would fit
TRAINING = {"batch_size": 32, "lr": 1e-4}
LEARNED_UPSAMPLER = "models/upsampler_learned_xy2_d4.pth"
DEGRADER = "models/degrader_pooled_native.pth"
TRANSLATED_SUFFIX = ".translated"
DOWNSAMPLED = {"data.zarr_suffix": {str(sid): TRANSLATED_SUFFIX for sid in campaign35.FINE_NATIVE_SCROLL_IDS}}
NOISE_BANK = "_ves_tmp/native_noise_bank.npy"
NATIVE_NOISE = {"data.native_noise": NOISE_BANK, "data.native_noise_scale": 1.0,
                "data.native_noise_ids": [int(sid) for sid in campaign35.FINE_NATIVE_SCROLL_IDS]}


def _test(tid: str, changes: dict, arch: dict | None = None, init_weights: str | None = None,
          upsampler: str | None = None) -> dict:
    """campaign 37's harness plus this arm's changes, at batch 32 / lr 1e-4."""
    test = campaign37._test(tid, changes, arch={**(arch or {}), **TRAINING})
    test["tag"] = f"39_{tid}"
    test["init_weights"] = init_weights
    test["upsampler"] = upsampler
    return test


TESTS = [
    _test("holdout_n96_combined_surface_norm", BASE, arch={"pretrain_key": FIBER_MAE}),
    _test("holdout_n96_upsampled_learned", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_upsampled_learned_native96.pth", upsampler=LEARNED_UPSAMPLER),
    _test("holdout_n96_upsampled_trilinear", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_upsampled_trilinear_native96.pth", upsampler="trilinear"),
    _test("holdout_n96_crossres_depth", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_crossres_depth.pth"),
    _test("holdout_n96_crossres_xyz", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_crossres_xyz.pth"),
    _test("holdout_n96_slab", NOFIBER, arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_slab_native96.pth"),
    # last: these read the translated fine-scan volumes (one shared prepared-dataset cache key)
    _test("holdout_n96_downsampled", {**BASE, **DOWNSAMPLED}, arch={"pretrain_key": FIBER_MAE}),
    # the degrader predicts the mean scan; real 113 keV noise is added per training crop (plan R.3)
    _test("holdout_n96_downsampled_noise", {**BASE, **DOWNSAMPLED, **NATIVE_NOISE},
          arch={"pretrain_key": FIBER_MAE}),
    _test("holdout_n96_downsampled_upsampled_trilinear", {**NOFIBER, **DOWNSAMPLED},
          arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_upsampled_trilinear_native96.pth", upsampler="trilinear"),
    _test("holdout_n96_downsampled_upsampled_learned", {**NOFIBER, **DOWNSAMPLED},
          arch={"pretrain_key": PRODUCTION_MAE},
          init_weights="models/mae_upsampled_learned_native96.pth", upsampler=LEARNED_UPSAMPLER),
]


def _upsampler_setting(test: dict) -> str:
    """the upsampler named by the MAE's sidecar, else the arm's default."""
    upsampler = str(test["upsampler"])
    sidecar = (ROOT / test["init_weights"]).with_suffix(".json")
    if sidecar.is_file():
        upsampler = str(json.loads(sidecar.read_text(encoding="utf-8")).get("upsampler") or upsampler)
    return upsampler


def _external_files(test: dict) -> list[str]:
    files = [test["init_weights"]] if test["init_weights"] else []
    if test["upsampler"]:
        upsampler = _upsampler_setting(test)
        if upsampler != "trilinear":
            files.append(upsampler)
    if test["config"].get("data.zarr_suffix"):
        files.append(DEGRADER)
    if test["config"].get("data.native_noise"):
        bank = str(test["config"]["data.native_noise"])
        files += [bank, str(Path(bank).with_suffix(".json"))]
    return files


def _translation_problems(test: dict) -> list[str]:
    """translated siblings that are missing, made by another degrader checkpoint, or lack a norm entry."""
    suffixes = test["config"].get("data.zarr_suffix") or {}
    if not suffixes or not (ROOT / DEGRADER).is_file():
        return []
    from utils.degrader import is_current
    from utils.norm import SURFACE_ANCHOR_CACHE_PATH, UNIFIED_CACHE_PATH, _read_json, load_cached_norm
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2"))
    anchors = _read_json(str(ROOT / SURFACE_ANCHOR_CACHE_PATH))
    problems = []
    for scroll_id, suffix in suffixes.items():
        volume = zarr_root / f"{scroll_id}{suffix}.zarr"
        if not volume.is_dir():
            problems.append(f"{volume} missing")
        elif not is_current(str(volume), str(ROOT / DEGRADER)):
            problems.append(f"{volume} was not made by {DEGRADER}")
        elif load_cached_norm(f"{scroll_id}{suffix}", str(ROOT / UNIFIED_CACHE_PATH)) is None:
            problems.append(f"{scroll_id}{suffix} has no norm_cache.json entry")
        elif f"{scroll_id}{suffix}" not in anchors:
            problems.append(f"{scroll_id}{suffix} has no surface_anchor_cache.json entry")
    return problems


def preflight_external(selected: list[dict], dry_run: bool) -> None:
    """externally pretrained checkpoints and plan R's translated volumes; stop before any arm if one is missing."""
    missing = sorted({
        path for test in selected for path in _external_files(test) if not (ROOT / path).is_file()
    })
    problems = sorted({problem for test in selected for problem in _translation_problems(test)})
    if not missing and not problems:
        return
    message = f"campaign 39 inputs not ready: missing={missing} translation={problems}"
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
        config.model.input_upsampler = _upsampler_setting(test)
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
            if not args.dry_run:
                ensure_surface_anchors(campaign38.SURFACE_ANCHOR_SCROLL_IDS, str(ROOT / "ves_zarrs2"))
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
                  f"upsampler={config.model.input_upsampler or 'none'} "
                  f"translated={len(config.data.zarr_suffix)} "
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
