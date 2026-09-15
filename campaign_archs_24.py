"""campaign 24: leave-one-fragment-out training over all 18 labeled fragments.

Each run holds out one complete fragment for visualization and trains on the
remaining 17 with the campaign-23 full-strength configuration plus fixed DANN.

Usage:
    python3 campaign_archs_24.py --dry-run
    python3 campaign_archs_24.py --only holdout_20250223000000
    python3 campaign_archs_24.py --from holdout_20260206000001
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from campaign_archs_23 import build_config as campaign23_build_config
from utils.config import DEFAULT_SCROLLS
from utils.dataloader import imread_gray

LOG_DIR = "./runs_archs24"
MODEL_DIR = "models/archs24"
TRAIN_MASK_DIR = Path("./train_masks")
MASK_DIR = Path("./masks")
INKLABEL_DIR = Path("./eroded_inklabels")
SURFACE_LABEL_DIR = Path("./surface_labels")
ZARR_DIR = Path(os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2"))

ALL_SCROLLS = tuple(DEFAULT_SCROLLS)
if len(ALL_SCROLLS) != 18:
    raise RuntimeError(f"campaign 24 requires exactly 18 scrolls, found {len(ALL_SCROLLS)}")
if len({int(scroll.scroll_id) for scroll in ALL_SCROLLS}) != len(ALL_SCROLLS):
    raise RuntimeError("campaign 24 scroll ids must be unique")
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in ALL_SCROLLS}
_HOLDOUT_IDS = (
    20250223000000,
    *(int(scroll.scroll_id) for scroll in ALL_SCROLLS if int(scroll.scroll_id) != 20250223000000),
)


def _test(holdout):
    holdout_id = int(holdout.scroll_id)
    train_scrolls = [
        scroll for scroll in ALL_SCROLLS
        if int(scroll.scroll_id) != holdout_id
    ]
    return {
        "tid": f"holdout_{holdout_id}",
        "tag": f"24_holdout_{holdout_id}",
        "holdout_id": holdout_id,
        "scrolls": train_scrolls,
        "max_samples_per_epoch": 6_667,
        "supcon_cross_frag": True,
        "supcon_curriculum": True,
        "supcon_lambda_start": 0.05,
        "supcon_lambda_end": 0.8,
        "supcon_curriculum_epochs": 8,
        "context_replace_prob": 0.35,
        "context_replace_margin": 20,
        "context_replace_feather": 40,
        "cutout_prob": 0.50,
        "cutout_max_frac": 0.16,
        "cutout_n_patches": 3,
        "depth_jitter": 1,
    }


TESTS = [_test(_SCROLLS_BY_ID[scroll_id]) for scroll_id in _HOLDOUT_IDS]


def _count_train_mask(scroll_id: int) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    train_path = TRAIN_MASK_DIR / f"{scroll_id}.png"
    mask_path = MASK_DIR / f"{scroll_id}.png"
    label_path = INKLABEL_DIR / f"{scroll_id}.png"
    zarr_path = ZARR_DIR / f"{scroll_id}.zarr"
    surface_dir = SURFACE_LABEL_DIR / str(scroll_id)
    surface_paths = (surface_dir / "depth.npy", surface_dir / "confidence.npy")
    required = (train_path, mask_path, label_path, zarr_path, *surface_paths)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {}, [f"missing: {', '.join(missing)}"]

    train_mask = imread_gray(str(train_path))
    papyrus_mask = imread_gray(str(mask_path))
    inklabels = imread_gray(str(label_path))
    if train_mask is None or papyrus_mask is None or inklabels is None:
        return {}, ["one or more mask images could not be decoded"]

    volume = zarr.open(str(zarr_path), mode="r")
    common_shape = (
        min(int(volume.shape[1]), int(papyrus_mask.shape[0]), int(inklabels.shape[0])),
        min(int(volume.shape[2]), int(papyrus_mask.shape[1]), int(inklabels.shape[1])),
    )
    if train_mask.shape != common_shape:
        errors.append(
            f"train mask shape {train_mask.shape} != dataloader shape {common_shape}"
        )
        return {}, errors

    papyrus = papyrus_mask[:common_shape[0], :common_shape[1]] > 0
    ink = inklabels[:common_shape[0], :common_shape[1]] > 0
    normal = train_mask >= 192
    explicit_negative = (train_mask >= 112) & (train_mask <= 143)
    invalid_value = (train_mask > 0) & ~normal & ~explicit_negative
    assigned = normal | explicit_negative

    stats = {
        "train_pixels": int(normal.sum()),
        "positive_pixels": int((normal & ink & papyrus).sum()),
        "nonink_pixels": int((normal & ~ink & papyrus).sum()),
        "explicit_negative_pixels": int((explicit_negative & papyrus).sum()),
        "explicit_ink_overlap": int((explicit_negative & ink).sum()),
        "outside_papyrus": int((assigned & ~papyrus).sum()),
        "invalid_value_pixels": int(invalid_value.sum()),
    }
    if stats["train_pixels"] == 0:
        errors.append("contains no full-intensity training pixels")
    if stats["positive_pixels"] == 0:
        errors.append("contains no detected positive ink pixels")
    if stats["nonink_pixels"] == 0:
        errors.append("contains no detected non-ink training pixels")
    if stats["outside_papyrus"]:
        errors.append(f"assigns {stats['outside_papyrus']:,} pixels outside the papyrus mask")
    if stats["invalid_value_pixels"]:
        errors.append(
            f"contains {stats['invalid_value_pixels']:,} nonzero pixels outside the full/half bands"
        )
    depth_shape = np.load(surface_paths[0], mmap_mode="r").shape
    confidence_shape = np.load(surface_paths[1], mmap_mode="r").shape
    volume_shape = tuple(int(value) for value in volume.shape[-2:])
    if depth_shape != volume_shape or confidence_shape != volume_shape:
        errors.append(
            "surface map shape mismatch: "
            f"expected {volume_shape}, depth={depth_shape}, confidence={confidence_shape}"
        )
    return stats, errors


def preflight_train_masks(scrolls=ALL_SCROLLS) -> None:
    """validate all manual masks before any leave-one-out run allocates data."""
    print(f"[preflight] validating {len(scrolls)} train masks", flush=True)
    failures: list[str] = []
    for scroll in scrolls:
        scroll_id = int(scroll.scroll_id)
        stats, errors = _count_train_mask(scroll_id)
        if stats:
            print(
                f"  {scroll_id}: train={stats['train_pixels']:,}"
                f" positive={stats['positive_pixels']:,}"
                f" nonink={stats['nonink_pixels']:,}"
                f" explicit_negative={stats['explicit_negative_pixels']:,}"
                f" explicit_label_overlap={stats['explicit_ink_overlap']:,}",
                flush=True,
            )
        if errors:
            failures.extend(f"{scroll_id}: {message}" for message in errors)
            for message in errors:
                print(f"  {scroll_id}: ERROR {message}", flush=True)
    if failures:
        raise RuntimeError(
            f"campaign-24 train-mask preflight failed ({len(failures)} issue(s)):\n  "
            + "\n  ".join(failures)
        )
    print(
        f"[preflight] all {len(scrolls)} train masks and surface inputs passed",
        flush=True,
    )


def build_config(test: dict):
    config = campaign23_build_config(test)
    holdout_id = int(test["holdout_id"])
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 20
    config.tra.eval_int = 20
    config.tra.fast_eval_figure = False
    config.tra.eval_int_scrolls = 1

    config.tra.dann = True
    config.tra.dann_lambda = 0.03
    config.tra.dann_grl_anneal = False
    config.tra.dann_n_domains = len(test["scrolls"])

    config.data.simple_split = False
    config.data.train_mask_dir = str(TRAIN_MASK_DIR)
    config.data.preload_volumes = True
    config.data.scrolls = list(test["scrolls"])
    config.data.vis_scroll_ids = [holdout_id]
    config.data.character_balance_scrolls = True
    config.data.character_balanced_sampling = True

    checkpoint_dir = os.path.join(MODEL_DIR, f"holdout_{holdout_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def run_test(config, holdout_id: int, dry_run: bool) -> bool:
    train_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    print(f"\n{'=' * 78}\n[campaign24] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  train_scrolls={len(train_ids)} holdout={holdout_id}"
        f" samples/scroll={config.data.max_samples_per_epoch}"
        f" total~={config.data.max_samples_per_epoch * len(train_ids)}",
        flush=True,
    )
    print(
        f"  epochs={config.tra.n_epochs} eval={config.tra.eval_int}"
        f" full_eval={not config.tra.fast_eval_figure}"
        f" DANN=fixed:{config.tra.dann_lambda} domains={config.tra.dann_n_domains}"
        f" xfrag={config.tra.supcon_cross_frag}"
        f" replace_margin={config.dl.context_replace_margin}",
        flush=True,
    )
    if holdout_id in train_ids or len(train_ids) != 17:
        raise RuntimeError("leave-one-out partition is invalid")
    if config.data.vis_scroll_ids != [holdout_id]:
        raise RuntimeError("held-out fragment must be the sole visualization scroll")
    if dry_run:
        print("  [DRY RUN] skipping", flush=True)
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
    parser = argparse.ArgumentParser(
        description="campaign 24: 18-way leave-one-fragment-out training"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preflight_train_masks()

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

    print(f"[campaign24] {len(selected)} run(s) queued (log -> {LOG_DIR})")
    results = {}
    for test in selected:
        config = build_config(test)
        results[test["tid"]] = (
            "OK" if run_test(config, int(test["holdout_id"]), args.dry_run) else "FAIL"
        )
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'=' * 78}\n[campaign24] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
