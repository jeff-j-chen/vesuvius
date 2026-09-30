"""campaign_finetune_96.py -- fine-tune the archs40 native-96 combined-surface model.

This starts from the campaign-40 `holdout_n96_combined_surface_norm` checkpoint and keeps the
same training fragments as archs40, then adds two new physical domains:

- PHerc0211: 20260928000003 (train-only, no validation split)
- PHerc1447: 20260930144758 / 20260930144760

Unlike the archs40 campaigns, this runner keeps the repository train masks
(`simple_split=False`). That lets PHerc0211 stay train-only via `train_only_scroll_ids`
without producing validation metrics for that scroll.

`--fewer` restricts training to the four-fragment bootstrap subset:

- PHerc0211: 20260928000003
- PHerc1447: 20260930144758 / 20260930144760
- PHerc0841: 20260221022814

Usage:
    python3 campaign_finetune_96.py --dry-run
    python3 campaign_finetune_96.py --fewer
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_40 as campaign40
from utils.config import ScrollConfig, startup_output
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_finetune"
MODEL_DIR = "models/finetune"
INKLABEL_DIR = "./dilated_inklabels"
DEFAULT_INIT_WEIGHTS = "/vesuvius/models/archs40/holdout_n96_combined_surface_norm/final.pth"
PHERC0211_ID = 20260928000003
PHERC1447_IDS = (20260930144758, 20260930144760)
PHERC0841_ID = 20260221022814
PHERC1447_DOMAIN = "pherc1447"
DEFAULT_VIS_SCROLL_IDS = [PHERC0211_ID, PHERC1447_IDS[1], 20260115000000]
FEWER_VIS_SCROLL_IDS = [PHERC0211_ID, PHERC1447_IDS[1], PHERC0841_ID]

DEFAULT_TRAIN_SCROLL_DICT = {
    **campaign40.BASE_SCROLL_DICT,
    campaign40.NEW_DOMAIN: [PHERC0211_ID],
    PHERC1447_DOMAIN: list(PHERC1447_IDS),
}
DEFAULT_SCROLLS = [
    *campaign40.SCROLLS_BASE,
    ScrollConfig(PHERC0211_ID, split_axis="x", train_split_frac=1.0),
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
]

FEWER_TRAIN_SCROLL_DICT = {
    campaign40.NEW_DOMAIN: [PHERC0211_ID],
    PHERC1447_DOMAIN: list(PHERC1447_IDS),
    "pherc0841": [PHERC0841_ID],
}
FEWER_SCROLLS = [
    ScrollConfig(PHERC0211_ID, split_axis="x", train_split_frac=1.0),
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC0841_ID, split_axis="x", train_split_frac=0.75),
]


def _test(init_weights: str, fewer: bool) -> dict:
    train_scroll_dict = FEWER_TRAIN_SCROLL_DICT if fewer else DEFAULT_TRAIN_SCROLL_DICT
    scrolls = FEWER_SCROLLS if fewer else DEFAULT_SCROLLS
    test = campaign40._test(
        "holdout_n96_combined_surface_norm",
        campaign40.SURFACE_FIBER_MAE_KEY,
        train_scroll_dict,
        scrolls,
        init_weights=init_weights,
        **campaign40.SCRATCH_TRAINING,
    )
    test["tag"] = "finetune_holdout_n96_combined_surface_norm_fewer" if fewer else "finetune_holdout_n96_combined_surface_norm"
    return test


def _selected_scrolls(selected: list[dict]) -> list[ScrollConfig]:
    ordered: dict[int, ScrollConfig] = {}
    for test in selected:
        for scroll in test["scrolls"]:
            ordered[int(scroll.scroll_id)] = scroll
    return list(ordered.values())


def _surface_anchor_ids(selected: list[dict]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(
        [int(scroll.scroll_id) for scroll in _selected_scrolls(selected)]
        + [int(scroll_id) for test in selected for scroll_id in test.get("vis_scroll_ids", [])]
    ))


@contextlib.contextmanager
def _campaign40_paths():
    saved = campaign40.LOG_DIR, campaign40.MODEL_DIR
    campaign40.LOG_DIR, campaign40.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign40.LOG_DIR, campaign40.MODEL_DIR = saved


def build_config(test: dict):
    with _campaign40_paths():
        config = campaign40.build_config(test)
    config.data.scrolls = list(test["scrolls"])
    config.data.train_scroll_dict = {
        domain: list(ids) for domain, ids in test["train_scroll_dict"].items()
    }
    config.data.train_scroll_weights = campaign40._train_scroll_weights(config.data.train_scroll_dict)
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.inklabel_dir = INKLABEL_DIR
    config.data.vis_scroll_ids = list(test["vis_scroll_ids"])
    config.data.vis_preload_persistent = True
    config.data.train_only_scroll_ids = [PHERC0211_ID]
    config.tra.n_epochs = 5
    config.tra.eval_int = 5
    config.tra.eval_int_scrolls = len(test["vis_scroll_ids"])
    config.tra.test_int = 999
    config.tra.fast_eval_figure = True
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    return config


def preflight_inputs(selected: list[dict], dry_run: bool) -> None:
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    failures = []
    for scroll in _selected_scrolls(selected):
        scroll_id = int(scroll.scroll_id)
        required = (
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / INKLABEL_DIR.strip("./") / f"{scroll_id}.png",
            ROOT / "train_masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            failures.append(f"{scroll_id}: {', '.join(missing)}")
    external = []
    for test in selected:
        for path in campaign40._external_files(test):
            if not (ROOT / path).is_file() and not Path(path).is_file():
                external.append(path)
    if failures or external:
        message = []
        if failures:
            message.append("fine-tune inputs missing:\n  " + "\n  ".join(failures))
        if external:
            message.append(f"external files missing: {sorted(set(external))}")
        text = "\n".join(message)
        if not dry_run:
            raise FileNotFoundError(text)
        print(f"[campaign_finetune] WARNING {text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="fine-tune archs40 native-96 combined-surface model with PHerc0211 + PHerc1447")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--fewer", action="store_true",
                        help="restrict training to PHerc0211, PHerc1447 w058/w060, and PHerc0841 auto-grown 405")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help="starting checkpoint (default: archs40 holdout_n96_combined_surface_norm final.pth)")
    args = parser.parse_args()

    selected = [_test(args.init_weights, args.fewer)]
    for test in selected:
        test["vis_scroll_ids"] = list(FEWER_VIS_SCROLL_IDS if args.fewer else DEFAULT_VIS_SCROLL_IDS)
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in selected if test["tid"] in wanted]
        missing = wanted - {test["tid"] for test in selected}
        if missing:
            raise ValueError(f"unknown test ids: {sorted(missing)}")

    with startup_output():
        preflight_inputs(selected, args.dry_run)
        if not args.dry_run:
            ensure_surface_anchors(_surface_anchor_ids(selected), str(ROOT / "ves_zarrs2"))
        print(f"[campaign_finetune] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign_finetune] {test['tid']}: ctx={config.data.context_size} "
                f"domains={list(config.data.train_scroll_dict)} "
                f"weights={config.data.train_scroll_weights} simple_split={config.data.simple_split} "
                f"fast_eval={config.tra.fast_eval_figure} vis={config.data.vis_scroll_ids} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"fewer={args.fewer} init={config.init_weights}",
                flush=True,
            )
        if args.dry_run:
            success = campaign31.run_test(config, True)
        else:
            campaign29.prewarm_data_cache(config)
            success = campaign31.run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        del config
        gc.collect()

    print(f"\n{'=' * 78}\n[campaign_finetune] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()