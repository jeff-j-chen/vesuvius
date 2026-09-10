"""campaign_archs_21.py -- isolate non-w013 data effects with the campaign-20 baseline

Runs the soft-BCE campaign-20 operating point on:
- 500P2_front only
- w044 only
- 500P2_front + w044 with equal scroll sampling

  python campaign_archs_21.py --dry-run
  python campaign_archs_21.py --only 500p2
  python campaign_archs_21.py --from w044
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


TESTS = [
    {
        "tid": "500p2",
        "tag": "21_500p2",
        "scrolls": _scrolls("500p2"),
        "max_samples_per_epoch": 20_000,
        "fast_eval_figure": True,
    },
    {
        "tid": "w044",
        "tag": "21_w044",
        "scrolls": _scrolls("w044"),
        "max_samples_per_epoch": 20_000,
        "fast_eval_figure": True,
    },
    {
        "tid": "multi2",
        "tag": "21_multi2",
        "scrolls": _scrolls("500p2", "w044"),
        # the cap is per child, keeping the merged epoch near 20k samples
        "max_samples_per_epoch": 10_000,
        "fast_eval_figure": True,
    },
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

    os.makedirs("models/archs21", exist_ok=True)
    config.save_final = f"models/archs21/{test['tid']}_final.pth"
    return config


def run_test(config, dry_run: bool) -> bool:
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    print(f"\n{'=' * 70}\n[archs21] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  scrolls={scroll_ids} samples/scroll={config.data.max_samples_per_epoch}"
        f" total_samples~={config.data.max_samples_per_epoch * len(scroll_ids)}"
    )
    print(
        f"  batch={config.dl.batch_size} lr={config.tra.lr:.2e}"
        f" context={config.data.context_size}/ds{config.data.context_downsample}"
        f" center={config.model.multitile_subtile * config.model.multitile_grid}"
        f" loss={config.tra.loss_type}"
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
