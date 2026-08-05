"""throughput sweep for the archived arch baseline on linux

clones the exact resolved config from a known-good runs_archs baseline, then only
overrides the short-run harness and throughput-sensitive knobs so batch / lr /
num_workers can be compared apples-to-apples on new hardware.

  python campaign_arch_throughput.py --dry-run
  python campaign_arch_throughput.py
  python campaign_arch_throughput.py --only tp0base,tp2bs128sqrt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from train import Trainer
from utils.config import Config, ProbeROI, ScrollConfig

BASE_RUN_CONFIG = Path(
    "runs_archs/cmp_archs_c0base_ctx48_baseline_arch_closed_31_08-39-48/config.json"
)
LOG_DIR = "./runs_archs"
INTER_RUN_COOLDOWN_SECS = 60
N_EP = 5
DISABLED_INTERVAL = 999


def _apply_dict(obj, data):
    for key, value in data.items():
        current = getattr(obj, key, None)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        elif key == "scrolls" and isinstance(value, list):
            obj.scrolls = [ScrollConfig(**item) for item in value]
        elif key == "probe_rois" and isinstance(value, dict):
            obj.probe_rois = {
                int(scroll_id): [ProbeROI(**roi) for roi in rois]
                for scroll_id, rois in value.items()
            }
        else:
            setattr(obj, key, value)


def _resolve_zarr_path() -> str:
    return os.getenv("VESUVIUS_ZARR_PATH", Config().data.zarr_path)


def _base_config(exp_name: str) -> Config:
    raw = json.loads(BASE_RUN_CONFIG.read_text(encoding="utf-8"))
    config = Config()
    _apply_dict(config, raw)

    config.exp_name = exp_name
    config.data.zarr_path = _resolve_zarr_path()
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = N_EP
    config.tra.eval_int = DISABLED_INTERVAL
    config.tra.test_int = DISABLED_INTERVAL
    config.tra.probe_int = DISABLED_INTERVAL
    config.tra.save_int = DISABLED_INTERVAL
    config.tra.epoch_cooldown_secs = 0
    config.tra.val_cooldown_secs = 0
    config.tra.eval_cooldown_secs = 0
    config.tra.fig_chunk_cooldown_ms = 0
    config.tra.save_vis = False
    config.tra.test_on_final = False

    return config


def _mk(tid: str, tag: str, *, batch_size: int, num_workers: int, lr: float):
    return {
        "tid": tid,
        "tag": tag,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
    }


TESTS = [
    _mk("tp0base", "ctx48_baseline_b64_w8_lr1e4_e5", batch_size=64, num_workers=8, lr=1.0e-4),
    _mk("tp1bs96", "ctx48_tp_b96_w12_lr15e5_e5", batch_size=96, num_workers=12, lr=1.5e-4),
    _mk("tp2bs128sqrt", "ctx48_tp_b128_w16_lr14e5_e5", batch_size=128, num_workers=16, lr=1.41421356e-4),
    _mk("tp3bs128lin", "ctx48_tp_b128_w16_lr2e4_e5", batch_size=128, num_workers=16, lr=2.0e-4),
]


def build_config(test: dict) -> Config:
    exp_name = f"cmp_archs_{test['tid']}_{test['tag']}"
    config = _base_config(exp_name)
    config.dl.batch_size = int(test["batch_size"])
    config.dl.num_workers = int(test["num_workers"])
    config.tra.lr = float(test["lr"])
    config.save_final = f"models/archs_throughput/{test['tid']}_{test['tag']}_final.pth"
    return config


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(config: Config, dry_run: bool) -> bool:
    print(f"\n{'=' * 70}\n[arch-tp] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  batch={config.dl.batch_size} workers={config.dl.num_workers} lr={config.tra.lr:.8f}"
    )
    print(
        f"  arch={config.model.arch} ctx={config.data.context_size} ds={config.data.context_downsample}"
    )
    print(
        f"  source={BASE_RUN_CONFIG}  zarr_path={config.data.zarr_path}"
    )
    if dry_run:
        print("  [DRY RUN] skipping")
        return True

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


def main():
    parser = argparse.ArgumentParser(description="campaign_arch_throughput: archived arch baseline throughput sweep")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not BASE_RUN_CONFIG.exists():
        raise FileNotFoundError(f"baseline config not found: {BASE_RUN_CONFIG}")

    selected = TESTS
    if args.only:
        want = {item.strip() for item in args.only.split(",") if item.strip()}
        selected = [test for test in TESTS if test["tid"] in want]
        missing = want - {test["tid"] for test in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: {[t['tid'] for t in TESTS]}")
            return

    print(f"[arch-tp] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[arch-tp] cloned baseline config -> {BASE_RUN_CONFIG}")

    results = {}
    for index, test in enumerate(selected):
        config = build_config(test)
        ok = run_test(config, args.dry_run)
        results[test["tid"]] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        if index < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {test['tid']}")

    print(f"\n{'=' * 70}\n[arch-tp] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        print(f"  {tid:12s}  {status}")


if __name__ == "__main__":
    main()