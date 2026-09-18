"""campaign 28: targeted combinations of Campaign 26/27 winners.

The full-IBN baseline is copied from Campaign 27 and is never queued.

Usage:
    python3 campaign_archs_28.py --dry-run
    python3 campaign_archs_28.py --only gradient_conflict_mid_gated
    python3 campaign_archs_28.py --from cluster_balance_sparse_deep
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

# Config probes CUDA availability before arm isolation. NVML keeps that probe from
# initializing CUDA runtime state that cannot be inherited across os.fork().
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from campaign_archs_24 import preflight_train_masks
from campaign_archs_25 import CAMPAIGN_SCROLLS
from campaign_archs_26 import ROOT, preflight_pretraining
from campaign_archs_27 import build_config as campaign27_build_config

LOG_DIR = "./runs_archs28"
MODEL_DIR = "models/archs28"
BASELINE_RUN = ROOT / "runs_archs28" / "baseline"


def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"28_{tid}",
        "scrolls": CAMPAIGN_SCROLLS,
        "max_samples_per_epoch": 6_667,
        "context_replace_prob": 0.35,
        "context_replace_margin": 20,
        "context_replace_feather": 40,
        "cutout_prob": 0.50,
        "cutout_max_frac": 0.16,
        "cutout_n_patches": 3,
        "depth_jitter": 1,
        "norm_mode": "ibn_full",
    }
    test.update(overrides)
    return test


TESTS = [
    _test(
        "gradient_conflict_mid_gated",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
        gated_stems=True,
        domain_gradient_mode="conflict_weighted",
        domain_gradient_threshold=0.8,
        domain_gradient_strength=8.0,
        compile_model=False,
    ),
    _test(
        "cluster_balance_sparse_deep",
        pretrain_key="full_ibn",
        sparse_deep_supervision=True,
        domain_gradient_mode="cluster_balance",
        domain_gradient_threshold=0.8,
        compile_model=False,
    ),
    _test(
        "gradient_conflict_gated_sparse_deep",
        pretrain_key="full_ibn",
        gated_stems=True,
        sparse_deep_supervision=True,
        domain_gradient_mode="conflict_weighted",
        domain_gradient_threshold=0.8,
        domain_gradient_strength=8.0,
        compile_model=False,
    ),
]


def build_config(test: dict):
    config = campaign27_build_config(test)
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 9
    config.tra.eval_int = 999
    config.tra.test_int = 9_999
    config.tra.probe_int = 9_999

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def run_test(config, dry_run: bool) -> bool:
    print(f"\n{'=' * 78}\n[campaign28] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  norm={config.model.norm_mode} mid_2d={config.model.mid_2d_unet} "
        f"gated_stems={config.model.gated_stems} "
        f"dual_scale={config.model.dual_scale}:mix{config.model.dual_scale_mix} "
        f"deep_supervision={config.model.sparse_deep_supervision}",
        flush=True,
    )
    print(
        f"  domain_gradient={config.tra.domain_gradient_mode}:"
        f"threshold={config.tra.domain_gradient_threshold}:"
        f"strength={config.tra.domain_gradient_strength}",
        flush=True,
    )
    print(f"  pretrained={config.init_weights}", flush=True)
    if dry_run:
        print("  [DRY RUN] skipping", flush=True)
        return True

    from train import Trainer
    from utils.dataloader import cleanup_mmap_files

    trainer = None
    try:
        trainer = Trainer(config)
        trainer.vis.writer.add_scalar("Run/Initialized", 1.0, 0)
        trainer.vis.writer.flush()
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return False
    finally:
        if trainer is not None:
            trainer.close()
        del trainer
        gc.collect()
        cleanup_mmap_files()


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def prewarm_data_cache(config) -> None:
    """Populate immutable dataset/chunk caches before any child initializes CUDA."""
    from train import Trainer
    from utils.chunk_cache import _CACHE_REGISTRY
    from utils.dataloader import _PREPARED_DATASET_CACHE

    if torch.cuda.is_initialized():
        raise RuntimeError("Campaign 28 controller initialized CUDA before cache warmup")
    warmup = Trainer.__new__(Trainer)
    warmup.c = config
    train_dataset = train_loader = valid_loader = None
    try:
        train_dataset, train_loader, valid_loader = warmup._setup_data()
    finally:
        del train_dataset, train_loader, valid_loader, warmup
        gc.collect()
    if not _PREPARED_DATASET_CACHE or not _CACHE_REGISTRY:
        raise RuntimeError("Campaign 28 RAM cache warmup did not populate global caches")
    if torch.cuda.is_initialized():
        raise RuntimeError("Campaign 28 cache warmup unexpectedly initialized CUDA")
    cached_gib = sum(volume.cached_nbytes for volume in _CACHE_REGISTRY.values()) / 1024**3
    print(
        f"[campaign28] parent RAM cache ready: datasets={len(_PREPARED_DATASET_CACHE)} "
        f"volumes={len(_CACHE_REGISTRY)} ram={cached_gib:.2f}GiB "
        f"controller_fds={_open_fd_count()}",
        flush=True,
    )


def run_test_isolated(config) -> bool:
    """Run one arm in a child so all DataLoader/resource-sharer FDs die with it."""
    if not hasattr(os, "fork"):
        return run_test(config, False)
    before = _open_fd_count()
    pid = os.fork()
    if pid == 0:
        try:
            success = run_test(config, False)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if success else 1)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    after = _open_fd_count()
    print(
        f"[campaign28] isolated arm pid={pid} status={status} "
        f"controller_fds={before}->{after}",
        flush=True,
    )
    if before >= 0 and after > before + 4:
        raise RuntimeError(f"Campaign 28 controller leaked file descriptors: {before}->{after}")
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="campaign 28: targeted combinations of Campaign 26/27 winners"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not BASELINE_RUN.is_dir():
        raise FileNotFoundError(f"Campaign 28 baseline copy is missing: {BASELINE_RUN}")
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

    print(f"[campaign28] external baseline (not queued): {BASELINE_RUN}")
    preflight_train_masks(CAMPAIGN_SCROLLS)
    preflight_pretraining(selected, args.dry_run)
    print(f"[campaign28] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    if not args.dry_run and selected:
        prewarm_data_cache(build_config(selected[0]))

    results = {}
    for test in selected:
        config = build_config(test)
        success = run_test(config, True) if args.dry_run else run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()

    print(f"\n{'=' * 78}\n[campaign28] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
