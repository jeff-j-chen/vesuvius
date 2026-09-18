"""campaign 27: full-IBN scale and physical-domain compatibility study.

The baseline is copied from Campaign 26 and is never queued by this runner.

Usage:
    python3 campaign_archs_27.py --dry-run
    python3 campaign_archs_27.py --only weldon_top_bottom4
    python3 campaign_archs_27.py --from mldg_rotating_beta_0p25
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

from campaign_archs_24 import preflight_train_masks
from campaign_archs_25 import CAMPAIGN_SCROLLS
from campaign_archs_26 import (
    ROOT,
    build_config as campaign26_build_config,
    preflight_pretraining,
)

LOG_DIR = "./runs_archs27"
MODEL_DIR = "models/archs27"
BASELINE_RUN = ROOT / "runs_archs27" / "26_pure_ibn_17_07-19-08"


def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"27_{tid}",
        "scrolls": CAMPAIGN_SCROLLS,
        "max_samples_per_epoch": 6_667,
        "context_replace_prob": 0.35,
        "context_replace_margin": 20,
        "context_replace_feather": 40,
        "cutout_prob": 0.50,
        "cutout_max_frac": 0.16,
        "cutout_n_patches": 3,
        "depth_jitter": 1,
        "pretrain_key": "full_ibn",
        "norm_mode": "ibn_full",
    }
    test.update(overrides)
    return test


TESTS = [
    _test("weldon_top_bottom4", weldon_k=4),
    _test(
        "ibn_triple_scale_local64_128",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        dual_scale_outer_size=128,
        dual_scale_outer_mix=0.50,
    ),
    _test(
        "dual_scale_mix_0p50",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "weldon_dual_scale_mix_0p50",
        weldon_k=4,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "mixstyle_amp_fixed",
        mixstyle=True,
        mixstyle_prob=0.7,
        mixstyle_alpha=0.2,
        compile_model=False,
    ),
    _test(
        "sagnet_amp_fixed",
        sagnet=True,
        sagnet_lambda=0.3,
        compile_model=False,
    ),
    _test(
        "mid_3d2d_full_ibn",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
    ),
    _test(
        "mid_3d2d_gated_stems",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
        gated_stems=True,
    ),
    _test(
        "mid_3d2d_dual_scale",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "adaptive_dual_scale_router",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        dual_scale_adaptive_gate=True,
        dual_scale_gate_max=1.0,
    ),
    _test(
        "mldg_rotating_beta_0p25",
        mldg=True,
        mldg_random_holdout=True,
        mldg_inner_lr=5e-4,
        mldg_beta=0.25,
        compile_model=False,
    ),
    _test(
        "mldg_rotating_dual_scale_beta_0p25",
        mldg=True,
        mldg_random_holdout=True,
        mldg_inner_lr=5e-4,
        mldg_beta=0.25,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        compile_model=False,
    ),
    _test(
        "gradient_cluster_balance",
        domain_gradient_mode="cluster_balance",
        domain_gradient_threshold=0.8,
        compile_model=False,
    ),
    _test(
        "gradient_cluster_worst",
        domain_gradient_mode="cluster_worst",
        domain_gradient_threshold=0.8,
        compile_model=False,
    ),
    _test(
        "gradient_conflict_weighted",
        domain_gradient_mode="conflict_weighted",
        domain_gradient_threshold=0.8,
        domain_gradient_strength=8.0,
        compile_model=False,
    ),
    _test(
        "gradient_conflict_dual_scale",
        domain_gradient_mode="conflict_weighted",
        domain_gradient_threshold=0.8,
        domain_gradient_strength=8.0,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        compile_model=False,
    ),
    _test(
        "gated_stems_dual_scale",
        gated_stems=True,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "physical_groupdro_dual_scale",
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "mid_3d2d_physical_groupdro",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
    ),
    _test(
        "stack_weldon_gated_dual_groupdro",
        weldon_k=4,
        gated_stems=True,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
    ),
    _test(
        "stack_weldon_gated_dual_gradient_conflict",
        weldon_k=4,
        gated_stems=True,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        domain_gradient_mode="conflict_weighted",
        domain_gradient_threshold=0.8,
        domain_gradient_strength=8.0,
        compile_model=False,
    ),
    _test(
        "stack_mid_gated_dual_groupdro",
        pretrain_key="mid_3d2d_full_ibn",
        mid_2d_unet=True,
        gated_stems=True,
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
    ),
]


def build_config(test: dict):
    config = campaign26_build_config(test)
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 9
    config.tra.eval_int = 999
    config.tra.test_int = 9_999
    config.tra.probe_int = 9_999
    config.model.weldon_k = int(test.get("weldon_k", 0))
    config.tra.domain_gradient_mode = str(test.get("domain_gradient_mode", ""))
    config.tra.domain_gradient_threshold = float(
        test.get("domain_gradient_threshold", 0.0)
    )
    config.tra.domain_gradient_strength = float(
        test.get("domain_gradient_strength", 4.0)
    )
    config.tra.domain_gradient_ema = float(test.get("domain_gradient_ema", 0.9))

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def run_test(config, dry_run: bool) -> bool:
    print(f"\n{'=' * 78}\n[campaign27] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  norm={config.model.norm_mode} weldon_k={config.model.weldon_k} "
        f"dual_scale={config.model.dual_scale}:local{config.model.dual_scale_local_size}:"
        f"mix{config.model.dual_scale_mix}:adaptive={config.model.dual_scale_adaptive_gate}",
        flush=True,
    )
    print(
        f"  mldg={config.tra.mldg}:random={config.tra.mldg_random_holdout}:"
        f"beta={config.tra.mldg_beta} "
        f"domain_gradient={config.tra.domain_gradient_mode}:"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="campaign 27: full-IBN scale and domain compatibility study"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not BASELINE_RUN.is_dir():
        raise FileNotFoundError(
            f"Campaign 27 baseline copy is missing: {BASELINE_RUN}"
        )
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

    print(f"[campaign27] external baseline (not queued): {BASELINE_RUN}")
    preflight_train_masks(CAMPAIGN_SCROLLS)
    preflight_pretraining(selected, args.dry_run)
    print(f"[campaign27] {len(selected)} run(s) queued (log -> {LOG_DIR})")

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

    print(f"\n{'=' * 78}\n[campaign27] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
