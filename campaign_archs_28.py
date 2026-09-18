"""campaign 28: expanded-data attribution study of Campaign 27 winners.

The first arm trains a new full-IBN baseline on the expanded scroll set.

Usage:
    python3 campaign_archs_28.py --dry-run
    python3 campaign_archs_28.py --only baseline
    python3 campaign_archs_28.py --from gradient_conflict_dual
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
from campaign_archs_26 import preflight_pretraining
from campaign_archs_27 import build_config as campaign27_build_config
from utils.config import DEFAULT_SCROLLS

LOG_DIR = "./runs_archs28"
MODEL_DIR = "models/archs28"
W044_SCROLL_ID = 20260115000000
CAMPAIGN28_SCROLL_DICT = {
    "pherc0139": [20260115000000, 20260317000000, 20250223000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc1667": [20240304141531, 20240304144031],
    "pherc0009b": [20250919125754],
    "phercparis4": [20231210121321],
    "pherc0500p2": [20250628074500],
    "pherc0814": [20260226000000],
    "phercparis2_fr143": [20230301213755],
    "pherc51cr4_fr8": [20231205222200],
    "phercparis1_fr34": [20230301213423],
}
CAMPAIGN28_SCROLL_WEIGHTS = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
CAMPAIGN28_SCROLL_IDS = tuple(
    scroll_id
    for scroll_ids in CAMPAIGN28_SCROLL_DICT.values()
    for scroll_id in scroll_ids
)
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}
_missing = set(CAMPAIGN28_SCROLL_IDS) - set(_SCROLLS_BY_ID)
if _missing:
    raise RuntimeError(f"campaign-28 scroll definitions missing: {sorted(_missing)}")
CAMPAIGN28_SCROLLS = [_SCROLLS_BY_ID[scroll_id] for scroll_id in CAMPAIGN28_SCROLL_IDS]

CAMPAIGN28_SETTINGS = {
    "max_samples_per_epoch": 6_667,
    "context_replace_prob": 0.35,
    "context_replace_margin": 20,
    "context_replace_feather": 40,
    "cutout_prob": 0.50,
    "cutout_max_frac": 0.16,
    "cutout_n_patches": 3,
    "depth_jitter": 1,
    "norm_mode": "ibn_full",
    "num_workers": 24,
    "prefetch_factor": 4,
    "simple_split": False,
    "preload_volumes": False,
    "selective_chunk_preload": True,
    "selective_chunk_workers": 8,
    "ram_safe_vis": False,
    "mask_memmap": os.name == "nt",
    "mask_bitpack": True,
    "train_mask_dir": "./train_masks",
    "character_balance_scrolls": True,
    "character_balanced_sampling": True,
    "inklabel_dir": "./inklabels",
    "label_dilate_r": 0,
    "ring_negatives": True,
    "ring_label_source": "closed",
    "ring_close_r": 2,
    "ring_gap_r": 2,
    "ring_shell_r": 4,
    "multitile_pos_only": True,
    "n_epochs": 9,
    "eval_int": 999,
    "eval_int_scrolls": 1,
    "test_int": 9_999,
    "probe_int": 9_999,
    "test_on_final": False,
    "fast_eval_figure": False,
    "dann": False,
    "dann_lambda": 0.0,
    "dann_grl_anneal": False,
    "supcon": False,
    "supcon_cross_frag": False,
    "per_scroll_metrics": True,
}


def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"28_{tid}",
        "scrolls": CAMPAIGN28_SCROLLS,
        **{
            name: CAMPAIGN28_SETTINGS[name]
            for name in (
                "max_samples_per_epoch",
                "context_replace_prob",
                "context_replace_margin",
                "context_replace_feather",
                "cutout_prob",
                "cutout_max_frac",
                "cutout_n_patches",
                "depth_jitter",
                "norm_mode",
            )
        },
    }
    test.update(overrides)
    return test


def _combo(tid: str, *components: dict, **overrides):
    options = {}
    for component in components:
        options.update(component)
    options.update(overrides)
    return _test(tid, **options)


WELDON = {"weldon_k": 4}
GATED = {"gated_stems": True}
DUAL = {
    "dual_scale": True,
    "dual_scale_local_size": 64,
    "dual_scale_mix": 0.50,
}
GROUPDRO = {
    "physical_domain_groupdro": True,
    "physical_domain_groupdro_eta": 0.05,
    "physical_domain_groupdro_max_ratio": 3.0,
}
GRADIENT_CONFLICT = {
    "domain_gradient_mode": "conflict_weighted",
    "domain_gradient_threshold": 0.8,
    "domain_gradient_strength": 8.0,
    "compile_model": False,
}
CLUSTER_BALANCE = {
    "domain_gradient_mode": "cluster_balance",
    "domain_gradient_threshold": 0.8,
    "compile_model": False,
}
MID_3D2D = {
    "pretrain_key": "mid_3d2d_full_ibn",
    "mid_2d_unet": True,
}
SPARSE_DEEP = {"sparse_deep_supervision": True}


TESTS = [
    # Expanded-data baseline and single-component effects.
    _test("baseline", pretrain_key="full_ibn"),
    _combo("weldon", WELDON, pretrain_key="full_ibn"),
    _combo("gated_stems", GATED, pretrain_key="full_ibn"),
    _combo("dual_scale", DUAL, pretrain_key="full_ibn"),
    _combo("physical_groupdro", GROUPDRO, pretrain_key="full_ibn"),
    _combo("gradient_conflict", GRADIENT_CONFLICT, pretrain_key="full_ibn"),
    _combo("gradient_cluster_balance", CLUSTER_BALANCE, pretrain_key="full_ibn"),

    # Attribute the WELDON + gated + dual-scale + GroupDRO winner.
    _combo("weldon_dual", WELDON, DUAL, pretrain_key="full_ibn"),
    _combo("gated_dual", GATED, DUAL, pretrain_key="full_ibn"),
    _combo("groupdro_dual", GROUPDRO, DUAL, pretrain_key="full_ibn"),
    _combo("weldon_gated_dual", WELDON, GATED, DUAL, pretrain_key="full_ibn"),
    _combo("gated_dual_groupdro", GATED, DUAL, GROUPDRO, pretrain_key="full_ibn"),
    _combo("weldon_dual_groupdro", WELDON, DUAL, GROUPDRO, pretrain_key="full_ibn"),
    _combo("weldon_gated_groupdro", WELDON, GATED, GROUPDRO, pretrain_key="full_ibn"),
    _combo(
        "weldon_gated_dual_groupdro",
        WELDON,
        GATED,
        DUAL,
        GROUPDRO,
        pretrain_key="full_ibn",
    ),

    # Compare robust objectives at identical architecture capacities.
    _combo("gradient_conflict_dual", GRADIENT_CONFLICT, DUAL, pretrain_key="full_ibn"),
    _combo(
        "gradient_conflict_gated_dual",
        GRADIENT_CONFLICT,
        GATED,
        DUAL,
        pretrain_key="full_ibn",
    ),
    _combo(
        "gradient_conflict_weldon_gated_dual",
        GRADIENT_CONFLICT,
        WELDON,
        GATED,
        DUAL,
        pretrain_key="full_ibn",
    ),
    _combo("cluster_balance_dual", CLUSTER_BALANCE, DUAL, pretrain_key="full_ibn"),
    _combo(
        "cluster_balance_gated_dual",
        CLUSTER_BALANCE,
        GATED,
        DUAL,
        pretrain_key="full_ibn",
    ),
    _combo(
        "cluster_balance_weldon_gated_dual",
        CLUSTER_BALANCE,
        WELDON,
        GATED,
        DUAL,
        pretrain_key="full_ibn",
    ),

    # Resolve the mid-3D/2D architecture and GroupDRO interaction.
    _combo("mid_3d2d", MID_3D2D),
    _combo("mid_3d2d_gated", MID_3D2D, GATED),
    _combo("mid_3d2d_groupdro", MID_3D2D, GROUPDRO),
    _combo("mid_3d2d_gated_groupdro", MID_3D2D, GATED, GROUPDRO),
    _combo("mid_3d2d_gated_dual_groupdro", MID_3D2D, GATED, DUAL, GROUPDRO),

    # Lower-priority checks for the original sparse-supervision hypothesis.
    _combo("sparse_deep", SPARSE_DEEP, pretrain_key="full_ibn"),
    _combo(
        "gradient_conflict_sparse_deep",
        GRADIENT_CONFLICT,
        SPARSE_DEEP,
        pretrain_key="full_ibn",
    ),
    _combo(
        "gradient_conflict_gated_sparse_deep",
        GRADIENT_CONFLICT,
        GATED,
        SPARSE_DEEP,
        pretrain_key="full_ibn",
    ),
]


def build_config(test: dict):
    config = campaign27_build_config(test)
    config.tra.log_dir = LOG_DIR
    for name in (
        "n_epochs", "eval_int", "eval_int_scrolls", "test_int", "probe_int",
        "test_on_final", "fast_eval_figure", "dann", "dann_lambda",
        "dann_grl_anneal", "supcon", "supcon_cross_frag", "per_scroll_metrics",
    ):
        setattr(config.tra, name, CAMPAIGN28_SETTINGS[name])
    for name in ("num_workers", "prefetch_factor"):
        setattr(config.dl, name, CAMPAIGN28_SETTINGS[name])
    for name in (
        "simple_split", "preload_volumes", "selective_chunk_preload",
        "selective_chunk_workers", "ram_safe_vis", "mask_memmap", "mask_bitpack",
        "train_mask_dir", "character_balance_scrolls", "character_balanced_sampling",
        "max_samples_per_epoch", "inklabel_dir", "label_dilate_r", "ring_negatives",
        "ring_label_source", "ring_close_r", "ring_gap_r", "ring_shell_r",
        "multitile_pos_only",
    ):
        setattr(config.data, name, CAMPAIGN28_SETTINGS[name])
    config.data.scrolls = list(CAMPAIGN28_SCROLLS)
    config.data.vis_scroll_ids = [W044_SCROLL_ID]
    config.data.train_scroll_dict = {
        name: list(scroll_ids) for name, scroll_ids in CAMPAIGN28_SCROLL_DICT.items()
    }
    config.data.train_scroll_weights = list(CAMPAIGN28_SCROLL_WEIGHTS)
    config.tra.dann_n_domains = len(CAMPAIGN28_SCROLL_DICT)
    config.model.norm_mode = str(test.get("norm_mode", CAMPAIGN28_SETTINGS["norm_mode"]))
    config.model.use_ibn = config.model.norm_mode == "ibn"

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
        description="campaign 28: expanded-data attribution study of Campaign 27 winners"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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

    preflight_train_masks(
        CAMPAIGN28_SCROLLS,
        inklabel_dir=Path(CAMPAIGN28_SETTINGS["inklabel_dir"]),
    )
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
