"""campaign 25: physically grouped multi-scroll baseline over 11 fragments.

The campaign combines five newly assembled fragments with six established anchors.
Sampling is round-robin by physical scroll, with PHerc0139 repeated four times per
cycle and every other physical scroll sampled once.

Usage:
    python3 campaign_archs_25.py --dry-run
    python3 campaign_archs_25.py --only minimum_support_top2
    python3 campaign_archs_25.py --from sam_rho_0p01
    python3 campaign_archs_25.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from campaign_archs_23 import build_config as campaign23_build_config
from campaign_archs_24 import preflight_train_masks
from utils.config import DEFAULT_SCROLLS, DEFAULT_TEST_SCROLL_IDS

LOG_DIR = "./runs_archs25"
MODEL_DIR = "models/archs25"
ROOT = Path(__file__).resolve().parent
PRETRAIN_STEPS = 2_000
PRETRAIN_SEED_PATH = ROOT / "models" / "mae_nnunet_192_ibn_depth8_22scroll_2k.pth"
PRETRAIN_SCROLL_IDS = tuple(
    dict.fromkeys(
        [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
        + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
    )
)

PRETRAIN_SPECS = {
    "base": (),
    "fiber": ("--fiber-coordinate-branch",),
    "early_3d2d": ("--early-2d-unet",),
    "divided_depth": ("--divided-attention",),
    "divided_space_depth": ("--divided-attention", "--divided-attention-spatial"),
}

PRETRAIN_REQUIRED_PREFIXES = {
    "base": (),
    "fiber": ("fiber_coordinate_input.",),
    "early_3d2d": ("early_depth_attn.", "early_depth_fuse.", "early2d_"),
    "divided_depth": ("divided_attention.",),
    "divided_space_depth": ("divided_attention.",),
}


def _expected_pretrain_key(test: dict) -> str:
    divided = bool(test.get("divided_attention", False))
    spatial = bool(test.get("divided_attention_spatial", False))
    if divided:
        return "divided_space_depth" if spatial else "divided_depth"
    if bool(test.get("early_2d_unet", False)):
        return "early_3d2d"
    if bool(test.get("fiber_coordinate_branch", False)):
        return "fiber"
    return "base"


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_ibn_depth8_campaign25_{key}_2k"


def _pretrain_path(key: str) -> Path:
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"


def _pretrain_marker(key: str) -> Path:
    return _pretrain_path(key).with_suffix(".complete.json")


def _pretraining_complete(key: str) -> bool:
    checkpoint = _pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    metadata_valid = (
        metadata.get("key") == key
        and metadata.get("steps") == PRETRAIN_STEPS
        and metadata.get("scroll_ids") == list(PRETRAIN_SCROLL_IDS)
        and metadata.get("architecture_args") == list(PRETRAIN_SPECS[key])
        and metadata.get("freeze_loaded_backbone") == (key != "base")
        and metadata.get("init_weights") == str(
            (_pretrain_path("base") if key != "base" else PRETRAIN_SEED_PATH).relative_to(ROOT)
        )
        and metadata.get("checkpoint") == str(checkpoint.relative_to(ROOT))
    )
    if not metadata_valid:
        return False
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return all(
        any(state_key.startswith(prefix) for state_key in state)
        for prefix in PRETRAIN_REQUIRED_PREFIXES[key]
    )

TRAIN_SCROLL_DICT = {
    "pherc0139": [20260115000000, 20260317000000, 20250223000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc1667": [20240304141531, 20240304144031],
    "pherc0009b": [20250919125754],
    "phercparis4": [20231210121321],
    "pherc0500p2": [20250628074500],
    "pherc0814": [20260226000000],
}
TRAIN_SCROLL_WEIGHTS = [4, 1, 1, 1, 1, 1, 1]
W044_SCROLL_ID = 20260115000000
DOMAIN_INDEX = {name: index for index, name in enumerate(TRAIN_SCROLL_DICT)}
_SCROLL_IDS = tuple(
    scroll_id
    for scroll_ids in TRAIN_SCROLL_DICT.values()
    for scroll_id in scroll_ids
)
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}
_missing = set(_SCROLL_IDS) - set(_SCROLLS_BY_ID)
if _missing:
    raise RuntimeError(f"campaign-25 scroll definitions missing: {sorted(_missing)}")
CAMPAIGN_SCROLLS = [_SCROLLS_BY_ID[scroll_id] for scroll_id in _SCROLL_IDS]

def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"25_{tid}",
        "scrolls": CAMPAIGN_SCROLLS,
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
    test.update(overrides)
    return test


TESTS = [
    _test("baseline"),
    _test("baseline_4x_weight"),
    _test("supcon_temp_0p02", supcon_temp=0.02),
    _test("supcon_temp_0p20", supcon_temp=0.20),
    _test("supcon_all_domains", supcon_cross_frag=False),
    _test("minimum_support_top2", minimum_support_k=2),
    _test("minimum_support_top4", minimum_support_k=4),
    _test("weldon_top_bottom4", weldon_k=4),
    _test("clam_lite_k4", clam_instance=True, clam_instance_k=4, clam_instance_lambda=0.1),
    _test("sam_rho_0p01", sam_rho=0.01),
    _test("sam_rho_0p05", sam_rho=0.05),
    _test("elr_epoch5", elr=True, elr_start_epoch=5, elr_beta=0.7, elr_lambda=0.1),
    _test("fiber_coordinates", pretrain_key="fiber", fiber_coordinate_branch=True),
    _test("early_3d2d_attnmax", pretrain_key="early_3d2d", early_2d_unet=True),
    _test("divided_depth_attention", pretrain_key="divided_depth", divided_attention=True),
    _test(
        "divided_space_depth_attention",
        pretrain_key="divided_space_depth",
        divided_attention=True,
        divided_attention_spatial=True,
    ),
    _test("dg_baseline_no_aux_newdata"),
]


def build_config(test: dict):
    config = campaign23_build_config(test)
    pretrain_key = str(test.get("pretrain_key", "base"))
    expected_pretrain_key = _expected_pretrain_key(test)
    if pretrain_key != expected_pretrain_key:
        raise RuntimeError(
            f"campaign-25 arm {test['tid']} maps to pretrain_key={pretrain_key!r}, "
            f"but its architecture requires {expected_pretrain_key!r}"
        )
    config.init_weights = str(_pretrain_path(pretrain_key).relative_to(ROOT))
    config.model.require_architecture_init = True
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 9
    config.tra.eval_int = 999
    config.tra.fast_eval_figure = False
    config.tra.eval_int_scrolls = 1
    config.tra.test_int = 9_999
    config.tra.probe_int = 9_999
    config.tra.test_on_final = False
    config.dl.num_workers = 24
    config.dl.prefetch_factor = 4

    config.tra.dann = False
    config.tra.dann_lambda = 0.0
    config.tra.dann_grl_anneal = False
    config.tra.dann_n_domains = len(TRAIN_SCROLL_DICT)
    config.tra.supcon = False
    config.tra.supcon_cross_frag = False
    config.tra.per_scroll_metrics = True
    config.tra.supcon_temp = float(test.get("supcon_temp", 0.07))
    config.tra.supcon_ignore_same_domain_same_class = bool(
        test.get("supcon_ignore_same_domain_same_class", False)
    )
    config.tra.prototype_align = bool(test.get("prototype_align", False))
    config.tra.prototype_align_lambda = float(test.get("prototype_align_lambda", 0.1))
    config.tra.prototype_margin = float(test.get("prototype_margin", 0.5))
    config.tra.coral_align = bool(test.get("coral_align", False))
    config.tra.coral_align_lambda = float(test.get("coral_align_lambda", 0.1))
    config.tra.coral_mean_weight = float(test.get("coral_mean_weight", 1.0))
    config.tra.cdan = bool(test.get("cdan", False))
    config.tra.cdan_lambda = float(test.get("cdan_lambda", 0.1))
    config.tra.mldg = bool(test.get("mldg", False))
    config.tra.mldg_holdout_domain = int(test.get("mldg_holdout_domain", -1))
    config.tra.mldg_inner_lr = float(test.get("mldg_inner_lr", 5e-4))
    config.tra.mldg_beta = float(test.get("mldg_beta", 1.0))
    config.tra.sagnet_lambda = float(test.get("sagnet_lambda", 0.1))
    config.tra.mae_reconstruction = bool(test.get("mae_reconstruction", False))
    config.tra.mae_reconstruction_lambda = float(
        test.get("mae_reconstruction_lambda", 0.1)
    )
    config.tra.mae_reconstruction_start_epoch = int(
        test.get("mae_reconstruction_start_epoch", 4)
    )
    config.tra.mae_reconstruction_mask_frac = float(
        test.get("mae_reconstruction_mask_frac", 0.5)
    )
    config.tra.mae_reconstruction_patch = int(test.get("mae_reconstruction_patch", 4))

    config.data.simple_split = False
    # Discover every candidate/donor context first, then eagerly retain its full-depth
    # spatial chunks in a campaign-lifetime RAM cache shared by forked workers and arms.
    config.data.preload_volumes = False
    config.data.selective_chunk_preload = True
    config.data.selective_chunk_workers = 8
    config.data.ram_safe_vis = False
    config.data.mask_memmap = os.name == "nt"
    config.data.train_mask_dir = "./train_masks"
    config.data.scrolls = list(CAMPAIGN_SCROLLS)
    config.data.vis_scroll_ids = [W044_SCROLL_ID]
    config.data.character_balance_scrolls = True
    config.data.character_balanced_sampling = True
    config.data.train_scroll_dict = {
        name: list(scroll_ids) for name, scroll_ids in TRAIN_SCROLL_DICT.items()
    }
    config.data.train_scroll_weights = list(TRAIN_SCROLL_WEIGHTS)

    config.model.minimum_support_k = int(test.get("minimum_support_k", 0))
    config.model.minimum_support_kernel = 3
    config.model.weldon_k = int(test.get("weldon_k", 0))
    config.model.fiber_coordinate_branch = bool(test.get("fiber_coordinate_branch", False))
    config.model.early_2d_unet = bool(test.get("early_2d_unet", False))
    config.model.divided_attention = bool(test.get("divided_attention", False))
    config.model.divided_attention_spatial = bool(
        test.get("divided_attention_spatial", False)
    )
    config.model.divided_attention_heads = 4
    config.model.divided_attention_window = 8
    config.model.mednext_adapters = bool(test.get("mednext_adapters", False))
    config.model.mednext_kernel = int(test.get("mednext_kernel", 5))
    config.model.mednext_expansion = 2
    config.model.mixstyle = bool(test.get("mixstyle", False))
    config.model.mixstyle_prob = float(test.get("mixstyle_prob", 0.8))
    config.model.mixstyle_alpha = float(test.get("mixstyle_alpha", 0.1))
    config.model.sagnet = bool(test.get("sagnet", False))
    config.model.dual_scale = bool(test.get("dual_scale", False))
    config.model.dual_scale_local_size = int(test.get("dual_scale_local_size", 64))
    config.model.dual_scale_mix = float(test.get("dual_scale_mix", 0.25))
    config.model.style_film = bool(test.get("style_film", False))
    config.model.style_film_hidden = int(test.get("style_film_hidden", 64))
    config.model.mae_reconstruction_head = config.tra.mae_reconstruction
    if "compile_model" in test:
        config.model.compile_model = bool(test["compile_model"])
    config.tra.clam_instance = bool(test.get("clam_instance", False))
    config.tra.clam_instance_k = int(test.get("clam_instance_k", 4))
    config.tra.clam_instance_lambda = float(test.get("clam_instance_lambda", 0.1))
    config.tra.sam_rho = float(test.get("sam_rho", 0.0))
    config.tra.elr = bool(test.get("elr", False))
    config.tra.elr_start_epoch = int(test.get("elr_start_epoch", 5))
    config.tra.elr_beta = float(test.get("elr_beta", 0.7))
    config.tra.elr_lambda = float(test.get("elr_lambda", 0.1))

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    """ensure the common and selected architecture MAE checkpoints are complete."""
    if not PRETRAIN_SEED_PATH.is_file() or PRETRAIN_SEED_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"campaign-25 MAE seed checkpoint is missing or empty: {PRETRAIN_SEED_PATH}"
        )
    requested = {str(test.get("pretrain_key", "base")) for test in selected}
    keys = ["base", *sorted(requested - {"base"})]
    for key in keys:
        if key not in PRETRAIN_SPECS:
            raise ValueError(f"unknown campaign-25 pretraining key: {key}")
        checkpoint = _pretrain_path(key)
        marker = _pretrain_marker(key)
        if _pretraining_complete(key):
            print(f"[campaign25] pretraining ready: {key} -> {checkpoint.relative_to(ROOT)}")
            continue

        command = [
            sys.executable,
            str(ROOT / "mae_pretrain_nnunet.py"),
            "--name", _pretrain_name(key),
            "--scroll-ids", *(str(scroll_id) for scroll_id in PRETRAIN_SCROLL_IDS),
            "--require-all-scrolls",
            "--ctx", "192",
            "--ds", "2",
            "--depth", "8",
            "--d-start", "10",
            "--d-end", "18",
            "--steps", str(PRETRAIN_STEPS),
            "--batch-size", "32",
            "--accum-steps", "1",
            *PRETRAIN_SPECS[key],
        ]
        init_weights = _pretrain_path("base") if key != "base" else PRETRAIN_SEED_PATH
        command.extend(("--init-weights", str(init_weights)))
        if key != "base":
            command.extend(
                (
                    "--freeze-loaded-backbone",
                )
            )
        if dry_run:
            print(f"[campaign25] pretraining missing: {key} (would run {PRETRAIN_STEPS} steps)")
            continue

        print(
            f"[campaign25] required pretraining missing: {key}; "
            f"running {PRETRAIN_STEPS} MAE steps",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"MAE pretraining did not create {checkpoint}")
        marker.write_text(
            json.dumps(
                {
                    "key": key,
                    "steps": PRETRAIN_STEPS,
                    "scroll_ids": list(PRETRAIN_SCROLL_IDS),
                    "architecture_args": list(PRETRAIN_SPECS[key]),
                    "freeze_loaded_backbone": key != "base",
                    "init_weights": str(init_weights.relative_to(ROOT)),
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )


def run_test(config, dry_run: bool) -> bool:
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    print(f"\n{'=' * 78}\n[campaign25] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  scrolls={scroll_ids} epochs={config.tra.n_epochs} eval={config.tra.eval_int}"
        f" eval_scroll_ids={config.data.vis_scroll_ids}"
        f" fast_eval={config.tra.fast_eval_figure}",
        flush=True,
    )
    print(
        f"  physical_groups={config.data.train_scroll_dict}"
        f" weights={config.data.train_scroll_weights}"
        f" DANN=fixed:{config.tra.dann_lambda}"
        f" mask_memmap={config.data.mask_memmap}"
        f" xfrag={config.tra.supcon_cross_frag}"
        f" supcon_temp={config.tra.supcon_temp}"
        f" ignore_local_same_class={config.tra.supcon_ignore_same_domain_same_class}",
        flush=True,
    )
    print(
        f"  aggregator=(support_k={config.model.minimum_support_k},"
        f" weldon_k={config.model.weldon_k})"
        f" clam=(enabled={config.tra.clam_instance}, k={config.tra.clam_instance_k},"
        f" lambda={config.tra.clam_instance_lambda})"
        f" sam_rho={config.tra.sam_rho}"
        f" elr=(enabled={config.tra.elr}, start={config.tra.elr_start_epoch},"
        f" beta={config.tra.elr_beta}, lambda={config.tra.elr_lambda})"
        f" fiber={config.model.fiber_coordinate_branch}",
        flush=True,
    )
    print(
        f"  architecture=(early_3d2d={config.model.early_2d_unet},"
        f" divided_depth={config.model.divided_attention},"
        f" divided_spatial={config.model.divided_attention_spatial},"
        f" mednext={config.model.mednext_adapters},"
        f" mednext_kernel={config.model.mednext_kernel})",
        flush=True,
    )
    print(
        f"  domain_generalization=(prototype={config.tra.prototype_align},"
        f" coral={config.tra.coral_align}, cdan={config.tra.cdan},"
        f" mldg={config.tra.mldg}:{config.tra.mldg_holdout_domain},"
        f" mixstyle={config.model.mixstyle}, sagnet={config.model.sagnet},"
        f" dual_scale={config.model.dual_scale}, film={config.model.style_film},"
        f" mae_reconstruction={config.tra.mae_reconstruction})",
        flush=True,
    )
    print(
        f"  pretrained={config.init_weights}"
        f" strict_architecture_init={config.model.require_architecture_init}",
        flush=True,
    )
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
        del trainer
        gc.collect()
        cleanup_mmap_files()


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 25: grouped 11-fragment baseline")
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

    preflight_train_masks(CAMPAIGN_SCROLLS)
    preflight_pretraining(selected, args.dry_run)

    print(f"[campaign25] {len(selected)} run(s) queued (log -> {LOG_DIR})")
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

    print(f"\n{'=' * 78}\n[campaign25] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
