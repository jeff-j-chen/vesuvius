"""campaign 29: expanded-data robustness and depth-localization study

Every arm uses the Campaign 28 scroll set and ten-epoch protocol unless it is one
of the two explicitly fragment-only controls. No arm renders evaluation figures.

Usage:
    python3 campaign_archs_29.py --dry-run
    python3 campaign_archs_29.py --only physical_groupdro
    python3 campaign_archs_29.py --from domain_vrex
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS, DEFAULT_TEST_SCROLL_IDS, startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs29"
MODEL_DIR = "models/archs29"
PRETRAIN_STEPS = 2_000
ALL_PRETRAIN_SCROLL_IDS = tuple(dict.fromkeys(
    [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
    + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
))

CAMPAIGN28_SCROLL_DICT = {
    "pherc0139": [20260115000000, 20260317000000, 20250223000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc1667": [20240304141531, 20240304144031, 20231201215900],
    "pherc0009b": [20250919125754],
    "phercparis4": [20231210121321],
    "pherc0500p2": [20250628074500],
    "pherc0814": [20260226000000],
    "phercparis2_fr143": [20230301213755],
    "pherc51cr4_fr8": [20231205222200],
    "phercparis1_fr34": [20230301213423],
    "pherc0343p": [20250511003658],
    "pherc0841": [20260221022814],
}
CAMPAIGN28_SCROLL_WEIGHTS = [4] + [1] * (len(CAMPAIGN28_SCROLL_DICT) - 1)
CAMPAIGN28_SCROLL_IDS = tuple(
    scroll_id
    for scroll_ids in CAMPAIGN28_SCROLL_DICT.values()
    for scroll_id in scroll_ids
)

FRAGMENT_SCROLL_DICT = {
    "pherc0500p2": [20250628074500],
    "phercparis2_fr143": [20230301213755],
    "pherc51cr4_fr8": [20231205222200],
    "phercparis1_fr34": [20230301213423],
    "pherc1667_cr1_fr3": [20231201215900],
}
FRAGMENT_SCROLL_IDS = tuple(
    scroll_id
    for scroll_ids in FRAGMENT_SCROLL_DICT.values()
    for scroll_id in scroll_ids
)
_SCROLLS_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}
_missing = set(CAMPAIGN28_SCROLL_IDS + FRAGMENT_SCROLL_IDS) - set(_SCROLLS_BY_ID)
if _missing:
    raise RuntimeError(f"campaign 29 scroll definitions missing: {sorted(_missing)}")
CAMPAIGN28_SCROLLS = [_SCROLLS_BY_ID[scroll_id] for scroll_id in CAMPAIGN28_SCROLL_IDS]
FRAGMENT_SCROLLS = [_SCROLLS_BY_ID[scroll_id] for scroll_id in FRAGMENT_SCROLL_IDS]

PRETRAIN_SPECS = {
    "mid_gated_all": {
        "depth": 8,
        "d_start": 10,
        "d_end": 18,
        "args": ("--mid-2d-unet", "--gated-stems", "--norm-mode", "ibn_full"),
        "required": ("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
    },
    "mid_gated_explicit_depth_all": {
        "depth": 8,
        "d_start": 10,
        "d_end": 18,
        "args": (
            "--mid-2d-unet", "--gated-stems", "--explicit-depth-channels",
            "--norm-mode", "ibn_full",
        ),
        "required": ("enc1.", "gated_cue_stem.", "mid2d_"),
    },
    "mid_gated_overlap12_all": {
        "depth": 12,
        "d_start": 8,
        "d_end": 20,
        "args": (
            "--mid-2d-unet", "--gated-stems", "--overlapping-depth-windows",
            "--overlapping-depth-window-size", "4",
            "--overlapping-depth-window-stride", "2",
            "--norm-mode", "ibn_full",
        ),
        "required": (
            "gated_cue_stem.", "mid2d_", "overlap_bottleneck_fuse.",
            "overlap_decoded_fuse.",
        ),
    },
    "mid_gated_antialias_all": {
        "depth": 8,
        "d_start": 10,
        "d_end": 18,
        "args": (
            "--mid-2d-unet", "--gated-stems", "--depth-antialias",
            "--norm-mode", "ibn_full",
        ),
        "required": ("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
    },
}


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_campaign29_{key}_2k"


def _pretrain_path(key: str) -> Path:
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"


def _pretrain_marker(key: str) -> Path:
    return _pretrain_path(key).with_suffix(".complete.json")


def _pretrain_metadata(key: str) -> dict:
    spec = PRETRAIN_SPECS[key]
    return {
        "campaign": 29,
        "key": key,
        "steps": PRETRAIN_STEPS,
        "scroll_ids": list(ALL_PRETRAIN_SCROLL_IDS),
        "architecture_args": list(spec["args"]),
        "depth": spec["depth"],
        "d_start": spec["d_start"],
        "d_end": spec["d_end"],
        "from_scratch": True,
        "sampling": "physical_round_robin",
        "checkpoint": str(_pretrain_path(key).relative_to(ROOT)),
    }


def _pretraining_complete(key: str) -> bool:
    checkpoint = _pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return metadata == _pretrain_metadata(key) and all(
        any(name.startswith(prefix) for name in state)
        for prefix in PRETRAIN_SPECS[key]["required"]
    )


def _test(tid: str, **overrides) -> dict:
    test = {
        "tid": tid,
        "tag": f"29_{tid}",
        "scrolls": CAMPAIGN28_SCROLLS,
        "pretrain_key": "mid_gated_all",
        "mid_2d_unet": True,
        "gated_stems": True,
        "norm_mode": "ibn_full",
        "compile_model": False,
    }
    test.update(overrides)
    return test


TESTS = [
    # Campaign 28 lock-in and attempts to retain conflict weighting without
    # sacrificing the new-domain gains of mid-3D/2D plus gated stems.
    # Completed: baseline, gradient_conflict, gradient_conflict_blend.
    # _test("baseline"),
    # _test(
    #     "gradient_conflict",
    #     domain_gradient_mode="conflict_weighted",
    #     domain_gradient_threshold=0.8,
    #     domain_gradient_strength=8.0,
    # ),
    # _test(
    #     "gradient_conflict_blend",
    #     domain_gradient_mode="conflict_weighted",
    #     domain_gradient_threshold=0.8,
    #     domain_gradient_strength=8.0,
    #     domain_gradient_blend=0.5,
    # ),
    _test(
        "physical_groupdro",
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
    ),
    _test(
        "physical_patch_groupdro",
        physical_patch_groupdro=True,
        physical_patch_groupdro_eta=0.05,
        physical_patch_groupdro_max_ratio=3.0,
    ),

    # New optimization and preservation mechanisms.
    _test("domain_vrex", domain_vrex=True, domain_vrex_lambda=1.0),
    # Completed: domain_cvar.
    # _test("domain_cvar", domain_cvar=True, domain_cvar_alpha=0.35),
    _test("pcgrad", pcgrad=True, compile_model=False),
    _test("cue_dropout", cue_dropout=0.33),
    _test("model_ema", model_ema=True, model_ema_decay=0.995),
    _test("mae_anchor", mae_anchor_lambda=0.001),

    # New depth/problem formulations.
    _test(
        "overlap_depth12",
        pretrain_key="mid_gated_overlap12_all",
        depth=12,
        overlapping_depth_windows=True,
        overlapping_depth_window_size=4,
        overlapping_depth_window_stride=2,
    ),
    _test(
        "explicit_depth",
        pretrain_key="mid_gated_explicit_depth_all",
        explicit_depth_channels=True,
    ),
    _test(
        "depth_antialias",
        pretrain_key="mid_gated_antialias_all",
        depth_antialias=True,
    ),
    _test(
        "depth_shift_aux",
        depth_shift_aux=True,
        depth_shift_aux_lambda=0.2,
        depth_shift_aux_classes=3,
    ),

    # Same 64x64 prediction center, but 8x8 targets instead of 16x16.
    _test("multitile_8px", multitile_subtile=8, multitile_grid=8),

    # Fragment-domain stress tests.
    _test("fragments_only", fragments_only=True),
    _test(
        "fragments_gap1_dilated2",
        fragments_only=True,
        inklabel_dir="./dilated_inklabels",
        ring_close_r=0,
        ring_gap_r=1,
        ring_shell_r=4,
        multitile_pos_only=False,
    ),
    _test(
        "fragments_gap1_dilated2_replica",
        fragments_only=True,
        inklabel_dir="./dilated_inklabels",
        ring_close_r=0,
        ring_gap_r=1,
        ring_shell_r=4,
        multitile_pos_only=False,
    ),
    _test(
        "fixed_depth_8_16",
        train_d_start=8,
        train_d_end=16,
        depth_jitter=0,
        surface_relative_depth_window=False,
    ),
    _test(
        "coordinate_hash_split",
        coordinate_hash_split=True,
        coordinate_hash_block_size=512,
        coordinate_hash_valid_fraction=0.25,
    ),
]


def build_config(test: dict):
    config = Config()
    config.exp_name = f"cmp_archs29_{test['tag']}"
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 10
    config.tra.eval_int = 9_999
    config.tra.eval_int_scrolls = 0
    config.tra.test_int = 9_999
    config.tra.probe_int = 9_999
    config.tra.test_on_final = False
    config.tra.save_vis = False
    config.tra.dann = False
    config.tra.dann_lambda = 0.0
    config.tra.dann_grl_anneal = False
    config.tra.supcon = False
    config.tra.supcon_cross_frag = False
    config.tra.per_scroll_metrics = True
    config.data.vis_scroll_ids = []
    config.data.scrolls = list(CAMPAIGN28_SCROLLS)
    config.data.train_scroll_dict = {
        name: list(ids) for name, ids in CAMPAIGN28_SCROLL_DICT.items()
    }
    config.data.train_scroll_weights = list(CAMPAIGN28_SCROLL_WEIGHTS)
    config.data.max_samples_per_epoch = 6_667
    config.data.simple_split = False
    config.data.preload_volumes = False
    config.data.selective_chunk_preload = True
    config.data.selective_chunk_workers = 8
    config.data.ram_safe_vis = False
    config.data.mask_memmap = os.name == "nt"
    config.data.mask_bitpack = True
    config.data.train_mask_dir = "./train_masks"
    config.data.character_balance_scrolls = True
    config.data.character_balanced_sampling = True
    config.data.label_dilate_r = 0
    config.data.ring_negatives = True
    config.data.ring_label_source = "closed"
    config.data.multitile_pos_only = True
    config.data.depth_jitter = 1
    config.dl.num_workers = 8
    config.dl.prefetch_factor = 2
    config.dl.context_replace_prob = 0.35
    config.dl.context_replace_margin = 20
    config.dl.context_replace_feather = 40
    config.dl.cutout_prob = 0.50
    config.dl.cutout_max_frac = 0.16
    config.dl.cutout_n_patches = 3

    if bool(test.get("fragments_only", False)):
        config.data.scrolls = list(FRAGMENT_SCROLLS)
        config.data.train_scroll_dict = {
            name: list(ids) for name, ids in FRAGMENT_SCROLL_DICT.items()
        }
        config.data.train_scroll_weights = [1] * len(FRAGMENT_SCROLL_DICT)
        config.tra.dann_n_domains = len(FRAGMENT_SCROLL_DICT)

    config.data.inklabel_dir = str(test.get("inklabel_dir", "./inklabels"))
    config.data.ring_close_r = int(test.get("ring_close_r", 2))
    config.data.ring_gap_r = int(test.get("ring_gap_r", 2))
    config.data.ring_shell_r = int(test.get("ring_shell_r", 4))
    config.data.multitile_pos_only = bool(test.get("multitile_pos_only", True))
    config.data.multitile_ring_gate = False  # reproduce campaign-29 legacy pos_only=False targets
    config.data.ring_from_inklabel_dir = False  # campaign 29 built rings from ./inklabels
    config.data.depth = int(test.get("depth", 8))
    config.data.train_d_start = int(test.get("train_d_start", config.data.train_d_start))
    config.data.train_d_end = int(test.get("train_d_end", config.data.train_d_end))
    config.data.depth_jitter = int(test.get("depth_jitter", config.data.depth_jitter))
    config.data.surface_relative_depth_window = bool(
        test.get(
            "surface_relative_depth_window",
            config.data.surface_relative_depth_window,
        )
    )
    config.data.coordinate_hash_split = bool(test.get("coordinate_hash_split", False))
    config.data.coordinate_hash_block_size = int(
        test.get("coordinate_hash_block_size", 512)
    )
    config.data.coordinate_hash_valid_fraction = float(
        test.get("coordinate_hash_valid_fraction", 0.25)
    )
    config.data.coordinate_hash_seed = int(test.get("coordinate_hash_seed", 29))

    config.tra.domain_gradient_mode = str(test.get("domain_gradient_mode", ""))
    config.tra.domain_gradient_threshold = float(test.get("domain_gradient_threshold", 0.0))
    config.tra.domain_gradient_strength = float(test.get("domain_gradient_strength", 4.0))
    config.tra.domain_gradient_blend = float(test.get("domain_gradient_blend", 1.0))
    config.tra.physical_domain_groupdro = bool(
        test.get("physical_domain_groupdro", False)
    )
    config.tra.physical_domain_groupdro_eta = float(
        test.get("physical_domain_groupdro_eta", 0.05)
    )
    config.tra.physical_domain_groupdro_max_ratio = float(
        test.get("physical_domain_groupdro_max_ratio", 3.0)
    )
    config.tra.physical_patch_groupdro = bool(
        test.get("physical_patch_groupdro", False)
    )
    config.tra.physical_patch_groupdro_eta = float(
        test.get("physical_patch_groupdro_eta", 0.05)
    )
    config.tra.physical_patch_groupdro_max_ratio = float(
        test.get("physical_patch_groupdro_max_ratio", 3.0)
    )
    config.tra.domain_vrex = bool(test.get("domain_vrex", False))
    config.tra.domain_vrex_lambda = float(test.get("domain_vrex_lambda", 1.0))
    config.tra.domain_vrex_warmup_epochs = int(test.get("domain_vrex_warmup_epochs", 2))
    config.tra.domain_cvar = bool(test.get("domain_cvar", False))
    config.tra.domain_cvar_alpha = float(test.get("domain_cvar_alpha", 0.25))
    config.tra.pcgrad = bool(test.get("pcgrad", False))
    config.tra.model_ema = bool(test.get("model_ema", False))
    config.tra.model_ema_decay = float(test.get("model_ema_decay", 0.999))
    config.tra.mae_anchor_lambda = float(test.get("mae_anchor_lambda", 0.0))
    config.tra.seed = int(test.get("seed", config.tra.seed))
    config.tra.encoder_freeze_epochs = int(test.get("encoder_freeze_epochs", 0))
    config.tra.encoder_lr_scale = float(test.get("encoder_lr_scale", 1.0))
    config.tra.depth_shift_aux = bool(test.get("depth_shift_aux", False))
    config.tra.depth_shift_aux_lambda = float(test.get("depth_shift_aux_lambda", 0.1))
    config.tra.depth_shift_aux_classes = int(test.get("depth_shift_aux_classes", 3))

    config.model.cue_dropout = float(test.get("cue_dropout", 0.0))
    config.model.mid_2d_unet = True
    config.model.gated_stems = True
    config.model.norm_mode = "ibn_full"
    config.model.use_ibn = False
    config.model.explicit_depth_channels = bool(test.get("explicit_depth_channels", False))
    config.model.overlapping_depth_windows = bool(
        test.get("overlapping_depth_windows", False)
    )
    config.model.overlapping_depth_window_size = int(
        test.get("overlapping_depth_window_size", 4)
    )
    config.model.overlapping_depth_window_stride = int(
        test.get("overlapping_depth_window_stride", 2)
    )
    config.model.depth_antialias = bool(test.get("depth_antialias", False))
    config.model.multitile_subtile = int(test.get("multitile_subtile", 16))
    config.model.multitile_grid = int(test.get("multitile_grid", 4))
    config.model.surface_teacher_input = True
    config.model.compile_model = False
    config.model.require_architecture_init = True
    config.init_weights = str(_pretrain_path(str(test["pretrain_key"])).relative_to(ROOT))

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def preflight_train_masks(scrolls, inklabel_dir: Path, strict: bool = True) -> None:
    failures = []
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2"))
    for scroll in scrolls:
        scroll_id = int(scroll.scroll_id)
        required = (
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            inklabel_dir / f"{scroll_id}.png",
            ROOT / "train_masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            failures.append(f"{scroll_id}: {', '.join(missing)}")
    if failures:
        message = "Campaign 29 input preflight failed:\n  " + "\n  ".join(failures)
        if strict:
            raise RuntimeError(message)
        print(f"[campaign29] WARNING {message}", flush=True)
        return
    print(f"[preflight] {len(scrolls)} Campaign 29 fragments passed", flush=True)


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    missing = [
        scroll_id for scroll_id in ALL_PRETRAIN_SCROLL_IDS
        if not (ROOT / "ves_zarrs2" / f"{scroll_id}.zarr").is_dir()
    ]
    if missing:
        message = f"Campaign 29 requires every training and test zarr; missing={missing}"
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign29] WARNING {message}", flush=True)
    keys = list(dict.fromkeys(str(test["pretrain_key"]) for test in selected))
    for key in keys:
        if key not in PRETRAIN_SPECS:
            raise ValueError(f"unknown Campaign 29 pretrain key: {key}")
        if not _pretraining_complete(key):
            spec = PRETRAIN_SPECS[key]
            if dry_run:
                print(
                    f"[campaign29] would pretrain {key}: {PRETRAIN_STEPS} steps, "
                    f"{len(ALL_PRETRAIN_SCROLL_IDS)} required zarrs",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(ROOT / "mae_pretrain_nnunet.py"),
                "--name", _pretrain_name(key),
                "--scroll-ids", *(str(value) for value in ALL_PRETRAIN_SCROLL_IDS),
                "--require-all-scrolls",
                "--ctx", "192",
                "--ds", "2",
                "--depth", str(spec["depth"]),
                "--d-start", str(spec["d_start"]),
                "--d-end", str(spec["d_end"]),
                "--steps", str(PRETRAIN_STEPS),
                "--batch-size", "24" if spec["depth"] > 8 else "32",
                "--accum-steps", "1",
                "--no-figures",
                "--physical-round-robin",
                *spec["args"],
            ]
            print(f"[campaign29] pretraining {key} from scratch", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            _pretrain_marker(key).write_text(
                json.dumps(_pretrain_metadata(key), indent=2) + "\n",
                encoding="utf-8",
            )
        if not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 29 MAE pretraining failed validation: {key}")


def run_test(config, dry_run: bool) -> bool:
    print(f"\n{'=' * 78}\n[campaign29] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  pretrain={config.init_weights} compile={config.model.compile_model} "
        f"depth={config.data.depth} target={config.model.multitile_subtile}px "
        f"grid={config.model.multitile_grid}x{config.model.multitile_grid}",
        flush=True,
    )
    print(
        f"  robust=gradient:{config.tra.domain_gradient_mode}:"
        f"blend{config.tra.domain_gradient_blend} groupdro:{config.tra.physical_domain_groupdro} "
        f"patchdro:{config.tra.physical_patch_groupdro} "
        f"vrex:{config.tra.domain_vrex} cvar:{config.tra.domain_cvar} "
        f"pcgrad:{config.tra.pcgrad}",
        flush=True,
    )
    print(
        f"  split={'coordinate_hash' if config.data.coordinate_hash_split else 'manual'} "
        f"block={config.data.coordinate_hash_block_size} "
        f"valid={config.data.coordinate_hash_valid_fraction}",
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


def _cgroup_oom_kill_count() -> int:
    try:
        entries = dict(
            line.split(maxsplit=1)
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
        return int(entries.get("oom_kill", 0))
    except (OSError, TypeError, ValueError):
        return -1


def _expected_prepared_cache_keys(config) -> set[tuple]:
    from utils.dataloader import DataManager

    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]
    domain_by_scroll = {scroll_id: index for index, scroll_id in enumerate(scroll_ids)}
    scroll_dict = getattr(config.data, "train_scroll_dict", None)
    if scroll_dict:
        domain_by_scroll = {
            int(scroll_id): domain_id
            for domain_id, group in enumerate(scroll_dict.values())
            for scroll_id in group
        }
    keys = set()
    for segment_index, scroll_id in enumerate(scroll_ids):
        manager = DataManager.__new__(DataManager)
        manager.c = config
        manager.scroll_id = scroll_id
        manager.domain_id = domain_by_scroll[scroll_id]
        manager.character_namespace = segment_index
        keys.add(manager._make_prepared_cache_key())
    return keys


def prewarm_data_cache(config) -> None:
    with startup_output():
        _prewarm_data_cache(config)


def _prewarm_data_cache(config) -> None:
    from train import Trainer
    from utils.chunk_cache import _CACHE_REGISTRY
    from utils.dataloader import _PREPARED_DATASET_CACHE, _PREPARED_DATASET_CACHE_LOCK

    if torch.cuda.is_initialized():
        raise RuntimeError("Campaign 29 controller initialized CUDA before cache warmup")
    expected_keys = _expected_prepared_cache_keys(config)
    with _PREPARED_DATASET_CACHE_LOCK:
        current_keys = set(_PREPARED_DATASET_CACHE)
        if current_keys != expected_keys:
            print(
                f"[campaign29] rotating prepared RAM cache: "
                f"{len(current_keys)} -> {len(expected_keys)} dataset(s)",
                flush=True,
            )
            _PREPARED_DATASET_CACHE.clear()
    if current_keys != expected_keys:
        gc.collect()
    warmup = Trainer.__new__(Trainer)
    warmup.c = config
    train_dataset = train_loader = valid_loader = None
    try:
        train_dataset, train_loader, valid_loader = warmup._setup_data()
    finally:
        del train_dataset, train_loader, valid_loader, warmup
        gc.collect()
    with _PREPARED_DATASET_CACHE_LOCK:
        prepared_keys = set(_PREPARED_DATASET_CACHE)
    if prepared_keys != expected_keys or not _CACHE_REGISTRY:
        raise RuntimeError("Campaign 29 RAM cache warmup did not populate global caches")
    if torch.cuda.is_initialized():
        raise RuntimeError("Campaign 29 cache warmup unexpectedly initialized CUDA")
    cached_gib = sum(volume.cached_nbytes for volume in _CACHE_REGISTRY.values()) / 1024**3
    print(
        f"[campaign29] parent RAM cache ready: datasets={len(_PREPARED_DATASET_CACHE)} "
        f"volumes={len(_CACHE_REGISTRY)} ram={cached_gib:.2f}GiB "
        f"controller_fds={_open_fd_count()}",
        flush=True,
    )


def run_test_isolated(config) -> bool:
    if not hasattr(os, "fork"):
        return run_test(config, False)
    before = _open_fd_count()
    oom_kills_before = _cgroup_oom_kill_count()
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
    oom_kills_after = _cgroup_oom_kill_count()
    status_detail = "unknown"
    if os.WIFEXITED(status):
        status_detail = f"exit={os.WEXITSTATUS(status)}"
    elif os.WIFSIGNALED(status):
        signal_number = os.WTERMSIG(status)
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = str(signal_number)
        status_detail = f"signal={signal_name}"
        if oom_kills_after > oom_kills_before >= 0:
            status_detail += f" cgroup_oom_kill={oom_kills_before}->{oom_kills_after}"
    print(
        f"[campaign29] isolated arm pid={pid} status={status} {status_detail} "
        f"controller_fds={before}->{after}",
        flush=True,
    )
    if before >= 0 and after > before + 4:
        raise RuntimeError(f"Campaign 29 controller leaked file descriptors: {before}->{after}")
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="campaign 29: robustness and depth-localization study"
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

    preflight_groups: dict[str, list] = {}
    for test in selected:
        source = str(test.get("inklabel_dir", "./inklabels"))
        scrolls = FRAGMENT_SCROLLS if test.get("fragments_only", False) else CAMPAIGN28_SCROLLS
        known = {int(scroll.scroll_id) for scroll in preflight_groups.get(source, [])}
        preflight_groups.setdefault(source, []).extend(
            scroll for scroll in scrolls if int(scroll.scroll_id) not in known
        )
    for source, scrolls in preflight_groups.items():
        preflight_train_masks(
            scrolls,
            inklabel_dir=Path(source),
            strict=not args.dry_run,
        )
    preflight_pretraining(selected, args.dry_run)
    print(f"[campaign29] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        if args.dry_run:
            success = run_test(config, True)
        else:
            prewarm_data_cache(config)
            success = run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()

    print(f"\n{'=' * 78}\n[campaign29] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()