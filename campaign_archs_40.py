"""campaign 40: campaign-39 cross-resolution arms plus scratch surface-anchor baselines

This keeps campaign 40 self-contained around the campaign-38 combined recipe: a 96 px native field,
ring close 2 / gap 2 / shell 4, sigma-8 soft edges (floor 0.55), bag ranking + fiber inputs + regime
weights + RSC + dropout/augs, and surface-anchor normalization. Campaign 39's external cross-resolution
arms are moved here, while the plain surface-anchor baselines still require campaign-40 scratch MAE
pretraining.

Pherc0841 and pherc0009b are folded back into supervised training. PHerc0211 is rendered for every arm,
kept persistently preloaded for visualization, and only enters supervised training in the dedicated
`holdout_n96_combined_surface_norm_pherc0211` arm. That arm's scroll has no validation region, so it is
split train-only there and contributes no validation metrics.

Arms include the campaign-39 cross-resolution and translated-volume variants, plus:

| arm                                           | what it changes                          |
|-----------------------------------------------|------------------------------------------|
| holdout_n96_combined_surface_norm             | scratch MAE baseline without PHerc0211   |
| holdout_n96_combined_surface_norm_pherc0211   | same, plus PHerc0211 in train + pretrain |

The scratch baselines pretrain before fine-tuning starts. The moved cross-resolution arms consume the
committed `models/c39_*` checkpoints and helper files directly.

Usage:
    python3 campaign_archs_40.py --dry-run
    python3 campaign_archs_40.py --smoke
    python3 campaign_archs_40.py --only holdout_n96_combined_surface_norm_pherc0211
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_34 as campaign34
import campaign_archs_35 as campaign35
from utils.config import DEFAULT_TEST_SCROLL_IDS, ScrollConfig, startup_output
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs40"
MODEL_DIR = "models/archs40"
INKLABEL_DIR = "./dilated_inklabels"
HOLDOUT_DOMAINS = ("pherc0841", "pherc0009b")
SURFACE_FIBER_MAE_KEY = "early_gated_fiber_native96_surfacenorm_campaign40"
SURFACE_FIBER_MAE_KEY_0211 = "early_gated_fiber_native96_surfacenorm_campaign40_pherc0211"
NEW_DOMAIN = "pherc0211"
NEW_SCROLL_ID = 20260928000003
VIS_SCROLL_IDS = [20260221022814, 20250919125754, NEW_SCROLL_ID]
SMOKE_SCROLL = 20260226000000  # pherc0814
BASE_TEST_SCROLL_IDS = tuple(
    int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS if int(scroll_id) != NEW_SCROLL_ID
)
SCRATCH_TRAINING = {"batch_size": 96, "lr": 1e-4}
CROSSRES_TRAINING = {"batch_size": 32, "lr": 1e-4}
BASE_MAE = "models/c39_mae_base.pth"
LEARNED_UPSAMPLER = "models/c39_upsampler_learned.pth"
UPSAMPLED_LEARNED_MAE = "models/c39_mae_upsampled_learned.pth"
UPSAMPLED_TRILINEAR_MAE = "models/c39_mae_upsampled_trilinear.pth"
CROSSRES_DEPTH_MAE = "models/c39_mae_crossres_depth.pth"
CROSSRES_XYZ_MAE = "models/c39_mae_crossres_xyz.pth"
SLAB_MAE = "models/c39_mae_slab.pth"
DEGRADER = "models/c39_degrader.pth"
TRANSLATED_SUFFIX = ".translated"

BASE_SCROLL_DICT = {
    "pherc0139": [
        20260115000000,
        20260317000000,
        20250223000000,
        20250108000005,
        20260112000000,
        20260126000000,
        20250831000000,
        20260108000000,
        20260302000000,
    ],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc1667": [20240304141531, 20240304144031, 20231201215900],
    "pherc0009b": [20250919125754],
    "phercparis4": [20231210121321],
    "pherc0500p2": [20250628074500],
    "pherc0814": [20260226000000],
    "phercparis2_fr143": [20230301213755, 20230205142449],
    "pherc51cr4_fr8": [20231205222200],
    "phercparis1_fr34": [20230301213423],
    "pherc0343p": [20250511003658],
    "pherc0841": [20260221022814],
}
TRAIN_SCROLL_DICT_WITH_0211 = {
    **BASE_SCROLL_DICT,
    NEW_DOMAIN: [NEW_SCROLL_ID],
}
SCROLLS_BASE = [
    ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),
    ScrollConfig(20260317000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20250223000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20250108000005, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260112000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260126000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20250831000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260108000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260302000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20251111010954, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20251112000002, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20240304141531, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20240304144031, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20231201215900, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20250919125754, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20231210121321, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20250628074500, split_axis="x", train_split_frac=0.6),
    ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20230301213755, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20230205142449, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20231205222200, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20230301213423, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20250511003658, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260221022814, split_axis="x", train_split_frac=0.75),
]
SCROLLS_WITH_0211 = [
    *SCROLLS_BASE,
    # PHerc0211 has no validation region, so this arm trains on the full fragment and skips val metrics there.
    ScrollConfig(NEW_SCROLL_ID, split_axis="x", train_split_frac=1.0),
]
REGIME_WEIGHT_BY_DOMAIN = {
    "pherc0139": 2,
    "pherc0343p": 4,
    "pherc0500p2": 3,
    "pherc0814": 5,
}
NO_DEFAULT_AUGS = {
    "dl.cutout_prob": 0.0,
    "dl.context_replace_prob": 0.0,
    "data.ctx_jitter": 0,
    "data.depth_jitter": 0,
}
RING_C2G2S4 = {"data.ring_close_r": 2, "data.ring_gap_r": 2, "data.ring_shell_r": 4}
EDGE_SOFT = {
    "data.edge_soft_sigma": 8.0,
    "data.edge_soft_floor": 0.55,
    "tra.label_smooth_pos": 0.0,
    "tra.label_smooth_neg": 0.0,
}
DROPOUT_AUGS = {
    "model.conv1_drop": 0.2,
    "model.conv2_drop": 0.2,
    "model.head_drop": 0.3,
    "dl.cutout_prob": 0.20,
    "dl.context_replace_prob": 0.15,
    "dl.context_replace_margin": 7,
    "dl.context_replace_feather": 13,
    "data.ctx_jitter": 10,
    "data.depth_jitter": 1,
}
DOWNSAMPLED = {
    "data.zarr_suffix": {str(sid): TRANSLATED_SUFFIX for sid in campaign35.FINE_NATIVE_SCROLL_IDS}
}
NATIVE_NOISE = {
    "data.native_noise": "_ves_tmp/native_noise_bank.npy",
    "data.native_noise_scale": 1.0,
    "data.native_noise_ids": [int(sid) for sid in campaign35.FINE_NATIVE_SCROLL_IDS],
}
BASE_CONFIG = {
    "tra.aug_start_epoch": 0,
    "tra.fast_eval_figure": False,
    "tra.eval_int_scrolls": len(VIS_SCROLL_IDS),
    "tra.eval_int": 10,
    "tra.test_int": 999,
    "data.vis_scroll_ids": VIS_SCROLL_IDS,
    "data.vis_preload_persistent": True,
    **NO_DEFAULT_AUGS,
}
COMBINED_SURFACE = {
    **RING_C2G2S4,
    **EDGE_SOFT,
    "tra.character_bag_ranking": True,
    "tra.character_bag_margin": 0.5,
    "tra.character_bag_topk_frac": 0.5,
    "tra.character_bag_lambda": 0.2,
    "model.fiber_coordinate_branch": True,
    "tra.rsc_prob": 0.33,
    "tra.rsc_drop_frac": 0.33,
    **DROPOUT_AUGS,
    "data.norm_mode": "surface_anchor",
}


def _scroll_ids_from_dict(train_scroll_dict: dict[str, list[int]]) -> tuple[int, ...]:
    return tuple(
        int(scroll_id)
        for scroll_ids in train_scroll_dict.values()
        for scroll_id in scroll_ids
    )


def _pretrain_scroll_ids(train_scroll_dict: dict[str, list[int]]) -> tuple[int, ...]:
    test_scroll_ids = DEFAULT_TEST_SCROLL_IDS if NEW_DOMAIN in train_scroll_dict else BASE_TEST_SCROLL_IDS
    return tuple(dict.fromkeys(
        [*(_scroll_ids_from_dict(train_scroll_dict))]
        + [int(scroll_id) for scroll_id in test_scroll_ids]
    ))


SURFACE_FIBER_REQUIRED = campaign34.EARLY_GATED_REQUIRED + ("fiber_coordinate_input.",)
campaign34.PRETRAIN_SPECS.update({
    SURFACE_FIBER_MAE_KEY: {
        **campaign31._spec(
            *campaign34.EARLY_GATED_ARGS,
            "--fiber-coordinate-branch",
            "--data-norm-mode", "surface_anchor",
            required=SURFACE_FIBER_REQUIRED,
            ctx=96,
            ds=1,
        ),
        "scroll_ids": _pretrain_scroll_ids(BASE_SCROLL_DICT),
    },
    SURFACE_FIBER_MAE_KEY_0211: {
        **campaign31._spec(
            *campaign34.EARLY_GATED_ARGS,
            "--fiber-coordinate-branch",
            "--data-norm-mode", "surface_anchor",
            required=SURFACE_FIBER_REQUIRED,
            ctx=96,
            ds=1,
        ),
        "scroll_ids": _pretrain_scroll_ids(TRAIN_SCROLL_DICT_WITH_0211),
    },
})


def _train_scroll_weights(train_scroll_dict: dict[str, list[int]]) -> list[int]:
    return [REGIME_WEIGHT_BY_DOMAIN.get(domain, 1) for domain in train_scroll_dict]


def _test(
    tid: str,
    pretrain_key: str,
    train_scroll_dict: dict[str, list[int]],
    scrolls: list[ScrollConfig],
    changes: dict | None = None,
    init_weights: str | None = None,
    upsampler: str | None = None,
    batch_size: int = 96,
    lr: float = 1.5e-4,
) -> dict:
    test = campaign34._test(
        tid,
        pretrain_key,
        holdout_domains=HOLDOUT_DOMAINS,
        context_size=96,
        context_downsample=1,
        batch_size=batch_size,
        lr=lr,
        config={
            **BASE_CONFIG,
            **COMBINED_SURFACE,
            **(changes or {}),
            "data.train_scroll_weights": _train_scroll_weights(train_scroll_dict),
        },
    )
    test["tag"] = f"40_{tid}"
    test["scrolls"] = list(scrolls)
    test["train_scroll_dict"] = {domain: list(ids) for domain, ids in train_scroll_dict.items()}
    test["train_scroll_weights"] = _train_scroll_weights(train_scroll_dict)
    test["init_weights_override"] = init_weights
    test["upsampler"] = upsampler
    return test


TESTS = [
    _test(
        "holdout_n96_combined_surface_norm",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        **SCRATCH_TRAINING,
    ),
    # _test(
    #     "holdout_n96_combined_surface_norm_pherc0211",
    #     SURFACE_FIBER_MAE_KEY_0211,
    #     TRAIN_SCROLL_DICT_WITH_0211,
    #     SCROLLS_WITH_0211,
    #     **SCRATCH_TRAINING,
    # ),
    _test(
        "holdout_n96_upsampled_learned",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        init_weights=UPSAMPLED_LEARNED_MAE,
        upsampler=LEARNED_UPSAMPLER,
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_upsampled_trilinear",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        init_weights=UPSAMPLED_TRILINEAR_MAE,
        upsampler="trilinear",
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_crossres_depth",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        init_weights=CROSSRES_DEPTH_MAE,
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_crossres_xyz",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        init_weights=CROSSRES_XYZ_MAE,
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_slab",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        init_weights=SLAB_MAE,
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_downsampled",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        changes=DOWNSAMPLED,
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_downsampled_noise",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        changes={**DOWNSAMPLED, **NATIVE_NOISE},
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_downsampled_upsampled_trilinear",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        changes=DOWNSAMPLED,
        init_weights=UPSAMPLED_TRILINEAR_MAE,
        upsampler="trilinear",
        **CROSSRES_TRAINING,
    ),
    _test(
        "holdout_n96_downsampled_upsampled_learned",
        SURFACE_FIBER_MAE_KEY,
        BASE_SCROLL_DICT,
        SCROLLS_BASE,
        changes=DOWNSAMPLED,
        init_weights=UPSAMPLED_LEARNED_MAE,
        upsampler=LEARNED_UPSAMPLER,
        **CROSSRES_TRAINING,
    ),
]


@contextlib.contextmanager
def _campaign34_paths():
    saved = campaign34.LOG_DIR, campaign34.MODEL_DIR
    campaign34.LOG_DIR, campaign34.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign34.LOG_DIR, campaign34.MODEL_DIR = saved


def build_config(test: dict):
    with _campaign34_paths():
        config = campaign34.build_config(test)
    config.data.scrolls = list(test["scrolls"])
    config.data.train_scroll_dict = {
        domain: list(ids) for domain, ids in test["train_scroll_dict"].items()
    }
    config.data.train_scroll_weights = list(test["train_scroll_weights"])
    config.data.train_only_scroll_ids = [NEW_SCROLL_ID] if NEW_DOMAIN in config.data.train_scroll_dict else []
    config.data.holdout_domains = []
    config.data.vis_scroll_ids = list(VIS_SCROLL_IDS)
    config.data.vis_preload_persistent = True
    config.data.inklabel_dir = INKLABEL_DIR
    config.tra.eval_int = 10
    config.tra.test_int = 999
    config.tra.eval_int_scrolls = len(VIS_SCROLL_IDS)
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    if test["init_weights_override"]:
        config.init_weights = str(test["init_weights_override"])
    if test["upsampler"]:
        config.model.input_upsampler = _upsampler_setting(test)
    return config


def _upsampler_setting(test: dict) -> str:
    """the upsampler named by the external MAE's sidecar, else the arm's default."""
    upsampler = str(test["upsampler"])
    sidecar = (ROOT / str(test["init_weights_override"])).with_suffix(".json")
    if sidecar.is_file():
        upsampler = str(json.loads(sidecar.read_text(encoding="utf-8")).get("upsampler") or upsampler)
    return upsampler


def _external_files(test: dict) -> list[str]:
    files = [str(test["init_weights_override"])] if test["init_weights_override"] else []
    if test["upsampler"]:
        upsampler = _upsampler_setting(test)
        if upsampler != "trilinear":
            files.append(upsampler)
    if test["config"].get("data.zarr_suffix"):
        files.append(DEGRADER)
    return files


def _translation_problems(test: dict) -> list[str]:
    """translated siblings that are missing, stale for this degrader, or lack norm/anchor entries."""
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
    """static external files that must exist before the campaign starts."""
    missing = sorted({
        path for test in selected for path in _external_files(test) if not (ROOT / path).is_file()
    })
    if not missing:
        return
    message = f"campaign 40 inputs not ready: missing={missing}"
    if not dry_run:
        raise FileNotFoundError(message)
    print(f"[campaign40] WARNING {message}", flush=True)


def _ensure_downsampled_inputs(test: dict, dry_run: bool) -> None:
    """build plan-R translated siblings on demand when the first downsampled arm starts."""
    problems = sorted(_translation_problems(test))
    if not problems:
        return
    command = [sys.executable, str(ROOT / "assemble_training_segments.py"), "--degrader", DEGRADER]
    if dry_run:
        print(
            f"[campaign40] would translate fine-scan siblings for {test['tid']} with: {' '.join(command)}",
            flush=True,
        )
        return
    print(f"[campaign40] preparing translated siblings for {test['tid']}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    remaining = sorted(_translation_problems(test))
    if remaining:
        raise FileNotFoundError(f"campaign 40 translated inputs still not ready: {remaining}")


def _ensure_noise_bank(test: dict, dry_run: bool) -> None:
    """build the native-noise bank only for the arm that actually uses it."""
    bank = test["config"].get("data.native_noise")
    if not bank:
        return
    paths = [ROOT / str(bank), ROOT / str(Path(str(bank)).with_suffix(".json"))]
    if all(path.is_file() for path in paths):
        return
    command = [sys.executable, str(ROOT / "crossres" / "build_noise_bank.py")]
    if dry_run:
        print(
            f"[campaign40] would build native noise bank for {test['tid']} with: {' '.join(command)}",
            flush=True,
        )
        return
    print(f"[campaign40] building native noise bank for {test['tid']}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"campaign 40 noise bank still missing after build: {missing}")


def _ensure_arm_inputs(test: dict, dry_run: bool) -> None:
    """prepare deferred inputs at the arm boundary instead of blocking campaign startup."""
    _ensure_downsampled_inputs(test, dry_run)
    _ensure_noise_bank(test, dry_run)


def _preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    """campaign 40 never retrains MAEs automatically; selected arms must reuse an existing checkpoint."""
    missing = []
    for key in dict.fromkeys(str(test["pretrain_key"]) for test in selected):
        if not campaign34._pretraining_complete(key):
            missing.append(str(campaign34._pretrain_path(key).relative_to(ROOT)))
    if not missing:
        return
    message = f"campaign 40 expects existing MAE checkpoints; missing or incomplete={missing}"
    if not dry_run:
        raise FileNotFoundError(message)
    print(f"[campaign40] WARNING {message}", flush=True)


def _selected_scrolls(selected: list[dict]) -> list[ScrollConfig]:
    ordered: dict[int, ScrollConfig] = {}
    for test in selected:
        for scroll in test["scrolls"]:
            ordered[int(scroll.scroll_id)] = scroll
    return list(ordered.values())


def _surface_anchor_ids(selected: list[dict]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(
        [int(scroll.scroll_id) for scroll in _selected_scrolls(selected)]
        + [int(scroll_id) for scroll_id in VIS_SCROLL_IDS]
        + [
            scroll_id
            for test in selected
            for scroll_id in _pretrain_scroll_ids(test["train_scroll_dict"])
        ]
    ))


def _smoke_config(config) -> list[int]:
    config.tra.n_epochs = 1
    config.tra.eval_int = 10**9
    config.dl.num_workers = 2
    config.data.max_samples_per_epoch = 2 * config.dl.num_workers * config.dl.batch_size
    config.init_weights = None
    config.model.require_architecture_init = False
    domain, weight = next(
        (name, weight)
        for (name, ids), weight in zip(config.data.train_scroll_dict.items(), config.data.train_scroll_weights)
        if SMOKE_SCROLL in map(int, ids)
    )
    config.data.train_scroll_dict = {domain: [SMOKE_SCROLL]}
    config.data.train_scroll_weights = [weight]
    config.data.holdout_domains = []
    dropped = [int(scroll.scroll_id) for scroll in config.data.scrolls if int(scroll.scroll_id) != SMOKE_SCROLL]
    config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) == SMOKE_SCROLL]
    config.data.vis_scroll_ids = [SMOKE_SCROLL]
    config.data.vis_preload_persistent = False
    config.data.ram_safe_vis = True
    config.tra.eval_int_scrolls = 0
    config.tra.dann_n_domains = 1
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="campaign 40: standalone combined+surface-anchor native-96 held-out tests"
    )
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one short epoch per arm on pherc0814 alone from random init, no figures, separate log/model dirs",
    )
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs40_smoke", "models/archs40_smoke"

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
            _selected_scrolls(selected),
            inklabel_dir=ROOT / INKLABEL_DIR,
            strict=not (args.dry_run or args.smoke),
        )
        if not args.smoke:
            preflight_external(selected, args.dry_run)
            ensure_surface_anchors(_surface_anchor_ids(selected), str(ROOT / "ves_zarrs2"))
            _preflight_pretraining(selected, args.dry_run)
        print(f"[campaign40] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        if not args.smoke:
            _ensure_arm_inputs(test, args.dry_run)
        config = build_config(test)
        dropped = _smoke_config(config) if args.smoke else []
        if args.smoke and getattr(config.model, "input_upsampler", "") and not (ROOT / config.model.input_upsampler).is_file():
            config.model.input_upsampler = "trilinear"
        with startup_output():
            print(
                f"[campaign40] {test['tid']}: ctx={config.data.context_size} "
                f"domains={list(config.data.train_scroll_dict)} "
                f"weights={config.data.train_scroll_weights} norm={config.data.norm_mode} "
                f"upsampler={config.model.input_upsampler or 'none'} "
                f"translated={len(config.data.zarr_suffix)} batch={config.dl.batch_size} "
                f"lr={config.tra.lr} init={config.init_weights} dann={config.tra.dann_n_domains} "
                f"dropped={len(dropped)} overrides={test['config']}",
                flush=True,
            )
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

    print(f"\n{'=' * 78}\n[campaign40] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()