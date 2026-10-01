"""campaign_finetune_96.py -- fine-tune the archs40 native-96 combined-surface model.

This starts from the campaign-40 `holdout_n96_combined_surface_norm` checkpoint and keeps the
same training fragments as archs40, then adds two new physical domains:

- PHerc0211: 20260928000003 (train-only, no validation split)
- PHerc1447: 20260930144758 / 20260930144760

Unlike the archs40 campaigns, this runner keeps the repository train masks
(`simple_split=False`). That lets PHerc0211 stay train-only via `train_only_scroll_ids`
without producing validation metrics for that scroll.

Runs (tids):
1. holdout_n96_combined_surface_norm            ring (2,2,4)
2. holdout_n96_combined_surface_norm_c0g0s4     ring (0,0,4), multitile_pos_only=False (this run only)
3. holdout_n96_combined_surface_norm_f12        ring (2,2,4), enc1/enc2 frozen,
                                                enc3+bottleneck lr=1e-6, decoder+head lr=5e-5

`--fewer` restricts training to the six-fragment bootstrap subset:

- PHerc0211: 20260928000003
- PHerc1447: 20260930144758 / 20260930144760
- PHerc0841: 20260221022814
- PHerc0009B: 20250919125754
- PHerc0139 w044: 20260115000000

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
PHERC9B_ID = 20250919125754
PHERC0139_W044_ID = 20260115000000
PHERC1447_DOMAIN = "pherc1447"
ENCODER_LR_SCALE = 0.25
DEFAULT_VIS_SCROLL_IDS = [PHERC0211_ID, PHERC1447_IDS[1], PHERC0139_W044_ID]
FEWER_VIS_SCROLL_IDS = [PHERC0211_ID, PHERC1447_IDS[1], PHERC0841_ID]
SCROLL_WEIGHT_BY_ID = {
    PHERC0211_ID: 3,
}

# test 3: absolute learning rates. enc1/enc2 are fully frozen (requires_grad=False and excluded
# from the optimizer); enc3 + bottleneck are the "lower encoder" group; everything else
# (decoder + head) is the task group.
FREEZE12_LAYER_LR = {
    "freeze_prefixes": ("enc1.", "enc2."),
    "encoder_lr": 1e-6,
    "task_lr": 5e-5,
}

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
    "pherc0009b": [PHERC9B_ID],
    "pherc0139": [PHERC0139_W044_ID],
}
FEWER_SCROLLS = [
    ScrollConfig(PHERC0211_ID, split_axis="x", train_split_frac=1.0),
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC0841_ID, split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC9B_ID, split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC0139_W044_ID, split_axis="y", train_split_frac=0.8055),
]


def _train_scroll_weights(train_scroll_dict: dict[str, list[int]]) -> list[int]:
    """domain weights derived from per-scroll overrides, falling back to campaign-40 defaults."""
    weights = []
    for domain, scroll_ids in train_scroll_dict.items():
        default = int(campaign40.REGIME_WEIGHT_BY_DOMAIN.get(domain, 1))
        chosen = {
            int(SCROLL_WEIGHT_BY_ID.get(int(scroll_id), default))
            for scroll_id in scroll_ids
        }
        if len(chosen) != 1:
            raise ValueError(f"domain {domain} has inconsistent per-scroll weights: {sorted(chosen)}")
        weight = chosen.pop()
        if weight < 1:
            raise ValueError(f"domain {domain} has non-positive weight {weight}")
        weights.append(weight)
    return weights


def _check_weights(config, test: dict) -> None:
    """post-condition: the weights on the final config match the exposed per-scroll/domain weights."""
    domains = list(config.data.train_scroll_dict)
    weights = [int(w) for w in config.data.train_scroll_weights]
    expected = _train_scroll_weights(config.data.train_scroll_dict)
    if len(weights) != len(domains):
        raise ValueError(f"{test['tid']}: {len(weights)} weights for {len(domains)} domains")
    if weights != expected:
        raise ValueError(f"{test['tid']}: train_scroll_weights {weights} != expected {expected} for {domains}")
    if [int(w) for w in test["train_scroll_weights"]] != expected:
        raise ValueError(f"{test['tid']}: test weights drifted from expected {expected}")
    if int(config.tra.dann_n_domains) != len(domains):
        raise ValueError(f"{test['tid']}: dann_n_domains {config.tra.dann_n_domains} != {len(domains)} domains")


def _test(
    init_weights: str,
    fewer: bool,
    tid_suffix: str,
    close_r: int,
    gap_r: int,
    shell_r: int,
    *,
    multitile_pos_only: bool | None = None,
    layer_lr: dict | None = None,
) -> dict:
    train_scroll_dict = FEWER_TRAIN_SCROLL_DICT if fewer else DEFAULT_TRAIN_SCROLL_DICT
    scrolls = FEWER_SCROLLS if fewer else DEFAULT_SCROLLS
    test = campaign40._test(
        f"holdout_n96_combined_surface_norm{tid_suffix}",
        campaign40.SURFACE_FIBER_MAE_KEY,
        train_scroll_dict,
        scrolls,
        init_weights=init_weights,
        changes={
            "data.ring_close_r": int(close_r),
            "data.ring_gap_r": int(gap_r),
            "data.ring_shell_r": int(shell_r),
        },
        **campaign40.SCRATCH_TRAINING,
    )
    test["tag"] = (
        f"AI_{tid_suffix}"
        if fewer else f"finetune_holdout_n96_combined_surface_norm{tid_suffix}"
    )
    test["ring_close_r"] = int(close_r)
    test["ring_gap_r"] = int(gap_r)
    test["ring_shell_r"] = int(shell_r)
    test["train_scroll_weights"] = _train_scroll_weights(train_scroll_dict)
    # None -> leave whatever campaign40 / DataConfig default is in place
    test["multitile_pos_only"] = None if multitile_pos_only is None else bool(multitile_pos_only)
    test["layer_lr"] = dict(layer_lr) if layer_lr else None
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
    config.data.train_scroll_weights = list(test["train_scroll_weights"])
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.inklabel_dir = INKLABEL_DIR
    config.data.vis_scroll_ids = list(test["vis_scroll_ids"])
    config.data.vis_preload_persistent = True
    config.data.train_only_scroll_ids = [PHERC0211_ID]
    config.data.ring_close_r = int(test["ring_close_r"])
    config.data.ring_gap_r = int(test["ring_gap_r"])
    config.data.ring_shell_r = int(test["ring_shell_r"])

    # per-test override of DataConfig.multitile_pos_only (only set when the test asks for it)
    if test.get("multitile_pos_only") is not None:
        if not hasattr(config.data, "multitile_pos_only"):
            raise AttributeError("DataConfig has no attribute 'multitile_pos_only'")
        config.data.multitile_pos_only = bool(test["multitile_pos_only"])

    config.tra.n_epochs = 10
    config.tra.eval_int = 10
    config.tra.eval_int_scrolls = len(test["vis_scroll_ids"])
    config.tra.test_int = 999
    config.tra.fast_eval_figure = False
    config.tra.encoder_lr_scale = ENCODER_LR_SCALE
    config.tra.freeze_prefixes = ()
    config.tra.encoder_lr = None
    config.tra.task_lr = None
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)

    layer_lr = test.get("layer_lr")
    if layer_lr:
        # absolute per-group LRs; the optimizer turns these into lr_scale vs config.tra.lr
        config.tra.freeze_prefixes = tuple(layer_lr["freeze_prefixes"])
        config.tra.encoder_lr = float(layer_lr["encoder_lr"])
        config.tra.task_lr = float(layer_lr["task_lr"])
        config.tra.lr = config.tra.task_lr          # base lr == decoder+head lr
        config.tra.encoder_lr_scale = 1.0           # superseded by encoder_lr
        config.tra.encoder_freeze_epochs = 0

    _check_weights(config, test)
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
                        help="restrict training to PHerc0211, PHerc1447 w058/w060, PHerc0841 auto-grown 405, PHerc0009B 487, and PHerc0139 w044")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help="starting checkpoint (default: archs40 holdout_n96_combined_surface_norm final.pth)")
    args = parser.parse_args()

    selected = [
        _test(args.init_weights, args.fewer, "", 2, 2, 4),
        _test(args.init_weights, args.fewer, "_c0g0s4", 0, 0, 4, multitile_pos_only=False),
        _test(args.init_weights, args.fewer, "_f12", 2, 2, 4, layer_lr=FREEZE12_LAYER_LR),
    ]
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
                f"weights={dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))} "
                f"simple_split={config.data.simple_split} "
                f"ring=({config.data.ring_close_r},{config.data.ring_gap_r},{config.data.ring_shell_r}) "
                f"multitile_pos_only={getattr(config.data, 'multitile_pos_only', None)} "
                f"fast_eval={config.tra.fast_eval_figure} vis={config.data.vis_scroll_ids} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"encoder_lr_scale={config.tra.encoder_lr_scale} "
                f"freeze={config.tra.freeze_prefixes} encoder_lr={config.tra.encoder_lr} "
                f"task_lr={config.tra.task_lr} fewer={args.fewer} init={config.init_weights}",
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