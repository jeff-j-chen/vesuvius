"""campaign_finetune.py -- fine-tune the archs40 native-96 nnPU model with PHerc1447.

This starts from the campaign-40 holdout_n96_combined_surface_norm_nnpu checkpoint (native 8 x 96 x 96
input) and trains on PHerc1447 (20260930144758 / 20260930144760), PHerc0500P2 (20250628074500),
PHerc0841 (20260221022814), PHerc0009B (20250919125754), PHerc0139 w044 (20260115000000) and
PHerc0814 seg46527 (20260226000000).

| tid          | change vs the archs40 nnPU recipe                                      |
|--------------|------------------------------------------------------------------------|
| baseline     | none                                                                   |
| hard         | hard targets (no sigma-8 soft edges)                                   |
| c0r0s4       | ring close 0 / gap 0 / shell 4, multitile_pos_only=False, soft targets |
| c0r0s4_hard  | ring close 0 / gap 0 / shell 4, multitile_pos_only=False, hard targets |

PHerc0211 (20260928000003, 20260928000004) and PHerc0813 (20260814140748, 20260928000007) are
label-free visualization scrolls: never sampled for training or validation, rendered only after
the final epoch.

Splits come from train_masks/ (`simple_split=False`): the nnPU far-negative sampler requires a
manual split.

Usage:
    python3 campaign_finetune.py --dry-run
    python3 campaign_finetune.py --smoke
    python3 campaign_finetune.py
    python3 campaign_finetune.py --only hard,c0r0s4
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
DEFAULT_INIT_WEIGHTS = "/vesuvius/models/archs40/holdout_n96_combined_surface_norm_nnpu/final.pth"
PHERC0211_IDS = (20260928000003, 20260928000004)
PHERC0813_IDS = (20260814140748, 20260928000007)
PHERC1447_IDS = (20260930144758, 20260930144760)#, 20260930144762)
PHERC1447_DOMAIN = "pherc1447"
PHERC0500P2_ID = 20250628074500
PHERC0841_ID = 20260221022814
PHERC0009B_ID = 20250919125754
W044_ID = 20260115000000
SEG46527_ID = 20260226000000
ADDED_IDS = (PHERC0500P2_ID, PHERC0841_ID, PHERC0009B_ID, W044_ID, SEG46527_ID)
LABEL_FREE_VIS_IDS = [*PHERC0211_IDS, *PHERC0813_IDS]
VIS_SCROLL_IDS = [*PHERC0211_IDS, *PHERC0813_IDS, *PHERC1447_IDS]
SMOKE_LOG_DIR = "./runs_finetune_smoke"
SMOKE_MODEL_DIR = "models/finetune_smoke"

TRAIN_SCROLL_DICT = {
    PHERC1447_DOMAIN: list(PHERC1447_IDS),
    "pherc0500p2": [PHERC0500P2_ID],
    "pherc0841": [PHERC0841_ID],
    "pherc0009b": [PHERC0009B_ID],
    "pherc0139": [W044_ID],
    "pherc0814": [SEG46527_ID],
}
SCROLLS = [
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
    # ScrollConfig(PHERC1447_IDS[2], split_axis="x", train_split_frac=0.75),
    *(scroll for scroll in campaign40.SCROLLS_BASE if int(scroll.scroll_id) in ADDED_IDS),
]


HARD = {"data.edge_soft_sigma": 0.0}
RING_C0R0S4 = {
    "data.ring_close_r": 0,
    "data.ring_gap_r": 0,
    "data.ring_shell_r": 4,
    "data.multitile_pos_only": False,
}


def _test(tid: str, changes: dict, init_weights: str) -> dict:
    test = campaign40._test(
        tid,
        campaign40.SURFACE_FIBER_MAE_KEY,
        TRAIN_SCROLL_DICT,
        SCROLLS,
        changes={**campaign40.NNPU, **changes},
        init_weights=init_weights,
        batch_size=campaign40.SCRATCH_TRAINING["batch_size"],
        lr=1e-4,
    )
    test["tag"] = f"ft_{tid}"
    return test


def _tests(init_weights: str) -> list[dict]:
    return [
        _test("baseline", {}, init_weights),
        _test("hard", HARD, init_weights),
        _test("c0r0s4", RING_C0R0S4, init_weights),
        _test("c0r0s4_hard", {**RING_C0R0S4, **HARD}, init_weights),
    ]


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
    config.data.vis_scroll_ids = list(VIS_SCROLL_IDS)
    config.data.vis_preload_persistent = True
    config.data.label_free_vis_scroll_ids = list(LABEL_FREE_VIS_IDS)
    config.data.train_only_scroll_ids = []
    config.tra.n_epochs = 10
    # fully trainable at a high LR; one warmup epoch instead of five
    config.tra.lr = 1e-3
    config.tra.warmup_epochs = 1
    config.tra.freeze_prefixes = ()
    config.tra.encoder_freeze_epochs = 0
    config.tra.encoder_lr_scale = 1.0
    config.tra.encoder_lr = None
    config.tra.task_lr = None
    # figures only after the final epoch
    config.tra.eval_int = config.tra.n_epochs
    config.tra.eval_int_scrolls = len(VIS_SCROLL_IDS)
    config.tra.test_int = 999
    config.tra.probe_int = 999
    config.tra.fast_eval_figure = False
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    trained = {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}
    trained |= {int(scroll.scroll_id) for scroll in config.data.scrolls}
    leaked = trained & set(LABEL_FREE_VIS_IDS)
    if leaked:
        raise ValueError(f"vis-only scrolls {sorted(leaked)} must not be trained")
    return config


def _smoke_config(config) -> None:
    """one tiny epoch, then the end-of-training figure on the smallest vis-only scroll."""
    config.tra.n_epochs = 1
    config.tra.eval_int = 1
    config.dl.num_workers = 2
    config.data.max_samples_per_epoch = 2 * config.dl.num_workers * config.dl.batch_size
    config.data.vis_scroll_ids = [PHERC0813_IDS[1]]
    config.data.vis_preload_persistent = False
    config.tra.eval_int_scrolls = 1


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
    for scroll_id in LABEL_FREE_VIS_IDS:
        required = (
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            failures.append(f"{scroll_id} (vis-only): {', '.join(missing)}")
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
    parser = argparse.ArgumentParser(description="fine-tune archs40 native-96 nnPU model with PHerc1447")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one tiny epoch + final vis on one vis-only scroll, separate log/model dirs")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help="starting checkpoint (default: archs40 holdout_n96_combined_surface_norm_nnpu final.pth)")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = SMOKE_LOG_DIR, SMOKE_MODEL_DIR

    selected = _tests(args.init_weights)
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
        if args.smoke:
            _smoke_config(config)
        with startup_output():
            print(
                f"[campaign_finetune] {test['tid']}: ctx={config.data.context_size} "
                f"domains={list(config.data.train_scroll_dict)} "
                f"weights={config.data.train_scroll_weights} simple_split={config.data.simple_split} "
                f"fast_eval={config.tra.fast_eval_figure} vis={config.data.vis_scroll_ids} "
                f"label_free={config.data.label_free_vis_scroll_ids} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"ring={config.data.ring_close_r}/{config.data.ring_gap_r}/{config.data.ring_shell_r} "
                f"pos_only={config.data.multitile_pos_only} edge_soft={config.data.edge_soft_sigma} "
                f"upsampler={config.model.input_upsampler or 'none'} "
                f"init={config.init_weights}",
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

    print(f"\n{'=' * 78}\n[campaign_finetune] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()