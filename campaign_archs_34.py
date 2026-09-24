"""campaign 34: resolution versus field of view for the early patch-GroupDRO recipe

Labels changed after Campaign 33 (dilated_inklabels rebuilt), so nothing here is
comparable with earlier campaigns; each arm has a matched reference inside Campaign 34.
The model average-pools the input by `context_downsample` before the first conv, so the
network grid is context_size / context_downsample pixels wide:

| arm                          | raw field | downsample | network grid | batch / lr     |
|------------------------------|-----------|------------|--------------|----------------|
| early_patchdro_ds2           | 192 px    | 2          | 96           | 96 / 1.5e-4    |
| early_patchdro_native128     | 128 px    | 1          | 128          | 96 / 1.5e-4    |
| early_patchdro_native196     | 196 px    | 1          | 196          | 48 / 1.2e-4    |
| early_patchdro_ctx384_ds2    | 384 px    | 2          | 192          | 48 / 1.2e-4    |
| early_patchdro_ds2_b48       | 192 px    | 2          | 96           | 48 / 1.2e-4    |

Background false-positive arms (all at the ds2 reference settings):
`early_patchdro_far_neg` draws 15% of negatives from whole windows >=160 px from any ink;
`early_patchdro_shell6` thickens the ring-negative shell from 4 to 6 tiles;
`early_patchdro_far_neg_region_gate` adds a coarse text-region head that gates ink logits,
trained to separate far background from ink-adjacent cells; `early_patchdro_pos_weight05`
halves the positive BCE weight.

Every arm renders one evaluation figure, for w035 (pherc0139), after the final epoch; that
volume stays in RAM for the whole run.

native128 halves the surround, so cutout, context replacement and context jitter are
weakened for it. The 196/192-grid arms run at batch 48 because the 196-grid EDR arm in
Campaign 33 died before its first epoch; `early_patchdro_ds2_b48` is their matched reference.

Usage:
    python3 campaign_archs_34.py --dry-run
    python3 campaign_archs_34.py --only early_patchdro_native196
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_33 as campaign33
from utils.config import startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs34"
MODEL_DIR = "models/archs34"
LARGE_BATCH_SIZE = 48
LARGE_LR = 1.2e-4
VIS_SCROLL_ID = 20260317000000  # pherc0139 w035
EARLY_GATED_ARGS = ("--early-2d-unet", "--gated-stems", "--norm-mode", "ibn_full")
EARLY_GATED_REQUIRED = ("gated_cue_stem.", "early_depth_attn.", "early2d_")
# the pretraining corpus is unchanged since Campaign 33, so its checkpoints are reused
PRETRAIN_SPECS = {
    "early_gated_native128": campaign31._spec(
        *EARLY_GATED_ARGS, required=EARLY_GATED_REQUIRED, ctx=128, ds=1,
    ),
    "early_gated_ctx384_ds2": campaign31._spec(
        *EARLY_GATED_ARGS, required=EARLY_GATED_REQUIRED, ctx=384, ds=2,
    ),
}
# native128 keeps 2/3 of the surround of the 192/196 arms; augmentations scale with it
NATIVE128_AUG = {
    "cutout_prob": 0.30,
    "context_replace_prob": 0.25,
    "context_replace_margin": 13,
    "context_replace_feather": 26,
    "ctx_jitter": 20,
}
LARGE = {"batch_size": LARGE_BATCH_SIZE, "lr": LARGE_LR}
# 15% of negative draws come from whole windows at least 160 px from any ink label
FAR_NEGATIVES = {"data.far_negative_share": 0.15, "data.far_negative_min_dist": 160}


def _test(tid: str, pretrain_key: str, **overrides) -> dict:
    extra = {
        "batch_size": int(overrides.pop("batch_size", campaign31.FINETUNE_BATCH_SIZE)),
        "lr": float(overrides.pop("lr", campaign31.FINETUNE_LR)),
        "aug": {key: overrides.pop(key) for key in list(NATIVE128_AUG) if key in overrides},
        "config": dict(overrides.pop("config", {})),
    }
    test = campaign33._test(
        tid, pretrain_key, early_2d_unet=True, mid_2d_unet=False,
        physical_patch_groupdro=True, **overrides,
    )
    test.update(extra)
    test["tag"] = f"34_{tid}"
    return test


TESTS = [
    _test("early_patchdro_ds2", "early_gated"),
    _test("early_patchdro_native128", "early_gated_native128",
          context_size=128, context_downsample=1, **NATIVE128_AUG),
    _test("early_patchdro_native196", "early_gated_native196",
          context_size=196, context_downsample=1, **LARGE),
    _test("early_patchdro_ctx384_ds2", "early_gated_ctx384_ds2",
          context_size=384, context_downsample=2, **LARGE),
    _test("early_patchdro_ds2_b48", "early_gated", **LARGE),

    # background false positives; each compares with early_patchdro_ds2
    _test("early_patchdro_far_neg", "early_gated", config=FAR_NEGATIVES),
    _test("early_patchdro_shell6", "early_gated", config={"data.ring_shell_r": 6}),
    _test("early_patchdro_far_neg_region_gate", "early_gated", config={
        **FAR_NEGATIVES, "model.text_region_gate": True, "tra.text_region_lambda": 0.5,
    }),
    # BCE pos_weight 0.5: the training-time equivalent of a -0.69 logit shift
    _test("early_patchdro_pos_weight05", "early_gated", config={"tra.tile_pos_weight": 0.5}),
]


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_campaign34_{key}_2k"


def _pretrain_path(key: str) -> Path:
    if key in campaign33.PRETRAIN_SPECS:
        return campaign33._pretrain_path(key)
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"


def _pretrain_metadata(key: str) -> dict:
    spec = PRETRAIN_SPECS[key]
    return {
        "campaign": 34,
        "key": key,
        "steps": campaign31.PRETRAIN_STEPS,
        "scroll_ids": list(campaign33.PRETRAIN_SCROLL_IDS),
        "architecture_args": list(spec["args"]),
        "depth": spec["depth"],
        "d_start": spec["d_start"],
        "d_end": spec["d_end"],
        "ctx": spec["ctx"],
        "ds": spec["ds"],
        "from_scratch": True,
        "sampling": "physical_round_robin",
        "checkpoint": str(_pretrain_path(key).relative_to(ROOT)),
    }


def _pretraining_complete(key: str) -> bool:
    if key in campaign33.PRETRAIN_SPECS:
        return campaign33._pretraining_complete(key)
    checkpoint = _pretrain_path(key)
    marker = checkpoint.with_suffix(".complete.json")
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata == _pretrain_metadata(key) and campaign33._checkpoint_has(
        checkpoint, PRETRAIN_SPECS[key]["required"],
    )


@contextlib.contextmanager
def _campaign33_paths():
    saved = campaign33.LOG_DIR, campaign33.MODEL_DIR
    campaign33.LOG_DIR, campaign33.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign33.LOG_DIR, campaign33.MODEL_DIR = saved


def build_config(test: dict):
    key = str(test["pretrain_key"])
    with _campaign33_paths():
        config = campaign33.build_config(test)
    config.init_weights = str(_pretrain_path(key).relative_to(ROOT))
    config.dl.batch_size = int(test["batch_size"])
    config.tra.lr = float(test["lr"])
    config.data.vis_scroll_ids = [VIS_SCROLL_ID]
    config.data.vis_preload_persistent = True
    config.tra.eval_int = config.tra.n_epochs
    config.tra.eval_int_scrolls = 1
    for name, value in test["aug"].items():
        section = config.data if name == "ctx_jitter" else config.dl
        setattr(section, name, type(getattr(section, name))(value))
    for path, value in test["config"].items():
        section_name, name = path.split(".")
        section = getattr(config, section_name)
        if not hasattr(section, name):
            raise AttributeError(f"unknown config field {path}")
        setattr(section, name, value)
    return config


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    keys = list(dict.fromkeys(str(test["pretrain_key"]) for test in selected))
    for key in keys:
        if key in campaign33.PRETRAIN_SPECS and not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 34 reuses the Campaign 33 checkpoint for {key}, but it is incomplete")
    pending = [key for key in keys if key in PRETRAIN_SPECS and not _pretraining_complete(key)]
    missing = [
        scroll_id for scroll_id in campaign33.PRETRAIN_SCROLL_IDS
        if not (ROOT / "ves_zarrs2" / f"{scroll_id}.zarr").is_dir()
    ]
    if pending and missing:
        message = f"Campaign 34 pretraining needs every training and test zarr; missing={missing}"
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign34] WARNING {message}", flush=True)
    tests_by_key = {str(test["pretrain_key"]): test for test in selected}
    for key in pending:
        spec = PRETRAIN_SPECS[key]
        if dry_run:
            print(
                f"[campaign34] would pretrain {key}: ctx={spec['ctx']} ds={spec['ds']} "
                f"{campaign31.PRETRAIN_STEPS} steps batch={campaign31.PRETRAIN_BATCH_SIZE} "
                f"lr={campaign31.PRETRAIN_LR}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "mae_pretrain_nnunet.py"),
            "--name", _pretrain_name(key),
            "--scroll-ids", *(str(value) for value in campaign33.PRETRAIN_SCROLL_IDS),
            "--require-all-scrolls",
            "--physical-round-robin",
            "--ctx", str(spec["ctx"]),
            "--ds", str(spec["ds"]),
            "--depth", str(spec["depth"]),
            "--d-start", str(spec["d_start"]),
            "--d-end", str(spec["d_end"]),
            "--steps", str(campaign31.PRETRAIN_STEPS),
            "--batch-size", str(campaign31.PRETRAIN_BATCH_SIZE),
            "--accum-steps", "1",
            "--lr", str(campaign31.PRETRAIN_LR),
            "--no-figures",
            *spec["args"],
        ]
        print(f"[campaign34] pretraining {key} from scratch", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        _pretrain_path(key).with_suffix(".complete.json").write_text(
            json.dumps(_pretrain_metadata(key), indent=2) + "\n",
            encoding="utf-8",
        )
        if not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 34 MAE pretraining failed validation: {key}")
        test = tests_by_key[key]
        fraction, matched, total, _, _ = campaign31._transfer_coverage(
            test, _pretrain_path(key), config=build_config(test)
        )
        print(f"[campaign34] {key} transfer {matched}/{total} ({fraction:.2%})", flush=True)
        if fraction < campaign31.MIN_TRANSFER_COVERAGE:
            raise RuntimeError(
                f"Campaign 34 MAE transfer below {campaign31.MIN_TRANSFER_COVERAGE:.0%} for {key}: "
                f"{matched}/{total} ({fraction:.2%})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 34: resolution versus field of view")
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

    with startup_output():
        campaign29.preflight_train_masks(
            campaign33.CAMPAIGN33_SCROLLS,
            inklabel_dir=ROOT / campaign33.INKLABEL_DIR,
            strict=not args.dry_run,
        )
        preflight_pretraining(selected, args.dry_run)
        print(f"[campaign34] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign34] {test['tid']}: ctx={config.data.context_size} "
                f"ds={config.data.context_downsample} batch={config.dl.batch_size} lr={config.tra.lr} "
                f"cutout={config.dl.cutout_prob} ctx_replace={config.dl.context_replace_prob}"
                f"/m{config.dl.context_replace_margin}/f{config.dl.context_replace_feather} "
                f"ctx_jitter={config.data.ctx_jitter}",
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

    print(f"\n{'=' * 78}\n[campaign34] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
