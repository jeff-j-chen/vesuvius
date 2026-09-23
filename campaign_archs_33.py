"""campaign 33: air-layer dependency, held-out domains, and per-slice early models

Campaign 31 found no breakthrough. Its only robust gains came from GroupDRO-style
reweighting and early depth collapse, and within-patch validation cannot detect a
shortcut that is patch-specific. Campaign 33 tests that directly, starting from a
`baseline` mid control because the labels changed after Campaign 31:

A. `mid_air_offset_m1` completes the +2/+3 window dose-response toward the papyrus
   side; the `lodo_*` arms remove one physical domain from training and score it on
   the same held-out letters it is validated on when seen; `early_patchdro_ink_band`
   centres the window on each domain's occlusion-measured ink band.
B. replicate the best Campaign 31 arm, stack EMA on it, and isolate EMA from anchor.
C. `early_planar*` makes the stem and enc1 convs per-slice so no learned convolution
   mixes depth before the early collapse.
D. `early_patchdro_native196` runs the c31 winner at 196px without XY downsampling;
   `mid_pcgrad_groups4` is c29 full PCGrad over four random domain groups per step.
E. early deep residual (EDR) and residual block-depth-3 heads with patch GroupDRO.

Usage:
    python3 campaign_archs_33.py --dry-run
    python3 campaign_archs_33.py --only early_patchdro_seed42
    python3 campaign_archs_33.py --from lodo_pherc0172_mid
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
import campaign_archs_30 as campaign30
import campaign_archs_31 as campaign31
from utils.config import ScrollConfig, startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs33"
MODEL_DIR = "models/archs33"
# characters are counted from ./inklabels when the label dir is dilated_inklabels
INKLABEL_DIR = "./dilated_inklabels"
# PHerc0139 w030, w043, w045: researcher-labelled patches added to the pherc0139 domain
NEW_PHERC0139_IDS = (20250108000005, 20260112000000, 20260126000000)
CAMPAIGN33_SCROLLS = list(campaign29.CAMPAIGN28_SCROLLS) + [
    ScrollConfig(scroll_id, split_axis="x", train_split_frac=0.75)
    for scroll_id in NEW_PHERC0139_IDS
]
PHERC0139_WEIGHT = 2
PRETRAIN_SPECS = {
    "early_gated_planar": campaign31._spec(
        "--early-2d-unet", "--gated-stems", "--norm-mode", "ibn_full",
        "--planar-early-convs",
        required=("gated_cue_stem.", "early_depth_attn.", "early2d_"),
    ),
    "early_gated_native196": campaign31._spec(
        "--early-2d-unet", "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early_depth_attn.", "early2d_"),
        ctx=196,
        ds=1,
    ),
}
EARLY = {"early_2d_unet": True, "mid_2d_unet": False}
# per-domain shift centring the 8-slice window on the ink band: round(centroid - 0.5) of the
# character-AP loss when each slice is occluded (early_gated_patch_groupdro, validation set,
# /data/extra/tmp/depth_occlusion.json); flat profiles (max loss < 0.02) stay at 0
INK_BAND_OFFSET_BY_DOMAIN = {
    "pherc0139": 0,
    "pherc0172": 0,
    "pherc1667": -1,
    "pherc0009b": -1,
    "phercparis4": 0,
    "pherc0500p2": 0,
    "pherc0814": -2,
    "phercparis2_fr143": -1,
    "pherc51cr4_fr8": 0,
    "phercparis1_fr34": -1,
    "pherc0343p": 0,
    "pherc0841": -2,
}
INK_BAND_OFFSET_BY_SCROLL = {
    int(scroll_id): offset
    for domain, offset in INK_BAND_OFFSET_BY_DOMAIN.items()
    for scroll_id in campaign29.CAMPAIGN28_SCROLL_DICT[domain]
}


def _test(tid: str, pretrain_key: str, **overrides) -> dict:
    extra = {
        "holdout_domains": tuple(overrides.pop("holdout_domains", ())),
        "planar_early_convs": bool(overrides.pop("planar_early_convs", False)),
        "depth_mask_prob": float(overrides.pop("depth_mask_prob", 0.0)),
        "depth_mask_mode": str(overrides.pop("depth_mask_mode", "zero")),
        "window_offset_by_scroll": dict(overrides.pop("window_offset_by_scroll", {})),
        "context_size": int(overrides.pop("context_size", 192)),
        "pcgrad_groups": int(overrides.pop("pcgrad_groups", 0)),
    }
    test = campaign31._test(tid, pretrain_key, **overrides)
    test.update(extra)
    test["tag"] = f"33_{tid}"
    return test


TESTS = [
    # B: replicate the best campaign 31 arm before building on it
    _test("early_patchdro_seed42", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, seed=42),

    # A1: window shifted toward the papyrus side (c31 tested +2/+3 toward air)
    _test("mid_air_offset_m1", "mid_gated_c30", surface_window_offset=-1),

    # A2: leave-one-domain-out; held-out letters match the in-domain validation set
    _test("lodo_pherc0172_mid", "mid_gated_c30", holdout_domains=("pherc0172",)),
    _test("lodo_pherc0172_early_patchdro", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, holdout_domains=("pherc0172",)),
    _test("lodo_phercparis4_mid", "mid_gated_c30", holdout_domains=("phercparis4",)),
    _test("lodo_phercparis4_early_patchdro", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, holdout_domains=("phercparis4",)),

    # B: EMA only helped combined with GroupDRO; anchor was null
    _test("early_patchdro_ema", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, model_ema=True),
    _test("mid_groupdro_ema", "mid_gated_c30",
          physical_domain_groupdro=True, model_ema=True),

    # C: no learned convolution mixes slices before the early collapse
    _test("early_planar", "early_gated_planar", **EARLY, planar_early_convs=True),
    _test("early_planar_patchdro", "early_gated_planar", **EARLY,
          planar_early_convs=True, physical_patch_groupdro=True),

    # one slice per sample (p=0.3) replaced by its neighbours' mean, never zeros
    _test("early_patchdro_depth_interp", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, depth_mask_prob=0.3, depth_mask_mode="interp"),

    # window centred per domain on the occlusion-measured ink band (train and eval)
    _test("early_patchdro_ink_band", "early_gated_c30", **EARLY,
          physical_patch_groupdro=True, window_offset_by_scroll=INK_BAND_OFFSET_BY_SCROLL),

    # c31 winner at native resolution: 196x196 context, no XY downsampling, matched pretrain
    _test("early_patchdro_native196", "early_gated_native196", **EARLY,
          physical_patch_groupdro=True, context_size=196, context_downsample=1),

    # c29 full pcgrad (train-mode, one forward, exact per-parameter projection) over four
    # random domain groups per step instead of ~12 domains: ~3.3x instead of ~12x
    _test("mid_pcgrad_groups4", "mid_gated_c30", pcgrad_groups=4),

    # E: deeper residual early heads under the c31 winner's objective; compare to early_patchdro_seed42
    _test("early_deep_residual_patchdro", "early_deep_residual_c30", **EARLY,
          residual_2d_unet=True, two_d_block_depth=3, two_d_extra_channels=(320,),
          physical_patch_groupdro=True),
    _test("early_residual_depth3_patchdro", "early_residual_depth3", **EARLY,
          residual_2d_unet=True, two_d_block_depth=3, physical_patch_groupdro=True),
]


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_campaign33_{key}_2k"


def _reference_scroll_ids() -> list[int]:
    # match the early_gated reference corpus; test scrolls added later must not leak in
    marker = campaign30._pretrain_marker("early_gated")
    return [int(value) for value in json.loads(marker.read_text(encoding="utf-8"))["scroll_ids"]]


def _pretrain_path(key: str) -> Path:
    if key in PRETRAIN_SPECS:
        return ROOT / "models" / f"{_pretrain_name(key)}.pth"
    return campaign31._pretrain_path(key)


def _pretrain_metadata(key: str) -> dict:
    spec = PRETRAIN_SPECS[key]
    return {
        "campaign": 33,
        "key": key,
        "steps": campaign31.PRETRAIN_STEPS,
        "scroll_ids": _reference_scroll_ids(),
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


def _checkpoint_has(path: Path, required) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return all(any(name.startswith(prefix) for name in state) for prefix in required)


def _pretraining_complete(key: str) -> bool:
    if key not in PRETRAIN_SPECS:
        # reference checkpoints predate the newest test scrolls, so match by content
        return _checkpoint_has(
            campaign31._pretrain_path(key),
            campaign31.PRETRAIN_SPECS[key]["required"],
        )
    checkpoint = _pretrain_path(key)
    marker = checkpoint.with_suffix(".complete.json")
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata == _pretrain_metadata(key) and _checkpoint_has(
        checkpoint,
        PRETRAIN_SPECS[key]["required"],
    )


@contextlib.contextmanager
def _campaign31_paths():
    saved = campaign31.LOG_DIR, campaign31.MODEL_DIR
    campaign31.LOG_DIR, campaign31.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign31.LOG_DIR, campaign31.MODEL_DIR = saved


def build_config(test: dict):
    key = str(test["pretrain_key"])
    # campaign 31 only knows its own pretrain keys; the planar arch swaps weights afterwards
    proxy = dict(test, pretrain_key="early_gated_c30") if key in PRETRAIN_SPECS else test
    with _campaign31_paths():
        config = campaign31.build_config(proxy)
    config.init_weights = str(_pretrain_path(key).relative_to(ROOT))
    config.data.holdout_domains = list(test["holdout_domains"])
    config.model.planar_early_convs = bool(test["planar_early_convs"])
    config.dl.depth_mask_prob = float(test["depth_mask_prob"])
    config.dl.depth_mask_mode = str(test["depth_mask_mode"])
    config.data.surface_window_offset_by_scroll = dict(test["window_offset_by_scroll"])
    config.data.context_size = int(test["context_size"])
    config.tra.pcgrad = int(test["pcgrad_groups"]) > 0
    config.tra.pcgrad_groups = int(test["pcgrad_groups"])
    config.data.eval_chunk_gb = 3.0
    config.data.eval_prefetch = 3
    config.data.inklabel_dir = INKLABEL_DIR
    config.data.ring_from_inklabel_dir = True
    config.data.multitile_ring_gate = True
    config.data.ring_close_r = 0
    config.data.ring_gap_r = 0
    config.data.scrolls = list(CAMPAIGN33_SCROLLS)
    scroll_dict = {domain: list(ids) for domain, ids in config.data.train_scroll_dict.items()}
    scroll_dict["pherc0139"] += [sid for sid in NEW_PHERC0139_IDS if sid not in scroll_dict["pherc0139"]]
    config.data.train_scroll_dict = scroll_dict
    weights = list(config.data.train_scroll_weights)
    weights[list(scroll_dict).index("pherc0139")] = PHERC0139_WEIGHT
    config.data.train_scroll_weights = weights
    return config


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    for key in dict.fromkeys(str(test["pretrain_key"]) for test in selected):
        if key not in PRETRAIN_SPECS and not _pretraining_complete(key):
            raise RuntimeError(
                f"Campaign 33 reuses the Campaign 30/31 checkpoint for {key}, but "
                f"{campaign31._pretrain_path(key)} is missing or incomplete"
            )
    for key in dict.fromkeys(
        str(test["pretrain_key"]) for test in selected
        if str(test["pretrain_key"]) in PRETRAIN_SPECS
    ):
        if _pretraining_complete(key):
            continue
        spec = PRETRAIN_SPECS[key]
        if dry_run:
            print(
                f"[campaign33] would pretrain {key}: {campaign31.PRETRAIN_STEPS} steps "
                f"batch={campaign31.PRETRAIN_BATCH_SIZE} lr={campaign31.PRETRAIN_LR}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "mae_pretrain_nnunet.py"),
            "--name", _pretrain_name(key),
            "--scroll-ids", *(str(value) for value in _reference_scroll_ids()),
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
        print(f"[campaign33] pretraining {key} from scratch", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        _pretrain_path(key).with_suffix(".complete.json").write_text(
            json.dumps(_pretrain_metadata(key), indent=2) + "\n",
            encoding="utf-8",
        )
        if not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 33 MAE pretraining failed validation: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 33: air, holdout, and planar tests")
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
            CAMPAIGN33_SCROLLS,
            inklabel_dir=ROOT / INKLABEL_DIR,
            strict=not args.dry_run,
        )
        preflight_pretraining(selected, args.dry_run)
        print(f"[campaign33] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign33] {test['tid']}: holdout={config.data.holdout_domains} "
                f"planar={config.model.planar_early_convs} "
                f"depth_mask={config.dl.depth_mask_prob}:{config.dl.depth_mask_mode} "
                f"offsets_by_scroll={config.data.surface_window_offset_by_scroll or '-'}",
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

    print(f"\n{'=' * 78}\n[campaign33] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
