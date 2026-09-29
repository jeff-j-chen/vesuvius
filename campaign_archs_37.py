"""campaign 37: native 96 px baseline on held-out scrolls

Same protocol as campaigns 35/36: every scroll except pherc0841 and pherc0009b is trained on, those
two are rendered at full extent after the final epoch in every arm and kept in RAM for the run, and
every arm uses the early-gated patch-GroupDRO recipe.

Campaign 36 showed native 96 matches native 192 on held-out recall (0009b) and trails it only
slightly on 0841; the larger field mostly sharpens letters. Base for every arm: a 96 px field at
native resolution, ring close 2 / gap 2 / shell 4, and softer edge positives (sigma-8 blurred ink
peak, floored at 0.55; cores 1, negatives 0, no label smoothing), batch 96 / lr 1.5e-4 (the
campaign-36 native96 recipe). Cutout, context replacement, context jitter and depth jitter are off
in every arm except holdout_n96_dropout_augs; flips/rotations and dropout 0.05 / 0.05 / head 0.1
stay on.

| arm                             | what it changes                                               |
|---------------------------------|---------------------------------------------------------------|
| holdout_n96                     | none; the campaign-37 base                                    |
| holdout_n96_dual                | dual-scale local expert on the 64 px centre, mixed 0.5        |
| holdout_n96_profile_expert      | depth-profile expert: 5x5 px box average, then convolutions   |
|                                 | along depth only; zero-initialised, mixed 0.5                 |
| holdout_n96_dual_expert_dropout | dual scale; per sample, the main score and (separately) the   |
|                                 | expert score are each dropped with p 0.3, never both          |
| holdout_n96_bag_rank            | each character's top half of ink cells must outrank the top   |
|                                 | half of its ring cells by margin 0.5 (lambda 0.2), added to   |
|                                 | the patch-GroupDRO loss                                       |
| holdout_n96_seed42              | the base at seed 42 (noise floor)                             |
| holdout_n96_rsc                 | RSC: on 1/3 of samples mute the top 1/3 head-input channels   |
|                                 | or positions that most support the correct logit              |
| holdout_n96_spectral_decoupling | L2 0.01 on supervised cell logits (anti gradient starvation)  |
| holdout_n96_fiber               | (arch) structure-tensor fibre inputs, own MAE                 |
| holdout_n96_private_heads       | per-domain residual output heads in training only             |
| holdout_n96_quality_norm        | fine scans filtered to the 9.36 um scanners' measured spectrum|
| holdout_n96_dropout_augs        | dropout 0.2 / 0.2 / head 0.3 plus cutout 0.2, context         |
|                                 | replacement 0.15 (margin 7 / feather 13), jitter 10, depth 1  |
| holdout_n96_label_shift         | control: training labels rolled half the scroll height off ink|
| holdout_n96_depth16             | (arch) 16-slice depth window instead of 8, own MAE            |

The dual-scale local expert is one conv block (a few px receptive field) on the 64 px centre, so it
differs from the 96 px U-Net by receptive field rather than by crop; the smaller field does not make
the two scales alike. It starts from initialisation (it is not part of the MAE), as does the
depth-profile expert. Arms are ordered so each prepared-dataset cache key is loaded once.

Usage:
    python3 campaign_archs_37.py --dry-run
    python3 campaign_archs_37.py --smoke
    python3 campaign_archs_37.py --only holdout_n96_dual
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
import campaign_archs_33 as campaign33
import campaign_archs_34 as campaign34
import campaign_archs_35 as campaign35
import campaign_archs_36 as campaign36
from utils.config import startup_output


LOG_DIR = "./runs_archs37"
MODEL_DIR = "models/archs37"
NATIVE96 = {
    "pretrain_key": "early_gated_native96", "context_size": 96, "context_downsample": 1,
    "batch_size": 96, "lr": 1.5e-4,
}
campaign34.PRETRAIN_SPECS.update({
    "early_gated_fiber_native96": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--fiber-coordinate-branch",
        required=campaign34.EARLY_GATED_REQUIRED + ("fiber_coordinate_input.",), ctx=96, ds=1,
    ),
    # centred on the same slices as the 8-slice window (10-18)
    "early_gated_native96_depth16": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
        depth=16, d_start=6, d_end=22,
    ),
})
BASE = {**campaign36.RING_C2G2S4, **campaign36.EDGE_SOFT}
# the campaign-36 native96 strengths: a 96 field keeps 16 px of surround per side
DROPOUT_AUGS = {**campaign36.MORE_DROPOUT, **campaign36.NATIVE96_AUG}


def _test(tid: str, changes: dict, arch: dict | None = None, **extra) -> dict:
    """the campaign-37 base (native 96, ring c2g2s4, softer edges) plus this arm's changes."""
    test = campaign35._test(tid, {**BASE, **changes}, arch={**NATIVE96, **(arch or {})}, **extra)
    test["tag"] = f"37_{tid}"
    return test


TESTS = [
    _test("holdout_n96", {}),
    _test("holdout_n96_dual", campaign36.DUAL_SCALE),
    # a third score from denoised slice columns: ink changes the density profile through the sheet
    _test("holdout_n96_profile_expert", {
        "model.depth_profile_expert": True, "model.depth_profile_mix": 0.5, "model.depth_profile_pool": 5,
    }),
    # keeps the two paths' errors different by making each predict alone on some samples
    _test("holdout_n96_dual_expert_dropout", {**campaign36.DUAL_SCALE, "model.expert_dropout": 0.3}),
    _test("holdout_n96_bag_rank", {
        "tra.character_bag_ranking": True, "tra.character_bag_margin": 0.5,
        "tra.character_bag_topk_frac": 0.5, "tra.character_bag_lambda": 0.2,
    }),
    _test("holdout_n96_seed42", {}, arch={"seed": 42}),
    # force the head onto evidence it would otherwise ignore (gradient starvation)
    _test("holdout_n96_rsc", {"tra.rsc_prob": 0.33, "tra.rsc_drop_frac": 0.33}),
    _test("holdout_n96_spectral_decoupling", {"tra.spectral_decoupling_lambda": 0.01}),
    _test("holdout_n96_fiber", {"model.fiber_coordinate_branch": True},
          arch={"pretrain_key": "early_gated_fiber_native96"}),
    _test("holdout_n96_private_heads", {
        "model.private_domain_heads": True,
        "tra.private_head_shared_weight": 0.5, "tra.private_head_l2": 0.01,
    }),
    _test("holdout_n96_quality_norm", {}, quality_normalize=True),
    # each arm below changes the prepared-dataset cache key, so each forces one full reload
    _test("holdout_n96_dropout_augs", DROPOUT_AUGS),
    # training labels moved off the ink; if its train fit matches the baseline, that fit is memorised papyrus
    _test("holdout_n96_label_shift", {"data.label_shift_frac": 0.5}),
    _test("holdout_n96_depth16", {}, arch={"pretrain_key": "early_gated_native96_depth16", "depth": 16}),
]


@contextlib.contextmanager
def _campaign35_paths():
    saved = campaign35.LOG_DIR, campaign35.MODEL_DIR
    campaign35.LOG_DIR, campaign35.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign35.LOG_DIR, campaign35.MODEL_DIR = saved


def build_config(test: dict):
    with _campaign35_paths():
        return campaign35.build_config(test)


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 37: native 96 baseline on held-out scrolls")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm on pherc0814 alone from random init, no figures, "
                             "separate log/model dirs")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs37_smoke", "models/archs37_smoke"
        # never re-measure quality filters from volumes that may be mid-rebuild
        campaign35._quality_sources_newer_than_cache = lambda: False

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
            inklabel_dir=Path(campaign33.ROOT) / campaign33.INKLABEL_DIR,
            strict=not (args.dry_run or args.smoke),
        )
        if not args.smoke:
            campaign34.preflight_pretraining(selected, args.dry_run)
        print(f"[campaign37] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        dropped = campaign36._smoke_config(config) if args.smoke else []
        with startup_output():
            print(f"[campaign37] {test['tid']}: ctx={config.data.context_size} "
                  f"ds={config.data.context_downsample} depth={config.data.depth} "
                  f"dual_scale={config.model.dual_scale} "
                  f"edge_soft={config.data.edge_soft_sigma}/{config.data.edge_soft_floor} "
                  f"batch={config.dl.batch_size} lr={config.tra.lr} "
                  f"init={config.init_weights} dropped={len(dropped)} overrides={test['config']}", flush=True)
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

    print(f"\n{'=' * 78}\n[campaign37] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
