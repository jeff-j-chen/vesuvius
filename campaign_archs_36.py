"""campaign 36: memorisation, label geometry, sampling and cross-scroll objectives on held-out scrolls

Same protocol as campaign 35: every scroll except pherc0841 and pherc0009b is trained on, those two
are rendered at full extent after the final epoch in every arm and kept in RAM for the run, and
every arm uses the early-gated patch-GroupDRO recipe.

Base for every arm unless the arm replaces that component: a 192 px field at native resolution,
the dual-scale head (independent local expert on the 64 px centre, mixed 0.5), ring close 2 / gap 2 /
shell 4, and softer edge positives (cores train to exactly 1, negatives to exactly 0, positive cells
grazing a stroke edge take their sigma-8 blurred ink peak, floored at 0.55; no label smoothing).
Every run tag notes both: it ends in _dual_softer. Every arm trains
at batch 32 / lr 1e-4. Cutout, context replacement, context jitter and depth jitter stay off unless
an arm turns one on; flips/rotations and dropout (0.05 / 0.05 / head 0.1) stay on.

| arm                         | what it changes                                                  |
|-----------------------------|------------------------------------------------------------------|
| holdout_dual_scale_softer   | none; the campaign-36 base (dual scale + softer sigma-8 edges)   |
| holdout_dual_softer_seed42  | replicate of the base at seed 42 (noise floor of the new base)   |
| holdout_three_scale         | second local expert on the 128 px centre, mixed 0.5 (64/128/192) |
| holdout_dual_deep           | the 64 px local expert is a 4-level 3D U-Net, not one conv block |
| holdout_rsc                 | RSC: on 1/3 of samples mute the top 1/3 head-input channels or   |
|                             | positions that most support the correct logit                    |
| holdout_spectral_decoupling | L2 0.01 on supervised cell logits (anti gradient starvation)     |
| holdout_fiber               | (arch) structure-tensor fibre inputs, own MAE                    |
| holdout_dual_mix1           | local expert mixed at 1.0 instead of 0.5                         |
| holdout_researcher_head     | (arch) 6-stage residual 2D U-Net, strided downsampling, own MAE  |
| holdout_private_heads       | per-domain residual output heads in training only; loss on       |
|                             | shared+private plus 0.5 x shared alone; inference uses shared    |
| holdout_quality_norm        | fine scans filtered to the 9.36 um scanners' measured spectrum   |
| holdout_cross_rank          | ink cells must outrank negative cells from other scrolls (0.2)   |
| holdout_surface_relief      | papyrus-air boundary geometry at enc1: relief, roughness, step,  |
|                             | missing top layer, lifted flakes, edge sharpness                 |
| holdout_label_shift         | control: training labels rolled half the scroll height off ink   |
| holdout_depth16             | (arch) 16-slice depth window instead of 8, own MAE               |
| holdout_native288           | (arch) 288 px field at native resolution, own MAE                |

Finished or dropped (kept below as comments): native128/96/ds2 dual, base_full_vis, dual_scale,
ds2, denoise, ring_c2g3s5, narrow2d, dropout, sampler_weights; and_mask, fishr, fish,
fish_and_mask and ema (invariance / re-weighting, null in every earlier campaign); randconv,
rotate_any, elastic_strong and ctx_replace_cross (perturb the surround that makes 192 beat 128).

Rotation and elastic warps leave the 64 px prediction centre plus an 8 px margin exactly in
place and reach full strength 24 px further out, so multitile targets stay valid without
warping labels; the centre still receives the exact 90-degree rotations and flips.
Every architecture is MAE-pretrained on the current corpus before its first arm; arms that keep
the base architecture share the base checkpoint. The dual-scale local expert and the surface-relief
input are not part of any MAE and start from initialisation.

Usage:
    python3 campaign_archs_36.py --dry-run
    python3 campaign_archs_36.py --only holdout_ctx_replace_cross
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
from utils.config import startup_output


LOG_DIR = "./runs_archs36"
MODEL_DIR = "models/archs36"
# every arm trains at batch 32 / lr 1e-4
TRAINING = {"batch_size": 32, "lr": 1e-4}
# every field is a multiple of 96 px
NATIVE192 = {"pretrain_key": "early_gated_native192", "context_size": 192, "context_downsample": 1, **TRAINING}
DS2_192 = {"pretrain_key": "early_gated_ds2_192", "context_size": 192, "context_downsample": 2, **TRAINING}
NATIVE128 = {"pretrain_key": "early_gated_native128", "context_size": 128, "context_downsample": 1, **TRAINING}
NATIVE96 = {"pretrain_key": "early_gated_native96", "context_size": 96, "context_downsample": 1, **TRAINING}
NATIVE288 = {"pretrain_key": "early_gated_native288", "context_size": 288, "context_downsample": 1, **TRAINING}
DEPTH16 = {"pretrain_key": "early_gated_native192_depth16", "depth": 16}
RESEARCHER_KEY = "early_gated_researcher_native192"
RESEARCHER_MODEL = {
    "model.residual_2d_unet": True,
    "model.two_d_extra_channels": (320, 320),
    "model.two_d_strided_down": True,
}
DENOISE_SIGMA = 0.6
# every arm's MAE pretrains on the current corpus (the campaign-33/34 checkpoints predate 20230205142449)
campaign34.PRETRAIN_SPECS.update({
    "early_gated_native96": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
    ),
    "early_gated_native192": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
    ),
    "early_gated_native288": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=288, ds=1,
    ),
    # centred on the same slices as the 8-slice window (10-18)
    "early_gated_native192_depth16": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
        depth=16, d_start=6, d_end=22,
    ),
    "early_gated_ds2_192": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=2,
    ),
    "early_gated_native192_denoise": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--input-denoise-sigma", str(DENOISE_SIGMA),
        required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
    ),
    RESEARCHER_KEY: campaign31._spec(
        *campaign34.EARLY_GATED_ARGS,
        "--residual-2d-unet", "--two-d-extra-channels", "320", "320", "--two-d-strided-down",
        required=campaign34.EARLY_GATED_REQUIRED + ("early2d_down.", "early2d_extra_encoders."),
        ctx=192, ds=1,
    ),
    "early_gated_narrow_native192": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--early-2d-channels-mult", "0.5",
        required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
    ),
    "early_gated_fiber_native192": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--fiber-coordinate-branch",
        required=campaign34.EARLY_GATED_REQUIRED + ("fiber_coordinate_input.",), ctx=192, ds=1,
    ),
})
# native 192 leaves a 64 px surround per side, the same raw-pixel geometry these were set for
CONTEXT_REPLACE = {
    "dl.context_replace_prob": 0.5,
    "dl.context_replace_margin": 20, "dl.context_replace_feather": 40,
}
PROTECTED_WARP = {"dl.protected_warp_margin": 8, "dl.protected_warp_feather": 24}
RING_C2G2S4 = {"data.ring_close_r": 2, "data.ring_gap_r": 2, "data.ring_shell_r": 4}
MORE_DROPOUT = {"model.conv1_drop": 0.2, "model.conv2_drop": 0.2, "model.head_drop": 0.3}
SAMPLER_WEIGHTS = {"pherc0139": 2, "pherc0343p": 4, "pherc0500p2": 3, "pherc0814": 5}
# cores train to exactly 1 and every negative to exactly 0; edge-grazing positives go soft
# (sigma 8 / floor 0.55 softens about half of a letter's cells, vs a third at sigma 4 / floor 0.6)
EDGE_SOFT = {
    "data.edge_soft_sigma": 8.0, "data.edge_soft_floor": 0.55,
    "tra.label_smooth_pos": 0.0, "tra.label_smooth_neg": 0.0,
}
BASE_LABELS = {**RING_C2G2S4, **EDGE_SOFT}
DUAL_SCALE = {"model.dual_scale": True, "model.dual_scale_local_size": 64, "model.dual_scale_mix": 0.5}
BASE = {**BASE_LABELS, **DUAL_SCALE}
# cutout / context replacement / context jitter (off in the base) at the config defaults
NATIVE_AUG = {
    "dl.cutout_prob": 0.5, "dl.context_replace_prob": 0.35,
    "dl.context_replace_margin": 20, "dl.context_replace_feather": 40,
    "data.ctx_jitter": 32, "data.depth_jitter": 1,
}
# campaign 34 strengths: a 128 field keeps 32 px of surround per side
NATIVE128_AUG = {
    "dl.cutout_prob": 0.30, "dl.context_replace_prob": 0.25,
    "dl.context_replace_margin": 13, "dl.context_replace_feather": 26,
    "data.ctx_jitter": 20, "data.depth_jitter": 1,
}
# a 96 field keeps 16 px of surround per side, half of 128's
NATIVE96_AUG = {
    "dl.cutout_prob": 0.20, "dl.context_replace_prob": 0.15,
    "dl.context_replace_margin": 7, "dl.context_replace_feather": 13,
    "data.ctx_jitter": 10, "data.depth_jitter": 1,
}
ALL_SCROLL_EVAL = {
    "data.vis_scroll_ids": list(campaign33.CAMPAIGN33_SCROLL_IDS),
    "tra.eval_int_scrolls": len(campaign33.CAMPAIGN33_SCROLL_IDS),
    "data.vis_preload_persistent": False,
    # stream every volume from zarr during the final render instead of materialising it in RAM
    "data.ram_safe_vis": True,
}


def _test(tid: str, changes: dict, arch: dict | None = None, **extra) -> dict:
    """the campaign-36 base (native 192, dual scale, ring c2g2s4, softer edges) plus this arm's changes."""
    test = campaign35._test(tid, {**BASE, **changes}, arch={**NATIVE192, **(arch or {})}, **extra)
    tag = f"36_{tid}"
    for note in ("dual", "softer"):
        if note not in tid:
            tag += f"_{note}"
    test["tag"] = tag
    return test


TESTS = [
    _test("holdout_dual_scale_softer", {}),
    # _test("holdout_native128_dual", {**DUAL_SCALE, **NATIVE128_AUG}, arch=NATIVE128),
    # _test("holdout_native96_dual", {**DUAL_SCALE, **NATIVE96_AUG}, arch=NATIVE96),
    # _test("holdout_ds2_dual", {**DUAL_SCALE, **NATIVE_AUG}, arch=DS2_192),
    # the only arm that renders every scroll, training and held out, streamed from zarr
    # _test("holdout_base_full_vis", ALL_SCROLL_EVAL),
    # independent local expert that only sees the 64 px prediction centre
    # _test("holdout_dual_scale", {
        # "model.dual_scale": True, "model.dual_scale_local_size": 64, "model.dual_scale_mix": 0.5,
    # }),
    # _test("holdout_ds2", {}, arch=DS2_192),
    # ds2's 2x2 pooling cuts voxel noise ~4x and shrinks each LSE bag 4x; this keeps the native
    # grid and receptive field but matches that noise reduction
    # _test("holdout_denoise", {"model.input_denoise_sigma": DENOISE_SIGMA},
        #   arch={"pretrain_key": "early_gated_native192_denoise"}),
    # _test("holdout_ring_c2g3s5", {"data.ring_close_r": 2, "data.ring_gap_r": 3, "data.ring_shell_r": 5}),
    # _test("holdout_narrow2d", {"model.early_2d_channels_mult": 0.5},
    #       arch={"pretrain_key": "early_gated_narrow_native192"}),
    # _test("holdout_dropout", MORE_DROPOUT),
    # _test("holdout_sampler_weights", {"data.train_scroll_weights": [
    #     SAMPLER_WEIGHTS.get(domain, 1) for domain in campaign33.CAMPAIGN33_SCROLL_DICT
    # ]}),
    # dropped: invariance / re-weighting objectives, null in every earlier campaign
    # _test("holdout_and_mask", {"tra.and_mask": True, "tra.and_mask_groups": 4, "tra.and_mask_threshold": 0.5}),
    # _test("holdout_fishr", {"tra.fishr_lambda": 1.0, "tra.fishr_warmup_epochs": 1}),
    # _test("holdout_fish", {"tra.fish": True, "tra.fish_meta_step": 0.5, "tra.and_mask_groups": 4}),
    # _test("holdout_fish_and_mask", {
    #     "tra.fish": True, "tra.fish_meta_step": 0.5, "tra.and_mask": True,
    #     "tra.and_mask_groups": 4, "tra.and_mask_threshold": 0.5,
    # }),
    # _test("holdout_ema_0995", {"tra.model_ema": True, "tra.model_ema_decay": 0.995}),
    # dropped: they perturb or replace the surround that makes 192 beat 128
    # _test("holdout_randconv", campaign35.RANDCONV),
    # _test("holdout_rotate_any", {**PROTECTED_WARP, "dl.protected_rotation_prob": 0.8}),
    # _test("holdout_elastic_strong", {
    #     **PROTECTED_WARP, "dl.protected_elastic_prob": 0.8, "dl.protected_elastic_alpha": 32.0,
    #     "dl.protected_elastic_sigma": 10.0,
    # }),
    # _test("holdout_ctx_replace_cross", {**CONTEXT_REPLACE, "dl.context_replace_cross_prob": 1.0}),

    # the base again at another seed: the only way to tell a recall gain from run-to-run noise
    _test("holdout_dual_softer_seed42", {}, arch={"seed": 42}),
    # dual scale is the only head change that found new ink; more and stronger local experts
    _test("holdout_three_scale", {"model.dual_scale_outer_size": 128, "model.dual_scale_outer_mix": 0.5}),
    _test("holdout_dual_deep", {"model.dual_scale_deep": True}),
    # force the head onto evidence it would otherwise ignore (gradient starvation)
    _test("holdout_rsc", {"tra.rsc_prob": 0.33, "tra.rsc_drop_frac": 0.33}),
    _test("holdout_spectral_decoupling", {"tra.spectral_decoupling_lambda": 0.01}),
    _test("holdout_fiber", {"model.fiber_coordinate_branch": True},
          arch={"pretrain_key": "early_gated_fiber_native192"}),
    _test("holdout_dual_mix1", {"model.dual_scale_mix": 1.0}),
    _test("holdout_researcher_head", RESEARCHER_MODEL, arch={"pretrain_key": RESEARCHER_KEY}),
    _test("holdout_private_heads", {
        "model.private_domain_heads": True,
        "tra.private_head_shared_weight": 0.5, "tra.private_head_l2": 0.01,
    }),
    _test("holdout_quality_norm", {}, quality_normalize=True),
    _test("holdout_cross_rank", {"tra.cross_scroll_rank_lambda": 0.2, "tra.cross_scroll_rank_pairs": 4096}),
    # the MAE runs without surface maps, so only this zero-initialised 1x1 conv starts untrained
    _test("holdout_surface_relief", {"model.surface_relief_input": True}),
    # each arm below changes the prepared-dataset cache key, so each forces one full reload
    # training labels moved off the ink; if its train fit matches the baseline, that fit is memorised papyrus
    _test("holdout_label_shift", {"data.label_shift_frac": 0.5}),
    _test("holdout_depth16", {}, arch=DEPTH16),
    _test("holdout_native288", {}, arch=NATIVE288),
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


SMOKE_SCROLL = 20260226000000  # pherc0814


def _smoke_config(config) -> list[int]:
    """shrink an arm to a code-path check on pherc0814 alone; returns the scroll ids dropped."""
    config.tra.n_epochs = 1
    # no epoch is a multiple of this, so no evaluation figure is rendered
    config.tra.eval_int = 10**9
    # the train set is sharded per worker with drop_last, so each worker needs whole batches
    config.dl.num_workers = 2
    config.data.max_samples_per_epoch = 2 * config.dl.num_workers * config.dl.batch_size
    # smoke tests exercise code paths, so they skip the MAE pretrains
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
    dropped = [int(s.scroll_id) for s in config.data.scrolls if int(s.scroll_id) != SMOKE_SCROLL]
    config.data.scrolls = [s for s in config.data.scrolls if int(s.scroll_id) == SMOKE_SCROLL]
    config.data.vis_scroll_ids = [SMOKE_SCROLL]
    config.data.vis_preload_persistent = False
    config.data.ram_safe_vis = True
    config.tra.eval_int_scrolls = 0
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 36: context invariance and researcher head")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm on pherc0814 alone from random init, no figures, "
                             "separate log/model dirs")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs36_smoke", "models/archs36_smoke"
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
        print(f"[campaign36] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        dropped = _smoke_config(config) if args.smoke else []
        with startup_output():
            print(f"[campaign36] {test['tid']}: ctx={config.data.context_size} "
                  f"ds={config.data.context_downsample} dual_scale={config.model.dual_scale} "
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

    print(f"\n{'=' * 78}\n[campaign36] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
