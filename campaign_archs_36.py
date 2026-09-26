"""campaign 36: memorisation, label geometry, sampling and context invariance on held-out scrolls

Same protocol as campaign 35: pherc0841 and pherc0009b are held out and rendered at full
extent after the final epoch, and every arm uses the early-gated patch-GroupDRO recipe. The
default field is 192 px at native resolution (batch 32 / lr 1e-4, eval batch 64 / 2 GB chunks).
Cutout, context replacement, context jitter and depth jitter stay off unless an arm turns one
on; flips/rotations and the default dropout (0.05 / 0.05 / head 0.1) stay on.

| arm                         | what it changes                                                  |
|-----------------------------|------------------------------------------------------------------|
| holdout_baseline_rep        | none; replicate for noise                                        |
| holdout_baseline_rep_2      | none; renders every scroll (training and held out) at the end    |
| holdout_ring_c2g2s4         | ring labels close 2 / gap 2 / shell 4 (campaign 33+ uses 0/0/4)  |
| holdout_ring_c2g2s4_gce     | the same ring with GCE loss, q=0.7                               |
| holdout_ring_c2g2s4_edge_soft | the same ring; positive cells grazing a stroke edge get soft   |
|                             | targets (floor 0.6); cores train to 1 and negatives to 0 (no     |
|                             | label smoothing, which every other arm has at 0.1 / 0.05)        |
| holdout_native96            | (arch) 96 px field at native resolution, own MAE                 |
| holdout_native96_ring_c2g2s4_edge_soft | native 96 with the c2g2s4 ring and soft edges         |
| holdout_narrow2d            | (arch) early 2D U-Net at half width, own MAE                     |
| holdout_dropout             | dropout 0.2 / 0.2 / head 0.3 instead of 0.05 / 0.05 / 0.1        |
| holdout_sampler_weights     | round robin 0139 x2, 0343p x4, 0500p2 x3, 0814 x5, others x1     |
| holdout_and_mask            | batch split into 4 random domain groups, each its own sub-batch; |
|                             | only weights whose gradient sign >=3 of 4 groups share update    |
| holdout_fishr               | Fishr on the output head: per-domain variances of per-sample     |
|                             | head gradients pulled together (lambda 1, from epoch 1)          |
| holdout_private_heads       | per-domain residual output heads in training only; loss on       |
|                             | shared+private plus 0.5 x shared alone; inference uses shared    |
| holdout_cross_rank          | ink cells must outrank negative cells from other scrolls (0.2)   |
| holdout_spectral_decoupling | L2 0.01 on supervised cell logits (anti gradient starvation)     |
| holdout_rsc                 | RSC: on 1/3 of samples mute the top 1/3 head-input channels or   |
|                             | positions that most support the correct logit                    |
| holdout_fish                | Fish: one optimizer step per domain group in sequence, then move |
|                             | halfway from the start to the end of that inner loop             |
| holdout_fish_and_mask       | Fish whose meta step keeps only coordinates where >=3 of 4 inner |
|                             | steps agree in sign                                              |
| holdout_quality_norm        | fine scans filtered to the 9.36 um scanners' measured spectrum   |
| holdout_fiber               | (arch) structure-tensor fibre inputs, own MAE                    |
| holdout_randconv            | aggressive random convolution texture re-rendering               |
| holdout_ema_0995            | weight EMA, decay 0.995 (~200-step horizon, as in c29/c31)       |
| holdout_surface_relief      | papyrus-air boundary geometry at enc1: relief, roughness, step,  |
|                             | missing top layer, lifted flakes, edge sharpness                 |
| holdout_ctx_replace_cross   | context replacement 0.5 with donors from other physical domains  |
| holdout_rotate_any          | context rotated by an arbitrary angle about the target           |
| holdout_elastic_strong      | strong smooth elastic displacement of the context (32 px peak)   |
| holdout_researcher_head     | (arch) 6-stage residual 2D U-Net, strided downsampling, own MAE  |
| holdout_label_shift         | control: training labels rolled half the scroll height off ink   |

Rotation and elastic warps leave the 64 px prediction centre plus an 8 px margin exactly in
place and reach full strength 24 px further out, so multitile targets stay valid without
warping labels; the centre still receives the exact 90-degree rotations and flips.
Every architecture is MAE-pretrained on the current corpus before its first arm; arms that keep
the default architecture share the default checkpoint.

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
# default field: 192 px at native resolution (native196 was the best full-extent 0814 reader);
# full scale needs batch 32 / lr 1e-4 and a smaller inference batch
NATIVE192 = {"pretrain_key": "early_gated_native192", "context_size": 192, "context_downsample": 1,
             "batch_size": 32, "lr": 1e-4}
FULL_SCALE_EVAL = {"data.eval_infer_bs": 64, "data.eval_chunk_gb": 2.0}
NATIVE96 = {"pretrain_key": "early_gated_native96", "context_size": 96, "context_downsample": 1}
RESEARCHER_KEY = "early_gated_researcher"
RESEARCHER_MODEL = {
    "model.residual_2d_unet": True,
    "model.two_d_extra_channels": (320, 320),
    "model.two_d_strided_down": True,
}
# every arm's MAE pretrains on the current corpus (the campaign-33/34 checkpoints predate 20230205142449)
campaign34.PRETRAIN_SPECS.update({
    "early_gated_native192": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
    ),
    RESEARCHER_KEY: campaign31._spec(
        *campaign34.EARLY_GATED_ARGS,
        "--residual-2d-unet", "--two-d-extra-channels", "320", "320", "--two-d-strided-down",
        required=campaign34.EARLY_GATED_REQUIRED + ("early2d_down.", "early2d_extra_encoders."),
        ctx=192, ds=1,
    ),
    "early_gated_native96": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
    ),
    "early_gated_narrow": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--early-2d-channels-mult", "0.5",
        required=campaign34.EARLY_GATED_REQUIRED, ctx=192, ds=1,
    ),
    "early_gated_fiber": campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, "--fiber-coordinate-branch",
        required=campaign34.EARLY_GATED_REQUIRED + ("fiber_coordinate_input.",), ctx=192, ds=1,
    ),
})
# the default 192 px surround (64 px per side) takes the campaign-33 replacement geometry;
# the protected warps double their native-128 pixel geometry for the 2x surround
CONTEXT_REPLACE = {
    "dl.context_replace_prob": 0.5,
    "dl.context_replace_margin": 20, "dl.context_replace_feather": 40,
}
PROTECTED_WARP = {"dl.protected_warp_margin": 8, "dl.protected_warp_feather": 24}
RING_C2G2S4 = {"data.ring_close_r": 2, "data.ring_gap_r": 2, "data.ring_shell_r": 4}
MORE_DROPOUT = {"model.conv1_drop": 0.2, "model.conv2_drop": 0.2, "model.head_drop": 0.3}
SAMPLER_WEIGHTS = {"pherc0139": 2, "pherc0343p": 4, "pherc0500p2": 3, "pherc0814": 5}
# cores train to exactly 1 and every negative to exactly 0; only edge-grazing positives go soft
EDGE_SOFT = {
    "data.edge_soft_sigma": 4.0, "data.edge_soft_floor": 0.6,
    "tra.label_smooth_pos": 0.0, "tra.label_smooth_neg": 0.0,
}
ALL_SCROLL_EVAL = {
    "data.vis_scroll_ids": list(campaign33.CAMPAIGN33_SCROLL_IDS),
    "tra.eval_int_scrolls": len(campaign33.CAMPAIGN33_SCROLL_IDS),
    "data.vis_preload_persistent": False,
    # stream every volume from zarr during the final render instead of materialising it in RAM
    "data.ram_safe_vis": True,
}


def _test(tid: str, augmentations: dict, arch: dict | None = None, **extra) -> dict:
    arch = arch or NATIVE192
    if arch["context_size"] == 192 and arch["context_downsample"] == 1:
        augmentations = {**FULL_SCALE_EVAL, **augmentations}
    test = campaign35._test(tid, augmentations, arch=arch, **extra)
    test["tag"] = f"36_{tid}"
    return test


TESTS = [
    _test("holdout_baseline_rep", {}),
    # the only arm that renders every scroll, training and held out, after the final epoch
    _test("holdout_baseline_rep_2", ALL_SCROLL_EVAL),
    _test("holdout_ring_c2g2s4", RING_C2G2S4),
    _test("holdout_ring_c2g2s4_gce", {**RING_C2G2S4, "tra.loss_type": "gce", "tra.gce_q": 0.7}),
    _test("holdout_ring_c2g2s4_edge_soft", {**RING_C2G2S4, **EDGE_SOFT}),
    _test("holdout_native96", {}, arch=NATIVE96),
    _test("holdout_native96_ring_c2g2s4_edge_soft", {**RING_C2G2S4, **EDGE_SOFT}, arch=NATIVE96),
    _test("holdout_narrow2d", {"model.early_2d_channels_mult": 0.5},
          arch={**NATIVE192, "pretrain_key": "early_gated_narrow"}),
    _test("holdout_dropout", MORE_DROPOUT),
    _test("holdout_sampler_weights", {"data.train_scroll_weights": [
        SAMPLER_WEIGHTS.get(domain, 1) for domain in campaign33.CAMPAIGN33_SCROLL_DICT
    ]}),
    _test("holdout_and_mask", {"tra.and_mask": True, "tra.and_mask_groups": 4, "tra.and_mask_threshold": 0.5}),
    _test("holdout_fishr", {"tra.fishr_lambda": 1.0, "tra.fishr_warmup_epochs": 1}),
    _test("holdout_private_heads", {
        "model.private_domain_heads": True,
        "tra.private_head_shared_weight": 0.5, "tra.private_head_l2": 0.01,
    }),
    _test("holdout_cross_rank", {"tra.cross_scroll_rank_lambda": 0.2, "tra.cross_scroll_rank_pairs": 4096}),
    _test("holdout_spectral_decoupling", {"tra.spectral_decoupling_lambda": 0.01}),
    _test("holdout_rsc", {"tra.rsc_prob": 0.33, "tra.rsc_drop_frac": 0.33}),
    _test("holdout_fish", {"tra.fish": True, "tra.fish_meta_step": 0.5, "tra.and_mask_groups": 4}),
    _test("holdout_fish_and_mask", {
        "tra.fish": True, "tra.fish_meta_step": 0.5, "tra.and_mask": True,
        "tra.and_mask_groups": 4, "tra.and_mask_threshold": 0.5,
    }),
    _test("holdout_quality_norm", {}, quality_normalize=True),
    _test("holdout_fiber", {"model.fiber_coordinate_branch": True},
          arch={**NATIVE192, "pretrain_key": "early_gated_fiber"}),
    _test("holdout_randconv", campaign35.RANDCONV),
    _test("holdout_ema_0995", {"tra.model_ema": True, "tra.model_ema_decay": 0.995}),
    # the MAE runs without surface maps, so only this zero-initialised 1x1 conv starts untrained
    _test("holdout_surface_relief", {"model.surface_relief_input": True}),
    _test("holdout_ctx_replace_cross", {**CONTEXT_REPLACE, "dl.context_replace_cross_prob": 1.0}),
    _test("holdout_rotate_any", {**PROTECTED_WARP, "dl.protected_rotation_prob": 0.8}),
    _test("holdout_elastic_strong", {
        **PROTECTED_WARP, "dl.protected_elastic_prob": 0.8, "dl.protected_elastic_alpha": 32.0,
        "dl.protected_elastic_sigma": 10.0,
    }),
    _test("holdout_researcher_head", RESEARCHER_MODEL, arch={**NATIVE192, "pretrain_key": RESEARCHER_KEY}),
    # training labels moved off the ink; if its train fit matches the baseline, that fit is memorised papyrus
    _test("holdout_label_shift", {"data.label_shift_frac": 0.5}),
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


def _smoke_config(config) -> list[int]:
    """shrink an arm to a code-path check; returns scroll ids dropped for lacking an assembled zarr."""
    config.tra.n_epochs = 1
    config.tra.eval_int = 1
    config.tra.fast_eval_figure = True
    config.data.max_samples_per_epoch = 96
    # smoke tests exercise code paths, so they skip the MAE pretrains
    config.init_weights = None
    config.model.require_architecture_init = False
    root = Path(campaign33.ROOT) / "ves_zarrs2"
    missing = [
        scroll_id for ids in config.data.train_scroll_dict.values() for scroll_id in ids
        if not (root / f"{scroll_id}.zarr").is_dir()
    ]
    if missing:
        groups = [
            (name, [s for s in ids if s not in missing], weight)
            for (name, ids), weight in zip(config.data.train_scroll_dict.items(), config.data.train_scroll_weights)
        ]
        groups = [group for group in groups if group[1]]
        config.data.train_scroll_dict = {name: ids for name, ids, _ in groups}
        config.data.train_scroll_weights = [weight for _, _, weight in groups]
        config.data.scrolls = [s for s in config.data.scrolls if int(s.scroll_id) not in missing]
        config.data.vis_scroll_ids = [s for s in config.data.vis_scroll_ids if int(s) not in missing]
        config.tra.eval_int_scrolls = min(config.tra.eval_int_scrolls, len(config.data.vis_scroll_ids))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 36: context invariance and researcher head")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm from random init, cropped figures, separate log/model dirs")
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
                  f"ds={config.data.context_downsample} batch={config.dl.batch_size} lr={config.tra.lr} "
                  f"init={config.init_weights} dropped={dropped} overrides={test['config']}", flush=True)
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
