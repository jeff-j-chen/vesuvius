"""campaign 36: context invariance, ring geometry and a researcher-style 2D head on held-out scrolls

Same protocol as campaign 35: pherc0841 and pherc0009b are held out and rendered at full
extent after the final epoch, every arm uses the native-128 early-gated patch-GroupDRO
recipe, and the in-scroll augmentations are off unless an arm turns one on.

| arm                         | what it changes                                                  |
|-----------------------------|------------------------------------------------------------------|
| holdout_baseline_rep        | none; replicate of campaign 35 holdout_baseline for noise        |
| holdout_ctx_replace         | context replacement 0.5 with same-scroll donors (control)        |
| holdout_ctx_replace_cross   | context replacement 0.5 with donors from other physical domains  |
| holdout_rotate_any          | context rotated by an arbitrary angle about the target           |
| holdout_elastic_strong      | strong smooth elastic displacement of the context (16 px peak)   |
| holdout_ring_c2g2s4         | ring labels close 2 / gap 2 / shell 4 (campaign 33+ uses 0/0/4)  |
| holdout_researcher_head     | 6-stage residual 2D U-Net, strided-conv downsampling, 4x4x320    |

Rotation and elastic warps leave the 64 px prediction centre plus a 4 px margin exactly in
place and reach full strength 12 px further out, so multitile targets stay valid without
warping labels; the centre still receives the exact 90-degree rotations and flips.
The researcher head keeps our stem, depth collapse and multitile output; it gets its own
matched MAE pretrain.

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
RESEARCHER_KEY = "early_gated_native128_researcher"
RESEARCHER_MODEL = {
    "model.residual_2d_unet": True,
    "model.two_d_extra_channels": (320, 320),
    "model.two_d_strided_down": True,
}
campaign34.PRETRAIN_SPECS[RESEARCHER_KEY] = campaign31._spec(
    *campaign34.EARLY_GATED_ARGS,
    "--residual-2d-unet", "--two-d-extra-channels", "320", "320", "--two-d-strided-down",
    required=campaign34.EARLY_GATED_REQUIRED + ("early2d_down.", "early2d_extra_encoders."),
    ctx=128, ds=1,
)
CONTEXT_REPLACE = {
    "dl.context_replace_prob": 0.5,
    "dl.context_replace_margin": 13, "dl.context_replace_feather": 26,
}


def _test(tid: str, augmentations: dict, arch: dict | None = None) -> dict:
    test = campaign35._test(tid, augmentations, arch=arch)
    test["tag"] = f"36_{tid}"
    return test


TESTS = [
    _test("holdout_baseline_rep", {}),
    _test("holdout_ctx_replace", CONTEXT_REPLACE),
    _test("holdout_ctx_replace_cross", {**CONTEXT_REPLACE, "dl.context_replace_cross_prob": 1.0}),
    _test("holdout_rotate_any", {"dl.protected_rotation_prob": 0.8}),
    _test("holdout_elastic_strong", {
        "dl.protected_elastic_prob": 0.8, "dl.protected_elastic_alpha": 16.0,
        "dl.protected_elastic_sigma": 5.0,
    }),
    _test("holdout_ring_c2g2s4", {"data.ring_close_r": 2, "data.ring_gap_r": 2, "data.ring_shell_r": 4}),
    _test("holdout_researcher_head", RESEARCHER_MODEL, arch={**campaign35.NATIVE128, "pretrain_key": RESEARCHER_KEY}),
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
    parser = argparse.ArgumentParser(description="campaign 36: context invariance and researcher head")
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
            inklabel_dir=Path(campaign33.ROOT) / campaign33.INKLABEL_DIR,
            strict=not args.dry_run,
        )
        campaign34.preflight_pretraining(selected, args.dry_run)
        print(f"[campaign36] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(f"[campaign36] {test['tid']}: ctx={config.data.context_size} "
                  f"ds={config.data.context_downsample} batch={config.dl.batch_size} lr={config.tra.lr} "
                  f"init={config.init_weights} overrides={test['config']}", flush=True)
        if args.dry_run:
            success = campaign31.run_test(config, True)
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
