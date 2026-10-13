"""campaign 38: native 9.36 um regime and unlabelled-surface tests on held-out scrolls

Same protocol and base as campaign 37 (logs to ./runs_archs37 beside it): a 96 px native field,
ring close 2 / gap 2 / shell 4, sigma-8 soft edges (floor 0.55), batch 96 / lr 1.5e-4, early-gated
patch-GroupDRO; pherc0841 and pherc0009b held out and rendered at full extent after the final epoch.
The campaign-37 holdout_n96 run is the baseline for every arm here.

Only three training domains were scanned natively on the 9.36 um / 1.2 m beamline like the test
scrolls (pherc0139, pherc0814, pherc0500p2); five are 2.4 / 3.24 um scans resampled to 9.36 um,
which are cleaner and dominate the round-robin.

| arm                          | what it changes                                                  |
|------------------------------|------------------------------------------------------------------|
| holdout_n96_combined         | bag rank + fiber + regime weights + RSC + dropout/augs, from a   |
|                              | c37 fiber MAE (full corpus; native-regime MAE not retrained)     |
| holdout_n96_ring_c2g1s4      | ring close 2 / gap 1 / shell 4                                   |
| ..._ring_c2g1s4_no_pos_only  | + multitile_pos_only off: ring negatives in ink windows are kept |
| ..._no_pos_only_private_detached | + per-domain private heads trained on detached shared logits |
|                              | (shared loss at full weight): only the features are shared      |
| holdout_n96_native_mae       | MAE pretrained only on natively scanned volumes (0139, 0814,     |
|                              | 0500p2, both held-out scrolls, every test scroll), no resampled  |
|                              | 2.4 / 3.24 um scans; fine-tuning is unchanged                    |
| holdout_n96_regime_weights   | round robin 0139 x2, 0343p x4, 0500p2 x3, 0814 x5, others x1     |
| holdout_n96_scan_film        | FiLM on the stem from each window's measured scan statistics     |
|                              | (mean, std, depth gradient, 8-band log radial power spectrum)    |
| holdout_n96_denoiser         | frozen self-supervised blind-spot denoiser (3x3 in-plane block   |
|                              | hidden, so noise correlated over one voxel is not copied) on the |
|                              | input; its own MAE is pretrained on denoised crops               |
| holdout_n96_surface_norm     | each scroll's gap level (1st pct) -> 0.1 and surface papyrus     |
|                              | (median at the fitted surface) -> 0.5 instead of whole-volume    |
|                              | z-score + raw min/max; own MAE on the same normalization         |
| ..._surface_norm_drop_pos    | + positive cells with papyrus at the surface (+-1 slice) under   |
|                              | 25% of pixels leave the loss: the inked layer is not in the input|
| ..._surface_norm_drop_both   | + the same for negative cells                                    |
| holdout_n96_nnpu             | far-background windows (share 0.15, >160 px from ink) enter as   |
|                              | unlabelled: non-negative PU risk with ink prior 0.03 instead of  |
|                              | BCE negatives (campaign 35's far_neg used them as negatives)     |
| holdout_n96_regime_specialist| trains on pherc0139 / 0814 / 0500p2 only                         |

Arms are ordered so each prepared-dataset cache key is loaded once: the first four share the base
data, the surface-norm trio shares its own, nnpu adds far-background windows, and the specialist
changes the scroll set. Surface anchors (surface_anchor_cache.json), the denoiser (4k steps), then
the native, denoised and surface-norm MAEs (2k steps each) are prepared before the first arm.

Usage:
    python3 campaign_archs_38.py --dry-run
    python3 campaign_archs_38.py --smoke
    python3 campaign_archs_38.py --only holdout_n96_nnpu
"""
from __future__ import annotations

import argparse
import contextlib
import gc
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
import campaign_archs_33 as campaign33
import campaign_archs_34 as campaign34
import campaign_archs_35 as campaign35
import campaign_archs_36 as campaign36
import campaign_archs_37 as campaign37
from utils.config import DEFAULT_TEST_SCROLL_IDS, startup_output
from utils.norm import ensure_surface_anchors


LOG_DIR = "./runs_archs37"
MODEL_DIR = "models/archs38"
REGIME_DOMAINS = campaign35.QUALITY_REFERENCE_DOMAINS
# unlabelled pretraining may use the held-out and test volumes, as every earlier MAE did
NATIVE_REGIME_SCROLL_IDS = tuple(dict.fromkeys(
    [
        int(scroll_id)
        for domain in (*REGIME_DOMAINS, *campaign35.HOLDOUT_DOMAINS)
        for scroll_id in campaign33.CAMPAIGN33_SCROLL_DICT[domain]
    ]
    + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
))
NATIVE_MAE_KEY = "early_gated_native96_nativeregime"
campaign34.PRETRAIN_SPECS[NATIVE_MAE_KEY] = {
    **campaign31._spec(
        *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
    ),
    "scroll_ids": NATIVE_REGIME_SCROLL_IDS,
}
# self-supervised blind-spot denoiser, trained on every pretraining volume, frozen in front of the
# model; its MAE pretrains on denoised crops so fine-tuning starts in the same input domain
DENOISER_NAME = "denoiser_n2v_block3_4k"
DENOISER_PATH = f"models/{DENOISER_NAME}.pth"
DENOISER_STEPS = 4000
DENOISED_MAE_KEY = "early_gated_native96_denoised"
campaign34.PRETRAIN_SPECS[DENOISED_MAE_KEY] = campaign31._spec(
    *campaign34.EARLY_GATED_ARGS, "--input-denoiser", DENOISER_PATH,
    required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
)
SURFACE_NORM = {"data.norm_mode": "surface_anchor"}
SURFACE_MAE_KEY = "early_gated_native96_surfacenorm"
campaign34.PRETRAIN_SPECS[SURFACE_MAE_KEY] = campaign31._spec(
    *campaign34.EARLY_GATED_ARGS, "--data-norm-mode", "surface_anchor",
    required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1,
)
SURFACE_ANCHOR_SCROLL_IDS = tuple(dict.fromkeys(
    [int(scroll_id) for scroll_id in campaign33.PRETRAIN_SCROLL_IDS]
    + [int(scroll_id) for scroll_id in campaign33.CAMPAIGN33_SCROLL_IDS]
    + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
))
REGIME_WEIGHTS = {"data.train_scroll_weights": [
    campaign36.SAMPLER_WEIGHTS.get(domain, 1) for domain in campaign33.CAMPAIGN33_SCROLL_DICT
]}
COMBINED = {
    "tra.character_bag_ranking": True, "tra.character_bag_margin": 0.5,
    "tra.character_bag_topk_frac": 0.5, "tra.character_bag_lambda": 0.2,
    "model.fiber_coordinate_branch": True,
    **REGIME_WEIGHTS,
    "tra.rsc_prob": 0.33, "tra.rsc_drop_frac": 0.33,
    **campaign37.DROPOUT_AUGS,
}
RING_C2G1S4 = {"data.ring_close_r": 2, "data.ring_gap_r": 1, "data.ring_shell_r": 4}


def _test(tid: str, changes: dict, arch: dict | None = None, **extra) -> dict:
    """the campaign-37 base plus this arm's changes."""
    test = campaign37._test(tid, changes, arch=arch, **extra)
    test["tag"] = f"38_{tid}"
    return test


TESTS = [
    # bag rank + fiber + regime weights + rsc + dropout/augs from the existing c37 fiber MAE (own cache key)
    _test("holdout_n96_combined", COMBINED, arch={"pretrain_key": "early_gated_fiber_native96"}),
    # gap 1 puts ink and ring negatives in the same window ~3-4x as often (18-24% of positive windows)
    _test("holdout_n96_ring_c2g1s4", RING_C2G1S4),
    # ...and keeps those negatives instead of dropping them from mixed windows; the gap stays ring-gated
    _test("holdout_n96_ring_c2g1s4_no_pos_only", {**RING_C2G1S4, "data.multitile_pos_only": False}),
    # private heads that shape shared features but cannot absorb the shared head's papyrus suppression
    _test("holdout_n96_ring_c2g1s4_no_pos_only_private_detached", {
        **RING_C2G1S4, "data.multitile_pos_only": False,
        "model.private_domain_heads": True, "model.private_head_detach_shared": True,
        "tra.private_head_shared_weight": 1.0, "tra.private_head_l2": 0.01,
    }),
    # _test("holdout_n96_native_mae", {}, arch={"pretrain_key": NATIVE_MAE_KEY}),
    # _test("holdout_n96_regime_weights", {"data.train_scroll_weights": [
    #     campaign36.SAMPLER_WEIGHTS.get(domain, 1) for domain in campaign33.CAMPAIGN33_SCROLL_DICT
    # ]}),
    # _test("holdout_n96_scan_film", {"model.scan_film": True}),
    # _test("holdout_n96_denoiser", {"model.input_denoiser": DENOISER_PATH},
    #       arch={"pretrain_key": DENOISED_MAE_KEY}),
    # surface-anchored normalization changes the prepared data, so these three share one reload
    _test("holdout_n96_surface_norm", SURFACE_NORM, arch={"pretrain_key": SURFACE_MAE_KEY}),
    _test("holdout_n96_surface_norm_drop_pos", {**SURFACE_NORM, "tra.support_drop": "pos"},
          arch={"pretrain_key": SURFACE_MAE_KEY}),
    _test("holdout_n96_surface_norm_drop_both", {**SURFACE_NORM, "tra.support_drop": "both"},
          arch={"pretrain_key": SURFACE_MAE_KEY}),
    # each arm below changes the prepared-dataset cache key, so each forces one full reload
    _test("holdout_n96_nnpu", {
        "data.far_negative_share": 0.15, "tra.pu_lambda": 1.0, "tra.pu_prior": 0.03,
    }),
    _test("holdout_n96_regime_specialist", {}, train_domains=REGIME_DOMAINS),
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
        config = campaign35.build_config(test)
    train_domains = test.get("train_domains")
    if train_domains:
        # held-out domains stay in the dict so they are still rendered; holdout_domains excludes them
        keep = set(train_domains) | set(config.data.holdout_domains)
        groups = [
            (name, ids, weight)
            for (name, ids), weight in zip(config.data.train_scroll_dict.items(), config.data.train_scroll_weights)
            if name in keep
        ]
        config.data.train_scroll_dict = {name: list(ids) for name, ids, _ in groups}
        config.data.train_scroll_weights = [weight for _, _, weight in groups]
        kept_ids = {int(scroll_id) for _, ids, _ in groups for scroll_id in ids}
        config.data.scrolls = [s for s in config.data.scrolls if int(s.scroll_id) in kept_ids]
        config.tra.dann_n_domains = len(groups)
    return config


def _train_denoiser(path: str, scroll_ids, steps: int, batch_size: int, dry_run: bool) -> None:
    checkpoint = campaign34.ROOT / path
    if checkpoint.with_suffix(".complete.json").is_file():
        return
    if dry_run:
        print(f"[campaign38] would train denoiser {path}: {len(scroll_ids)} volumes, {steps} steps", flush=True)
        return
    subprocess.run([
        sys.executable, str(campaign34.ROOT / "train_denoiser.py"),
        "--name", checkpoint.stem, "--out-dir", str(checkpoint.parent),
        "--scroll-ids", *(str(value) for value in scroll_ids),
        "--steps", str(steps), "--batch-size", str(batch_size),
    ], cwd=campaign34.ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 38: native-regime and PU tests on held-out scrolls")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm on pherc0814 alone from random init, no figures, "
                             "separate log/model dirs")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs38_smoke", "models/archs38_smoke"
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
            if any(test["config"].get("model.input_denoiser") for test in selected):
                _train_denoiser(DENOISER_PATH, campaign33.PRETRAIN_SCROLL_IDS, DENOISER_STEPS, 16, args.dry_run)
            if any(test["config"].get("data.norm_mode") == "surface_anchor" for test in selected):
                ensure_surface_anchors(SURFACE_ANCHOR_SCROLL_IDS, str(campaign34.ROOT / "ves_zarrs2"))
            campaign34.preflight_pretraining(selected, args.dry_run)
        print(f"[campaign38] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        dropped = campaign36._smoke_config(config) if args.smoke else []
        if args.smoke and config.model.input_denoiser:
            config.model.input_denoiser = f"{MODEL_DIR}/denoiser_smoke.pth"
            _train_denoiser(config.model.input_denoiser, [campaign36.SMOKE_SCROLL], 20, 2, False)
        with startup_output():
            print(f"[campaign38] {test['tid']}: ctx={config.data.context_size} "
                  f"domains={list(config.data.train_scroll_dict)} "
                  f"weights={config.data.train_scroll_weights} scan_film={config.model.scan_film} "
                  f"far_share={config.data.far_negative_share} pu={config.tra.pu_lambda}/{config.tra.pu_prior} "
                  f"batch={config.dl.batch_size} lr={config.tra.lr} denoiser={config.model.input_denoiser!r} "
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

    print(f"\n{'=' * 78}\n[campaign38] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
