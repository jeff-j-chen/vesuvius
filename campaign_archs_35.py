"""campaign 35: extreme augmentation scored on held-out scrolls

Every arm trains without pherc0841 and pherc0009b (few labels, closest in size and
acquisition to the unlabeled test scrolls) and is judged on those two domains. Earlier
augmentation tests scored only held-out regions of trained scrolls, which rewards per-scroll
memorisation and cannot show a gain in cross-scroll invariance.

Each family is pushed far past a mild setting so the model cannot absorb it; transforms run
from epoch 0. The base recipe for every arm is early_gated + patch GroupDRO on a native
128 px field (campaign 34 `early_patchdro_native128`, batch 96 / lr 1.5e-4); only the arms
marked (arch) change the field or resolution. The in-scroll augmentations (cutout, context
replacement, context jitter, depth jitter) are off everywhere except `holdout_default_augs`,
so each family is measured alone against a bare baseline. Rotation/flip stay on.

| arm                     | what it changes                                                |
|-------------------------|----------------------------------------------------------------|
| holdout_ctx384_ds2      | (arch) 384 px field at 2x downsampling, batch 32 / lr 1e-4     |
| holdout_ds2_b32         | (arch) 192 px field at 2x downsampling, batch 32 / lr 1e-4     |
| holdout_far_neg         | 15% of negatives from windows >=160 px from any ink            |
| holdout_shell6          | ring-negative shell 6 tiles instead of 4                       |
| holdout_far_neg_region_gate | far negatives + coarse text-region head gating ink logits  |
| holdout_pos_weight05    | positive BCE weight 0.5                                        |
| holdout_baseline        | nothing beyond rotation/flip                                   |
| holdout_dice_bce        | paper loss: 0.5 soft Dice + 0.5 BCE, label smoothing 0.25      |
| holdout_multi_collapse  | volumetric enc2/enc3 max-collapsed into the 2D stages too      |
| holdout_seven_stage     | three extra 320-ch 2D stages (7 stages, 2x2 bottleneck)        |
| holdout_dual_scale      | independent local-64 expert added to the score (mix 0.5)       |
| holdout_default_augs    | cutout, context replacement, context and depth jitter          |
| holdout_photometric     | brightness, contrast, white noise                              |
| holdout_acquisition     | PSF blur, correlated noise, coarse-scan resampling (fine scans)|
| holdout_geometric       | depth undulation, depth stretch, surface-band contrast         |
| holdout_fda             | cross-scroll low-frequency amplitude swap                      |
| holdout_phase           | energy: propagation phase contrast (paganin smoothing/fringes) |
| holdout_tone            | energy: nonlinear monotonic intensity transfer                 |
| holdout_regime          | beamline regime swap: fine scans hazed/blurred, coarse dehazed |
| holdout_soft_labels     | label smoothing 0.2/0.1 (every other arm uses 0.1/0.05)        |
| holdout_gce             | noise-robust GCE loss, q=0.7                                   |
| holdout_edge_soft       | positive cells grazing a stroke edge get soft targets          |
| holdout_ema_0999        | weight EMA, decay 0.999 (~1k steps, about one epoch)           |
| holdout_ema_long        | weight EMA, decay 0.9998 from epoch 3 (~5k steps, SWAD-like)   |
| holdout_all             | photometric..regime families at once                           |

`holdout_quality_norm`, `holdout_fiber`, `holdout_randconv` and `holdout_ema_0995` moved to
campaign 36.

The final epoch renders both held-out scrolls at full extent; both volumes stay in RAM.

Usage:
    python3 campaign_archs_35.py --dry-run
    python3 campaign_archs_35.py --smoke --only holdout_baseline
    python3 campaign_archs_35.py --only holdout_baseline,holdout_all
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
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
from utils.config import startup_output


LOG_DIR = "./runs_archs35"
MODEL_DIR = "models/archs35"
HOLDOUT_DOMAINS = ("pherc0841", "pherc0009b")
VIS_SCROLL_IDS = [20260221022814, 20250919125754]  # pherc0841, pherc0009b (both held out)
# resampled down to the 9.36 um grid from 2.4 um (w013, w018, paris4) or 3.24 um (fragments)
FINE_NATIVE_SCROLL_IDS = [
    20240304141531, 20240304144031, 20231210121321,
    20230301213755, 20231205222200, 20230301213423, 20231201215900, 20230205142449,
]
# training scrolls scanned natively on the 9.36 um / 1.2 m beamline, like the test scrolls
QUALITY_REFERENCE_DOMAINS = ("pherc0139", "pherc0814", "pherc0500p2")
QUALITY_TRANSFER_PATH = Path(__file__).resolve().parent / "quality_transfer.json"
QUALITY_BINS = 24

PHOTOMETRIC = {
    "dl.brightness_prob": 1.0, "dl.brightness_delta": 0.5,
    "dl.contrast_prob": 1.0, "dl.contrast_delta": 0.6,
    "dl.noise_prob": 1.0, "dl.noise_std_min": 0.02, "dl.noise_std_max": 0.10,
}
ACQUISITION = {
    "dl.acquisition_blur_prob": 0.8, "dl.acquisition_blur_min": 1.0, "dl.acquisition_blur_max": 4.0,
    "dl.correlated_noise_prob": 0.8, "dl.correlated_noise_min": 0.03,
    "dl.correlated_noise_max": 0.12, "dl.correlated_noise_sigma": 1.5,
    "dl.resolution_degrade_prob": 0.8, "dl.resolution_degrade_min": 2.0, "dl.resolution_degrade_max": 8.0,
}
# the input window is only 8 slices deep, so +-3 slices of undulation is severe
GEOMETRIC = {
    "dl.depth_warp_prob": 0.9, "dl.depth_warp_max": 3.0, "dl.depth_warp_sigma": 12.0,
    "dl.depth_scale_prob": 0.8, "dl.depth_scale_min": 0.5, "dl.depth_scale_max": 2.0,
    "dl.surface_atten_prob": 0.8, "dl.surface_atten_min": 0.6,
    "dl.surface_atten_max": 0.95, "dl.surface_atten_sigma": 3.0,
}
FDA = {"dl.fda_prob": 1.0, "dl.fda_beta": 0.3}
PHASE = {
    "dl.phase_filter_prob": 0.9, "dl.phase_filter_min": 2.0,
    "dl.phase_filter_max": 200.0, "dl.phase_filter_max_gain": 8.0,
}
TONE = {"dl.tone_curve_prob": 0.9, "dl.tone_curve_knots": 8, "dl.tone_curve_concentration": 0.2}
REGIME = {"dl.scan_regime_prob": 0.9, "dl.scan_regime_min": 0.5, "dl.scan_regime_max": 1.0}
RANDCONV = {
    "dl.randconv_prob": 0.9, "dl.randconv_kernel_sizes": [3, 5, 7, 9, 11],
    "dl.randconv_depth_kernel": 3, "dl.randconv_layers_max": 2, "dl.randconv_mix_min": 0.5,
}
NO_DEFAULT_AUGS = {
    "dl.cutout_prob": 0.0, "dl.context_replace_prob": 0.0,
    "data.ctx_jitter": 0, "data.depth_jitter": 0,
}
# campaign 34 native128 strengths: the surround is 2/3 of a 192 px field
DEFAULT_AUGS = {
    "dl.cutout_prob": 0.30, "dl.context_replace_prob": 0.25,
    "dl.context_replace_margin": 13, "dl.context_replace_feather": 26,
    "data.ctx_jitter": 20, "data.depth_jitter": 1,
}
NATIVE128 = {"pretrain_key": "early_gated_native128", "context_size": 128, "context_downsample": 1}
DS2_B32 = {"pretrain_key": "early_gated", "context_size": 192, "context_downsample": 2,
           "batch_size": 32, "lr": 1e-4}
# 384 px OOMed host RAM at batch 48 in campaign 34
CTX384_B32 = {"pretrain_key": "early_gated_ctx384_ds2", "context_size": 384, "context_downsample": 2,
              "batch_size": 32, "lr": 1e-4}
FAR_NEGATIVES = {"data.far_negative_share": 0.15, "data.far_negative_min_dist": 160}
# the extra modules are absent from the MAE checkpoint and start from initialisation
NEW_MODULES = {"model.require_architecture_init": False}
BASE = {
    "tra.aug_start_epoch": 0,
    "tra.fast_eval_figure": False,
    "tra.eval_int_scrolls": len(VIS_SCROLL_IDS),
    "data.vis_scroll_ids": VIS_SCROLL_IDS,
    "data.vis_preload_persistent": True,
    "dl.fine_native_scroll_ids": FINE_NATIVE_SCROLL_IDS,
    **NO_DEFAULT_AUGS,
}


def _test(tid: str, augmentations: dict, arch: dict | None = None, **extra) -> dict:
    arch = dict(arch or NATIVE128)
    test = campaign34._test(
        tid, arch.pop("pretrain_key"), holdout_domains=HOLDOUT_DOMAINS,
        config={**BASE, **augmentations}, **arch,
    )
    test["tag"] = f"35_{tid}"
    test.update(extra)
    return test


TESTS = [
    # moved from campaign 34
    _test("holdout_ctx384_ds2", {}, arch=CTX384_B32),
    _test("holdout_ds2_b32", {}, arch=DS2_B32),
    _test("holdout_far_neg", FAR_NEGATIVES),
    _test("holdout_shell6", {"data.ring_shell_r": 6}),
    _test("holdout_far_neg_region_gate", {
        **FAR_NEGATIVES, "model.text_region_gate": True, "tra.text_region_lambda": 0.5,
    }),
    _test("holdout_pos_weight05", {"tra.tile_pos_weight": 0.5}),

    _test("holdout_baseline", {}),
    # smp SoftBCE smooth=0.25 maps targets to 0.75 / 0.25
    _test("holdout_dice_bce", {
        "tra.dice_weight": 0.5, "tra.label_smooth_pos": 0.25, "tra.label_smooth_neg": 0.25,
    }),
    _test("holdout_multi_collapse", {"model.multi_depth_collapse": True}),
    _test("holdout_seven_stage", {"model.two_d_extra_channels": (320, 320, 320), **NEW_MODULES}),
    _test("holdout_dual_scale", {
        "model.dual_scale": True, "model.dual_scale_local_size": 64, "model.dual_scale_mix": 0.5,
    }),
    _test("holdout_default_augs", DEFAULT_AUGS),
    _test("holdout_photometric", PHOTOMETRIC),
    _test("holdout_acquisition", ACQUISITION),
    _test("holdout_geometric", GEOMETRIC),
    _test("holdout_fda", FDA),
    _test("holdout_phase", PHASE),
    _test("holdout_tone", TONE),
    _test("holdout_regime", REGIME),
    _test("holdout_soft_labels", {"tra.label_smooth_pos": 0.2, "tra.label_smooth_neg": 0.1}),
    _test("holdout_gce", {"tra.loss_type": "gce", "tra.gce_q": 0.7}),
    _test("holdout_edge_soft", {"data.edge_soft_sigma": 4.0, "data.edge_soft_floor": 0.6}),
    _test("holdout_ema_0999", {"tra.model_ema": True, "tra.model_ema_decay": 0.999}),
    # before the start epoch the average just tracks the live weights
    _test("holdout_ema_long", {
        "tra.model_ema": True, "tra.model_ema_decay": 0.9998, "tra.model_ema_start_epoch": 3,
    }),
    _test("holdout_all", {**PHOTOMETRIC, **ACQUISITION, **GEOMETRIC, **FDA, **PHASE, **TONE, **REGIME}),
]


@contextlib.contextmanager
def _campaign34_paths():
    saved = campaign34.LOG_DIR, campaign34.MODEL_DIR
    campaign34.LOG_DIR, campaign34.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign34.LOG_DIR, campaign34.MODEL_DIR = saved


def _radial_power(scroll_id: int, rng, samples: int = 64, size: int = 192):
    import cv2
    import numpy as np
    import zarr
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm

    root = Path(campaign33.ROOT)
    volume = zarr.open(str(root / "ves_zarrs2" / f"{scroll_id}.zarr"), mode="r")
    mask = cv2.imread(str(root / "masks" / f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE)
    mean, std = load_cached_norm(str(scroll_id), UNIFIED_CACHE_PATH)[:2]
    radius = np.sqrt(np.fft.fftfreq(size)[:, None] ** 2 + np.fft.rfftfreq(size)[None, :] ** 2)
    bins = np.minimum((radius / 0.5 * QUALITY_BINS).astype(int), QUALITY_BINS)
    counts = np.bincount(bins.ravel(), minlength=QUALITY_BINS + 1)[:QUALITY_BINS]
    ys, xs = np.nonzero(mask[::32, ::32] > 0)
    middle = volume.shape[0] // 2
    total, used = np.zeros(QUALITY_BINS), 0
    for index in rng.permutation(len(ys)):
        y, x = int(ys[index]) * 32, int(xs[index]) * 32
        if y + size > mask.shape[0] or x + size > mask.shape[1]:
            continue
        if (mask[y:y + size, x:x + size] > 0).mean() < 0.98:
            continue
        block = (np.asarray(volume[middle - 4:middle + 4, y:y + size, x:x + size], np.float32) - mean) / std
        block -= block.mean(axis=(1, 2), keepdims=True)
        power = (np.abs(np.fft.rfft2(block, axes=(1, 2))) ** 2).mean(axis=0)
        total += np.bincount(bins.ravel(), weights=power.ravel(), minlength=QUALITY_BINS + 1)[:QUALITY_BINS]
        used += 1
        if used >= samples:
            break
    if used == 0:
        raise RuntimeError(f"no fully masked {size}px crop found in scroll {scroll_id}")
    spectrum = total / counts / used
    return spectrum / spectrum[1:4].mean()


def _quality_sources_newer_than_cache() -> bool:
    cached = QUALITY_TRANSFER_PATH.stat().st_mtime
    ids = list(FINE_NATIVE_SCROLL_IDS) + [
        scroll_id for domain in QUALITY_REFERENCE_DOMAINS
        for scroll_id in campaign33.CAMPAIGN33_SCROLL_DICT[domain]
    ]
    root = Path(campaign33.ROOT) / "ves_zarrs2"
    for scroll_id in ids:
        zarr_dir = root / f"{scroll_id}.zarr"
        stamps = [p.stat().st_mtime for p in (zarr_dir, zarr_dir / ".zattrs", zarr_dir / ".zarray") if p.exists()]
        if stamps and max(stamps) > cached:
            return True
    return False


def quality_transfer() -> dict:
    """per fine scroll, the gain that makes its radial spectrum match the reference beamline."""
    if QUALITY_TRANSFER_PATH.exists():
        cached = json.loads(QUALITY_TRANSFER_PATH.read_text())
        # re-rendered volumes (e.g. the 88 keV fragment rescans) invalidate the measured filters
        if set(map(str, FINE_NATIVE_SCROLL_IDS)) <= set(cached) and not _quality_sources_newer_than_cache():
            return cached
        print("[campaign35] quality transfer cache is stale; re-measuring", flush=True)
    import numpy as np

    rng = np.random.default_rng(0)
    reference_ids = [
        scroll_id for domain in QUALITY_REFERENCE_DOMAINS
        for scroll_id in campaign33.CAMPAIGN33_SCROLL_DICT[domain]
    ]
    reference = np.exp(np.mean([np.log(_radial_power(s, rng)) for s in reference_ids], axis=0))
    table = {}
    for scroll_id in FINE_NATIVE_SCROLL_IDS:
        ratio = np.sqrt(reference / _radial_power(scroll_id, rng))
        smoothed = np.convolve(np.pad(ratio, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
        gains = np.minimum.accumulate(np.clip(smoothed, 0.0, 1.0))
        table[str(scroll_id)] = [round(float(g), 4) for g in gains]
        print(f"[campaign35] quality transfer {scroll_id}: "
              + " ".join(f"{g:.2f}" for g in gains[::3]), flush=True)
    QUALITY_TRANSFER_PATH.write_text(json.dumps({"reference": reference_ids, **table}, indent=1))
    return json.loads(QUALITY_TRANSFER_PATH.read_text())


def build_config(test: dict):
    with _campaign34_paths():
        config = campaign34.build_config(test)
    if test.get("quality_normalize"):
        config.dl.quality_transfer = {
            key: value for key, value in quality_transfer().items() if key != "reference"
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 35: extreme augmentation on held-out scrolls")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one short epoch per arm, cropped figures, separate log/model dirs")
    args = parser.parse_args()
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = "./runs_archs35_smoke", "models/archs35_smoke"

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
        print(f"[campaign35] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        if args.smoke:
            config.tra.n_epochs = 1
            config.tra.eval_int = 1
            config.tra.fast_eval_figure = True
            config.data.max_samples_per_epoch = 96
        with startup_output():
            print(f"[campaign35] {test['tid']}: holdout={config.data.holdout_domains} "
                  f"ctx={config.data.context_size} ds={config.data.context_downsample} "
                  f"batch={config.dl.batch_size} lr={config.tra.lr} init={config.init_weights} "
                  f"aug_start={config.tra.aug_start_epoch} overrides={test['config']}", flush=True)
        if args.dry_run:
            success = campaign31.run_test(config, True)
        else:
            campaign29.prewarm_data_cache(config)
            success = campaign31.run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        del config
        gc.collect()

    print(f"\n{'=' * 78}\n[campaign35] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
