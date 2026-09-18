"""campaign 26: post-baseline domain-generalization and architecture study.

All arms inherit Campaign 25's new-label 11-fragment no-DANN/no-SupCon baseline,
four-times PHerc0139 sampling, nine epochs, and disabled periodic figure evaluation.

Usage:
    python3 campaign_archs_26.py --dry-run
    python3 campaign_archs_26.py --only baseline_ibn
    python3 campaign_archs_26.py --from weldon_asymmetric
    python3 campaign_archs_26.py --from mednext_adapter_k7 --restart-baseline
    python3 campaign_archs_26.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from campaign_archs_24 import preflight_train_masks
from campaign_archs_25 import (
    CAMPAIGN_SCROLLS,
    DOMAIN_INDEX,
    PRETRAIN_SCROLL_IDS,
    W044_SCROLL_ID,
    _pretrain_path as campaign25_pretrain_path,
    build_config as campaign25_build_config,
)

LOG_DIR = "./runs_archs26"
MODEL_DIR = "models/archs26"
ROOT = Path(__file__).resolve().parent
PRETRAIN_STEPS = 2_000
AVAILABLE_PRETRAIN_SCROLL_IDS = tuple(
    scroll_id
    for scroll_id in PRETRAIN_SCROLL_IDS
    if (ROOT / "ves_zarrs2" / f"{scroll_id}.zarr").is_dir()
)
MISSING_PRETRAIN_SCROLL_IDS = tuple(
    scroll_id for scroll_id in PRETRAIN_SCROLL_IDS
    if scroll_id not in AVAILABLE_PRETRAIN_SCROLL_IDS
)

PRETRAIN_SPECS = {
    "full_ibn": ("--norm-mode", "ibn_full"),
    "batch_norm": ("--norm-mode", "batch"),
    "early_wide": (
        "--early-2d-unet",
        "--early-2d-channels-mult",
        "1.5",
    ),
    "mid_3d2d": ("--mid-2d-unet",),
    "mid_3d2d_full_ibn": ("--mid-2d-unet", "--norm-mode", "ibn_full"),
    "factorized_2plus1d": ("--factorized-2plus1d",),
}

PRETRAIN_REQUIRED_PREFIXES = {
    "full_ibn": ("enc1.", "enc2.", "enc3.", "bottleneck.", "dec3.", "dec2.", "dec1."),
    "batch_norm": ("enc1.", "enc2.", "enc3.", "bottleneck.", "dec3.", "dec2.", "dec1."),
    "early_wide": ("early_depth_attn.", "early_depth_fuse.", "early2d_"),
    "mid_3d2d": ("mid_depth_attn.", "mid_depth_fuse.", "mid_skip1_fuse.", "mid2d_"),
    "mid_3d2d_full_ibn": (
        "enc1.", "enc2.", "mid_depth_attn.", "mid_depth_fuse.",
        "mid_skip1_fuse.", "mid2d_",
    ),
    "factorized_2plus1d": ("enc1.", "enc2.", "enc3.", "bottleneck.", "dec3.", "dec2.", "dec1."),
}


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_depth8_campaign26_{key}_2k"


def _campaign26_pretrain_path(key: str) -> Path:
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"

PRETRAIN_ROUTES = {
    "base": (campaign25_pretrain_path("base"), True, "exact Campaign 25 backbone"),
    "mednext_k5": (
        campaign25_pretrain_path("mednext_k5"), True, "exact Campaign 25 MedNeXt-k5"
    ),
    "mednext_k7": (
        campaign25_pretrain_path("mednext_k7"), True, "exact Campaign 25 MedNeXt-k7"
    ),
    "mednext_k7_divided": (
        campaign25_pretrain_path("mednext_k7_divided"),
        True,
        "exact Campaign 25 MedNeXt-k7 plus divided attention",
    ),
    "full_ibn": (_campaign26_pretrain_path("full_ibn"), True, "matched Campaign 26 MAE"),
    "batch_norm": (_campaign26_pretrain_path("batch_norm"), True, "matched Campaign 26 MAE"),
    "early_wide": (
        _campaign26_pretrain_path("early_wide"),
        True,
        "matched Campaign 26 MAE",
    ),
    "mid_3d2d": (
        _campaign26_pretrain_path("mid_3d2d"), True, "matched Campaign 26 MAE"
    ),
    "mid_3d2d_full_ibn": (
        _campaign26_pretrain_path("mid_3d2d_full_ibn"),
        True,
        "matched full-IBN mid-3D/2D MAE",
    ),
    "factorized_2plus1d": (
        _campaign26_pretrain_path("factorized_2plus1d"),
        True,
        "matched Campaign 26 MAE",
    ),
    "depth_attention_2d": (
        campaign25_pretrain_path("base"), False, "complete 3D backbone; new attention head"
    ),
}


def _test(tid: str, **overrides):
    test = {
        "tid": tid,
        "tag": f"26_{tid}",
        "scrolls": CAMPAIGN_SCROLLS,
        "max_samples_per_epoch": 6_667,
        "context_replace_prob": 0.35,
        "context_replace_margin": 20,
        "context_replace_feather": 40,
        "cutout_prob": 0.50,
        "cutout_max_frac": 0.16,
        "cutout_n_patches": 3,
        "depth_jitter": 1,
    }
    test.update(overrides)
    return test


TESTS = [
    _test("baseline", eval_int=9),
    _test("mednext_adapter_k5", pretrain_key="mednext_k5", mednext_adapters=True, mednext_kernel=5),
    _test("mednext_adapter_k7", pretrain_key="mednext_k7", mednext_adapters=True, mednext_kernel=7),
    _test(
        "mednext_k7_divided_attention",
        pretrain_key="mednext_k7_divided",
        mednext_adapters=True,
        mednext_kernel=7,
        divided_attention=True,
        divided_attention_spatial=True,
    ),
    _test("dg_prototype_align_strong", prototype_align=True, prototype_align_lambda=0.3),
    _test("dg_conditional_coral_strong", coral_align=True, coral_align_lambda=0.3),
    _test("dg_cdan_strong", cdan=True, cdan_lambda=0.3),
    _test(
        "dg_mldg_holdout_pherc1667",
        mldg=True,
        mldg_holdout_domain=DOMAIN_INDEX["pherc1667"],
        mldg_inner_lr=5e-4,
        mldg_beta=1.0,
        compile_model=False,
    ),
    _test(
        "dg_mldg_holdout_pherc0139",
        mldg=True,
        mldg_holdout_domain=DOMAIN_INDEX["pherc0139"],
        mldg_inner_lr=5e-4,
        mldg_beta=1.0,
        compile_model=False,
    ),
    _test(
        "dg_mixstyle_strong",
        mixstyle=True,
        mixstyle_prob=0.7,
        mixstyle_alpha=0.2,
        compile_model=False,
    ),
    _test("dg_sagnet_strong", sagnet=True, sagnet_lambda=0.3, compile_model=False),
    _test(
        "dg_dual_scale_local64",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.25,
    ),
    _test(
        "dg_dual_scale_mix_0p50",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test("dg_continuous_style_film_h128", style_film=True, style_film_hidden=128),
    _test(
        "dg_mae_continuation_strong",
        mae_reconstruction=True,
        mae_reconstruction_lambda=0.3,
        mae_reconstruction_start_epoch=4,
        mae_reconstruction_mask_frac=0.5,
        mae_reconstruction_patch=4,
        compile_model=False,
    ),
    _test("pure_ibn", pretrain_key="full_ibn", norm_mode="ibn_full"),
    _test("pure_batch_norm", pretrain_key="batch_norm", norm_mode="batch"),
    _test(
        "weldon_asymmetric",
        weldon_top_k=4,
        weldon_bottom_k=16,
        weldon_top_weight=0.7,
    ),
    _test(
        "weldon_multi_k",
        weldon_top_k=4,
        weldon_bottom_k=16,
        weldon_top_weight=0.7,
        weldon_multi_k=True,
        weldon_top_k2=12,
        weldon_bottom_k2=24,
        weldon_multi_mix=0.5,
    ),
    _test(
        "weldon_depth_supported",
        weldon_top_k=4,
        weldon_bottom_k=16,
        weldon_top_weight=0.7,
        weldon_depth_support_k=2,
    ),
    _test(
        "early_3d2d_wide",
        pretrain_key="early_wide",
        early_2d_unet=True,
        early_2d_channels_mult=1.5,
    ),
    _test(
        "mid_3d2d",
        pretrain_key="mid_3d2d",
        mid_2d_unet=True,
    ),
    _test(
        "factorized_2plus1d",
        pretrain_key="factorized_2plus1d",
        factorized_2plus1d=True,
    ),
    _test(
        "depth_attention_2d_head",
        pretrain_key="depth_attention_2d",
        depth_attention_2d_head=True,
    ),
    _test("full_residual_unet", residual_unet=True),
    _test("sparse_deep_supervision", sparse_deep_supervision=True),
    _test("gated_stems", gated_stems=True),
    _test(
        "physical_domain_groupdro",
        physical_domain_groupdro=True,
        physical_domain_groupdro_eta=0.05,
        physical_domain_groupdro_max_ratio=3.0,
    ),
    _test(
        "tight_ring_uneroded",
        inklabel_dir="./eroded_inklabels",
        label_dilate_r=8,
        ring_label_source="closed",
        ring_close_r=0,
        ring_gap_r=1,
        ring_shell_r=2,
        multitile_pos_only=False,
    ),
    _test(
        "eroded_touching",
        inklabel_dir="./eroded_inklabels",
        ring_label_source="eroded",
        ring_close_r=0,
        ring_gap_r=0,
        ring_shell_r=1,
        multitile_pos_only=False,
    ),
    _test(
        "dg_mixstyle_strong_retry",
        mixstyle=True,
        mixstyle_prob=0.7,
        mixstyle_alpha=0.2,
        compile_model=False,
    ),
    _test(
        "dg_sagnet_strong_retry",
        sagnet=True,
        sagnet_lambda=0.3,
        compile_model=False,
    ),
    _test(
        "ibn_dual_scale_deep_local64",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_deep=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.50,
    ),
    _test(
        "ibn_dual_scale_local_only64",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_only=True,
        dual_scale_local_size=64,
    ),
    _test(
        "ibn_dual_scale_local64_mix_0p75",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=0.75,
    ),
    _test(
        "ibn_dual_scale_local64_mix_1p00",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_size=64,
        dual_scale_mix=1.00,
    ),
    _test(
        "ibn_dual_scale_local96_mix_0p75",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_size=96,
        dual_scale_mix=0.75,
    ),
    _test(
        "ibn_dual_scale_local128_mix_1p00",
        pretrain_key="full_ibn",
        norm_mode="ibn_full",
        dual_scale=True,
        dual_scale_local_size=128,
        dual_scale_mix=1.00,
    ),
]


def _expected_pretrain_key(test: dict) -> str:
    mednext = bool(test.get("mednext_adapters", False))
    divided = bool(test.get("divided_attention", False))
    spatial = bool(test.get("divided_attention_spatial", False))
    if mednext and divided and spatial and int(test.get("mednext_kernel", 5)) == 7:
        return "mednext_k7_divided"
    if mednext:
        return f"mednext_k{int(test.get('mednext_kernel', 5))}"
    if bool(test.get("early_2d_unet", False)):
        return "early_wide"
    if bool(test.get("mid_2d_unet", False)):
        return (
            "mid_3d2d_full_ibn"
            if str(test.get("norm_mode", "ibn")) == "ibn_full"
            else "mid_3d2d"
        )
    if bool(test.get("factorized_2plus1d", False)):
        return "factorized_2plus1d"
    if bool(test.get("depth_attention_2d_head", False)):
        return "depth_attention_2d"
    norm_mode = str(test.get("norm_mode", "ibn"))
    if norm_mode == "ibn_full":
        return "full_ibn"
    if norm_mode == "batch":
        return "batch_norm"
    return "base"


def _pretrain_path(key: str) -> Path:
    try:
        return PRETRAIN_ROUTES[key][0]
    except KeyError as exc:
        raise ValueError(f"unknown campaign-26 transfer key: {key}") from exc


def _pretrain_marker(key: str) -> Path:
    return _campaign26_pretrain_path(key).with_suffix(".complete.json")


def _pretrain_metadata(key: str) -> dict:
    checkpoint = _campaign26_pretrain_path(key)
    return {
        "key": key,
        "steps": PRETRAIN_STEPS,
        "scroll_ids": list(AVAILABLE_PRETRAIN_SCROLL_IDS),
        "omitted_missing_scroll_ids": list(MISSING_PRETRAIN_SCROLL_IDS),
        "architecture_args": list(PRETRAIN_SPECS[key]),
        "freeze_loaded_backbone": True,
        "init_weights": str(campaign25_pretrain_path("base").relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }


def _pretraining_complete(key: str) -> bool:
    checkpoint = _campaign26_pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if metadata != _pretrain_metadata(key):
        return False
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return all(
        any(state_key.startswith(prefix) for state_key in state)
        for prefix in PRETRAIN_REQUIRED_PREFIXES[key]
    )


def build_config(test: dict):
    base_request = {
        "tid": "dg_baseline_no_aux",
        "tag": test["tag"],
        "scrolls": test["scrolls"],
        "max_samples_per_epoch": test["max_samples_per_epoch"],
        "context_replace_prob": test["context_replace_prob"],
        "context_replace_margin": test["context_replace_margin"],
        "context_replace_feather": test["context_replace_feather"],
        "cutout_prob": test["cutout_prob"],
        "cutout_max_frac": test["cutout_max_frac"],
        "cutout_n_patches": test["cutout_n_patches"],
        "depth_jitter": test["depth_jitter"],
    }
    config = campaign25_build_config(base_request)
    pretrain_key = str(test.get("pretrain_key", "base"))
    expected_key = _expected_pretrain_key(test)
    if pretrain_key != expected_key:
        raise RuntimeError(
            f"campaign-26 arm {test['tid']} maps to pretrain_key={pretrain_key!r}, "
            f"but its architecture requires {expected_key!r}"
        )

    config.init_weights = str(_pretrain_path(pretrain_key).relative_to(ROOT))
    config.model.require_architecture_init = bool(PRETRAIN_ROUTES[pretrain_key][1])
    config.tra.log_dir = LOG_DIR
    config.tra.n_epochs = 9
    config.tra.eval_int = 9 if test["tid"] == "baseline" else 999
    config.tra.test_int = 9_999
    config.tra.probe_int = 9_999
    config.tra.fast_eval_figure = False
    config.tra.eval_int_scrolls = 1
    config.tra.dann = False
    config.tra.dann_lambda = 0.0
    config.tra.supcon = False
    config.tra.supcon_cross_frag = False
    config.tra.per_scroll_metrics = True
    config.tra.prototype_align = bool(test.get("prototype_align", False))
    config.tra.prototype_align_lambda = float(test.get("prototype_align_lambda", 0.1))
    config.tra.prototype_margin = float(test.get("prototype_margin", 0.5))
    config.tra.coral_align = bool(test.get("coral_align", False))
    config.tra.coral_align_lambda = float(test.get("coral_align_lambda", 0.1))
    config.tra.coral_mean_weight = float(test.get("coral_mean_weight", 1.0))
    config.tra.cdan = bool(test.get("cdan", False))
    config.tra.cdan_lambda = float(test.get("cdan_lambda", 0.1))
    config.tra.mldg = bool(test.get("mldg", False))
    config.tra.mldg_holdout_domain = int(test.get("mldg_holdout_domain", -1))
    config.tra.mldg_random_holdout = bool(test.get("mldg_random_holdout", False))
    config.tra.mldg_inner_lr = float(test.get("mldg_inner_lr", 5e-4))
    config.tra.mldg_beta = float(test.get("mldg_beta", 1.0))
    config.tra.physical_domain_groupdro = bool(
        test.get("physical_domain_groupdro", False)
    )
    config.tra.physical_domain_groupdro_eta = float(
        test.get("physical_domain_groupdro_eta", 0.05)
    )
    config.tra.physical_domain_groupdro_max_ratio = float(
        test.get("physical_domain_groupdro_max_ratio", 3.0)
    )
    config.tra.sagnet_lambda = float(test.get("sagnet_lambda", 0.1))
    config.tra.mae_reconstruction = bool(test.get("mae_reconstruction", False))
    config.tra.mae_reconstruction_lambda = float(
        test.get("mae_reconstruction_lambda", 0.1)
    )
    config.tra.mae_reconstruction_start_epoch = int(
        test.get("mae_reconstruction_start_epoch", 4)
    )
    config.tra.mae_reconstruction_mask_frac = float(
        test.get("mae_reconstruction_mask_frac", 0.5)
    )
    config.tra.mae_reconstruction_patch = int(test.get("mae_reconstruction_patch", 4))
    config.data.vis_scroll_ids = [W044_SCROLL_ID]
    config.data.inklabel_dir = str(
        test.get("inklabel_dir", config.data.inklabel_dir)
    )
    config.data.label_dilate_r = int(
        test.get("label_dilate_r", config.data.label_dilate_r)
    )
    config.data.ring_label_source = str(
        test.get("ring_label_source", config.data.ring_label_source)
    )
    config.data.ring_close_r = int(
        test.get("ring_close_r", config.data.ring_close_r)
    )
    config.data.ring_gap_r = int(
        test.get("ring_gap_r", config.data.ring_gap_r)
    )
    config.data.ring_shell_r = int(
        test.get("ring_shell_r", config.data.ring_shell_r)
    )
    config.data.multitile_pos_only = bool(
        test.get("multitile_pos_only", config.data.multitile_pos_only)
    )

    config.model.norm_mode = str(test.get("norm_mode", "ibn"))
    config.model.use_ibn = config.model.norm_mode == "ibn"
    config.model.weldon_k = 0
    config.model.weldon_top_k = int(test.get("weldon_top_k", 0))
    config.model.weldon_bottom_k = int(test.get("weldon_bottom_k", 0))
    config.model.weldon_top_weight = float(test.get("weldon_top_weight", 0.5))
    config.model.weldon_multi_k = bool(test.get("weldon_multi_k", False))
    config.model.weldon_top_k2 = int(test.get("weldon_top_k2", 0))
    config.model.weldon_bottom_k2 = int(test.get("weldon_bottom_k2", 0))
    config.model.weldon_multi_mix = float(test.get("weldon_multi_mix", 0.5))
    config.model.weldon_depth_support_k = int(test.get("weldon_depth_support_k", 0))
    config.model.early_2d_unet = bool(test.get("early_2d_unet", False))
    config.model.early_2d_channels_mult = float(test.get("early_2d_channels_mult", 1.0))
    config.model.mid_2d_unet = bool(test.get("mid_2d_unet", False))
    config.model.factorized_2plus1d = bool(test.get("factorized_2plus1d", False))
    config.model.residual_unet = bool(test.get("residual_unet", False))
    config.model.gated_stems = bool(test.get("gated_stems", False))
    config.model.sparse_deep_supervision = bool(
        test.get("sparse_deep_supervision", False)
    )
    config.model.sparse_deep_supervision_dec2_weight = float(
        test.get("sparse_deep_supervision_dec2_weight", 0.3)
    )
    config.model.sparse_deep_supervision_dec3_weight = float(
        test.get("sparse_deep_supervision_dec3_weight", 0.1)
    )
    config.model.depth_attention_2d_head = bool(test.get("depth_attention_2d_head", False))
    config.model.divided_attention = bool(test.get("divided_attention", False))
    config.model.divided_attention_spatial = bool(
        test.get("divided_attention_spatial", False)
    )
    config.model.mednext_adapters = bool(test.get("mednext_adapters", False))
    config.model.mednext_kernel = int(test.get("mednext_kernel", 5))
    config.model.mednext_expansion = 2
    config.model.mixstyle = bool(test.get("mixstyle", False))
    config.model.mixstyle_prob = float(test.get("mixstyle_prob", 0.8))
    config.model.mixstyle_alpha = float(test.get("mixstyle_alpha", 0.1))
    config.model.sagnet = bool(test.get("sagnet", False))
    config.model.dual_scale = bool(test.get("dual_scale", False))
    config.model.dual_scale_local_size = int(test.get("dual_scale_local_size", 64))
    config.model.dual_scale_mix = float(test.get("dual_scale_mix", 0.25))
    config.model.dual_scale_deep = bool(test.get("dual_scale_deep", False))
    config.model.dual_scale_local_only = bool(
        test.get("dual_scale_local_only", False)
    )
    config.model.dual_scale_outer_size = int(
        test.get("dual_scale_outer_size", 0)
    )
    config.model.dual_scale_outer_mix = float(
        test.get("dual_scale_outer_mix", 0.0)
    )
    config.model.dual_scale_adaptive_gate = bool(
        test.get("dual_scale_adaptive_gate", False)
    )
    config.model.dual_scale_gate_max = float(
        test.get("dual_scale_gate_max", 1.0)
    )
    config.model.style_film = bool(test.get("style_film", False))
    config.model.style_film_hidden = int(test.get("style_film_hidden", 64))
    config.model.mae_reconstruction_head = config.tra.mae_reconstruction
    if "compile_model" in test:
        config.model.compile_model = bool(test["compile_model"])

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def preflight_pretraining(
    selected: list[dict],
    dry_run: bool,
    reuse_compatible: bool = False,
) -> None:
    from utils.model import create_model

    if not AVAILABLE_PRETRAIN_SCROLL_IDS:
        raise RuntimeError("no intended Campaign 26 pretraining zarrs are available")
    print(
        f"[campaign26] MAE corpus: present={len(AVAILABLE_PRETRAIN_SCROLL_IDS)}/"
        f"{len(PRETRAIN_SCROLL_IDS)} omitted_missing={list(MISSING_PRETRAIN_SCROLL_IDS)}",
        flush=True,
    )

    states = {}
    reported = set()
    for test in selected:
        key = str(test.get("pretrain_key", "base"))
        if key in reported:
            continue
        if key not in PRETRAIN_ROUTES:
            raise ValueError(f"unknown campaign-26 transfer key: {key}")
        checkpoint, strict, note = PRETRAIN_ROUTES[key]
        checkpoint_exists = checkpoint.is_file() and checkpoint.stat().st_size > 0
        if key in PRETRAIN_SPECS and not _pretraining_complete(key) and not (
            reuse_compatible and checkpoint_exists
        ):
            if dry_run:
                print(
                    f"[campaign26] matched pretraining missing: {key} "
                    f"(would run {PRETRAIN_STEPS} steps on "
                    f"{len(AVAILABLE_PRETRAIN_SCROLL_IDS)} zarrs)",
                    flush=True,
                )
                reported.add(key)
                continue
            command = [
                sys.executable,
                str(ROOT / "mae_pretrain_nnunet.py"),
                "--name", _pretrain_name(key),
                "--scroll-ids", *(str(scroll_id) for scroll_id in AVAILABLE_PRETRAIN_SCROLL_IDS),
                "--require-all-scrolls",
                "--ctx", "192",
                "--ds", "2",
                "--depth", "8",
                "--d-start", "10",
                "--d-end", "18",
                "--steps", str(PRETRAIN_STEPS),
                "--batch-size", "32",
                "--accum-steps", "1",
                "--init-weights", str(campaign25_pretrain_path("base")),
                "--freeze-loaded-backbone",
                *PRETRAIN_SPECS[key],
            ]
            print(
                f"[campaign26] running matched pretraining: {key} "
                f"({PRETRAIN_STEPS} steps, {len(AVAILABLE_PRETRAIN_SCROLL_IDS)} zarrs)",
                flush=True,
            )
            subprocess.run(command, cwd=ROOT, check=True)
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise RuntimeError(f"MAE pretraining did not create {checkpoint}")
            _pretrain_marker(key).write_text(
                json.dumps(_pretrain_metadata(key), indent=2) + "\n",
                encoding="utf-8",
            )
            if not _pretraining_complete(key):
                raise RuntimeError(f"Campaign 26 MAE checkpoint failed validation: {checkpoint}")
        elif key in PRETRAIN_SPECS and not _pretraining_complete(key):
            print(
                f"[campaign26] reusing existing compatible pretrain despite "
                f"corpus-marker drift: {key}",
                flush=True,
            )
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Campaign 26 pretraining checkpoint is missing: {checkpoint}")
        if checkpoint not in states:
            states[checkpoint] = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        state = states[checkpoint]
        config = build_config(test)
        config.device = "cpu"
        config.model.compile_model = False
        model, _ = create_model(config)
        target = model.state_dict()
        compatible = {
            name: value
            for name, value in state.items()
            if name in target and value.shape == target[name].shape
        }
        compatible_numel = sum(value.numel() for value in compatible.values())
        target_numel = sum(value.numel() for value in target.values())
        print(
            f"[campaign26] transfer {key}: {checkpoint.relative_to(ROOT)} "
            f"tensors={len(compatible)}/{len(target)} "
            f"numel={100.0 * compatible_numel / max(target_numel, 1):.2f}% "
            f"strict={strict} ({note})",
            flush=True,
        )
        del model, target, compatible
        reported.add(key)


def run_test(config, dry_run: bool) -> bool:
    print(f"\n{'=' * 78}\n[campaign26] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  epochs={config.tra.n_epochs} eval={config.tra.eval_int} "
        f"DANN={config.tra.dann} SupCon={config.tra.supcon} "
        f"weights={config.data.train_scroll_weights}",
        flush=True,
    )
    print(
        f"  norm={config.model.norm_mode} "
        f"weldon=({config.model.weldon_top_k},{config.model.weldon_bottom_k},"
        f"w={config.model.weldon_top_weight},multi={config.model.weldon_multi_k},"
        f"depth={config.model.weldon_depth_support_k})",
        flush=True,
    )
    print(
        f"  architecture=(early_2d={config.model.early_2d_unet}:"
        f"x{config.model.early_2d_channels_mult}, mid_2d={config.model.mid_2d_unet}, "
        f"factorized_2plus1d={config.model.factorized_2plus1d}, "
        f"residual={config.model.residual_unet}, gated_stems={config.model.gated_stems}, "
        f"deep_supervision={config.model.sparse_deep_supervision}, "
        f"depth_attention_2d={config.model.depth_attention_2d_head}, "
        f"mednext={config.model.mednext_adapters}:k{config.model.mednext_kernel}, "
        f"divided_attention={config.model.divided_attention}:"
        f"spatial={config.model.divided_attention_spatial})",
        flush=True,
    )
    print(
        f"  domain_generalization=(prototype={config.tra.prototype_align}, "
        f"coral={config.tra.coral_align}, cdan={config.tra.cdan}, "
        f"mldg={config.tra.mldg}:{config.tra.mldg_holdout_domain}:"
        f"random={config.tra.mldg_random_holdout}, "
        f"mixstyle={config.model.mixstyle}, sagnet={config.model.sagnet}, "
        f"dual_scale={config.model.dual_scale}:local{config.model.dual_scale_local_size}:"
        f"mix{config.model.dual_scale_mix}:deep={config.model.dual_scale_deep}:"
        f"only={config.model.dual_scale_local_only}:"
        f"outer{config.model.dual_scale_outer_size}:"
        f"mix{config.model.dual_scale_outer_mix}, film={config.model.style_film}, "
        f"mae_reconstruction={config.tra.mae_reconstruction}, "
        f"physical_groupdro={config.tra.physical_domain_groupdro})",
        flush=True,
    )
    print(
        f"  supervision=(labels={config.data.inklabel_dir}, "
        f"dilate={config.data.label_dilate_r}px, "
        f"ring={config.data.ring_label_source}:close{config.data.ring_close_r}:"
        f"gap{config.data.ring_gap_r}:shell{config.data.ring_shell_r}, "
        f"pos_only={config.data.multitile_pos_only})",
        flush=True,
    )
    print(
        f"  pretrained={config.init_weights} "
        f"strict_architecture_init={config.model.require_architecture_init}",
        flush=True,
    )
    if dry_run:
        print("  [DRY RUN] skipping", flush=True)
        return True

    from train import Trainer
    from utils.dataloader import cleanup_mmap_files

    trainer = None
    try:
        trainer = Trainer(config)
        trainer.vis.writer.add_scalar("Run/Initialized", 1.0, 0)
        trainer.vis.writer.flush()
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return False
    finally:
        if trainer is not None:
            trainer.close()
        del trainer
        gc.collect()
        cleanup_mmap_files()


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 26: normalization and dimensionality study")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument(
        "--restart-baseline",
        action="store_true",
        help="prepend baseline to a --from suffix without rerunning earlier arms",
    )
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
        if args.restart_baseline and args.from_id != "baseline":
            selected = [TESTS[ids.index("baseline")], *selected]
    elif args.restart_baseline:
        raise ValueError("--restart-baseline requires --from")

    preflight_train_masks(CAMPAIGN_SCROLLS)
    preflight_pretraining(selected, args.dry_run)

    print(f"[campaign26] {len(selected)} run(s) queued (log -> {LOG_DIR})")
    results = {}
    for test in selected:
        config = build_config(test)
        results[test["tid"]] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'=' * 78}\n[campaign26] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
