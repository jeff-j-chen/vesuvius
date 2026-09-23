"""campaign 31: mechanism combinations and early-2D follow-ups

Campaign 31 retains the Campaign 30 data protocol and tests complementary robust
objectives, promising architecture/objective combinations, and controlled early-2D
ablations. `mid_control_seed42` measures run-to-run noise for the mid control; the
researcher recipe is tested bundled at ds1 and ds2 and decomposed only if competitive.

PCGrad-Gram arms: projected PCGrad gradients stay in the span of the domain
gradients, so the update equals one backward of a reweighted domain loss whose
weights depend only on the domain-gradient gram matrix. The gram is measured from
disjoint per-domain sub-batches (about one extra forward/backward in total) rather
than one full-batch backward per domain (about 12x). `mid_domain_mean` isolates the
equal-domain weighting that full PCGrad also silently introduced.

Depth arms decompose Campaign 29 `overlap_depth12` (12 slices, five 4-slice windows):
`mid_depth12` adds slices without windows, `mid_overlap8` adds windows without slices,
and `mid_overlap12` replicates the bundle with threshold calibration. `early_overlap12`
applies the same 12-slice five-window bundle to the early_gated architecture.

Depth-latching arms: the window can shift toward the air side, a minimum-entropy
floor keeps mid depth attention from collapsing onto one or two slices, top-k
replaces amax at the collapse, and a low lse cap stops single voxels deciding a cell.

Usage:
    python3 campaign_archs_31.py --dry-run
    python3 campaign_archs_31.py --only mid_groupdro_anchor_ema
    python3 campaign_archs_31.py --from early_residual_depth3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_30 as campaign30
from utils.config import DEFAULT_SCROLLS, DEFAULT_TEST_SCROLL_IDS, startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs31"
MODEL_DIR = "models/archs31"
PRETRAIN_STEPS = 2_000
MIN_TRANSFER_COVERAGE = 0.85
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 3e-4
FINETUNE_BATCH_SIZE = 96
FINETUNE_LR = 1.5e-4
ALL_PRETRAIN_SCROLL_IDS = tuple(dict.fromkeys(
    [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
    + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
))


def _spec(*args, required=(), ctx=192, ds=2, depth=8, d_start=10, d_end=18):
    return {
        "args": tuple(args),
        "required": tuple(required),
        "ctx": int(ctx),
        "ds": int(ds),
        "depth": int(depth),
        "d_start": int(d_start),
        "d_end": int(d_end),
    }


PRETRAIN_SPECS = {
    "mid_gated_c30": {
        "reuse_campaign30": "mid_gated_c29",
        "required": ("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
        "ctx": 192,
        "ds": 2,
    },
    "early_deep_residual_c30": {
        "reuse_campaign30": "early_deep_residual2d",
        "required": (
            "gated_cue_stem.", "early2d_enc2.shortcut.",
            "early2d_extra_encoders.", "early2d_",
        ),
        "ctx": 192,
        "ds": 2,
    },
    "early_raw_instance_c30": {
        "reuse_campaign30": "early_raw_instance",
        "required": ("enc1.", "early_depth_attn.", "early2d_"),
        "ctx": 192,
        "ds": 2,
    },
    "mid_wide15_c30": {
        "reuse_campaign30": "mid_wide15_gated",
        "required": ("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
        "ctx": 192,
        "ds": 2,
    },
    "early_gated_c30": {
        "reuse_campaign30": "early_gated",
        "required": ("gated_cue_stem.", "early_depth_attn.", "early2d_"),
        "ctx": 192,
        "ds": 2,
    },
    "mid_overlap12_c29": {
        "reuse_campaign29": "mid_gated_overlap12_all",
        "required": campaign29.PRETRAIN_SPECS["mid_gated_overlap12_all"]["required"],
        "ctx": 192,
        "ds": 2,
    },
    "mid_depth12": _spec(
        "--mid-2d-unet", "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
        depth=12,
        d_start=8,
        d_end=20,
    ),
    "mid_overlap8": _spec(
        "--mid-2d-unet", "--gated-stems", "--overlapping-depth-windows",
        "--overlapping-depth-window-size", "4",
        "--overlapping-depth-window-stride", "2",
        "--norm-mode", "ibn_full",
        required=(
            "gated_cue_stem.", "mid2d_", "overlap_bottleneck_fuse.",
            "overlap_decoded_fuse.",
        ),
    ),
    "early_overlap12": _spec(
        "--early-2d-unet", "--gated-stems", "--overlapping-depth-windows",
        "--overlapping-depth-window-size", "4",
        "--overlapping-depth-window-stride", "2",
        "--norm-mode", "ibn_full",
        required=(
            "gated_cue_stem.", "early_depth_attn.", "early2d_",
            "overlap_bottleneck_fuse.", "overlap_decoded_fuse.",
        ),
        depth=12,
        d_start=8,
        d_end=20,
    ),
    "early_residual_depth3": _spec(
        "--early-2d-unet", "--residual-2d-unet", "--two-d-block-depth", "3",
        "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early2d_enc2.shortcut.", "early2d_"),
    ),
    "early_residual_extra320": _spec(
        "--early-2d-unet", "--residual-2d-unet",
        "--two-d-extra-channels", "320", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=(
            "gated_cue_stem.", "early2d_enc2.shortcut.",
            "early2d_extra_encoders.", "early2d_",
        ),
    ),
    "early_deep_nonresidual": _spec(
        "--early-2d-unet", "--two-d-block-depth", "3",
        "--two-d-extra-channels", "320", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early2d_extra_encoders.", "early2d_"),
    ),
    "early_gated_instance": _spec(
        "--early-2d-unet", "--gated-stems", "--norm-mode", "instance",
        required=("gated_cue_stem.", "early_depth_attn.", "early2d_"),
    ),
    "early_raw_ibn": _spec(
        "--early-2d-unet", "--raw-only-stem", "--norm-mode", "ibn_full",
        required=("enc1.", "early_depth_attn.", "early2d_"),
    ),
    "researcher_ds2_full": _spec(
        "--early-2d-unet", "--raw-only-stem", "--norm-mode", "instance",
        "--channels-mult", "0.5", "--residual-2d-unet",
        "--two-d-extra-channels", "256", "320",
        required=(
            "enc1.", "early2d_enc2.shortcut.",
            "early2d_extra_encoders.", "early2d_",
        ),
    ),
    "researcher_full": _spec(
        "--early-2d-unet", "--raw-only-stem", "--norm-mode", "instance",
        "--channels-mult", "0.5", "--residual-2d-unet",
        "--two-d-extra-channels", "256", "320",
        required=(
            "enc1.", "early2d_enc2.shortcut.",
            "early2d_extra_encoders.", "early2d_",
        ),
        ds=1,
    ),
}


def _test(tid: str, pretrain_key: str, **overrides) -> dict:
    test = {
        "tid": tid,
        "tag": f"31_{tid}",
        "pretrain_key": pretrain_key,
        "early_2d_unet": False,
        "mid_2d_unet": True,
        "gated_stems": True,
        "raw_only_stem": False,
        "norm_mode": "ibn_full",
        "context_downsample": 2,
        "channels_mult": 1.0,
        "early_2d_channels_mult": 1.0,
        "mid_2d_channels_mult": 1.0,
        "residual_2d_unet": False,
        "two_d_block_depth": 2,
        "two_d_extra_levels": 0,
        "two_d_extra_channels": (),
        "physical_domain_groupdro": False,
        "physical_patch_groupdro": False,
        "mae_anchor_lambda": 0.0,
        "model_ema": False,
        "model_ema_decay": 0.995,
        "pcgrad_gram": False,
        "pcgrad_gram_interval": 1,
        "pcgrad_gram_ema": 0.0,
        "calibrate_character_threshold": True,
        "seed": 41,
        "depth": 8,
        "overlapping_depth_windows": False,
        "overlapping_depth_window_size": 4,
        "overlapping_depth_window_stride": 2,
        "surface_window_offset": 0,
        "mid_depth_entropy_floor": 0.0,
        "mid_depth_entropy_lambda": 0.0,
        "mid_depth_max_mode": "amax",
        "mt_lse_r_max": 10.0,
        "topk_positive_fraction": 0.0,
        "explicit_negative_share": 0.0,
        "character_forgetting": False,
    }
    test.update(overrides)
    return test


TESTS = [
    # noise floor: nominally identical c29 baseline and c30 control differed by 0.058 f1
    _test("mid_control_seed42", "mid_gated_c30", seed=42),

    # groupdro+anchor and the triple; single-factor references come from c29
    _test(
        "mid_groupdro_anchor",
        "mid_gated_c30",
        physical_domain_groupdro=True,
        mae_anchor_lambda=0.001,
    ),
    _test(
        "mid_groupdro_anchor_ema",
        "mid_gated_c30",
        physical_domain_groupdro=True,
        mae_anchor_lambda=0.001,
        model_ema=True,
    ),

    # cheap pcgrad: equal-domain control, exact per-step gram, sparse ema gram
    _test(
        "mid_domain_mean",
        "mid_gated_c30",
        pcgrad_gram=True,
        pcgrad_gram_interval=0,
    ),
    _test("mid_depth12", "mid_depth12", depth=12),


    _test(
        "mid_pcgrad_gram_exact",
        "mid_gated_c30",
        pcgrad_gram=True,
        pcgrad_gram_interval=1,
    ),
    _test(
        "mid_pcgrad_gram_sparse4",
        "mid_gated_c30",
        pcgrad_gram=True,
        pcgrad_gram_interval=4,
        pcgrad_gram_ema=0.8,
    ),

    # overlap_depth12 decomposition: extra slices vs windowed architecture
    _test("mid_overlap8", "mid_overlap8", overlapping_depth_windows=True),
    _test(
        "mid_overlap12",
        "mid_overlap12_c29",
        depth=12,
        overlapping_depth_windows=True,
    ),
    _test(
        "early_overlap12",
        "early_overlap12",
        early_2d_unet=True,
        mid_2d_unet=False,
        depth=12,
        overlapping_depth_windows=True,
    ),

    # window shifted toward the air side; + matches the pherc0841 jitter that helped
    _test("mid_air_offset2", "mid_gated_c30", surface_window_offset=2),
    _test("mid_air_offset3", "mid_gated_c30", surface_window_offset=3),

    # depth latching at the mid collapse; floors are fractions of log(enc2 depth)
    _test(
        "mid_depth_entropy_weak",
        "mid_gated_c30",
        mid_depth_entropy_floor=0.5,
        mid_depth_entropy_lambda=0.03,
    ),
    _test(
        "mid_depth_entropy_strong",
        "mid_gated_c30",
        mid_depth_entropy_floor=0.8,
        mid_depth_entropy_lambda=0.1,
    ),
    _test(
        "mid_depth_entropy_strong_topk",
        "mid_gated_c30",
        mid_depth_entropy_floor=0.8,
        mid_depth_entropy_lambda=0.1,
        mid_depth_max_mode="topk",
    ),
    _test("mid_lse_capped", "mid_gated_c30", mt_lse_r_max=1.0),
    # deliberately overstrong upper bracket
    _test(
        "mid_depth_entropy_overstrong_lse",
        "mid_gated_c30",
        mid_depth_entropy_floor=0.95,
        mid_depth_entropy_lambda=0.3,
        mt_lse_r_max=1.0,
    ),

    # confirm the only arm above seed noise
    _test(
        "triple_seed42",
        "mid_gated_c30",
        physical_domain_groupdro=True,
        mae_anchor_lambda=0.001,
        model_ema=True,
        seed=42,
    ),

    # label-uncertainty-safe supervision and mining
    _test("mid_topk_bag_positive", "mid_gated_c30", topk_positive_fraction=0.5),
    _test(
        "mid_trusted_negative_mining",
        "mid_gated_c30",
        explicit_negative_share=0.2,
        character_forgetting=True,
    ),

    # architecture and robust-objective combinations
    _test(
        "early_deep_residual_groupdro_anchor",
        "early_deep_residual_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
        physical_domain_groupdro=True,
        mae_anchor_lambda=0.001,
    ),
    _test(
        "early_raw_instance_groupdro",
        "early_raw_instance_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
        physical_domain_groupdro=True,
    ),
    _test(
        "early_raw_instance_groupdro_anchor",
        "early_raw_instance_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
        physical_domain_groupdro=True,
        mae_anchor_lambda=0.001,
    ),
    _test(
        "mid_wide15_groupdro",
        "mid_wide15_c30",
        mid_2d_channels_mult=1.5,
        physical_domain_groupdro=True,
    ),
    _test(
        "early_gated_patch_groupdro",
        "early_gated_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        physical_patch_groupdro=True,
    ),

    # controlled early-depth and hierarchy decomposition
    _test(
        "early_residual_depth3",
        "early_residual_depth3",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
        two_d_block_depth=3,
    ),
    _test(
        "early_residual_extra320",
        "early_residual_extra320",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
        two_d_extra_channels=(320,),
    ),
    _test(
        "early_deep_nonresidual",
        "early_deep_nonresidual",
        early_2d_unet=True,
        mid_2d_unet=False,
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
    ),
    _test(
        "early_deep_residual_replica",
        "early_deep_residual_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
    ),

    # raw-stem and normalization controls for early_raw_instance
    _test(
        "early_gated_instance",
        "early_gated_instance",
        early_2d_unet=True,
        mid_2d_unet=False,
        norm_mode="instance",
    ),
    _test(
        "early_raw_ibn",
        "early_raw_ibn",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
    ),

    # researcher recipe bundled at ds2 and ds1; decompose later only if competitive
    _test(
        "researcher_ds2_full",
        "researcher_ds2_full",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
        channels_mult=0.5,
        residual_2d_unet=True,
        two_d_extra_channels=(256, 320),
    ),
    _test(
        "researcher_full",
        "researcher_full",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
        context_downsample=1,
        channels_mult=0.5,
        residual_2d_unet=True,
        two_d_extra_channels=(256, 320),
    ),

    
    # pcgrad-gram weights multiplied by physical groupdro weights
    _test(
        "mid_pcgrad_gram_groupdro",
        "mid_gated_c30",
        pcgrad_gram=True,
        pcgrad_gram_interval=4,
        pcgrad_gram_ema=0.8,
        physical_domain_groupdro=True,
    ),
]


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_campaign31_{key}_2k"


def _pretrain_path(key: str) -> Path:
    spec = PRETRAIN_SPECS[key]
    if spec.get("reuse_campaign29"):
        return campaign29._pretrain_path(str(spec["reuse_campaign29"]))
    reused = spec.get("reuse_campaign30")
    if reused:
        return campaign30._pretrain_path(str(reused))
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"


def _pretrain_marker(key: str) -> Path:
    return _pretrain_path(key).with_suffix(".complete.json")


def _pretrain_metadata(key: str) -> dict:
    spec = PRETRAIN_SPECS[key]
    return {
        "campaign": 31,
        "key": key,
        "steps": PRETRAIN_STEPS,
        "scroll_ids": list(ALL_PRETRAIN_SCROLL_IDS),
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


def _existing_pretraining_reusable(key: str) -> bool:
    """keep an existing checkpoint whose scroll list predates newly added test scrolls."""
    checkpoint = _pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    trained_ids = {int(value) for value in metadata.get("scroll_ids", ())}
    added_ids = set(ALL_PRETRAIN_SCROLL_IDS) - trained_ids
    spec = PRETRAIN_SPECS[key]
    if spec.get("reuse_campaign29"):
        expected = campaign29._pretrain_metadata(str(spec["reuse_campaign29"]))
    elif spec.get("reuse_campaign30"):
        key30 = str(spec["reuse_campaign30"])
        spec30 = campaign30.PRETRAIN_SPECS[key30]
        if spec30.get("reuse_campaign29") and key30 not in campaign30._MATCHED_PRETRAIN_OVERRIDES:
            expected = campaign29._pretrain_metadata(str(spec30["reuse_campaign29"]))
        else:
            expected = campaign30._pretrain_metadata(key30)
    else:
        expected = _pretrain_metadata(key)
    strip = lambda values: {k: v for k, v in values.items() if k != "scroll_ids"}
    return (
        bool(trained_ids)
        and strip(metadata) == strip(expected)
        and trained_ids <= set(ALL_PRETRAIN_SCROLL_IDS)
        and added_ids <= {int(value) for value in DEFAULT_TEST_SCROLL_IDS}
        and all(
            any(name.startswith(prefix) for name in state)
            for prefix in PRETRAIN_SPECS[key]["required"]
        )
    )


def _pretraining_complete(key: str) -> bool:
    spec = PRETRAIN_SPECS[key]
    if _existing_pretraining_reusable(key):
        return True
    if spec.get("reuse_campaign29"):
        return campaign29._pretraining_complete(str(spec["reuse_campaign29"]))
    reused = spec.get("reuse_campaign30")
    if reused:
        return campaign30._pretraining_complete(str(reused))
    checkpoint = _pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return metadata == _pretrain_metadata(key) and all(
        any(name.startswith(prefix) for name in state)
        for prefix in spec["required"]
    )


def build_config(test: dict):
    base = {
        "tid": test["tid"],
        "tag": test["tag"],
        "pretrain_key": "mid_gated_all",
    }
    config = campaign29.build_config(base)
    config.exp_name = f"{test['tag']}"
    config.tra.log_dir = LOG_DIR
    config.dl.batch_size = FINETUNE_BATCH_SIZE
    config.tra.lr = FINETUNE_LR
    config.data.context_size = 192
    config.data.context_downsample = int(test["context_downsample"])
    config.data.depth = int(test["depth"])
    config.data.surface_window_offset = int(test["surface_window_offset"])
    config.data.explicit_negative_share = float(test["explicit_negative_share"])
    config.model.mid_depth_entropy_floor = float(test["mid_depth_entropy_floor"])
    config.model.mid_depth_max_mode = str(test["mid_depth_max_mode"])
    config.model.mt_lse_r_max = float(test["mt_lse_r_max"])
    config.model.overlapping_depth_windows = bool(test["overlapping_depth_windows"])
    config.model.overlapping_depth_window_size = int(test["overlapping_depth_window_size"])
    config.model.overlapping_depth_window_stride = int(
        test["overlapping_depth_window_stride"]
    )
    config.model.multitile = True
    config.model.multitile_subtile = 16
    config.model.multitile_grid = 4
    config.model.surface_teacher_input = True
    config.model.early_2d_unet = bool(test["early_2d_unet"])
    config.model.mid_2d_unet = bool(test["mid_2d_unet"])
    config.model.gated_stems = bool(test["gated_stems"])
    config.model.raw_only_stem = bool(test["raw_only_stem"])
    config.model.norm_mode = str(test["norm_mode"])
    config.model.use_ibn = config.model.norm_mode == "ibn"
    config.model.channels_mult = float(test["channels_mult"])
    config.model.early_2d_channels_mult = float(test["early_2d_channels_mult"])
    config.model.mid_2d_channels_mult = float(test["mid_2d_channels_mult"])
    config.model.residual_2d_unet = bool(test["residual_2d_unet"])
    config.model.two_d_block_depth = int(test["two_d_block_depth"])
    config.model.two_d_extra_levels = int(test["two_d_extra_levels"])
    config.model.two_d_extra_channels = tuple(test["two_d_extra_channels"])
    config.model.two_d_bottleneck_channels = 0
    config.model.compile_model = False
    config.model.require_architecture_init = True

    config.tra.domain_gradient_mode = ""
    config.tra.domain_vrex = False
    config.tra.domain_cvar = False
    config.tra.pcgrad = False
    config.tra.pcgrad_lite = False
    config.tra.pcgrad_gram = bool(test["pcgrad_gram"])
    config.tra.pcgrad_gram_interval = int(test["pcgrad_gram_interval"])
    config.tra.pcgrad_gram_ema = float(test["pcgrad_gram_ema"])
    config.tra.physical_domain_groupdro = bool(test["physical_domain_groupdro"])
    config.tra.physical_domain_groupdro_eta = 0.05
    config.tra.physical_domain_groupdro_max_ratio = 3.0
    config.tra.physical_patch_groupdro = bool(test["physical_patch_groupdro"])
    config.tra.physical_patch_groupdro_eta = 0.05
    config.tra.physical_patch_groupdro_max_ratio = 3.0
    config.tra.mae_anchor_lambda = float(test["mae_anchor_lambda"])
    config.tra.model_ema = bool(test["model_ema"])
    config.tra.model_ema_decay = float(test["model_ema_decay"])
    config.tra.character_calibrate_threshold = bool(test["calibrate_character_threshold"])
    config.tra.seed = int(test["seed"])
    config.tra.mid_depth_entropy_lambda = float(test["mid_depth_entropy_lambda"])
    config.tra.topk_positive_fraction = float(test["topk_positive_fraction"])
    config.tra.character_forgetting = bool(test["character_forgetting"])
    config.init_weights = str(_pretrain_path(str(test["pretrain_key"])).relative_to(ROOT))

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    config.tra.character_forgetting_path = (
        os.path.join(checkpoint_dir, "character_forgetting.json")
        if config.tra.character_forgetting else ""
    )
    return config


def _transfer_coverage(test: dict, checkpoint: Path) -> tuple[float, int, int, int, int]:
    from utils.model import create_model

    config = build_config(test)
    config.device = "cpu"
    config.model.compile_model = False
    model, _ = create_model(config)
    if config.model.early_2d_unet:
        prefixes = ["enc1.", "early_depth_attn.", "early_depth_fuse.", "early2d_"]
    else:
        prefixes = [
            "enc1.", "enc2.", "mid_depth_attn.", "mid_depth_fuse.",
            "mid_skip1_fuse.", "mid2d_",
        ]
    if config.model.gated_stems:
        prefixes.append("gated_cue_stem.")
    excluded = ("early2d_head.", "mid2d_head.", "new_surface_input.")
    active = {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
        and not name.startswith(excluded)
    }
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    compatible = {
        name: parameter
        for name, parameter in active.items()
        if name in state and tuple(state[name].shape) == tuple(parameter.shape)
    }
    compatible_parameters = sum(parameter.numel() for parameter in compatible.values())
    active_parameters = sum(parameter.numel() for parameter in active.values())
    del state, model
    gc.collect()
    return (
        compatible_parameters / max(active_parameters, 1),
        compatible_parameters,
        active_parameters,
        len(compatible),
        len(active),
    )


def _campaign30_source_test(pretrain_key: str) -> dict:
    return next(
        test for test in campaign30.TESTS
        if str(test["pretrain_key"]) == pretrain_key
    )


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    missing = [
        scroll_id for scroll_id in ALL_PRETRAIN_SCROLL_IDS
        if not (ROOT / "ves_zarrs2" / f"{scroll_id}.zarr").is_dir()
    ]
    if missing:
        message = f"Campaign 31 requires every training and test zarr; missing={missing}"
        needs_pretraining = any(
            not _pretraining_complete(str(test["pretrain_key"])) for test in selected
        )
        if not dry_run and needs_pretraining:
            raise FileNotFoundError(message)
        print(f"[campaign31] WARNING {message}", flush=True)

    keys = list(dict.fromkeys(str(test["pretrain_key"]) for test in selected))
    tests_by_key = {str(test["pretrain_key"]): test for test in selected}
    for key in keys:
        if key not in PRETRAIN_SPECS:
            raise ValueError(f"unknown Campaign 31 pretrain key: {key}")
        spec = PRETRAIN_SPECS[key]
        if spec.get("reuse_campaign29"):
            if not _pretraining_complete(key):
                campaign29.preflight_pretraining(
                    [{"pretrain_key": str(spec["reuse_campaign29"])}],
                    dry_run,
                )
            continue
        reused = spec.get("reuse_campaign30")
        if reused:
            if not _pretraining_complete(key):
                campaign30.preflight_pretraining(
                    [_campaign30_source_test(str(reused))],
                    dry_run,
                )
            if not dry_run and not _pretraining_complete(key):
                raise RuntimeError(f"Campaign 31 reused pretraining is incomplete: {key}")
            continue
        if _pretraining_complete(key):
            continue
        if dry_run:
            print(
                f"[campaign31] would pretrain {key}: {PRETRAIN_STEPS} steps, "
                f"ctx={spec['ctx']} ds={spec['ds']} depth={spec['depth']} "
                f"batch={PRETRAIN_BATCH_SIZE} lr={PRETRAIN_LR}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "mae_pretrain_nnunet.py"),
            "--name", _pretrain_name(key),
            "--scroll-ids", *(str(value) for value in ALL_PRETRAIN_SCROLL_IDS),
            "--require-all-scrolls",
            "--physical-round-robin",
            "--ctx", str(spec["ctx"]),
            "--ds", str(spec["ds"]),
            "--depth", str(spec["depth"]),
            "--d-start", str(spec["d_start"]),
            "--d-end", str(spec["d_end"]),
            "--steps", str(PRETRAIN_STEPS),
            "--batch-size", str(PRETRAIN_BATCH_SIZE),
            "--accum-steps", "1",
            "--lr", str(PRETRAIN_LR),
            "--no-figures",
            *spec["args"],
        ]
        print(f"[campaign31] pretraining {key} from scratch", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        _pretrain_marker(key).write_text(
            json.dumps(_pretrain_metadata(key), indent=2) + "\n",
            encoding="utf-8",
        )
        if not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 31 MAE pretraining failed validation: {key}")
        test = tests_by_key[key]
        fraction, matched, total, _, _ = _transfer_coverage(test, _pretrain_path(key))
        if fraction < MIN_TRANSFER_COVERAGE:
            raise RuntimeError(
                f"Campaign 31 MAE transfer below {MIN_TRANSFER_COVERAGE:.0%} for {key}: "
                f"{matched}/{total} ({fraction:.2%})"
            )


def _print_run_header(config) -> None:
    print(f"\n{'=' * 78}\n[campaign31] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  pretrain={config.init_weights} batch={config.dl.batch_size} "
        f"lr={config.tra.lr} ds={config.data.context_downsample} "
        f"early={config.model.early_2d_unet} mid={config.model.mid_2d_unet} "
        f"residual={config.model.residual_2d_unet} "
        f"block_depth={config.model.two_d_block_depth} "
        f"extra={config.model.two_d_extra_channels} "
        f"depth={config.data.depth} overlap={config.model.overlapping_depth_windows}",
        flush=True,
    )
    print(
        f"  groupdro={config.tra.physical_domain_groupdro} "
        f"patchdro={config.tra.physical_patch_groupdro} "
        f"anchor={config.tra.mae_anchor_lambda} "
        f"ema={config.tra.model_ema}:{config.tra.model_ema_decay} "
        f"pcgrad_gram={config.tra.pcgrad_gram}:"
        f"every{config.tra.pcgrad_gram_interval}:ema{config.tra.pcgrad_gram_ema}",
        flush=True,
    )
    print(
        f"  window_offset={config.data.surface_window_offset} "
        f"depth_entropy={config.model.mid_depth_entropy_floor}"
        f"x{config.tra.mid_depth_entropy_lambda}:{config.model.mid_depth_max_mode} "
        f"lse_r_max={config.model.mt_lse_r_max} topk_pos={config.tra.topk_positive_fraction} "
        f"explicit_neg={config.data.explicit_negative_share} "
        f"forgetting={config.tra.character_forgetting} seed={config.tra.seed}",
        flush=True,
    )


def run_test(config, dry_run: bool) -> bool:
    with startup_output():
        _print_run_header(config)
    if dry_run:
        print(f"[DRY RUN] {config.exp_name}", flush=True)
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


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _cgroup_oom_kill_count() -> int:
    try:
        entries = dict(
            line.split(maxsplit=1)
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
        return int(entries.get("oom_kill", 0))
    except (OSError, TypeError, ValueError):
        return -1


def run_test_isolated(config) -> bool:
    if not hasattr(os, "fork"):
        return run_test(config, False)
    before = _open_fd_count()
    oom_before = _cgroup_oom_kill_count()
    pid = os.fork()
    if pid == 0:
        try:
            success = run_test(config, False)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if success else 1)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    after = _open_fd_count()
    oom_after = _cgroup_oom_kill_count()
    if os.WIFEXITED(status):
        detail = f"exit={os.WEXITSTATUS(status)}"
    elif os.WIFSIGNALED(status):
        number = os.WTERMSIG(status)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        detail = f"signal={name}"
        if oom_after > oom_before >= 0:
            detail += f" cgroup_oom_kill={oom_before}->{oom_after}"
    else:
        detail = "unknown"
    print(
        f"[campaign31] isolated arm pid={pid} status={status} {detail} "
        f"controller_fds={before}->{after}",
        flush=True,
    )
    if before >= 0 and after > before + 4:
        raise RuntimeError(f"Campaign 31 controller leaked file descriptors: {before}->{after}")
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="campaign 31: mechanism combinations and early-2D follow-ups"
    )
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

    campaign29.preflight_train_masks(
        campaign29.CAMPAIGN28_SCROLLS,
        inklabel_dir=ROOT / "inklabels",
        strict=not args.dry_run,
    )
    preflight_pretraining(selected, args.dry_run)
    print(f"[campaign31] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        if args.dry_run:
            success = run_test(config, True)
        else:
            campaign29.prewarm_data_cache(config)
            success = run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()

    print(f"\n{'=' * 78}\n[campaign31] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
