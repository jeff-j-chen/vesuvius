"""config.py -- current training configuration for the nnunet3d_lcndz path."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ScrollConfig:
    scroll_id: int
    split_axis: str = "y"
    train_split_frac: float = 0.8055
    crop_x_frac: tuple[float, float] = (0.0, 1.0)
    crop_y_frac: tuple[float, float] = (0.0, 1.0)


@dataclass
class ProbeROI:
    x: int
    y: int
    label: str = ""
    size: int = 576


def _load_probe_rois(cache_path: str = "probe_rois.json") -> Dict[int, List[ProbeROI]]:
    """load probe rois from the cache written by roi.py."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}

    out: Dict[int, List[ProbeROI]] = {}
    for scroll_id, boxes in (raw or {}).items():
        try:
            sid = int(scroll_id)
        except (TypeError, ValueError):
            continue
        rois: List[ProbeROI] = []
        for label in ("easy", "hard"):
            box = (boxes or {}).get(label)
            if not box or "x" not in box or "y" not in box:
                continue
            rois.append(ProbeROI(int(box["x"]), int(box["y"]), label, int(box.get("size", 576))))
        if rois:
            out[sid] = rois
    return out


DEFAULT_SCROLLS: List[ScrollConfig] = [
    ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),
    ScrollConfig(20260317000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20250223000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20251111010954, split_axis="x", train_split_frac=0.75),  # w068 PHerc0172
    ScrollConfig(20251112000002, split_axis="x", train_split_frac=0.75),  # w087 PHerc0172
    ScrollConfig(20240304141531, split_axis="x", train_split_frac=0.75),  # w013 PHerc1667
    ScrollConfig(20240304144031, split_axis="x", train_split_frac=0.75),  # w018 PHerc1667
    ScrollConfig(20231201215900, split_axis="x", train_split_frac=0.75),  # PHerc1667 Cr1 Fr3
    ScrollConfig(20250919125754, split_axis="x", train_split_frac=0.75),  # PHerc0009B 487
    ScrollConfig(20231210121321, split_axis="x", train_split_frac=0.75),  # PHercParis4
    ScrollConfig(20250628074500, split_axis="x", train_split_frac=0.6),
    ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20230301213755, split_axis="x", train_split_frac=0.75),  # PHercParis2 Fr143
    ScrollConfig(20231205222200, split_axis="x", train_split_frac=0.75),  # PHerc51 Cr4 Fr8
    ScrollConfig(20230301213423, split_axis="x", train_split_frac=0.75),  # PHercParis1 Fr34
    ScrollConfig(20250511003658, split_axis="x", train_split_frac=0.75),  # PHerc0343P
    ScrollConfig(20260221022814, split_axis="x", train_split_frac=0.75),  # PHerc0841
]

DEFAULT_TRAIN_SCROLL_DICT = {
    "pherc0139": [20260115000000, 20260317000000, 20250223000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc1667": [20240304141531, 20240304144031, 20231201215900],
    "pherc0009b": [20250919125754],
    "phercparis4": [20231210121321],
    "pherc0500p2": [20250628074500],
    "pherc0814": [20260226000000],
    "phercparis2_fr143": [20230301213755],
    "pherc51cr4_fr8": [20231205222200],
    "phercparis1_fr34": [20230301213423],
    "pherc0343p": [20250511003658],
    "pherc0841": [20260221022814],
}

DEFAULT_TEST_SCROLL_IDS = (
    20260814140748,
    20260717193517,
    20260720090842,
    20250703034159,
    20260723112922,
)


@dataclass
class DataConfig:
    zarr_path: str = field(
        default_factory=lambda: os.getenv(
            "VESUVIUS_ZARR_PATH",
            "/vesuvius/ves_zarrs2" if os.name == "posix" else r"C:\Users\ChenJeff\Documents\ves_zarrs2",
        )
    )
    scrolls: List[ScrollConfig] = field(default_factory=lambda: list(DEFAULT_SCROLLS))
    train_scroll_dict: Optional[Dict[str, List[int]]] = field(
        default_factory=lambda: {
            name: list(scroll_ids) for name, scroll_ids in DEFAULT_TRAIN_SCROLL_DICT.items()
        }
    )
    train_scroll_weights: Optional[List[int]] = field(
        default_factory=lambda: [4] + [1] * (len(DEFAULT_TRAIN_SCROLL_DICT) - 1)
    )
    test_scroll_ids: List[int] = field(
        default_factory=lambda: list(DEFAULT_TEST_SCROLL_IDS)
    )
    holdout_scroll_ids: List[int] = field(default_factory=lambda: [20251226000000])

    tile_size: int = 16
    depth: int = 8
    d_start: int = 4
    d_end: int = 28
    train_d_start: int = 4
    train_d_end: int = 28

    composite_method: str = "maxproj"
    composite_d0: int = 10
    composite_d1: int = 18
    composite_display: str = "raw"
    voxel_um: float = 9.362

    mask_memmap: bool = False
    mask_bitpack: bool = True
    preload_volumes: bool = False
    selective_chunk_preload: bool = True
    selective_chunk_workers: int = 8
    ram_safe_vis: bool = False
    ring_negatives: bool = True
    ring_label_source: str = "closed"
    ring_close_r: int = 2
    ring_gap_r: int = 2
    ring_shell_r: int = 4
    simple_split: bool = False  # true: axis/fraction split; false: train_masks/<scroll_id>.png
    coordinate_hash_split: bool = False
    coordinate_hash_block_size: int = 256
    coordinate_hash_valid_fraction: float = 0.25
    coordinate_hash_seed: int = 41
    train_mask_dir: str = "./train_masks"
    surface_label_dir: str = "./surface_labels"
    context_size: int = 192
    context_downsample: int = 2
    eval_infer_bs: int = 192
    eval_prefetch: int = 0   # >0 reads eval rows in N background threads to overlap disk i/o with gpu inference (0=serial)
    eval_chunk_gb: float = 0.25  # bounded host-RAM target for the final W044 figure
    tta_mode: str = "light"  # eval TTA view set: "light"=id+hflip (2x), "flips"=id+h+v+180 (4x), "dihedral"=+/-90 too (6x)
    probe_rois: Dict[int, List[ProbeROI]] = field(default_factory=_load_probe_rois)
    vis_scroll_ids: Optional[List[int]] = field(default_factory=lambda: [20260115000000])
    inklabel_dir: str = "./inklabels"
    label_dilate_r: int = 0
    dot_inklabel_dir: str = ""  # optional dir of binary dot labels; only positives are added to train
    dot_scroll_whitelist: List[int] = field(default_factory=list)  # if non-empty, load dots ONLY for these scroll ids
    ctx_jitter: int = 32  # max pixel jitter for context window; varies surrounding context during training
    target_aware_ctx_jitter: bool = True
    depth_jitter: int = 1  # max slice jitter for depth window start; attacks depth-profile position memorization
    surface_relative_depth_window: bool = True  # center the source window on the literal map
    multitile_train_step: int = 16  # dataloader window stride (px) in multitile mode
    multitile_pos_only: bool = True  # in ink-containing windows, supervise ONLY ink sub-tiles (mask out non-ink ones to avoid labelling unlabelled-ink neighbours as negatives); ink-free ring windows still give negatives
    character_balanced_sampling: bool = True
    character_balance_scrolls: bool = True
    character_min_pixels: int = 8
    max_samples_per_epoch: Optional[int] = 6667

    @property
    def test_scroll_id(self) -> Optional[int]:
        return self.test_scroll_ids[0] if self.test_scroll_ids else None


@dataclass
class DataloaderConfig:
    batch_size: int = 96
    num_workers: int = 12
    prefetch_factor: int = 2
    data_aug: bool = True
    rotation_prob: float = 0.6
    flip_prob: float = 0.6
    noise_prob: float = 0.0
    brightness_prob: float = 0.0
    contrast_prob: float = 0.0
    brightness_delta: float = 0.15
    contrast_delta: float = 0.15
    noise_std_min: float = 0.001
    noise_std_max: float = 0.005
    cutout_prob: float = 0.5
    cutout_max_frac: float = 0.16
    cutout_n_patches: int = 3
    depth_mask_prob: float = 0.0
    fda_prob: float = 0.0
    fda_beta: float = 0.05  # fraction of low-freq spectrum to swap
    elastic_prob: float = 0.0
    elastic_alpha: float = 15.0  # displacement magnitude in pixels
    elastic_sigma: float = 5.0   # gaussian smoothing sigma for displacement field
    depth_warp_prob: float = 0.0
    depth_warp_max: float = 2.0
    depth_warp_sigma: float = 24.0
    surface_atten_prob: float = 0.0
    surface_atten_min: float = 0.1
    surface_atten_max: float = 0.35
    surface_atten_sigma: float = 2.0
    acquisition_blur_prob: float = 0.0
    acquisition_blur_min: float = 0.4
    acquisition_blur_max: float = 0.9
    correlated_noise_prob: float = 0.0
    correlated_noise_min: float = 0.003
    correlated_noise_max: float = 0.015
    correlated_noise_sigma: float = 6.0
    cutout_protect_center: bool = True
    context_replace_prob: float = 0.35
    context_replace_keep_size: int = 0  # 0 = prediction center + 2*margin
    context_replace_margin: int = 20
    context_replace_feather: int = 40
    context_replace_min_mask_frac: float = 0.8
    context_replace_surface_align: bool = True


@dataclass
class TrainingConfig:
    n_epochs: int = 10
    lr: float = 1.5e-4
    encoder_lr_scale: float = 1.0
    encoder_freeze_epochs: int = 0
    warmup_epochs: int = 5
    weight_decay: float = 0.0
    l1_lambda: float = 0.0
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 15
    log_dir: str = "./runs_archs28"
    eval_int: int = 10
    eval_int_scrolls: int = 1
    test_int: int = 9999
    probe_int: int = 9999
    loss_type: str = "bce"
    gce_q: float = 0.9
    label_smooth_pos: float = 0.1
    label_smooth_neg: float = 0.05
    tta_consistency: bool = False
    tta_consistency_lambda: float = 0.3
    tta_consistency_mode: str = "flips"
    tta_consistency_prob: float = 1.0  # fraction of steps that run the consistency 2nd forward; <1 trades signal for speed
    tile_pos_weight: float = 1.0  # >0 up-weights positive tiles in the loss; needed for multitile (8px sub-tile imbalance ~5:1)
    tile_pos_weight_auto: bool = False  # when tile_pos_weight==0, compute pos_weight from data (neg/pos over supervised units) and cache per scroll+mode
    dann: bool = False
    dann_lambda: float = 0.0
    dann_n_domains: int = 12
    dann_grl_anneal: bool = False
    spill_reduction: bool = False
    spill_lambda: float = 0.0
    spill_depth_threshold: float = 0.35
    spill_active_depth_tau: float = 0.08
    spill_max_active_depth_frac: float = 0.35
    spill_min_depth_var: float = 0.8
    spill_prob: bool = False  # old probability-based active-depth-fraction approach
    spill_entropy: bool = False
    spill_entropy_lambda: float = 0.3
    spill_max_depth_entropy: float = 2.1
    new_surface_lambda: float = 0.2
    new_surface_smooth_lambda: float = 0.02
    surface_target_sigma: float = 0.75
    seed: int = 41
    deterministic: bool = False
    epoch_cooldown_secs: int = 0
    val_cooldown_secs: int = 0
    eval_cooldown_secs: int = 0
    fig_chunk_cooldown_ms: int = 0
    save_vis: bool = True
    fast_eval_figure: bool = True
    test_on_final: bool = False

    supcon: bool = False
    supcon_lambda: float = 0.1
    supcon_temp: float = 0.07
    supcon_proj_dim: int = 128
    supcon_hidden_dim: int = 256
    supcon_curriculum: bool = True
    supcon_lambda_start: float = 0.05
    supcon_lambda_end: float = 0.5
    supcon_curriculum_epochs: int = 8
    supcon_cross_frag: bool = False  # restrict supcon positives to cross-fragment pairs only
    supcon_ignore_same_domain_same_class: bool = False
    per_scroll_metrics: bool = True
    prototype_align: bool = False
    prototype_align_lambda: float = 0.1
    prototype_margin: float = 0.5
    coral_align: bool = False
    coral_align_lambda: float = 0.1
    coral_mean_weight: float = 1.0
    cdan: bool = False
    cdan_lambda: float = 0.1
    mldg: bool = False
    mldg_holdout_domain: int = -1
    mldg_random_holdout: bool = False
    mldg_inner_lr: float = 5e-4
    mldg_beta: float = 1.0
    sagnet_lambda: float = 0.1
    mae_reconstruction: bool = False
    mae_reconstruction_lambda: float = 0.1
    mae_reconstruction_start_epoch: int = 4
    mae_reconstruction_mask_frac: float = 0.5
    mae_reconstruction_patch: int = 4
    entropy_min_lambda: float = 0.0  # weight for entropy reward on unlabeled (validation) tiles
    entropy_min_batch_size: int = 8  # unlabeled samples per step; kept small to avoid OOM
    character_macro_metrics: bool = True
    character_score_threshold: float = 0.5
    character_calibrate_threshold: bool = False
    character_threshold_min: float = 0.1
    character_threshold_max: float = 0.9
    character_threshold_steps: int = 33
    character_recall_target: float = 0.5
    character_max_ring_fpr: float = 0.1
    character_checkpoint_metric: str = "character_ap_macro"
    sanity_guard_epoch: int = 0
    sanity_min_character_ap: float = 0.0
    sanity_min_specificity: float = 0.0
    context_consistency: bool = False
    context_consistency_prob: float = 0.25
    context_consistency_lambda: float = 0.1
    depth_view_consistency: bool = False
    depth_view_consistency_prob: float = 0.5
    depth_view_consistency_lambda: float = 0.2
    depth_view_consistency_offset: int = 2
    character_bag_ranking: bool = False
    character_bag_margin: float = 0.5
    character_bag_topk_frac: float = 0.5
    character_bag_lambda: float = 0.2
    character_groupdro: bool = False
    character_groupdro_eta: float = 0.05
    character_groupdro_max_ratio: float = 3.0
    physical_domain_groupdro: bool = False
    physical_domain_groupdro_eta: float = 0.05
    physical_domain_groupdro_max_ratio: float = 3.0
    physical_patch_groupdro: bool = False
    physical_patch_groupdro_eta: float = 0.05
    physical_patch_groupdro_max_ratio: float = 3.0
    domain_vrex: bool = False
    domain_vrex_lambda: float = 1.0
    domain_vrex_warmup_epochs: int = 2
    domain_cvar: bool = False
    domain_cvar_alpha: float = 0.25
    pcgrad: bool = False
    pcgrad_lite: bool = False
    pcgrad_lite_max_domains: int = 4
    pcgrad_lite_scope: str = "head"
    pcgrad_gram: bool = False
    pcgrad_gram_interval: int = 1  # 0 never measures conflicts: equal domain weights only
    pcgrad_gram_ema: float = 0.0
    model_ema: bool = False
    model_ema_decay: float = 0.999
    model_ema_start_epoch: int = 0
    mae_anchor_lambda: float = 0.0
    domain_gradient_mode: str = ""
    domain_gradient_threshold: float = 0.0
    domain_gradient_strength: float = 4.0
    domain_gradient_ema: float = 0.9
    domain_gradient_blend: float = 1.0
    character_cvar: bool = False
    character_cvar_alpha: float = 0.25
    clam_instance: bool = False
    clam_instance_k: int = 4
    clam_instance_lambda: float = 0.1
    sam_rho: float = 0.0
    elr: bool = False
    elr_start_epoch: int = 5
    elr_beta: float = 0.7
    elr_lambda: float = 0.1
    depth_shift_aux: bool = False
    depth_shift_aux_lambda: float = 0.1
    depth_shift_aux_classes: int = 3

@dataclass
class ModelConfig:
    arch: str = "nnunet3d_lcndz"
    compile_model: bool = False
    require_architecture_init: bool = True
    conv1_drop: float = 0.05
    conv2_drop: float = 0.05
    head_drop: float = 0.1
    attn_mil: bool = False
    attn_entropy_weight: float = 0.03
    feature_attn_mil: bool = False
    feature_depth_fusion: bool = False
    minimum_support_k: int = 0
    minimum_support_kernel: int = 3
    weldon_k: int = 0
    weldon_top_k: int = 0
    weldon_bottom_k: int = 0
    weldon_top_weight: float = 0.5
    weldon_multi_k: bool = False
    weldon_top_k2: int = 0
    weldon_bottom_k2: int = 0
    weldon_multi_mix: float = 0.5
    weldon_depth_support_k: int = 0
    fiber_coordinate_branch: bool = False
    early_2d_unet: bool = False
    early_2d_channels_mult: float = 1.0
    mid_2d_unet: bool = True
    mid_2d_channels_mult: float = 1.0
    residual_2d_unet: bool = False
    two_d_block_depth: int = 2
    two_d_bottleneck_channels: int = 0
    two_d_extra_levels: int = 0
    two_d_extra_channels: tuple[int, ...] = ()
    factorized_2plus1d: bool = False
    residual_unet: bool = False
    gated_stems: bool = True
    raw_only_stem: bool = False
    cue_dropout: float = 0.0
    explicit_depth_channels: bool = False
    overlapping_depth_windows: bool = False
    overlapping_depth_window_size: int = 4
    overlapping_depth_window_stride: int = 2
    depth_antialias: bool = False
    sparse_deep_supervision: bool = False
    sparse_deep_supervision_dec2_weight: float = 0.3
    sparse_deep_supervision_dec3_weight: float = 0.1
    depth_attention_2d_head: bool = False
    divided_attention: bool = False
    divided_attention_spatial: bool = False
    divided_attention_heads: int = 4
    divided_attention_window: int = 8
    mednext_adapters: bool = False
    mednext_kernel: int = 5
    mednext_expansion: int = 2
    mixstyle: bool = False
    mixstyle_prob: float = 0.8
    mixstyle_alpha: float = 0.1
    sagnet: bool = False
    dual_scale: bool = False
    dual_scale_local_size: int = 64
    dual_scale_mix: float = 0.25
    dual_scale_deep: bool = False
    dual_scale_local_only: bool = False
    dual_scale_outer_size: int = 0
    dual_scale_outer_mix: float = 0.0
    dual_scale_adaptive_gate: bool = False
    dual_scale_gate_max: float = 1.0
    style_film: bool = False
    style_film_hidden: int = 64
    mae_reconstruction_head: bool = False
    learned_surface: bool = False
    new_learned_surface: bool = False
    better_surface: bool = False
    surface_teacher_input: bool = True
    surface_canonicalize: bool = False
    surface_canonical_depth: int = 24
    surface_guided_mil: bool = False
    surface_guided_mix: float = 0.5
    surface_band_sigma: float = 1.5
    use_ibn: bool = False  # IBN-a: IN+BN hybrid in shallow encoder blocks
    norm_mode: str = "ibn_full"  # auto follows use_ibn; explicit: instance, ibn, ibn_full, batch
    use_prototype: bool = False  # replace bag-score with online prototype cosine classifier
    prototype_ema: float = 0.99
    skip_drop: float = 0.2  # prob of zeroing each skip connection; forces bottleneck reliance
    use_depth_profile: bool = False  # replace bag-score with depth-profile-only MLP (no spatial info)
    no_dz: bool = False  # zero the dz input channel; tests whether depth gradient or raw signal carries ink
    channels_mult: float = 1.0  # width multiplier on the 32/64/128/256 channel ladder (0.5 = half)
    allow_depth4: bool = False  # preserve depth at the third pool for four-slice experiments
    multitile: bool = True       # predict a grid of sub-tiles over the center instead of one 16px tile
    multitile_subtile: int = 16    # px per sub-tile prediction
    multitile_grid: int = 4       # grid side: 4 -> 16 predictions over a 64px center


@dataclass
class HardMiningConfig:
    enabled: bool = False
    hn_cutoff: float = 0.8
    hp_cutoff: float = 0.45
    hm_frac: float = 0.1
    dir: str = "./hard_negs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    dl: DataloaderConfig = field(default_factory=DataloaderConfig)
    tra: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hm: HardMiningConfig = field(default_factory=HardMiningConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model_dir: str = "models/archs28/mid_3d2d_gated"
    exp_name: Optional[str] = None
    init_weights: Optional[str] = "models/mae_nnunet_192_depth8_campaign26_mid_3d2d_full_ibn_2k.pth"
    save_final: Optional[str] = "models/archs28/mid_3d2d_gated/final.pth"

    def scroll_ids(self) -> List[int]:
        return [scroll.scroll_id for scroll in self.data.scrolls]

    def split_overrides(self) -> dict:
        return {
            scroll.scroll_id: {"axis": scroll.split_axis, "frac": scroll.train_split_frac}
            for scroll in self.data.scrolls
        }