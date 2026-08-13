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
    ScrollConfig(20250223000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260206000001, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260115000001, split_axis="y", train_split_frac=0.5),
    ScrollConfig(20260210000000, split_axis="x", train_split_frac=0.5),
    ScrollConfig(20260227000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260318000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260325000000, split_axis="x", train_split_frac=0.6),
    ScrollConfig(20260108000000, split_axis="x", train_split_frac=0.7),
    ScrollConfig(20250831000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260302000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260306000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260310000000, split_axis="x", train_split_frac=0.75),
    ScrollConfig(20260303000000, split_axis="y", train_split_frac=0.5),
    ScrollConfig(20260317000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),
    ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75),
]


@dataclass
class DataConfig:
    zarr_path: str = field(
        default_factory=lambda: os.getenv(
            "VESUVIUS_ZARR_PATH",
            "/vesuvius/ves_zarrs2" if os.name == "posix" else r"C:\Users\ChenJeff\Documents\ves_zarrs2",
        )
    )
    scrolls: List[ScrollConfig] = field(default_factory=lambda: list(DEFAULT_SCROLLS))
    test_scroll_ids: List[int] = field(
        default_factory=lambda: [
            20260716083545,
            20260717193517,
            20260720090842,
            20250703034159,
            20260723112922,
        ]
    )
    holdout_scroll_ids: List[int] = field(default_factory=lambda: [20251226000000])

    tile_size: int = 16
    depth: int = 8
    d_start: int = 0
    d_end: int = 28
    train_d_start: int = 0
    train_d_end: int = 28

    composite_method: str = "maxproj"
    composite_d0: int = 10
    composite_d1: int = 18
    composite_display: str = "raw"
    voxel_um: float = 9.362

    mask_memmap: bool = True
    mask_bitpack: bool = True
    ring_negatives: bool = True
    ring_label_source: str = "eroded"
    ring_close_r: int = 3
    ring_gap_r: int = 3
    ring_shell_r: int = 2
    context_size: int = 0
    context_downsample: int = 1
    eval_infer_bs: int = 128
    probe_rois: Dict[int, List[ProbeROI]] = field(default_factory=_load_probe_rois)
    vis_scroll_ids: Optional[List[int]] = None

    @property
    def test_scroll_id(self) -> Optional[int]:
        return self.test_scroll_ids[0] if self.test_scroll_ids else None


@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    data_aug: bool = False
    rotation_prob: float = 0.25
    flip_prob: float = 0.25
    noise_prob: float = 0.30
    brightness_prob: float = 0.50
    contrast_prob: float = 0.50
    brightness_delta: float = 0.15
    contrast_delta: float = 0.15
    noise_std_min: float = 0.001
    noise_std_max: float = 0.005
    cutout_prob: float = 0.0
    cutout_max_frac: float = 0.35
    cutout_n_patches: int = 1
    depth_mask_prob: float = 0.0


@dataclass
class TrainingConfig:
    n_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    l1_lambda: float = 3e-7
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = "./runs"
    eval_int: int = 20
    eval_int_scrolls: int = 2
    test_int: int = 9999
    probe_int: int = 5
    loss_type: str = "gce"
    gce_q: float = 0.7
    label_smooth_pos: float = 0.0
    label_smooth_neg: float = 0.0
    tta_consistency: bool = False
    tta_consistency_lambda: float = 0.5
    tta_consistency_mode: str = "flips"
    seed: int = 41
    deterministic: bool = True
    epoch_cooldown_secs: int = 9
    val_cooldown_secs: int = 12
    eval_cooldown_secs: int = 60
    fig_chunk_cooldown_ms: int = 60
    save_vis: bool = False
    fast_eval_figure: bool = False
    test_on_final: bool = False

    supcon: bool = False
    supcon_lambda: float = 0.1
    supcon_temp: float = 0.07
    supcon_proj_dim: int = 128
    supcon_hidden_dim: int = 256
    supcon_curriculum: bool = False
    supcon_lambda_start: float = 0.1
    supcon_lambda_end: float = 0.5
    supcon_curriculum_epochs: int = 15


@dataclass
class ModelConfig:
    arch: str = "nnunet3d_lcndz"
    conv1_drop: float = 0.05
    conv2_drop: float = 0.075
    head_drop: float = 0.0
    attn_mil: bool = False
    attn_entropy_weight: float = 0.0
    learned_surface: bool = False


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
    model_dir: str = "models"
    exp_name: Optional[str] = None
    init_weights: Optional[str] = None

    def scroll_ids(self) -> List[int]:
        return [scroll.scroll_id for scroll in self.data.scrolls]

    def split_overrides(self) -> dict:
        return {
            scroll.scroll_id: {"axis": scroll.split_axis, "frac": scroll.train_split_frac}
            for scroll in self.data.scrolls
        }