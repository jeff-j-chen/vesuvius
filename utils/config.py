"""config.py -- single source of truth for all training configuration.

train.py takes only `-n experiment_name`. everything else is set here by default
or overridden per-test in campaign runners by constructing a Config() and mutating
fields before passing to Trainer.

SCROLL CONFIGURATION:
  `scrolls` is a list of ScrollConfig objects (one per training fragment).
  each entry carries its zarr id, split axis/fraction, and any fragment-specific
  overrides.

CURRENT TRAINING SCROLLS (14 PHerc0139 9.362um / 113keV fragments):
  original 4:
    20260115000000  w044  split y 0.8055
    20250223000000  w059  split x 0.75
    20260206000001  w047  split x 0.75
    20260115000001  w056  split y 0.50
  new 10 (2026-07-21, all split x 0.75):
    w058 w052 w049 w046 w041 w040 w039 w038 w037 w034

CURRENT MODEL: v14_mil_deep (MIL with per-voxel logits + LSE aggregation).
  physics variants: v14b_mil_zgrad (depth-gradient channel), v14c_mil_lcn (local contrast norm + depth PE).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import torch


@dataclass
class ScrollConfig:
    scroll_id: int
    split_axis: str = "y"         # 'x' = vertical (left/right), 'y' = horizontal (top/bottom)
    train_split_frac: float = 0.8055
    crop_x_frac: tuple = (0.0, 1.0)
    crop_y_frac: tuple = (0.0, 1.0)


@dataclass
class ProbeROI:
    """a fixed readability probe: a window of tiles centred on (x, y) in pixel coords.
    x/y are snapped to tile_size multiples at render time so they align with the training grid."""
    x: int
    y: int
    label: str = ""   # e.g. 'easy' or 'hard'; used as tag prefix in TensorBoard


# default per-scroll probe ROIs — the same window size and grid snapping apply to all.
DEFAULT_PROBE_ROIS: Dict[int, List[ProbeROI]] = {
    20260115000000: [
        ProbeROI(3702, 3885, "easy"),
        ProbeROI(2612, 4900, "hard"),
    ],
    20250223000000: [
        ProbeROI(3826, 4096, "easy"),
        ProbeROI(5210, 2496, "hard"),
    ],
    20260206000001: [
        ProbeROI(5858, 1585, "easy"),
        ProbeROI(6419, 2431, "hard"),
    ],
    # w056 — labeled band spans y≈1837-4472; split at y=3590 (50%)
    # probes are approximate; adjust once ink distribution is known
    20260115000001: [
        ProbeROI(4870, 2500, "easy"),   # training half, center-ish
        ProbeROI(4870, 4000, "hard"),   # validation half
    ],
}


DEFAULT_SCROLLS: List[ScrollConfig] = [
    # original 4 PHerc0139 fragments
    ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044
    ScrollConfig(20250223000000, split_axis="x", train_split_frac=0.75),    # w059
    ScrollConfig(20260206000001, split_axis="x", train_split_frac=0.75),    # w047
    ScrollConfig(20260115000001, split_axis="y", train_split_frac=0.5),     # w056
    # 10 new PHerc0139 fragments (2026-07-21). vertical split, left 75% train / right 25% valid.
    # HOLDOUT FOR SANITY:
    # ScrollConfig(20260210000000, split_axis="x", train_split_frac=0.75),    # w058
    ScrollConfig(20260227000000, split_axis="x", train_split_frac=0.75),    # w052
    ScrollConfig(20260318000000, split_axis="x", train_split_frac=0.75),    # w049
    ScrollConfig(20260325000000, split_axis="x", train_split_frac=0.75),    # w046
    ScrollConfig(20260108000000, split_axis="x", train_split_frac=0.75),    # w041
    ScrollConfig(20250831000000, split_axis="x", train_split_frac=0.75),    # w040
    ScrollConfig(20260302000000, split_axis="x", train_split_frac=0.75),    # w039
    ScrollConfig(20260306000000, split_axis="x", train_split_frac=0.75),    # w038
    ScrollConfig(20260310000000, split_axis="x", train_split_frac=0.75),    # w037
    ScrollConfig(20260303000000, split_axis="x", train_split_frac=0.75),    # w034
]


@dataclass
class DataConfig:
    zarr_path: str = field(default_factory=lambda: os.getenv(
        "VESUVIUS_ZARR_PATH",
        "/vesuvius/ves_zarrs2" if os.name == "posix"
        else r"C:\Users\ChenJeff\Documents\ves_zarrs2"))

    scrolls: List[ScrollConfig] = field(default_factory=lambda: list(DEFAULT_SCROLLS))

    # test/inference scrolls (all rendered when test_int fires).
    # default = VC3D segments grown so far (PHerc0813 x3, PHerc1203 x1).
    test_scroll_ids: List[int] = field(default_factory=lambda: [
        20260716083545,   # auto_grown_20260716083545968  2.98cm²  max_gen=175  PHerc0813
        20260717193517,   # auto_grown_20260717193517520  11.49cm² max_gen=740  PHerc0211
        20260719202304,   # auto_grown_20260719202304218  10.74cm² max_gen=392  PHerc0211
        20260720090842,   # auto_grown_20260720090842117  7.90cm²  max_gen=345  PHerc1203
    ])

    @property
    def test_scroll_id(self) -> Optional[int]:
        """backward-compat: returns first test scroll id, or None"""
        return self.test_scroll_ids[0] if self.test_scroll_ids else None

    tile_size: int = 16
    depth: int = 8
    d_start: int = 0
    d_end: int = 28
    train_d_start: int = 0
    train_d_end: int = 28

    mask_memmap: bool = True
    ring_negatives: bool = True
    ring_label_source: str = "eroded"
    dense_labels: bool = False        # dense per-pixel BCE (model emits (B,1,T,T) map, not a tile scalar)
    dense_soft_labels: bool = False   # use soft_inklabels probability map as the dense target
    preload_to_ram: bool = False  # load full zarr into RAM; only useful if disk I/O is the bottleneck (it's not — chunks are uncompressed, OS caches them)
    # per-scroll probe ROIs: {scroll_id: [ProbeROI, ...]}
    probe_rois: Dict[int, List[ProbeROI]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROBE_ROIS.items()})

    # scroll2/3/4 ids: legacy eval targets; not used in the new pipeline.
    @property
    def scroll2_id(self) -> Optional[int]: return None
    @property
    def scroll3_id(self) -> Optional[int]: return None
    @property
    def scroll4_id(self) -> Optional[int]: return None
    @property
    def vis_scroll_ids(self) -> Optional[list]: return None

@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    data_aug: bool = False
    channel_mixing_prob: float = 0.25
    rotation_prob: float = 0.25
    flip_prob: float = 0.25
    noise_prob: float = 0.30
    brightness_prob: float = 0.50
    contrast_prob: float = 0.50


@dataclass
class TrainingConfig:
    n_epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 0.0
    l1_lambda: float = 3e-7
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = "./runs_p0139_triple"
    eval_int: int = 20
    test_int: int = 9999
    probe_int: int = 5               # render probe ROI figures every N epochs; set > n_epochs to disable
    probe_rois_enabled: bool = True
    label_smooth: float = 0.0        # label smoothing: 0 = hard 0/1, 0.05 = soft 0.05/0.95
    focal_gamma: float = 0.0         # >0 = focal loss (down-weights easy tiles); 0 = plain BCE
    ranking_lambda: float = 0.0      # weight on pairwise ranking (AUC surrogate) added to BCE; 0 = off
    ranking_neg_frac: float = 1.0    # 1.0 = full-AUC ranking; <1.0 = partial-AUC (hardest negatives only)
    ranking_margin: float = 0.3      # margin for the ranking hinge
    seed: int = 41                   # base RNG seed (torch/cuda/numpy/random + dataloader workers)
    deterministic: bool = False      # True = exact reproducibility (cudnn deterministic, no benchmark);
                                     # costs ~10-20% speed. False = fast path (cudnn benchmark, GPU atomics
                                     # -> tiny run-to-run differences even with a fixed seed)
    epoch_cooldown_secs: int = 9
    val_cooldown_secs: int = 12
    eval_cooldown_secs: int = 60
    fig_chunk_cooldown_ms: int = 60


@dataclass
class ModelConfig:
    arch: str = "v14_mil_deep"
    conv1_drop: float = 0.05   # Dropout3d between depth-mix conv blocks (channel-wise)
    conv2_drop: float = 0.075  # Dropout3d at end of depth-mix stage (channel-wise)
    head_drop:  float = 0.0    # Dropout3d before voxel head (closest to old FC-head dropout)


@dataclass
class HardMiningConfig:
    """hard negative / hard positive mining. set enabled=True to activate.
    mining files are written per-scroll by the evaluator; on multi-scroll runs each
    scroll mines independently (keyed by scroll_id) and the injector routes records
    back to the correct volume via the scroll_id field in each JSONL record."""
    enabled: bool = False
    hn_cutoff: float = 0.8    # tiles scoring above this with label=0 are hard negatives
    hp_cutoff: float = 0.45   # tiles scoring below this with label=1 are hard positives
    hm_frac: float = 0.1      # fraction of epoch tiles to replace with hard examples
    dir: str = "./hard_negs"  # root directory for per-experiment mining JSONL files


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

    def scroll_ids(self) -> List[int]:
        return [s.scroll_id for s in self.data.scrolls]

    def split_overrides(self) -> dict:
        return {
            s.scroll_id: {"axis": s.split_axis, "frac": s.train_split_frac}
            for s in self.data.scrolls
        }

    def tra_scroll_ids(self) -> List[int]:
        return self.scroll_ids()
