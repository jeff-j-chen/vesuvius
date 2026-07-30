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
    """a fixed readability probe: a square window whose TOP-LEFT corner is (x, y) in full-res
    pixel coords. x/y are snapped to the model grid (max(tile_size, context_size)) at render
    time so the window's tiles land exactly on the grid the model trains on."""
    x: int
    y: int
    label: str = ""   # e.g. 'easy' or 'hard'; used as tag prefix in TensorBoard
    size: int = 576   # window side in px; 576 = LCM(16,32,48)*6 so it fits any model grid


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


def _load_probe_rois_from_disk(cache_path: str = "probe_rois.json") -> Dict[int, List[ProbeROI]]:
    """read the unified probe-ROI cache written by roi.py (a single file, like norm_cache.json).

    structure: {"<scroll_id>": {"easy": {x,y,size}, "hard": {x,y,size}}} with top-left coords in
    full-res px. returns {} when the file is missing so the hardcoded DEFAULT_PROBE_ROIS remain
    the fallback."""
    out: Dict[int, List[ProbeROI]] = {}
    if not os.path.isfile(cache_path):
        return out
    import json as _json
    try:
        with open(cache_path) as f:
            data = _json.load(f)
    except Exception:
        return out
    for sid_str, boxes in (data or {}).items():
        try:
            sid = int(sid_str)
        except (ValueError, TypeError):
            continue
        rois: List[ProbeROI] = []
        for label in ("easy", "hard"):
            b = (boxes or {}).get(label)
            if b and "x" in b and "y" in b:
                rois.append(ProbeROI(int(b["x"]), int(b["y"]), label, int(b.get("size", 576))))
        if rois:
            out[sid] = rois
    return out


def _default_probe_rois() -> Dict[int, List[ProbeROI]]:
    """disk-annotated ROIs (roi.py) override the hardcoded defaults per scroll."""
    merged = {k: list(v) for k, v in DEFAULT_PROBE_ROIS.items()}
    merged.update(_load_probe_rois_from_disk())
    return merged


DEFAULT_SCROLLS: List[ScrollConfig] = [
    # original 4 PHerc0139 fragments
    ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044
    ScrollConfig(20250223000000, split_axis="x", train_split_frac=0.75),    # w059
    ScrollConfig(20260206000001, split_axis="x", train_split_frac=0.75),    # w047
    ScrollConfig(20260115000001, split_axis="y", train_split_frac=0.5),     # w056
    # 10 new PHerc0139 fragments (2026-07-21). vertical split, left 75% train / right 25% valid.
    ScrollConfig(20260210000000, split_axis="x", train_split_frac=0.75),    # w058
    ScrollConfig(20260227000000, split_axis="x", train_split_frac=0.75),    # w052
    ScrollConfig(20260318000000, split_axis="x", train_split_frac=0.75),    # w049
    ScrollConfig(20260325000000, split_axis="x", train_split_frac=0.75),    # w046
    ScrollConfig(20260108000000, split_axis="x", train_split_frac=0.75),    # w041
    ScrollConfig(20250831000000, split_axis="x", train_split_frac=0.75),    # w040
    ScrollConfig(20260302000000, split_axis="x", train_split_frac=0.75),    # w039
    ScrollConfig(20260306000000, split_axis="x", train_split_frac=0.75),    # w038
    ScrollConfig(20260310000000, split_axis="x", train_split_frac=0.75),    # w037
    ScrollConfig(20260303000000, split_axis="x", train_split_frac=0.75),    # w034
    # PHerc0814 segment 46527 (2026-07-22) — different scroll; horizontal split (top 75% train)
    ScrollConfig(20260226000000, split_axis="y", train_split_frac=0.75),    # seg46527 P0814
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
        20250703034159,   # auto_grown_20250703034159599  51.27cm² max_gen=638  PHerc1447 (8.64µm src)
    ])

    # holdout scroll(s): rendered as full-size test figures alongside test_scroll_ids but
    # NEVER trained on — the hallucination sanity check. w055 = PHerc0139.
    holdout_scroll_ids: List[int] = field(default_factory=lambda: [20251226000000])

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

    # composite (fiber-visibility) test-figure render, matched to VC3D Compositing.hpp:
    # a MAX filter over a limited ~8-layer surface window, displayed as raw 0-255.
    composite_method: str = "maxproj"
    composite_d0: int = 10
    composite_d1: int = 18
    composite_display: str = "raw"   # "raw" = linear 0-255 (VC3D); "stretch" = 1-99 in-mask
    voxel_um: float = 9.362          # microns/pixel at full res (drives the 1 cm scale bar)

    mask_memmap: bool = True
    ring_negatives: bool = True
    ring_label_source: str = "eroded"
    # 'closed' ring params (only used when ring_label_source == 'closed'). all radii are in
    # TILE units, so physical distance = radius * tile_size (16px). closed now builds off the
    # (hand-cleaned) eroded map, not original inklabels.
    ring_close_r: int = 3      # tiles: close letter interior holes (dilate then erode)
    ring_gap_r: int = 3        # tiles: air gap between ink edge and ring start (96px @ tile=16)
    ring_shell_r: int = 2      # ring shell width; 0 = dynamic (balance to ink count), >0 = fixed
    context_size: int = 0      # >0: input crop size (px) centered on each tile; model center-pools MIL
                               # over the tile region. label/mask stay the center tile. 0 = off (plain tile)
    context_downsample: int = 1    # >1: avg-pool the context crop by this factor at the stem, so the
                                   # model keeps the FULL context extent but at a coarser resolution
                                   # (~1/ds^2 the activations -> near-plain compute, less overfit, no OOM)
    eval_cmap_norm: str = "raw"    # display-only contrast for eval pred panels: "raw" | "percentile" | "rank"
                                   # raw = true probability (DEFAULT; pool-independent, matches test figs).
                                   # rank (histogram-equalize) spreads saturated outputs but is pool-relative;
                                   # percentile stretches [p2,p98]
    tta_mode: str = "flips"        # TTA transforms: "flips" (4: id,h,v,180 -- fast, contiguous, label-natural)
                                   # or "dihedral" (6: adds +/-90 rot -- slower, non-contiguous, unnatural for text)
    eval_infer_bs: int = 128         # 0 = auto-size the eval/test inference batch; >0 = manual override (use spare VRAM)
    dense_labels: bool = False        # dense per-pixel BCE (model emits (B,1,T,T) map, not a tile scalar)
    dense_soft_labels: bool = False   # use soft_inklabels probability map as the dense target
    preload_to_ram: bool = False  # load full zarr into RAM; only useful if disk I/O is the bottleneck (it's not — chunks are uncompressed, OS caches them)
    # per-scroll probe ROIs: {scroll_id: [ProbeROI, ...]}
    probe_rois: Dict[int, List[ProbeROI]] = field(default_factory=_default_probe_rois)

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
    # --- augmentation MAGNITUDES (were hardcoded in dataloader.Transform; now tracked here) ---
    # brightness/contrast: factor ~ uniform(1-delta, 1+delta), one shared factor across depth.
    brightness_delta: float = 0.15
    contrast_delta: float = 0.15
    # gaussian noise: per-voxel std ~ uniform(noise_std_min, noise_std_max) on [0,1] data.
    noise_std_min: float = 0.001
    noise_std_max: float = 0.005
    # specaugment-style masking (were untracked loose attrs; declared so they hit config.json).
    cutout_prob: float = 0.0        # prob of applying XY cutout patches to a block
    cutout_max_frac: float = 0.35   # each patch side up to this fraction of H/W
    cutout_n_patches: int = 1       # number of cutout patches per block
    depth_mask_prob: float = 0.0    # per-depth-slice independent zero-out probability

@dataclass
class TrainingConfig:
    n_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0        # AdamW decoupled weight decay (overfit lever). 0 = off; sane on-value ~1e-2 for this small model
    l1_lambda: float = 3e-7
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = "./runs_p0139_triple"
    eval_int: int = 20
    eval_int_scrolls: int = 2         # eval figures: render only the FIRST N scrolls (the slow part); probes/test unaffected
    test_int: int = 9999
    probe_int: int = 5               # render probe ROI figures every N epochs; set > n_epochs to disable
    probe_rois_enabled: bool = True
    label_smooth: float = 0.1        # label smoothing: 0 = hard 0/1, 0.1 = soft 0.1/0.9 (default on)
    pos_weight_enabled: bool = False  # False = ignore the neg/pos pos_weight (temper upward bias toward 1.0)
    loss_type: str = "gce"           # "bce" | "focal" | "gce" (noise-robust generalized CE) -- default on
    gce_q: float = 0.7               # GCE q in (0,1]: ->0 behaves like CE, =1 like noise-robust MAE loss
    focal_gamma: float = 0.0         # >0 = focal loss (down-weights easy tiles); 0 = plain BCE
    ranking_lambda: float = 0.0      # weight on pairwise ranking (AUC surrogate) added to BCE; 0 = off
    ranking_neg_frac: float = 1.0    # 1.0 = full-AUC ranking; <1.0 = partial-AUC (hardest negatives only)
    ranking_margin: float = 0.3      # margin for the ranking hinge
    # TTA-consistency regularizer: each step forward an EXTRA flipped view and penalize the two
    # tile predictions' disagreement (invariance regularizer -> fewer holdout hallucinations than
    # augmentation alone, which never forces two views to AGREE). ~2x forward cost when on.
    tta_consistency: bool = False         # master switch (False = off, no extra forward). on at run 4
    tta_consistency_lambda: float = 0.5   # weight of the consistency term (sane default when enabled)
    tta_consistency_mode: str = "flips"   # "flips" = random h/v/180 flip per step (label-natural for text)
    seed: int = 41                   # base RNG seed (torch/cuda/numpy/random + dataloader workers)
    deterministic: bool = True      # True = exact reproducibility (cudnn deterministic, no benchmark);
                                     # costs ~10-20% speed. False = fast path (cudnn benchmark, GPU atomics
                                     # -> tiny run-to-run differences even with a fixed seed)
    epoch_cooldown_secs: int = 9
    val_cooldown_secs: int = 12
    eval_cooldown_secs: int = 60
    fig_chunk_cooldown_ms: int = 60
    # save the full-size eval figures (one per training scroll) to
    # ./output/visualizations/<exp_name>/ at eval_int. off by default (TensorBoard only).
    save_vis: bool = False
    # render the test/holdout figure ONCE on the final epoch even when test_int never fires
    # (e.g. test_int=9999). used by leave-one-out campaigns to infer the held-out fragment.
    test_on_final: bool = False

    # ---- campaign_archs: architectural regularization levers (all off by default) ----
    # DANN: domain-adversarial head + gradient reversal (DANN, Ganin 2015). forces the backbone
    # to produce scroll-invariant features by training a domain classifier adversarially.
    dann: bool = False
    dann_lambda: float = 0.1           # gradient reversal strength (ramps up over dann_ramp_epochs)
    dann_ramp_epochs: int = 5          # linear ramp from 0 -> dann_lambda over this many epochs
    dann_n_domains: int = 15           # number of scroll-id classes (= number of training scrolls)

    # SupCon: supervised contrastive learning auxiliary head (Khosla 2020). pulls same-class tile
    # embeddings together and pushes cross-class ones apart -> transferable ink boundary.
    supcon: bool = False
    supcon_lambda: float = 0.1         # weight of the supcon loss term
    supcon_temp: float = 0.07          # temperature for the contrastive softmax

    # mean-teacher self-training on verified negatives from 2.4um inklabels + test scrolls.
    # VERIFIED NEGATIVES: tiles where 2.4um inklabel < verified_neg_threshold are trusted
    # papyrus -> hard-negative supervision reinforced during training.
    mean_teacher: bool = False
    mean_teacher_alpha: float = 0.999  # EMA decay for teacher weights (higher = slower update)
    mean_teacher_ramp_epochs: int = 3  # ramp consistency weight from 0 over this many epochs
    mean_teacher_lambda: float = 0.1   # consistency loss weight (student vs teacher, unlabeled)
    verified_neg_threshold: int = 26   # 2.4um label < this = definite papyrus (~0.1 ink prob)
    verified_neg_lambda: float = 0.2   # extra weight on verified-negative supervised tiles
    # test-scroll consistency: enforce student==teacher on unlabeled test-scroll tiles
    test_consistency: bool = False
    test_consistency_lambda: float = 0.1


@dataclass
class ModelConfig:
    arch: str = "v14_mil_deep"
    conv1_drop: float = 0.05   # Dropout3d between depth-mix conv blocks (channel-wise)
    conv2_drop: float = 0.075  # Dropout3d at end of depth-mix stage (channel-wise)
    head_drop:  float = 0.0    # Dropout3d before voxel head (closest to old FC-head dropout)
    attn_mil:   bool  = False  # use gated attention-MIL instead of fixed LSE aggregation


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
    init_weights: Optional[str] = None   # path to a (MAE) checkpoint to warm-start from (strict=False)

    def scroll_ids(self) -> List[int]:
        return [s.scroll_id for s in self.data.scrolls]

    def split_overrides(self) -> dict:
        return {
            s.scroll_id: {"axis": s.split_axis, "frac": s.train_split_frac}
            for s in self.data.scrolls
        }

    def tra_scroll_ids(self) -> List[int]:
        return self.scroll_ids()
