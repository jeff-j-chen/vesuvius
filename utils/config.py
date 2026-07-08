from dataclasses import dataclass, field
from typing import Optional
import os
import torch

@dataclass
class DataConfig:
    # on linux (remote server) scrolls live on the mounted network volume at /workspace;
    # on windows fall back to the old local docs path. override via VESUVIUS_ZARR_PATH.
    zarr_path: str = field(default_factory=lambda: os.getenv(
        "VESUVIUS_ZARR_PATH",
        "/workspace/ves_zarrs2" if os.name == "posix" else "C:\\Users\\ChenJeff\\Documents\\ves_zarrs2"))
    # tra_scroll_id / tra_scroll_ids: the scroll fragment(s) we TRAIN on. renamed from
    # scroll1_id to make clear the training set is a distinct (and now switchable) set
    # of scrolls, independent of the goal/test scrolls (scroll2/scroll3/scroll4).
    # now points at the scroll4 (PHerc1667) w018 dense-text patch, reconstructed at
    # 2.399um in-plane / depth-resampled 109->64 via reconstruct_scroll4_patch.py.
    # override with --scroll-id to train on a different fragment.
    tra_scroll_id: int = 20240304144031
    # tra_scroll_ids: list of scroll fragments to train on simultaneously (merged batches).
    # defaults to [tra_scroll_id]; set via --scroll-ids to add more fragments.
    # each fragment uses the same 75/25 right/left train/valid split.
    tra_scroll_ids: list = field(default_factory=lambda: [20240304144031])
    # subset of tra_scroll_ids that render EVALUATION figures (and, when hard mining is
    # on, mine) on each eval step. None = all training scrolls (default, = current
    # behavior). set via --vis-scroll-ids to watch only some fragments when training
    # on many. NOTE: test figures are unaffected (they render for every scroll unless
    # test_scroll2_only); probes are scroll-independent and always render once.
    vis_scroll_ids: Optional[list] = None
    scroll2_id: int = 20230709155141
    scroll4_id: int = 20231210132040
    # scroll3 (PHerc332) goal-scroll transfer target: same 7.91um modality as the scroll4
    # training run. rendered as its OWN separate test figure alongside scroll2 when the test
    # interval fires. loaded lazily/defensively (skipped if its zarr/mask are absent).
    scroll3_id: int = 20240716140050
    # when false, 'test' figures use scroll2 instead of scroll4
    test_on_scroll4: bool = False
    # when true, the test figure renders ONLY the goal scroll2 fragment (full extent)
    # and skips the expensive full training-scroll "Test" figure + scroll4.
    # used to keep end-of-training test inference affordable in campaigns.
    test_scroll2_only: bool = False
    tile_size: int = 32
    depth: int = 8
    d_start: int = 28
    d_end: int = 48
    # training-only depth window; inference still uses d_start/d_end
    train_d_start: int = 32
    train_d_end: int = 40
    # gaussian blur applied to prediction maps at inference time (0 = off)
    # promotes spatial coherence without changing what the model learns
    smooth_sigma: float = 0.0
    # input representation mode for training
    #   single: 8-slice ink-band window (current behavior)
    #   diff:   ink_band - pre_band  (differential absorption = ink signal)
    #   triple: concat(pre_band, ink_band, post_band) = 24 depth channels
    input_mode: str = "single"
    pre_band_start: int = 20   # guaranteed no-ink band before the ink window
    pre_band_end: int = 28
    post_band_start: int = 40  # guaranteed no-ink band after the ink window
    post_band_end: int = 48
    # when true: load the full zarr volume into a numpy array at startup
    # makes all reads RAM-speed instead of disk-speed; only viable for small scrolls
    # set false when training on terabytes of data
    preload_to_ram: bool = False
    # when true: back each dataset's binary mask/labels with an on-disk memmap so
    # they pickle as a file path (a few bytes) instead of the full array. removes the
    # per-worker RAM duplication on windows spawn — matters at the 5-10 fragment scale
    # where N x hundreds-of-MB x num_workers would otherwise balloon. default off so
    # small/existing runs are unchanged. scratch dir via env VESUVIUS_MMAP_DIR.
    mask_memmap: bool = False
    # when true: restrict training negatives to a ring of dilated-inklabel pixels
    # eliminates the risk of training on unlabeled ink as negatives; balances to ~1:1
    ring_negatives: bool = False
    # which inklabels to use for ring boundary computation:
    #   'eroded' (old behavior): ring computed from eroded_inklabels/
    #   'original': ring computed from inklabels/ — no original ink pixel enters ring as negative
    #   'closed': dilate+erode original labels to close letter holes, add air gap, then ring
    # training positive labels always come from eroded_inklabels (conservative, clean)
    ring_label_source: str = 'closed'
    # alternating ring: on odd epochs use the ring training set, on even epochs use the
    # full training set. hard mining runs only on ring-epoch data (closer to the decision
    # boundary). ring set sees ~2x the epochs relative to a pure full-set run.
    alternating_ring: bool = False
    # soft depth label sampling: for labeled ink tiles, with prob soft_label_prob
    # fetch from the flanking band instead and assign label soft_label_value
    soft_label_prob: float = 0.0
    soft_label_value: float = 0.3
    # train/val split axis: 'x' = legacy vertical (left train / right valid);
    # 'y' = horizontal (top train / bottom valid). the scroll4 7.91 dual-res test uses 'y'.
    split_axis: str = 'x'
    # fraction of the (cropped) split axis given to TRAIN (rest = valid). 0.75 = legacy.
    train_split_frac: float = 0.75
    # optional region crop as fractions of the full frame (start, end) per axis; (0,1)=full.
    # lets a run train on only a sub-region (e.g. right 60% x, top 75% y) to save compute.
    crop_x_frac: tuple = (0.0, 1.0)
    crop_y_frac: tuple = (0.0, 1.0)

@dataclass
class DataloaderConfig:
    batch_size: int = 96
    num_workers: int = 2
    data_aug: bool = True
    channel_mixing_prob: float = 0.0
    rotation_prob: float = 0.25
    flip_prob: float = 0.25
    noise_prob: float = 0.30
    brightness_prob: float = 0.50
    contrast_prob: float = 0.50

@dataclass
class TrainingConfig:
    n_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0
    l1_lambda: float = 7e-6
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = './runs_campaign4'
    eval_int: int = 10
    test_int: int = 30
    probe_int: int = 5
    eval_aggregate: bool = True  # show one aggregated (depth-averaged) eval figure in addition to per-depth
    # probe ROIs: fixed readability probe regions rendered every probe_int epochs.
    # disabled for now (regions are pinned to the old scroll2/scroll4 coords and will be
    # re-introduced once the new scroll4/scroll3 probe locations are known). --probe-rois
    # re-enables. when off, no probe specs are built and no probe figures render.
    probe_rois_enabled: bool = False
    focal_gamma: float = 0.0   # >0 activates focal loss: down-weights easy negatives, pushes gradient toward hard tiles
    ranking_lambda: float = 0.0  # weight of pairwise ranking loss term
    ranking_margin: float = 0.3  # margin: pos_score must exceed neg_score by at least this
    ranking_neg_frac: float = 1.0  # partial-AUC: fraction of hardest (top-scoring) negatives to rank against; 1.0=all pairs (full AUC), <1.0 focuses gradient on the low-FPR region
    pretrain_epochs: int = 0     # epochs of band-identity pretraining before BCE fine-tuning
    eval_cooldown_secs: int = 0   # sleep this many seconds after probe/eval epochs to let hardware cool

@dataclass
class FinetuneConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 25

@dataclass
class ModelConfig:
    # conv1_drop: float = 0
    # conv2_drop: float = 0
    # fc1_drop: float = 0
    # fc2_drop: float = 0
    conv1_drop: float = 0.0
    conv2_drop: float = 0.05
    fc1_drop: float = 0.2
    fc2_drop: float = 0.1
    pooling: str = "avg"
    gem_p: float = 3.0
    conv3_dilation: int = 1
    arch: str = "v1"

@dataclass
class HardMining:
    enabled: bool = True
    hn_cutoff: float = 0.8
    hp_cutoff: float = 0.45
    hm_frac: float = 0.1
    dir: str = './hard_negs'

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    dl: DataloaderConfig = field(default_factory=DataloaderConfig)
    tra: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    hm: HardMining = field(default_factory=HardMining)
    
    # Derived properties
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models"
    exp_name: Optional[str] = None
