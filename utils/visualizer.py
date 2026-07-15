from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing import _32Bit
import os
from typing import Any, Literal
from collections import defaultdict
import json
import re

# raise OpenCV decode pixel cap before cv2 import (native 2.4 masks/labels ~1.3 Gpx)
os.environ.setdefault("CV_IO_MAX_IMAGE_PIXELS", str(2**34))
import cv2
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import scipy.ndimage as ndimage
from scipy import stats as scipy_stats

from .config import Config
from .dataloader import DataManager, imread_gray
from .training_utils import calculate_metrics

# Pillow>=10 removed Image.ANTIALIAS, but some torch/tensorboard paths still
# reference it. Keep a compatibility alias so add_figure/add_image does not fail.
if not hasattr(Image, "ANTIALIAS") and hasattr(Image, "Resampling"):
    setattr(Image, "ANTIALIAS", Image.Resampling.LANCZOS)

# NaN tiles (outside mask) render as mid-gray instead of black-zero so the
# train/valid split line is not confused with actual low-confidence predictions
import copy as _copy
_inferno_nan = _copy.copy(plt.cm.inferno)
_inferno_nan.set_bad(color=(0.45, 0.45, 0.45, 1.0))
# register_cmap was removed in matplotlib 3.9; use colormaps.register instead
try:
    import matplotlib as _mpl
    _mpl.colormaps.register(_inferno_nan, name='inferno_nan', force=True)
except Exception:
    plt.cm.inferno_nan = _inferno_nan  # fallback: attach directly

def group_by_depth(coords):
    """group tile coordinates by their depth offset"""
    grouped = defaultdict(list)
    for d_off, y_off, x_off in coords:
        grouped[d_off].append((d_off, y_off, x_off))
    return grouped

def predict_tiles(config, model, vol, mask, coords, y_range, x_range, depth_start, volume_name, g_mean, g_std, g_min, g_max):
    """run batched prediction over given coords returning downsampled map.

    reads tiles sequentially (no ThreadPoolExecutor) to avoid PCIe bus saturation
    that causes hard system crashes on Blackwell GPUs under concurrent zarr+inference load.
    """
    tile  = config.data.tile_size
    depth = config.data.depth
    H = y_range[1] - y_range[0]
    W = x_range[1] - x_range[0]
    h_small = H // tile
    w_small = W // tile
    pmap = np.full((h_small, w_small), np.nan, dtype=np.float32)

    infer_bs = min(max(config.dl.batch_size * 2, 256), 256)
    # scale inference batch size by tile area relative to the baseline T=32.
    # at T=106/D=16, B=256 requires ~10GB for input alone and OOMs during eval.
    # formula: 256 * (32/T)^2 * (8/D), clamped to [1, 256].
    tile_scale = (32.0 / tile) ** 2 * (8.0 / max(depth, 1))
    infer_bs = max(1, min(infer_bs, int(256 * tile_scale)))
    device = config.device if torch.cuda.is_available() else "cpu"

    tile_list = [
        (depth_start, y_range[0] + y_off, x_range[0] + x_off, y_off, x_off)
        for _, y_off, x_off in coords
    ]

    def _read_one(args):
        d, y, x, y_off, x_off = args
        mode = getattr(config.data, "input_mode", "single")

        def _fetch(z_start, n_depth):
            if z_start + n_depth > vol.shape[0]:
                return None
            blk = np.array(vol[z_start:z_start + n_depth, y:y + tile, x:x + tile]).astype(np.float32)
            blk = (blk - g_mean) / g_std
            if mask.ndim == 2:
                m_bin = (mask[y:y + tile, x:x + tile] > 0).astype(np.uint8)
                blk[np.broadcast_to(np.expand_dims(m_bin, 0), blk.shape) == 0] = 0
            return np.clip((blk - g_min) / (g_max - g_min + 1e-12), 0, 1)

        if mode == "diff":
            pre_z = getattr(config.data, "pre_band_start", 20)
            ink = _fetch(d, depth)
            pre = _fetch(pre_z, depth)
            if ink is None or pre is None:
                return None, y_off, x_off
            blk = np.clip(ink - pre, 0, None)
            expected = (depth, tile, tile)
        elif mode == "triple":
            pre_z  = getattr(config.data, "pre_band_start", 20)
            post_z = getattr(config.data, "post_band_start", 40)
            pre  = _fetch(pre_z,  depth)
            ink  = _fetch(d,      depth)
            post = _fetch(post_z, depth)
            if any(b is None for b in (pre, ink, post)):
                return None, y_off, x_off
            blk = np.concatenate([pre, ink, post], axis=0)
            expected = (depth * 3, tile, tile)
        elif mode == "double":
            pre_z = getattr(config.data, "pre_band_start", 20)
            ink = _fetch(d,     depth)
            pre = _fetch(pre_z, depth)
            if ink is None or pre is None:
                return None, y_off, x_off
            blk = np.concatenate([ink, pre], axis=0)
            expected = (depth * 2, tile, tile)
        elif mode == "fulldepth":
            full_d = int(vol.shape[0])
            if full_d > vol.shape[0]:
                return None, y_off, x_off
            blk_full = np.array(vol[0:full_d, y:y+tile, x:x+tile]).astype(np.float32)
            blk_full = (blk_full - g_mean) / g_std
            if mask.ndim == 2:
                m_bin = (mask[y:y+tile, x:x+tile] > 0).astype(np.uint8)
                blk_full[np.broadcast_to(np.expand_dims(m_bin, 0), blk_full.shape) == 0] = 0
            blk = np.clip((blk_full - g_min) / (g_max - g_min + 1e-12), 0, 1)
            expected = (full_d, tile, tile)
        else:
            if d + depth > vol.shape[0]:
                return None, y_off, x_off
            blk = _fetch(d, depth)
            if blk is None:
                return None, y_off, x_off
            expected = (depth, tile, tile)

        if blk.shape != expected:
            return None, y_off, x_off
        return blk, y_off, x_off

    # read tiles sequentially — avoids concurrent zarr+GPU load that triggers WHEA crashes
    print(f"[predict] reading {len(tile_list)} tiles sequentially...")
    results = [_read_one(t) for t in tqdm(tile_list, desc=f"Read {volume_name}", leave=False)]

    valid = [(blk, y_off, x_off) for blk, y_off, x_off in results if blk is not None]

    with torch.no_grad():
        for i in tqdm(range(0, len(valid), infer_bs), desc=f"Predict {volume_name}", leave=True):
            chunk   = valid[i:i + infer_bs]
            b_blocks = [b for b, _, _ in chunk]
            b_idx    = [(yo, xo) for _, yo, xo in chunk]

            bt     = torch.from_numpy(np.stack(b_blocks)).float().unsqueeze(1).to(device)
            logits = model(bt)
            if logits.dim() == 4:
                logits = logits.flatten(1).max(dim=1, keepdim=True).values
            preds  = torch.sigmoid(logits).cpu().numpy().flatten()

            for (y_off, x_off), pred in zip(b_idx, preds):
                yi = y_off // tile
                xi = x_off // tile
                if 0 <= yi < h_small and 0 <= xi < w_small:
                    pmap[yi, xi] = float(pred)
            del bt

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # optional post-hoc spatial smoothing — only applied to valid (non-NaN) tiles
    sigma = float(getattr(config.data, "smooth_sigma", 0.0))
    if sigma > 0:
        valid_mask = np.isfinite(pmap)
        filled = np.where(valid_mask, pmap, 0.0)
        filled = ndimage.gaussian_filter(filled, sigma=sigma)
        weight = ndimage.gaussian_filter(valid_mask.astype(np.float32), sigma=sigma)
        # normalize by the contribution of valid neighbors; leave NaN positions as NaN
        with np.errstate(invalid='ignore'):
            smoothed = np.where(weight > 0, filled / weight, np.nan)
        pmap = np.clip(smoothed, 0.0, 1.0)

    return pmap

class _ScopedWriter:
    """thin proxy over a shared SummaryWriter that namespaces each tag with a per-
    scroll scope (e.g. 's<scroll_id>'), so multiple per-scroll visualizers can write
    into ONE tensorboard run without their identical tags colliding.

    the scope is inserted AFTER the top-level category rather than at the root, so
    tensorboard groups by category first and splits scrolls within each:
        Test/s<sid>/...   Evaluation/s<sid>/...   R_M/s<sid>/...
    (not s<sid>/Test/... which would bury every category under the scroll id).

    this consolidates the multi-scroll layout from (1 metrics + N per-scroll)
    folders down to a single folder. tags listed in global_prefixes are written
    WITHOUT the scope — used for the probe ROIs, a fixed global set rendered once."""
    def __init__(self, writer, prefix: str = "", global_prefixes: tuple = ()):
        self._w = writer
        # accept either 's<sid>/' (legacy prefix form) or 's<sid>'; store bare scope
        self._scope = (prefix or "").strip("/")
        self._global_prefixes = tuple(global_prefixes)

    def _tag(self, tag):
        tag = str(tag)
        # leave globally-scoped tags (e.g. probe ROIs) unscoped
        if not self._scope or any(tag.startswith(g) for g in self._global_prefixes):
            return tag
        # insert scope after the first path segment so the category stays on top
        head, sep, rest = tag.partition("/")
        return f"{head}/{self._scope}/{rest}" if sep else f"{head}/{self._scope}"

    def add_scalar(self, tag, *a, **k):
        return self._w.add_scalar(self._tag(tag), *a, **k)

    def add_figure(self, tag, *a, **k):
        return self._w.add_figure(self._tag(tag), *a, **k)

    def add_images(self, tag, *a, **k):
        return self._w.add_images(self._tag(tag), *a, **k)

    def add_histogram(self, tag, *a, **k):
        return self._w.add_histogram(self._tag(tag), *a, **k)

    def add_custom_scalars(self, *a, **k):
        # layout is applied once by the owning (main) writer; ignore here
        return None

    def add_graph(self, *a, **k):
        return self._w.add_graph(*a, **k)

    def flush(self):
        return self._w.flush()

    def close(self):
        # the shared writer is owned by the main visualizer; do not close it here
        return None

    def __getattr__(self, name):
        # delegate any other attribute/method to the wrapped writer
        return getattr(self._w, name)


class TensorboardVisualizer:
    def __init__(self, config: Config, mode: str = 'train', scroll_id=None, log_suffix: str = None,
                 shared_writer=None, tag_prefix: str = ""):
        """initialize tensorboard visualizer and precompute datasets and stats.

        scroll_id: which scroll fragment this visualizer renders figures for;
            defaults to config.data.tra_scroll_id. lets the trainer build one
            figure-visualizer per fragment.
        mode: 'train' loads figure assets and can render figures; 'metrics' only
            logs scalar metrics (used for the merged multi-scroll training stream);
            'finetune' behaves like the legacy finetune path.
        log_suffix: appended to the run directory name so per-scroll visualizers
            write to separate tensorboard runs.
        """
        self.c = config
        self.mode = mode
        self.scroll1_id = int(scroll_id) if scroll_id is not None else int(config.data.tra_scroll_id)
        # whether this scroll renders evaluation figures. vis_scroll_ids=None => all
        # scrolls render (default); otherwise only listed scrolls do. test/probe
        # figures are unaffected.
        _vis_ids = getattr(config.data, "vis_scroll_ids", None)
        self.eval_enabled = (not _vis_ids) or (int(self.scroll1_id) in [int(v) for v in _vis_ids])
        self.probe_log_interval = max(1, int(getattr(config.tra, "probe_int", 5)))
        # probe ROIs can be toggled off (default off); when off no specs are built and
        # no probe figures render
        self.probe_rois_enabled = bool(getattr(config.tra, "probe_rois_enabled", False))

        if config.exp_name is None:
            if self.mode == 'finetune':
                experiment_name = f"finetune_{datetime.now().strftime('%d.%m_%H-%M-%S')}"
            else:
                experiment_name = f"ink_detection_{datetime.now().strftime('%d.%m_%H-%M-%S')}"
        else:
            experiment_name = config.exp_name + "_" + datetime.now().strftime('%d_%H-%M-%S')

        if log_suffix:
            experiment_name = f"{experiment_name}_{log_suffix}"

        self.log_path = os.path.join(config.tra.log_dir, experiment_name)


        # layout for dashboards unchanged to keep metric names
        self.layout = {
            "Training_Overview": {
                "loss": ["Multiline", ["G_M/Loss/Train", "G_M/Loss/Train_Raw", "G_M/Loss/Valid"]],
                "accuracy": ["Multiline", ["G_M/Acc/Train", "G_M/Acc/Valid"]],
            },
            "P_M_Metrics": {
                "precision_recall": [
                    "Multiline", [
                        "P_M/Precision/Train", "P_M/Precision/Valid",
                        "P_M/Recall/Train", "P_M/Recall/Valid"
                    ]
                ],
                "f1_specificity": [
                    "Multiline", [
                        "P_M/F1_Score/Train", "P_M/F1_Score/Valid",
                        "P_M/Specificity/Train", "P_M/Specificity/Valid"
                    ]
                ],
            },
            "AUC_Metrics": {
                "roc_auc": ["Multiline", ["AUC/ROC_AUC/Train", "AUC/ROC_AUC/Valid"]],
                "pr_auc": ["Multiline", ["AUC/PR_AUC/Train", "AUC/PR_AUC/Valid"]],
            },
            "Readability": {
                "contrast_ranking": [
                    "Multiline", [
                        "R_M/LocalContrast",
                        "R_M/LocalRanking",
                        "R_M/TopKPrecision",
                        "R_M/InkFractionSpearman"
                    ]
                ],
                "low_fpr_spill": [
                    "Multiline", [
                        "R_M/RecallAt1PctFPR",
                        "R_M/PartialAUCAt1PctFPR",
                        "R_M/SpillRatio",
                        "R_M/ReadabilityComposite"
                    ]
                ],
            },
        }

        # training mode preloads training and evaluation assets
        if self.mode == 'train':
            self._init_training_assets()

        if shared_writer is not None:
            # consolidated multi-scroll layout: write into the shared run folder,
            # namespacing this scroll's tags via tag_prefix. probe ROIs stay global.
            self.writer = _ScopedWriter(shared_writer, tag_prefix,
                                        global_prefixes=("ProbeROIs", "R_M/Probe"))
            self.log_path = getattr(shared_writer, "log_dir", self.log_path)
        else:
            self.writer = SummaryWriter(self.log_path)
            self.writer.add_custom_scalars(self.layout)

        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.tra.log_dir}")

    def _init_training_assets(self):
        """load training and auxiliary datasets and normalization stats"""
        # data manager holds main training volume mask labels and splits
        dm = DataManager(self.c, scroll_id=self.scroll1_id)
        self.dm = dm

        self.volume = dm.vol
        self.mask = dm.mask
        self.labels = dm.labels
        # eval figure always uses the FULL papyrus mask so the prediction map covers
        # the entire cropped scroll region — this is intentional and distinct from
        # the training/validation loop which uses the ring mask when ring_negatives=True.
        # the figure is a visual diagnostic of where the model fires across the scroll,
        # not a metric computation; restricting it to the ring would defeat the purpose.
        self.eval_mask = self.mask
        # use the dataloader's coordinate system directly — the old hardcoded crop
        # (y0=200, x0=1000) was NOT tile-aligned (200%32=8, 1000%32=8), so the eval
        # figure was evaluating tiles at (200+k*32, 1000+l*32) while training used
        # (k*32, l*32). these grids never overlap: the eval was always evaluating
        # untrained background tiles, never the actual ring/ink training tiles.
        self.train_x_range = dm.train_x
        self.valid_x_range = dm.valid_x
        self.y_range = dm.y_range
        # axis-aware split (see DataManager): 'y' = horizontal (top train / bottom valid),
        # 'x' = legacy vertical. eval figure composition below branches on this.
        self.split_axis = getattr(dm, "split_axis", "x")
        self.train_range = getattr(dm, "train_range", dm.train_x)
        self.valid_range = getattr(dm, "valid_range", dm.valid_x)
        self.shared_range = getattr(dm, "shared_range", dm.y_range)
        self.global_mean, self.global_std, self.global_min, self.global_max = dm.norm_stats

        # load test data region and scroll4 data with stats
        self.test_volume, self.test_mask, self.test_y_range, self.test_x_range = self._load_test_region()
        self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max = self._get_or_compute_norm(
            self.test_volume, self.test_mask, str(self.scroll1_id)
        )

        # scroll4 transfer region: loaded DEFENSIVELY (its zarr/mask may be absent on a minimal
        # setup that only has the training scroll). on failure -> None, and its test figure is
        # skipped. the test figure never runs anyway when test_int > epochs.
        self.scroll4_volume = self.scroll4_mask = None
        self.scroll4_y_range = self.scroll4_x_range = None
        self.scroll4_global_mean = self.scroll4_global_std = None
        self.scroll4_global_min = self.scroll4_global_max = None
        try:
            (self.scroll4_volume, self.scroll4_mask,
             self.scroll4_y_range, self.scroll4_x_range) = self._load_scroll4_region()
            (self.scroll4_global_mean, self.scroll4_global_std,
             self.scroll4_global_min, self.scroll4_global_max) = self._get_or_compute_norm(
                self.scroll4_volume, self.scroll4_mask, str(self.c.data.scroll4_id))
        except Exception as e:
            print(f"[scroll4] not available, skipping its test figure ({e})")
            self.scroll4_volume = None

        # scroll2 goal-scroll: also DEFENSIVE. probe ROIs would include it, but probes are gated
        # on probe_rois_enabled; if scroll2 is missing we skip its figure/probes gracefully.
        self.scroll2_volume = self.scroll2_mask = None
        self.scroll2_y_range = self.scroll2_x_range = None
        self.scroll2_global_mean = self.scroll2_global_std = None
        self.scroll2_global_min = self.scroll2_global_max = None
        try:
            (self.scroll2_volume, self.scroll2_mask,
             self.scroll2_y_range, self.scroll2_x_range) = self._load_scroll2_region()
            (self.scroll2_global_mean, self.scroll2_global_std,
             self.scroll2_global_min, self.scroll2_global_max) = self._get_or_compute_norm(
                self.scroll2_volume, self.scroll2_mask, str(self.c.data.scroll2_id))
        except Exception as e:
            print(f"[scroll2] not available, skipping its test figure ({e})")
            self.scroll2_volume = None

        # scroll3 goal-scroll: loaded DEFENSIVELY (its zarr/mask may not exist yet, e.g. while
        # it is still downloading). on any failure we set it to None and simply skip its test
        # figure — training and the scroll2 figure are unaffected.
        self.scroll3_volume = self.scroll3_mask = None
        self.scroll3_y_range = self.scroll3_x_range = None
        self.scroll3_global_mean = self.scroll3_global_std = None
        self.scroll3_global_min = self.scroll3_global_max = None
        try:
            (self.scroll3_volume, self.scroll3_mask,
             self.scroll3_y_range, self.scroll3_x_range) = self._load_scroll3_region()
            (self.scroll3_global_mean, self.scroll3_global_std,
             self.scroll3_global_min, self.scroll3_global_max) = self._get_or_compute_norm(
                self.scroll3_volume, self.scroll3_mask, str(self.c.data.scroll3_id))
        except Exception as e:
            print(f"[scroll3] not available, skipping its test figure ({e})")
            self.scroll3_volume = None

        self._segment_assets = {}
        self.probe_specs = self._build_probe_specs() if self.probe_rois_enabled else []
        self._debug_scroll4_ranges_once()

        # dense probe specs: named ROIs rendered every probe_int epochs.
        # built once at init; add_dense_probe_figure iterates them all.
        self._dense_probe_specs = []
        if getattr(self.c.data, "dense_labels", False):
            self._dense_probe_specs = self._build_dense_probe_specs()

        # legacy single auto-probe (kept for backward compat with non-dense runs)
        self._probe_py0 = self._probe_px0 = self._probe_size = None

    def _build_dense_probe_specs(self):
        """build list of named probe ROI specs for dense runs.

        each entry:
          tag   : TensorBoard tag suffix and log filename prefix
          vol   : zarr/array to read from
          norm  : (mean, std, g_min, g_max) normalisation stats
          labels: pixel label array (None for transfer scrolls with no GT)
          mask  : binary papyrus mask
          y0,x0 : top-left corner of the probe window (tile AND 8-pixel aligned)
          size  : patch side length in pixels (divisible by 8 for U-Net)
        """
        T    = self.c.data.tile_size
        # snap to tile_size — multiples of T are always multiples of 8 when T=32,
        # satisfying both the training tile grid alignment AND the U-Net div-by-8 requirement.
        snap = lambda v: (v // T) * T
        # probe size: 512 is divisible by both 32 and 8 — no adjustment needed.
        ps = 512
        n  = (self.global_mean, self.global_std, self.global_min, self.global_max)
        specs = []

        # full eroded labels for the training scroll (pixel array, already loaded as float)
        # dm.labels is cast to uint8 after get_datasets(), so reload from file
        def _load_eroded(sid):
            img = imread_gray(f"./eroded_inklabels/{sid}.png")
            return (img / 255.0).astype(np.float32) if img is not None else None

        # named probes on the training scroll (scroll4 w023)
        SCROLL4_W023 = 20240304161941
        if int(self.scroll1_id) == SCROLL4_W023:
            train_labels = _load_eroded(SCROLL4_W023)   # full-res float labels
            for tag, y, x in [("Easy", 5085, 25596), ("Medium", 4943, 22575), ("Hard", 8966, 21726)]:
                specs.append({"tag": tag, "vol": self.volume, "norm": n,
                               "labels": train_labels, "mask": self.mask,
                               "y0": snap(y), "x0": snap(x), "size": ps})
        else:
            # fall back to auto-detected densest window for other training scrolls
            try:
                py0, px0, _ = self._find_dense_probe_region(probe_px=ps)
                train_labels = _load_eroded(int(self.scroll1_id))
                specs.append({"tag": "Auto", "vol": self.volume, "norm": n,
                               "labels": train_labels, "mask": self.mask,
                               "y0": py0, "x0": px0, "size": ps})
            except Exception as e:
                print(f"[dense probe] auto-detect failed: {e}")

        # scroll4_pi probe — uses the small scroll4 test segment (20231210132040),
        # which HAS eroded inklabels. loaded defensively.
        if self.scroll4_volume is not None:
            sn = (self.scroll4_global_mean, self.scroll4_global_std,
                  self.scroll4_global_min, self.scroll4_global_max)
            scroll4_labels = _load_eroded(int(self.c.data.scroll4_id))
            _d4, H4, W4 = (int(v) for v in self.scroll4_volume.shape)
            # pi region: y=7968, x=1952 (already tile-aligned per legacy specs)
            s4y, s4x = snap(7968), snap(1952)
            # clamp in case this test scroll is smaller than expected
            s4y = min(s4y, max(0, H4 - ps))
            s4x = min(s4x, max(0, W4 - ps))
            specs.append({"tag": "Scroll4_Pi", "vol": self.scroll4_volume,
                          "norm": sn, "labels": scroll4_labels, "mask": self.scroll4_mask,
                          "y0": s4y, "x0": s4x, "size": ps})

        # scroll2 and scroll3: no inklabels — labels=None, no GT panel
        if self.scroll2_volume is not None:
            sn = (self.scroll2_global_mean, self.scroll2_global_std,
                  self.scroll2_global_min, self.scroll2_global_max)
            _d2, H2, W2 = (int(v) for v in self.scroll2_volume.shape)
            cy = snap(max(0, (H2 - ps) // 2))
            cx = snap(max(0, (W2 - ps) // 2))
            specs.append({"tag": "Scroll2_Center", "vol": self.scroll2_volume,
                          "norm": sn, "labels": None, "mask": self.scroll2_mask,
                          "y0": cy, "x0": cx, "size": ps})

        if self.scroll3_volume is not None:
            sn = (self.scroll3_global_mean, self.scroll3_global_std,
                  self.scroll3_global_min, self.scroll3_global_max)
            _d3, H3, W3 = (int(v) for v in self.scroll3_volume.shape)
            cy = snap(max(0, (H3 - ps) // 2))
            cx = snap(max(0, (W3 - ps) // 2))
            specs.append({"tag": "Scroll3_Center", "vol": self.scroll3_volume,
                          "norm": sn, "labels": None, "mask": self.scroll3_mask,
                          "y0": cy, "x0": cx, "size": ps})

        tags = [s['tag'] for s in specs]
        print(f"[dense probe] {len(specs)} probe specs: {tags}  (size={ps}px, snap_unit={T}px)")
        return specs

    def _find_dense_probe_region(self, probe_px=512):
        """scan the training region for the densest probe_px x probe_px ink window.
        returns absolute pixel (py0, px0, probe_px) within the training area."""
        T = self.c.data.tile_size
        lab = np.asarray(self.labels)

        # training-region bounds in pixel space
        if self.split_axis == "y":
            (tr_lo, tr_hi) = self.train_range
            (sx0, sx1)     = self.shared_range
            abs_y0, abs_y1 = tr_lo, tr_hi
            abs_x0, abs_x1 = sx0, sx1
        else:
            (sy0, sy1)     = self.shared_range
            (tr_lo, tr_hi) = self.train_range
            abs_y0, abs_y1 = sy0, sy1
            abs_x0, abs_x1 = tr_lo, tr_hi

        # build tile-level ink count map (vectorised reshape; cheap even for large labels)
        ht = lab.shape[0] // T
        wt = lab.shape[1] // T
        lab_bin = (lab[:ht*T, :wt*T] > 0.5).astype(np.float32)
        ink_tile = lab_bin.reshape(ht, T, wt, T).sum(axis=(1, 3))  # (ht, wt)

        # tile-index bounds of the training region
        ty0, ty1 = abs_y0 // T, min(abs_y1 // T, ht)
        tx0, tx1 = abs_x0 // T, min(abs_x1 // T, wt)
        probe_tiles = max(1, probe_px // T)

        H_c, W_c = ty1 - ty0, tx1 - tx0
        if H_c < probe_tiles or W_c < probe_tiles:
            # training region smaller than the probe window -- use its top-left corner
            print(f"[dense probe] region ({H_c}x{W_c} tiles) smaller than probe ({probe_tiles}x{probe_tiles}); using top-left")
            return abs_y0, abs_x0, probe_px

        crop = ink_tile[ty0:ty1, tx0:tx1]
        cs = np.cumsum(np.cumsum(crop, axis=0), axis=1)
        cs_pad = np.pad(cs, ((1, 0), (1, 0)))

        # vectorised sliding-window sum over the crop
        r_max = H_c - probe_tiles + 1
        c_max = W_c - probe_tiles + 1
        win = (cs_pad[probe_tiles:probe_tiles+r_max, probe_tiles:probe_tiles+c_max]
             - cs_pad[0:r_max,                      probe_tiles:probe_tiles+c_max]
             - cs_pad[probe_tiles:probe_tiles+r_max, 0:c_max]
             + cs_pad[0:r_max,                      0:c_max])
        best_r, best_c = np.unravel_index(win.argmax(), win.shape)
        best_score = float(win[best_r, best_c])

        py0 = (ty0 + int(best_r)) * T
        px0 = (tx0 + int(best_c)) * T
        max_ink = probe_tiles * probe_tiles * T * T  # total px in window
        print(f"[dense probe] best window y[{py0},{py0+probe_px}] x[{px0},{px0+probe_px}] "
              f"ink_px={int(best_score)}/{max_ink} ({100*best_score/max_ink:.1f}%)")
        return py0, px0, probe_px

    def add_dense_probe_figure(self, epoch, model):
        """combined probe ROI figure across all named probes and all depth windows.

        layout (matches historical ProbeROIs style, adapted for dense inference):
          rows  : one per depth window (no half-step: z_step=depth), + final composite row
          cols  : two per probe — (pred, pred+inklabel overlay)
                  scroll2/scroll3 (no GT) still get two columns; the overlay is pred only.
          probes: Easy | Medium | Hard | Scroll4_Pi | Scroll2_Center | Scroll3_Center

        each cell is a single forward pass on the probe's 512x512 region.
        no raw scan, no standalone GT column.
        """
        if not self._dense_probe_specs:
            return

        D   = self.c.data.depth
        zf0 = int(self.c.data.d_start)
        zf1 = int(self.c.data.d_end)
        dev = self.c.device
        model.eval()

        out_dir = os.path.join(os.path.dirname(self.log_path), "dense_figs")
        os.makedirs(out_dir, exist_ok=True)

        # depth windows: NO half-stepping — step = D so windows don't overlap
        z_step   = max(1, D)
        z_starts = list(range(zf0, max(zf0 + 1, zf1 - D + 1), z_step))
        if not z_starts:
            z_starts = [zf0]
        # add a final non-overlapping window if the last one doesn't reach zf1-D
        if z_starts[-1] + D < zf1:
            z_starts.append(zf1 - D)
        n_depths = len(z_starts)
        n_rows   = n_depths + 1      # depth rows + composite row

        specs   = self._dense_probe_specs
        n_probes = len(specs)
        n_cols   = n_probes * 2      # pred + overlay per probe

        # gold overlay RGBA (reused for every probe that has labels)
        _GOLD = np.array([0.98, 0.85, 0.37, 0.50], dtype=np.float32)

        # ── per-probe inference ─────────────────────────────────────────────
        # probe_preds[pi] = list of (oh, ow, pred_map) per depth, + composite
        probe_data = []   # list of dicts per probe
        for spec in specs:
            tag      = spec.get("tag", "probe")
            vol      = spec["vol"]
            g_mean, g_std, g_min, g_max = spec["norm"]
            lab_src  = spec.get("labels")
            mask_src = spec.get("mask")
            py0 = int(spec["y0"])
            px0 = int(spec["x0"])
            ps  = int(spec["size"])

            vol_d = int(vol.shape[0])
            py1   = min(py0 + ps, int(vol.shape[1]))
            px1   = min(px0 + ps, int(vol.shape[2]))
            oh, ow = py1 - py0, px1 - px0
            if oh < 8 or ow < 8:
                probe_data.append(None)
                continue

            ph = (-oh) % 8; pw = (-ow) % 8  # U-Net pad-to-8

            mk = ((np.asarray(mask_src[py0:py1, px0:px1]) > 0.5).astype(np.float32)
                  if mask_src is not None else np.ones((oh, ow), np.float32))
            gt = ((np.asarray(lab_src[py0:py1, px0:px1]) > 0.5).astype(np.float32)
                  if lab_src is not None else None)

            depth_preds = []
            for z0 in z_starts:
                z0c = max(0, min(z0, vol_d - D))
                blk = np.asarray(vol[z0c:z0c+D, py0:py1, px0:px1]).astype(np.float32)
                if blk.shape[0] != D:
                    depth_preds.append(np.zeros((oh, ow), np.float32))
                    continue
                blk_n = np.clip(((blk - g_mean) / g_std - g_min) / (g_max - g_min + 1e-12), 0, 1)
                if ph or pw:
                    blk_n = np.pad(blk_n, ((0, 0), (0, ph), (0, pw)), mode="reflect")
                bt = torch.from_numpy(blk_n).unsqueeze(0).unsqueeze(0).float().to(dev)
                try:
                    with torch.no_grad(), autocast(dev):
                        p = torch.sigmoid(model(bt))[0, 0, :oh, :ow].float().cpu().numpy()
                except Exception:
                    p = np.zeros((oh, ow), np.float32)
                depth_preds.append(p * mk)

            composite = np.max(np.stack(depth_preds, axis=0), axis=0)
            probe_data.append({"tag": tag, "oh": oh, "ow": ow,
                               "mk": mk, "gt": gt,
                               "depth_preds": depth_preds,
                               "composite": composite,
                               "z_starts": z_starts})

        # ── figure layout ───────────────────────────────────────────────────
        cell_h = 2.4   # inches per row
        cell_w = 2.4   # inches per column
        fig_h  = cell_h * n_rows + 0.6
        fig_w  = cell_w * n_cols + 0.3
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)

        col_headers = []
        for sp in specs:
            t = sp.get("tag", "?")
            col_headers += [t, f"{t} overlay"]

        for ci, hdr in enumerate(col_headers):
            axes[0, ci].set_title(hdr, fontsize=6, pad=2)

        for pi, (spec, pd) in enumerate(zip(specs, probe_data)):
            c_pred    = pi * 2
            c_overlay = pi * 2 + 1

            for ri in range(n_rows):
                ax_p = axes[ri, c_pred]
                ax_o = axes[ri, c_overlay]
                ax_p.axis("off"); ax_o.axis("off")

                if pd is None:
                    continue

                if ri < n_depths:
                    pred = pd["depth_preds"][ri]
                    z0   = z_starts[ri]
                    row_label = f"z{z0}-{z0+D}"
                else:
                    pred = pd["composite"]
                    row_label = "composite"

                ax_p.imshow(pred, cmap="magma", vmin=0, vmax=1)
                ax_p.set_title(row_label, fontsize=5, pad=1)

                # overlay column: pred + gold inklabel if GT exists
                ax_o.imshow(pred, cmap="magma", vmin=0, vmax=1)
                if pd["gt"] is not None:
                    gt_rgba        = np.zeros((*pred.shape, 4), dtype=np.float32)
                    gt_rgba[..., :] = _GOLD
                    gt_rgba[..., 3] = pd["gt"] * _GOLD[3]
                    ax_o.imshow(gt_rgba)

        plt.suptitle(f"Dense probe ROIs — ep{epoch+1}  z_step={D}  {n_depths} depths + composite",
                     fontsize=8, y=1.002)
        plt.tight_layout(pad=0.3)
        self.writer.add_figure("Dense/ProbeROIs_Combined", fig, epoch)
        try:
            out_png = os.path.join(out_dir, f"probe_combined_ep{epoch+1:02d}.png")
            fig.savefig(out_png, dpi=100, bbox_inches="tight")
            # shrink saved file to half resolution; inference canvas stays full-res
            import cv2 as _cv2
            _img = _cv2.imread(out_png)
            if _img is not None:
                _h, _w = _img.shape[:2]
                _cv2.imwrite(out_png, _cv2.resize(_img, (_w//2, _h//2), interpolation=_cv2.INTER_AREA))
            print(f"[dense probe] combined ep{epoch+1} -> {out_png}")
        except Exception as e:
            print(f"[dense probe] save failed: {e}")
        plt.close(fig)

        # READABILITY METRICS from training-scroll labeled probes only (Easy, Medium, Hard).
        # Scroll4_Pi is a test segment, Scroll2/Scroll3 have no GT — all excluded.
        try:
            T = self.c.data.tile_size
            _TRAIN_TAGS = {"Easy", "Medium", "Hard"}
            agg_rm = []
            for spec, pd in zip(specs, probe_data):
                if pd is None or pd.get("gt") is None:
                    continue   # no GT (scroll2, scroll3)
                if spec.get("tag") not in _TRAIN_TAGS:
                    continue   # scroll4_pi is test segment, not training corpus
                py0   = int(spec["y0"]); px0 = int(spec["x0"]); ps = int(spec["size"])
                tag   = spec.get("tag", "probe")
                comp  = pd["composite"]  # (oh, ow) pixel-space composite prediction
                oh, ow = comp.shape
                mask_src = spec.get("mask")
                lab_src  = pd["gt"]      # already loaded as float32 binary

                # build tile-level maps for this probe's spatial extent
                # use full-res label/mask arrays cropped to the probe window
                mk_full = ((np.asarray(mask_src[py0:py0+oh, px0:px0+ow]) > 0.5)
                           if mask_src is not None else np.ones((oh, ow), np.float32))
                lb_tile = np.zeros((oh // T, ow // T), dtype=bool)
                lf_tile = np.zeros((oh // T, ow // T), dtype=np.float32)
                vt_tile = np.zeros((oh // T, ow // T), dtype=bool)
                for ty in range(oh // T):
                    for tx in range(ow // T):
                        l_patch = lab_src[ty*T:(ty+1)*T, tx*T:(tx+1)*T]
                        m_patch = mk_full[ty*T:(ty+1)*T, tx*T:(tx+1)*T]
                        if m_patch.sum() <= 0:
                            continue
                        vt_tile[ty, tx] = True
                        lb_tile[ty, tx] = bool(np.any(l_patch > 0.5))
                        lf_tile[ty, tx] = float(np.mean(l_patch > 0.5))

                # center-sample composite into tile resolution
                tile_ys = np.arange(oh // T) * T + T // 2
                tile_xs = np.arange(ow // T) * T + T // 2
                tile_ys = np.clip(tile_ys, 0, oh - 1)
                tile_xs = np.clip(tile_xs, 0, ow - 1)
                tp = comp[np.ix_(tile_ys, tile_xs)]

                rm = self._compute_readability_metrics(tp, lb_tile, lf_tile, vt_tile)
                agg_rm.append(rm)

                # per-probe contrast scalar for quick comparison (training probes only)
                lc = rm.get("local_contrast", float("nan"))
                rc = rm.get("readability_composite", float("nan"))
                if np.isfinite(lc):
                    self.writer.add_scalar(f"R_M/Probe/{tag}/LocalContrast", lc, epoch)
                if np.isfinite(rc):
                    self.writer.add_scalar(f"R_M/Probe/{tag}/ReadabilityComposite", rc, epoch)

            if agg_rm:
                merged = self._aggregate_metric_dicts(agg_rm)
                self._log_readability_metrics(epoch, merged, [], [])
                print(f"[dense probe] readability logged ep{epoch+1}  "
                      f"composite={merged.get('readability_composite', float('nan')):.4f}")
        except Exception as _rm_err:
            import traceback
            print(f"[dense probe] readability metrics failed: {_rm_err}")
            traceback.print_exc()


    def _get_or_compute_norm(self, vol, mask, seg_id):
        """compute or load cached normalization stats for a volume using a mask"""
        cache_path = "./norm_cache.json"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                if isinstance(cache, dict) and seg_id in cache:
                    stats = cache[seg_id]
                    return stats["mean"], stats["std"], stats["min"], stats["max"]
            except Exception:
                pass

        total_sum, total_sq_sum, total_count = 0.0, 0.0, 0

        for z in tqdm(range(vol.shape[0]), desc="norm pass1", leave=False):
            chunk = vol[z, :, :]
            mask_chunk = mask[:, :]
            valid = chunk[mask_chunk > 0]
            if valid.size == 0:
                continue
            total_sum += float(np.sum(valid, dtype=np.float64))
            total_sq_sum += float(np.sum(np.square(valid, dtype=np.float64), dtype=np.float64))
            total_count += int(valid.size)

        if total_count == 0:
            raise ValueError("no valid pixels found for normalization")

        mean = total_sum / total_count
        std = float(np.sqrt(max((total_sq_sum / total_count) - (mean * mean), 1e-12)))

        g_min, g_max = float('inf'), float('-inf')
        for z in tqdm(range(vol.shape[0]), desc="norm pass2", leave=False):
            chunk = vol[z, :, :]
            mask_chunk = mask[:, :]
            valid = chunk[mask_chunk > 0]
            if valid.size == 0:
                continue
            norm = (valid.astype(np.float64) - mean) / std
            g_min = min(g_min, float(norm.min()))
            g_max = max(g_max, float(norm.max()))

        try:
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

            if not isinstance(cache, dict):
                cache = {}

            entry = cache.get(seg_id, {})
            if not isinstance(entry, dict):
                entry = {}
            entry["mean"] = mean
            entry["std"] = std
            entry["min"] = g_min
            entry["max"] = g_max
            cache[seg_id] = entry

            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=4)
        except Exception:
            pass

        return mean, std, g_min, g_max

    def _gen_tile_coords(self, z_range, y_range, x_range, mask):
        """generate valid tile coords within ranges filtered by mask"""
        z0, z1 = z_range
        y0, y1 = y_range
        x0, x1 = x_range

        depth = self.c.data.depth
        tile = self.c.data.tile_size

        z_span = max(0, z1 - z0 - depth + 1)
        y_span = max(0, y1 - y0 - tile + 1)
        x_span = max(0, x1 - x0 - tile + 1)

        coords = []
        z_step = max(1, depth // 2)

        for d in range(0, z_span, z_step):
            if z0 + d + depth > z1:
                continue
            for y in range(0, y_span, tile):
                for x in range(0, x_span, tile):
                    m_tile = mask[y0 + y:y0 + y + tile, x0 + x:x0 + x + tile]
                    if np.sum(m_tile) > 0:
                        coords.append((d, y, x))

        return coords

    def _load_test_region(self):
        """load test region based on training segment bottom slice"""
        sid = self.scroll1_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        vol = None
        try:
            import zarr
            vol = zarr.open(zarr_path, mode='r')
        except Exception as e:
            raise RuntimeError(f"could not open zarr at {zarr_path}: {e}")

        D, H, W = map(int, vol.shape)

        mask_path = f"./masks/{sid}.png"
        mask = imread_gray(mask_path) / 255.0

        # y_range = (max(0, H - max(0, H - 4200)), H)
        y_range = (0, H)
        x_range = (0, W)

        return vol, mask, y_range, x_range

    def _load_scroll4_region(self):
        """load scroll4 region with predefined slicing"""
        sid = self.c.data.scroll4_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        vol = None
        try:
            import zarr
            vol = zarr.open(zarr_path, mode='r')
        except Exception as e:
            raise RuntimeError(f"could not open zarr at {zarr_path}: {e}")

        D, H, W = map(int, vol.shape)

        mask_path = f"./masks/{sid}.png"
        mask = imread_gray(mask_path) / 255.0

        y_range = (6500 if H > 6500 else 0, H)
        x_range = (0, min(5000, W))

        return vol, mask, y_range, x_range

    def _load_scroll2_region(self):
        """load the ENTIRE scroll2 fragment as the goal-scroll test region.

        scroll2 is our target scroll: we want to see whether a model trained on the
        scroll1 fragments transfers any ink signal to it. the test figure renders the
        full fragment over full depth (tiles outside the papyrus mask are skipped, so
        cost tracks the actual segment area, not the bounding box)."""
        sid = self.c.data.scroll2_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        try:
            import zarr
            vol = zarr.open(zarr_path, mode='r')
        except Exception as e:
            raise RuntimeError(f"could not open zarr at {zarr_path}: {e}")

        mask_path = f"./masks/{sid}.png"
        mask = imread_gray(mask_path) / 255.0

        # full fragment extent (was a fixed 2048x1024 crop at x=3080,y=748)
        D, H, W = map(int, vol.shape)
        y_range = (0, H)
        x_range = (0, W)

        return vol, mask, y_range, x_range

    def _load_scroll3_region(self):
        """load the ENTIRE scroll3 fragment as a second goal-scroll test region.

        scroll3 (PHerc332) is the same 7.91um modality as the scroll4 training run — the
        real transfer target. long-and-skinny fragment; full extent over full depth (tiles
        outside the papyrus mask are skipped). raises if its zarr/mask are missing so the
        caller can skip the figure gracefully (it may still be downloading)."""
        sid = self.c.data.scroll3_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        import zarr
        vol = zarr.open(zarr_path, mode='r')

        mask_path = f"./masks/{sid}.png"
        mask = imread_gray(mask_path)
        if mask is None:
            raise FileNotFoundError(f"scroll3 mask not found at {mask_path}")
        mask = mask / 255.0

        D, H, W = map(int, vol.shape)
        y_range = (0, H)
        x_range = (0, W)

        return vol, mask, y_range, x_range

        """fixed readability probe regions, generated per active training scroll.

        each training scroll listed in config.data.tra_scroll_ids contributes its own
        easy/medium/hard ROIs (when defined below); the standard scroll4 and scroll2
        transfer checks are always appended. so a big-scroll-only run yields 3 probes +
        scroll4 + scroll2, while a small+big multiscroll run yields 6 probes + the two
        standard checks. all x/y snap to a tile multiple so the probe grid is co-aligned
        with the training tile grid (small-scroll values kept floor-aligned for
        historical comparability; big-scroll values snap to NEAREST per request)."""
        T = self.c.data.tile_size  # 32
        def nearest(v): return int((v + T // 2) // T) * T

        SMALL = 20230827161847
        BIG   = 20230702185753

        # per-training-scroll ROI definitions (already tile-aligned literals for SMALL)
        per_scroll = {
            SMALL: [
                {"tag": "Easy",   "title": "small scroll easy",   "segment_id": SMALL, "x": 2080, "y": 4352, "size": 608},
                {"tag": "Medium", "title": "small scroll medium", "segment_id": SMALL, "x": 2560, "y": 928,  "size": 608},
                {"tag": "Hard",   "title": "small scroll hard",   "segment_id": SMALL, "x": 3744, "y": 3840, "size": 608},
            ],
            BIG: [
                # coords snapped to nearest tile multiple; train/valid notes are the
                # user's annotations. dataloader splits train=left 75% (split_x=13024),
                # so easy/medium fall in train and hard (x~13344) is genuinely valid.
                {"tag": "BigEasy",   "title": "big scroll easy (train)",   "segment_id": BIG, "x": nearest(1728),  "y": nearest(6749),  "size": 608},
                {"tag": "BigMedium", "title": "big scroll medium (train)", "segment_id": BIG, "x": nearest(6285),  "y": nearest(10429), "size": 608},
                {"tag": "BigHard",   "title": "big scroll hard (valid)",   "segment_id": BIG, "x": nearest(13349), "y": nearest(6372),  "size": 608},
            ],
        }

        train_ids = [int(s) for s in (getattr(self.c.data, "tra_scroll_ids", None) or [self.c.data.tra_scroll_id])]
        specs = []
        for sid in train_ids:
            specs.extend(per_scroll.get(sid, []))

        # always-on transfer checks (scroll4 pi region + scroll2 goal scroll)
        specs.append({"tag": "Scroll4_Pi", "title": "scroll4 pi", "segment_id": self.c.data.scroll4_id, "x": 1952, "y": 7968, "size": 608})
        specs.append({"tag": "Scroll2",    "title": "scroll2",    "segment_id": self.c.data.scroll2_id, "x": 3072, "y": 736,  "size": 608})
        return specs

    def _build_probe_specs_legacy(self):
        """fixed readability probe regions used for qualitative tracking.
        all x/y coordinates are snapped to multiples of tile_size (32) so the
        probe tile grid is co-aligned with the training tile grid."""
        T = self.c.data.tile_size  # 32
        def align(v): return (v // T) * T
        return [
            {
                "tag": "Easy",
                "title": "small scroll easy",
                "segment_id": 20230827161847,
                "x": align(2100),   # 2080
                "y": align(4370),   # 4352
                "size": 608,
            },
            {
                "tag": "Medium",
                "title": "small scroll medium",
                "segment_id": 20230827161847,
                "x": align(2578),   # 2560
                "y": align(948),    # 928
                "size": 608,
            },
            {
                "tag": "Hard",
                "title": "small scroll hard",
                "segment_id": 20230827161847,
                "x": align(3744),   # 3744 (already aligned)
                "y": align(3862),   # 3840
                "size": 608,
            },
            {
                "tag": "Scroll4_Pi",
                "title": "scroll4 pi",
                "segment_id": 20231210132040,
                "x": align(1960),   # 1952
                "y": align(7968),   # 7968 (already aligned)
                "size": 608,
            },
            {
                "tag": "Scroll2",
                "title": "scroll2",
                "segment_id": 20230709155141,
                "x": align(3080),   # 3072
                "y": align(748),    # 736
                "size": 608,
            },
        ]

    def _load_segment_labels(self, seg_id):
        """load eroded labels for a segment"""
        path = f"./eroded_inklabels/{seg_id}.png"
        labels = imread_gray(path)
        if labels is None:
            raise RuntimeError(f"could not read labels at {path}")
        return labels / 255.0

    def _load_segment_mask(self, seg_id):
        """load mask for a segment"""
        path = f"./masks/{seg_id}.png"
        mask = imread_gray(path)
        if mask is None:
            raise RuntimeError(f"could not read mask at {path}")
        return mask / 255.0

    def _get_segment_asset(self, seg_id):
        """return cached volume mask labels and normalization stats for a segment"""
        if seg_id in self._segment_assets:
            return self._segment_assets[seg_id]

        if seg_id == self.scroll1_id:
            asset = {
                "volume": self.volume,
                "mask": self.mask,
                "labels": self.labels,
                "norm": (self.global_mean, self.global_std, self.global_min, self.global_max),
            }
        elif seg_id == self.c.data.scroll4_id:
            asset = {
                "volume": self.scroll4_volume,
                "mask": self.scroll4_mask,
                "labels": self._load_segment_labels(seg_id),
                "norm": (
                    self.scroll4_global_mean,
                    self.scroll4_global_std,
                    self.scroll4_global_min,
                    self.scroll4_global_max,
                ),
            }
        elif seg_id == self.c.data.scroll2_id:
            # scroll2 has no ink labels — substitute zeros so overlay is prediction-only
            labels = np.zeros(self.scroll2_mask.shape, dtype=np.float32)
            asset = {
                "volume": self.scroll2_volume,
                "mask": self.scroll2_mask,
                "labels": labels,
                "norm": (self.scroll2_global_mean, self.scroll2_global_std,
                         self.scroll2_global_min, self.scroll2_global_max),
            }
        else:
            import zarr

            volume = zarr.open(os.path.join(self.c.data.zarr_path, f"{seg_id}.zarr"), mode="r")
            mask = self._load_segment_mask(seg_id)
            labels = self._load_segment_labels(seg_id)
            g_mean, g_std, g_min, g_max = self._get_or_compute_norm(volume, mask, str(seg_id))
            asset = {
                "volume": volume,
                "mask": mask,
                "labels": labels,
                "norm": (g_mean, g_std, g_min, g_max),
            }

        self._segment_assets[seg_id] = asset
        return asset

    def _compute_tile_maps(self, labels, mask, y_range, x_range):
        """derive tile-aligned label fraction and validity maps anchored to the eval grid"""
        tile = self.c.data.tile_size
        y0, y1 = y_range
        x0, x1 = x_range
        h_small = max(0, (y1 - y0) // tile)
        w_small = max(0, (x1 - x0) // tile)

        label_binary = np.zeros((h_small, w_small), dtype=bool)
        label_fraction = np.zeros((h_small, w_small), dtype=np.float32)
        valid_tiles = np.zeros((h_small, w_small), dtype=bool)

        for yi in range(h_small):
            y = y0 + yi * tile
            for xi in range(w_small):
                x = x0 + xi * tile
                label_tile = labels[y:y + tile, x:x + tile]
                mask_tile = mask[y:y + tile, x:x + tile]
                if label_tile.shape != (tile, tile) or mask_tile.shape != (tile, tile):
                    continue
                if np.sum(mask_tile) <= 0:
                    continue
                ink = label_tile > 0.5
                valid_tiles[yi, xi] = True
                label_binary[yi, xi] = bool(np.any(ink))
                label_fraction[yi, xi] = float(np.mean(ink))

        return label_binary, label_fraction, valid_tiles

    def _compute_local_contrast_metrics(self, pred_map, label_binary, valid_tiles, radius=2):
        """measure local score separation around positive tiles"""
        pos_coords = np.argwhere(valid_tiles & label_binary)
        contrasts = []
        rankings = []

        for yi, xi in pos_coords:
            y0 = max(0, yi - radius)
            y1 = min(pred_map.shape[0], yi + radius + 1)
            x0 = max(0, xi - radius)
            x1 = min(pred_map.shape[1], xi + radius + 1)

            local_valid = valid_tiles[y0:y1, x0:x1]
            local_neg = local_valid & (~label_binary[y0:y1, x0:x1])
            if not np.any(local_neg):
                continue

            pos_score = float(pred_map[yi, xi])
            if not np.isfinite(pos_score):
                continue
            neg_scores = pred_map[y0:y1, x0:x1][local_neg]
            neg_scores = neg_scores[np.isfinite(neg_scores)]
            if neg_scores.size == 0:
                continue

            contrasts.append(pos_score - float(np.mean(neg_scores)))
            rankings.append(float(np.mean(pos_score > neg_scores)))

        if not contrasts:
            return np.nan, np.nan

        return float(np.mean(contrasts)), float(np.mean(rankings))

    def _compute_low_fpr_metrics(self, scores, labels, max_fpr=0.01):
        """measure recall and partial auc in the very low-fpr regime"""
        if scores.size == 0 or len(np.unique(labels)) < 2:
            return np.nan, np.nan

        fpr, tpr, _ = roc_curve(labels, scores)
        keep = fpr <= max_fpr
        if not np.any(keep):
            return 0.0, 0.0

        recall_at_low_fpr = float(np.max(tpr[keep]))
        tpr_at_max = float(np.interp(max_fpr, fpr, tpr))
        fpr_part = fpr[keep]
        tpr_part = tpr[keep]
        if fpr_part[-1] < max_fpr:
            fpr_part = np.concatenate([fpr_part, [max_fpr]])
            tpr_part = np.concatenate([tpr_part, [tpr_at_max]])

        partial_auc = float(np.trapz(tpr_part, fpr_part) / max_fpr)
        return recall_at_low_fpr, partial_auc

    def _compute_topk_precision(self, scores, labels):
        """precision among the top-k scores where k equals positive-tile count"""
        k = int(np.sum(labels))
        if k <= 0 or scores.size == 0:
            return np.nan

        k = min(k, scores.size)
        top_idx = np.argsort(scores)[::-1][:k]
        return float(np.mean(labels[top_idx]))

    def _compute_fraction_correlation(self, scores, fractions):
        """correlation between score and per-tile ink fraction"""
        if scores.size < 2 or np.std(scores) <= 1e-12 or np.std(fractions) <= 1e-12:
            return np.nan, np.nan

        pearson = float(np.corrcoef(scores, fractions)[0, 1])
        spearman = float(scipy_stats.spearmanr(scores, fractions).correlation)
        return pearson, spearman

    def _compute_spill_metrics(self, pred_map, label_binary, valid_tiles):
        """measure positive mass spill and binary component structure at ink budget"""
        valid_scores = pred_map[valid_tiles]
        valid_scores = valid_scores[np.isfinite(valid_scores)]  # drop NaN (outside-volume tiles)
        if valid_scores.size == 0:
            return np.nan, np.nan, np.nan

        dilated_gt = ndimage.binary_dilation(label_binary, iterations=1)
        outside_dilated = valid_tiles & (~dilated_gt)
        spill_ratio = float(pred_map[outside_dilated].sum() / max(valid_scores.sum(), 1e-8))

        labels_flat = label_binary[valid_tiles].astype(int)
        k = int(np.sum(labels_flat))
        if k <= 0:
            return spill_ratio, np.nan, np.nan

        k = min(k, valid_scores.size)
        valid_indices = np.argwhere(valid_tiles)
        top_idx = np.argsort(valid_scores)[::-1][:k]
        budget_mask = np.zeros_like(pred_map, dtype=np.uint8)
        for idx in top_idx:
            yi, xi = valid_indices[idx]
            budget_mask[yi, xi] = 1

        components, num_components = ndimage.label(budget_mask)
        if num_components <= 0:
            return spill_ratio, 0.0, 0.0

        component_sizes = ndimage.sum(np.ones_like(components), components, index=np.arange(1, num_components + 1))
        mean_component_size = float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0
        return spill_ratio, float(num_components), mean_component_size

    def _compute_readability_metrics(self, pred_map, label_binary, label_fraction, valid_tiles):
        """compute readability-aligned scalar metrics for a prediction map.

        composite redesigned to favour COVERAGE and SPATIAL COHERENCE over precision:
          - removed: topk_precision, spill_good  (both reward high-precision / conservative abstention)
          - added:   recall@5%fpr, pauc@5%fpr   (broader recall budget)
          - added:   coverage_recall             (fraction of labeled ink tiles with score > 0.3)
          - added:   coherence                   (mean_component_size, normalised)
        """
        # defensive: crop all maps to a common shape. tile-grid rounding can leave a
        # one-tile mismatch between the prediction map and the label/valid maps; a hard
        # crop here prevents a boolean-index crash from killing figure generation.
        h = min(pred_map.shape[0], label_binary.shape[0], label_fraction.shape[0], valid_tiles.shape[0])
        w = min(pred_map.shape[1], label_binary.shape[1], label_fraction.shape[1], valid_tiles.shape[1])
        pred_map       = pred_map[:h, :w]
        label_binary   = label_binary[:h, :w]
        label_fraction = label_fraction[:h, :w]
        valid_tiles    = valid_tiles[:h, :w]

        valid_scores    = pred_map[valid_tiles]
        valid_labels    = label_binary[valid_tiles].astype(int)
        valid_fraction  = label_fraction[valid_tiles]

        # drop tiles where the prediction map has NaN (tile outside volume or input-mode failed)
        finite_mask    = np.isfinite(valid_scores)
        valid_scores   = valid_scores[finite_mask]
        valid_labels   = valid_labels[finite_mask]
        valid_fraction = valid_fraction[finite_mask]

        local_contrast, local_ranking = self._compute_local_contrast_metrics(pred_map, label_binary, valid_tiles)
        recall_at_1pct_fpr, partial_auc_at_1pct_fpr = self._compute_low_fpr_metrics(valid_scores, valid_labels, max_fpr=0.01)
        recall_at_5pct_fpr, partial_auc_at_5pct_fpr = self._compute_low_fpr_metrics(valid_scores, valid_labels, max_fpr=0.05)
        topk_precision = self._compute_topk_precision(valid_scores, valid_labels)
        fraction_corr_pearson, fraction_corr_spearman = self._compute_fraction_correlation(valid_scores, valid_fraction)
        spill_ratio, component_count, mean_component_size = self._compute_spill_metrics(pred_map, label_binary, valid_tiles)

        # coverage: fraction of labeled positive tiles that score above a moderate threshold
        # measures whether the model is FINDING most of the ink, not just the easiest ink
        COVERAGE_THRESHOLD = 0.3
        pos_scores = pred_map[valid_tiles & label_binary]
        pos_scores = pos_scores[np.isfinite(pos_scores)]   # drop NaN before mean
        coverage_recall = float(np.mean(pos_scores > COVERAGE_THRESHOLD)) if pos_scores.size > 0 else np.nan

        # coherence: normalise mean_component_size — larger blobs = more letter-like structure
        # cap at 20 tiles (a reasonable stroke width in tile units); values above are noise
        coherence = np.clip(np.nan_to_num(mean_component_size, nan=0.0) / 20.0, 0.0, 1.0)

        contrast_norm   = np.clip(np.nan_to_num(local_contrast, nan=0.0), 0.0, 1.0)
        ranking_norm    = np.clip(np.nan_to_num(local_ranking, nan=0.0), 0.0, 1.0)
        # 1%fpr metrics still logged but excluded from composite (too strict, rewards abstention)
        recall5_norm    = np.clip(np.nan_to_num(recall_at_5pct_fpr, nan=0.0), 0.0, 1.0)
        pauc5_norm      = np.clip(np.nan_to_num(partial_auc_at_5pct_fpr, nan=0.0), 0.0, 1.0)
        coverage_norm   = np.clip(np.nan_to_num(coverage_recall, nan=0.0), 0.0, 1.0)
        corr_norm       = np.clip((np.nan_to_num(fraction_corr_spearman, nan=-1.0) + 1.0) / 2.0, 0.0, 1.0)

        # weighted composite: local_contrast is the primary signal — high weight.
        # coverage is deliberately down-weighted (0.5×): ring-trained models saturate it
        # at 1.0 trivially (everything fires), making it uninformative for model selection.
        # coherence up-weighted (2.0×): letter-shaped blobs = meaningful structure.
        weights = [2.0, 1.0, 1.0, 1.0, 0.5, 1.0, 2.0]
        terms   = [contrast_norm, ranking_norm, recall5_norm, pauc5_norm, coverage_norm, corr_norm, coherence]
        readability_composite = float(np.average(terms, weights=weights))

        return {
            "local_contrast":           float(local_contrast),
            "local_ranking":            float(local_ranking),
            "recall_at_1pct_fpr":       float(recall_at_1pct_fpr),
            "partial_auc_at_1pct_fpr":  float(partial_auc_at_1pct_fpr),
            "recall_at_5pct_fpr":       float(recall_at_5pct_fpr),
            "partial_auc_at_5pct_fpr":  float(partial_auc_at_5pct_fpr),
            "coverage_recall":          float(coverage_recall),
            "topk_precision":           float(topk_precision),
            "ink_fraction_corr_pearson": float(fraction_corr_pearson),
            "ink_fraction_corr_spearman": float(fraction_corr_spearman),
            "spill_ratio":              float(spill_ratio),
            "component_count":          float(component_count),
            "mean_component_size":      float(mean_component_size),
            "readability_composite":    readability_composite,
        }

    def _aggregate_metric_dicts(self, metrics_list):
        """average scalar metrics across depth blocks while ignoring missing values"""
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        aggregate = {}
        for key in keys:
            vals = [m[key] for m in metrics_list if np.isfinite(m[key])]
            aggregate[key] = float(np.mean(vals)) if vals else np.nan
        return aggregate

    def log_epoch_metrics(self, epoch, model, train_metrics, val_metrics, learning_rate, time_elapsed, params, pos_weight):
        """log metrics images and hparams"""
        print(f"Logging metrics for epoch: {epoch+1}")

        self.writer.add_scalar("G_M/Loss/Train", train_metrics['loss'], epoch)
        self.writer.add_scalar("G_M/Loss/Train_Raw", train_metrics['raw_loss'], epoch)
        self.writer.add_scalar("G_M/Loss/Valid", val_metrics['loss'], epoch)

        self.writer.add_scalar("G_M/Acc/Train", train_metrics['accuracy'], epoch)
        self.writer.add_scalar("G_M/Acc/Valid", val_metrics['accuracy'], epoch)
        # balanced accuracy: mean(sensitivity, specificity) — invariant to ring imbalance.
        # unlike raw accuracy it is 0.5 when the model predicts all-one-class, not 0.566x.
        self.writer.add_scalar("G_M/BalAcc/Train", train_metrics.get('balanced_accuracy', 0.0), epoch)
        self.writer.add_scalar("G_M/BalAcc/Valid", val_metrics.get('balanced_accuracy', 0.0), epoch)

        self.writer.add_scalar("P_M/Precision/Train", train_metrics['precision'], epoch)
        self.writer.add_scalar("P_M/Precision/Valid", val_metrics['precision'], epoch)
        self.writer.add_scalar("P_M/Recall/Train", train_metrics['recall'], epoch)
        self.writer.add_scalar("P_M/Recall/Valid", val_metrics['recall'], epoch)
        self.writer.add_scalar("P_M/F1_Score/Train", train_metrics['f1'], epoch)
        self.writer.add_scalar("P_M/F1_Score/Valid", val_metrics['f1'], epoch)
        self.writer.add_scalar("P_M/Specificity/Train", train_metrics['specificity'], epoch)
        self.writer.add_scalar("P_M/Specificity/Valid", val_metrics['specificity'], epoch)

        self.writer.add_scalar("AUC/ROC_AUC/Train", train_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/ROC_AUC/Valid", val_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Train", train_metrics['pr_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Valid", val_metrics['pr_auc'], epoch)

        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        self.writer.add_scalar('Time_Elapsed', time_elapsed, epoch)

        self.log_confusion_matrix(train_metrics, val_metrics, epoch)
        self.log_output_histogram(train_metrics, val_metrics, epoch)
        self.log_metrics_comparison(train_metrics, val_metrics, epoch)

        # weight/gradient histogram logging disabled — too expensive on large models
        # self.log_weight_histograms(model, epoch)

        if epoch == 0:
            print("Logging hyperparameters and model graph")
            ex = torch.randn(1, self.c.data.depth, self.c.data.tile_size, self.c.data.tile_size).to(self.c.device)
            ex = ex.unsqueeze(0)
            # self.log_model_graph(model, ex)
            self.log_hyperparameters(params, pos_weight)

        if self.mode == 'train' and self.eval_enabled and (epoch + 1) % self.c.tra.eval_int == 0:
            try:
                if getattr(self.c.data, "dense_labels", False):
                    self.add_dense_evaluation_figure(epoch, model)
                else:
                    self.add_evaluation_figures(epoch, model)
            except Exception as e:
                print(f"[ERROR] add_evaluation_figures failed at epoch {epoch}: {e}")
                import traceback; traceback.print_exc()

        if self.mode == 'train' and (epoch + 1) % self.c.tra.test_int == 0:
            try:
                self.add_test_figures(epoch, model)
            except Exception as e:
                print(f"[ERROR] add_test_figures failed at epoch {epoch}: {e}")
                import traceback; traceback.print_exc()

        if self.mode == 'train' and self.probe_rois_enabled and (epoch + 1) % self.probe_log_interval == 0:
            try:
                self.add_probe_region_figures(epoch, model)
            except Exception as e:
                print(f"[ERROR] add_probe_region_figures failed at epoch {epoch}: {e}")
                import traceback; traceback.print_exc()

        # dense per-pixel probes: fire every probe_int epochs for all named probe specs.
        if (self.mode == 'train'
                and getattr(self.c.data, "dense_labels", False)
                and self._dense_probe_specs
                and (epoch + 1) % self.probe_log_interval == 0):
            try:
                self.add_dense_probe_figure(epoch, model)
            except Exception as e:
                print(f"[ERROR] add_dense_probe_figure failed at epoch {epoch}: {e}")
                import traceback; traceback.print_exc()

        self.writer.flush()

    def log_confusion_matrix(self, train_metrics, val_metrics, epoch):
        """create and log confusion matrix visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        train_tp = train_metrics['positive_samples'] * train_metrics['recall']
        train_fp = train_tp * (1 - train_metrics['precision']) / train_metrics['precision'] if train_metrics['precision'] > 0 else 0
        train_fn = train_metrics['positive_samples'] - train_tp
        train_tn = train_metrics['negative_samples'] - train_fp

        train_cm = np.array([[train_tn, train_fp], [train_fn, train_tp]])

        sns.heatmap(train_cm, annot=True, fmt='.0f', cmap='Blues', ax=ax1,
                    xticklabels=['Predicted No Ink', 'Predicted Ink'],
                    yticklabels=['Actual No Ink', 'Actual Ink'])
        ax1.set_title(f'Training Confusion Matrix\nPrecision: {train_metrics["precision"]:.3f}, Recall: {train_metrics["recall"]:.3f}')

        val_tp = val_metrics['positive_samples'] * val_metrics['recall']
        val_fp = val_tp * (1 - val_metrics['precision']) / val_metrics['precision'] if val_metrics['precision'] > 0 else 0
        val_fn = val_metrics['positive_samples'] - val_tp
        val_tn = val_metrics['negative_samples'] - val_fp

        val_cm = np.array([[val_tn, val_fp], [val_fn, val_tp]])

        sns.heatmap(val_cm, annot=True, fmt='.0f', cmap='Oranges', ax=ax2,
                    xticklabels=['Predicted No Ink', 'Predicted Ink'],
                    yticklabels=['Actual No Ink', 'Actual Ink'])
        ax2.set_title(f'Valid Confusion Matrix\nPrecision: {val_metrics["precision"]:.3f}, Recall: {val_metrics["recall"]:.3f}')

        plt.tight_layout()
        self.writer.add_figure('Confusion_Matrix', fig, epoch)
        plt.close(fig)

    def log_output_histogram(self, train_metrics, val_metrics, epoch):
        """create and log histogram of model outputs for training and validation"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        bins = np.linspace(0, 1, 51)

        ax.hist(train_metrics['scores'], bins=bins, alpha=0.6, label='Training', color='skyblue', edgecolor='black', density=True)  # type: ignore
        ax.hist(val_metrics['scores'], bins=bins, alpha=0.6, label='Validation', color='lightcoral', edgecolor='black', density=True)  # type: ignore

        ax.set_xlabel('Model Output (Sigmoid Score)')
        ax.set_ylabel('Density')
        ax.set_title('Model Output Distribution\nTraining vs Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1)

        plt.tight_layout()
        self.writer.add_figure('Output_Histogram', fig, epoch)
        plt.close(fig)

    def log_metrics_comparison(self, train_metrics, val_metrics, epoch):
        """create and log a comprehensive metrics comparison chart"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        metrics_to_plot = ['precision', 'recall', 'f1', 'specificity', 'roc_auc', 'pr_auc']

        ax1 = axes[0]
        train_vals = [train_metrics[m] for m in metrics_to_plot]
        val_vals = [val_metrics[m] for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, train_vals, width, label='Train', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, val_vals, width, label='Valid', color='lightcoral', alpha=0.8)

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Training vs Valid Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.annotate(f'{h:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

        categories = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC', 'PR-AUC']
        radar_ax = fig.add_subplot(1, 2, 2, projection='polar')
        self._plot_radar_chart(
            radar_ax,
            categories,
            [
                ("Train", train_vals, "blue"),
                ("Valid", val_vals, "red"),
            ],
            title='Performance Radar Chart',
            ylim=(0, 1),
        )

        plt.tight_layout()
        self.writer.add_figure('Metrics_Comparison', fig, epoch)
        plt.close(fig)

    def _plot_radar_chart(self, ax, categories, series, title, ylim=(0, 1)):
        """plot one or more normalized series on a radar chart"""
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for label, values, color in series:
            values_c = [float(np.nan_to_num(v, nan=0.0)) for v in values]
            values_c += values_c[:1]
            ax.plot(angles, values_c, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values_c, alpha=0.2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.set_title(title, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

    def _hard_mining_dir(self):
        """return hard-mining directory for the current experiment (per scroll fragment)"""
        base = getattr(self.c.hm, "dir", "./hard_negs")
        # keep mined files separated per scroll so global (z,y,x) keys never collide
        if getattr(self, "scroll1_id", None) is not None:
            return os.path.join(base, f"scroll_{self.scroll1_id}")
        return base

    def add_dense_evaluation_figure(self, epoch, model):
        """dense per-pixel eval figure (for dense_labels runs).

        renders the ENTIRE region given to the model (the whole cropped scroll = train range
        UNION valid range on the split axis, full shared range on the other) — matching the
        historical add_evaluation_figures which spans the full train+valid region, NOT a small
        window. per the README convention it ALWAYS sweeps the FULL inference depth
        d_start..d_end (e.g. 0->64) in blocks of `depth`, regardless of the (narrower) training
        window. shows one full-region prediction map per depth block, a depth-MAX composite, and
        the ground truth, with the train/valid split boundary drawn. inference is fully-
        convolutional over large chunks; each chunk's per-pixel prediction is downsampled into a
        canvas so the whole ~17k x 31k region fits in memory. the composite drives the logged
        per-pixel validation AUC (computed over the VALID sub-region only)."""
        import matplotlib.pyplot as plt
        model.eval()
        dev = self.c.device
        D = int(self.c.data.depth)
        zf0 = int(self.c.data.d_start)
        zf1 = min(int(self.c.data.d_end), int(self.volume.shape[0]))
        stride_z = max(1, D)
        z_starts = list(range(zf0, max(zf0 + 1, zf1 - D + 1), stride_z))
        if not z_starts:
            z_starts = [zf0]
        if z_starts[-1] != zf1 - D and (zf1 - D) >= zf0:
            z_starts.append(zf1 - D)
        g_mean, g_std, g_min, g_max = self.global_mean, self.global_std, self.global_min, self.global_max
        vol, lab, mk = self.volume, self.labels, self.eval_mask

        # FULL given region = union of train+valid on the split axis, shared range on the other.
        if getattr(self, "split_axis", "x") == "y":
            (tr_lo, tr_hi) = self.train_range
            (va_lo, va_hi) = self.valid_range
            (sx0, sx1) = self.shared_range
            y0, y1 = min(tr_lo, va_lo), max(tr_hi, va_hi)
            x0, x1 = sx0, sx1
            split_is_y = True
            split_at = va_lo   # valid starts here on the y axis
        else:
            (tr_lo, tr_hi) = self.train_range
            (va_lo, va_hi) = self.valid_range
            (sy0, sy1) = self.shared_range
            x0, x1 = min(tr_lo, va_lo), max(tr_hi, va_hi)
            y0, y1 = sy0, sy1
            split_is_y = False
            split_at = va_lo
        y1 = min(y1, int(vol.shape[1])); x1 = min(x1, int(vol.shape[2]))
        Hreg, Wreg = y1 - y0, x1 - x0

        # downsample factor so the whole region fits a ~3000px-max canvas
        DS = max(1, int(np.ceil(max(Hreg, Wreg) / 3000.0)))
        Hc, Wc = Hreg // DS, Wreg // DS

        # fully-convolutional chunk size (divisible by 8 for the 3-pool U-Net AND by DS so the
        # downsampled placement tiles exactly)
        CH = 768
        CH = (CH // 8) * 8
        while CH % DS != 0:
            CH -= 8

        def _norm(blk):
            blk = (blk - g_mean) / g_std
            return np.clip((blk - g_min) / (g_max - g_min + 1e-12), 0, 1)

        def _pad8(a):
            _, h, w = a.shape
            ph = (-h) % 8; pw = (-w) % 8
            if ph or pw:
                a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            return a, h, w

        # overlapping tiled inference: chunks overlap by CH//4 on each side and are blended
        # with a 2D Hann window. this eliminates the grid artifact caused by the U-Net's
        # receptive field context differing at non-overlapping chunk boundaries.
        _OVERLAP = max(CH // 4, 8)
        _stride  = max(CH - _OVERLAP, CH // 2)
        ys = list(range(y0, y1, _stride))
        xs = list(range(x0, x1, _stride))

        def _predict_region_at_depth(z0):
            accum  = np.zeros((Hc, Wc), np.float32)
            weight = np.zeros((Hc, Wc), np.float32)
            import cv2 as _cv2
            import time as _time
            _chunk_sleep_s = int(getattr(self.c.tra, 'fig_chunk_cooldown_ms', 0)) / 1000.0
            with torch.no_grad():
                for yy in ys:
                    ch = min(CH, y1 - yy)
                    for xx in xs:
                        cw = min(CH, x1 - xx)
                        blk = np.asarray(vol[z0:z0 + D, yy:yy + ch, xx:xx + cw]).astype(np.float32)
                        if blk.shape[0] != D:
                            continue
                        blk = _norm(blk)
                        blk, oh, ow = _pad8(blk)
                        bt = torch.from_numpy(blk).unsqueeze(0).unsqueeze(0).float().to(dev)
                        with autocast(dev):
                            p = torch.sigmoid(model(bt))[0, 0, :oh, :ow].float().cpu().numpy()
                        if _chunk_sleep_s > 0:
                            _time.sleep(_chunk_sleep_s)
                        # 2D Hann taper — N+2 trick avoids exact-zero endpoints
                        wy = np.hanning(oh + 2)[1:-1].astype(np.float32).reshape(-1, 1)
                        wx = np.hanning(ow + 2)[1:-1].astype(np.float32).reshape(1, -1)
                        w2d = (wy * wx)
                        # downsample into canvas with weighted accumulation
                        cyd = (yy - y0) // DS
                        cxd = (xx - x0) // DS
                        chd = oh // DS
                        cwd = ow // DS
                        if chd < 1 or cwd < 1:
                            continue
                        pd = _cv2.resize(p,   (cwd, chd), interpolation=_cv2.INTER_AREA)
                        wd = _cv2.resize(w2d, (cwd, chd), interpolation=_cv2.INTER_AREA)
                        ey = min(cyd + chd, Hc)
                        ex = min(cxd + cwd, Wc)
                        accum [cyd:ey, cxd:ex] += (pd * wd)[:ey - cyd, :ex - cxd]
                        weight[cyd:ey, cxd:ex] += wd       [:ey - cyd, :ex - cxd]
            with np.errstate(invalid='ignore'):
                canvas = np.where(weight > 1e-6, accum / weight, 0.0)
            return canvas

        # downsampled mask and GT (raw scan not displayed; removed to reduce memory + clutter)
        import cv2 as _cv
        mask_ds = _cv.resize((np.asarray(mk[y0:y1, x0:x1]) > 0).astype(np.float32), (Wc, Hc), interpolation=_cv.INTER_AREA)
        mask_ds = (mask_ds > 0.5).astype(np.float32)
        gt_ds = _cv.resize((np.asarray(lab[y0:y1, x0:x1]) > 0.5).astype(np.float32), (Wc, Hc), interpolation=_cv.INTER_AREA)

        preds = [(z0, _predict_region_at_depth(z0) * mask_ds) for z0 in z_starts]
        composite = np.max(np.stack([p for _, p in preds], axis=0), axis=0) * mask_ds

        # validation AUC over the VALID sub-region only (the held-out part)
        split_ds = max(0, min(Hc if split_is_y else Wc, (split_at - (y0 if split_is_y else x0)) // DS))
        if split_is_y:
            comp_val, gt_val, m_val = composite[split_ds:], gt_ds[split_ds:], mask_ds[split_ds:]
        else:
            comp_val, gt_val, m_val = composite[:, split_ds:], gt_ds[:, split_ds:], mask_ds[:, split_ds:]
        sel = m_val.reshape(-1) > 0
        try:
            from sklearn.metrics import roc_auc_score
            yv = (gt_val.reshape(-1)[sel] > 0.5).astype(int)
            auc = roc_auc_score(yv, comp_val.reshape(-1)[sel]) if len(np.unique(yv)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        self.writer.add_scalar("Dense/Valid_PixelAUC", auc, epoch)

        # READABILITY METRICS: downsample composite to tile resolution and reuse the
        # standard readability pipeline so R_M/* scalars appear for dense runs too.
        # we tile-max-pool the pixel-space composite: each tile's score = max prediction
        # over its T×T px region (mapped through DS). center-sampling from the DS canvas
        # is equivalent and avoids a second resize.
        try:
            T = self.c.data.tile_size
            lab_full  = np.asarray(self.labels)
            mask_full = np.asarray(self.mask)
            # compute tile-level label maps over the full eval region (train+valid)
            lb, lf, vt = self._compute_tile_maps(lab_full, mask_full, (y0, y1), (x0, x1))
            h_tiles, w_tiles = lb.shape
            # center-sample the DS-downsampled composite into tile resolution
            tile_ys = np.arange(h_tiles) * T // DS + (T // DS // 2)
            tile_xs = np.arange(w_tiles) * T // DS + (T // DS // 2)
            tile_ys = np.clip(tile_ys, 0, Hc - 1)
            tile_xs = np.clip(tile_xs, 0, Wc - 1)
            tile_pred = composite[np.ix_(tile_ys, tile_xs)]  # (h_tiles, w_tiles)
            rm = self._compute_readability_metrics(tile_pred, lb, lf, vt)
            self._log_readability_metrics(epoch, rm, {}, [])
        except Exception as _rm_err:
            print(f"[dense eval] readability metrics failed: {_rm_err}")

        def _mark_split(ax):
            if split_is_y:
                ax.axhline(split_ds, color="cyan", lw=1.0, ls="--")
            else:
                ax.axvline(split_ds, color="cyan", lw=1.0, ls="--")

        # 2-column layout: left = prediction, right = prediction + GT label overlay (gold)
        # no raw scan, no standalone GT row — refer to attached reference image
        n_panels = len(preds) + 1   # one row per depth block + composite
        gt_ov = np.zeros((Hc, Wc, 4), dtype=np.float32)   # gold RGBA overlay
        gt_ov[..., 0] = 0.98
        gt_ov[..., 1] = 0.85
        gt_ov[..., 2] = 0.37
        gt_ov[..., 3] = np.clip(gt_ds, 0.0, 1.0) * 0.50

        panel_h = max(3.0, min(8.0, Hc / 250))
        fig_w   = min(22, 2.0 * Wc / 150)
        fig, axes = plt.subplots(n_panels, 2,
                                 figsize=(fig_w, panel_h * n_panels),
                                 squeeze=False)

        for i, (z0_b, p) in enumerate(preds):
            axes[i, 0].imshow(p, cmap="magma", vmin=0, vmax=1)
            axes[i, 0].set_title(f"pred z{z0_b}-{z0_b + D}", fontsize=7)
            _mark_split(axes[i, 0])
            axes[i, 1].imshow(p, cmap="magma", vmin=0, vmax=1)
            axes[i, 1].imshow(gt_ov)                             # GT overlay
            axes[i, 1].set_title(f"pred z{z0_b}-{z0_b + D}  + GT", fontsize=7)
            _mark_split(axes[i, 1])

        axes[-1, 0].imshow(composite, cmap="magma", vmin=0, vmax=1)
        axes[-1, 0].set_title("depth-MAX composite", fontsize=7)
        _mark_split(axes[-1, 0])
        axes[-1, 1].imshow(composite, cmap="magma", vmin=0, vmax=1)
        axes[-1, 1].imshow(gt_ov)
        axes[-1, 1].set_title(f"composite  + GT   VALID_auc={auc:.3f}", fontsize=7)
        _mark_split(axes[-1, 1])

        for row in axes:
            for a_ in row:
                a_.axis("off")
        plt.suptitle(f"ep{epoch+1}  y[{y0},{y1}] x[{x0},{x1}]  "
                     f"train z{self.c.data.train_d_start}-{self.c.data.train_d_end}  "
                     f"cyan=train/valid boundary", fontsize=8)
        plt.tight_layout()
        self.writer.add_figure("Dense/FullRegion_Prediction", fig, epoch)
        try:
            out_dir = os.path.join(os.path.dirname(self.log_path), "dense_figs")
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f"dense_eval_ep{epoch+1:02d}.png")
            fig.savefig(out_png, dpi=100)
            # shrink saved file to half resolution; inference canvas stays full-res
            import cv2 as _cv2
            _img = _cv2.imread(out_png)
            if _img is not None:
                _h, _w = _img.shape[:2]
                _cv2.imwrite(out_png, _cv2.resize(_img, (_w//2, _h//2), interpolation=_cv2.INTER_AREA))
            print(f"[dense eval] ep{epoch+1} FULL region y[{y0},{y1}] x[{x0},{x1}] "
                  f"canvas={Hc}x{Wc} DS={DS} depth-blocks={[z for z,_ in preds]} "
                  f"VALID_auc={auc:.4f} -> {out_png}")
        except Exception as e:
            print(f"[dense eval] figure save failed: {e}")
        plt.close(fig)

    def add_evaluation_figures(self, epoch, model):
        """run eval on train and valid splits produce mining and figures"""
        print("Starting evaluation figure generation...")
        model.eval()
        z_range = (self.c.data.d_start, self.c.data.d_end)

        # use ring mask when ring_negatives=True so the eval figure only renders
        # ring+ink tiles — everything else stays NaN and renders black, matching
        # the actual training distribution instead of swamping the signal with OOD noise
        eval_mask = getattr(self, 'eval_mask', self.mask)
        # axis-aware split: 'y' -> train=top rows / valid=bottom rows, shared x, stack vertically;
        # 'x' -> legacy train=left / valid=right, shared y, stack horizontally.
        if getattr(self, "split_axis", "x") == "y":
            tr_y, tr_x = self.train_range, self.shared_range
            va_y, va_x = self.valid_range, self.shared_range
            concat_axis = 0
        else:
            tr_y, tr_x = self.shared_range, self.train_range
            va_y, va_x = self.shared_range, self.valid_range
            concat_axis = 1
        train_coords = self._gen_tile_coords(z_range, tr_y, tr_x, eval_mask)
        valid_coords = self._gen_tile_coords(z_range, va_y, va_x, eval_mask)

        train_grouped = group_by_depth(train_coords)
        valid_grouped = group_by_depth(valid_coords)
        depth_offsets = sorted(set(train_grouped.keys()) | set(valid_grouped.keys()))
        all_pred_data = []

        hm_dir = self._hard_mining_dir()
        hm_enabled = getattr(self.c.hm, "enabled", True)

        if hm_enabled:
            os.makedirs(hm_dir, exist_ok=True)
            mining_path = os.path.join(hm_dir, f"hard_mining_epoch_{epoch}.jsonl")
            mining_f = open(mining_path, "w")
            print(f"[HARD][Eval] Writing mining file to: {mining_path}")
        else:
            mining_path = None
            mining_f = None

        hn_cut = self.c.hm.hn_cutoff
        hp_cut = self.c.hm.hp_cutoff
        hn_cnt = 0
        hp_cnt = 0

        # load a set of existing mined keys across all previous files to prevent duplicates
        existing_keys = self._load_existing_mined_keys() if hm_enabled else set()
        # also track keys added in this epoch to avoid intra epoch duplicates
        new_keys = set()

        for d_off in depth_offsets:
            depth_start = d_off + self.c.data.d_start
            depth_end = depth_start + self.c.data.depth

            t_coords = train_grouped.get(d_off, [])
            v_coords = valid_grouped.get(d_off, [])

            t_pred = predict_tiles(
                self.c, model, self.volume, eval_mask, t_coords, tr_y, tr_x,
                depth_start, "train", self.global_mean, self.global_std, self.global_min, self.global_max
            )

            v_pred = predict_tiles(
                self.c, model, self.volume, eval_mask, v_coords, va_y, va_x,
                depth_start, "valid", self.global_mean, self.global_std, self.global_min, self.global_max
            )

            tile = self.c.data.tile_size

            for (_, y_off, x_off) in t_coords:
                yi = y_off // tile
                xi = x_off // tile
                if yi < 0 or yi >= t_pred.shape[0] or xi < 0 or xi >= t_pred.shape[1]:
                    continue

                score = float(t_pred[yi, xi])

                z_global = depth_start
                y_global = tr_y[0] + y_off
                x_global = tr_x[0] + x_off

                l_tile = self.labels[y_global:y_global + tile, x_global:x_global + tile]
                has_ink = int(np.any(l_tile > 0.5))

                # dedup key includes z y x and label
                key = (int(z_global), int(y_global), int(x_global), int(has_ink))

                # scroll_id makes each record self-describing so the injector can
                # route it back to the correct volume in multi-scroll training
                sid_rec = int(getattr(self, "scroll1_id", 0) or 0)
                if has_ink == 0 and score >= hn_cut:
                    if key not in existing_keys and key not in new_keys:
                        if mining_f is not None:
                            mining_f.write(json.dumps({"scroll_id": sid_rec, "z": z_global, "y": y_global, "x": x_global, "score": score, "label": 0}) + "\n")
                        new_keys.add(key)
                        hn_cnt += 1
                elif has_ink == 1 and score <= hp_cut:
                    if key not in existing_keys and key not in new_keys:
                        if mining_f is not None:
                            mining_f.write(json.dumps({"scroll_id": sid_rec, "z": z_global, "y": y_global, "x": x_global, "score": score, "label": 1}) + "\n")
                        new_keys.add(key)
                        hp_cnt += 1

            full_pred = np.concatenate([t_pred, v_pred], axis=concat_axis)
            all_pred_data.append((full_pred, t_pred, depth_start, depth_end))

        if mining_f is not None:
            mining_f.write(json.dumps({"_type": "meta", "hard_negatives": hn_cnt, "hard_positives": hp_cnt}) + "\n")
            mining_f.close()
            print(f"[HARD][Eval] Finished mining epoch {epoch}: neg={hn_cnt} pos={hp_cnt}")
        # no print when disabled — the user already knows

        if hm_enabled:
            self.writer.add_scalar("HardMining/HardNegatives", hn_cnt, epoch)
            self.writer.add_scalar("HardMining/HardPositives", hp_cnt, epoch)

        fig = self._create_hard_examples_overlay(mining_path) if mining_path else None
        if fig is not None:
            self.writer.add_figure(f"HardMined/Overlay", fig, epoch)
            plt.close(fig)

        if all_pred_data:
            # label map spans the full split extent along the split axis, shared range on the other
            if getattr(self, "split_axis", "x") == "y":
                full_y_range = (self.train_range[0], self.valid_range[1])
                full_x_range = self.shared_range
            else:
                full_y_range = self.shared_range
                full_x_range = (self.train_range[0], self.valid_range[1])
            label_binary, label_fraction, valid_tiles = self._compute_tile_maps(
                self.labels,
                self.mask,
                full_y_range,
                full_x_range,
            )
            per_depth_metrics = []
            depth_labels = []

            for pred_data in all_pred_data:
                depth_start = pred_data[2]
                depth_end = pred_data[3]

                per_depth_metrics.append(
                    self._compute_readability_metrics(pred_data[0], label_binary, label_fraction, valid_tiles)
                )
                depth_labels.append(f"{depth_start}-{depth_end}")

            aggregate_metrics = self._aggregate_metric_dicts(per_depth_metrics)
            self._log_readability_metrics(epoch, aggregate_metrics, per_depth_metrics, depth_labels)

            if getattr(self.c.tra, "eval_aggregate", True):
                # size of the train portion in tile units + orientation of the split line
                if getattr(self, "split_axis", "x") == "y":
                    train_split_n = (self.train_range[1] - self.train_range[0]) // self.c.data.tile_size
                    split_axis = "y"
                else:
                    train_split_n = (self.train_range[1] - self.train_range[0]) // self.c.data.tile_size
                    split_axis = "x"
                fig = self._create_aggregate_eval_figure(all_pred_data, train_split_n, label_binary, split_axis)
                self.writer.add_figure('Evaluation/Aggregated', fig, epoch)
                plt.close(fig)

            # ---- voxel map visualization (v13_mil only) ----
            # log a grid of representative tiles showing WHERE the model fires,
            # not just WHETHER it fires. only runs when the model exposes last_voxel_map
            # (i.e. arch=v13_mil). adds 'VoxelMap/InkTiles' and 'VoxelMap/BlankTiles'
            # to TensorBoard under the Images tab.
            self._log_voxel_maps(epoch, model)

    def _log_voxel_maps(self, epoch, model):
        """log per-tile voxel maps for v13_mil to TensorBoard Images tab.

        for each tile: shows (left) the depth-mean raw scan slice and (right) the
        depth-max of the model's per-voxel logit map, both normalized to [0,1].
        logged under VoxelMap/InkTiles and VoxelMap/BlankTiles.

        what to look for:
          INK TILES — the right panel should show bright spots/streaks at ink stroke
          positions: thin horizontal ribbons for letter strokes, NOT diffuse blobs.
          BLANK TILES — right panel should be uniformly dark (low activation everywhere).
          if ink and blank panels look identical (diffuse/random), the model has not
          learned spatially localized ink — it's still doing coarse intensity detection.
        """
        if not hasattr(model, 'last_voxel_map') or model.last_voxel_map is None:
            return   # non-MIL architecture; skip silently
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            tile   = self.c.data.tile_size
            depth  = self.c.data.depth
            z_start = self.c.data.d_start
            device = self.c.device
            z_range = (self.c.data.d_start, self.c.data.d_end)
            if getattr(self, "split_axis", "x") == "y":
                tr_y, tr_x = self.train_range, self.shared_range
            else:
                tr_y, tr_x = self.shared_range, self.train_range
            all_coords  = self._gen_tile_coords(z_range, tr_y, tr_x, self.mask)
            ink_tiles   = [(z, y, x) for z, y, x in all_coords
                           if self.labels[tr_y[0] + y: tr_y[0] + y + tile,
                                          tr_x[0] + x: tr_x[0] + x + tile].any()]
            blank_tiles = [(z, y, x) for z, y, x in all_coords
                           if not self.labels[tr_y[0] + y: tr_y[0] + y + tile,
                                              tr_x[0] + x: tr_x[0] + x + tile].any()]
            n_show = 6
            rng = np.random.default_rng(epoch)
            ink_sample   = [ink_tiles[i]   for i in rng.choice(len(ink_tiles),   min(n_show, len(ink_tiles)),   replace=False)] if ink_tiles   else []
            blank_sample = [blank_tiles[i] for i in rng.choice(len(blank_tiles), min(n_show, len(blank_tiles)), replace=False)] if blank_tiles else []
            g_mean, g_std = self.global_mean, self.global_std
            g_min,  g_max = self.global_min,  self.global_max

            def _fetch(d_off, y_off, x_off):
                y = tr_y[0] + y_off; x = tr_x[0] + x_off; z = z_start + d_off
                if z + depth > self.volume.shape[0]: return None, None
                blk = np.array(self.volume[z:z + depth, y:y + tile, x:x + tile]).astype(np.float32)
                blk = np.clip((blk - g_mean) / (g_std + 1e-8), -5, 5)
                blk = np.clip((blk - g_min) / (g_max - g_min + 1e-8), 0, 1)
                t = torch.from_numpy(blk).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad(): model(t)
                vmap = torch.sigmoid(model.last_voxel_map[0, 0]).max(0).values.cpu().numpy()
                return blk.mean(0), vmap   # (H,W) depth-mean scan; (H',W') depth-max logit

            def _make_grid(samples, title):
                if not samples: return None
                n = len(samples)
                fig, axes = plt.subplots(n, 2, figsize=(4, n * 2))
                if n == 1: axes = [axes]
                for row, (d_off, y_off, x_off) in enumerate(samples):
                    raw, vmap = _fetch(d_off, y_off, x_off)
                    if raw is None: continue
                    axes[row][0].imshow(raw,  cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    axes[row][1].imshow(vmap, cmap='hot',  vmin=0, vmax=1, interpolation='nearest')
                    axes[row][0].axis('off'); axes[row][1].axis('off')
                axes[0][0].set_title('scan (depth-mean)', fontsize=7)
                axes[0][1].set_title('voxel map (depth-max sigmoid)', fontsize=7)
                fig.suptitle(title, fontsize=8); plt.tight_layout()
                return fig

            for tag, samples in [('VoxelMap/InkTiles', ink_sample), ('VoxelMap/BlankTiles', blank_sample)]:
                fig = _make_grid(samples, f'{tag.split("/")[1]} — epoch {epoch}')
                if fig is not None:
                    self.writer.add_figure(tag, fig, epoch)
                    plt.close(fig)
        except Exception as e:
            print(f"[voxel map logging] skipped: {e}")

    def _run_and_log_hard_mining_evaluation(self, current_epoch, model):
        """evaluate previously mined files and log metrics"""
        if not getattr(self.c.hm, "enabled", True):
            return
        print("Starting hard-mining file evaluation...")
        try:
            hm_dir = self._hard_mining_dir()
            if not os.path.isdir(hm_dir):
                print("No hard-mining directory found")
                return

            hm_files = [f for f in os.listdir(hm_dir) if re.match(r'hard_mining_epoch_\d+\.jsonl', f)]
            if not hm_files:
                print("No hard-mining files found to evaluate.")
                return

            for hm_file in sorted(hm_files):
                m = re.search(r'(\d+)', hm_file)
                if not m:
                    continue
                source_epoch = int(m.group(1))
                if source_epoch > current_epoch:
                    print(f"Skipping future mining file: {hm_file}")
                    continue

                file_path = os.path.join(hm_dir, hm_file)
                print(f"Evaluating hard-mining file: {hm_file}")

                metrics = self._evaluate_hard_mining_file(model, file_path)

                if metrics:
                    self._log_hard_mining_metrics(metrics, current_epoch, source_epoch)
                else:
                    print(f"Skipping logging for {hm_file} due to no valid samples or error.")
        except Exception as e:
            print(f"[ERROR] Failed during hard-mining evaluation: {e}")

    def _evaluate_hard_mining_file(self, model, file_path):
        """run inference on samples from a hard mining file and calculate metrics"""
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "_type" not in data:
                        samples.append(data)
                except json.JSONDecodeError:
                    continue

        if not samples:
            return None

        all_labels = []
        all_scores = []

        device = self.c.device
        tile = self.c.data.tile_size
        bs = self.c.dl.batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(samples), bs), desc=f"Eval HM {os.path.basename(file_path)}", leave=False):
                b_samp = samples[i:i + bs]
                b_blocks = []
                b_labels = []

                for s in b_samp:
                    z, y, x, lbl = s['z'], s['y'], s['x'], s['label']

                    if z + self.c.data.depth > self.volume.shape[0]:
                        continue

                    blk = np.array(self.volume[z:z + self.c.data.depth, y:y + tile, x:x + tile]).astype(np.float32)

                    blk = (blk - self.global_mean) / self.global_std

                    m_tile = self.mask[y:y + tile, x:x + tile]
                    m_bin = (m_tile > 0).astype(np.uint8)
                    m_exp = np.broadcast_to(np.expand_dims(m_bin, axis=0), blk.shape)
                    blk[m_exp == 0] = 0

                    blk = (blk - self.global_min) / (self.global_max - self.global_min + 1e-12)
                    blk = np.clip(blk, 0, 1)

                    b_blocks.append(blk)
                    b_labels.append(lbl)

                if not b_blocks:
                    continue

                bt = torch.from_numpy(np.stack(b_blocks)).float().unsqueeze(1).to(device)
                logits = model(bt)
                if logits.dim() == 4:
                    logits = logits.flatten(1).max(dim=1, keepdim=True).values
                scores = torch.sigmoid(logits).cpu().numpy().flatten()

                all_scores.extend(scores)
                all_labels.extend(b_labels)

        if not all_labels:
            return None

        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        y_pred = (y_scores > 0.5).astype(int)

        return calculate_metrics(y_true, y_pred, y_scores)

    def _log_hard_mining_metrics(self, metrics, current_epoch, source_epoch):
        """log metrics for a mined file with source epoch tag"""
        tag = f"HM_{source_epoch}"

        self.writer.add_scalar(f"G_M/Loss/{tag}", metrics.get('loss', 0), current_epoch)
        self.writer.add_scalar(f"G_M/Acc/{tag}", metrics['accuracy'], current_epoch)
        self.writer.add_scalar(f"P_M/Precision/{tag}", metrics['precision'], current_epoch)
        self.writer.add_scalar(f"P_M/Recall/{tag}", metrics['recall'], current_epoch)
        self.writer.add_scalar(f"P_M/F1_Score/{tag}", metrics['f1'], current_epoch)
        self.writer.add_scalar(f"P_M/Specificity/{tag}", metrics['specificity'], current_epoch)
        self.writer.add_scalar(f"AUC/ROC_AUC/{tag}", metrics['roc_auc'], current_epoch)
        self.writer.add_scalar(f"AUC/PR_AUC/{tag}", metrics['pr_auc'], current_epoch)
        print(f"Logged metrics for HM from epoch {source_epoch} at eval epoch {current_epoch}. F1: {metrics['f1']:.4f}")

    def add_test_figures(self, epoch, model):
        """add test figures for test scroll and the active secondary target (scroll2 or scroll4)"""
        print("Starting test figure generation...")
        model.eval()

        # cost-control: when test_scroll2_only is set, render ONLY the goal scroll2
        # fragment. this skips the very expensive full training-scroll "Test" figure
        # (e.g. the big fragment is ~2.3M tile reads / several hours) so end-of-training
        # test inference stays affordable across a campaign.
        scroll2_only = bool(getattr(self.c.data, "test_scroll2_only", False))

        if not scroll2_only:
            try:
                self._add_single_test_figure(epoch, model, self.test_volume, self.test_mask, self.test_y_range, self.test_x_range, self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max, "Test")
            except Exception as e:
                print(f"[ERROR] Test (training-scroll) figure failed: {e}")
                import traceback; traceback.print_exc()

        if (not scroll2_only) and self.c.data.test_on_scroll4:
            if self.scroll4_volume is not None:
                try:
                    self._add_single_test_figure(epoch, model, self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range, self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max, "Scroll4")
                except Exception as e:
                    print(f"[ERROR] Scroll4 test figure failed: {e}")
                    import traceback; traceback.print_exc()
        else:
            if self.scroll2_volume is not None:
                try:
                    self._add_single_test_figure(epoch, model, self.scroll2_volume, self.scroll2_mask, self.scroll2_y_range, self.scroll2_x_range, self.scroll2_global_mean, self.scroll2_global_std, self.scroll2_global_min, self.scroll2_global_max, "Scroll2")
                except Exception as e:
                    print(f"[ERROR] Scroll2 test figure failed: {e}")
                    import traceback; traceback.print_exc()

        # scroll3 goal-scroll: its OWN separate figure, always rendered alongside scroll2 when
        # available. skipped silently if scroll3 data was not loaded (e.g. still downloading).
        if self.scroll3_volume is not None:
            try:
                self._add_single_test_figure(epoch, model, self.scroll3_volume, self.scroll3_mask, self.scroll3_y_range, self.scroll3_x_range, self.scroll3_global_mean, self.scroll3_global_std, self.scroll3_global_min, self.scroll3_global_max, "Scroll3")
            except Exception as e:
                print(f"[ERROR] Scroll3 test figure failed: {e}")
                import traceback; traceback.print_exc()

    def _add_single_test_figure(self, epoch, model, vol, mask, y_range, x_range, g_mean, g_std, g_min, g_max, name):
        """predict per depth and create a mosaic figure for a test dataset"""
        z_range = (0, vol.shape[0])

        coords = self._gen_tile_coords(z_range, y_range, x_range, mask)
        grp = group_by_depth(coords)
        depths = sorted(grp.keys())

        all_data = []

        for d_start in depths:
            b_coords = grp[d_start]
            pred = predict_tiles(
                self.c, model, vol, mask, b_coords, y_range, x_range,
                d_start, name, g_mean, g_std, g_min, g_max
            )
            d_end = d_start + self.c.data.depth
            all_data.append((pred, d_start, d_end))

        if all_data:
            fig = self._create_combined_test_figure(all_data, len(all_data), name)
            self.writer.add_figure(f'Test/{name}_All_Depth_Blocks', fig, epoch)
            plt.close(fig)

    def _create_evaluation_figure(self, pred_data, label_binary):
        """create evaluation figure for a single depth block"""
        full_pred, train_pred, d_start, d_end = pred_data

        fig, axes = plt.subplots(1, 2, figsize=(15, 9))

        ax_pred = axes[0]
        im1 = ax_pred.imshow(full_pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
        ax_pred.set_title(f'Predictions (Depth {d_start}-{d_end})', fontsize=9)

        split_pos = train_pred.shape[1] - 0.5
        ax_pred.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_pred.axis('off')

        ax_overlay = axes[1]
        ax_overlay.imshow(full_pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
        ax_overlay.set_title(f'Overlay (Depth {d_start}-{d_end})', fontsize=9)

        if label_binary is not None:
            overlay = np.zeros((*full_pred.shape, 4))
            h = min(label_binary.shape[0], overlay.shape[0])
            w = min(label_binary.shape[1], overlay.shape[1])
            overlay[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
            ax_overlay.imshow(overlay)

        ax_overlay.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_overlay.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def _create_aggregate_eval_figure(self, all_pred_data, train_split_n, label_binary, split_axis="x"):
        """n_blocks-row × 2-col figure: left col = predictions, right col = overlay with inklabels.

        figure size adapts to the map's tile dimensions and aspect ratio so the image
        is never distorted regardless of scroll geometry. split_axis controls whether the
        train/valid divider is drawn as a vertical (x-split) or horizontal (y-split) line.
        """
        n_blocks = len(all_pred_data)
        if n_blocks == 0:
            return None

        # derive panel size from the actual tile-unit dimensions of the first map
        sample_pred = all_pred_data[0][0]
        h_tiles, w_tiles = sample_pred.shape
        aspect = w_tiles / max(h_tiles, 1)      # width / height of one panel

        # target a panel width of ~0.06 in per tile column, capped [6, 16] in
        panel_w = max(6.0, min(16.0, w_tiles * 0.06))
        panel_h = max(2.0, min(12.0, panel_w / aspect))
        # recompute panel_w in case panel_h was clamped
        panel_w = panel_h * aspect

        fig_w = panel_w * 2 + 0.3           # two columns + small gap
        fig_h = panel_h * n_blocks + 0.4    # one row per depth block + title margin

        fig, axes = plt.subplots(n_blocks, 2, figsize=(fig_w, fig_h),
                                 squeeze=False)

        split_pos = train_split_n - 0.5

        def _draw_split(ax):
            if split_axis == "y":
                ax.axhline(y=split_pos, color='red', linestyle='--', linewidth=0.8)
            else:
                ax.axvline(x=split_pos, color='red', linestyle='--', linewidth=0.8)

        for row, (full_pred, train_pred, d_start, d_end) in enumerate(all_pred_data):
            # left: raw prediction
            ax_pred = axes[row, 0]
            ax_pred.imshow(full_pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
            ax_pred.set_title(f'Depth {d_start}-{d_end}', fontsize=8)
            _draw_split(ax_pred)
            ax_pred.axis('off')

            # right: same prediction + inklabel overlay
            ax_ov = axes[row, 1]
            ax_ov.imshow(full_pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
            ax_ov.set_title(f'Overlay {d_start}-{d_end}', fontsize=8)
            if label_binary is not None:
                ov = np.zeros((*full_pred.shape, 4))
                h = min(label_binary.shape[0], ov.shape[0])
                w = min(label_binary.shape[1], ov.shape[1])
                ov[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
                ax_ov.imshow(ov)
            _draw_split(ax_ov)
            ax_ov.axis('off')

        plt.subplots_adjust(wspace=0.04, hspace=0.12,
                            left=0.01, right=0.99,
                            top=0.98, bottom=0.01)
        return fig

    def _create_combined_test_figure(self, all_data, n_blocks, test_type):
        """create combined test figure showing prediction mosaics.

        panel height is derived from the actual tile-grid aspect of the maps so that
        very wide/flat scrolls (e.g. scroll2) do not waste large vertical whitespace
        bands around each thin strip. aspect='equal' keeps the image undistorted; the
        cell is sized to match the image so there is little leftover space."""
        cols = 2
        rows = (n_blocks + cols - 1) // cols

        # aspect = width / height of one prediction map, in tile units
        sample = all_data[0][0]
        h_tiles, w_tiles = sample.shape
        aspect = w_tiles / max(h_tiles, 1)

        panel_w = 6.0                                  # inches per column
        # match cell height to the image aspect so whitespace is minimal; clamp so
        # extreme aspect ratios stay legible (tall training scroll vs flat scroll2)
        panel_h = max(1.3, min(7.0, panel_w / max(aspect, 1e-6)))

        fig_w = panel_w * cols
        fig_h = panel_h * rows + 0.4                   # small margin for titles

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (pred, d_start, d_end) in enumerate(all_data):
            ax = axes[idx // cols, idx % cols]
            im = ax.imshow(pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
            ax.set_title(f'Depth Block {d_start}-{d_end}', fontsize=9)
            ax.axis('off')

        for idx in range(len(all_data), rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.18, left=0.03, right=0.97, top=0.97, bottom=0.03)
        return fig

    def _log_readability_metrics(self, epoch, aggregate_metrics, per_depth_metrics, depth_labels):
        """log readability-aligned scalar and figure summaries"""
        if not aggregate_metrics:
            return

        scalar_tags = {
            "R_M/LocalContrast":            aggregate_metrics.get("local_contrast", np.nan),
            "R_M/LocalRanking":             aggregate_metrics.get("local_ranking", np.nan),
            "R_M/RecallAt1PctFPR":          aggregate_metrics.get("recall_at_1pct_fpr", np.nan),
            "R_M/PartialAUCAt1PctFPR":      aggregate_metrics.get("partial_auc_at_1pct_fpr", np.nan),
            "R_M/RecallAt5PctFPR":          aggregate_metrics.get("recall_at_5pct_fpr", np.nan),
            "R_M/PartialAUCAt5PctFPR":      aggregate_metrics.get("partial_auc_at_5pct_fpr", np.nan),
            "R_M/CoverageRecall":           aggregate_metrics.get("coverage_recall", np.nan),
            "R_M/TopKPrecision":            aggregate_metrics.get("topk_precision", np.nan),
            "R_M/InkFractionSpearman":      aggregate_metrics.get("ink_fraction_corr_spearman", np.nan),
            "R_M/SpillRatio":               aggregate_metrics.get("spill_ratio", np.nan),
            "R_M/ComponentCount":           aggregate_metrics.get("component_count", np.nan),
            "R_M/MeanComponentSize":        aggregate_metrics.get("mean_component_size", np.nan),
            "R_M/ReadabilityComposite":     aggregate_metrics.get("readability_composite", np.nan),
        }

        for tag, value in scalar_tags.items():
            if np.isfinite(value):
                self.writer.add_scalar(tag, float(value), epoch)

        fig = self._create_readability_summary_figure(aggregate_metrics, per_depth_metrics, depth_labels)
        self.writer.add_figure("Readability/Summary", fig, epoch)
        plt.close(fig)

        fig = self._create_readability_compass_figure(aggregate_metrics, per_depth_metrics, depth_labels)
        self.writer.add_figure("Readability/Compass", fig, epoch)
        plt.close(fig)

    def _readability_compass_values(self, metrics):
        """map raw readability metrics into 0..1 values used by compass plot"""
        local_contrast = np.clip(np.nan_to_num(metrics.get("local_contrast", np.nan), nan=0.0), 0.0, 1.0)
        local_ranking = np.clip(np.nan_to_num(metrics.get("local_ranking", np.nan), nan=0.0), 0.0, 1.0)
        local_contrast = np.clip(np.nan_to_num(metrics.get("local_contrast", np.nan), nan=0.0), 0.0, 1.0)
        local_ranking  = np.clip(np.nan_to_num(metrics.get("local_ranking", np.nan), nan=0.0), 0.0, 1.0)
        recall_5pct    = np.clip(np.nan_to_num(metrics.get("recall_at_5pct_fpr", np.nan), nan=0.0), 0.0, 1.0)
        pauc_5pct      = np.clip(np.nan_to_num(metrics.get("partial_auc_at_5pct_fpr", np.nan), nan=0.0), 0.0, 1.0)
        coverage       = np.clip(np.nan_to_num(metrics.get("coverage_recall", np.nan), nan=0.0), 0.0, 1.0)
        spearman       = np.clip((np.nan_to_num(metrics.get("ink_fraction_corr_spearman", np.nan), nan=-1.0) + 1.0) / 2.0, 0.0, 1.0)
        coherence      = np.clip(np.nan_to_num(metrics.get("mean_component_size", np.nan), nan=0.0) / 20.0, 0.0, 1.0)
        composite      = np.clip(np.nan_to_num(metrics.get("readability_composite", np.nan), nan=0.0), 0.0, 1.0)
        return [
            float(local_contrast),
            float(local_ranking),
            float(recall_5pct),
            float(pauc_5pct),
            float(coverage),
            float(spearman),
            float(coherence),
            float(composite),
        ]

    def _readability_good_targets(self):
        """heuristic target values used as visual reference markers (updated for coverage focus)"""
        return {
            "local_contrast":             0.15,
            "local_ranking":              0.70,
            "recall_at_5pct_fpr":         0.50,
            "partial_auc_at_5pct_fpr":    0.40,
            "coverage_recall":            0.40,
            "ink_fraction_corr_spearman": 0.40,
            "mean_component_size":        8.0,   # raw tile units; normalised by /20 in compass
            "readability_composite":      0.60,
        }

    def _create_readability_compass_figure(self, aggregate_metrics, per_depth_metrics, depth_labels):
        """create a readability-focused radar chart using normalized readability terms"""
        categories = [
            "local contrast",
            "local ranking",
            "recall@5%fpr",
            "pauc@5%fpr",
            "coverage@0.3",
            "spearman",
            "coherence",
            "composite",
        ]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"projection": "polar"})

        series = [
            ("aggregate", self._readability_compass_values(aggregate_metrics), "teal"),
        ]

        best_idx = None
        best_value = float("-inf")
        for idx, metrics in enumerate(per_depth_metrics):
            value = float(np.nan_to_num(metrics.get("readability_composite", np.nan), nan=-1.0))
            if value > best_value:
                best_value = value
                best_idx = idx

        if best_idx is not None:
            best_label = "best depth"
            if best_idx < len(depth_labels):
                best_label = f"best depth ({depth_labels[best_idx]})"
            series.append((best_label, self._readability_compass_values(per_depth_metrics[best_idx]), "darkorange"))

        self._plot_radar_chart(
            ax,
            categories,
            series,
            title="Readability Compass",
            ylim=(0, 1),
        )

        good_targets = self._readability_good_targets()
        good_values = self._readability_compass_values(good_targets)
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        good_values_c = good_values + good_values[:1]
        ax.plot(angles, good_values_c, color="red", marker="o", linestyle="None", markersize=3, label="good target")
        for ang, val in zip(angles[:-1], good_values):
            ax.text(ang, min(0.99, val + 0.04), f"{val:.2f}", color="red", fontsize=7, ha="center", va="bottom")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        return fig

    def _create_readability_summary_figure(self, aggregate_metrics, per_depth_metrics, depth_labels):
        """create a combined readability dashboard figure"""
        metric_keys = [
            ("local_contrast",            "local contrast"),
            ("local_ranking",             "ranking"),
            ("recall_at_5pct_fpr",        "recall@5%fpr"),
            ("partial_auc_at_5pct_fpr",   "pauc@5%fpr"),
            ("coverage_recall",           "coverage@0.3"),
            ("ink_fraction_corr_spearman","fraction corr"),
            ("mean_component_size",       "coherence"),
            ("readability_composite",     "composite"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        agg_labels = [label for _, label in metric_keys]
        agg_values = []
        for key, _ in metric_keys:
            value = float(np.nan_to_num(aggregate_metrics.get(key, np.nan), nan=0.0))
            # normalise coherence (mean_component_size) same way as composite: /20
            if key == "mean_component_size":
                value = float(np.clip(value / 20.0, 0.0, 1.0))
            agg_values.append(value)

        good_targets = self._readability_good_targets()
        good_values = []
        for key, _ in metric_keys:
            target = float(np.nan_to_num(good_targets.get(key, np.nan), nan=0.0))
            if key == "mean_component_size":
                target = float(np.clip(target / 20.0, 0.0, 1.0))
            good_values.append(float(np.clip(target, 0.0, 1.0)))

        axes[0].bar(np.arange(len(agg_values)), agg_values, color="steelblue", alpha=0.85)
        axes[0].scatter(np.arange(len(good_values)), good_values, color="red", marker="o", s=20, zorder=4, label="good target")
        axes[0].set_xticks(np.arange(len(agg_values)))
        axes[0].set_xticklabels(agg_labels, rotation=35, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("aggregate readability metrics (coverage + coherence focused)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        for idx, value in enumerate(agg_values):
            axes[0].annotate(f"{value:.3f}", (idx, value), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        for idx, value in enumerate(good_values):
            axes[0].annotate(f"{value:.2f}", (idx, value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color="red")

        # right: per-depth annotated heatmap (skipped when no per-depth data provided)
        raw_matrix = np.array([
            [float(metric.get(key, np.nan)) for key, _ in metric_keys]
            for metric in per_depth_metrics
        ], dtype=np.float32)

        if raw_matrix.ndim < 2 or raw_matrix.shape[0] == 0:
            # no per-depth data — leave the right panel blank
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "no per-depth data", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=9, color="gray")
        else:
            norm_matrix = np.zeros_like(raw_matrix)
            for col in range(raw_matrix.shape[1]):
                col_vals = raw_matrix[:, col]
                finite_mask = np.isfinite(col_vals)
                if not np.any(finite_mask):
                    continue
                vmin = float(np.min(col_vals[finite_mask]))
                vmax = float(np.max(col_vals[finite_mask]))
                if abs(vmax - vmin) < 1e-12:
                    norm_matrix[finite_mask, col] = 0.5
                else:
                    norm_matrix[finite_mask, col] = (col_vals[finite_mask] - vmin) / (vmax - vmin)

            annot = np.empty(raw_matrix.shape, dtype=object)
            for yi in range(raw_matrix.shape[0]):
                for xi in range(raw_matrix.shape[1]):
                    annot[yi, xi] = "nan" if not np.isfinite(raw_matrix[yi, xi]) else f"{raw_matrix[yi, xi]:.3f}"

            sns.heatmap(
                norm_matrix,
                annot=annot,
                fmt="",
                cmap="viridis",
                xticklabels=[label for _, label in metric_keys],
                yticklabels=depth_labels,
                ax=axes[1],
                cbar=False,
            )
            axes[1].set_title("per-depth readability summary\ncolumn-normalized colors with raw annotations")
            axes[1].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        return fig

    def add_probe_region_figures(self, epoch, model):
        """log fixed readability probe regions as image panels and scalar scorecards"""
        print("Logging probe-region figures...")
        model.eval()

        probe_data_list = []
        for spec in self.probe_specs:
            probe_data = self._collect_probe_region_predictions(model, spec)
            if probe_data is None:
                continue

            probe_data_list.append(probe_data)

            _, aggregate_metrics = self._create_probe_region_figure(probe_data)

            if aggregate_metrics:
                probe_tag = spec["tag"]
                for key, value in {
                    f"R_M/Probe/{probe_tag}/LocalContrast":        aggregate_metrics.get("local_contrast", np.nan),
                    f"R_M/Probe/{probe_tag}/CoverageRecall":       aggregate_metrics.get("coverage_recall", np.nan),
                    f"R_M/Probe/{probe_tag}/RecallAt5PctFPR":      aggregate_metrics.get("recall_at_5pct_fpr", np.nan),
                    f"R_M/Probe/{probe_tag}/ReadabilityComposite": aggregate_metrics.get("readability_composite", np.nan),
                }.items():
                    if np.isfinite(value):
                        self.writer.add_scalar(key, float(value), epoch)

        if probe_data_list:
            fig = self._create_combined_probe_depth_figure(probe_data_list)
            self.writer.add_figure("ProbeROIs/AllPatches_ByDepth", fig, epoch)
            plt.close(fig)

    def _collect_probe_region_predictions(self, model, spec):
        """prepare per-depth predictions and readability stats for one fixed probe region"""
        try:
            asset = self._get_segment_asset(spec["segment_id"])
        except Exception as e:
            print(f"[PROBE] Skipping {spec['tag']} due to asset load error: {e}")
            return None

        volume = asset["volume"]
        mask = asset["mask"]
        labels = asset["labels"]
        g_mean, g_std, g_min, g_max = asset["norm"]

        x0 = int(spec["x"])
        y0 = int(spec["y"])
        size = int(spec["size"])
        y1 = min(y0 + size, volume.shape[1])
        x1 = min(x0 + size, volume.shape[2])
        y_range = (y0, y1)
        x_range = (x0, x1)

        z_range = (self.c.data.d_start, self.c.data.d_end)
        coords = self._gen_tile_coords(z_range, y_range, x_range, mask)
        if not coords:
            print(f"[PROBE] No valid coords for {spec['tag']}")
            return None

        grouped = group_by_depth(coords)
        depth_offsets = sorted(grouped.keys())
        label_binary, label_fraction, valid_tiles = self._compute_tile_maps(labels, mask, y_range, x_range)

        depth_rows = []
        for d_off in depth_offsets:
            depth_start = self.c.data.d_start + d_off
            depth_end = depth_start + self.c.data.depth
            pred = predict_tiles(
                self.c,
                model,
                volume,
                mask,
                grouped[d_off],
                y_range,
                x_range,
                depth_start,
                spec["tag"],
                g_mean,
                g_std,
                g_min,
                g_max,
            )

            metrics = self._compute_readability_metrics(pred, label_binary, label_fraction, valid_tiles)
            depth_rows.append(
                {
                    "depth_start": depth_start,
                    "depth_end": depth_end,
                    "pred": pred,
                    "metrics": metrics,
                }
            )

        aggregate_metrics = self._aggregate_metric_dicts([row["metrics"] for row in depth_rows])
        return {
            "spec": spec,
            "label_binary": label_binary,
            "depth_rows": depth_rows,
            "aggregate_metrics": aggregate_metrics,
            "x0": x0,
            "y0": y0,
            "size": size,
        }

    def _create_probe_region_figure(self, probe_data):
        """predict a fixed roi across depth blocks and render prediction plus label overlay"""
        spec = probe_data["spec"]
        label_binary = probe_data["label_binary"]
        depth_rows = probe_data["depth_rows"]
        aggregate_metrics = probe_data["aggregate_metrics"]

        if not depth_rows:
            return None, None

        fig, axes = plt.subplots(len(depth_rows), 2, figsize=(10, max(4, 4 * len(depth_rows))))
        axes = np.array(axes).reshape(len(depth_rows), 2)

        for idx, row in enumerate(depth_rows):
            depth_start = row["depth_start"]
            depth_end = row["depth_end"]
            pred = row["pred"]
            metrics = row["metrics"]

            axes[idx, 0].imshow(pred, cmap="inferno", vmin=0, vmax=1, aspect="equal")
            axes[idx, 0].set_title(f"pred {depth_start}-{depth_end}", fontsize=9)
            axes[idx, 0].axis("off")

            overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
            h = min(label_binary.shape[0], pred.shape[0])
            w = min(label_binary.shape[1], pred.shape[1])
            overlay[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
            axes[idx, 1].imshow(pred, cmap="inferno", vmin=0, vmax=1, aspect="equal")
            axes[idx, 1].imshow(overlay)
            axes[idx, 1].set_title(
                f"overlay {depth_start}-{depth_end}\nC={np.nan_to_num(metrics['local_contrast'], nan=0.0):.3f} P@K={np.nan_to_num(metrics['topk_precision'], nan=0.0):.3f}",
                fontsize=9,
            )
            axes[idx, 1].axis("off")

        x0 = probe_data["x0"]
        y0 = probe_data["y0"]
        size = probe_data["size"]
        fig.suptitle(
            f"{spec['title']} | seg={spec['segment_id']} | x={x0}, y={y0}, size={size} | composite={np.nan_to_num(aggregate_metrics.get('readability_composite', np.nan), nan=0.0):.3f}",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, aggregate_metrics

    def _create_combined_probe_depth_figure(self, probe_data_list):
        """render easy/hard/scroll4 probes side-by-side per depth with pred and overlay"""
        depth_values = sorted({
            row["depth_start"]
            for probe_data in probe_data_list
            for row in probe_data["depth_rows"]
        })

        rows = max(1, len(depth_values))
        cols = 2 * len(probe_data_list)
        fig_w = max(14, 4 * len(probe_data_list))
        fig_h = max(4, 3 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(rows, cols)

        for row_idx, depth_start in enumerate(depth_values):
            for probe_idx, probe_data in enumerate(probe_data_list):
                spec = probe_data["spec"]
                label_binary = probe_data["label_binary"]
                by_depth = {row["depth_start"]: row for row in probe_data["depth_rows"]}
                pred_ax = axes[row_idx, 2 * probe_idx]
                ov_ax = axes[row_idx, 2 * probe_idx + 1]

                if depth_start not in by_depth:
                    pred_ax.axis("off")
                    ov_ax.axis("off")
                    continue

                row = by_depth[depth_start]
                depth_end = row["depth_end"]
                pred = row["pred"]
                metrics = row["metrics"]

                pred_ax.imshow(pred, cmap="inferno", vmin=0, vmax=1, aspect="equal")
                pred_ax.axis("off")

                ov_ax.imshow(pred, cmap="inferno", vmin=0, vmax=1, aspect="equal")
                overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
                h = min(label_binary.shape[0], pred.shape[0])
                w = min(label_binary.shape[1], pred.shape[1])
                overlay[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
                ov_ax.imshow(overlay)
                ov_ax.axis("off")

                if row_idx == 0:
                    pred_ax.set_title(f"{spec['tag']} pred", fontsize=9)
                    ov_ax.set_title(f"{spec['tag']} overlay", fontsize=9)

                if probe_idx == 0:
                    pred_ax.text(
                        -0.03,
                        0.5,
                        f"{depth_start}-{depth_end}",
                        transform=pred_ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=8,
                    )

                ov_ax.text(
                    0.02,
                    0.02,
                    f"C {np.nan_to_num(metrics['local_contrast'], nan=0.0):.2f} | P@K {np.nan_to_num(metrics['topk_precision'], nan=0.0):.2f}",
                    transform=ov_ax.transAxes,
                    fontsize=7,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
                )

        fig.suptitle("Probe patches by depth: easy | medium | hard | scroll4", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def log_model_graph(self, model, example_input):
        """log the model graph"""
        self.writer.add_graph(model, example_input)

    def log_activation_maps(self, activations, epoch):
        """log activation maps with safe handling of shapes"""
        for layer, act in activations.items():
            if act.dim() == 5:
                act4 = act.mean(dim=2)
                self.writer.add_images(f"Activations/{layer.__class__.__name__}", act4, epoch, dataformats="NCHW")
            elif act.dim() == 2:
                act_r = act.unsqueeze(1).unsqueeze(-1)
                self.writer.add_images(f"Activations/{layer.__class__.__name__}", act_r, epoch, dataformats="NCHW")
            else:
                raise ValueError(f"Unexpected activation map dimensions: {act.shape}")

    def log_weight_histograms(self, model, epoch):
        """log weight and gradient histograms with guards"""
        if getattr(self, "_disable_histogram_logging", False):
            return

        for name, p in model.named_parameters():
            if p.requires_grad:
                data = p.data.detach().cpu().numpy()
                if data.size > 0 and not np.isnan(data).all():
                    try:
                        self.writer.add_histogram(f"Weights/{name}", data, epoch)
                    except Exception as e:
                        print(f"[WARNING] Disabling histogram logging (Weights/{name}) due to compatibility error: {e}")
                        self._disable_histogram_logging = True
                        return

                if p.grad is not None:
                    g = p.grad.detach().cpu().numpy()
                    if g.size > 0 and not np.isnan(g).all() and np.abs(g).sum() > 0:
                        try:
                            self.writer.add_histogram(f"Gradients/{name}", g, epoch)
                        except Exception as e:
                            print(f"[WARNING] Disabling histogram logging (Gradients/{name}) due to compatibility error: {e}")
                            self._disable_histogram_logging = True
                            return

    def _create_hard_examples_overlay(self, mining_path):
        """
        downsampled tile grid overlay for mined examples
        base is grayscale eroded labels converted to rgb then downsampled
        negatives add blue intensity equal to score and positives add red intensity equal to one minus score
        alpha blend per tile with fixed alpha
        """
        if not os.path.exists(mining_path):
            return None

        seg_id = self.scroll1_id
        label_path = f"./eroded_inklabels/{seg_id}.png"
        if not os.path.exists(label_path):
            return None

        label_gray = imread_gray(label_path)
        if label_gray is None:
            return None

        if seg_id == 20230827161847:
            y0, y1 = 200, 5600
            x0, x1 = 1000, 4600
        else:
            y0, y1 = 0, label_gray.shape[0]
            x0, x1 = 0, label_gray.shape[1]

        crop = label_gray[y0:y1, x0:x1]
        tile = self.c.data.tile_size
        Ht = crop.shape[0] // tile
        Wt = crop.shape[1] // tile
        if Ht <= 0 or Wt <= 0:
            return None

        base_small = crop[:Ht * tile:tile, :Wt * tile:tile].astype(np.float32)
        if base_small.shape != (Ht, Wt):
            return None

        # build base in rgb to avoid channel confusion
        base_small_rgb = np.stack([base_small, base_small, base_small], axis=-1)
        canvas_tmpl = base_small_rgb.copy()

        by_z = {}
        with open(mining_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_type"):
                    continue
                z = obj.get("z")
                lbl = obj.get("label")
                if z is None or lbl not in (0, 1):
                    continue
                by_z.setdefault(z, {"neg": [], "pos": []})
                if lbl == 0:
                    by_z[z]["neg"].append(obj)
                else:
                    by_z[z]["pos"].append(obj)

        if not by_z:
            return None

        zs = sorted(by_z.keys())
        cols = 2
        rows = (len(zs) + cols - 1) // cols
        fig_w = 10
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, rows * 4))
        axes = np.array(axes).reshape(rows, cols)
        alpha = 0.45

        for idx, z in enumerate(zs):
            ax = axes[idx // cols, idx % cols]
            canvas = canvas_tmpl.copy()  # rgb float canvas

            # negatives: blue in rgb with intensity = score
            for rec in by_z[z]["neg"]:
                xg, yg = rec["x"], rec["y"]
                xr, yr = xg - x0, yg - y0
                if xr < 0 or yr < 0:
                    continue
                xi = xr // tile
                yi = yr // tile
                if not (0 <= xi < Wt and 0 <= yi < Ht):
                    continue
                score = float(rec.get("score", 0.0))
                b_val = 255.0 * max(0.0, min(1.0, score))
                orig = canvas[yi, xi]
                blend_rgb = np.array([0.0, 0.0, b_val], dtype=np.float32)
                canvas[yi, xi] = alpha * blend_rgb + (1 - alpha) * orig

            # positives: red in rgb with intensity = 1 - score
            for rec in by_z[z]["pos"]:
                xg, yg = rec["x"], rec["y"]
                xr, yr = xg - x0, yg - y0
                if xr < 0 or yr < 0:
                    continue
                xi = xr // tile
                yi = yr // tile
                if not (0 <= xi < Wt and 0 <= yi < Ht):
                    continue
                score = float(rec.get("score", 0.0))
                r_val = 255.0 * max(0.0, min(1.0, 1.0 - score))
                orig = canvas[yi, xi]
                blend_rgb = np.array([r_val, 0.0, 0.0], dtype=np.float32)
                canvas[yi, xi] = alpha * blend_rgb + (1 - alpha) * orig

            ax.imshow(canvas.astype(np.uint8), interpolation='nearest')
            ax.set_title(f"z={z}\nN={len(by_z[z]['neg'])} P={len(by_z[z]['pos'])}", fontsize=8)
            ax.axis("off")

        for j in range(len(zs), rows * cols):
            axes[j // cols, j % cols].axis("off")

        fig.suptitle("Hard Examples (Per Z, Tile Grid Overlay)", fontsize=12)
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def log_hyperparameters(self, params, pos_weight):
        """log run hyperparameters"""
        self.writer.add_scalar("Hyperparameters/Tile Size", self.c.data.tile_size)
        self.writer.add_scalar("Hyperparameters/Depth", self.c.data.depth)
        self.writer.add_scalar("Hyperparameters/Batch Size", self.c.dl.batch_size)
        self.writer.add_scalar("Hyperparameters/Num Workers", self.c.dl.num_workers)
        self.writer.add_scalar("Hyperparameters/Learning Rate", self.c.tra.lr)
        self.writer.add_scalar("Hyperparameters/Weight Decay", self.c.tra.weight_decay)
        self.writer.add_scalar("Hyperparameters/L1 Lambda", self.c.tra.l1_lambda)
        self.writer.add_scalar("Hyperparameters/Conv1 Dropout", self.c.model.conv1_drop)
        self.writer.add_scalar("Hyperparameters/Conv2 Dropout", self.c.model.conv2_drop)
        self.writer.add_scalar("Hyperparameters/FC1 Dropout", self.c.model.fc1_drop)
        self.writer.add_scalar("Hyperparameters/FC2 Dropout", self.c.model.fc2_drop)
        self.writer.add_scalar("Hyperparameters/Max Grad Norm", self.c.tra.grad_norm)
        self.writer.add_scalar("Hyperparameters/Patience", self.c.tra.patience)
        self.writer.add_scalar("Hyperparameters/LR Scheduler Factor", self.c.tra.lr_decay)
        self.writer.add_scalar("Hyperparameters/Probe Interval", self.c.tra.probe_int)
        self.writer.add_scalar("Hyperparameters/Model Complexity", params)
        self.writer.add_scalar("Hyperparameters/Pos Weight", pos_weight)
        self.writer.add_scalar("Hyperparameters/HN Cutoff", self.c.hm.hn_cutoff)
        self.writer.add_scalar("Hyperparameters/HP Cutoff", self.c.hm.hp_cutoff)

    def close(self):
        """close the tensorboard writer"""
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_path}")

    def _debug_scroll4_ranges_once(self):
        """one time sanity checks for scroll4 alignment"""
        if getattr(self, "scroll4_volume", None) is None:
            return  # scroll4 not loaded (minimal setup) — nothing to sanity-check
        try:
            vol = self.scroll4_volume
            mask = self.scroll4_mask
            y_range = self.scroll4_y_range
            x_range = self.scroll4_x_range
            issues = []

            if mask.shape != (vol.shape[1], vol.shape[2]):
                issues.append(f"Mask shape {mask.shape} != volume spatial {(vol.shape[1], vol.shape[2])}")

            if not (0 <= y_range[0] < y_range[1] <= vol.shape[1]):  # type: ignore
                issues.append(f"Y range {y_range} out of bounds (0,{vol.shape[1]})")
            if not (0 <= x_range[0] < x_range[1] <= vol.shape[2]):  # type: ignore
                issues.append(f"X range {x_range} out of bounds (0,{vol.shape[2]})")

            tile = self.c.data.tile_size
            if (y_range[0] % tile != 0) or (x_range[0] % tile != 0):
                issues.append(f"Ranges not tile aligned: y_start%tile={y_range[0]%tile}, x_start%tile={x_range[0]%tile}")

            region_mask = mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            if region_mask.size == 0:
                issues.append("Region mask slice empty")
            else:
                nz_frac = (region_mask > 0).mean()
                print(f"[SCROLL4 DEBUG] Region mask non-zero fraction: {nz_frac:.4f}")
                if nz_frac == 0:
                    issues.append("Region mask entirely zero")

            if issues:
                print("[SCROLL4 DEBUG] Potential issues detected:")
                for iss in issues:
                    print(" -", iss)
            else:
                print("[SCROLL4 DEBUG] Scroll4 mask / range basic checks passed.")
        except Exception as e:
            print(f"[SCROLL4 DEBUG] Exception during range debug: {e}")

    def _load_existing_mined_keys(self):
        """scan all existing mining files and return a set of keys (z y x label) to prevent duplicates"""
        hm_dir = self._hard_mining_dir()
        keys = set()
        try:
            if not os.path.isdir(hm_dir):
                return keys
            for fname in os.listdir(hm_dir):
                if not re.match(r'hard_mining_epoch_\d+\.jsonl', fname):
                    continue
                fpath = os.path.join(hm_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if obj.get("_type"):
                                continue
                            z = obj.get("z"); y = obj.get("y"); x = obj.get("x"); lbl = obj.get("label")
                            if z is None or y is None or x is None or lbl is None:
                                continue
                            keys.add((int(z), int(y), int(x), int(lbl)))
                except Exception:
                    continue
        except Exception:
            pass
        return keys
