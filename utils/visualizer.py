from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing import _32Bit
import os
from typing import Any, Literal
from collections import defaultdict
import json
import re
import warnings

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
# YlGnBu_r: reversed so high prediction score (ink) = bright yellow, low = dark blue.
# _r suffix reverses standard YlGnBu (which runs pale-yellow -> dark-blue).
_ylgnbu_nan = _copy.copy(plt.cm.YlGnBu_r)
_ylgnbu_nan.set_bad(color=(0.45, 0.45, 0.45, 1.0))
# register_cmap was removed in matplotlib 3.9; use colormaps.register instead

_purples_nan = _copy.copy(plt.cm.Purples)
_purples_nan.set_bad(color=(0.45, 0.45, 0.45, 1.0))
try:
    import matplotlib as _mpl
    _mpl.colormaps.register(_inferno_nan, name='inferno_nan', force=True)
    _mpl.colormaps.register(_ylgnbu_nan, name='ylgnbu_nan', force=True)
    _mpl.colormaps.register(_purples_nan, name='purples_nan', force=True)
except Exception:
    plt.cm.inferno_nan = _inferno_nan  # fallback: attach directly
    plt.cm.ylgnbu_nan = _ylgnbu_nan
    plt.cm.purples_nan = _purples_nan

# single knob for all scroll prediction colormaps. high score (ink) = bright yellow.
SCROLL_CMAP = 'inferno_nan'

def group_by_depth(coords):
    """group tile coordinates by their depth offset"""
    grouped = defaultdict(list)
    for d_off, y_off, x_off in coords:
        grouped[d_off].append((d_off, y_off, x_off))
    return grouped


class _RegionCache:
    """serves a preloaded [:, ry0:ry1, rx0:rx1] crop of a (zarr or ndarray) volume from RAM.
    lets probe inference read its small fixed region ONCE and reuse it every epoch instead of
    hitting zarr each time. any read outside the cached box transparently falls back to the
    underlying volume. exposes .shape/.dtype so predict_tiles treats it like the real volume."""

    def __init__(self, vol, ry0, ry1, rx0, rx1):
        self.vol = vol
        self.shape = tuple(int(s) for s in vol.shape)
        self.dtype = getattr(vol, "dtype", None)
        self.ry0, self.ry1, self.rx0, self.rx1 = ry0, ry1, rx0, rx1
        self._buf = np.asarray(vol[:, ry0:ry1, rx0:rx1])   # (D, h, w) raw crop held in RAM

    def __getitem__(self, idx):
        zs, ys, xs = idx
        if (isinstance(ys, slice) and isinstance(xs, slice)
                and ys.start is not None and ys.stop is not None
                and xs.start is not None and xs.stop is not None
                and ys.start >= self.ry0 and ys.stop <= self.ry1
                and xs.start >= self.rx0 and xs.stop <= self.rx1):
            return self._buf[zs, ys.start - self.ry0:ys.stop - self.ry0,
                                 xs.start - self.rx0:xs.stop - self.rx0]
        return self.vol[idx]   # outside the cached box (shouldn't happen for a probe)


def _process_chunk(valid, pmap, pmap_tta, model, device, tile, h_small, w_small, 
                   transforms, collapse_fn, infer_bs, also_tta, tta):
    """helper for predict_tiles: run inference on a chunk of tiles and update pmaps.
    extracted to enable chunked processing that prevents OOM from loading all tiles at once."""
    import torch
    from tqdm import tqdm
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid), infer_bs), desc="Predict chunk", leave=False):
            chunk   = valid[i:i + infer_bs]
            b_blocks = [b for b, _, _ in chunk]
            b_idx    = [(yo, xo) for _, yo, xo in chunk]
            bt = torch.from_numpy(np.stack(b_blocks)).float().unsqueeze(1).to(device)

            if also_tta:
                # ONE read of the blocks -> BOTH the regular (identity) and TTA-averaged maps.
                # TTA re-augments the already-loaded tiles in-memory; it does NOT re-read disk.
                prob_sum = None; id_probs = None
                for j, op in enumerate(transforms):
                    p = torch.sigmoid(collapse_fn(model(op(bt).contiguous())))
                    if j == 0:
                        id_probs = p
                    prob_sum = p if prob_sum is None else prob_sum + p
                reg_preds = id_probs.cpu().numpy().flatten()
                tta_preds = (prob_sum / len(transforms)).cpu().numpy().flatten()
                for (y_off, x_off), rp, tp in zip(b_idx, reg_preds, tta_preds):
                    yi = y_off // tile; xi = x_off // tile
                    if 0 <= yi < h_small and 0 <= xi < w_small:
                        pmap[yi, xi] = float(rp); pmap_tta[yi, xi] = float(tp)
            elif tta:
                prob_sum = None
                for op in transforms:
                    p = torch.sigmoid(collapse_fn(model(op(bt).contiguous())))
                    prob_sum = p if prob_sum is None else prob_sum + p
                preds = (prob_sum / len(transforms)).cpu().numpy().flatten()
                for (y_off, x_off), pred in zip(b_idx, preds):
                    yi = y_off // tile; xi = x_off // tile
                    if 0 <= yi < h_small and 0 <= xi < w_small:
                        pmap[yi, xi] = float(pred)
            else:
                preds = torch.sigmoid(collapse_fn(model(bt))).cpu().numpy().flatten()
                for (y_off, x_off), pred in zip(b_idx, preds):
                    yi = y_off // tile; xi = x_off // tile
                    if 0 <= yi < h_small and 0 <= xi < w_small:
                        pmap[yi, xi] = float(pred)
            del bt


def predict_tiles(config, model, vol, mask, coords, y_range, x_range, depth_start, volume_name, g_mean, g_std, g_min, g_max, tta=False, also_tta=False):
    """run batched prediction over given coords returning downsampled map.

    reads tiles as y-row strips: one zarr call per row of tiles instead of one
    per tile. for a scroll ~8000px wide at tile=16 this is ~500 fewer zarr calls
    per row, cutting read time by ~10-100x while producing identical results.

    tta: when True, average the model output over the 6 spatial dihedral transforms
    (identity, h/v flip, 180, +/-90 rot) of each tile. the model emits a tile scalar
    (or a 4D map collapsed via max), so the transforms need no inverse -- we just mean
    the sigmoid. suppresses hallucinations that are inconsistent across orientations.
    """
    from collections import defaultdict

    tile  = config.data.tile_size
    depth = config.data.depth
    H = y_range[1] - y_range[0]
    W = x_range[1] - x_range[0]
    h_small = H // tile
    w_small = W // tile
    pmap = np.full((h_small, w_small), np.nan, dtype=np.float32)

    tile_scale = (32.0 / tile) ** 2 * (8.0 / max(depth, 1))
    infer_bs = max(1, min(int(256 * tile_scale), 256))
    device = config.device if torch.cuda.is_available() else "cpu"
    mode = getattr(config.data, "input_mode", "single")

    # context-window toggle: when context_size>tile (single mode only) each tile is read as a
    # larger crop centered on it; the model center-pools MIL over the tile region. shrink the
    # inference batch by the area ratio so the bigger crops don't blow VRAM.
    ctx = int(getattr(config.data, "context_size", 0) or 0)
    use_ctx = (ctx > tile and mode == "single")
    if use_ctx:
        infer_bs = max(1, int(infer_bs * (tile / float(ctx)) ** 2))

    # optional manual override: crank this up to exploit spare VRAM (throughput is often
    # launch/underutilization-bound, not memory-bound, on a large card). 0 = auto (above).
    _ib = int(getattr(config.data, "eval_infer_bs", 0) or 0)
    if _ib > 0:
        infer_bs = _ib

    # group x offsets by y_off so each y-row becomes one contiguous zarr read
    by_y = defaultdict(list)
    for _, y_off, x_off in coords:
        by_y[y_off].append(x_off)

    def _read_strip(z_start, n_depth, y_abs, x_abs_min, width):
        """read, normalize and mask one full-width y-strip in a single zarr call."""
        if z_start + n_depth > vol.shape[0] or width <= 0:
            return None
        s = np.array(vol[z_start:z_start + n_depth, y_abs:y_abs + tile, x_abs_min:x_abs_min + width], dtype=np.float32)
        if s.shape != (n_depth, tile, width):
            return None
        s = (s - g_mean) / g_std
        if mask.ndim == 2:
            m = (mask[y_abs:y_abs + tile, x_abs_min:x_abs_min + width] > 0)
            s[:, ~m] = 0.0
        return np.clip((s - g_min) / (g_max - g_min + 1e-12), 0.0, 1.0)

    def _read_ctx_strip(z_start, n_depth, y0, x0, hgt, wid):
        """read a zero-padded (OOB-safe) strip of height hgt for the context-window model.
        matches the dataloader: zero-pad in RAW space, then normalize the whole block."""
        if z_start + n_depth > vol.shape[0] or wid <= 0 or hgt <= 0:
            return None
        H, W = int(vol.shape[1]), int(vol.shape[2])
        out = np.zeros((n_depth, hgt, wid), dtype=np.float32)
        ys, ye = max(0, y0), min(H, y0 + hgt)
        xs, xe = max(0, x0), min(W, x0 + wid)
        if ys < ye and xs < xe:
            out[:, ys - y0:ye - y0, xs - x0:xe - x0] = np.array(
                vol[z_start:z_start + n_depth, ys:ye, xs:xe], dtype=np.float32)
        out = (out - g_mean) / g_std
        return np.clip((out - g_min) / (g_max - g_min + 1e-12), 0.0, 1.0)

    valid = []
    n_rows = len(by_y)
    total_tiles = sum(len(v) for v in by_y.values())
    print(f"[predict] reading {total_tiles} tiles in {n_rows} row-strips ({volume_name})")
    
    # MEMORY FIX: process tiles in chunks to avoid OOM. at 48x48x24, 170k tiles = ~37GB RAM.
    # instead of loading all tiles then inferring, we read+infer in batches of ~20k tiles
    # (~4GB), keeping peak RAM bounded. batch size scales with tile/depth (smaller tiles/depth
    # -> more tiles per GB, so larger batch allowed).
    tile_area = (ctx if (ctx > tile and mode == "single") else tile) ** 2
    depth_eff = {"triple": depth * 3, "double": depth * 2, "fulldepth": int(vol.shape[0])}.get(mode, depth)
    tiles_per_gb = 1e9 / (tile_area * depth_eff * 4)  # 4 bytes per float32
    chunk_tiles = max(5000, int(tiles_per_gb * 4))     # target ~4GB per chunk

    # TTA transforms — defined HERE (before the read loop) so _process_chunk can use them.
    _flips = (
        lambda t: t,
        lambda t: torch.flip(t, dims=[4]),        # h-flip
        lambda t: torch.flip(t, dims=[3]),        # v-flip
        lambda t: torch.flip(t, dims=[3, 4]),     # 180
    )
    if str(getattr(config.data, "tta_mode", "flips")).lower() == "dihedral":
        _tf = _flips + (
            lambda t: torch.rot90(t, 1, dims=[3, 4]),   # +90
            lambda t: torch.rot90(t, -1, dims=[3, 4]),  # -90
        )
    else:
        _tf = _flips

    def _collapse(lg):
        return lg.flatten(1).max(dim=1, keepdim=True).values if lg.dim() == 4 else lg

    pmap_tta = np.full((h_small, w_small), np.nan, dtype=np.float32) if also_tta else None

    pbar_read = tqdm(sorted(by_y), desc=f"Read {volume_name}", leave=False)
    
    for y_off in pbar_read:
        x_offs    = sorted(by_y[y_off])
        y_abs     = y_range[0] + y_off
        x_abs_min = x_range[0] + min(x_offs)
        x_abs_max = x_range[0] + max(x_offs) + tile
        width     = x_abs_max - x_abs_min

        if mode == "diff":
            pre_z = getattr(config.data, "pre_band_start", 20)
            s_ink = _read_strip(depth_start, depth, y_abs, x_abs_min, width)
            s_pre = _read_strip(pre_z,       depth, y_abs, x_abs_min, width)
            if s_ink is None or s_pre is None:
                continue
            strip = np.clip(s_ink - s_pre, 0.0, None)
        elif mode == "triple":
            pre_z  = getattr(config.data, "pre_band_start", 20)
            post_z = getattr(config.data, "post_band_start", 40)
            s_pre  = _read_strip(pre_z,       depth, y_abs, x_abs_min, width)
            s_ink  = _read_strip(depth_start, depth, y_abs, x_abs_min, width)
            s_post = _read_strip(post_z,      depth, y_abs, x_abs_min, width)
            if any(s is None for s in (s_pre, s_ink, s_post)):
                continue
            strip = np.concatenate([s_pre, s_ink, s_post], axis=0)
        elif mode == "double":
            pre_z = getattr(config.data, "pre_band_start", 20)
            s_ink = _read_strip(depth_start, depth, y_abs, x_abs_min, width)
            s_pre = _read_strip(pre_z,       depth, y_abs, x_abs_min, width)
            if s_ink is None or s_pre is None:
                continue
            strip = np.concatenate([s_ink, s_pre], axis=0)
        elif mode == "fulldepth":
            full_d = int(vol.shape[0])
            strip = _read_strip(0, full_d, y_abs, x_abs_min, width)
            if strip is None:
                continue
        else:  # single
            if use_ctx:
                pad = (ctx - tile) // 2
                strip = _read_ctx_strip(depth_start, depth, y_abs - pad, x_abs_min - pad,
                                        ctx, width + 2 * pad)
            else:
                strip = _read_strip(depth_start, depth, y_abs, x_abs_min, width)
            if strip is None:
                continue

        expected_d = strip.shape[0]
        sp = ctx if use_ctx else tile
        for x_off in x_offs:
            xl = (x_range[0] + x_off) - x_abs_min
            blk = strip[:, :, xl:xl + sp]
            if blk.shape == (expected_d, sp, sp):
                valid.append((np.ascontiguousarray(blk), y_off, x_off))
        
        # CHUNKED INFERENCE: when buffer reaches chunk_tiles, run inference and clear buffer.
        # this keeps peak RAM bounded instead of accumulating all 170k tiles before inference.
        if len(valid) >= chunk_tiles:
            _process_chunk(valid, pmap, pmap_tta, model, device, tile, h_small, w_small,
                           _tf, _collapse, infer_bs, also_tta, tta)
            valid.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    pbar_read.close()
    
    # process remaining tiles
    if valid:
        _process_chunk(valid, pmap, pmap_tta, model, device, tile, h_small, w_small,
                       _tf, _collapse, infer_bs, also_tta, tta)
        valid.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # optional post-hoc spatial smoothing -- only applied to valid (non-NaN) tiles
    sigma = float(getattr(config.data, "smooth_sigma", 0.0))

    def _smooth(pm):
        if pm is None or sigma <= 0:
            return pm
        vmask = np.isfinite(pm)
        filled = ndimage.gaussian_filter(np.where(vmask, pm, 0.0), sigma=sigma)
        weight = ndimage.gaussian_filter(vmask.astype(np.float32), sigma=sigma)
        with np.errstate(invalid='ignore'):
            sm = np.where(weight > 0, filled / weight, np.nan)
        return np.clip(sm, 0.0, 1.0)

    pmap = _smooth(pmap)
    if also_tta:
        return pmap, _smooth(pmap_tta)
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
                 shared_writer=None, tag_prefix: str = "", load_test_frags: bool = True):
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
        # only the primary per-scroll visualizer loads the (large) test-fragment assets;
        # test figures are scroll-independent so loading them in every per-scroll vis would
        # multiply RAM by the training-scroll count (OOM with many scrolls).
        self._load_test_frags = load_test_frags
        self.scroll1_id = int(scroll_id) if scroll_id is not None else int(config.data.scrolls[0].scroll_id)
        # whether this scroll renders evaluation figures. vis_scroll_ids=None => all
        # scrolls render (default); otherwise only listed scrolls do. test/probe
        # figures are unaffected.
        _vis_ids = getattr(config.data, "vis_scroll_ids", None)
        self.eval_enabled = (not _vis_ids) or (int(self.scroll1_id) in [int(v) for v in _vis_ids])
        self.probe_log_interval = max(1, int(getattr(config.tra, "probe_int", 5)))

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
                "loss": [
                    "Multiline", [
                        "G_M/Loss/Valid"
                    ]
                ],
                "accuracy": [
                    "Multiline", [
                        "G_M/Acc/Valid"
                    ]
                ],
            },
            "P_M_Metrics": {
                # "precision_recall": [
                #     "Multiline", [
                #         "P_M/Precision/Train", "P_M/Precision/Valid",
                #         "P_M/Recall/Train", "P_M/Recall/Valid"
                #     ]
                # ],
                # "f1_specificity": [
                #     "Multiline", [
                #         "P_M/F1_Score/Train", "P_M/F1_Score/Valid",
                #         "P_M/Specificity/Train", "P_M/Specificity/Valid"
                #     ]
                # ],
                "F1": [
                    "Multiline", [
                        "P_M/F1_Score/Valid"
                    ]
                ],
            },
            "AUC_Metrics": {
                "roc_auc": ["Multiline", ["AUC/ROC_AUC/Valid"]],
                "pr_auc": ["Multiline", ["AUC/PR_AUC/Valid"]],
            },
            "Readability": {
                # "contrast_ranking": [
                #     "Multiline", [
                #         "R_M/Train/LocalContrast",
                #         "R_M/Valid/LocalContrast",
                #         "R_M/Train/LocalRanking",
                #         "R_M/Valid/LocalRanking",
                #     ]
                # ],
                # "composite": [
                #     "Multiline", [
                #         "R_M/Train/ReadabilityComposite",
                #         "R_M/Valid/ReadabilityComposite",
                #         "R_M/Train_tta/ReadabilityComposite",
                #         "R_M/Valid_tta/ReadabilityComposite",
                #     ]
                # ],
                # pinnable aggregate: mean ReadabilityComposite across all 30 probe ROIs,
                # split by probe label (easy = letter-tracing, hard = ambiguous region)
                "probe_aggregate": [
                    "Multiline", [
                        "R_M/Probe/ALL/ReadabilityComposite",
                        "R_M/Probe/Easy/ReadabilityComposite",
                        "R_M/Probe/Hard/ReadabilityComposite",
                    ]
                ],
                "probe_aggregate_tta": [
                    "Multiline", [
                        "R_M/Probe/ALL/ReadabilityComposite_tta",
                        "R_M/Probe/Easy/ReadabilityComposite_tta",
                        "R_M/Probe/Hard/ReadabilityComposite_tta",
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

        # print the log location once (primary visualizer only; per-scroll secondaries
        # share the same run dir, so repeating it per scroll is just noise)
        if shared_writer is None:
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

        # load test data region with stats
        self.test_volume, self.test_mask, self.test_y_range, self.test_x_range = self._load_test_region()
        self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max = self._get_or_compute_norm(
            self.test_volume, self.test_mask, str(self.scroll1_id)
        )

        # test fragments: one entry per test_scroll_id. each is loaded defensively
        # (missing zarr/mask is skipped). stored as a list of dicts so add_test_figures
        # can iterate them one at a time, clearing CUDA cache between to prevent OOM
        # on large segments.
        self.testfrags = []
        # test fragments + holdout(s). holdouts are rendered as full-size test figures too
        # (the hallucination sanity check) but are never in the training corpus.
        _tfs = []
        if self._load_test_frags:
            _tfs = [(int(t), False) for t in (getattr(self.c.data, 'test_scroll_ids', None) or [])]
            if not _tfs:
                _tf1 = getattr(self.c.data, 'test_scroll_id', None)
                if _tf1 is not None:
                    _tfs = [(int(_tf1), False)]
            _tfs += [(int(h), True) for h in (getattr(self.c.data, 'holdout_scroll_ids', None) or [])]
        for _tf, _is_hold in _tfs:
            try:
                import zarr as _zarr
                _tv = _zarr.open(os.path.join(self.c.data.zarr_path, f'{_tf}.zarr'), mode='r')
                _tm = imread_gray(f'./masks/{_tf}.png')
                if _tm is None:
                    raise FileNotFoundError(f'./masks/{_tf}.png missing')
                _tm = _tm / 255.0
                _D, _H, _W = map(int, _tv.shape)
                _norm = self._get_or_compute_norm(_tv, _tm, str(_tf))
                self.testfrags.append({
                    'id': _tf, 'name': str(_tf), 'is_holdout': _is_hold,
                    'volume': _tv, 'mask': _tm,
                    'y_range': (0, _H), 'x_range': (0, _W),
                    'mean': _norm[0], 'std': _norm[1],
                    'min': _norm[2], 'max': _norm[3],
                })
                print(f'[testfrag] loaded {_tf}{" (holdout)" if _is_hold else ""} shape ({_D},{_H},{_W}) for test figures')
            except Exception as e:
                print(f'[testfrag] {_tf} not available, skipping ({e})')

        self._segment_assets = {}
        self._probe_vol_cache = {}
        self.probe_specs = self._build_probe_specs()

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
          probes: named training-scroll ROIs (Easy | Medium | Hard | Auto)

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

        out_dir = os.path.join(self.log_path, "dense_figs")
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

                ax_p.imshow(pred, cmap=SCROLL_CMAP, vmin=0, vmax=1)
                ax_p.set_title(row_label, fontsize=5, pad=1)

                # overlay column: pred + gold inklabel if GT exists
                ax_o.imshow(pred, cmap=SCROLL_CMAP, vmin=0, vmax=1)
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
        try:
            T = self.c.data.tile_size
            _TRAIN_TAGS = {"Easy", "Medium", "Hard"}
            agg_rm = []
            for spec, pd in zip(specs, probe_data):
                if pd is None or pd.get("gt") is None:
                    continue   # no GT
                if spec.get("tag") not in _TRAIN_TAGS:
                    continue   # only labeled training-scroll ROIs contribute
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

    def _gen_tile_coords(self, z_range, y_range, x_range, mask, z_step=None):
        """generate valid tile coords within ranges filtered by mask.
        z_step defaults to depth//2 (overlapping windows, used for hard mining);
        pass z_step=depth for non-overlapping eval figure passes (2x faster)."""
        z0, z1 = z_range
        y0, y1 = y_range
        x0, x1 = x_range

        depth = self.c.data.depth
        tile = self.c.data.tile_size

        z_span = max(0, z1 - z0 - depth + 1)
        y_span = max(0, y1 - y0 - tile + 1)
        x_span = max(0, x1 - x0 - tile + 1)

        coords = []
        if z_step is None:
            z_step = max(1, depth // 2)
        else:
            z_step = max(1, z_step)

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

        return vol, mask, y_range, x_range

    def _build_probe_specs_legacy(self):
        """DEPRECATED — superseded by _build_probe_specs which reads from config.data.probe_rois."""
        return []

    def _build_probe_specs(self):
        """build probe specs from config.data.probe_rois for ALL training scrolls.

        each ProbeROI's (x, y) is the window TOP-LEFT in full-res px. it is snapped to the
        effective model grid unit G = max(tile_size, context_size) -- i.e. 16, 32 or 48 --
        so the probe window's tiles land on the exact grid the model trains/infers on. the
        full-scroll eval grid is already anchored to the same multiples (crop origin snapped
        to tile_size, tiles stepped by tile_size), so probe and scroll stay phase-aligned.
        returns a list of spec-dicts understood by _collect_probe_region_predictions.
        """
        T = self.c.data.tile_size
        ctx = int(getattr(self.c.data, "context_size", 0) or 0)
        G = ctx if ctx > T else T          # model grid unit: 16 (plain), 32 or 48 (context)
        probe_rois = getattr(self.c.data, "probe_rois", {}) or {}

        def align(v): return (int(v) // G) * G

        specs = []
        for scroll in self.c.data.scrolls:
            sid = int(scroll.scroll_id)
            rois = probe_rois.get(sid, probe_rois.get(str(sid), []))
            for roi in rois:
                # snap the window side to a whole number of grid cells (576 already fits any G)
                size = int(getattr(roi, "size", 576) or 576)
                size = max(G, (size // G) * G)
                specs.append({
                    "tag": f"{roi.label}_{sid}" if roi.label else str(sid),
                    "title": f"{roi.label} (scroll {sid})" if roi.label else f"probe {sid}",
                    "label": roi.label,
                    "segment_id": sid,
                    "x": align(roi.x),
                    "y": align(roi.y),
                    "size": size,
                })
        return specs

    def _load_segment_labels(self, seg_id):
        """load eroded labels as a compact binary (uint8 0/1) map. only the >0.5 threshold
        is ever used downstream (label_fraction is the mean of the binarized ink), so the old
        float64 /255 was 8x wasted RAM. >127 matches the prior `/255 > 0.5` cut exactly."""
        path = f"./eroded_inklabels/{seg_id}.png"
        labels = imread_gray(path)
        if labels is None:
            raise RuntimeError(f"could not read labels at {path}")
        return (labels > 127).astype(np.uint8)

    def _load_segment_mask(self, seg_id):
        """load mask as a compact binary (uint8 0/1) map (only >0 is ever tested)."""
        path = f"./masks/{seg_id}.png"
        mask = imread_gray(path)
        if mask is None:
            raise RuntimeError(f"could not read mask at {path}")
        return (mask > 127).astype(np.uint8)

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

    def _get_probe_volume(self, spec, vol):
        """RAM-cached volume crop covering a probe's read region (ROI + context margin, full
        depth). built once per (segment,x,y,size) and reused every render, so probe inference
        stops re-reading zarr each epoch. ~23 MB per probe (uint16); ~0.7 GB for all 30."""
        key = (spec["segment_id"], spec["x"], spec["y"], spec["size"])
        rc = self._probe_vol_cache.get(key)
        if rc is not None:
            return rc
        D, H, W = (int(s) for s in vol.shape)
        M = 32   # margin >= max context pad (ctx<=48 -> pad 16); covers ctx up to 80
        ry0 = max(0, spec["y"] - M); ry1 = min(H, spec["y"] + spec["size"] + M)
        rx0 = max(0, spec["x"] - M); rx1 = min(W, spec["x"] + spec["size"] + M)
        rc = _RegionCache(vol, ry0, ry1, rx0, rx1)
        self._probe_vol_cache[key] = rc
        print(f"[probe-cache] {spec['tag']}: preloaded {rc._buf.shape} {rc._buf.dtype} "
              f"({rc._buf.nbytes / 1e6:.1f} MB)")
        return rc

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
        if scores.size < 2 or np.unique(scores).size < 2 or np.unique(fractions).size < 2:
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

        if self.mode == 'train' and (epoch + 1) % self.probe_log_interval == 0:
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
            axes[i, 0].imshow(p, cmap=SCROLL_CMAP, vmin=0, vmax=1)
            axes[i, 0].set_title(f"pred z{z0_b}-{z0_b + D}", fontsize=7)
            _mark_split(axes[i, 0])
            axes[i, 1].imshow(p, cmap=SCROLL_CMAP, vmin=0, vmax=1)
            axes[i, 1].imshow(gt_ov)                             # GT overlay
            axes[i, 1].set_title(f"pred z{z0_b}-{z0_b + D}  + GT", fontsize=7)
            _mark_split(axes[i, 1])

        axes[-1, 0].imshow(composite, cmap=SCROLL_CMAP, vmin=0, vmax=1)
        axes[-1, 0].set_title("depth-MAX composite", fontsize=7)
        _mark_split(axes[-1, 0])
        axes[-1, 1].imshow(composite, cmap=SCROLL_CMAP, vmin=0, vmax=1)
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
            out_dir = os.path.join(self.log_path, "dense_figs")
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

        eval_mask = getattr(self, 'eval_mask', self.mask)
        # the train/valid split is PURELY VISUAL (a line drawn on the figure afterward); this is
        # functionally a single eval over the whole region. so read+predict the FULL region in
        # ONE pass instead of two (train then valid) -- half the per-call overhead, and for an
        # x-split one wide strip read per row instead of two.
        if getattr(self, "split_axis", "x") == "y":
            tr_y, tr_x = self.train_range, self.shared_range
            full_y = (self.train_range[0], self.valid_range[1]); full_x = self.shared_range
        else:
            tr_y, tr_x = self.shared_range, self.train_range
            full_y = self.shared_range; full_x = (self.train_range[0], self.valid_range[1])
        
        # fast_eval_figure: only render left 40% x-dimension AND bottom 40% y-dimension
        # (makes 16% area figures - much faster rendering for single-scroll campaigns)
        if getattr(self.c.tra, "fast_eval_figure", False):
            def _snap_start(candidate, anchor, stop, tile):
                """keep the cropped eval region on the same tile grid as training."""
                rem = (candidate - anchor) % tile
                if rem:
                    candidate += tile - rem
                return max(anchor, min(candidate, stop - tile))

            tile = self.c.data.tile_size
            full_width_x = full_x[1] - full_x[0]
            fast_x_end = full_x[0] + int(full_width_x * 0.4)
            full_x = (full_x[0], fast_x_end)
            
            full_height_y = full_y[1] - full_y[0]
            fast_y_start = full_y[1] - int(full_height_y * 0.4)
            fast_y_start = _snap_start(fast_y_start, full_y[0], full_y[1], tile)
            full_y = (fast_y_start, full_y[1])

        hm_dir = self._hard_mining_dir()
        hm_enabled = getattr(self.c.hm, "enabled", True)

        full_coords = self._gen_tile_coords(z_range, full_y, full_x, eval_mask, z_step=self.c.data.depth)
        full_grouped = group_by_depth(full_coords)
        # train-region coords (coord-only, no read) -- only needed to route hard-negative mining
        # to the training portion; the full-region origin == train origin so indices line up.
        train_grouped = group_by_depth(
            self._gen_tile_coords(z_range, tr_y, tr_x, eval_mask, z_step=self.c.data.depth)
        ) if hm_enabled else {}
        depth_offsets = sorted(full_grouped.keys())
        all_pred_data = []
        all_tta_data = []

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

            # ONE read+predict pass over the full train+valid region (also_tta -> raw + TTA maps)
            full_pred, full_tta = predict_tiles(
                self.c, model, self.volume, eval_mask, full_grouped.get(d_off, []), full_y, full_x,
                depth_start, "eval", self.global_mean, self.global_std, self.global_min, self.global_max,
                also_tta=True
            )

            tile = self.c.data.tile_size

            # hard-negative mining from the TRAIN portion only (full origin == train origin, so
            # the train-tile indices line up directly with the full prediction map)
            for (_, y_off, x_off) in t_coords:
                yi = y_off // tile
                xi = x_off // tile
                if yi < 0 or yi >= full_pred.shape[0] or xi < 0 or xi >= full_pred.shape[1]:
                    continue

                score = float(full_pred[yi, xi])

                z_global = depth_start
                y_global = full_y[0] + y_off
                x_global = full_x[0] + x_off

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

            all_pred_data.append((full_pred, None, depth_start, depth_end))
            all_tta_data.append(full_tta)

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
            # when fast_eval_figure is enabled, use the cropped full_y/full_x instead
            if getattr(self.c.tra, "fast_eval_figure", False):
                # full_y and full_x were already cropped above
                full_y_range = full_y
                full_x_range = full_x
            else:
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
            # split the tile-grid maps into train/valid along the split axis so readability is
            # reported PER SPLIT (the combined region was ~75% train, masking the valid signal).
            # each depth's TTA map (all_tta_data, from the same read) yields the *_tta groups.
            n_split = (self.train_range[1] - self.train_range[0]) // self.c.data.tile_size
            _ax = "y" if getattr(self, "split_axis", "x") == "y" else "x"
            def _sp(m):
                return (m[:n_split], m[n_split:]) if _ax == "y" else (m[:, :n_split], m[:, n_split:])
            lb_t, lb_v = _sp(label_binary); lf_t, lf_v = _sp(label_fraction); vt_t, vt_v = _sp(valid_tiles)

            depth_labels = [f"{pd[2]}-{pd[3]}" for pd in all_pred_data]
            groups = {"Train": [], "Valid": [], "Train_tta": [], "Valid_tta": []}
            for i, pd in enumerate(all_pred_data):
                rp = pd[0]
                tp = all_tta_data[i] if (i < len(all_tta_data) and all_tta_data[i] is not None) else rp
                rp_t, rp_v = _sp(rp); tp_t, tp_v = _sp(tp)
                groups["Train"].append(self._compute_readability_metrics(rp_t, lb_t, lf_t, vt_t))
                groups["Valid"].append(self._compute_readability_metrics(rp_v, lb_v, lf_v, vt_v))
                groups["Train_tta"].append(self._compute_readability_metrics(tp_t, lb_t, lf_t, vt_t))
                groups["Valid_tta"].append(self._compute_readability_metrics(tp_v, lb_v, lf_v, vt_v))
            group_aggr = {k: self._aggregate_metric_dicts(v) for k, v in groups.items()}

            for gname, aggr in group_aggr.items():
                self._log_readability_scalars(epoch, aggr, gname)
            self._log_readability_compass(epoch, group_aggr)
            # per-depth summary heatmap for the train split (representative)
            _sumfig = self._create_readability_summary_figure(group_aggr["Train"], groups["Train"], depth_labels)
            self.writer.add_figure("Readability/Summary", _sumfig, epoch); plt.close(_sumfig)

            if getattr(self.c.tra, "eval_aggregate", True):
                # train_split_n: number of tiles in the train region along the split axis
                # when fast_eval_figure is enabled, compute based on the cropped extent
                if getattr(self.c.tra, "fast_eval_figure", False):
                    # full_y and full_x were cropped above; compute train_split_n relative to crop
                    split_axis = "y" if getattr(self, "split_axis", "x") == "y" else "x"
                    if split_axis == "y":
                        # y-split: train is top portion, find where train_range[1] falls in cropped full_y
                        crop_start = full_y[0]
                        crop_end = full_y[1]
                        train_end = min(self.train_range[1], crop_end)
                        train_split_n = max(0, (train_end - crop_start)) // self.c.data.tile_size
                    else:
                        # x-split: train is left portion, find where train_range[1] falls in cropped full_x
                        crop_start = full_x[0]
                        crop_end = full_x[1]
                        train_end = min(self.train_range[1], crop_end)
                        train_split_n = max(0, (train_end - crop_start)) // self.c.data.tile_size
                else:
                    train_split_n = (self.train_range[1] - self.train_range[0]) // self.c.data.tile_size
                    split_axis = "y" if getattr(self, "split_axis", "x") == "y" else "x"

                # regular inference map (primary depth block)
                reg_pred = all_pred_data[0][0]

                # TTA map: computed in the SAME read pass as reg_pred (also_tta=True above),
                # so we do NOT re-read the zarr -- TTA just re-augments the already-loaded tiles.
                tta_pred = all_tta_data[0] if all_tta_data else None

                # raw 1.1um / 2.4um inklabels (visual reference), cropped to the eval extent.
                # width is aligned to the mask (download resized to mask width); height is mapped
                # proportionally since the raw inklabel aspect differs slightly from the mask.
                # Use the already-cropped full_y and full_x (respects fast_eval_figure)
                ext_y = full_y
                ext_x = full_x
                mh, mw = self.mask.shape[:2]
                pred_h, pred_w = reg_pred.shape  # tile dimensions of prediction
                def _raw_ink(sub):
                    img = imread_gray(f"./inklabels/{sub}/{self.scroll1_id}.png")
                    if img is None:
                        return None
                    rh, rw = img.shape[:2]
                    ry0 = int(ext_y[0] * rh / max(mh, 1)); ry1 = int(ext_y[1] * rh / max(mh, 1))
                    rx0 = int(ext_x[0] * rw / max(mw, 1)); rx1 = int(ext_x[1] * rw / max(mw, 1))
                    crop = img[ry0:ry1, rx0:rx1]
                    if crop.size == 0:
                        return None
                    # resize to match prediction tile dimensions for proper aspect in figure
                    resized = cv2.resize(crop, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)
                    return resized
                raw_1_1 = _raw_ink("1_1um"); raw_2_4 = _raw_ink("2_4um")

                fig = self._create_eval_figure_2x3(reg_pred, tta_pred, label_binary,
                                                   raw_1_1, raw_2_4, train_split_n, split_axis)
                if fig is not None:
                    # ALWAYS save the full-size eval figure to <log>/eval_figs/ (highest res)
                    try:
                        _ed = os.path.join(self.log_path, "eval_figs"); os.makedirs(_ed, exist_ok=True)
                        _lp = os.path.join(_ed, f"eval_s{self.scroll1_id}_ep{epoch+1:02d}.png")
                        fig.savefig(_lp, dpi=200, bbox_inches="tight")
                        print(f"[eval-fig] full-size -> {_lp}")
                    except Exception as _e:
                        print(f"[eval-fig] save failed: {_e}")
                    # also drop a copy into ./output/visualizations/<exp>/ when save_vis is on
                    if getattr(self.c.tra, "save_vis", False):
                        try:
                            _p = os.path.join(self._save_vis_dir(), f"eval_s{self.scroll1_id}_ep{epoch+1:02d}.png")
                            fig.savefig(_p, dpi=200, bbox_inches="tight")
                            print(f"[save-vis] eval figure -> {_p}")
                        except Exception as _e:
                            print(f"[save-vis] eval save failed: {_e}")
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

    def _save_vis_dir(self):
        """per-run visualization output folder: ./output/visualizations/<exp_name>/.
        each run (unique exp_name) gets its own folder so leave-one-out runs never collide."""
        name = getattr(self.c, "exp_name", None) or "run"
        d = os.path.join("output", "visualizations", str(name))
        os.makedirs(d, exist_ok=True)
        return d

    def add_test_figures(self, epoch, model):
        """FULL-SIZE, notebook-style 4-panel test figures for every test fragment + holdout.

        each figure (one per fragment): [raw pred | TTA pred | composite | overlay], inferno
        heatmaps upsampled to native resolution, native-resolution VC3D-style composite,
        cropped to the mask bbox with white padding on a black background. NO downscaling: the
        model runs over the full-resolution fragment and the saved JPG is full size.

        the FULL-SIZE JPG is written to <log>/test_figs/ AND ./output/test_visualizations/<exp>/;
        a bounded copy (<=4096 px) is logged to tensorboard so event files stay small."""
        if not self.testfrags:
            return
        print(f"Generating {len(self.testfrags)} full-size test figures...")
        model.eval()
        for frag in self.testfrags:
            try:
                self._render_test_fragment(epoch, model, frag)
            except Exception as e:
                print(f"[ERROR] test fragment {frag.get('name', frag.get('id'))} figure failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                # free VRAM between large renders so subsequent (very large) segments don't OOM
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass

    # ---- full-size test-fragment rendering (ported from test_inference.ipynb) ----
    def _frag_project_depth(self, vol, d0, d1, method):
        """memory-bounded depth projection over slices [d0, d1) (one slice at a time)."""
        Z, H, W = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        d0, d1 = max(0, d0), min(d1, Z)
        if method == "meanproj":
            acc = np.zeros((H, W), np.float64)
            for d in range(d0, d1):
                acc += np.asarray(vol[d])
            return (acc / max(d1 - d0, 1)).astype(np.float32)
        # default: maxproj (VC3D's max filter over the surface window)
        acc = np.zeros((H, W), np.float32)
        for d in range(d0, d1):
            acc = np.maximum(acc, np.asarray(vol[d]).astype(np.float32))
        return acc

    def _frag_composite(self, vol, mask_bool, d0, d1, method, display):
        """native-resolution fiber-visibility composite (uint8), matched to VC3D."""
        proj = self._frag_project_depth(vol, d0, d1, method)
        m = mask_bool if mask_bool.shape == proj.shape else (proj > 0)
        if display == "raw":
            img = np.clip(proj, 0, 255).astype(np.uint8)   # VC3D linear volume window
        else:
            lo, hi = (np.percentile(proj[m], [1, 99]) if m.any()
                      else (float(proj.min()), float(proj.max())))
            img = (np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        return img * m.astype(np.uint8)

    def _frag_bbox(self, mask_bool, margin):
        """outer bounding box of the mask (+margin), clamped to image bounds."""
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return 0, mask_bool.shape[0], 0, mask_bool.shape[1]
        y0 = max(0, int(ys.min()) - margin); y1 = min(mask_bool.shape[0], int(ys.max()) + 1 + margin)
        x0 = max(0, int(xs.min()) - margin); x1 = min(mask_bool.shape[1], int(xs.max()) + 1 + margin)
        return y0, y1, x0, x1

    def _frag_colorize(self, pmap, out_hw):
        """tile-res prob map -> full-res inferno BGR; NaN (outside mask) -> gray."""
        m = np.isfinite(pmap)
        p8 = (np.clip(np.nan_to_num(pmap, nan=0.0), 0, 1) * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(p8, cv2.COLORMAP_INFERNO)
        bgr[~m] = (115, 115, 115)
        return cv2.resize(bgr, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)

    def _frag_scale_bar(self, bgr, voxel_um):
        """draw a RED 1 cm scale bar bottom-right; oriented along the longer axis of the panel."""
        H, W = bgr.shape[:2]
        L = int(round(10000.0 / voxel_um))   # 1 cm in pixels at full res
        red = (0, 0, 255)                    # BGR
        th = max(4, min(H, W) // 250)
        pad = int(0.05 * min(H, W)) + th
        fs = 1.2; ft = 3
        if W >= H:
            x1 = W - pad; x0 = max(0, x1 - L); y = H - pad
            cv2.rectangle(bgr, (x0, y - th), (x1, y), red, -1)
            cv2.putText(bgr, "1 cm", (x0, y - th - 12), cv2.FONT_HERSHEY_SIMPLEX, fs, red, ft, cv2.LINE_AA)
        else:
            y1 = H - pad; y0 = max(0, y1 - L); x = W - pad
            cv2.rectangle(bgr, (x - th, y0), (x, y1), red, -1)
            cv2.putText(bgr, "1 cm", (max(0, x - th - int(4 * fs * 22)), y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, red, ft, cv2.LINE_AA)
        return bgr

    def _frag_pad(self, panel, pad_px):
        """white border around a panel for visual separation."""
        return cv2.copyMakeBorder(panel, pad_px, pad_px, pad_px, pad_px,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def _frag_label(self, img, text):
        """prepend a black banner with white text above a BGR panel (orientation-independent
        labeling so the mosaic reads name [ raw | TTA | composite | overlay ])."""
        h, w = img.shape[:2]
        bar = max(28, w // 40)
        fs = bar / 34.0
        strip = np.zeros((bar, w, 3), dtype=img.dtype)
        cv2.putText(strip, str(text), (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), max(1, int(round(fs * 1.4))), cv2.LINE_AA)
        return np.vstack([strip, img])

    def _render_test_fragment(self, epoch, model, frag):
        """render + save + log ONE full-size 4-panel test figure for a fragment."""
        name = frag["name"]; sid = int(frag["id"])
        vol = frag["volume"]
        mask01 = np.asarray(frag["mask"], dtype=np.float32)
        gm, gs, gmin, gmax = frag["mean"], frag["std"], frag["min"], frag["max"]
        H, W = mask01.shape
        mask_bool = mask01 > 0
        T = int(self.c.data.tile_size)
        d_start = int(self.c.data.d_start)

        # full-fragment tile coords (only tiles that touch the mask)
        coords = [(0, y, x) for y in range(0, H - T + 1, T) for x in range(0, W - T + 1, T)
                  if mask01[y:y+T, x:x+T].sum() > 0]
        # single read -> BOTH the raw and TTA maps (also_tta), matching the eval-figure
        # optimization: TTA re-augments the already-loaded tiles instead of re-reading the zarr.
        raw, ttp = predict_tiles(self.c, model, vol, mask01, coords, (0, H), (0, W),
                                 d_start, f"Frag_{sid}", gm, gs, gmin, gmax, also_tta=True)

        cm    = getattr(self.c.data, "composite_method", "maxproj")
        cd0   = int(getattr(self.c.data, "composite_d0", 10))
        cd1   = int(getattr(self.c.data, "composite_d1", 18))
        cdisp = getattr(self.c.data, "composite_display", "raw")
        comp  = self._frag_composite(vol, mask_bool, cd0, cd1, cm, cdisp)   # native-res uint8

        y0, y1, x0, x1 = self._frag_bbox(mask_bool, 32)
        Hc, Wc = y1 - y0, x1 - x0
        p1 = self._frag_colorize(raw, (H, W))[y0:y1, x0:x1]
        p2 = self._frag_colorize(ttp, (H, W))[y0:y1, x0:x1]
        p3 = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)[y0:y1, x0:x1]
        panels = [p1, p2, p3.copy(), p3.copy()]
        panels[3] = self._frag_scale_bar(panels[3], float(getattr(self.c.data, "voxel_um", 9.362)))
        # label panels so the mosaic reads: name [ raw | TTA | composite | overlay ]
        # (the 4th 'overlay' panel is the composite the user annotates by hand)
        _labels = [f"{name}  raw", "TTA", "composite", "overlay"]
        panels = [self._frag_label(p, lb) for p, lb in zip(panels, _labels)]
        panels = [self._frag_pad(p, 24) for p in panels]
        big = np.hstack(panels) if Hc >= Wc else np.vstack(panels)   # tall->side by side; wide->stacked

        out_dir = os.path.join(self.log_path, "test_figs")
        os.makedirs(out_dir, exist_ok=True)
        tag = "holdout" if frag.get("is_holdout") else "test"
        out_p = os.path.join(out_dir, f"{tag}_{name}_ep{epoch+1:02d}.jpg")
        cv2.imwrite(out_p, big, [cv2.IMWRITE_JPEG_QUALITY, 92])   # FULL SIZE on disk
        # ALWAYS save the full-size figure to ./output/test_visualizations/<exp>/ too
        try:
            _tv = os.path.join("output", "test_visualizations", str(self.c.exp_name or "run"))
            os.makedirs(_tv, exist_ok=True)
            cv2.imwrite(os.path.join(_tv, f"{tag}_{name}_ep{epoch+1:02d}.jpg"),
                        big, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as _e:
            print(f"[test-vis] full-size save failed: {_e}")

        # tensorboard preview: full-size event images are impractical (the big fragment is
        # ~30k px wide). cap the longest side at 4096 for TB ONLY; the disk JPG stays full size.
        prev = big
        mx = max(big.shape[0], big.shape[1])
        if mx > 4096:
            s = 4096.0 / mx
            prev = cv2.resize(big, (int(big.shape[1] * s), int(big.shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
        self.writer.add_image(f"TestFrag/{tag}_{name}",
                              cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), epoch, dataformats="HWC")
        print(f"[testfrag] {tag} {name} ep{epoch+1} FULL {big.shape[1]}x{big.shape[0]} -> {out_p}")

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
        im1 = ax_pred.imshow(full_pred, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
        ax_pred.set_title(f'Predictions (Depth {d_start}-{d_end})', fontsize=9)

        split_pos = train_pred.shape[1] - 0.5
        ax_pred.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_pred.axis('off')

        ax_overlay = axes[1]
        ax_overlay.imshow(full_pred, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
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

    def _display_norm(self, pmap):
        """display-only contrast for a prediction map (does NOT affect metrics or saved values).
        raw: as-is. percentile: stretch [p2,p98] -> [0,1]. rank: histogram-equalize (rank/N) so
        the relative ordering fills the colormap even when outputs saturate near 1.0."""
        mode = getattr(self.c.data, "eval_cmap_norm", "raw")
        if mode == "raw":
            return pmap
        m = np.isfinite(pmap)
        if not m.any():
            return pmap
        out = pmap.copy()
        vals = pmap[m].astype(np.float64)
        if mode == "percentile":
            lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
            if hi - lo < 1e-6:
                return pmap
            out[m] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        elif mode == "rank":
            ranks = np.argsort(np.argsort(vals))
            out[m] = ranks / max(len(vals) - 1, 1)
        return out

    def _create_eval_figure_2x3(self, reg_pred, tta_pred, label_binary,
                                raw_1_1, raw_2_4, train_split_n, split_axis="x"):
        """2-column x 3-row eval figure:
            row 0: regular inference        | inference + inklabel overlay
            row 1: TTA inference            | TTA + inklabel overlay
            row 2: 1.1um inklabel_raw       | 2.4um inklabel_raw
        every panel occupies the same cell footprint; predictions use SCROLL_CMAP (0-1),
        the raw inklabel references are grayscale. split line = train/valid divider.
        """
        h_tiles, w_tiles = reg_pred.shape
        aspect = w_tiles / max(h_tiles, 1)
        panel_w = max(6.0, min(16.0, w_tiles * 0.06))
        panel_h = max(2.0, min(12.0, panel_w / aspect))
        panel_w = panel_h * aspect
        fig_w = panel_w * 2 + 0.3
        fig_h = panel_h * 3 + 0.5

        fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h), squeeze=False)
        split_pos = train_split_n - 0.5

        def _draw_split(ax):
            if split_axis == "y":
                ax.axhline(y=split_pos, color='red', linestyle='--', linewidth=0.8)
            else:
                ax.axvline(x=split_pos, color='red', linestyle='--', linewidth=0.8)

        def _overlay(ax, pred):
            if label_binary is not None:
                ov = np.zeros((*pred.shape, 4))
                h = min(label_binary.shape[0], ov.shape[0])
                w = min(label_binary.shape[1], ov.shape[1])
                ov[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
                ax.imshow(ov)

        def _pred_panel(ax, pmap, title, overlay=False):
            ax.imshow(pmap, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
            if overlay:
                _overlay(ax, pmap)
            _draw_split(ax)
            ax.set_title(title, fontsize=8); ax.axis('off')

        tp = tta_pred if tta_pred is not None else reg_pred
        reg_disp = self._display_norm(reg_pred)
        tp_disp = self._display_norm(tp)
        _pred_panel(axes[0, 0], reg_disp, "inference")
        _pred_panel(axes[0, 1], reg_disp, "inference + inklabel", overlay=True)
        _pred_panel(axes[1, 0], tp_disp, "TTA inference")
        _pred_panel(axes[1, 1], tp_disp, "TTA + inklabel", overlay=True)

        for ax, raw, ttl in ((axes[2, 0], raw_1_1, "1.1um inklabel_raw"),
                             (axes[2, 1], raw_2_4, "2.4um inklabel_raw")):
            if raw is not None:
                ax.imshow(raw, cmap="gray", vmin=0, vmax=255, aspect='equal')
            ax.set_title(ttl, fontsize=8); ax.axis('off')

        plt.subplots_adjust(wspace=0.04, hspace=0.12, left=0.01, right=0.99, top=0.98, bottom=0.01)
        return fig

    def _create_aggregate_eval_figure(self, all_pred_data, train_split_n, label_binary, split_axis="x"):
        """one row per depth block × 2 cols: left = prediction, right = prediction + inklabel overlay.

        with a single depth window (e.g. 4->28) this is exactly 1 row × 2 cols — the overlay IS the
        full-depth prediction, so no separate "MAX across depths" row is drawn.

        figure size adapts to the map's tile dimensions and aspect ratio so the image is never
        distorted. split_axis controls whether the train/valid divider is vertical (x) or
        horizontal (y).
        """
        n_blocks = len(all_pred_data)
        if n_blocks == 0:
            return None

        # derive panel size from the actual tile-unit dimensions of the first map
        sample_pred = all_pred_data[0][0]
        h_tiles, w_tiles = sample_pred.shape
        aspect = w_tiles / max(h_tiles, 1)      # width / height of one panel

        panel_w = max(6.0, min(16.0, w_tiles * 0.06))
        panel_h = max(2.0, min(12.0, panel_w / aspect))
        panel_w = panel_h * aspect

        n_rows = n_blocks                   # one row per depth block; NO extra max row
        fig_w = panel_w * 2 + 0.3           # two columns + small gap
        fig_h = panel_h * n_rows + 0.4      # one row per depth block + title margin

        fig, axes = plt.subplots(n_rows, 2, figsize=(fig_w, fig_h),
                                 squeeze=False)

        split_pos = train_split_n - 0.5

        def _draw_split(ax):
            if split_axis == "y":
                ax.axhline(y=split_pos, color='red', linestyle='--', linewidth=0.8)
            else:
                ax.axvline(x=split_pos, color='red', linestyle='--', linewidth=0.8)

        def _overlay(ax, pred):
            """paint the gold inklabel overlay on top of a prediction panel."""
            if label_binary is not None:
                ov = np.zeros((*pred.shape, 4))
                h = min(label_binary.shape[0], ov.shape[0])
                w = min(label_binary.shape[1], ov.shape[1])
                ov[:h, :w][label_binary[:h, :w] > 0.5] = [1, 1, 1, 0.4]
                ax.imshow(ov)

        for row, (full_pred, train_pred, d_start, d_end) in enumerate(all_pred_data):
            # left: raw prediction
            ax_pred = axes[row, 0]
            ax_pred.imshow(full_pred, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
            ax_pred.set_title(f'Depth {d_start}-{d_end}', fontsize=8)
            _draw_split(ax_pred)
            ax_pred.axis('off')

            # right: same prediction + inklabel overlay
            ax_ov = axes[row, 1]
            ax_ov.imshow(full_pred, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
            ax_ov.set_title(f'Overlay {d_start}-{d_end}', fontsize=8)
            _overlay(ax_ov, full_pred)
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
            im = ax.imshow(pred, cmap=SCROLL_CMAP, vmin=0, vmax=1, aspect='equal')
            ax.set_title(f'Depth Block {d_start}-{d_end}', fontsize=9)
            ax.axis('off')

        for idx in range(len(all_data), rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.18, left=0.03, right=0.97, top=0.97, bottom=0.03)
        return fig

    def _log_readability_scalars(self, epoch, aggregate_metrics, prefix):
        """log the readability scalar set for one split/tta group under R_M/<prefix>/* (prefix is
        Train / Valid / Train_tta / Valid_tta)."""
        if not aggregate_metrics:
            return
        p = f"R_M/{prefix}"
        scalar_tags = {
            f"{p}/LocalContrast":         aggregate_metrics.get("local_contrast", np.nan),
            f"{p}/LocalRanking":          aggregate_metrics.get("local_ranking", np.nan),
            f"{p}/RecallAt1PctFPR":       aggregate_metrics.get("recall_at_1pct_fpr", np.nan),
            f"{p}/PartialAUCAt1PctFPR":   aggregate_metrics.get("partial_auc_at_1pct_fpr", np.nan),
            f"{p}/RecallAt5PctFPR":       aggregate_metrics.get("recall_at_5pct_fpr", np.nan),
            f"{p}/PartialAUCAt5PctFPR":   aggregate_metrics.get("partial_auc_at_5pct_fpr", np.nan),
            f"{p}/CoverageRecall":        aggregate_metrics.get("coverage_recall", np.nan),
            f"{p}/TopKPrecision":         aggregate_metrics.get("topk_precision", np.nan),
            f"{p}/InkFractionSpearman":   aggregate_metrics.get("ink_fraction_corr_spearman", np.nan),
            f"{p}/SpillRatio":            aggregate_metrics.get("spill_ratio", np.nan),
            f"{p}/ComponentCount":        aggregate_metrics.get("component_count", np.nan),
            f"{p}/MeanComponentSize":     aggregate_metrics.get("mean_component_size", np.nan),
            f"{p}/ReadabilityComposite":  aggregate_metrics.get("readability_composite", np.nan),
        }
        for tag, value in scalar_tags.items():
            if np.isfinite(value):
                self.writer.add_scalar(tag, float(value), epoch)

    def _log_readability_compass(self, epoch, group_aggr):
        """one radar chart with a series per split/tta group (Train/Valid/Train_tta/Valid_tta)."""
        if not group_aggr:
            return
        fig = self._create_readability_compass_figure(group_aggr)
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

    def _create_readability_compass_figure(self, group_aggr):
        """radar chart of the readability terms, one series per split/tta group."""
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

        _colors = {"Train": "teal", "Valid": "darkorange",
                   "Train_tta": "steelblue", "Valid_tta": "crimson"}
        series = [(name, self._readability_compass_values(aggr), _colors.get(name, "gray"))
                  for name, aggr in group_aggr.items()]

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

            aggregate_metrics = probe_data["aggregate_metrics"]
            # per-scroll probe composites removed — only the ALL aggregate is logged below

            aggregate_tta = probe_data.get("aggregate_metrics_tta")
            # tta ReadabilityComposite rolled into the ALL/ReadabilityComposite_tta aggregate;
            # skip per-probe tta scalars to avoid 30 more clutter tags.

        if probe_data_list:
            fig = self._create_combined_probe_depth_figure(probe_data_list)
            # patch the epoch number into the suptitle (avoids passing epoch deep into the figure method)
            fig.texts[0].set_text(fig.texts[0].get_text().format(epoch + 1))
            self.writer.add_figure("ProbeROIs/Grid", fig, epoch)
            plt.close(fig)

            # aggregate readability composite across all probes — total + per-label splits
            all_composites = {
                "ALL":  [], "Easy": [], "Hard": []
            }
            all_composites_tta = {
                "ALL":  [], "Easy": [], "Hard": []
            }
            for pd in probe_data_list:
                lbl   = (pd["spec"].get("label") or "").capitalize()  # "easy"->"Easy" etc.
                rc    = pd["aggregate_metrics"].get("readability_composite", float("nan"))
                rc_t  = pd.get("aggregate_metrics_tta", {}).get("readability_composite", float("nan"))
                all_composites["ALL"].append(rc)
                all_composites_tta["ALL"].append(rc_t)
                if lbl in all_composites:
                    all_composites[lbl].append(rc)
                    all_composites_tta[lbl].append(rc_t)

            for group, vals in all_composites.items():
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    self.writer.add_scalar(f"R_M/Probe/{group}/ReadabilityComposite",
                                          float(np.mean(finite)), epoch)
                    if group == "ALL":
                        print(f"[probe-agg] {group}: {np.mean(finite):.4f} "
                              f"({len(finite)} probes)")
            for group, vals in all_composites_tta.items():
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    self.writer.add_scalar(f"R_M/Probe/{group}/ReadabilityComposite_tta",
                                          float(np.mean(finite)), epoch)

    def _collect_probe_region_predictions(self, model, spec):
        """prepare per-depth predictions and readability stats for one fixed probe region"""
        try:
            asset = self._get_segment_asset(spec["segment_id"])
        except Exception as e:
            print(f"[PROBE] Skipping {spec['tag']} due to asset load error: {e}")
            return None

        volume = self._get_probe_volume(spec, asset["volume"])
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
            # single read -> raw + TTA maps (TTA re-augments the loaded tiles; no extra disk read)
            pred, pred_tta = predict_tiles(
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
                also_tta=True,
            )

            metrics = self._compute_readability_metrics(pred, label_binary, label_fraction, valid_tiles)
            metrics_tta = self._compute_readability_metrics(pred_tta, label_binary, label_fraction, valid_tiles)
            depth_rows.append(
                {
                    "depth_start": depth_start,
                    "depth_end": depth_end,
                    "pred": pred,
                    "pred_tta": pred_tta,
                    "metrics": metrics,
                    "metrics_tta": metrics_tta,
                }
            )

        aggregate_metrics = self._aggregate_metric_dicts([row["metrics"] for row in depth_rows])
        aggregate_metrics_tta = self._aggregate_metric_dicts([row["metrics_tta"] for row in depth_rows])
        return {
            "spec": spec,
            "label_binary": label_binary,
            "depth_rows": depth_rows,
            "aggregate_metrics": aggregate_metrics,
            "aggregate_metrics_tta": aggregate_metrics_tta,
            "x0": x0,
            "y0": y0,
            "size": size,
        }

    def _create_combined_probe_depth_figure(self, probe_data_list):
        """dynamic rows x 12 cols: 3 scrolls per row-pair, each scroll = easy | easy+overlay |
        hard | hard+overlay. every scroll-row is DOUBLED -- top sub-row = raw prediction,
        bottom sub-row = TTA prediction.

        predictions use the SAME display normalization as the eval figure (_display_norm). overlay
        cells dim the base to half brightness and paint the eroded inklabel in white. the list is
        ordered easy,hard per scroll; each probe_data carries per-depth 'pred' and 'pred_tta'.
        """
        n_scrolls = max(1, int(np.ceil(len(probe_data_list) / 2.0)))
        n_scroll_rows = max(1, int(np.ceil(n_scrolls / 3.0)))
        n_cols = 12
        n_rows = n_scroll_rows * 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.9 * n_rows),
                                 gridspec_kw={"hspace": 0.28, "wspace": 0.03})
        axes = np.array(axes).reshape(n_rows, n_cols)
        cmap = plt.get_cmap(SCROLL_CMAP)
        alpha = 0.45   # inklabel overlay strength
        import warnings

        def _maxpred(pd, key):
            preds = [row.get(key) for row in pd["depth_rows"] if row.get(key) is not None]
            if not preds:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mp = np.nanmax(np.stack(preds, axis=0), axis=0)
            return self._display_norm(np.nan_to_num(mp, nan=0.0))

        for pr in range(n_rows):
            scroll_row = pr // 2
            is_tta = (pr % 2 == 1)                  # each scroll-row: raw on top, TTA below
            key = "pred_tta" if is_tta else "pred"
            for col in range(n_cols):
                ax = axes[pr, col]
                ax.axis("off")
                sub = col % 4                       # 0 easy-raw 1 easy-ov 2 hard-raw 3 hard-ov
                scroll_idx = scroll_row * 3 + (col // 4)
                pidx = 2 * scroll_idx + (0 if sub < 2 else 1)
                if pidx >= len(probe_data_list):
                    continue
                pd = probe_data_list[pidx]
                mp = _maxpred(pd, key)
                if mp is None:
                    continue
                overlay = sub in (1, 3)
                rgb = cmap(np.clip(mp, 0.0, 1.0))[..., :3]
                if overlay:
                    rgb = rgb * 0.5             # dim base so the white inklabel pops
                    lb = pd["label_binary"]
                    h = min(lb.shape[0], rgb.shape[0]); w = min(lb.shape[1], rgb.shape[1])
                    g = lb[:h, :w] > 0.5
                    rgb[:h, :w][g] = (1.0 - alpha) * rgb[:h, :w][g] + alpha * np.array([1.0, 1.0, 1.0])
                ax.imshow(rgb, aspect="equal", interpolation="nearest")
                lab = pd["spec"].get("label") or pd["spec"]["tag"]
                suf = "-tta" if is_tta else ""
                if overlay:
                    ax.set_title(f"{lab}{suf}+GT", fontsize=5, pad=2)
                elif is_tta:
                    ax.set_title(f"{lab}{suf} {pd['spec']['segment_id']}", fontsize=5, pad=2)
                else:
                    c_val = np.nan_to_num(
                        pd["aggregate_metrics"].get("local_contrast", float("nan")), nan=0.0)
                    ax.set_title(f"{lab} {pd['spec']['segment_id']} C{c_val:.2f}", fontsize=5, pad=2)

        fig.suptitle("Probe ROIs ep{}", fontsize=9, y=0.997)
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
        self.writer.add_scalar("Hyperparameters/Conv1 Dropout", getattr(self.c.model, 'conv1_drop', 0.0))
        self.writer.add_scalar("Hyperparameters/Conv2 Dropout", getattr(self.c.model, 'conv2_drop', 0.0))
        _fc1 = getattr(self.c.model, 'fc1_drop', None)
        _fc2 = getattr(self.c.model, 'fc2_drop', None)
        if _fc1 is not None:
            self.writer.add_scalar("Hyperparameters/FC1 Dropout", _fc1)
        if _fc2 is not None:
            self.writer.add_scalar("Hyperparameters/FC2 Dropout", _fc2)
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
