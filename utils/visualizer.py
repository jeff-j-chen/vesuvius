from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing import _32Bit
import os
from typing import Any, Literal
from collections import defaultdict
import json
import re

import cv2
import numpy as np
import torch
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
from .dataloader import DataManager
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
plt.cm.register_cmap(name='inferno_nan', cmap=_inferno_nan)

def group_by_depth(coords):
    """group tile coordinates by their depth offset"""
    grouped = defaultdict(list)
    for d_off, y_off, x_off in coords:
        grouped[d_off].append((d_off, y_off, x_off))
    return grouped

def predict_tiles(config, model, vol, mask, coords, y_range, x_range, depth_start, volume_name, g_mean, g_std, g_min, g_max):
    """run batched prediction over given coords returning downsampled map.

    zarr reads dominate inference time and are IO-bound; they release the GIL so
    ThreadPoolExecutor parallelises them effectively on windows without spawn overhead.
    all tiles are read in parallel first, then sent to the gpu in large batches.
    """
    from concurrent.futures import ThreadPoolExecutor

    tile  = config.data.tile_size
    depth = config.data.depth
    H = y_range[1] - y_range[0]
    W = x_range[1] - x_range[0]
    h_small = H // tile
    w_small = W // tile
    pmap = np.full((h_small, w_small), np.nan, dtype=np.float32)

    # inference has no gradient overhead so use a much larger batch than training
    infer_bs = max(config.dl.batch_size * 8, 512)
    device = config.device if torch.cuda.is_available() else "cpu"

    tile_list = [
        (depth_start, y_range[0] + y_off, x_range[0] + x_off, y_off, x_off)
        for _, y_off, x_off in coords
    ]

    def _read_one(args):
        d, y, x, y_off, x_off = args
        if d + depth > vol.shape[0]:
            return None, y_off, x_off
        blk = np.array(vol[d:d + depth, y:y + tile, x:x + tile]).astype(np.float32)
        blk = (blk - g_mean) / g_std
        if blk.ndim == 3 and mask.ndim == 2:
            m_tile = mask[y:y + tile, x:x + tile]
            m_bin  = (m_tile > 0).astype(np.uint8)
            blk[np.broadcast_to(np.expand_dims(m_bin, 0), blk.shape) == 0] = 0
        blk = np.clip((blk - g_min) / (g_max - g_min + 1e-12), 0, 1)
        if blk.shape != (depth, tile, tile):
            return None, y_off, x_off
        return blk, y_off, x_off

    # read tiles in parallel; threads release GIL during zarr/numcodecs decompression
    n_workers = min(8, max(1, len(tile_list)))
    print(f"[predict] reading {len(tile_list)} tiles with {n_workers} threads...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(_read_one, tile_list),
                            total=len(tile_list),
                            desc=f"Read {volume_name}", leave=False))

    valid = [(blk, y_off, x_off) for blk, y_off, x_off in results if blk is not None]

    with torch.no_grad():
        for i in tqdm(range(0, len(valid), infer_bs), desc=f"Predict {volume_name}", leave=True):
            chunk   = valid[i:i + infer_bs]
            b_blocks = [b for b, _, _ in chunk]
            b_idx    = [(yo, xo) for _, yo, xo in chunk]

            bt     = torch.from_numpy(np.stack(b_blocks)).float().unsqueeze(1).to(device)
            logits = model(bt)
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

class TensorboardVisualizer:
    def __init__(self, config: Config, mode: str = 'train'):
        """initialize tensorboard visualizer and precompute datasets and stats"""
        self.c = config
        self.mode = mode
        self.probe_log_interval = max(1, int(getattr(config.tra, "probe_int", 5)))

        if config.exp_name is None:
            if self.mode == 'finetune':
                experiment_name = f"finetune_{datetime.now().strftime('%d.%m_%H-%M-%S')}"
            else:
                experiment_name = f"ink_detection_{datetime.now().strftime('%d.%m_%H-%M-%S')}"
        else:
            experiment_name = config.exp_name + "_" + datetime.now().strftime('%d_%H-%M-%S')

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

        self.writer = SummaryWriter(self.log_path)
        self.writer.add_custom_scalars(self.layout)

        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.tra.log_dir}")

    def _init_training_assets(self):
        """load training and auxiliary datasets and normalization stats"""
        # data manager holds main training volume mask labels and splits
        dm = DataManager(self.c)
        self.dm = dm

        self.volume = dm.vol
        self.mask = dm.mask
        self.labels = dm.labels
        # respect original bounds for the main training scroll when applicable
        if self.c.data.scroll1_id == 20230827161847:
            # original spatial crop
            y0, y1 = 200, 5600
            x0, x1 = 1000, 4600
            self.y_range = (y0, y1)
            # split the cropped x-range 75/25 for train/valid to mirror original behavior
            x_len = x1 - x0
            split = int(x_len * 0.75)
            self.train_x_range = (x0, x0 + split)
            self.valid_x_range = (x0 + split, x1)
        else:
            self.train_x_range = dm.train_x
            self.valid_x_range = dm.valid_x
            self.y_range = dm.y_range
        self.global_mean, self.global_std, self.global_min, self.global_max = dm.norm_stats

        # load test data region and scroll4 data with stats
        self.test_volume, self.test_mask, self.test_y_range, self.test_x_range = self._load_test_region()
        self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max = self._get_or_compute_norm(
            self.test_volume, self.test_mask, str(self.c.data.scroll1_id)
        )

        self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range = self._load_scroll4_region()
        self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max = self._get_or_compute_norm(
            self.scroll4_volume, self.scroll4_mask, str(self.c.data.scroll4_id)
        )

        # scroll2 is always loaded — probe ROIs always include it regardless of test_on_scroll4
        self.scroll2_volume, self.scroll2_mask, self.scroll2_y_range, self.scroll2_x_range = self._load_scroll2_region()
        self.scroll2_global_mean, self.scroll2_global_std, self.scroll2_global_min, self.scroll2_global_max = self._get_or_compute_norm(
            self.scroll2_volume, self.scroll2_mask, str(self.c.data.scroll2_id)
        )

        self._segment_assets = {}
        self.probe_specs = self._build_probe_specs()
        self._debug_scroll4_ranges_once()

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
        sid = self.c.data.scroll1_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        vol = None
        try:
            import zarr
            vol = zarr.open(zarr_path, mode='r')
        except Exception as e:
            raise RuntimeError(f"could not open zarr at {zarr_path}: {e}")

        D, H, W = map(int, vol.shape)

        mask_path = f"./masks/{sid}.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

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
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        y_range = (6500 if H > 6500 else 0, H)
        x_range = (0, min(5000, W))

        return vol, mask, y_range, x_range

    def _load_scroll2_region(self):
        """load scroll2 region: 2048×1024 window at x=3080, y=748"""
        sid = self.c.data.scroll2_id
        zarr_path = os.path.join(self.c.data.zarr_path, f"{sid}.zarr")
        try:
            import zarr
            vol = zarr.open(zarr_path, mode='r')
        except Exception as e:
            raise RuntimeError(f"could not open zarr at {zarr_path}: {e}")

        mask_path = f"./masks/{sid}.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # fixed 2048 (width) × 1024 (height) window starting at x=3080, y=748
        y_range = (748, 748 + 1024)
        x_range = (3080, 3080 + 2048)

        return vol, mask, y_range, x_range

    def _build_probe_specs(self):
        """fixed readability probe regions used for qualitative tracking"""
        return [
            {
                "tag": "Easy",
                "title": "small scroll easy",
                "segment_id": 20230827161847,
                "x": 2100,
                "y": 4370,
                "size": 608,
            },
            {
                "tag": "Hard",
                "title": "small scroll hard",
                "segment_id": 20230827161847,
                "x": 3744,
                "y": 3862,
                "size": 608,
            },
            {
                "tag": "Scroll4_Pi",
                "title": "scroll4 pi",
                "segment_id": 20231210132040,
                "x": 1960,
                "y": 7968,
                "size": 608,
            },
            {
                "tag": "Scroll2",
                "title": "scroll2",
                "segment_id": 20230709155141,
                "x": 3080,
                "y": 748,
                "size": 608,
            },
        ]

    def _load_segment_labels(self, seg_id):
        """load eroded labels for a segment"""
        path = f"./eroded_inklabels/{seg_id}.png"
        labels = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if labels is None:
            raise RuntimeError(f"could not read labels at {path}")
        return labels / 255.0

    def _load_segment_mask(self, seg_id):
        """load mask for a segment"""
        path = f"./masks/{seg_id}.png"
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"could not read mask at {path}")
        return mask / 255.0

    def _get_segment_asset(self, seg_id):
        """return cached volume mask labels and normalization stats for a segment"""
        if seg_id in self._segment_assets:
            return self._segment_assets[seg_id]

        if seg_id == self.c.data.scroll1_id:
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
            neg_scores = pred_map[y0:y1, x0:x1][local_neg]
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
        valid_scores = pred_map[valid_tiles]
        valid_labels = label_binary[valid_tiles].astype(int)
        valid_fraction = label_fraction[valid_tiles]

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

        # weighted composite: coverage and coherence get 1.5× because they most directly
        # capture whether a human would see readable structure in the prediction map
        weights = [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.5]
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

        self.log_weight_histograms(model, epoch)

        if epoch == 0:
            print("Logging hyperparameters and model graph")
            ex = torch.randn(1, self.c.data.depth, self.c.data.tile_size, self.c.data.tile_size).to(self.c.device)
            ex = ex.unsqueeze(0)
            # self.log_model_graph(model, ex)
            self.log_hyperparameters(params, pos_weight)

        if self.mode == 'train' and (epoch + 1) % self.c.tra.eval_int == 0:
            try:
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
        """return hard-mining directory for the current experiment"""
        return getattr(self.c.hm, "dir", "./hard_negs")

    def add_evaluation_figures(self, epoch, model):
        """run eval on train and valid splits produce mining and figures"""
        print("Starting evaluation figure generation...")
        model.eval()

        z_range = (self.c.data.d_start, self.c.data.d_end)

        train_coords = self._gen_tile_coords(z_range, self.y_range, self.train_x_range, self.mask)
        valid_coords = self._gen_tile_coords(z_range, self.y_range, self.valid_x_range, self.mask)

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
                self.c, model, self.volume, self.mask, t_coords, self.y_range, self.train_x_range,
                depth_start, "train", self.global_mean, self.global_std, self.global_min, self.global_max
            )

            v_pred = predict_tiles(
                self.c, model, self.volume, self.mask, v_coords, self.y_range, self.valid_x_range,
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
                y_global = self.y_range[0] + y_off
                x_global = self.train_x_range[0] + x_off

                l_tile = self.labels[y_global:y_global + tile, x_global:x_global + tile]
                has_ink = int(np.any(l_tile > 0.5))

                # dedup key includes z y x and label
                key = (int(z_global), int(y_global), int(x_global), int(has_ink))

                if has_ink == 0 and score >= hn_cut:
                    if key not in existing_keys and key not in new_keys:
                        if mining_f is not None:
                            mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 0}) + "\n")
                        new_keys.add(key)
                        hn_cnt += 1
                elif has_ink == 1 and score <= hp_cut:
                    if key not in existing_keys and key not in new_keys:
                        if mining_f is not None:
                            mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 1}) + "\n")
                        new_keys.add(key)
                        hp_cnt += 1

            full_pred = np.concatenate([t_pred, v_pred], axis=1)
            all_pred_data.append((full_pred, t_pred, depth_start, depth_end))

        if mining_f is not None:
            mining_f.write(json.dumps({"_type": "meta", "hard_negatives": hn_cnt, "hard_positives": hp_cnt}) + "\n")
            mining_f.close()
            print(f"[HARD][Eval] Finished mining epoch {epoch}: neg={hn_cnt} pos={hp_cnt}")
        else:
            print("[HARD][Eval] mining disabled, skipping file write")

        if hm_enabled:
            self.writer.add_scalar("HardMining/HardNegatives", hn_cnt, epoch)
            self.writer.add_scalar("HardMining/HardPositives", hp_cnt, epoch)

        fig = self._create_hard_examples_overlay(mining_path) if mining_path else None
        if fig is not None:
            self.writer.add_figure(f"HardMined/Overlay", fig, epoch)
            plt.close(fig)

        if all_pred_data:
            full_x_range = (self.train_x_range[0], self.valid_x_range[1])
            label_binary, label_fraction, valid_tiles = self._compute_tile_maps(
                self.labels,
                self.mask,
                self.y_range,
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
                # width of the train portion in tile units (used to draw the split line)
                train_split_w = (self.train_x_range[1] - self.train_x_range[0]) // self.c.data.tile_size
                fig = self._create_aggregate_eval_figure(all_pred_data, train_split_w, label_binary)
                self.writer.add_figure('Evaluation/Aggregated', fig, epoch)
                plt.close(fig)

        self._run_and_log_hard_mining_evaluation(epoch, model)

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

        self._add_single_test_figure(epoch, model, self.test_volume, self.test_mask, self.test_y_range, self.test_x_range, self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max, "Test")

        if self.c.data.test_on_scroll4:
            self._add_single_test_figure(epoch, model, self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range, self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max, "Scroll4")
        else:
            self._add_single_test_figure(epoch, model, self.scroll2_volume, self.scroll2_mask, self.scroll2_y_range, self.scroll2_x_range, self.scroll2_global_mean, self.scroll2_global_std, self.scroll2_global_min, self.scroll2_global_max, "Scroll2")

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

    def _create_aggregate_eval_figure(self, all_pred_data, train_split_w, label_binary):
        """n_blocks-row × 2-col figure: left col = predictions, right col = overlay with inklabels.

        figure size adapts to the map's tile dimensions and aspect ratio so the image
        is never distorted regardless of scroll geometry.
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

        split_pos = train_split_w - 0.5

        for row, (full_pred, train_pred, d_start, d_end) in enumerate(all_pred_data):
            # left: raw prediction
            ax_pred = axes[row, 0]
            ax_pred.imshow(full_pred, cmap='inferno_nan', vmin=0, vmax=1, aspect='equal')
            ax_pred.set_title(f'Depth {d_start}-{d_end}', fontsize=8)
            ax_pred.axvline(x=split_pos, color='red', linestyle='--', linewidth=0.8)
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
            ax_ov.axvline(x=split_pos, color='red', linestyle='--', linewidth=0.8)
            ax_ov.axis('off')

        plt.subplots_adjust(wspace=0.04, hspace=0.12,
                            left=0.01, right=0.99,
                            top=0.98, bottom=0.01)
        return fig

    def _create_combined_test_figure(self, all_data, n_blocks, test_type):
        """create combined test figure showing prediction mosaics"""
        cols = 2
        rows = (n_blocks + cols - 1) // cols

        fig_w = 8
        h_mult = 7 if test_type == "scroll1" else 3
        fig_h = h_mult * rows

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

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
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

        # right: per-depth annotated heatmap
        raw_matrix = np.array([
            [float(metric.get(key, np.nan)) for key, _ in metric_keys]
            for metric in per_depth_metrics
        ], dtype=np.float32)

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

        fig.suptitle("Probe patches by depth: easy | hard | scroll4", fontsize=11)
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

        seg_id = self.c.data.scroll1_id
        label_path = f"./eroded_inklabels/{seg_id}.png"
        if not os.path.exists(label_path):
            return None

        label_gray = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
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
