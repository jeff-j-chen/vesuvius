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
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.ndimage as ndimage

from .config import Config
from .dataloader import DataManager
from .training_utils import calculate_metrics

def group_by_depth(coords):
    """group tile coordinates by their depth offset"""
    grouped = defaultdict(list)
    for d_off, y_off, x_off in coords:
        grouped[d_off].append((d_off, y_off, x_off))
    return grouped

def predict_tiles(config, model, vol, mask, coords, y_range, x_range, depth_start, volume_name, g_mean, g_std, g_min, g_max):
    """run batched prediction over given coords returning downsampled map"""
    tile = config.data.tile_size

    H = y_range[1] - y_range[0]
    W = x_range[1] - x_range[0]

    h_small = H // tile
    w_small = W // tile
    pmap = np.zeros((h_small, w_small), dtype=np.float32)

    bs = config.dl.batch_size
    device = config.device if torch.cuda.is_available() else "cpu"

    tiles = []
    for _, y_off, x_off in coords:
        # each call to predict_tiles targets a single depth window [depth_start, depth_start + depth)
        d = depth_start
        y = y_range[0] + y_off
        x = x_range[0] + x_off
        tiles.append((d, y, x, y_off, x_off))

    with torch.no_grad():
        for i in tqdm(range(0, len(tiles), bs), desc=f"Predict {volume_name}", leave=True):
            batch = tiles[i:i + bs]
            b_blocks = []
            b_idx = []
            for d, y, x, y_off, x_off in batch:
                if d + config.data.depth > vol.shape[0]:
                    continue

                blk = np.array(vol[d:d + config.data.depth, y:y + tile, x:x + tile]).astype(np.float32)

                blk = (blk - g_mean) / g_std

                if blk.ndim == 3 and mask.ndim == 2:
                    m_tile = mask[y:y + tile, x:x + tile]
                    m_bin = (m_tile > 0).astype(np.uint8)
                    m_exp = np.broadcast_to(np.expand_dims(m_bin, axis=0), blk.shape)
                    blk[m_exp == 0] = 0

                blk = (blk - g_min) / (g_max - g_min + 1e-12)
                blk = np.clip(blk, 0, 1)

                if blk.shape != (config.data.depth, tile, tile):
                    continue

                b_blocks.append(blk)
                b_idx.append((y_off, x_off))

            if not b_blocks:
                continue

            bt = torch.from_numpy(np.stack(b_blocks)).float().unsqueeze(1).to(device)
            logits = model(bt)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

            for (y_off, x_off), pred in zip(b_idx, preds):
                yi = y_off // tile
                xi = x_off // tile
                if 0 <= yi < h_small and 0 <= xi < w_small:
                    pmap[yi, xi] = float(pred)

            del bt

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pmap

class TensorboardVisualizer:
    def __init__(self, config: Config, mode: str = 'train'):
        """initialize tensorboard visualizer and precompute datasets and stats"""
        self.c = config
        self.mode = mode

        if config.exp_name is None:
            if self.mode == 'finetune':
                experiment_name = f"finetune_{datetime.now().strftime('%d.%m_%H:%M:%S')}"
            else:
                experiment_name = f"ink_detection_{datetime.now().strftime('%d.%m_%H:%M:%S')}"
        else:
            experiment_name = config.exp_name + "_" + datetime.now().strftime('%d_%H:%M:%S')

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

        self._debug_scroll4_ranges_once()

    def _get_or_compute_norm(self, vol, mask, seg_id):
        """compute or load cached normalization stats for a volume using a mask"""
        cache_path = "./norm_cache.json"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                if seg_id in cache:
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
            cache[seg_id] = {"mean": mean, "std": std, "min": g_min, "max": g_max}
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
            self.add_evaluation_figures(epoch, model)

        if self.mode == 'train' and (epoch + 1) % self.c.tra.test_int == 0:
            self.add_test_figures(epoch, model)

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
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        train_vals_r = train_vals + train_vals[:1]
        val_vals_r = val_vals + val_vals[:1]

        radar_ax = fig.add_subplot(1, 2, 2, projection='polar')
        radar_ax.plot(angles, train_vals_r, 'o-', linewidth=2, label='Train', color='blue')
        radar_ax.fill(angles, train_vals_r, alpha=0.25, color='blue')
        radar_ax.plot(angles, val_vals_r, 'o-', linewidth=2, label='Valid', color='red')
        radar_ax.fill(angles, val_vals_r, alpha=0.25, color='red')

        radar_ax.set_xticks(angles[:-1])
        radar_ax.set_xticklabels(categories)
        radar_ax.set_ylim(0, 1)
        radar_ax.set_title('Performance Radar Chart', y=1.08)
        radar_ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        radar_ax.grid(True)

        plt.tight_layout()
        self.writer.add_figure('Metrics_Comparison', fig, epoch)
        plt.close(fig)

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

        os.makedirs("hard_negs", exist_ok=True)
        mining_path = os.path.join("hard_negs", f"hard_mining_epoch_{epoch}.jsonl")
        mining_f = open(mining_path, "w")
        print(f"[HARD][Eval] Writing mining file to: {mining_path}")
        hn_cut = self.c.hm.hn_cutoff
        hp_cut = self.c.hm.hp_cutoff
        hn_cnt = 0
        hp_cnt = 0

        # load a set of existing mined keys across all previous files to prevent duplicates
        existing_keys = self._load_existing_mined_keys()
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
                        mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 0}) + "\n")
                        new_keys.add(key)
                        hn_cnt += 1
                elif has_ink == 1 and score <= hp_cut:
                    if key not in existing_keys and key not in new_keys:
                        mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 1}) + "\n")
                        new_keys.add(key)
                        hp_cnt += 1

            full_pred = np.concatenate([t_pred, v_pred], axis=1)
            all_pred_data.append((full_pred, t_pred, depth_start, depth_end))

        mining_f.write(json.dumps({"_type": "meta", "hard_negatives": hn_cnt, "hard_positives": hp_cnt}) + "\n")
        mining_f.close()
        print(f"[HARD][Eval] Finished mining epoch {epoch}: neg={hn_cnt} pos={hp_cnt}")

        self.writer.add_scalar("HardMining/HardNegatives", hn_cnt, epoch)
        self.writer.add_scalar("HardMining/HardPositives", hp_cnt, epoch)

        fig = self._create_hard_examples_overlay(mining_path)
        if fig is not None:
            self.writer.add_figure(f"HardMined/Overlay", fig, epoch)
            plt.close(fig)

        if all_pred_data:
            labels_crop = self.labels[self.y_range[0]:self.y_range[1], self.train_x_range[0]:self.valid_x_range[1]]
            for pred_data in all_pred_data:
                depth_start = pred_data[2]
                depth_end = pred_data[3]
                fig = self._create_evaluation_figure(pred_data, labels_crop)
                self.writer.add_figure(f'Evaluation/Depth_Block_{depth_start}-{depth_end}', fig, epoch)
                plt.close(fig)

        self._run_and_log_hard_mining_evaluation(epoch, model)

    def _run_and_log_hard_mining_evaluation(self, current_epoch, model):
        """evaluate previously mined files and log metrics"""
        print("Starting hard-mining file evaluation...")
        try:
            hm_dir = "./hard_negs"
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
        """add test figures for test and scroll4 data"""
        print("Starting test figure generation...")
        model.eval()

        self._add_single_test_figure(epoch, model, self.test_volume, self.test_mask, self.test_y_range, self.test_x_range, self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max, "Test")
        
        self._add_single_test_figure(epoch, model, self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range, self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max, "Scroll4")

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

    def _create_evaluation_figure(self, pred_data, labels):
        """create evaluation figure for a single depth block"""
        full_pred, train_pred, d_start, d_end = pred_data

        fig, axes = plt.subplots(1, 2, figsize=(15, 9))

        tile = self.c.data.tile_size
        d_labels = labels[::tile, ::tile]

        scaled_labels = ndimage.zoom(d_labels, 1, order=0)

        ax_pred = axes[0]
        im1 = ax_pred.imshow(full_pred, cmap='inferno', vmin=0, vmax=1, aspect='equal')
        ax_pred.set_title(f'Predictions (Depth {d_start}-{d_end})', fontsize=9)

        split_pos = train_pred.shape[1] - 0.5
        ax_pred.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_pred.axis('off')

        ax_overlay = axes[1]
        ax_overlay.imshow(full_pred, cmap='inferno', vmin=0, vmax=1, aspect='equal')
        ax_overlay.set_title(f'Overlay (Depth {d_start}-{d_end})', fontsize=9)

        if scaled_labels is not None:
            overlay = np.zeros((*full_pred.shape, 4))
            h = min(scaled_labels.shape[0], overlay.shape[0])
            w = min(scaled_labels.shape[1], overlay.shape[1])
            overlay[:h, :w][scaled_labels[:h, :w] > 0.5] = [1, 1, 1, 0.4]
            ax_overlay.imshow(overlay)

        ax_overlay.axvline(x=split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_overlay.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
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
            im = ax.imshow(pred, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            ax.set_title(f'Depth Block {d_start}-{d_end}', fontsize=9)
            ax.axis('off')

        for idx in range(len(all_data), rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
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
        for name, p in model.named_parameters():
            if p.requires_grad:
                data = p.data.detach().cpu().numpy()
                if data.size > 0 and not np.isnan(data).all():
                    try:
                        self.writer.add_histogram(f"Weights/{name}", data, epoch)
                    except ValueError as e:
                        print(f"[WARNING] Could not log histogram for Weights/{name}: {e}")

                if p.grad is not None:
                    g = p.grad.detach().cpu().numpy()
                    if g.size > 0 and not np.isnan(g).all() and np.abs(g).sum() > 0:
                        try:
                            self.writer.add_histogram(f"Gradients/{name}", g, epoch)
                        except ValueError as e:
                            print(f"[WARNING] Could not log histogram for Gradients/{name}: {e}")

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
        hm_dir = "./hard_negs"
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