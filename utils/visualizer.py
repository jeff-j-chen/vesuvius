from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing import _32Bit
import os
from typing import Any, Literal
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from .config import Config
from .dataloader import get_test_dataset, load_scroll4_data, get_tile_coords_for_split, generate_tile_coords, get_or_compute_normalization, load_tv_data
import scipy.ndimage as ndimage
from collections import defaultdict
import json

def group_by_depth(coords):
    """Group tile coordinates by their depth offset."""
    grouped = defaultdict(list)
    for d_off, y_off, x_off in coords:
        grouped[d_off].append((d_off, y_off, x_off))
    return grouped

def predict_tiles(config, model, volume, mask, block_coords, y_range, x_range, depth_start, volume_name, global_mean, global_std, global_min, global_max):
    tile_size = config.data.tile_size
    region_H = y_range[1] - y_range[0]
    region_W = x_range[1] - x_range[0]
    
    # Create a downsampled prediction map
    downsampled_H = region_H // tile_size
    downsampled_W = region_W // tile_size
    prediction_map = np.zeros((downsampled_H, downsampled_W), dtype=np.float32)
    
    batch_size = config.dataloader.batch_size
    device = config.device if torch.cuda.is_available() else "cpu"
    
    desc = f"Predicting tiles (depth {depth_start}-{depth_start + config.data.depth})"
    
    # Prepare all tile info for this block
    tile_infos = []
    for d_off, y_off, x_off in block_coords:
        # Apply start_level offset only for training/validation volumes
        if volume_name in ["training", "validation"]:
            d = d_off + config.data.start_level
        else:
            d = d_off
        if not (depth_start <= d < depth_start + config.data.depth):
            continue
        y = y_range[0] + y_off
        x = x_range[0] + x_off
        tile_infos.append((d, y, x, y_off, x_off))
    
    debug_samples = []
    debug_limit = 50
    did_debug_log = False
    # Batched inference
    with torch.no_grad():
        for i in tqdm(range(0, len(tile_infos), batch_size), desc=desc, leave=True):
            batch = tile_infos[i:i+batch_size]
            batch_blocks = []
            batch_indices = []
            
            for d, y, x, y_off, x_off in batch:
                # Skip if the block would exceed the available depth
                if d + config.data.depth > volume.shape[0]:
                    continue
                
                block = np.array(volume[d:d+config.data.depth, y:y+tile_size, x:x+tile_size]).astype(np.float32)
                
                # Apply normalization
                block = (block - global_mean) / global_std
                
                # Apply mask
                if block.ndim == 3 and mask.ndim == 2:
                    mask_tile = mask[y:y+tile_size, x:x+tile_size]
                    # Previous code cast to uint8, turning fractional mask values (<1) into 0 (problem for scroll4).
                    # Treat any positive value as valid region.
                    binary_mask = (mask_tile > 0).astype(np.uint8)
                    mask_exp = np.expand_dims(binary_mask, axis=0)
                    mask_exp = np.broadcast_to(mask_exp, block.shape)
                    block[mask_exp == 0] = 0
                
                block = (block - global_min) / (global_max - global_min)
                block = np.clip(block, 0, 1)
                
                if block.shape != (config.data.depth, tile_size, tile_size):
                    print(f"Block shape mismatch: {block.shape} != ({config.data.depth}, {tile_size}, {tile_size})")
                    continue
                
                batch_blocks.append(block)
                batch_indices.append((y_off, x_off))
            
            if not batch_blocks:
                continue
            
            batch_tensor = torch.from_numpy(np.stack(batch_blocks)).float().unsqueeze(1).to(device)
            logits = model(batch_tensor)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            
            for (y_off, x_off), pred in zip(batch_indices, preds):
                # Update the downsampled map at the corresponding pixel
                y_idx = y_off // tile_size
                x_idx = x_off // tile_size
                if 0 <= y_idx < downsampled_H and 0 <= x_idx < downsampled_W:
                    prediction_map[y_idx, x_idx] = pred
            
            del batch_tensor
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction_map

class TensorboardVisualizer:
    def __init__(self, config: Config, mode: str = 'train'):
        self.config = config
        self.mode = mode

        if config.experiment_name is None:
            if self.mode == 'finetune':
                experiment_name = f"finetune_{datetime.now().strftime('%d.%m_%H:%M:%S')}"
            else:
                experiment_name = f"ink_detection_{datetime.now().strftime('%d.%m_%H:%M:%S')}"
        else:
            experiment_name = config.experiment_name + "_" +  datetime.now().strftime('%d_%H:%M:%S')
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        
        # Enhanced layout with class-wise metrics
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

        if self.mode == 'train':
            self.volume, self.mask, self.labels, _, _, _ = load_tv_data(self.config)
            self.global_mean, self.global_std, self.global_min, self.global_max = get_or_compute_normalization(self.config.data.train_segment_id, self.volume, self.mask)
            
            # Load test datasets with normalization
            self.test_volume, self.test_mask, self.test_y_range, self.test_x_range = get_test_dataset(self.config)
            self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max = get_or_compute_normalization(self.config.data.train_segment_id, self.test_volume, self.test_mask)

            self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range = load_scroll4_data(self.config)
            self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max = get_or_compute_normalization(self.config.data.scroll4_segment_id, self.scroll4_volume, self.scroll4_mask)
        
            self._debug_scroll4_ranges_once()
        
        self.writer = SummaryWriter(self.log_path)
        self.writer.add_custom_scalars(self.layout)

        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")

    def log_epoch_metrics(self, epoch, model, train_metrics, val_metrics, learning_rate, time_elapsed, params, pos_weight):
        """Log comprehensive metrics including class-wise metrics"""
        print(f"Logging metrics for epoch: {epoch+1}")
        
        # Original loss and accuracy metrics
        self.writer.add_scalar("G_M/Loss/Train", train_metrics['loss'], epoch)
        self.writer.add_scalar("G_M/Loss/Train_Raw", train_metrics['raw_loss'], epoch)
        self.writer.add_scalar("G_M/Loss/Valid", val_metrics['loss'], epoch)
        
        self.writer.add_scalar("G_M/Acc/Train", train_metrics['accuracy'], epoch)
        self.writer.add_scalar("G_M/Acc/Valid", val_metrics['accuracy'], epoch)
        
        # P_M metrics
        self.writer.add_scalar("P_M/Precision/Train", train_metrics['precision'], epoch)
        self.writer.add_scalar("P_M/Precision/Valid", val_metrics['precision'], epoch)
        self.writer.add_scalar("P_M/Recall/Train", train_metrics['recall'], epoch)
        self.writer.add_scalar("P_M/Recall/Valid", val_metrics['recall'], epoch)
        self.writer.add_scalar("P_M/F1_Score/Train", train_metrics['f1'], epoch)
        self.writer.add_scalar("P_M/F1_Score/Valid", val_metrics['f1'], epoch)
        self.writer.add_scalar("P_M/Specificity/Train", train_metrics['specificity'], epoch)
        self.writer.add_scalar("P_M/Specificity/Valid", val_metrics['specificity'], epoch)
        
        # AUC metrics
        self.writer.add_scalar("AUC/ROC_AUC/Train", train_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/ROC_AUC/Valid", val_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Train", train_metrics['pr_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Valid", val_metrics['pr_auc'], epoch)
        
        # Learning rate and time
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        self.writer.add_scalar('Time_Elapsed', time_elapsed, epoch)

        # Create and log confusion matrix
        self.log_confusion_matrix(train_metrics, val_metrics, epoch)
        
        # Create and log output histogram
        self.log_output_histogram(train_metrics, val_metrics, epoch)
        
        # Create and log metrics comparison chart
        self.log_metrics_comparison(train_metrics, val_metrics, epoch)

        # Log weight histograms
        self.log_weight_histograms(model, epoch)

        # Log model graph once at the beginning
        if epoch == 0:
            print("Logging hyperparameters and model graph")
            example_input = torch.randn(1, self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size).to(self.config.device)
            example_input = example_input.unsqueeze(0)
            # self.log_model_graph(model, example_input)
            self.log_hyperparameters(params, pos_weight)

        # Add evaluation figures at specified intervals
        if self.mode == 'train' and (epoch+1) % self.config.training.evaluation_interval == 0:
            self.add_evaluation_figures(epoch, model)

        # Add test figures at specified intervals
        if self.mode == 'train' and (epoch+1) % self.config.training.test_interval == 0:
            self.add_test_figures(epoch, model)
        
        self.writer.flush()

    def log_confusion_matrix(self, train_metrics, val_metrics, epoch):
        """Create and log confusion matrix visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training confusion matrix
        train_tp = train_metrics['positive_samples'] * train_metrics['recall']
        train_fp = train_tp * (1 - train_metrics['precision']) / train_metrics['precision'] if train_metrics['precision'] > 0 else 0
        train_fn = train_metrics['positive_samples'] - train_tp
        train_tn = train_metrics['negative_samples'] - train_fp
        
        train_cm = np.array([[train_tn, train_fp], [train_fn, train_tp]])
        
        sns.heatmap(train_cm, annot=True, fmt='.0f', cmap='Blues', ax=ax1,
                   xticklabels=['Predicted No Ink', 'Predicted Ink'],
                   yticklabels=['Actual No Ink', 'Actual Ink'])
        ax1.set_title(f'Training Confusion Matrix\nPrecision: {train_metrics["precision"]:.3f}, Recall: {train_metrics["recall"]:.3f}')
        
        # Valid confusion matrix
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
        """Create and log histogram of model outputs for training and validation"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bins = np.linspace(0, 1, 51)
        
        # Plot overlapping histograms
        ax.hist(train_metrics['scores'], bins=bins, alpha=0.6, label='Training', color='skyblue', edgecolor='black', density=True) # type: ignore
        ax.hist(val_metrics['scores'], bins=bins, alpha=0.6, label='Validation', color='lightcoral', edgecolor='black', density=True) # type: ignore
        
        ax.set_xlabel('Model Output (Sigmoid Score)')
        ax.set_ylabel('Density')
        ax.set_title('Model Output Distribution\nTraining vs Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        # Add vertical line at decision boundary
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        self.writer.add_figure('Output_Histogram', fig, epoch)
        plt.close(fig)

    def log_metrics_comparison(self, train_metrics, val_metrics, epoch):
        """Create and log a comprehensive metrics comparison chart"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # Creates a 1D array of axes
        
        # Metrics to compare
        metrics_to_plot = ['precision', 'recall', 'f1', 'specificity', 'roc_auc', 'pr_auc']
        
        # Bar plot comparing train vs validation
        ax1 = axes[0]  # First subplot
        train_values = [train_metrics[metric] for metric in metrics_to_plot]
        val_values = [val_metrics[metric] for metric in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_values, width, label='Train', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, val_values, width, label='Valid', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Training vs Valid Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)
        
        # Performance radar chart
        ax2 = axes[1]  # Second subplot
        categories = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC', 'PR-AUC']
        
        # Create angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Add values for train and validation
        train_values_radar = [train_metrics[metric] for metric in metrics_to_plot]
        val_values_radar = [val_metrics[metric] for metric in metrics_to_plot]
        train_values_radar += train_values_radar[:1]  # Complete the circle
        val_values_radar += val_values_radar[:1]  # Complete the circle
        
        # Radar chart must use a polar axis
        radar_ax = fig.add_subplot(1, 2, 2, projection='polar')
        radar_ax.plot(angles, train_values_radar, 'o-', linewidth=2, label='Train', color='blue')
        radar_ax.fill(angles, train_values_radar, alpha=0.25, color='blue')
        radar_ax.plot(angles, val_values_radar, 'o-', linewidth=2, label='Valid', color='red')
        radar_ax.fill(angles, val_values_radar, alpha=0.25, color='red')
        
        radar_ax.set_xticks(angles[:-1])
        radar_ax.set_xticklabels(categories)
        radar_ax.set_ylim(0, 1)
        radar_ax.set_title('Performance Radar Chart', y=1.08)
        radar_ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        radar_ax.grid(True)
        
        plt.tight_layout()
        self.writer.add_figure('Metrics_Comparison', fig, epoch)
        plt.close(fig)
    
    def log_performance_trends(self, train_metrics, val_metrics, epoch):
        """Log performance trend analysis"""
        # Create a summary figure showing key performance indicators
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Performance interpretation text
        performance_text = []
        
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
        val_pr_auc = val_metrics['pr_auc']
        val_pos_ratio = val_metrics['positive_ratio']
        
        # Analyze performance
        if val_precision > 0.7 and val_recall > 0.6:
            performance_text.append("✓ EXCELLENT: High precision and recall")
        elif val_precision > 0.5 and val_recall > 0.4:
            performance_text.append("~ GOOD: Moderate precision and recall")
        else:
            performance_text.append("✗ NEEDS IMPROVEMENT: Low precision or recall")
        
        if val_pr_auc > 0.4:
            performance_text.append("✓ EXCELLENT: High PR-AUC score")
        elif val_pr_auc > 0.2:
            performance_text.append("~ GOOD: Moderate PR-AUC score")
        else:
            performance_text.append("✗ NEEDS IMPROVEMENT: Low PR-AUC score")
        
        # Class imbalance analysis
        if val_pos_ratio < 0.1:
            performance_text.append(f"⚠ SEVERE IMBALANCE: Only {val_pos_ratio:.1%} positive samples")
        elif val_pos_ratio < 0.3:
            performance_text.append(f"⚠ MODERATE IMBALANCE: {val_pos_ratio:.1%} positive samples")
        else:
            performance_text.append(f"✓ BALANCED: {val_pos_ratio:.1%} positive s")
        
        # Model behavior analysis
        if val_precision > val_recall + 0.1:
            performance_text.append("📊 MODEL BEHAVIOR: Conservative (few false positives)")
        elif val_recall > val_precision + 0.1:
            performance_text.append("📊 MODEL BEHAVIOR: Aggressive (few false negatives)")
        else:
            performance_text.append("📊 MODEL BEHAVIOR: Balanced precision/recall")
        
        ax.text(0.1, 0.9, '\n'.join(performance_text), transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Analysis - Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        self.writer.add_figure('Performance_Analysis', fig, epoch)
        plt.close(fig)

    def add_evaluation_figures(self, epoch, model):
        print("Starting evaluation figure generation...")
        model.eval()
        
        train_coords, y_range, train_x_range, z_range = get_tile_coords_for_split(self.config, 'train')
        valid_coords, _, valid_x_range, _ = get_tile_coords_for_split(self.config, 'valid')
        
        train_grouped = group_by_depth(train_coords)
        valid_grouped = group_by_depth(valid_coords)
        all_depth_offsets = sorted(set(train_grouped.keys()) | set(valid_grouped.keys()))
        all_predictions_data = []
        # Hard mining setup
        do_mining = self.config.hard_mining.next_iter_ratio > 0
        if do_mining:
            # Ensure mining file is written where HardMiningManager expects
            mining_path = os.path.join(self.log_path, f"hard_mining_epoch_{epoch}.jsonl")
            os.makedirs(self.log_path, exist_ok=True)
            mining_f = open(mining_path, "w")
            print(f"[HARD][Eval] Writing mining file to: {mining_path}")
            hn_cut = self.config.hard_mining.hard_negative_cutoff
            hp_cut = self.config.hard_mining.hard_positive_cutoff
            hard_neg_cnt = 0
            hard_pos_cnt = 0
        
        for d_off in all_depth_offsets:
            depth_start = d_off + self.config.data.start_level
            depth_end = depth_start + self.config.data.depth
            train_block_coords = train_grouped.get(d_off, [])
            valid_block_coords = valid_grouped.get(d_off, [])
            train_predictions = predict_tiles(self.config, model, self.volume, self.mask, train_block_coords, y_range, train_x_range, depth_start, "training", self.global_mean, self.global_std, self.global_min, self.global_max)
            valid_predictions = predict_tiles(self.config, model, self.volume, self.mask, valid_block_coords, y_range, valid_x_range, depth_start, "validation", self.global_mean, self.global_std, self.global_min, self.global_max)

            # Hard mining (per train tile)
            if do_mining and train_block_coords:
                tile_size = self.config.data.tile_size
                # Map coords to downsample indices
                for (z_off, y_off, x_off) in train_block_coords:
                    # indices
                    y_idx = y_off // tile_size
                    x_idx = x_off // tile_size
                    if y_idx < 0 or y_idx >= train_predictions.shape[0] or x_idx < 0 or x_idx >= train_predictions.shape[1]:
                        continue
                    score = float(train_predictions[y_idx, x_idx])
                    # Global coordinates
                    z_global = depth_start
                    y_global = y_range[0] + y_off
                    x_global = train_x_range[0] + x_off
                    # Label
                    label_tile = self.labels[
                        y_global:y_global+tile_size,
                        x_global:x_global+tile_size
                    ]
                    has_ink = int(np.any(label_tile > 0.5))
                    # Decide hard
                    if has_ink == 0 and score >= hn_cut:
                        mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 0}) + "\n")
                        hard_neg_cnt += 1
                    elif has_ink == 1 and score <= hp_cut:
                        mining_f.write(json.dumps({"z": z_global, "y": y_global, "x": x_global, "score": score, "label": 1}) + "\n")
                        hard_pos_cnt += 1

            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            all_predictions_data.append((full_predictions, train_predictions, depth_start, depth_end))
        
        if do_mining:
            # Meta line with counts
            mining_f.write(json.dumps({"_type": "meta", "hard_negatives": hard_neg_cnt, "hard_positives": hard_pos_cnt}) + "\n")
            mining_f.close()
            print(f"[HARD][Eval] Finished mining epoch {epoch}: neg={hard_neg_cnt} pos={hard_pos_cnt}")
            # Log counts
            self.writer.add_scalar("HardMining/HardNegatives", hard_neg_cnt, epoch)
            self.writer.add_scalar("HardMining/HardPositives", hard_pos_cnt, epoch)

        if all_predictions_data:
            cropped_labels = self.labels[y_range[0]:y_range[1], train_x_range[0]:valid_x_range[1]]
            for prediction_data in all_predictions_data:
                depth_start = prediction_data[2]
                depth_end = prediction_data[3]
                fig = self._create_evaluation_figure(prediction_data, cropped_labels)
                self.writer.add_figure(f'Evaluation/Depth_Block_{depth_start}-{depth_end}', fig, epoch)
                plt.close(fig)



    def add_test_figures(self, epoch, model):
        """Add test figures for both test and scroll4 data"""
        print("Starting test figure generation...")
        model.eval()
        
        # Test data (bottom section of training segment)
        self._add_single_test_figure(epoch, model, self.test_volume, self.test_mask, self.test_y_range, self.test_x_range, self.test_global_mean, self.test_global_std, self.test_global_min, self.test_global_max, "Test")
        
        # Scroll4 data
        self._add_single_test_figure(epoch, model, self.scroll4_volume, self.scroll4_mask, self.scroll4_y_range, self.scroll4_x_range, self.scroll4_global_mean, self.scroll4_global_std, self.scroll4_global_min, self.scroll4_global_max, "Scroll4")

    def _add_single_test_figure(self, epoch, model, volume, mask, y_range, x_range, global_mean, global_std, global_min, global_max, test_name):
        """Generate predictions and create figure for a single test dataset"""
        z_range = (0, volume.shape[0])
        test_coords = generate_tile_coords(z_range, y_range, x_range, self.config, volume)
        grouped = group_by_depth(test_coords)
        all_depth_offsets = sorted(grouped.keys())
        all_predictions_data = []
        
        for depth_start in all_depth_offsets:
            block_coords = grouped[depth_start]
            predictions = predict_tiles(self.config, model, volume, mask, block_coords, y_range, x_range, depth_start, test_name, global_mean, global_std, global_min, global_max)
            depth_end = depth_start + self.config.data.depth
            all_predictions_data.append((predictions, depth_start, depth_end))
        
        if all_predictions_data:
            fig = self._create_combined_test_figure(all_predictions_data, len(all_predictions_data), test_name)
            self.writer.add_figure(f'Test/{test_name}_All_Depth_Blocks', fig, epoch)
            plt.close(fig)

    def _create_evaluation_figure(self, prediction_data, labels):
        """Create an evaluation figure for a single depth block."""
        full_predictions, train_predictions, depth_start, depth_end = prediction_data
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        
        # Downsample labels to match prediction map dimensions
        tile_size = self.config.data.tile_size
        downsampled_labels = labels[::tile_size, ::tile_size]

        # Keep zoom function with factor of 1 as requested, using downsampled data directly.
        # scaled_full_predictions = ndimage.zoom(full_predictions, 1, order=1)
        # scaled_train_predictions = ndimage.zoom(train_predictions, 1, order=1)
        scaled_labels = ndimage.zoom(downsampled_labels, 1, order=0)
        
        # Left plot: Model predictions
        ax_pred = axes[0]
        im1 = ax_pred.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
        ax_pred.set_title(f'Predictions (Depth {depth_start}-{depth_end})', fontsize=9)
        
        # Adjust the dividing line position based on scaling
        train_split_pos = train_predictions.shape[1] - 0.5
        ax_pred.axvline(x=train_split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_pred.axis('off')
        
        # Right plot: Predictions + Ground Truth overlay
        ax_overlay = axes[1]
        ax_overlay.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
        ax_overlay.set_title(f'Overlay (Depth {depth_start}-{depth_end})', fontsize=9)
         
        if scaled_labels is not None:
            # Ensure label overlay matches the shape of the predictions
            label_overlay = np.zeros((*full_predictions.shape, 4))
            # We need to handle the case where downsampled labels might be slightly different in size
            h, w = min(scaled_labels.shape[0], label_overlay.shape[0]), min(scaled_labels.shape[1], label_overlay.shape[1])
            label_overlay[:h, :w][scaled_labels[:h, :w] > 0.5] = [1, 1, 1, 0.4]  # White with 40% opacity
            ax_overlay.imshow(label_overlay)
        
        ax_overlay.axvline(x=train_split_pos, color='red', linestyle='--', linewidth=1.2)
        ax_overlay.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def _create_combined_test_figure(self, all_predictions_data, num_depth_blocks, test_type):
        """Create combined test figure with predictions (no ground truth overlay)"""
        
        cols = 2
        rows = (num_depth_blocks + 1) // 2
        
        fig_width = 10
        height_mult = 7 if test_type == "scroll1" else 3
        fig_height = height_mult * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        # Ensure axes is always 2D for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for block_idx, (predictions, depth_start, depth_end) in enumerate(all_predictions_data):
            ax1 = axes[block_idx // cols, block_idx % cols]
            # scaled_predictions = ndimage.zoom(predictions, 1, order=1)
            im = ax1.imshow(predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            ax1.set_title(f'Depth Block {depth_start}-{depth_end}', fontsize=9)
            ax1.axis('off')
        # Hide any unused subplots
        for idx in range(len(all_predictions_data), rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis('off')
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def log_model_graph(self, model, example_input):
        self.writer.add_graph(model, example_input)
    
    def log_activation_maps(self, activations, epoch):
        for layer, activation_map in activations.items():
            if activation_map.dim() == 5:  # Shape: (B, C, D, H, W)
                # Collapse depth dimension (D) using mean
                activation_map_4d = activation_map.mean(dim=2)  # Shape: (B, C, H, W)
                self.writer.add_images(f"Activations/{layer.__class__.__name__}", activation_map_4d, epoch, dataformats="NCHW")
            elif activation_map.dim() == 2:  # Shape: (B, N)
                # Reshape to (B, 1, N, 1) for compatibility
                activation_map_reshaped = activation_map.unsqueeze(1).unsqueeze(-1)  # Shape: (B, 1, N, 1)
                self.writer.add_images(f"Activations/{layer.__class__.__name__}", activation_map_reshaped, epoch, dataformats="NCHW")
            else:
                raise ValueError(f"Unexpected activation map dimensions: {activation_map.shape}")
    
    def log_weight_histograms(self, model, epoch):
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weights
                data = param.data.detach().cpu().numpy()
                if data.size > 0 and not np.isnan(data).all():
                    try:
                        self.writer.add_histogram(f"Weights/{name}", data, epoch)
                    except ValueError as e:
                        print(f"[WARNING] Could not log histogram for Weights/{name}: {e}")
                # Log gradients (with safety checks)
                if param.grad is not None:
                    grad = param.grad.detach().cpu().numpy()
                    if grad.size > 0 and not np.isnan(grad).all() and np.abs(grad).sum() > 0:
                        try:
                            self.writer.add_histogram(f"Gradients/{name}", grad, epoch)
                        except ValueError as e:
                            print(f"[WARNING] Could not log histogram for Gradients/{name}: {e}")

    def log_hyperparameters(self, params, pos_weight):
        self.writer.add_scalar("Hyperparameters/Tile Size", self.config.data.tile_size)
        self.writer.add_scalar("Hyperparameters/Depth", self.config.data.depth)
        self.writer.add_scalar("Hyperparameters/Batch Size", self.config.dataloader.batch_size)
        self.writer.add_scalar("Hyperparameters/Num Workers", self.config.dataloader.num_workers)
        self.writer.add_scalar("Hyperparameters/Learning Rate", self.config.training.learning_rate)
        self.writer.add_scalar("Hyperparameters/Weight Decay", self.config.training.weight_decay)
        self.writer.add_scalar("Hyperparameters/L1 Lambda", self.config.training.l1_lambda)
        self.writer.add_scalar("Hyperparameters/Conv1 Dropout", self.config.model.conv1_drop)
        self.writer.add_scalar("Hyperparameters/Conv2 Dropout", self.config.model.conv2_drop)
        self.writer.add_scalar("Hyperparameters/FC1 Dropout", self.config.model.fc1_drop)
        self.writer.add_scalar("Hyperparameters/FC2 Dropout", self.config.model.fc2_drop)
        self.writer.add_scalar("Hyperparameters/Max Grad Norm", self.config.training.max_grad_norm)
        self.writer.add_scalar("Hyperparameters/Patience", self.config.training.patience)
        self.writer.add_scalar("Hyperparameters/LR Scheduler Factor", self.config.training.lr_scheduler_factor)
        self.writer.add_scalar("Hyperparameters/Model Complexity", params)
        self.writer.add_scalar("Hyperparameters/Pos Weight", pos_weight)
    
    def close(self):
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_path}")
    
    def _debug_scroll4_ranges_once(self):
        """One-time detailed sanity checks for scroll4 coordinate / mask alignment."""
        try:
            vol = self.scroll4_volume
            mask = self.scroll4_mask
            y_range = self.scroll4_y_range
            x_range = self.scroll4_x_range
            issues = []
            # Shape checks
            if mask.shape != (vol.shape[1], vol.shape[2]):
                issues.append(f"Mask shape {mask.shape} != volume spatial {(vol.shape[1], vol.shape[2])}")
            # Range bounds
            if not (0 <= y_range[0] < y_range[1] <= vol.shape[1]):
                issues.append(f"Y range {y_range} out of bounds (0,{vol.shape[1]})")
            if not (0 <= x_range[0] < x_range[1] <= vol.shape[2]):
                issues.append(f"X range {x_range} out of bounds (0,{vol.shape[2]})")
            # Tile alignment
            tile = self.config.data.tile_size
            if (y_range[0] % tile != 0) or (x_range[0] % tile != 0):
                issues.append(f"Ranges not tile aligned: y_start%tile={y_range[0]%tile}, x_start%tile={x_range[0]%tile}")
            # Mask coverage stats inside region
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