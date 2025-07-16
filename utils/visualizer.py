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
from .dataloader import get_test_dataset, load_scroll4_data, get_tile_coords_for_split, generate_tile_coords
import scipy.ndimage as ndimage
from collections import defaultdict

class TensorboardVisualizer:
    def __init__(self, config: Config):
        self.config = config

        if config.experiment_name is None:
            experiment_name = f"ink_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = config.experiment_name + "_" +  datetime.now().strftime('%d_%H%M%S')
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        
        # Enhanced layout with class-wise metrics
        self.layout = {
            "Training_Overview": {
                "loss": ["Multiline", ["Training_Metrics/Loss/Train", "Training_Metrics/Loss/Train_Raw", "Training_Metrics/Loss/Validation"]],
                "accuracy": ["Multiline", ["Training_Metrics/Accuracy/Train", "Training_Metrics/Accuracy/Validation"]],
            },
            "Classification_Metrics": {
                "precision_recall": ["Multiline", ["Classification/Precision/Train", "Classification/Precision/Validation", 
                                                   "Classification/Recall/Train", "Classification/Recall/Validation"]],
                "f1_specificity": ["Multiline", ["Classification/F1/Train", "Classification/F1/Validation",
                                                  "Classification/Specificity/Train", "Classification/Specificity/Validation"]],
            },
            "AUC_Metrics": {
                "roc_auc": ["Multiline", ["AUC/ROC_AUC/Train", "AUC/ROC_AUC/Validation"]],
                "pr_auc": ["Multiline", ["AUC/PR_AUC/Train", "AUC/PR_AUC/Validation"]],
            },
        }
        
        self.writer = SummaryWriter(self.log_path)
        self.writer.add_custom_scalars(self.layout)
        # self.test_volume = get_test_dataset(self.config)
        # self.scroll4_volume = load_scroll4_data(self.config)

        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")
    
    def log_epoch_metrics(self, epoch, model, train_metrics, val_metrics, learning_rate, time_elapsed, volume, labels, params):
        """Log comprehensive metrics including class-wise metrics"""
        print(f"Logging metrics for epoch: {epoch+1}")
        
        # Original loss and accuracy metrics
        self.writer.add_scalar("Training_Metrics/Loss/Train", train_metrics['loss'], epoch)
        self.writer.add_scalar("Training_Metrics/Loss/Train_Raw", train_metrics['raw_loss'], epoch)
        self.writer.add_scalar("Training_Metrics/Loss/Validation", val_metrics['loss'], epoch)
        
        self.writer.add_scalar("Training_Metrics/Accuracy/Train", train_metrics['accuracy'], epoch)
        self.writer.add_scalar("Training_Metrics/Accuracy/Validation", val_metrics['accuracy'], epoch)
        
        # Classification metrics
        self.writer.add_scalar("Classification/Precision/Train", train_metrics['precision'], epoch)
        self.writer.add_scalar("Classification/Precision/Validation", val_metrics['precision'], epoch)
        self.writer.add_scalar("Classification/Recall/Train", train_metrics['recall'], epoch)
        self.writer.add_scalar("Classification/Recall/Validation", val_metrics['recall'], epoch)
        self.writer.add_scalar("Classification/F1/Train", train_metrics['f1'], epoch)
        self.writer.add_scalar("Classification/F1/Validation", val_metrics['f1'], epoch)
        self.writer.add_scalar("Classification/Specificity/Train", train_metrics['specificity'], epoch)
        self.writer.add_scalar("Classification/Specificity/Validation", val_metrics['specificity'], epoch)
        
        # AUC metrics
        self.writer.add_scalar("AUC/ROC_AUC/Train", train_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/ROC_AUC/Validation", val_metrics['roc_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Train", train_metrics['pr_auc'], epoch)
        self.writer.add_scalar("AUC/PR_AUC/Validation", val_metrics['pr_auc'], epoch)
        
        # Learning rate and time
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        self.writer.add_scalar('Time_Elapsed', time_elapsed, epoch)

        # Create and log confusion matrix
        self.log_confusion_matrix(train_metrics, val_metrics, epoch)
        
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
            self.log_hyperparameters(params)

        # Add evaluation figures at specified intervals
        if (epoch+1) % self.config.training.evaluation_interval == 0:
            self.add_evaluation_figures(epoch, model, volume, labels)

        # # Add test figures at specified intervals
        # if (epoch+1) % self.config.training.test_interval == 0:
        #     self.add_test_figures(epoch, model, self.test_volume, "scroll1")
        #     self.add_test_figures(epoch, model, self.scroll4_volume, "scroll4")
        
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
        
        # Validation confusion matrix
        val_tp = val_metrics['positive_samples'] * val_metrics['recall']
        val_fp = val_tp * (1 - val_metrics['precision']) / val_metrics['precision'] if val_metrics['precision'] > 0 else 0
        val_fn = val_metrics['positive_samples'] - val_tp
        val_tn = val_metrics['negative_samples'] - val_fp
        
        val_cm = np.array([[val_tn, val_fp], [val_fn, val_tp]])
        
        sns.heatmap(val_cm, annot=True, fmt='.0f', cmap='Oranges', ax=ax2,
                   xticklabels=['Predicted No Ink', 'Predicted Ink'],
                   yticklabels=['Actual No Ink', 'Actual Ink'])
        ax2.set_title(f'Validation Confusion Matrix\nPrecision: {val_metrics["precision"]:.3f}, Recall: {val_metrics["recall"]:.3f}')
        
        plt.tight_layout()
        self.writer.add_figure('Confusion_Matrix', fig, epoch)
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
        bars2 = ax1.bar(x + width/2, val_values, width, label='Validation', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Training vs Validation Metrics Comparison')
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
        radar_ax.plot(angles, val_values_radar, 'o-', linewidth=2, label='Validation', color='red')
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

    def group_by_depth(self, coords):
        grouped = defaultdict(list)
        for d_off, y_off, x_off in coords:
            grouped[d_off].append((d_off, y_off, x_off))
        return grouped
    
    def add_evaluation_figures(self, epoch, model, volume, labels):
        print("Starting evaluation figure generation...")
        model.eval()
        # Use the exact same tile logic as the dataloader
        train_coords, y_range, train_x_range, z_range = get_tile_coords_for_split(self.config, 'train')
        valid_coords, _, valid_x_range, _ = get_tile_coords_for_split(self.config, 'valid')
        print(f"[EVAL] y_range: {y_range}, train_x_range: {train_x_range}, valid_x_range: {valid_x_range}")
        # Group tile coordinates by depth block (z_offset)

        train_grouped = self.group_by_depth(train_coords)
        valid_grouped = self.group_by_depth(valid_coords)
        all_depth_offsets = sorted(set(train_grouped.keys()) | set(valid_grouped.keys()))
        all_predictions_data = []
        for d_off in all_depth_offsets:
            depth_start = d_off + self.config.data.start_level
            depth_end = depth_start + self.config.data.depth
            train_block_coords = train_grouped.get(d_off, [])
            valid_block_coords = valid_grouped.get(d_off, [])
            train_predictions = self._predict_tiles(model, volume, train_block_coords, y_range, train_x_range, depth_start, volume_name="training")
            valid_predictions = self._predict_tiles(model, volume, valid_block_coords, y_range, valid_x_range, depth_start, volume_name="validation")
            print(f"[EVAL] train_predictions shape: {train_predictions.shape}, valid_predictions shape: {valid_predictions.shape}")
            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            all_predictions_data.append((full_predictions, train_predictions, depth_start, depth_end))
        if all_predictions_data:
            cropped_labels = labels[y_range[0]:y_range[1], train_x_range[0]:valid_x_range[1]]
            print(f"[EVAL] cropped_labels shape: {cropped_labels.shape}")
            fig = self._create_combined_evaluation_figure(all_predictions_data, cropped_labels, len(all_predictions_data))
            self.writer.add_figure('Evaluation/All_Depth_Blocks', fig, epoch)
            plt.close(fig)

    def _predict_tiles(self, model, volume, block_coords, y_range, x_range, depth_start, volume_name="training"):
        tile_size = self.config.data.tile_size
        region_H = y_range[1] - y_range[0]
        region_W = x_range[1] - x_range[0]
        prediction_map = np.zeros((region_H, region_W), dtype=np.float32)
        batch_size = self.config.dataloader.batch_size
        if torch.cuda.is_available():
            device = self.config.device
        else:
            device = "cpu"
        from tqdm import tqdm
        desc = f"Predicting tiles (depth {depth_start}-{depth_start + self.config.data.depth})"
        # Prepare all tile info for this block
        tile_infos = []
        for d_off, y_off, x_off in block_coords:
            # Apply start_level offset only for training/validation volumes
            if volume_name in ["training", "validation"]:
                d = d_off + self.config.data.start_level
            else:
                d = d_off
            if not (depth_start <= d < depth_start + self.config.data.depth):
                continue
            y = y_range[0] + y_off
            x = x_range[0] + x_off
            tile_infos.append((d, y, x, y_off, x_off))
        # Batched inference
        with torch.no_grad():
            for i in tqdm(range(0, len(tile_infos), batch_size), desc=desc, leave=True):
                batch = tile_infos[i:i+batch_size]
                batch_blocks = []
                batch_indices = []
                for d, y, x, y_off, x_off in batch:
                    # Skip if the block would exceed the available depth
                    if d + self.config.data.depth > volume.shape[0]:
                        continue
                    block = np.array(volume[d:d+self.config.data.depth, y:y+tile_size, x:x+tile_size]).astype(np.float32)
                    # Only normalize for training/validation
                    if volume_name in ["training", "validation"]:
                        block = block / 65535.0
                    if block.shape != (self.config.data.depth, tile_size, tile_size):
                        print(f"Block shape mismatch: {block.shape} != ({self.config.data.depth}, {tile_size}, {tile_size})")
                        continue
                    batch_blocks.append(block)
                    batch_indices.append((y_off, x_off))
                if not batch_blocks:
                    continue
                batch_tensor = torch.from_numpy(np.stack(batch_blocks)).float().unsqueeze(1).to(device)  # [B, 1, D, H, W]
                logits = model(batch_tensor)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                for (y_off, x_off), pred in zip(batch_indices, preds):
                    y = y_off
                    x = x_off
                    prediction_map[y:y+tile_size, x:x+tile_size] = pred
                del batch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prediction_map

    def _create_combined_evaluation_figure(self, all_predictions_data, labels, num_depth_blocks, scale_factor=0.3):
        
        # Calculate figure dimensions based on scaled data
        fig_height = 6 * num_depth_blocks
        fig_width = 10
        
        # Create subplots directly with minimal spacing
        fig, axes = plt.subplots(num_depth_blocks, 2, figsize=(fig_width, fig_height))
        
        # Handle single row case
        if num_depth_blocks == 1:
            axes = axes.reshape(1, -1)
        
        for block_idx, (full_predictions, train_predictions, depth_start, depth_end) in enumerate(all_predictions_data):
            
            # Scale the prediction arrays
            scaled_full_predictions = ndimage.zoom(full_predictions, scale_factor, order=1)
            scaled_train_predictions = ndimage.zoom(train_predictions, scale_factor, order=1)
            scaled_labels = ndimage.zoom(labels, scale_factor, order=0)
            
            # Left plot: Model predictions
            ax_pred = axes[block_idx, 0]
            im1 = ax_pred.imshow(scaled_full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            ax_pred.set_title(f'Depth Block {depth_start}-{depth_end}', fontsize=9)
            
            # Adjust the dividing line position based on scaling
            train_split_pos = scaled_train_predictions.shape[1] - 0.5
            ax_pred.axvline(x=train_split_pos, color='red', linestyle='--', linewidth=1.2)
            ax_pred.axis('off')
            
            # Right plot: Predictions + Ground Truth overlay
            ax_overlay = axes[block_idx, 1]
            ax_overlay.imshow(scaled_full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
             
            if scaled_labels is not None:
                label_overlay = np.zeros((*scaled_labels.shape, 4))
                label_overlay[scaled_labels > 0.5] = [1, 1, 1, 0.4]  # White with 40% opacity
                ax_overlay.imshow(label_overlay)
            
            ax_overlay.axvline(x=train_split_pos, color='red', linestyle='--', linewidth=1.2)
            ax_overlay.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, right=0.9, top=0.90, bottom=0.10)
        return fig


    def _create_combined_test_figure(self, all_predictions_data, num_depth_blocks, scale_factor, test_type):
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
            scaled_predictions = ndimage.zoom(predictions, scale_factor, order=1)
            im = ax1.imshow(scaled_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            ax1.set_title(f'Depth Block {depth_start}-{depth_end}', fontsize=9)
            ax1.axis('off')
        # Hide any unused subplots
        for idx in range(len(all_predictions_data), rows * cols):
            ax = axes[idx // cols, idx % cols]
            ax.axis('off')
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        return fig

    def add_test_figures(self, epoch, model, test_volume, test_type):
        """
        Run test evaluation and create one combined figure with all depth blocks
        No ground truth overlay for test data
        Uses unified logic based on scroll1's correct tile coordinate generation
        """
        if test_type not in ["scroll1", "scroll4"]:
            print(f"Invalid test type: {test_type}. Expected 'scroll1' or 'scroll4'.")
            return
            
        print("Starting test figure generation...")
        model.eval()
        D = test_volume.shape[0]
        all_predictions_data = []
        
        # Unified logic - use scroll1's exact approach for both test types
        y_range = (0, test_volume.shape[1])
        x_range = (0, test_volume.shape[2])
        tile_size = self.config.data.tile_size
        depth = self.config.data.depth
        z_range = (0, D)
        print("test_volume.shape:", test_volume.shape)
        print("tile_size:", tile_size)
        print("depth:", depth)
        print("z_range:", z_range)
        test_coords = generate_tile_coords(z_range, y_range, x_range, self.config, test_volume)
        grouped = self.group_by_depth(test_coords)
        all_depth_offsets = sorted(grouped.keys())
        
        for depth_start in all_depth_offsets:
            block_coords = grouped[depth_start]
            
            volume_name = f"test_{test_type}"
            predictions = self._predict_tiles(
                model, test_volume, block_coords, y_range, x_range, depth_start, volume_name=volume_name
            )
            print(f"[TEST] predictions shape: {predictions.shape}")
            depth_end = depth_start + depth
            all_predictions_data.append((predictions, depth_start, depth_end))
        
        if all_predictions_data:
            print(f"got {len(all_predictions_data)} depth blocks for test")
            fig = self._create_combined_test_figure(all_predictions_data, len(all_predictions_data), 0.3, test_type)
            self.writer.add_figure(f'Test/{test_type.capitalize()}_All_Depth_Blocks', fig, epoch)
            plt.close(fig)  # Important: close figure to free memory
        
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

    def log_hyperparameters(self, params):
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
    
    def close(self):
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_path}")