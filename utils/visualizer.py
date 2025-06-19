import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import dataclass
from .config import Config
from typing import Optional, Dict, Any, Tuple, List

class TensorboardVisualizer:
    def __init__(self, config: Config):
        self.config = config

        if config.experiment_name is None:
            experiment_name = f"ink_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = config.experiment_name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_path)
        
        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")
    
    def log_epoch_metrics(self, epoch, train_acc, val_acc, train_loss, val_loss, learning_rate):
        """
        Log all metrics for a single epoch
        
        Args:
            epoch: Current epoch number
            train_acc: Training accuracy
            val_acc: Validation accuracy  
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
        """
        # Log accuracies
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # Log losses
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Log combined accuracy plot
        self.writer.add_scalars('Accuracy_Comparison', {
            'Train': train_acc,
            'Validation': val_acc
        }, epoch)
        
        # Log combined loss plot
        self.writer.add_scalars('Loss_Comparison', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
    
    def _process_volume_depth_block(self, model, volume, volume_name, depth_start, depth_end):
        """Helper function to process a single volume at a specific depth range"""
        model.eval()
        D, H, W = volume.shape
        
        prediction_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Create list of all tile coordinates
        tile_coords = []
        for y in range(0, H - self.config.data.tile_size + 1, self.config.data.tile_size):
            for x in range(0, W - self.config.data.tile_size + 1, self.config.data.tile_size):
                tile_coords.append((y, x))
        
        with torch.no_grad():
            # Process tiles with tqdm progress bar
            for y, x in tqdm(tile_coords, desc=f"Processing {volume_name} volume (depth {depth_start}-{depth_end-1})", leave=False):
                # Extract block from the specified depth range
                block = volume[depth_start:depth_end, y:y+self.config.data.tile_size, x:x+self.config.data.tile_size]
                
                if block.shape == (self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size):
                    block_tensor = torch.from_numpy(block).float().unsqueeze(0).unsqueeze(0).to(self.config.device)
                    logits = model(block_tensor)
                    pred = torch.sigmoid(logits).item()
                    
                    prediction_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += pred
                    count_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += 1
        
        # Normalize predictions
        prediction_map = np.divide(prediction_map, count_map, where=count_map>0)
        return prediction_map
    
    def _create_validation_figure(self, full_volume_slice, full_labels, full_predictions, train_predictions, block_idx, depth_start, depth_end, middle_slice_idx):
        """Create a single validation figure for one depth block"""
        fig = plt.figure(figsize=(24, 6))
        
        # Original slice
        plt.subplot(1, 4, 1)
        plt.imshow(full_volume_slice, cmap='gray')
        plt.title(f'Full Volume (Slice {middle_slice_idx})\nDepth Block {block_idx + 1} ({depth_start}-{depth_end-1})\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 4, 2)
        plt.imshow(full_labels, cmap='binary')
        plt.title(f'Ground Truth Labels\nDepth Block {block_idx + 1}\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Predictions
        plt.subplot(1, 4, 3)
        img = plt.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1)
        plt.colorbar(img, fraction=0.046, pad=0.04)
        plt.title(f'Model Predictions\nDepth Block {block_idx + 1}\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 4, 4)
        plt.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1)
        
        # Create overlay for ground truth
        label_overlay = np.zeros((*full_labels.shape, 4))  # RGBA
        label_overlay[full_labels > 0.5] = [1, 1, 1, 0.4]  # White with transparency
        plt.imshow(label_overlay)
        
        plt.title(f'Predictions + Ground Truth\nDepth Block {block_idx + 1}\nTrain | Valid\n(White = True Labels)')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def log_validation_results(self, epoch: int, model, train_volume, train_labels, valid_volume, valid_labels):
        """
        Run full validation and log all visualization figures to TensorBoard
        
        Args:
            epoch: Current epoch number
            model: PyTorch model to evaluate
            train_volume: Training volume data
            train_labels: Training labels
            valid_volume: Validation volume data  
            valid_labels: Validation labels
        """
        # Calculate number of depth blocks to match dataset creation logic
        D = train_volume.shape[0]
        num_depth_blocks = (D - self.config.data.depth + 1) // self.config.data.depth
        
        # Process each depth block
        for block_idx in range(num_depth_blocks):
            depth_start = block_idx * self.config.data.depth
            depth_end = depth_start + self.config.data.depth
            
            # Process both volumes for this depth block
            train_predictions = self._process_volume_depth_block(
                model, train_volume, "training", depth_start, depth_end
            )
            valid_predictions = self._process_volume_depth_block(
                model, valid_volume, "validation", depth_start, depth_end
            )
            
            # Stitch everything back together horizontally
            # Use the middle slice of the current depth block for visualization
            middle_slice_idx = depth_start + self.config.data.depth // 2
            full_volume_slice = np.concatenate([train_volume[middle_slice_idx], valid_volume[middle_slice_idx]], axis=1)
            full_labels = np.concatenate([train_labels, valid_labels], axis=1)
            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            
            
            # Log some metrics for this depth block
            ink_pixels = (full_labels > 0.5).sum()
            predicted_ink_pixels = (full_predictions > 0.5).sum()
            
            self.writer.add_scalar(f'Validation_Metrics/Ink_Pixels_Block_{block_idx + 1}', ink_pixels, epoch)
            self.writer.add_scalar(f'Validation_Metrics/Predicted_Ink_Pixels_Block_{block_idx + 1}', predicted_ink_pixels, epoch)
            
            # Create and log the validation figure
            fig = self._create_validation_figure(
                full_volume_slice, full_labels, full_predictions, train_predictions,
                block_idx, depth_start, depth_end, middle_slice_idx
            )
            # Log figure to TensorBoard
            self.writer.add_figure(f'Validation/Depth_Block_{block_idx + 1}', fig, epoch)
            
            # Close the figure to free memory
            plt.close(fig)
    
    def should_run_validation(self, epoch):
        return epoch % self.config.training.validation_interval == 0
    
    def close(self):
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_path}")
