import os
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
from .dataloader import load_test_data, load_scroll4_data




class TensorboardVisualizer:
    def __init__(self, config: Config):
        self.config = config

        if config.experiment_name is None:
            experiment_name = f"ink_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = config.experiment_name + "_" +  datetime.now().strftime('%d_%H%M%S')
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_path)
        self.test_volume = load_test_data(self.config)
        # self.scroll4_volume = load_scroll4_data(self.config)
        
        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")
    
    def log_epoch_metrics(self, epoch, model, train_acc, val_acc, train_loss, val_loss, learning_rate, time_elapsed, train_volume, valid_volume, labels, params):
        print(f"Logging metrics for epoch: {epoch}")
        # Log accuracies
        self.writer.add_scalar('Metrics/Train_Acc', train_acc, epoch)
        self.writer.add_scalar('Metrics/Validation_Acc', val_acc, epoch)
        
        # Log losses
        self.writer.add_scalar('Metrics/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('Metrics/Validation_Loss', val_loss, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Metrics/Learning_Rate', learning_rate, epoch)

        # Time elapsed
        self.writer.add_scalar('Time_Elapsed', time_elapsed, epoch)

        # Log weight histograms
        self.log_weight_histograms(model, epoch)

        # # Log activation maps if available
        # if hasattr(model, "activations"):
        #     self.log_activation_maps(model.activations, epoch)

        # Log model graph once at the beginning
        if epoch == 0:
            print("Logging hyperparameters and model graph")
            example_input = torch.randn(1, self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size).to(self.config.device)
            example_input = example_input.unsqueeze(0)
            self.log_model_graph(model, example_input)
            self.log_hyperparameters(params)

        # Add test figures at specified intervals
        # if (epoch) % self.config.training.test_interval == 0:
            # self.add_test_figures(epoch, model, self.test_volume)
            # self.add_scroll4_figures(epoch, model, self.scroll4_volume)

        # Add evaluation figures at specified intervals
        if (epoch) % self.config.training.evaluation_interval == 0:
            self.add_evaluation_figures(epoch, model, train_volume, valid_volume, labels)
        
        self.writer.flush()
    
    def _process_volume_depth_block(self, model, volume, volume_name, depth_start, depth_end):
        """Helper function to process a single volume at a specific depth range"""
        D, H, W = volume.shape
        
        prediction_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # Create list of all tile coordinates
        tile_coords = []
        for y in range(0, H - self.config.data.tile_size + 1, self.config.data.tile_size):
            for x in range(0, W - self.config.data.tile_size + 1, self.config.data.tile_size):
                tile_coords.append((y, x))
        
        # Pre-allocate tensor on GPU to avoid repeated allocations
        if torch.cuda.is_available():
            block_tensor = torch.zeros(
                (1, 1, self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size),
                dtype=torch.float32,
                device=self.config.device
            )
        else:
            block_tensor = torch.zeros(
                (1, 1, self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size),
                dtype=torch.float32
            )
        with torch.no_grad():
            for y, x in tqdm(tile_coords, desc=f"Processing {volume_name} volume (depth {depth_start}-{depth_end})"):
                # Extract block from the specified depth range
                block = volume[depth_start:depth_end, y:y+self.config.data.tile_size, x:x+self.config.data.tile_size]
                # print(f"extracted {block.shape} from {depth_start}:{depth_end}")
                
                if block.shape != (self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size):
                    # Clean up before returning
                    del block_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return np.zeros((H, W), dtype=np.float32)
                
                # Copy data to pre-allocated tensor instead of creating new ones
                block_tensor[0, 0] = torch.from_numpy(block).float()
                
                logits = model(block_tensor)
                pred = torch.sigmoid(logits).item()
                
                prediction_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += pred
                count_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += 1
        
        # Clean up GPU memory
        del block_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Normalize predictions
        prediction_map = np.divide(prediction_map, count_map, where=count_map>0)
        return prediction_map

    def _create_combined_evaluation_figure(self, all_predictions_data, labels, num_depth_blocks):
        print("Creating combined evaluation figure")

        fig_height = 3 * num_depth_blocks
        fig = plt.figure(figsize=(10, fig_height))  # Set consistent width
        gs = gridspec.GridSpec(num_depth_blocks, 2, width_ratios=[1, 1], wspace=0.05, hspace=0.15)

        for block_idx, (full_predictions, train_predictions, depth_start, depth_end) in enumerate(all_predictions_data):
            s = self.config.data.start_level

            # Predictions only
            ax_pred = fig.add_subplot(gs[block_idx, 0])
            im1 = ax_pred.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='auto')
            ax_pred.set_title(f'Model Predictions\nDepth Block {s+depth_start}-{s+depth_end}\nTrain | Valid', fontsize=9)
            ax_pred.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=1.5)
            ax_pred.axis('off')
            cbar = plt.colorbar(im1, ax=ax_pred, fraction=0.046, pad=0.01)
            cbar.ax.tick_params(labelsize=6)

            # Predictions + GT overlay
            ax_overlay = fig.add_subplot(gs[block_idx, 1])
            ax_overlay.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='auto')

            # GT overlay
            label_overlay = np.zeros((*labels.shape, 4))  # RGBA
            label_overlay[labels > 0.5] = [1, 1, 1, 0.4]
            ax_overlay.imshow(label_overlay)

            ax_overlay.set_title(f'Predictions + Ground Truth\nDepth Block {s+depth_start}-{s+depth_end}\n(White = GT)', fontsize=9)
            ax_overlay.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=1.5)
            ax_overlay.axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
        return fig


    def _create_combined_test_figure(self, all_predictions_data, num_depth_blocks):
        """Create combined test figure with predictions (no ground truth overlay)"""
        
        # Rearrange from 8x1 to 4x2 for better space utilization
        if num_depth_blocks <= 4:
            # Keep as single column if 4 or fewer blocks
            cols = 1
            rows = num_depth_blocks
        else:
            # Use 2 columns for more than 4 blocks
            cols = 2
            rows = (num_depth_blocks + 1) // 2  # Ceiling division
        
        # Optimized sizing for square plots
        fig_width = 6 * cols  # 6 inches per column
        fig_height = 3 * rows  # 3 inches per row (good for square plots)
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle different subplot arrangements
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for block_idx, (predictions, depth_start, depth_end) in enumerate(all_predictions_data):
            ax = axes[block_idx]
            im = ax.imshow(predictions, cmap='inferno', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)  # Smaller colorbar
            ax.set_title(f'Model Predictions\nDepth Block {depth_start}-{depth_end}', fontsize=10)
            ax.axis('off')
        
        # Hide any unused subplots
        for block_idx in range(len(all_predictions_data), len(axes)):
            axes[block_idx].axis('off')
        
        # CRITICAL FIX: Minimal spacing to maximize plot size and eliminate left whitespace
        plt.subplots_adjust(
            left=0.02,    # Minimal left margin - eliminates disgusting whitespace
            right=0.98,   # Minimal right margin  
            top=0.95,     # Small top margin for titles
            bottom=0.02,  # Minimal bottom margin
            wspace=0.05,  # Very small horizontal spacing between plots
            hspace=0.15   # Small vertical spacing between rows
        )
        
        return fig


    def add_evaluation_figures(self, epoch, model, train_volume, valid_volume, labels):
        """
        Run full validation and create one combined figure with all depth blocks
        Each depth block gets 2 subplots: predictions only, predictions + ground truth overlay
        """
        print("Starting evaluation figure generation...")
        # Set model to eval mode once at the beginning
        model.eval()
        
        # Calculate number of depth blocks
        D = train_volume.shape[0]
        num_depth_blocks = (D - self.config.data.depth) // int(self.config.data.depth // 2) + 1
        
        all_predictions_data = []
        
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for evaluation...")
            depth_start = block_idx * int(self.config.data.depth // 2)
            depth_end = min(depth_start + self.config.data.depth, D)
            
            if depth_start >= D or depth_end > D: 
                continue
            
            train_predictions = self._process_volume_depth_block(
                model, train_volume, "training", depth_start, depth_end
            )
            valid_predictions = self._process_volume_depth_block(
                model, valid_volume, "validation", depth_start, depth_end
            )
            
            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            all_predictions_data.append((full_predictions, train_predictions, depth_start, depth_end))
        
        if all_predictions_data:
            # Create one combined figure with all depth blocks (2 subplots per depth block)
            fig = self._create_combined_evaluation_figure(all_predictions_data, labels, len(all_predictions_data))
            self.writer.add_figure('Evaluation/All_Depth_Blocks', fig, epoch)
            plt.close(fig)  # Important: close figure to free memory
    
    def add_test_figures(self, epoch, model, test_volume):
        """
        Run test evaluation and create one combined figure with all depth blocks
        No ground truth overlay for test data
        """
        print("Starting test figure generation...")
        model.eval()
        
        # Calculate number of depth blocks
        D = test_volume.shape[0]
        num_depth_blocks = (D + self.config.data.depth - 1) // self.config.data.depth  # Ceiling division
        
        all_predictions_data = []
        
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for test...")
            
            # Determine depth range for the current block
            depth_start = block_idx * self.config.data.depth
            depth_end = min(depth_start + self.config.data.depth, D)
            
            # Skip if depth_start or depth_end is out of bounds
            if depth_start >= D or depth_end > D or depth_start < 0:
                continue
            
            # Check if the depth range is valid
            if depth_end - depth_start != self.config.data.depth:
                print(f"Skipping block {block_idx + 1} due to mismatched depth dimensions.")
                continue
            
            predictions = self._process_volume_depth_block(
                model, test_volume, "test", depth_start, depth_end
            )
            
            all_predictions_data.append((predictions, depth_start, depth_end))
        
        if all_predictions_data:
            # Create one combined figure with all test depth blocks (no ground truth overlay)
            print(f"got {len(all_predictions_data)} depth blocks for test")

            fig = self._create_combined_test_figure(all_predictions_data, len(all_predictions_data))
            self.writer.add_figure('Test/All_Depth_Blocks', fig, epoch)
            plt.close(fig)  # Important: close figure to free memory
    
    def add_scroll4_figures(self, epoch, model, scroll4_volume):
        """
        Run scroll4 evaluation and create one combined figure with all depth blocks
        No ground truth available for scroll4
        """
        print("Starting scroll4 figure generation...")
        model.eval()
        
        # Calculate number of depth blocks
        D = scroll4_volume.shape[0]
        num_depth_blocks = (D - self.config.data.depth) // int(self.config.data.depth // 2) + 1
        
        all_predictions_data = []
        
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for scroll4...")
            depth_start = block_idx * int(self.config.data.depth // 2)
            depth_end = min(depth_start + self.config.data.depth, D)
            
            if depth_start >= D or depth_end > D: 
                continue
            
            predictions = self._process_volume_depth_block(
                model, scroll4_volume, "scroll4", depth_start, depth_end
            )
            
            all_predictions_data.append((predictions, depth_start, depth_end))
        
        if all_predictions_data:
            # Create one combined figure with all scroll4 depth blocks (no ground truth)
            fig = self._create_combined_test_figure(all_predictions_data, len(all_predictions_data), "Scroll4")
            self.writer.add_figure('Scroll4/All_Depth_Blocks', fig, epoch)
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
                self.writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad.cpu().numpy(), epoch)

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
