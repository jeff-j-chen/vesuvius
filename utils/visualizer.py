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
from scipy.ndimage import zoom

class TensorboardVisualizer:
    def __init__(self, config: Config):
        self.config = config

        if config.experiment_name is None:
            experiment_name = f"ink_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = config.experiment_name + "_" +  datetime.now().strftime('%d_%H%M%S')
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        self.layout = {
            "Training_Overview": {
                "loss": ["Multiline", ["Training_Metrics/Loss/Train", "Training_Metrics/Loss/Train_Raw", "Training_Metrics/Loss/Validation"]],
                "accuracy": ["Multiline", ["Training_Metrics/Accuracy/Train", "Training_Metrics/Accuracy/Validation"]],
            },
        }
        self.writer = SummaryWriter(self.log_path)
        self.writer.add_custom_scalars(self.layout)
        self.test_volume = None
        self.scroll4_volume = None
        # self.test_volume = load_test_data(self.config)
        # self.scroll4_volume = load_scroll4_data(self.config)

        
        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")
    
    def log_epoch_metrics(self, epoch, model, train_acc, val_acc, train_loss, train_raw_loss, val_loss, learning_rate, time_elapsed, train_volume, valid_volume, labels, params):
        print(f"Logging metrics for epoch: {epoch}")
        self.writer.add_scalar("Training_Metrics/Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Training_Metrics/Loss/Train_Raw", train_raw_loss, epoch)
        self.writer.add_scalar("Training_Metrics/Loss/Validation", val_loss, epoch)
        
        self.writer.add_scalar("Training_Metrics/Accuracy/Train", train_acc, epoch)
        self.writer.add_scalar("Training_Metrics/Accuracy/Validation", val_acc, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)

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
        if (epoch+1) % self.config.training.test_interval == 0:
            self.add_test_figures(epoch, model, self.test_volume, "scroll1")
            self.add_test_figures(epoch, model, self.scroll4_volume, "scroll4")

        # Add evaluation figures at specified intervals
        if (epoch+1) % self.config.training.evaluation_interval == 0:
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
            scaled_full_predictions = zoom(full_predictions, scale_factor, order=1)
            scaled_train_predictions = zoom(train_predictions, scale_factor, order=1)
            scaled_labels = zoom(labels, scale_factor, order=0)
            
            # Left plot: Model predictions
            ax_pred = axes[block_idx, 0]
            im1 = ax_pred.imshow(scaled_full_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            s = self.config.data.start_level
            ax_pred.set_title(f'Depth Block {s+depth_start}-{s+depth_end}', fontsize=9)
            
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
        
        if num_depth_blocks == 1:
            axes = axes.reshape(1, -1)
        
        for block_idx, (predictions, depth_start, depth_end) in enumerate(all_predictions_data):
            ax1 = axes[block_idx // cols, block_idx % cols]
            scaled_predictions = zoom(predictions, scale_factor, order=1)
            im = ax1.imshow(scaled_predictions, cmap='inferno', vmin=0, vmax=1, aspect='equal')
            ax1.set_title(f'Depth Block {depth_start}-{depth_end}', fontsize=9)
            ax1.axis('off')

        
        plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
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
    
    def add_test_figures(self, epoch, model, test_volume, test_type):
        """
        Run test evaluation and create one combined figure with all depth blocks
        No ground truth overlay for test data
        """
        if test_type not in ["scroll1", "scroll4"]:
            print(f"Invalid test type: {test_type}. Expected 'scroll1' or 'scroll4'.")
        print("Starting test figure generation...")
        model.eval()
        
        # Calculate number of depth blocks
        D = test_volume.shape[0]
        all_predictions_data = []
        num_depth_blocks = (D - self.config.data.depth) // int(self.config.data.depth // 2) + 1
        
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for evaluation...")
            depth_start = block_idx * int(self.config.data.depth // 2)
            depth_end = min(depth_start + self.config.data.depth, D)
            
            # Skip if depth_start or depth_end is out of bounds
            if depth_start >= D or depth_end > D or depth_start < 0:
                continue
            
            # Check if the depth range is valid
            if depth_end - depth_start != self.config.data.depth:
                print(f"Skipping block {block_idx + 1} due to mismatched depth dimensions.")
                continue
            
            predictions = self._process_volume_depth_block(
                model, test_volume, test_type, depth_start, depth_end
            )
            
            all_predictions_data.append((predictions, depth_start, depth_end))
        
        if all_predictions_data:
            # Create one combined figure with all test depth blocks (no ground truth overlay)
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
                self.writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad.cpu().numpy(), epoch)

    def log_hyperparameters(self, params):
        self.writer.add_scalar("Hyperparameters/Tile Size", self.config.data.tile_size)
        self.writer.add_scalar("Hyperparameters/Depth", self.config.data.depth)
        self.writer.add_scalar("Hyperparameters/Batch Size", self.config.dataloader.train_batch_size)
        self.writer.add_scalar("Hyperparameters/Num Workers", self.config.dataloader.train_num_workers)
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
