import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from .config import Config

class TensorboardVisualizer:
    def __init__(self, config: Config):
        self.config = config

        if config.experiment_name is None:
            experiment_name = f"ink_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = config.experiment_name + "_" +  datetime.now().strftime('%d_%H%M%S')
        
        self.log_path = os.path.join(config.training.log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_path)
        
        print(f"TensorBoard logs will be saved to: {self.log_path}")
        print(f"To view, run: tensorboard --logdir={config.training.log_dir}")
    
    def log_epoch_metrics(self, epoch, model, train_acc, val_acc, train_loss, val_loss, learning_rate, time_elapsed, train_volume, train_labels, valid_volume, valid_labels, params):
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

        if (epoch+1) % self.config.training.evaluation_interval == 0:
            print(f"Running full evaluation on epoch {epoch} due to evaluation interval {self.config.training.evaluation_interval}")
            self.add_evaluation_figures(epoch, model, train_volume, train_labels, valid_volume, valid_labels)
        
        self.writer.flush()
    

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
        print("tiles created")
        
        with torch.no_grad():
            # Process tiles with tqdm progress bar
            print("starting processing with no grad")
            for y, x in tqdm(tile_coords, desc=f"Processing {volume_name} volume (depth {depth_start}-{depth_end-1})"):
                # Extract block from the specified depth range
                block = volume[depth_start:depth_end, y:y+self.config.data.tile_size, x:x+self.config.data.tile_size]
                
                if block.shape == (self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size):
                    block_tensor = torch.from_numpy(block).float().unsqueeze(0).unsqueeze(0).to(self.config.device)
                    logits = model(block_tensor)
                    pred = torch.sigmoid(logits).item()
                    
                    prediction_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += pred
                    count_map[y:y+self.config.data.tile_size, x:x+self.config.data.tile_size] += 1
            print("grad back on")
        
        # Normalize predictions
        print("done, now normalizing")
        prediction_map = np.divide(prediction_map, count_map, where=count_map>0)
        print("normalization complete, returning")
        return prediction_map

    def _create_evaluation_figure(self, full_labels, full_predictions, train_predictions, block_idx, depth_start, depth_end, middle_slice_idx):
        """Create a single validation figure for one depth block with confusion matrix"""
        fig = plt.figure(figsize=(15, 4))  # Increased width for 3 subplots
        
        # Predictions
        plt.subplot(1, 3, 1)
        img = plt.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1)
        plt.colorbar(img, fraction=0.046, pad=0.04)
        plt.title(f'Model Predictions\nDepth Block {block_idx + 1}\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 2)
        plt.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1)
        
        # Create overlay for ground truth
        label_overlay = np.zeros((*full_labels.shape, 4))  # RGBA
        label_overlay[full_labels > 0.5] = [1, 1, 1, 0.4]  # White with transparency
        plt.imshow(label_overlay)
        
        plt.title(f'Predictions + Ground Truth\nDepth Block {block_idx + 1}\nTrain | Valid\n(White = True Labels)')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Confusion Matrix
        plt.subplot(1, 3, 3)
        
        # Convert continuous predictions to binary using 0.5 threshold
        binary_predictions = (full_predictions > 0.5).astype(int)
        binary_labels = (full_labels > 0.5).astype(int)
        
        # Flatten arrays for confusion matrix calculation
        y_true_flat = binary_labels.flatten()
        y_pred_flat = binary_predictions.flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Ink (0)', 'Ink (1)'], 
                    yticklabels=['No Ink (0)', 'Ink (1)'],
                    cbar_kws={'shrink': 0.8})
        
        plt.title(f'Confusion Matrix\nDepth Block {block_idx + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Calculate and add metrics as text
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add metrics text below the confusion matrix
        metrics_text = f'Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}'
        plt.figtext(0.85, 0.02, metrics_text, fontsize=10, ha='center')
        
        plt.tight_layout()
        return fig

    def add_evaluation_figures(self, epoch, model, train_volume, train_labels, valid_volume, valid_labels):
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
        num_depth_blocks = (D - self.config.data.depth) // int(self.config.data.depth // 2) + 1
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for evaluation...")
            depth_start = block_idx * int(self.config.data.depth // 2)
            depth_end = min(depth_start + self.config.data.depth, D)  # Ensure depth_end does not exceed D
            if depth_start >= D or depth_end > D:  # Check if depth_start or depth_end is out of bounds
                print(f"Skipping depth block {block_idx + 1} due to out-of-bounds indices.")
                continue
            
            # Process both volumes for this depth block
            print('beginning preds')
            train_predictions = self._process_volume_depth_block(
                model, train_volume, "training", depth_start, depth_end
            )
            valid_predictions = self._process_volume_depth_block(
                model, valid_volume, "validation", depth_start, depth_end
            )
            
            # Stitch everything back together horizontally
            # Use the middle slice of the current depth block for visualization
            middle_slice_idx = depth_start + self.config.data.depth // 2
            if middle_slice_idx >= D:  # Ensure middle_slice_idx is within bounds
                print(f"Skipping visualization for depth block {block_idx + 1} due to out-of-bounds middle slice index.")
                continue
            
            full_labels = np.concatenate([train_labels, valid_labels], axis=1)
            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            
            # Create and log the validation figure
            fig = self._create_evaluation_figure(
                full_labels, full_predictions, train_predictions,
                block_idx, depth_start, depth_end, middle_slice_idx
            )
            # Log figure to TensorBoard
            self.writer.add_figure(f'Evaluation/Depth_Block_{self.config.data.start_level + depth_start}-{self.config.data.start_level + depth_end}', fig, epoch)
            
            # Close the figure to free memory
            plt.close(fig)
    
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
