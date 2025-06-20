import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .config import Config

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
    
    def log_epoch_metrics(self, epoch, model, train_acc, val_acc, train_loss, val_loss, learning_rate, time_elapsed, train_volume, train_labels, valid_volume, valid_labels, params):
        print(f"Logging metrics for epoch: {epoch}")
        # Log accuracies
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # Log losses
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)

        # Time elapsed
        self.writer.add_scalar('Time_Elapsed', time_elapsed, epoch)
        
        # # Log combined accuracy plot
        # self.writer.add_scalars('Accuracy_Comparison', {
        #     'Train': train_acc,
        #     'Validation': val_acc
        # }, epoch)
        
        # # Log combined loss plot
        # self.writer.add_scalars('Loss_Comparison', {
        #     'Train': train_loss,
        #     'Validation': val_loss
        # }, epoch)

        # Log weight histograms
        # self.log_weight_histograms(model, epoch)

        # # Log activation maps if available
        # if hasattr(model, "activations"):
        #     self.log_activation_maps(model.activations, epoch)

        # # Log confusion matrix if available
        # if hasattr(model, "confusion_matrix"):
        #     confusion_matrix = model.confusion_matrix.cpu().numpy()
        #     class_names = ["0", "1"]
        #     self.log_confusion_matrix(confusion_matrix, class_names, epoch)

        # Log model graph once at the beginning
        if epoch == 0:
            print("loggin hparams")
            example_input = torch.randn(1, 1, self.config.data.depth, self.config.data.tile_size, self.config.data.tile_size).to(self.config.device)
            self.log_model_graph(model, example_input)
            self.log_hyperparameters(params)

        if epoch % self.config.training.evaluation_interval == 0:
            print(f"Running full evaluation on epoch {epoch} due to evaluation interval {self.config.training.evaluation_interval}")
            self.log_evaluation_results(epoch, model, train_volume, train_labels, valid_volume, valid_labels)
        
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
    
    def _create_evaluation_figure(self, full_volume_slice, full_labels, full_predictions, train_predictions, block_idx, depth_start, depth_end, middle_slice_idx):
        """Create a single validation figure for one depth block"""
        fig = plt.figure(figsize=(12, 12))
        
        # Original slice
        plt.subplot(2, 2, 1)
        plt.imshow(full_volume_slice, cmap='gray')
        plt.title(f'Full Volume (Slice {middle_slice_idx})\nDepth Block {block_idx + 1} ({depth_start}-{depth_end-1})\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Ground truth
        plt.subplot(2, 2, 2)
        plt.imshow(full_labels, cmap='binary')
        plt.title(f'Ground Truth Labels\nDepth Block {block_idx + 1}\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Predictions
        plt.subplot(2, 2, 3)
        img = plt.imshow(full_predictions, cmap='inferno', vmin=0, vmax=1)
        plt.colorbar(img, fraction=0.046, pad=0.04)
        plt.title(f'Model Predictions\nDepth Block {block_idx + 1}\nTrain | Valid')
        plt.axvline(x=train_predictions.shape[1]-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axis('off')
        
        # Overlay
        plt.subplot(2, 2, 4)
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
    
    def log_evaluation_results(self, epoch, model, train_volume, train_labels, valid_volume, valid_labels):
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
        num_depth_blocks = (D - self.config.data.depth + 1) // int(self.config.data.depth // 2)
        
        # Process each depth block
        for block_idx in range(num_depth_blocks):
            print(f"Processing depth block {block_idx + 1}/{num_depth_blocks} for evaluation...")
            depth_start = block_idx * int(self.config.data.depth // 2)
            depth_end = min(depth_start + self.config.data.depth, D)  # Ensure depth_end does not exceed D
            
            if depth_start >= D or depth_end > D:  # Check if depth_start or depth_end is out of bounds
                print(f"Skipping depth block {block_idx + 1} due to out-of-bounds indices.")
                continue
            
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
            if middle_slice_idx >= D:  # Ensure middle_slice_idx is within bounds
                print(f"Skipping visualization for depth block {block_idx + 1} due to out-of-bounds middle slice index.")
                continue
            
            full_volume_slice = np.concatenate([train_volume[middle_slice_idx], valid_volume[middle_slice_idx]], axis=1)
            full_labels = np.concatenate([train_labels, valid_labels], axis=1)
            full_predictions = np.concatenate([train_predictions, valid_predictions], axis=1)
            
            # Create and log the validation figure
            fig = self._create_evaluation_figure(
                full_volume_slice, full_labels, full_predictions, train_predictions,
                block_idx, depth_start, depth_end, middle_slice_idx
            )
            # Log figure to TensorBoard
            self.writer.add_figure(f'Evaluation/Depth_Block_{block_idx + 1}', fig, epoch)
            
            # Close the figure to free memory
            plt.close(fig)
    
    def log_model_graph(self, model, example_input):
        self.writer.add_graph(model, example_input)
    
    def log_activation_maps(self, activations, epoch):
        for layer_name, activation_map in activations.items():
            self.writer.add_images(f"Activations/{layer_name}", activation_map, epoch, dataformats="NCHW")
    
    def log_confusion_matrix(self, confusion_matrix, class_names, epoch):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(confusion_matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.colorbar()
        self.writer.add_figure("Confusion_Matrix", fig, epoch)
        plt.close(fig)

    # Log constants to TensorBoard
    def log_hyperparameters(self, params):
        self.writer.add_text("Hyperparameters/Tile Size", str(self.config.data.tile_size))
        self.writer.add_text("Hyperparameters/Depth", str(self.config.data.depth))
        self.writer.add_text("Hyperparameters/Batch Size", str(self.config.dataloader.batch_size))
        self.writer.add_text("Hyperparameters/Num Workers", str(self.config.dataloader.num_workers))
        self.writer.add_text("Hyperparameters/Num Epochs", str(self.config.training.num_epochs))
        self.writer.add_text("Hyperparameters/Learning Rate", str(self.config.training.learning_rate))
        self.writer.add_text("Hyperparameters/Weight Decay", str(self.config.training.weight_decay))
        self.writer.add_text("Hyperparameters/Max Grad Norm", str(self.config.training.max_grad_norm))
        self.writer.add_text("Hyperparameters/Patience", str(self.config.training.patience))
        self.writer.add_text("Hyperparameters/LR Scheduler Factor", str(self.config.training.lr_scheduler_factor))
        self.writer.add_text("Hyperparameters/Save Every N Epochs", str(self.config.training.save_every_n_epochs))
        self.writer.add_text("Hyperparameters/Evaluation Interval", str(self.config.training.evaluation_interval))
        self.writer.add_text("Hyperparameters/Model Complexity", str(params))
    
    def close(self):
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.log_path}")
