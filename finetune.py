import torch
import numpy as np
import argparse
import time
import cv2
import os
from torch.cuda.amp.grad_scaler import GradScaler

from utils.config import Config
from utils.model import create_model
from utils.dataloader import load_scroll4_data, get_or_compute_normalization
from utils.finetune_dataloader import get_finetune_dataloaders
from utils.training_utils import create_optimizer_and_scheduler, create_loss_function, save_model
from train import train_epoch, validate_epoch, set_seed
from utils.visualizer import TensorboardVisualizer

# Locations from your provided file
locs = [
    # x, y, width, height
    # train
    [5612, 4110, 448, 448], [5060, 3452, 576, 576], [4992, 5032, 576, 576],
    [4871, 6682, 576, 576], [8746, 3678, 576, 576], [9369, 1574, 576, 576],
    [6690, 5182, 1280, 576], [2330, 5794, 672, 672], [1097, 3710, 672, 672],
    [9978, 3384, 448, 1280],
    # valid
    [3284, 7940, 832, 448], [1962, 7972, 448, 448], [771, 5132, 384, 384]
]

def main(config: Config, model_path: str):
    set_seed(42)
    
    # --- 1. Load Model and Freeze Layers ---
    print("Loading pre-trained model...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model, params = create_model(config)
    model.load_state_dict(torch.load(model_path))
    
    print("Freezing feature extractor layers...")
    for param in model.features.parameters():
        param.requires_grad = False

    # Verify which layers are frozen
    for name, param in model.named_parameters():
        print(f"{name}: trainable={param.requires_grad}")

    # --- 2. Load Data and Dataloaders ---
    print("Loading Scroll 4 data for fine-tuning...")
    volume, mask, _, _ = load_scroll4_data(config)
    labels_path = f"./eroded_inklabels/{config.data.scroll4_segment_id}.png"
    labels = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE) / 255.0
    
    print("Getting normalization stats for Scroll 4...")
    norm_stats = get_or_compute_normalization(config.data.scroll4_segment_id, volume, mask)
    
    print("Creating fine-tuning dataloaders...")
    train_loader, valid_loader = get_finetune_dataloaders(config, volume, labels, mask, norm_stats, locs)

    # --- 3. Setup Training Components ---
    # Use a smaller learning rate for fine-tuning
    config.training.learning_rate = 1e-5
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # For fine-tuning, we can use a simple pos_weight or calculate it on the small dataset
    pos_weight = torch.tensor([2.0]) # A reasonable default for imbalanced data
    criterion = create_loss_function(pos_weight, config)
    
    # --- 4. Training Loop ---
    print("Initializing Tensorboard for fine-tuning...")
    vis = TensorboardVisualizer(config, mode='finetune')
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    scaler = GradScaler()
    
    config.training.num_epochs = 25 # More epochs for fine-tuning on small data
    
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config, scaler)
        val_metrics = validate_epoch(model, valid_loader, criterion, config, scaler)
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model(model, f'{config.model_dir}/finetuned_best_f1.pth')
            print(f"New best F1 model saved! Val F1: {best_val_f1:.4f}")

        time_elapsed = time.time() - start_time
        vis.log_epoch_metrics(epoch, model, train_metrics, val_metrics, current_lr, time_elapsed, params)

    vis.close()
    print("Fine-tuning completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script for Vesuvius model.")
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        default="models/best_model_f1.pth", 
        help="Path to the pre-trained model file to fine-tune."
    )
    parser.add_argument("-n", "--experiment_name", type=str, default="finetune", help="Name of the fine-tuning experiment")
    args = parser.parse_args()
    
    config = Config()
    config.experiment_name = args.experiment_name
    
    main(config, args.model_path)