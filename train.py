
import torch
from tqdm import tqdm
from utils.config import Config
from utils.dataloader import load_data, create_datasets, create_dataloaders, calculate_class_weights
from utils.model import create_model
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    save_model
)
from utils.visualizer import TensorboardVisualizer
import time
import random
import argparse

def train_epoch(model, train_loader, criterion, optimizer, config: Config):
    """Train for one epoch with L1 regularization"""
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch_images, batch_labels in tqdm(train_loader, desc="Training"):
        batch_images = batch_images.to(config.device)
        batch_labels = batch_labels.to(config.device).view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(batch_images)
        
        loss = criterion(outputs, batch_labels)
        
        # Add L1 regularization
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        current_lr = optimizer.param_groups[0]['lr']
        scaled_l1_lambda = config.training.l1_lambda * (current_lr / config.training.learning_rate)
        loss += scaled_l1_lambda * l1_loss
        loss += config.training.l1_lambda * l1_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
        
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predicted == batch_labels).sum().item()
        train_total += batch_labels.size(0)
    
    return train_loss / len(train_loader), train_correct / train_total

def validate_epoch(model, valid_loader, criterion, config: Config):
    """Validate for one epoch (unchanged)"""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        i = 0
        for images, labels in valid_loader:
            # i += 1
            # if i % 100 != 1:
            #     continue
            images = images.to(config.device)
            labels = labels.to(config.device).view(-1, 1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    return val_loss / len(valid_loader), val_correct / val_total

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.experiment_name = args.experiment_name

    # Load and prepare data
    print("Loading data...", end="")
    start_time = time.time()
    volume, labels = load_data(config)
    print(f" done in {time.time() - start_time:.2f}s")
    
    # Create datasets and dataloaders, in addition to the class weights
    print("Creating datasets...", end="")
    start_time = time.time()
    train_dataset, valid_dataset, train_volume, train_labels, valid_volume, valid_labels = create_datasets(volume, labels, config)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, config)
    pos_weight = calculate_class_weights(train_dataset)
    print(f" done in {time.time() - start_time:.2f}s")
    
    
    # Create model
    print("Creating model and loss...", end="")
    start_time = time.time()
    model, params = create_model(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    criterion = create_loss_function(pos_weight, config)
    print(f" done in {time.time() - start_time:.2f}s")

    # Initialize Tensorboard
    print("Initializing Tensorboard...")
    start_time = time.time()
    vis = TensorboardVisualizer(config)
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, config)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log progress
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/{config.training.num_epochs} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'{config.model_dir}/best_model.pth')
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            save_model(model, f'{config.model_dir}/model_epoch_{epoch+1}.pth')

        time_elapsed = time.time() - start_time
        vis.log_epoch_metrics(epoch, model, train_acc, val_acc, train_loss, val_loss, current_lr, time_elapsed, train_volume, train_labels, valid_volume, valid_labels, params)
    
    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    main()