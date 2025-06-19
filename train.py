
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

def train_epoch(model, train_loader, criterion, optimizer, config: Config):
    """Train for one epoch"""
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for batch_images, batch_labels in tqdm(train_loader, desc="Training"):
        batch_images = batch_images.to(config.device)
        batch_labels = batch_labels.to(config.device).view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(batch_images)
        
        loss = criterion(outputs, batch_labels)
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
    """Validate for one epoch"""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(config.device)
            labels = labels.to(config.device).view(-1, 1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    return val_loss / len(valid_loader), val_correct / val_total

def main():
    # Load configuration
    config = Config()
    config.experiment_name = "HELLO_WORLD"

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
    model = create_model(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    criterion = create_loss_function(pos_weight, config)
    print(f" done in {time.time() - start_time:.2f}s")

    # Initialize Tensorboard
    print("Initializing Tensorboard...")
    start_time = time.time()
    vis = TensorboardVisualizer(config)
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
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

            vis.log_epoch_metrics(epoch, train_acc, val_acc, train_loss, val_loss, current_lr)
    
        # Run validation visualization if needed
        if vis.should_run_validation(epoch):
            vis.log_validation_results(epoch, model, train_volume, train_labels, valid_volume, valid_labels)
    
    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    main()