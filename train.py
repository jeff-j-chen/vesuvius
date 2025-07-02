
import torch
from tqdm import tqdm
from utils.config import Config
from utils.dataloader import create_datasets, create_dataloaders, calculate_class_weights
from utils.model import create_model, InkDetector, CBAM3D
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    save_model
)
from utils.visualizer import TensorboardVisualizer
import time
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import argparse

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, config: Config, scaler: GradScaler):
    """Train for one epoch with L1 regularization"""
    model.train()
    train_loss, train_raw_loss, train_correct, train_total = 0.0, 0.0, 0, 0
    i = 0
    for batch_images, batch_labels in tqdm(train_loader, desc="Training"):
        # i += 1
        # if i > 10:
        #     break
        batch_images = batch_images.to(config.device)
        batch_labels = batch_labels.to(config.device).view(-1, 1)
        
        optimizer.zero_grad()

        with autocast(config.device):
            outputs = model(batch_images)
            raw_loss = criterion(outputs, batch_labels)
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            loss = raw_loss + config.training.l1_lambda * l1_loss
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        train_raw_loss += raw_loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (predicted == batch_labels).sum().item()
        train_total += batch_labels.size(0)
    
    return train_loss / len(train_loader), train_raw_loss / len(train_loader), train_correct / train_total

def validate_epoch(model, valid_loader, criterion, config: Config, scaler: GradScaler):
    """Validate for one epoch (unchanged)"""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    i = 0
    with torch.no_grad(), autocast(config.device):
        for images, labels in tqdm(valid_loader, desc="Validating"):
            # i += 1
            # if i > 10:
            #     break
            images = images.to(config.device)
            labels = labels.to(config.device).view(-1, 1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    return val_loss / len(valid_loader), val_correct / val_total

def main(config: Config):
    set_seed(41)
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, dict):
            for subfield, subvalue in value.items():
                print(f"{field}.{subfield}: {subvalue}")
        else:
            print(f"{field}: {value}")

    print("Creating datasets...", end="")
    start_time = time.time()
    train_dataset, valid_dataset, train_volume, valid_volume, labels = create_datasets(config)
    train_loader, valid_loader = create_dataloaders(train_dataset, valid_dataset, config)
    pos_weight = calculate_class_weights(train_dataset)
    print(f" done in {time.time() - start_time:.2f}s")
    
    # Create model
    print(f"Creating model and loss... l1 lamba {config.training.l1_lambda}... ", end="")
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

    
    scaler = GradScaler()
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        # Train
        if epoch >= 5 and config.dataloader.apply_transforms:
            train_dataset.apply_transforms = True
        train_loss, train_raw_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config, scaler)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, config, scaler)
        
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
        for layer in model.features:
            if isinstance(layer, CBAM3D):
                print(f"Spatial Scale: {layer.spatial_scale.item():.4f}\nChannel Scale: {layer.channel_scale.item():.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'{config.model_dir}/best_model.pth')
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch+1) % config.training.save_every_n_epochs == 0:
            save_model(model, f'{config.model_dir}/model_epoch_{epoch+1}.pth')

        time_elapsed = time.time() - start_time
        vis.log_epoch_metrics(epoch, model, train_acc, val_acc, train_loss, train_raw_loss, val_loss, current_lr, time_elapsed, train_volume, valid_volume, labels, params)
    
    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    # parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    # args = parser.parse_args()
    # config = Config()
    # config.experiment_name = args.experiment_name
    # main(config)

    # test stronger augmentations
    for transform_type in ["brightness", "contrast"]:
        config = Config()
        config.dataloader.apply_transforms = True
        config.dataloader.transform_type = transform_type
        config.experiment_name = f"transform_{transform_type}_strong"
        print(f"Training with transform type: {transform_type}...")
        main(config)
    
    # test if a lower l1 will allow for mix to generalize
    config = Config()
    config.dataloader.transform_type = "mix"
    config.experiment_name = f"mix_1e-4l1"
    config.training.l1_lambda = 1e-4
    main(config)
    
    # test if a lower l1 will allow for mix to generalize
    config = Config()
    config.dataloader.apply_transforms = False
    config.experiment_name = f"transform_off_128b"
    config.training.l1_lambda = 7e-4
    main(config)

    # re-run tests on every augmentation, to see if they work better with a lower probability (33%)
    for transform_type in ["brightness", "mix", "contrast", "noise", "rotate", "flip"]:
        config = Config()
        config.training.l1_lambda = 7e-4
        config.dataloader.apply_transforms = True
        config.dataloader.low_trans_prob = True
        config.dataloader.transform_type = transform_type
        config.experiment_name = f"transform_{transform_type}_lowprob"
        print(f"Training with transform type: {transform_type}...")
        main(config)

    # add additional testing for every augmentation type, but this time have them turn on only 33% of the time.

    # scroll_ids = [
    #     20231007101619,
    #     20231005123336,
    #     20231022170901,
    #     20230929220926,
    #     20231210121321,
    #     20230702185753,
    #     20231106155351, # x > 4500,
    #     20231016151002,
    #     20231031143852,
    #     20231221180251,
    #     20231012184420,
    #     20230827161847
    # ]
    # for scroll_id in scroll_ids:
    #     config = Config()
    #     config.data.segment_id = scroll_id
    #     config.experiment_name = f"{scroll_id}"
    #     print(f"Training on scroll {scroll_id}...")
    #     main(config)

    # l1s = [7.5e-4]
    # for l1 in l1s:
    #     config = Config()
    #     if l1 == 0: 
    #         config.experiment_name = "cbam3d_28-48_l1_0"
    #     else:
    #         config.experiment_name = f"cbam3d_28-48_l1_{l1:.0e}"
    #     config.training.l1_lambda = l1
    #     main(config)

    # conv1 conv2 fc1 fc2
    # drops = [
    #     [0.0, 0.3, 0.8, 0.6],
    # ]
    # for drop in drops:
    #     config = Config()
    #     config.model.conv1_drop = drop[0]
    #     config.model.conv2_drop = drop[1]
    #     config.model.fc1_drop = drop[2]
    #     config.model.fc2_drop = drop[3]
    #     config.experiment_name = f"sanity-{drop[0]}-{drop[1]}-{drop[2]}-{drop[3]}"
    #     main(config)
    
    # config = Config()
    # config.data.start_level = 32
    # config.data.end_level = 48
    # config.experiment_name = f"3dmodel_redo_{config.data.start_level}_{config.data.end_level}"
    # main(config)

    # while config.data.end_level - config.data.start_level > 4:
    #     config.data.start_level += 4
    #     print(f"entry {config.data.start_level} to {config.data.end_level}")
    #     config.experiment_name = f"3dmodel_redo_{config.data.start_level}_{config.data.end_level}"
    #     main(config)

    #     config.data.end_level -= 4
    #     print(f"entry {config.data.start_level} to {config.data.end_level}")
    #     config.experiment_name = f"3dmodel_redo_{config.data.start_level}_{config.data.end_level}"
    #     main(config)
    