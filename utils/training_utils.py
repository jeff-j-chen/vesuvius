import torch
import torch.nn as nn
import torch.optim as optim
import os
from .config import Config

def create_optimizer_and_scheduler(model, config: Config):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.lr_scheduler_factor,
        patience=config.training.patience,
    )
    
    return optimizer, scheduler

def create_loss_function(pos_weight, config: Config):
    """Create loss function with optional class weighting"""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(config.device) if pos_weight is not None else None
    )
    return criterion

def save_model(model, path):
    """Save model state dict"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model state dict"""
    model.load_state_dict(torch.load(path))
    return model