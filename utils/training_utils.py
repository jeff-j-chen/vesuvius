import torch
import torch.nn as nn
import torch.optim as optim
import os
from .config import Config

class WarmupThenPlateau:
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, val_loss=None):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.plateau_scheduler.step(val_loss)
        self.current_epoch += 1


def create_optimizer_and_scheduler(model, config: Config):
    """Create optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = WarmupThenPlateau(
        optimizer,
        warmup_epochs=config.training.patience,
        plateau_scheduler=optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.lr_scheduler_factor,
            patience=config.training.patience
        ),
        base_lr=config.training.learning_rate
    )
    
    return optimizer, scheduler

def create_loss_function(pos_weight, config: Config):
    """Create loss function with optional class weighting"""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(config.device) if pos_weight is not None else None
    )
    return criterion

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model