import torch
import torch.nn as nn
import torch.optim as optim
import os
from .config import Config
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

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
    """Create the loss function, optionally with positional weights."""
    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(config.device) if pos_weight is not None else None
    )

def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculate comprehensive metrics for binary classification
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1) 
        y_scores: Predicted probabilities (0 to 1)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Accuracy
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['specificity'] = specificity
    
    # Class distribution
    pos_samples = np.sum(y_true == 1)
    neg_samples = np.sum(y_true == 0)
    total_samples = len(y_true)
    metrics['positive_samples'] = pos_samples
    metrics['negative_samples'] = neg_samples
    metrics['positive_ratio'] = pos_samples / total_samples if total_samples > 0 else 0.0

    
    # ROC-AUC and PR-AUC (handle edge cases)
    try:
        if len(np.unique(y_true)) == 2:  # Both classes present
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.5  # Random performance when only one class
            metrics['pr_auc'] = pos_samples / total_samples if total_samples > 0 else 0.0 # Baseline for PR-AUC
    except Exception as e:
        print(f"Warning: Could not calculate AUC metrics: {e}")
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.0
    
    return metrics

def save_model(model, path):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model