import torch
import torch.nn as nn
import torch.optim as optim
import os
from .config import Config
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class WarmupThenPlateau:
    """a learning rate scheduler with a warmup phase followed by a plateau phase"""
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler, base_lr):
        """initializes the scheduler"""
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, val_loss=None):
        """updates the learning rate based on the current epoch"""
        if self.current_epoch < self.warmup_epochs:
            # linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # switch to plateau scheduler after warmup
            self.plateau_scheduler.step(val_loss)
            
        self.current_epoch += 1


def create_optimizer_and_scheduler(model, config: Config):
    """creates an optimizer and a learning rate scheduler"""
    # create adamw optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.tra.lr,
        weight_decay=config.tra.weight_decay
    )
    
    # create the combined warmup and plateau scheduler
    scheduler = WarmupThenPlateau(
        optimizer,
        warmup_epochs=config.tra.patience,
        plateau_scheduler=optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.tra.lr_decay,
            patience=config.tra.patience
        ),
        base_lr=config.tra.lr
    )
    
    return optimizer, scheduler

class FocalBCELoss(nn.Module):
    """focal loss wrapper around BCE: (1-p_t)^gamma * BCE.
    gamma=0 reduces to standard BCE; gamma>0 down-weights easy examples
    so the gradient is dominated by hard-to-classify tiles."""
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p_t = torch.exp(-bce_loss)              # probability of correct class
        focal_weight = (1 - p_t) ** self.gamma
        return focal_weight * bce_loss          # same shape as input; caller handles masking/reduction


def create_loss_function(pos_weight, config: Config):
    """creates the loss function; focal loss when focal_gamma > 0"""
    gamma = float(getattr(config.tra, 'focal_gamma', 0.0))
    pw = pos_weight.to(config.device) if pos_weight is not None else None
    if gamma > 0:
        print(f"using focal loss with gamma={gamma}")
        return FocalBCELoss(pos_weight=pw, gamma=gamma)
    return nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

def calculate_metrics(y_true, y_pred, y_scores):
    """calculates comprehensive metrics for binary classification"""
    metrics = {}
    
    # basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # accuracy
    total = tp + tn + fp + fn
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0.0
    
    # precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # specificity (true negative rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # class distribution
    pos_samples = np.sum(y_true == 1)
    neg_samples = np.sum(y_true == 0)
    total_samples = len(y_true)
    metrics['positive_samples'] = pos_samples
    metrics['negative_samples'] = neg_samples
    metrics['positive_ratio'] = pos_samples / total_samples if total_samples > 0 else 0.0
    
    # roc-auc and pr-auc
    try:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = metrics['positive_ratio']
    except Exception as e:
        print(f"warning: could not calculate auc metrics: {e}")
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.0
    
    return metrics

def save_model(model, path):
    """saves the model state dictionary to a file"""
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """loads a model state dictionary from a file"""
    model.load_state_dict(torch.load(path))
    return model