import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config

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

class GCEBCELoss(nn.Module):
    """generalized cross-entropy for binary labels (Zhang & Sabuncu, NeurIPS 2018).

    loss = (1 - p_t**q) / q, where p_t is the probability mass the model assigns to the
    TARGET class. q in (0,1] interpolates between cross-entropy (q->0, trusts every label)
    and the bounded, noise-robust MAE loss (q=1). the point: a mislabeled tile has low p_t,
    and (1 - p_t**q)/q SATURATES as p_t->0, so its gradient is bounded -- probable-wrong
    labels can no longer dominate the update the way -log(p_t) does. the model fits the
    self-consistent labels and effectively shrugs off the noisy ones. supports soft targets
    (label smoothing) via p_t = y*p + (1-y)*(1-p). reduction='none' to match the BCE
    interface (caller applies the mask + reduction)."""
    def __init__(self, q=0.7, pos_weight=None):
        super().__init__()
        self.q = float(q)
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)   # prob assigned to target class
        p_t = p_t.clamp(min=1e-6)
        loss = (1.0 - p_t ** self.q) / self.q
        if self.pos_weight is not None:
            # up-weight the positive class like BCEWithLogitsLoss(pos_weight=...)
            loss = loss * (1.0 + (self.pos_weight - 1.0) * targets)
        return loss

def create_loss_function(pos_weight, config: Config):
    """create the current tile loss function."""
    pw = pos_weight.to(config.device) if pos_weight is not None else None
    loss_type = str(getattr(config.tra, 'loss_type', 'bce')).lower()
    if loss_type == 'gce':
        q = float(getattr(config.tra, 'gce_q', 0.7))
        print(f"using GCE (noise-robust) loss with q={q}")
        return GCEBCELoss(q=q, pos_weight=pw)
    return nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

def calculate_metrics(y_true, y_pred, y_scores):
    """calculates comprehensive metrics for binary classification"""
    metrics = {}
    
    # basic counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # accuracy (raw — biased by class imbalance; use balanced_accuracy for ring datasets)
    total = tp + tn + fp + fn
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0.0
    # balanced accuracy: mean(sensitivity, specificity) — 0.5 when model predicts all one class,
    # 1.0 when perfect. unaffected by ring imbalance ratio, so comparable across splits.
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['balanced_accuracy'] = (sensitivity + specificity_val) / 2.0
    
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