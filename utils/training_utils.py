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
        self.warmup_epochs = int(warmup_epochs)
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self.plateau_scheduler = plateau_scheduler
        self.base_lr = base_lr
        self.completed_epochs = 0
        if self.warmup_epochs > 0:
            self._set_lr(self.base_lr / self.warmup_epochs)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            scale = float(param_group.get("lr_scale", 1.0))
            param_group['lr'] = lr * scale if param_group.get("lr_enabled", True) else 0.0

    def step(self, val_loss=None):
        """sets the learning rate for the next epoch"""
        self.completed_epochs += 1
        if self.completed_epochs < self.warmup_epochs:
            next_epoch = self.completed_epochs + 1
            self._set_lr(self.base_lr * next_epoch / self.warmup_epochs)
        else:
            self.plateau_scheduler.step(val_loss)


def create_optimizer_and_scheduler(model, config: Config):
    """creates an optimizer and a learning rate scheduler"""
    encoder_lr_scale = float(getattr(config.tra, "encoder_lr_scale", 1.0))
    encoder_freeze_epochs = int(getattr(config.tra, "encoder_freeze_epochs", 0))
    if encoder_lr_scale <= 0:
        raise ValueError("encoder_lr_scale must be positive")
    if encoder_freeze_epochs < 0:
        raise ValueError("encoder_freeze_epochs must be non-negative")

    parameter_groups = model.parameters()
    if encoder_lr_scale != 1.0 or encoder_freeze_epochs > 0:
        encoder_prefixes = ("enc1.", "enc2.", "enc3.", "bottleneck.")
        encoder_parameters = []
        task_parameters = []
        for name, parameter in model.named_parameters():
            destination = encoder_parameters if name.startswith(encoder_prefixes) else task_parameters
            destination.append(parameter)
        parameter_groups = [
            {
                "params": task_parameters,
                "group_name": "task",
                "lr_scale": 1.0,
                "lr_enabled": True,
            },
            {
                "params": encoder_parameters,
                "group_name": "encoder",
                "lr_scale": encoder_lr_scale,
                "lr_enabled": encoder_freeze_epochs == 0,
            },
        ]

    optimizer = optim.AdamW(
        parameter_groups,
        lr=config.tra.lr,
        weight_decay=config.tra.weight_decay,
        fused=bool(str(config.device).startswith("cuda") and torch.cuda.is_available()),
    )
    
    # create the combined warmup and plateau scheduler
    scheduler = WarmupThenPlateau(
        optimizer,
        warmup_epochs=config.tra.warmup_epochs,
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


def calculate_character_metrics(
    y_true,
    y_scores,
    character_ids,
    score_threshold=0.5,
    recall_target=0.5,
    max_ring_fpr=0.1,
):
    """macro metrics over connected ink characters and their associated ring negatives."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_scores = np.asarray(y_scores).reshape(-1)
    character_ids = np.asarray(character_ids).reshape(-1).astype(np.int64)
    keep = (character_ids > 0) & np.isfinite(y_scores)
    y_true, y_scores, character_ids = y_true[keep], y_scores[keep], character_ids[keep]

    recalls, fprs, f1s, aps, successes = [], [], [], [], []
    for component_id in np.unique(character_ids):
        selected = character_ids == component_id
        labels = y_true[selected]
        scores = y_scores[selected]
        positive = labels == 1
        negative = labels == 0
        if not positive.any() or not negative.any():
            continue
        predicted = scores >= float(score_threshold)
        recall = float(predicted[positive].mean())
        fpr = float(predicted[negative].mean())
        tp = int((predicted & positive).sum())
        fp = int((predicted & negative).sum())
        fn = int((~predicted & positive).sum())
        f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
        ap = float(average_precision_score(labels, scores))
        recalls.append(recall)
        fprs.append(fpr)
        f1s.append(f1)
        aps.append(ap)
        successes.append(recall >= float(recall_target) and fpr <= float(max_ring_fpr))

    if not recalls:
        return {
            "character_count": 0,
            "character_recall_macro": 0.0,
            "character_ring_fpr_macro": 0.0,
            "character_f1_macro": 0.0,
            "character_ap_macro": 0.0,
            "character_success_fraction": 0.0,
        }
    return {
        "character_count": len(recalls),
        "character_recall_macro": float(np.mean(recalls)),
        "character_ring_fpr_macro": float(np.mean(fprs)),
        "character_f1_macro": float(np.mean(f1s)),
        "character_ap_macro": float(np.mean(aps)),
        "character_success_fraction": float(np.mean(successes)),
    }


def calibrate_character_threshold(
    y_true,
    y_scores,
    character_ids,
    threshold_min=0.1,
    threshold_max=0.9,
    threshold_steps=33,
    recall_target=0.5,
    max_ring_fpr=0.1,
):
    """select one validation threshold that maximizes character macro f1."""
    if threshold_steps < 2 or not 0.0 <= threshold_min < threshold_max <= 1.0:
        raise ValueError("character threshold sweep bounds are invalid")
    labels = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(y_scores).reshape(-1)
    component_ids = np.asarray(character_ids).reshape(-1).astype(np.int64)
    keep = (component_ids > 0) & np.isfinite(scores)
    labels, scores, component_ids = labels[keep], scores[keep], component_ids[keep]
    thresholds = np.linspace(
        float(threshold_min),
        float(threshold_max),
        int(threshold_steps),
        dtype=np.float64,
    )
    component_f1 = []
    for component_id in np.unique(component_ids):
        selected = component_ids == component_id
        component_labels = labels[selected]
        positive = component_labels == 1
        negative = component_labels == 0
        if not positive.any() or not negative.any():
            continue
        predicted = scores[selected, None] >= thresholds[None, :]
        true_positive = (predicted & positive[:, None]).sum(axis=0)
        false_positive = (predicted & negative[:, None]).sum(axis=0)
        false_negative = ((~predicted) & positive[:, None]).sum(axis=0)
        component_f1.append(
            2.0 * true_positive
            / np.maximum(2.0 * true_positive + false_positive + false_negative, 1)
        )
    if not component_f1:
        return {
            "character_calibrated_threshold": 0.5,
            "character_calibrated_f1_macro": 0.0,
            "character_calibrated_recall_macro": 0.0,
            "character_calibrated_ring_fpr_macro": 0.0,
            "character_calibrated_success_fraction": 0.0,
        }
    macro_f1 = np.mean(np.stack(component_f1), axis=0)
    best_value = float(macro_f1.max())
    candidates = np.flatnonzero(np.isclose(macro_f1, best_value))
    best_index = min(candidates, key=lambda index: abs(float(thresholds[index]) - 0.5))
    threshold = float(thresholds[best_index])
    metrics = calculate_character_metrics(
        labels,
        scores,
        component_ids,
        score_threshold=threshold,
        recall_target=recall_target,
        max_ring_fpr=max_ring_fpr,
    )
    return {
        "character_calibrated_threshold": threshold,
        "character_calibrated_f1_macro": metrics["character_f1_macro"],
        "character_calibrated_recall_macro": metrics["character_recall_macro"],
        "character_calibrated_ring_fpr_macro": metrics["character_ring_fpr_macro"],
        "character_calibrated_success_fraction": metrics["character_success_fraction"],
    }

def save_model(model, path):
    """saves the model state dictionary to a file"""
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """loads a model state dictionary from a file"""
    model.load_state_dict(torch.load(path))
    return model