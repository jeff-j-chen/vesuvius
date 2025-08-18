import torch
from tqdm import tqdm
from utils.config import Config
from utils.dataloader import get_tv_datasets, get_dataloaders, calculate_class_weights, load_tv_data
from utils.model import create_model, InkDetector, CBAM3D
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    save_model
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from utils.visualizer import TensorboardVisualizer
import time
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import argparse
from utils.hard_mining import list_mining_files, load_mining_records, create_hard_mined_loader, ensure_hard_negs_dir, mining_filename, analyze_mining_files
import os

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    metrics['positive_ratio'] = pos_samples / total_samples

    
    # ROC-AUC and PR-AUC (handle edge cases)
    try:
        if len(np.unique(y_true)) == 2:  # Both classes present
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        else:
            metrics['roc_auc'] = 0.5  # Random performance when only one class
            metrics['pr_auc'] = pos_samples / total_samples  # Baseline for PR-AUC
    except Exception as e:
        print(f"Warning: Could not calculate AUC metrics: {e}")
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.0
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, config: Config, scaler: GradScaler):
    """Standard training epoch (no hard injection)."""
    model.train()
    train_loss, train_raw_loss = 0.0, 0.0
    all_labels=[]; all_preds=[]; all_scores=[]
    for batch_images, batch_labels, mask in tqdm(train_loader, desc="Training"):
        batch_images = batch_images.to(config.device)
        batch_labels = batch_labels.to(config.device).view(-1,1)
        mask = mask.to(config.device).view(-1,1)
        optimizer.zero_grad()
        with autocast(config.device):
            outputs = model(batch_images)
            raw_loss = criterion(outputs, batch_labels)*mask
            if mask.sum() <= 0:
                continue
            raw_loss = raw_loss.sum()/mask.sum()
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            loss = raw_loss + config.training.l1_lambda*l1_loss
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
        scaler.step(optimizer); scaler.update()
        train_loss += loss.item(); train_raw_loss += raw_loss.item()
        scores = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        preds = (scores>0.5).astype(int)
        labels = batch_labels.detach().cpu().numpy().flatten().astype(int)
        all_scores.extend(scores); all_preds.extend(preds); all_labels.extend(labels)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    metrics['loss']=train_loss/len(train_loader)
    metrics['raw_loss']=train_raw_loss/len(train_loader)
    metrics['scores']=all_scores
    return metrics


def validate_epoch(model, valid_loader, criterion, config: Config, scaler: GradScaler):
    """Validate for one epoch with mask-based loss zeroing."""
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    with torch.no_grad(), autocast(config.device):
        for batch_images, batch_labels, mask in tqdm(valid_loader, desc="Validating"):
            # Safety check: mask sum must be non-zero
            if mask.sum() <= 0:
                print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")

            batch_images = batch_images.to(config.device)
            batch_labels = batch_labels.to(config.device).view(-1, 1)
            mask = mask.to(config.device).view(-1, 1)  # Ensure mask matches the shape of the loss

            outputs = model(batch_images)
            raw_loss = criterion(outputs, batch_labels)

            # Zero out loss in masked-out regions
            raw_loss = raw_loss * mask  # Apply mask to the loss
            if mask.sum() <= 0:
                print("[ERROR] Mask sum is zero, skipping loss calculation.")
                continue
            raw_loss = raw_loss.sum() / mask.sum()  # Normalize by the number of valid regions

            val_loss += raw_loss.item()

            # Predictions and metrics
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            predicted = (scores > 0.5).astype(int)
            labels_np = batch_labels.cpu().numpy().flatten().astype(int)

            all_labels.extend(labels_np)
            all_predictions.extend(predicted)
            all_scores.extend(scores)

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_scores)
    )

    metrics['loss'] = val_loss / len(valid_loader)
    metrics['scores'] = all_scores

    return metrics

def periodic_model_save(model, epoch, val_metrics, best_val_f1, best_val_loss):
    # Save best model based on F1 score (better metric for imbalanced data)
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        save_model(model, f'{config.model_dir}/best_model_f1.pth')
        print(f"New best F1 model saved! Val F1: {best_val_f1:.4f}")
    
    # Also save based on loss for comparison
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        save_model(model, f'{config.model_dir}/best_model_loss.pth')
        print(f"New best loss model saved! Val Loss: {best_val_loss:.4f}")
    
    # Save periodic checkpoints
    if (epoch+1) % config.training.epoch_save_freq == 0:
        save_model(model, f'{config.model_dir}/model_epoch_{epoch+1}.pth')
    
    return best_val_f1, best_val_loss  # ensure updated values propagate

def train_mined(model, criterion, optimizer, scaler, config, mining_files, volume, mask, labels, stats):
    """
    Train sequentially over each mining file (one mini-epoch per file).
    Fixes:
      - Expect scalar mask per sample (1.), no per-pixel flatten.
      - Remove unintended huge normalization from pixel mask sum.
      - Consistent raw_loss averaging with standard training.
    """
    per_file_metrics = {}
    all_labels = []; all_preds=[]; all_scores=[]
    total_loss = 0.0; total_raw_loss = 0.0; total_batches=0
    did_shape_debug = False
    for mf in mining_files:
        records = load_mining_records(mf)
        if not records:
            print(f"[HARD][Mined Train] Skipping empty file {mf}")
            continue
        ds, loader = create_hard_mined_loader(records, volume, mask, labels, stats, config)
        file_loss=0.0; file_raw_loss=0.0
        f_labels=[]; f_preds=[]; f_scores=[]
        model.train()
        for batch_images, batch_labels, batch_mask in tqdm(loader, desc=f"Mining {os.path.basename(mf)}"):
            # Expected shapes: batch_images (B,1,D,H,W), batch_labels (B,1), batch_mask (B,1)
            if not did_shape_debug:
                print(f"[HARD][dbg] shapes images={tuple(batch_images.shape)} labels={tuple(batch_labels.shape)} mask={tuple(batch_mask.shape)}")
                did_shape_debug = True
            batch_images = batch_images.to(config.device)
            batch_labels = batch_labels.to(config.device).view(-1,1)
            batch_mask = batch_mask.to(config.device).view(-1,1)  # now (B,1) scalars of 1.
            optimizer.zero_grad()
            with autocast(config.device):
                out = model(batch_images)
                # Per-sample unreduced loss
                raw_loss_full = criterion(out, batch_labels)  # scalar OR (B,1) depending on reduction
                if raw_loss_full.ndim > 0:
                    # Apply sample mask then mean over valid
                    masked = raw_loss_full * batch_mask
                    valid = batch_mask.sum()
                    if valid <= 0:
                        continue
                    raw_loss = masked.sum() / valid
                else:
                    raw_loss = raw_loss_full  # already scalar (BCE default mean)
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = raw_loss + config.training.l1_lambda * l1_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
            scaler.step(optimizer); scaler.update()
            file_loss += loss.item(); file_raw_loss += raw_loss.item()
            total_loss += loss.item(); total_raw_loss += raw_loss.item()
            total_batches += 1
            scores = torch.sigmoid(out).detach().cpu().numpy().flatten()
            preds = (scores>0.5).astype(int)
            labs = batch_labels.detach().cpu().numpy().flatten().astype(int)
            f_scores.extend(scores); f_preds.extend(preds); f_labels.extend(labs)
            all_scores.extend(scores); all_preds.extend(preds); all_labels.extend(labs)
        if f_labels:
            fm = calculate_metrics(np.array(f_labels), np.array(f_preds), np.array(f_scores))
            fm['loss']=file_loss/ max(1,len(loader))
            fm['raw_loss']=file_raw_loss/max(1,len(loader))
            fm['scores']=f_scores
            per_file_metrics[mf]=fm
    return per_file_metrics

def main(config: Config):
    set_seed(41)
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if isinstance(value, dict):
            for subfield, subvalue in value.items():
                print(f"{field}.{subfield}: {subvalue}")
        else:
            print(f"{field}: {value}")

    print("Creating datasets...")
    start_time = time.time()
    volume, mask, labels, train_x_range, valid_x_range, y_range = load_tv_data(config)
    train_dataset, valid_dataset = get_tv_datasets(config)
    train_loader, valid_loader = get_dataloaders(train_dataset, valid_dataset, config)
    # Normalization stats for mined data reuse
    norm_stats = {
        "mean": train_dataset.global_mean,
        "std": train_dataset.global_std,
        "min": train_dataset.global_min,
        "max": train_dataset.global_max
    }
    print(f" done in {time.time() - start_time:.2f}s")
    # Create model
    print(f"Creating model and loss... l1 lamba {config.training.l1_lambda}... ", end="")
    start_time = time.time()
    model, params = create_model(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    pos_weight = calculate_class_weights(train_dataset, valid_dataset)
    criterion = create_loss_function(pos_weight, config)
    print(f" done in {time.time() - start_time:.2f}s")

    # Initialize Tensorboard
    print("Initializing Tensorboard...")
    start_time = time.time()
    vis = TensorboardVisualizer(config)
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    scaler = GradScaler()
    mining_debug_done = False
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        # Activate transforms only after epoch 5
        if epoch >= 5 and config.dataloader.apply_transforms:
            train_dataset.apply_transforms = True
        
        # --- Standard Train ---
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config, scaler)
        # --- Validation ---
        val_metrics = validate_epoch(model, valid_loader, criterion, config, scaler)
        # --- Mining ---
        if epoch + 1 > config.training.evaluation_interval:
            mining_files = list_mining_files(upto_epoch=epoch)
            # One-time integrity diagnostics
            if not mining_debug_done:
                analyze_mining_files(
                    mining_files,
                    labels,
                    volume_depth=volume.shape[0],
                    tile_size=config.data.tile_size,
                    depth=config.data.depth,
                    sample_cap=3000
                )
                mining_debug_done = True
            mine_metrics = train_mined(
                model, criterion, optimizer, scaler, config,
                mining_files, volume, mask, labels, norm_stats
            )
            vis.log_hard_mined_metrics(epoch, mine_metrics)
            
        scheduler.step(val_metrics['loss'], epoch)
        current_lr = optimizer.param_groups[0]['lr']
        time_elapsed = time.time() - start_time

        vis.log_epoch_metrics(epoch, model, train_metrics, val_metrics, current_lr, time_elapsed, params, pos_weight)

        best_val_f1, best_val_loss = periodic_model_save(model, epoch, val_metrics, best_val_f1, best_val_loss)

    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    args = parser.parse_args()
    config = Config()
    config.experiment_name = args.experiment_name
    main(config)