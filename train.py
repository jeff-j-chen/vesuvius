# VERSION 2
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
    """Train for one epoch with L1 regularization"""
    model.train()
    train_loss, train_raw_loss = 0.0, 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
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
        # with torch.no_grad():
        #     scores = torch.sigmoid(outputs).cpu().numpy().flatten()
        #     predicted = (scores > 0.5).astype(int)
        #     labels = batch_labels.cpu().numpy().flatten().astype(int)
            
        #     all_labels.extend(labels)
        #     all_predictions.extend(predicted)
        #     all_scores.extend(scores)
        scores = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
        predicted = (scores > 0.5).astype(int)
        labels = batch_labels.cpu().detach().numpy().flatten().astype(int)

        all_labels.extend(labels)
        all_predictions.extend(predicted)
        all_scores.extend(scores)

    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_predictions), 
        np.array(all_scores)
    )
    
    metrics['loss'] = train_loss / len(train_loader)
    metrics['raw_loss'] = train_raw_loss / len(train_loader)
    
    return metrics

def validate_epoch(model, valid_loader, criterion, config: Config, scaler: GradScaler):
    """Validate for one epoch (unchanged)"""
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    with torch.no_grad(), autocast(config.device):
        for images, labels in tqdm(valid_loader, desc="Validating"):
            images = images.to(config.device)
            labels = labels.to(config.device).view(-1, 1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            predicted = (scores > 0.5).astype(int)
            labels_np = labels.cpu().numpy().flatten().astype(int)

            all_labels.extend(labels_np)
            all_predictions.extend(predicted)
            all_scores.extend(scores)
    
    metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_predictions), 
        np.array(all_scores)
    )
    
    metrics['loss'] = val_loss / len(valid_loader)
    
    return metrics

def print_metrics_summary(epoch, train_metrics, val_metrics, current_lr):
    """Print a comprehensive summary of metrics"""
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1} SUMMARY")
    print(f"{'='*80}")
    print(f"Learning Rate: {current_lr:.6f}")
    print(f"{'='*80}")
    
    # Loss metrics
    print(f"{'LOSS METRICS':<25} {'TRAIN':<15} {'VALIDATION':<15}")
    print(f"{'-'*55}")
    print(f"{'Total Loss':<25} {train_metrics['loss']:<15.4f} {val_metrics['loss']:<15.4f}")
    print(f"{'Raw Loss':<25} {train_metrics['raw_loss']:<15.4f} {val_metrics['loss']:<15.4f}")
    
    # Classification metrics
    print(f"\n{'CLASSIFICATION METRICS':<25} {'TRAIN':<15} {'VALIDATION':<15}")
    print(f"{'-'*55}")
    print(f"{'Accuracy':<25} {train_metrics['accuracy']:<15.4f} {val_metrics['accuracy']:<15.4f}")
    print(f"{'Precision':<25} {train_metrics['precision']:<15.4f} {val_metrics['precision']:<15.4f}")
    print(f"{'Recall':<25} {train_metrics['recall']:<15.4f} {val_metrics['recall']:<15.4f}")
    print(f"{'F1-Score':<25} {train_metrics['f1']:<15.4f} {val_metrics['f1']:<15.4f}")
    print(f"{'Specificity':<25} {train_metrics['specificity']:<15.4f} {val_metrics['specificity']:<15.4f}")
    
    # AUC metrics
    print(f"\n{'AUC METRICS':<25} {'TRAIN':<15} {'VALIDATION':<15}")
    print(f"{'-'*55}")
    print(f"{'ROC-AUC':<25} {train_metrics['roc_auc']:<15.4f} {val_metrics['roc_auc']:<15.4f}")
    print(f"{'PR-AUC':<25} {train_metrics['pr_auc']:<15.4f} {val_metrics['pr_auc']:<15.4f}")
    
    # Dataset balance
    print(f"\n{'DATASET BALANCE':<25} {'TRAIN':<15} {'VALIDATION':<15}")
    print(f"{'-'*55}")
    
    # Performance interpretation
    print(f"\n{'PERFORMANCE INTERPRETATION':<50}")
    print(f"{'-'*50}")
    
    val_precision = val_metrics['precision']
    val_recall = val_metrics['recall']
    val_f1 = val_metrics['f1']
    val_pr_auc = val_metrics['pr_auc']
    
    if val_precision > 0.7 and val_recall > 0.6:
        print("✓ GOOD: High precision and recall - model is performing well")
    elif val_precision > 0.5 and val_recall > 0.4:
        print("~ FAIR: Moderate precision and recall - room for improvement")
    else:
        print("✗ POOR: Low precision or recall - model needs significant improvement")
    
    if val_pr_auc > 0.4:
        print("✓ GOOD: High PR-AUC - model distinguishes classes well")
    elif val_pr_auc > 0.2:
        print("~ FAIR: Moderate PR-AUC - some discriminative ability")
    else:
        print("✗ POOR: Low PR-AUC - model struggles to distinguish classes")
    
    print(f"{'='*80}\n")

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
    if (epoch+1) % config.training.save_every_n_epochs == 0:
        save_model(model, f'{config.model_dir}/model_epoch_{epoch+1}.pth')

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
    volume, mask, labels, train_x_range, valid_x_range, y_range = load_tv_data(config)
    train_dataset, valid_dataset = get_tv_datasets(config)
    train_loader, valid_loader = get_dataloaders(train_dataset, valid_dataset, config)
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
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        # Train
        if epoch >= 5 and config.dataloader.apply_transforms:
            train_dataset.apply_transforms = True
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config, scaler)
        val_metrics = validate_epoch(model, valid_loader, criterion, config, scaler)
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        print_metrics_summary(epoch, train_metrics, val_metrics, current_lr)
        
        periodic_model_save(model, epoch, val_metrics, best_val_f1, best_val_loss)

        time_elapsed = time.time() - start_time
        vis.log_epoch_metrics(epoch, model, train_metrics, val_metrics, current_lr, time_elapsed, volume, labels, params)

    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    args = parser.parse_args()
    config = Config()
    config.experiment_name = args.experiment_name
    main(config)

    # # test stronger augmentations
    # for transform_type in ["brightness", "contrast"]:
    #     config = Config()
    #     config.dataloader.apply_transforms = True
    #     config.dataloader.transform_type = transform_type
    #     config.experiment_name = f"transform_{transform_type}_strong"
    #     print(f"Training with transform type: {transform_type}...")
    #     main(config)
    
    # for l1 in [7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4]:
    #     config = Config()
    #     config.training.l1_lambda = l1
    #     drops = [
    #         [0.0, 0.0, 0.2, 0.1],
    #         [0.0, 0.05, 0.2, 0.1],
    #         [0.0, 0.0, 0.3, 0.1],
    #         [0.0, 0.05, 0.3, 0.1],
    #         [0.0, 0.1, 0.3, 0.1],
    #         [0.0, 0, 0.4, 0.2],
    #         [0.0, 0.05, 0.4, 0.2],
    #         [0.0, 0.1, 0.4, 0.2],
    #         [0.0, 0.2, 0.5, 0.3],
    #         [0.0, 0.3, 0.6, 0.8],
    #     ]
    #     for drop in drops:
    #         config.model.conv1_drop = drop[0]
    #         config.model.conv2_drop = drop[1]
    #         config.model.fc1_drop = drop[2]
    #         config.model.fc2_drop = drop[3]
    #         config.experiment_name = f"alltrans_{l1:.0e}l1_{drop[0]}-{drop[1]}-{drop[2]}-{drop[3]}"
    #         main(config)

    # # test if a lower l1 will allow for mix to generalize
    # config = Config()
    # config.dataloader.transform_type = "mix"
    # config.experiment_name = f"mix_1e-4l1"
    # config.training.l1_lambda = 1e-4
    # main(config)
    
    # # test if a lower l1 will allow for mix to generalize
    # config = Config()
    # config.dataloader.apply_transforms = False
    # config.experiment_name = f"transform_off_128b"
    # config.training.l1_lambda = 7e-4
    # main(config)

    # # re-run tests on every augmentation, to see if they work better with a lower probability (33%)
    # for transform_type in ["brightness", "mix", "contrast", "noise", "rotate", "flip"]:
    #     config = Config()
    #     config.training.l1_lambda = 7e-4
    #     config.dataloader.apply_transforms = True
    #     config.dataloader.low_trans_prob = True
    #     config.dataloader.transform_type = transform_type
    #     config.experiment_name = f"transform_{transform_type}_lowprob"
    #     print(f"Training with transform type: {transform_type}...")
    #     main(config)

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
    