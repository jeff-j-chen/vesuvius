import torch
from tqdm import tqdm
from utils.config import Config
from utils.dataloader import get_tv_datasets, get_dataloaders, calculate_class_weights, load_tv_data
from utils.model import create_model, InkDetector, CBAM3D
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    calculate_metrics,
    save_model
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from utils.visualizer import TensorboardVisualizer
import time
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import argparse
from utils.hard_mining import HardMiningManager, HardMiningInjector
import os

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, config: Config, scaler: GradScaler,
                hard_injector: HardMiningInjector):
    """Train for one epoch with L1 regularization and mask-based loss zeroing."""
    model.train()
    train_loss, train_raw_loss = 0.0, 0.0
    all_labels = []; all_predictions = []; all_scores = []
    total_batches = len(train_loader)
    # --- diagnostics counters ---
    hard_planned_total = 0
    hard_injected_total = 0
    hard_skipped_total = 0

    for batch_idx, (batch_images, batch_labels, mask) in enumerate(tqdm(train_loader, desc="Training")):
        # Hard mining injection
        if hard_injector and hard_injector.has_next():
            remaining_batches = total_batches - batch_idx
            remaining_needed = hard_injector.remaining()
            
            # Prevent division by zero if there are no remaining batches
            if remaining_batches <= 0:
                inject_n = 0
            else:
                inject_n = min(batch_images.size(0), (remaining_needed + remaining_batches - 1) // remaining_batches)

            hard_planned_total += inject_n
            actual_injected = 0
            if inject_n > 0:
                replace_indices = np.random.choice(batch_images.size(0), inject_n, replace=False)
                for ri in replace_indices:
                    sample = None
                    while hard_injector.has_next() and sample is None:
                        sample = hard_injector.next_sample()
                    if sample is None:
                        break
                    hi_block, hi_label, hi_mask = sample
                    batch_images[ri] = hi_block.to(config.device)
                    batch_labels[ri] = hi_label.to(config.device)
                    mask[ri] = hi_mask.to(config.device)
                    actual_injected += 1
                hard_injected_total += actual_injected
                hard_skipped_total += (inject_n - actual_injected)
        batch_images = batch_images.to(config.device)
        batch_labels = batch_labels.to(config.device).view(-1, 1)
        mask = mask.to(config.device).view(-1, 1)

        optimizer.zero_grad()
        with autocast(config.device):
            outputs = model(batch_images)
            raw_loss = criterion(outputs, batch_labels)
            raw_loss = raw_loss * mask
            if mask.sum() <= 0:
                print("[ERROR] Mask sum is zero, skipping loss calculation.")
                continue
            raw_loss = raw_loss.sum() / mask.sum()
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            loss = raw_loss + config.training.l1_lambda * l1_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_raw_loss += raw_loss.item()

        scores = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
        predicted = (scores > 0.5).astype(int)
        labels = batch_labels.cpu().detach().numpy().flatten().astype(int)
        all_labels.extend(labels); all_predictions.extend(predicted); all_scores.extend(scores)

    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions), np.array(all_scores))
    metrics['loss'] = train_loss / len(train_loader)
    metrics['raw_loss'] = train_raw_loss / len(train_loader)
    metrics['scores'] = all_scores
    # Diagnostics summary
    metrics['hard_planned'] = hard_planned_total
    metrics['hard_injected'] = hard_injected_total
    metrics['hard_skipped'] = hard_skipped_total
    if hard_injector:
        st = hard_injector.stats()
        print(f"[HARD][Epoch Summary] planned={hard_planned_total} injected={hard_injected_total} "
              f"skipped={hard_skipped_total} injector_used={st['used']} injector_skipped={st['skipped']} "
              f"reasons={st['skip_reasons']}")
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
    if (epoch+1) % config.training.save_every_n_epochs == 0:
        save_model(model, f'{config.model_dir}/model_epoch_{epoch+1}.pth')
    
    return best_val_f1, best_val_loss  # ensure updated values propagate

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
    hard_manager = HardMiningManager()
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    scaler = GradScaler()
    all_hard_samples = []  # Accumulate all hard samples here
    
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        # Activate transforms only after epoch 5
        if epoch >= 5 and config.dataloader.apply_transforms:
            train_dataset.apply_transforms = True
        
        # Check for new hard mining files at evaluation intervals
        prev_eval_epoch = epoch - 1
        if epoch > 5 and (prev_eval_epoch + 1) % config.training.evaluation_interval == 0:
            target_hard = int(config.hard_mining.next_iter_ratio * len(train_dataset))
            # Pass the visualizer's log_path to find the file
            new_samples = hard_manager.sample_for_epoch(prev_eval_epoch, target_hard)
            if new_samples:
                all_hard_samples.extend(new_samples)
                print(f"[HARD][Epoch {epoch}] Added {len(new_samples)} new hard samples. Total is now {len(all_hard_samples)}.")
                vis.writer.add_scalar("HardMining/TotalSamplesInPool", len(all_hard_samples), epoch)
            else:
                print(f"[HARD][Epoch {epoch}] Mining file processed but no new samples were added.")

        # Create injector if we have any hard samples in our pool
        hard_injector = None
        if all_hard_samples:
            hard_injector = HardMiningInjector(all_hard_samples, train_dataset)
            if (prev_eval_epoch + 1) % config.training.evaluation_interval == 0: # Log on new injection
                 vis.writer.add_scalar("HardMining/InjectedSamplesPlanned", len(all_hard_samples), epoch)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config, scaler, hard_injector)
        # Validate
        val_metrics = validate_epoch(model, valid_loader, criterion, config, scaler)
        # Scheduler & model save
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        best_val_f1, best_val_loss = periodic_model_save(model, epoch, val_metrics, best_val_f1, best_val_loss)
        # Logging
        time_elapsed = time.time() - start_time
        vis.log_epoch_metrics(epoch, model, train_metrics, val_metrics, current_lr, time_elapsed, params, pos_weight)
        # Extra logging to Tensorboard
        vis.writer.add_scalar("HardMining/Planned", train_metrics.get('hard_planned', 0), epoch)
        vis.writer.add_scalar("HardMining/Injected", train_metrics.get('hard_injected', 0), epoch)
        vis.writer.add_scalar("HardMining/Skipped", train_metrics.get('hard_skipped', 0), epoch)
    vis.close()
    print("Training completed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    args = parser.parse_args()
    config = Config()
    config.experiment_name = args.experiment_name
    main(config)