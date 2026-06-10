import os

# keep tensorboard/tensorflow startup noise low and deterministic across windows workers
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from utils.visualizer import TensorboardVisualizer
from utils.hard_mining import HardMiningManager, HardMiningInjector
from utils.dataloader import DataManager, get_dataloaders, calc_class_wgts
from utils.model import create_model
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    calculate_metrics,
    save_model
)

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from tqdm import tqdm
import time
import argparse
import random


def _apply_cli_overrides(c: Config, args):
    """apply optional CLI overrides to config"""
    if args.epochs is not None:
        c.tra.n_epochs = int(args.epochs)
    if args.eval_int is not None:
        c.tra.eval_int = int(args.eval_int)
    if args.test_int is not None:
        c.tra.test_int = int(args.test_int)
    if args.probe_int is not None:
        c.tra.probe_int = int(args.probe_int)

    if args.scroll_id is not None:
        c.data.scroll1_id = int(args.scroll_id)
    if args.scroll4_id is not None:
        c.data.scroll4_id = int(args.scroll4_id)
    if args.zarr_path is not None:
        c.data.zarr_path = args.zarr_path

    if args.batch_size is not None:
        c.dl.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        c.dl.num_workers = int(args.num_workers)
    if args.data_aug is not None:
        c.dl.data_aug = bool(int(args.data_aug))

    if args.lr is not None:
        c.tra.lr = float(args.lr)
    if args.weight_decay is not None:
        c.tra.weight_decay = float(args.weight_decay)
    if args.l1_lambda is not None:
        c.tra.l1_lambda = float(args.l1_lambda)

    if args.no_hard_mining:
        c.hm.enabled = False
    if args.hm_frac is not None:
        c.hm.hm_frac = float(args.hm_frac)
    if args.hn_cutoff is not None:
        c.hm.hn_cutoff = float(args.hn_cutoff)
    if args.hp_cutoff is not None:
        c.hm.hp_cutoff = float(args.hp_cutoff)
    if args.hm_dir is not None:
        c.hm.dir = args.hm_dir

    if args.channel_mixing_prob is not None:
        c.dl.channel_mixing_prob = float(args.channel_mixing_prob)

    if args.pooling is not None:
        c.model.pooling = str(args.pooling)
    if args.gem_p is not None:
        c.model.gem_p = float(args.gem_p)
    if args.conv3_dilation is not None:
        c.model.conv3_dilation = int(args.conv3_dilation)
    if args.arch is not None:
        c.model.arch = str(args.arch)
    if args.conv1_drop is not None:
        c.model.conv1_drop = float(args.conv1_drop)
    if args.conv2_drop is not None:
        c.model.conv2_drop = float(args.conv2_drop)
    if args.fc1_drop is not None:
        c.model.fc1_drop = float(args.fc1_drop)
    if args.fc2_drop is not None:
        c.model.fc2_drop = float(args.fc2_drop)

def set_seed(seed=42):
    """sets the seed for reproducibility across all relevant libraries"""
    # benchmark=True: cuDNN profiles algorithms on the first batch and caches the winner
    # deterministic=True would conflict — it prevents caching, causing re-benchmarking
    # on every forward pass and growing cuDNN workspace memory until OOM
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer:
    """manages the model training and validation process"""
    def __init__(self, config):
        """initializes the trainer, setting up data, model, and logging"""
        self.c = config
        set_seed(41)
        self._print_config()
        
        # initialize data, model, and optimizer
        self.train_dataset, \
        self.train_loader, \
        self.valid_loader, \
        self.pos_weight = self._setup_data()
        
        self.model, \
        self.params, \
        self.optimizer, \
        self.scheduler, \
        self.criterion = self._setup_model_optim()
        
        # setup logging and visualization
        print("Initializing Tensorboard...")
        self.vis = TensorboardVisualizer(self.c)
        
        # initialize training components
        self.scaler = GradScaler()
        self.hard_manager = HardMiningManager(self.c.hm.dir)
        self.hard_samples = []
        
        # initialize tracking variables for best model saving
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0

    def _print_config(self):
        """prints the configuration parameters"""
        print("--- Configuration ---")
        for field in self.c.__dataclass_fields__:
            value = getattr(self.c, field)
            if isinstance(value, dict):
                for subfield, subvalue in value.items():
                    print(f"{field}.{subfield}: {subvalue}")
            else:
                print(f"{field}: {value}")
        print("---------------------")

    def _setup_data(self):
        """loads and prepares the datasets and dataloaders"""
        print("Creating datasets...")
        start_time = time.time()
        
        data_manager = DataManager(self.c)
        t_set, v_set = data_manager.get_datasets()
        t_loader, v_loader = get_dataloaders(t_set, v_set, self.c)
        pos_weight = calc_class_wgts(t_set, v_set, scroll_id=self.c.data.scroll1_id)
        
        print(f"Data setup done in {time.time() - start_time:.2f}s")
        return t_set, t_loader, v_loader, pos_weight

    def _setup_model_optim(self):
        """creates the model, optimizer, scheduler, and loss function"""
        print(f"Creating model and loss... l1 lambda {self.c.tra.l1_lambda}... ", end="")
        start_time = time.time()
        
        model, params = create_model(self.c)
        optimizer, scheduler = create_optimizer_and_scheduler(model, self.c)
        criterion = create_loss_function(self.pos_weight, self.c)
        
        print(f" done in {time.time() - start_time:.2f}s")
        return model, params, optimizer, scheduler, criterion

    def _train_batch(self, b_imgs, b_labels, mask):
        """trains the model on a single batch of data"""
        b_imgs = b_imgs.to(self.c.device)
        b_labels = b_labels.to(self.c.device).view(-1, 1)
        mask = mask.to(self.c.device).view(-1, 1)

        self.optimizer.zero_grad()

        # forward pass with automatic mixed precision
        with autocast(self.c.device):
            outputs = self.model(b_imgs)
            raw_loss = self.criterion(outputs, b_labels)
            
            # apply mask to zero out loss in irrelevant regions
            raw_loss = raw_loss * mask
            if mask.sum() <= 0:
                print("[ERROR] Mask sum is zero, skipping loss calculation.")
                return np.empty([]), np.empty([]), 0.0, 0.0
            
            # normalize loss and add l1 regularization
            raw_loss_val = (raw_loss.sum() / mask.sum()).item()
            l1_loss = sum(p.abs().sum() for p in self.model.parameters())
            loss = (raw_loss.sum() / mask.sum()) + self.c.tra.l1_lambda * l1_loss

        # backward pass and optimization step
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # calculate scores and labels for metrics
        scores = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
        labels = b_labels.cpu().detach().numpy().flatten().astype(int)
        
        return scores, labels, loss.item(), raw_loss_val

    def _ins_hard_samples(self, b_imgs, b_labels, mask, hard_injector, rem_batches):
        """injects hard-mined samples into the current batch"""
        if not hard_injector or not hard_injector.has_next():
            return 0

        # calculate how many samples to inject in this batch to distribute them evenly
        if rem_batches <= 0:
            inject_n = 0
        else:
            inject_n = min(b_imgs.size(0), (hard_injector.remaining() + rem_batches - 1) // rem_batches)

        injected_n = 0
        if inject_n > 0:
            # randomly select indices in the batch to replace with hard samples
            replace_indices = np.random.choice(b_imgs.size(0), inject_n, replace=False)
            for ri in replace_indices:
                sample = None
                while hard_injector.has_next() and sample is None:
                    sample = hard_injector.next_sample()
                
                if sample is None:
                    break
                
                # replace batch data with hard-mined sample data
                hi_block, hi_label, hi_mask = sample
                b_imgs[ri] = hi_block.to(self.c.device)
                b_labels[ri] = hi_label.to(self.c.device)
                mask[ri] = hi_mask.to(self.c.device)
                injected_n += 1
        
        return injected_n

    def train_epoch(self, hard_injector):
        """runs a single training epoch"""
        self.model.train()
        loss, raw_loss = 0.0, 0.0
        labels, preds, scores = [], [], []
        total_inj = 0

        for batch_idx, (b_imgs, b_labels, mask) in enumerate(tqdm(self.train_loader, desc="Training")):
            # inject hard-mined samples into the batch
            total_inj += self._ins_hard_samples(
                b_imgs, b_labels, mask, hard_injector, len(self.train_loader) - batch_idx
            )

            # train on the (potentially modified) batch
            b_scores, b_labels, b_loss, b_raw_loss = self._train_batch(b_imgs, b_labels, mask)

            # accumulate results for epoch-level metrics
            if b_scores.size > 0:  # check if the batch was skipped
                loss += b_loss
                raw_loss += b_raw_loss
                labels.extend(b_labels)
                preds.extend((b_scores > 0.5).astype(int))
                scores.extend(b_scores)

        # calculate and return epoch metrics
        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        metrics['loss'] = loss / len(self.train_loader)
        metrics['raw_loss'] = raw_loss / len(self.train_loader)
        metrics['scores'] = scores
        metrics['hard_injected'] = total_inj
        
        if hard_injector:
            st = hard_injector.stats()
            print(f"[HARD][Epoch Summary] injected={total_inj} "
                  f"injector_used={st['used']} injector_skipped={st['skipped']}")
        return metrics

    def validate_epoch(self):
        """runs a single validation epoch"""
        self.model.eval()
        loss = 0.0
        labels, preds, scores = [], [], []
        
        with torch.no_grad(), autocast(self.c.device):
            for b_imgs, b_labels, mask in tqdm(self.valid_loader, desc="Validating"):
                if mask.sum() <= 0:
                    print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")
                    continue

                b_imgs = b_imgs.to(self.c.device)
                b_labels = b_labels.to(self.c.device).view(-1, 1)
                mask = mask.to(self.c.device).view(-1, 1)

                # forward pass
                outputs = self.model(b_imgs)
                
                # calculate loss
                raw_loss = self.criterion(outputs, b_labels)
                raw_loss = (raw_loss * mask).sum() / mask.sum()
                loss += raw_loss.item()

                # calculate scores and accumulate for metrics
                b_scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                labels.extend(b_labels.cpu().numpy().flatten().astype(int))
                preds.extend((b_scores > 0.5).astype(int))
                scores.extend(b_scores)

        # calculate and return epoch metrics
        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        metrics['loss'] = loss / len(self.valid_loader)
        metrics['scores'] = scores
        return metrics

    def _periodic_model_save(self, epoch, val_metrics):
        """saves the model periodically and based on performance"""
        # save best model based on f1 score
        if val_metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = val_metrics['f1']
            save_model(self.model, f'{self.c.model_dir}/best_model_f1.pth')
            print(f"New best F1 model saved! Val F1: {self.best_val_f1:.4f}")
        
        # save best model based on validation loss
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            save_model(self.model, f'{self.c.model_dir}/best_model_loss.pth')
            print(f"New best loss model saved! Val Loss: {self.best_val_loss:.4f}")
        
        # save periodic checkpoint
        if (epoch + 1) % self.c.tra.save_int == 0:
            save_model(self.model, f'{self.c.model_dir}/model_epoch_{epoch+1}.pth')

    def _update_hard_mining_samples(self, epoch):
        """checks for and loads new hard-mined samples at specified intervals"""
        if not self.c.hm.enabled:
            return
        # periodically check for new hard mining data
        if epoch % self.c.tra.eval_int == 0 and epoch > 5:
            target_hard = int(self.c.hm.hm_frac * len(self.train_dataset))
            
            # attempt to load new samples from the previous epoch's evaluation
            new_samples = self.hard_manager.sample_for_epoch(epoch - 1, target_hard)
            
            if new_samples:
                self.hard_samples.extend(new_samples)
                print(f"[HARD][Epoch {epoch}] Added {len(new_samples)} new hard samples. Total is now {len(self.hard_samples)}.")
                self.vis.writer.add_scalar("HardMining/TotalSamplesInPool", len(self.hard_samples), epoch)
            else:
                print(f"[HARD][Epoch {epoch}] Mining file processed but no new samples were added.")

    def _log_epoch(self, epoch, train_metrics, val_metrics, time_elapsed):
        """logs metrics for the completed epoch to tensorboard"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.vis.log_epoch_metrics(epoch, self.model, train_metrics, val_metrics, current_lr, time_elapsed, self.params, self.pos_weight)
        self.vis.writer.add_scalar("HardMining/Injected", train_metrics.get('hard_injected', 0), epoch)

    def run(self):
        """executes the main training loop for all epochs"""
        for epoch in range(self.c.tra.n_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.c.tra.n_epochs} ---")
            start_time = time.time()
            
            # activate data augmentation after a few initial epochs
            if epoch >= 5 and self.c.dl.data_aug:
                self.train_dataset.apply_transforms = True
            
            # check for new hard-mined samples to add to the pool
            self._update_hard_mining_samples(epoch)
            
            # create a new injector for the epoch if hard samples are available
            hard_injector = None
            if self.hard_samples and self.c.hm.enabled:
                hard_injector = HardMiningInjector(self.hard_samples, self.train_dataset)
                if (epoch % self.c.tra.eval_int) == 0:
                    self.vis.writer.add_scalar("HardMining/InjectedSamplesPlanned", len(self.hard_samples), epoch)

            # run training and validation for the epoch
            train_metrics = self.train_epoch(hard_injector)
            val_metrics = self.validate_epoch()
            
            # update learning rate scheduler and save models
            self.scheduler.step(val_metrics['loss'])
            self._periodic_model_save(epoch, val_metrics)
            
            # log results
            time_elapsed = time.time() - start_time
            self._log_epoch(epoch, train_metrics, val_metrics, time_elapsed)

        self.vis.close()
        print("Training completed.")

def main():
    """parses arguments, initializes the configuration, and starts training"""
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--eval-int", type=int, default=None, help="Override evaluation interval")
    parser.add_argument("--test-int", type=int, default=None, help="Override test figure interval")
    parser.add_argument("--probe-int", type=int, default=None, help="Override probe interval")

    parser.add_argument("--scroll-id", type=int, default=None, help="Scroll id for train/valid")
    parser.add_argument("--scroll4-id", type=int, default=None, help="Scroll id for scroll4 eval path")
    parser.add_argument("--zarr-path", type=str, default=None, help="Path to zarr root")

    parser.add_argument("--batch-size", type=int, default=None, help="Dataloader batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument("--data-aug", type=int, choices=[0, 1], default=None, help="Enable/disable data augmentation")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--l1-lambda", type=float, default=None, help="L1 regularization strength")

    parser.add_argument("--no-hard-mining", action="store_true", help="Disable hard mining entirely")
    parser.add_argument("--hm-frac", type=float, default=None, help="Hard-mining sample fraction")
    parser.add_argument("--hn-cutoff", type=float, default=None, help="Hard-negative score cutoff")
    parser.add_argument("--hp-cutoff", type=float, default=None, help="Hard-positive score cutoff")
    parser.add_argument("--hm-dir", type=str, default=None, help="Hard-mining directory")

    parser.add_argument("--channel-mixing-prob", type=float, default=None, help="Depth channel permutation probability")
    parser.add_argument("--pooling", type=str, choices=["avg", "max", "gem"], default=None, help="Pooling mode")
    parser.add_argument("--gem-p", type=float, default=None, help="Initial GeM pooling p")
    parser.add_argument("--conv3-dilation", type=int, default=None, help="Dilation for final conv stage")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture variant (v1, v2_slim_head, ...)")    
    parser.add_argument("--conv1-drop", type=float, default=None, help="Dropout after first conv block")
    parser.add_argument("--conv2-drop", type=float, default=None, help="Dropout after second conv block")
    parser.add_argument("--fc1-drop", type=float, default=None, help="Dropout on first FC layers")
    parser.add_argument("--fc2-drop", type=float, default=None, help="Dropout on final FC layer")
    args = parser.parse_args()
    
    # load configuration and optionally override experiment name
    c = Config()
    if args.experiment_name:
        c.exp_name = args.experiment_name
    _apply_cli_overrides(c, args)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if c.exp_name:
        c.hm.dir = os.path.join(c.hm.dir, c.exp_name)

    if not os.path.isabs(c.tra.log_dir):
        c.tra.log_dir = os.path.normpath(os.path.join(repo_root, c.tra.log_dir))
    if not os.path.isabs(c.hm.dir):
        c.hm.dir = os.path.normpath(os.path.join(repo_root, c.hm.dir))
    if not os.path.isabs(c.model_dir):
        c.model_dir = os.path.normpath(os.path.join(repo_root, c.model_dir))
        
    # initialize and run the trainer
    trainer = Trainer(c)
    trainer.run()

if __name__ == "__main__":
    main()