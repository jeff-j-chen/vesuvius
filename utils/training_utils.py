import torch
import torch.nn as nn
import torch.optim as optim
import os
from .config import Config

class WarmupThenPlateau:
    def __init__(self, optimizer, config, plateau_scheduler, base_lr):
        self.optimizer = optimizer
        self.config = config
        self.plateau_scheduler = plateau_scheduler
        self.base_lr = base_lr
        self.current_epoch = 0
        # Evaluation-interval re-warm state
        self.in_eval_rewarm = False
        self.eval_rewarm_step = 0
        self.eval_target_lr = -1
        self.eval_start_lr = -1

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, val_loss, epoch):
        """
        Advance scheduler one epoch.
        - Initial warmup (linear to base_lr)
        - Normal phase: ReduceLROnPlateau
        - At evaluation interval epochs: instantaneous drop (factor) + linear re-warm over next N epochs back to pre-drop LR.
        """
        # 1. Initial warmup phase
        if self.current_epoch < self.config.training.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.config.training.warmup_epochs
            self._set_lr(lr)
            self.current_epoch += 1
            return

        # 2. If currently in evaluation re-warm phase
        if self.in_eval_rewarm:
            # progress AFTER this call
            progress = (self.eval_rewarm_step + 1) / self.config.training.eval_rewarm_epochs
            new_lr = self.eval_start_lr + (self.eval_target_lr - self.eval_start_lr) * progress
            self._set_lr(new_lr)
            self.eval_rewarm_step += 1
            if self.eval_rewarm_step >= self.config.training.eval_rewarm_epochs:
                # Finish re-warm
                self.in_eval_rewarm = False
                self.eval_rewarm_step = 0
                self.eval_start_lr = -1
                # Ensure exact target
                self._set_lr(self.eval_target_lr)
            self.current_epoch += 1
            return

        # 3. Normal plateau step
        # Only run plateau scheduler if not in re-warm
        if val_loss is not None:
            self.plateau_scheduler.step(val_loss)

        # Capture post-plateau LR (baseline for potential drop)
        current_lr = self.get_lr()

        # 4. Check evaluation interval trigger
        if (epoch + 1) % self.config.training.evaluation_interval == 0 and not self.in_eval_rewarm:
            # Start eval-induced cooldown + re-warm
            self.eval_target_lr = current_lr  # store target to return to
            self.eval_start_lr = current_lr * self.config.training.eval_drop_factor
            self._set_lr(self.eval_start_lr)  # immediate drop
            self.in_eval_rewarm = True
            self.eval_rewarm_step = 0
            self.current_epoch += 1
            return

        # 5. No special action; keep plateau LR
        self.current_epoch += 1


def create_optimizer_and_scheduler(model, config: Config):
    """Create optimizer and learning rate scheduler with evaluation-interval cooldown."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = WarmupThenPlateau(
        optimizer,
        config,
        plateau_scheduler=optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.lr_scheduler_factor,
            patience=config.training.patience
        ),
        base_lr=config.training.learning_rate,
    )
    return optimizer, scheduler

def create_loss_function(pos_weight, config: Config):
    """Create loss function with optional class weighting"""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(config.device) if pos_weight is not None else None
    )
    return criterion

def save_model(model, path):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model