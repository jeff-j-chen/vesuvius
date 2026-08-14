import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from utils.config import Config
from utils.dataloader import DataManager, MultiScrollIterableDataset, get_dataloaders
from utils.hard_mining import HardMiningInjector, HardMiningManager
from utils.model import create_model, supcon_loss
from utils.training_utils import (
    calculate_metrics,
    create_loss_function,
    create_optimizer_and_scheduler,
    save_model,
)
from utils.visualizer import TensorboardVisualizer


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """set the global RNG state for reproducible training."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            print(f"[seed] use_deterministic_algorithms unavailable: {exc}")
        print(f"[seed] DETERMINISTIC mode (seed={seed}) — exact reproducibility, slower")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"[seed] fast mode (seed={seed}) — cudnn benchmark on, tiny run-to-run fp noise")


class Trainer:
    """manage the current training and validation loop."""

    def __init__(self, config: Config):
        self.c = config
        set_seed(
            int(getattr(config.tra, "seed", 41)),
            deterministic=bool(getattr(config.tra, "deterministic", False)),
        )
        self._print_config()

        self.train_dataset, self.train_loader, self.valid_loader = self._setup_data()
        self.model, self.params, self.optimizer, self.scheduler, self.criterion = self._setup_model_optim()

        print("Initializing Tensorboard...")
        self._init_visualizers()
        self._dump_run_config()

        self.scaler = GradScaler(enabled=self.c.device == "cuda")
        self.hard_manager = HardMiningManager(self.c.hm.dir)
        self.hard_samples: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0

    def _print_config(self) -> None:
        print("--- Configuration ---")
        for field in self.c.__dataclass_fields__:
            print(f"{field}: {getattr(self.c, field)}")
        print("---------------------")

    def _setup_data(self):
        print("Creating datasets...")
        start_time = time.time()

        scroll_ids = [int(scroll.scroll_id) for scroll in self.c.data.scrolls]
        if bool(getattr(self.c.tra, "dann", False)) and int(getattr(self.c.tra, "dann_n_domains", 0)) <= 0:
            self.c.tra.dann_n_domains = len(scroll_ids)
        self._scroll_ids = scroll_ids
        self._scroll_train_sets = None

        if len(scroll_ids) > 1:
            train_sets = []
            valid_sets = []
            self._scroll_dms = {}
            self._scroll_train_sets = {}
            for domain_id, scroll_id in enumerate(scroll_ids):
                data_manager = DataManager(self.c, scroll_id=scroll_id, domain_id=domain_id)
                train_set, valid_set = data_manager.get_datasets()
                train_sets.append(train_set)
                valid_sets.append(valid_set)
                self._scroll_dms[scroll_id] = data_manager
                self._scroll_train_sets[scroll_id] = train_set
                print(
                    f"[multi-scroll] scroll {scroll_id}: "
                    f"train_tiles={len(train_set)} valid_tiles={len(valid_set)}"
                )

            merged_train = MultiScrollIterableDataset(train_sets)
            merged_valid = MultiScrollIterableDataset(valid_sets)
            train_loader, valid_loader = get_dataloaders(merged_train, merged_valid, self.c)
            print(
                f"[multi-scroll] merged train_tiles={len(merged_train)} "
                f"valid_tiles={len(merged_valid)}"
            )
            print(f"Data setup done in {time.time() - start_time:.2f}s")
            return merged_train, train_loader, valid_loader

        data_manager = DataManager(self.c, scroll_id=scroll_ids[0], domain_id=0)
        train_set, valid_set = data_manager.get_datasets()
        train_loader, valid_loader = get_dataloaders(train_set, valid_set, self.c)
        self._scroll_dms = {scroll_ids[0]: data_manager}
        print(f"Data setup done in {time.time() - start_time:.2f}s")
        return train_set, train_loader, valid_loader

    def _setup_model_optim(self):
        print(f"Creating model and loss... l1 lambda {self.c.tra.l1_lambda}... ", end="")
        start_time = time.time()

        model, params = create_model(self.c)
        init_path = getattr(self.c, "init_weights", None)
        if init_path:
            state_dict = torch.load(init_path, map_location=self.c.device)
            model_state = model.state_dict()
            compatible = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped = len(state_dict) - len(compatible)
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            print(
                f"[init-weights] loaded {len(compatible)}/{len(state_dict)} tensors from {init_path} "
                f"(shape-skipped={skipped} missing={len(missing)} unexpected={len(unexpected)})"
            )

        optimizer, scheduler = create_optimizer_and_scheduler(model, self.c)
        criterion = create_loss_function(None, self.c)
        print(f" done in {time.time() - start_time:.2f}s")
        return model, params, optimizer, scheduler, criterion

    def _init_visualizers(self) -> None:
        scroll_ids = self._scroll_ids
        tra = self.c.tra
        will_test = (tra.test_int <= tra.n_epochs) or bool(getattr(tra, "test_on_final", False))
        if not will_test:
            print(
                f"[test] test_int={tra.test_int} > n_epochs={tra.n_epochs} and "
                f"test_on_final={getattr(tra, 'test_on_final', False)} -> skipping test-frag load"
            )

        if len(scroll_ids) > 1:
            self.vis = TensorboardVisualizer(self.c, mode="metrics")
            self.scroll_vis = {}
            for index, scroll_id in enumerate(scroll_ids):
                self.scroll_vis[scroll_id] = TensorboardVisualizer(
                    self.c,
                    mode="train",
                    scroll_id=scroll_id,
                    shared_writer=self.vis.writer,
                    tag_prefix=f"s{scroll_id}/",
                    load_test_frags=(index == 0 and will_test),
                )
        else:
            self.vis = TensorboardVisualizer(self.c, load_test_frags=will_test)
            self.scroll_vis = None

    def _dump_run_config(self) -> None:
        import dataclasses
        import json

        try:
            run_dir = getattr(self.vis, "log_path", None) or getattr(self.vis.writer, "log_dir", None)
            if not run_dir:
                print("[config] no run dir resolved -- skipping config dump", flush=True)
                return
            config_data = dataclasses.asdict(self.c) if dataclasses.is_dataclass(self.c) else vars(self.c)
            text = json.dumps(config_data, indent=2, default=str, sort_keys=True)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
                handle.write(text)
            self.vis.writer.add_text("config", "```json\n" + text + "\n```", 0)
            print(f"[config] saved run config -> {os.path.join(run_dir, 'config.json')}", flush=True)
        except Exception as exc:
            print(f"[config] WARN could not dump run config: {exc}", flush=True)

    def _train_batch(self, images, labels, mask, domain_ids=None, epoch: int = 0):
        images = images.to(self.c.device)
        labels = labels.to(self.c.device).view(-1, 1)
        mask = (mask.to(self.c.device).view(images.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)
        if domain_ids is not None:
            domain_ids = domain_ids.to(self.c.device, non_blocking=True).view(-1)

        self.optimizer.zero_grad()
        with autocast(self.c.device, enabled=self.c.device == "cuda"):
            use_extras = hasattr(self.model, "forward_with_extras") and any([
                bool(getattr(self.c.tra, "supcon", False)),
                bool(getattr(self.c.tra, "dann", False)),
                bool(getattr(self.c.tra, "spill_reduction", False)),
                bool(getattr(self.c.tra, "spill_entropy", False)),
            ])
            if bool(getattr(self.c.tra, "dann_grl_anneal", False)):
                n_ep = float(getattr(self.c.tra, "n_epochs", 12))
                p = float(epoch) / max(1.0, n_ep)
                grl_scale = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            else:
                grl_scale = 1.0
            if use_extras:
                outputs, _, domain_logits, supcon_z = self.model.forward_with_extras(
                    images,
                    grl_scale=grl_scale,
                )
            else:
                outputs = self.model(images)
                domain_logits = None
                supcon_z = None

            if outputs.dim() == 4:
                outputs = outputs.flatten(1).max(dim=1, keepdim=True).values

            targets = labels.float()
            pos_smooth = float(getattr(self.c.tra, "label_smooth_pos", 0.0))
            neg_smooth = float(getattr(self.c.tra, "label_smooth_neg", 0.0))
            if pos_smooth > 0 or neg_smooth > 0:
                pos_mask = labels > 0.5
                targets = torch.where(
                    pos_mask,
                    torch.full_like(targets, 1.0 - pos_smooth),
                    torch.full_like(targets, neg_smooth),
                )

            raw_loss = self.criterion(outputs, targets) * mask
            denom = mask.sum()
            if denom <= 0:
                print("[ERROR] Mask sum is zero, skipping loss calculation.")
                return np.empty([]), np.empty([]), 0.0, 0.0, 0.0

            raw_loss_value = (raw_loss.sum() / denom).item()
            l1_loss = sum(param.abs().sum() for param in self.model.parameters())
            loss = (raw_loss.sum() / denom) + self.c.tra.l1_lambda * l1_loss

            tta_lambda = float(getattr(self.c.tra, "tta_consistency_lambda", 0.0))
            if getattr(self.c.tra, "tta_consistency", False) and tta_lambda > 0:
                mode = str(getattr(self.c.tra, "tta_consistency_mode", "flips")).lower()
                if mode == "dihedral":
                    transforms = (
                        lambda tensor: torch.flip(tensor, dims=[-1]),
                        lambda tensor: torch.flip(tensor, dims=[-2]),
                        lambda tensor: torch.flip(tensor, dims=[-1, -2]),
                        lambda tensor: torch.rot90(tensor, 1, dims=[-2, -1]),
                        lambda tensor: torch.rot90(tensor, -1, dims=[-2, -1]),
                    )
                    view = transforms[int(torch.randint(0, len(transforms), (1,)).item())](images)
                else:
                    dims = ([-1], [-2], [-1, -2])[int(torch.randint(0, 3, (1,)).item())]
                    view = torch.flip(images, dims=dims)
                other = self.model(view.contiguous())
                if other.dim() == 4:
                    other = other.flatten(1).max(dim=1, keepdim=True).values
                p1 = torch.sigmoid(outputs)
                p2 = torch.sigmoid(other)
                consistency = ((p2 - p1.detach()) ** 2) * mask
                loss = loss + tta_lambda * (consistency.sum() / denom.clamp(min=1))

            attn_entropy_loss = getattr(self.model, "last_attn_entropy_loss", None)
            if attn_entropy_loss is not None and attn_entropy_loss.item() != 0.0:
                loss = loss + attn_entropy_loss

            dann_loss_value = outputs.new_zeros(())
            if (
                bool(getattr(self.c.tra, "dann", False))
                and domain_ids is not None
                and domain_logits is not None
            ):
                dann_loss_value = F.cross_entropy(domain_logits, domain_ids)
                loss = loss + float(getattr(self.c.tra, "dann_lambda", 0.0)) * dann_loss_value

            spill_loss_value = outputs.new_zeros(())
            center_voxel_map = getattr(self.model, "last_center_voxel_map", None)
            if bool(getattr(self.c.tra, "spill_reduction", False)) and center_voxel_map is not None:
                pos_mask = (labels.view(-1) > 0.5).float()
                if pos_mask.sum() > 0:
                    # variance of mean logit per depth slice: high var = depth-selective (good)
                    # low var = uniform across all layers (spill); no cap on prediction confidence
                    depth_logits = center_voxel_map.mean(dim=(3, 4)).squeeze(1)  # [B, D]
                    depth_var = depth_logits.var(dim=1, unbiased=False)           # [B]
                    min_var = float(getattr(self.c.tra, "spill_min_depth_var", 0.5))
                    spill_loss_value = (F.relu(min_var - depth_var) * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
                    loss = loss + float(getattr(self.c.tra, "spill_lambda", 0.0)) * spill_loss_value

            if bool(getattr(self.c.tra, "spill_entropy", False)):
                full_voxel_map = getattr(self.model, "last_voxel_map_full", None)
                if full_voxel_map is not None:
                    pos_mask_e = (labels.view(-1) > 0.5).float()
                    if pos_mask_e.sum() > 0:
                        # softmax entropy of full-context depth profile: scale-invariant depth sparsity
                        # uses full spatial extent (not just center) for a robust depth estimate
                        full_depth = full_voxel_map.mean(dim=(3, 4)).squeeze(1)  # [B, D]
                        depth_attn = F.softmax(full_depth, dim=1)
                        entropy = -(depth_attn * depth_attn.log()).sum(dim=1)    # [B]
                        max_ent = float(getattr(self.c.tra, "spill_max_depth_entropy", 2.1))
                        ent_spill = (F.relu(entropy - max_ent) * pos_mask_e).sum() / pos_mask_e.sum().clamp(min=1.0)
                        spill_loss_value = spill_loss_value + ent_spill
                        loss = loss + float(getattr(self.c.tra, "spill_entropy_lambda", 0.3)) * ent_spill

            if getattr(self.c.tra, "supcon", False) and supcon_z is not None:
                if getattr(self.c.tra, "supcon_curriculum", False):
                    curriculum_epochs = int(getattr(self.c.tra, "supcon_curriculum_epochs", 15))
                    lambda_start = float(getattr(self.c.tra, "supcon_lambda_start", 0.1))
                    lambda_end = float(getattr(self.c.tra, "supcon_lambda_end", 0.5))
                    progress = min(1.0, epoch / max(1, curriculum_epochs))
                    supcon_lambda = lambda_start + (lambda_end - lambda_start) * progress
                else:
                    supcon_lambda = float(getattr(self.c.tra, "supcon_lambda", 0.1))
                supcon_temp = float(getattr(self.c.tra, "supcon_temp", 0.07))
                loss = loss + supcon_lambda * supcon_loss(supcon_z, labels.view(-1).long(), temp=supcon_temp)

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        scores = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        label_values = labels.detach().cpu().numpy().flatten().astype(int)
        return (
            scores,
            label_values,
            loss.item(),
            raw_loss_value,
            float(dann_loss_value.detach().item()),
            float(spill_loss_value.detach().item()),
        )

    def _ins_hard_samples(self, images, labels, mask, hard_injector, remaining_batches: int) -> int:
        if not hard_injector or not hard_injector.has_next():
            return 0

        if remaining_batches <= 0:
            inject_n = 0
        else:
            inject_n = min(images.size(0), (hard_injector.remaining() + remaining_batches - 1) // remaining_batches)

        injected = 0
        if inject_n > 0:
            replace_indices = np.random.choice(images.size(0), inject_n, replace=False)
            for replace_index in replace_indices:
                sample = None
                while hard_injector.has_next() and sample is None:
                    sample = hard_injector.next_sample()
                if sample is None:
                    break
                hard_block, hard_label, hard_mask = sample
                images[replace_index] = hard_block.to(self.c.device)
                labels[replace_index] = hard_label.to(self.c.device)
                mask[replace_index] = hard_mask.to(self.c.device)
                injected += 1
        return injected

    def train_epoch(self, hard_injector, epoch: int = 0):
        self.model.train()
        loss_total = 0.0
        raw_loss_total = 0.0
        dann_loss_total = 0.0
        spill_loss_total = 0.0
        labels = []
        preds = []
        scores = []
        total_injected = 0

        for batch_index, batch in enumerate(
            tqdm(self.train_loader, desc="Training", mininterval=5, miniters=1, file=sys.stderr)
        ):
            images, batch_labels, mask = batch[:3]
            domain_ids = batch[3] if len(batch) > 3 else None
            total_injected += self._ins_hard_samples(
                images,
                batch_labels,
                mask,
                hard_injector,
                len(self.train_loader) - batch_index,
            )
            (
                batch_scores,
                batch_labels_out,
                batch_loss,
                batch_raw_loss,
                batch_dann_loss,
                batch_spill_loss,
            ) = self._train_batch(
                images,
                batch_labels,
                mask,
                domain_ids=domain_ids,
                epoch=epoch,
            )
            if batch_scores.size == 0:
                continue
            loss_total += batch_loss
            raw_loss_total += batch_raw_loss
            dann_loss_total += batch_dann_loss
            spill_loss_total += batch_spill_loss
            labels.extend(batch_labels_out)
            preds.extend((batch_scores > 0.5).astype(int))
            scores.extend(batch_scores)

        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        metrics["loss"] = loss_total / len(self.train_loader)
        metrics["raw_loss"] = raw_loss_total / len(self.train_loader)
        metrics["dann_loss"] = dann_loss_total / len(self.train_loader)
        metrics["spill_loss"] = spill_loss_total / len(self.train_loader)
        metrics["scores"] = scores
        metrics["hard_injected"] = total_injected

        if hard_injector:
            stats = hard_injector.stats()
            print(
                f"[HARD][Epoch Summary] injected={total_injected} "
                f"injector_used={stats['used']} injector_skipped={stats['skipped']}"
            )
        return metrics

    def validate_epoch(self):
        self.model.eval()
        loss_total = 0.0
        labels = []
        preds = []
        scores = []

        with torch.no_grad(), autocast(self.c.device, enabled=self.c.device == "cuda"):
            for batch in tqdm(
                self.valid_loader,
                desc="Validating",
                mininterval=5,
                miniters=1,
                file=sys.stderr,
            ):
                images, batch_labels, mask = batch[:3]
                if mask.view(mask.size(0), -1).sum() <= 0:
                    print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")
                    continue

                images = images.to(self.c.device, non_blocking=True)
                batch_labels = batch_labels.to(self.c.device, non_blocking=True).view(-1, 1)
                mask = (mask.to(self.c.device).view(images.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)

                outputs = self.model(images)
                if outputs.dim() == 4:
                    outputs = outputs.flatten(1).max(dim=1, keepdim=True).values

                raw_loss = self.criterion(outputs, batch_labels)
                loss_total += ((raw_loss * mask).sum() / mask.sum()).item()

                batch_scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                labels.extend(batch_labels.cpu().numpy().flatten().astype(int))
                preds.extend((batch_scores > 0.5).astype(int))
                scores.extend(batch_scores)

        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        metrics["loss"] = loss_total / len(self.valid_loader)
        metrics["scores"] = scores
        return metrics

    def _periodic_model_save(self, epoch: int, val_metrics: dict) -> None:
        if val_metrics["f1"] > self.best_val_f1:
            self.best_val_f1 = val_metrics["f1"]
            save_model(self.model, f"{self.c.model_dir}/best_model_f1.pth")
            print(f"New best F1 model saved! Val F1: {self.best_val_f1:.4f}")

        if val_metrics["loss"] < self.best_val_loss:
            self.best_val_loss = val_metrics["loss"]
            save_model(self.model, f"{self.c.model_dir}/best_model_loss.pth")
            print(f"New best loss model saved! Val Loss: {self.best_val_loss:.4f}")

        if (epoch + 1) % self.c.tra.save_int == 0:
            save_model(self.model, f"{self.c.model_dir}/model_epoch_{epoch + 1}.pth")

    def _update_hard_mining_samples(self, epoch: int) -> None:
        if not self.c.hm.enabled:
            return
        if epoch % self.c.tra.eval_int != 0 or epoch <= 5:
            return

        target_hard = int(self.c.hm.hm_frac * len(self.train_dataset))
        new_samples = self.hard_manager.sample_for_epoch_scrolls(epoch - 1, target_hard, self._scroll_ids)
        if new_samples:
            self.hard_samples.extend(new_samples)
            print(
                f"[HARD][Epoch {epoch}] Added {len(new_samples)} new hard samples. "
                f"Total is now {len(self.hard_samples)}."
            )
            self.vis.writer.add_scalar("HardMining/TotalSamplesInPool", len(self.hard_samples), epoch)
        else:
            print(f"[HARD][Epoch {epoch}] Mining file processed but no new samples were added.")

    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, time_elapsed: float) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"[METRICS] epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f}"
        )

        self.vis.log_epoch_metrics(
            epoch,
            self.model,
            train_metrics,
            val_metrics,
            current_lr,
            time_elapsed,
            self.params,
            None,
        )
        if self.c.hm.enabled:
            self.vis.writer.add_scalar("HardMining/Injected", train_metrics.get("hard_injected", 0), epoch)

        if not self.scroll_vis:
            return

        eval_due = (epoch + 1) % self.c.tra.eval_int == 0
        test_due = (epoch + 1) % self.c.tra.test_int == 0
        probe_due = (epoch + 1) % self.c.tra.probe_int == 0
        if getattr(self.c.tra, "test_on_final", False) and (epoch + 1) == self.c.tra.n_epochs:
            test_due = True

        max_eval_scrolls = getattr(self.c.tra, "eval_int_scrolls", 2)
        eval_rendered = 0
        for index, (scroll_id, visualizer) in enumerate(self.scroll_vis.items()):
            if eval_due and getattr(visualizer, "eval_enabled", True) and eval_rendered < max_eval_scrolls:
                try:
                    visualizer.add_evaluation_figures(epoch, self.model)
                    eval_rendered += 1
                except Exception as exc:
                    print(f"[ERROR] eval figures failed for scroll {scroll_id}: {exc}")
                    import traceback

                    traceback.print_exc()
            if test_due and index == 0:
                try:
                    visualizer.add_test_figures(epoch, self.model)
                except Exception as exc:
                    print(f"[ERROR] test figures failed for scroll {scroll_id}: {exc}")
            if probe_due and index == 0:
                try:
                    visualizer.add_probe_region_figures(epoch, self.model)
                except Exception as exc:
                    print(f"[ERROR] probe figures failed for scroll {scroll_id}: {exc}")
        for visualizer in self.scroll_vis.values():
            visualizer.writer.flush()

    def run(self) -> None:
        self._hm_active_this_epoch = self.c.hm.enabled
        for epoch in range(self.c.tra.n_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.c.tra.n_epochs} ---")
            start_time = time.time()

            self.train_dataset.apply_transforms = bool(epoch >= 5 and self.c.dl.data_aug)
            self._update_hard_mining_samples(epoch)

            hard_injector = None
            if self.hard_samples and self._hm_active_this_epoch:
                if self._scroll_train_sets:
                    dataset_map = self._scroll_train_sets
                else:
                    dataset_map = {int(self._scroll_ids[0]): self.train_dataset}
                hard_injector = HardMiningInjector(self.hard_samples, dataset_map)
                if epoch % self.c.tra.eval_int == 0:
                    self.vis.writer.add_scalar("HardMining/InjectedSamplesPlanned", len(self.hard_samples), epoch)

            train_metrics = self.train_epoch(hard_injector, epoch=epoch)

            val_cooldown = int(getattr(self.c.tra, "val_cooldown_secs", 0))
            if val_cooldown > 0:
                print(f"[COOLDOWN] train->val pause {val_cooldown}s...")
                time.sleep(val_cooldown)

            val_metrics = self.validate_epoch()
            self.scheduler.step(val_metrics["loss"])
            self._periodic_model_save(epoch, val_metrics)
            self._log_epoch(epoch, train_metrics, val_metrics, time.time() - start_time)

            eval_cooldown = int(getattr(self.c.tra, "eval_cooldown_secs", 0))
            is_probe_epoch = (epoch + 1) % self.c.tra.probe_int == 0
            is_eval_epoch = (epoch + 1) % self.c.tra.eval_int == 0
            if eval_cooldown > 0 and (is_probe_epoch or is_eval_epoch):
                kind = "eval+probe" if is_eval_epoch and is_probe_epoch else ("eval" if is_eval_epoch else "probe")
                print(f"[COOLDOWN] {kind} epoch — sleeping {eval_cooldown}s for hardware to cool...")
                time.sleep(eval_cooldown)

            epoch_cooldown = int(getattr(self.c.tra, "epoch_cooldown_secs", 0))
            if epoch_cooldown > 0 and not (eval_cooldown > 0 and (is_probe_epoch or is_eval_epoch)):
                print(f"[COOLDOWN] end-of-epoch pause {epoch_cooldown}s for hardware to cool...")
                time.sleep(epoch_cooldown)

        final_path = getattr(self.c, "save_final", None)
        if final_path:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            save_model(self.model, final_path)
            print(f"[save-final] wrote final model to {final_path}")

        self.vis.close()
        if self.scroll_vis:
            for visualizer in self.scroll_vis.values():
                visualizer.close()
        print("Training completed.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Vesuvius ink-detection training")
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        default="",
        help="experiment name (used for TensorBoard log dir and checkpoint naming)",
    )
    args = parser.parse_args()

    config = Config()
    if args.experiment_name:
        config.exp_name = args.experiment_name

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config.tra.log_dir):
        config.tra.log_dir = os.path.normpath(os.path.join(repo_root, config.tra.log_dir))
    if not os.path.isabs(config.model_dir):
        config.model_dir = os.path.normpath(os.path.join(repo_root, config.model_dir))

    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()