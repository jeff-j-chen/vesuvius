import os

# keep tensorboard/tensorflow startup noise low and deterministic across windows workers
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from utils.visualizer import TensorboardVisualizer
from utils.hard_mining import HardMiningManager, HardMiningInjector
from utils.dataloader import DataManager, get_dataloaders, calc_class_wgts, calc_dense_pos_weight, MultiScrollIterableDataset
from utils.model import create_model
from utils.training_utils import (
    create_optimizer_and_scheduler, 
    create_loss_function,
    calculate_metrics,
    save_model,
    pairwise_ranking_loss,
)

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from tqdm import tqdm
import sys
import time
import random


def set_seed(seed=42, deterministic=False):
    """sets the seed for reproducibility across all relevant libraries.

    deterministic=False (default, FAST): cuDNN profiles+caches the fastest conv algo
      (benchmark=True). conv/pool backward use non-deterministic GPU atomics, so two
      runs with the SAME seed differ very slightly (fp noise that compounds over epochs).
      this is the speed/memory path.
    deterministic=True (EXACT): forces deterministic cuDNN algos, disables benchmark, and
      requires CUBLAS_WORKSPACE_CONFIG for deterministic cuBLAS matmuls. two runs with the
      same seed are then bit-for-bit identical, at ~10-20% speed cost. warn_only=True so an
      op without a deterministic kernel warns instead of hard-crashing mid-run.
    """
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
        except Exception as e:
            print(f"[seed] use_deterministic_algorithms unavailable: {e}")
        print(f"[seed] DETERMINISTIC mode (seed={seed}) — exact reproducibility, slower")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"[seed] fast mode (seed={seed}) — cudnn benchmark on, tiny run-to-run fp noise")

class Trainer:
    """manages the model training and validation process"""
    def __init__(self, config):
        """initializes the trainer, setting up data, model, and logging"""
        self.c = config
        set_seed(int(getattr(config.tra, "seed", 41)),
                 deterministic=bool(getattr(config.tra, "deterministic", False)))
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
        scroll_ids = self._scroll_ids
        # will a test/holdout figure ever fire? periodic tests need test_int <= n_epochs;
        # the one-shot final render needs test_on_final. campaigns set test_int > n_epochs
        # (e.g. 9999) and no final, so skip opening every test/holdout zarr for nothing.
        tra = self.c.tra
        will_test = (tra.test_int <= tra.n_epochs) or bool(getattr(tra, "test_on_final", False))
        if not will_test:
            print(f"[test] test_int={tra.test_int} > n_epochs={tra.n_epochs} and "
                  f"test_on_final={getattr(tra, 'test_on_final', False)} -> skipping test-frag load")
        if len(scroll_ids) > 1:
            # merged training stream: the main visualizer owns the single tensorboard
            # run folder and logs scalar metrics; one figure-visualizer per scroll
            # renders its own eval/test figures into that SAME folder, namespacing its
            # tags with s<sid>/. this keeps the run list at one folder regardless of
            # scroll count (tag '/' is UI grouping only, not a folder on disk). probe
            # ROIs stay global (rendered once, unprefixed).
            self.vis = TensorboardVisualizer(self.c, mode='metrics')
            self.scroll_vis = {}
            for i, sid in enumerate(scroll_ids):
                self.scroll_vis[sid] = TensorboardVisualizer(
                    self.c, mode='train', scroll_id=sid,
                    shared_writer=self.vis.writer, tag_prefix=f"s{sid}/",
                    # only primary loads the large test-frag assets, and only if a test can fire
                    load_test_frags=(i == 0 and will_test),
                )
        else:
            self.vis = TensorboardVisualizer(self.c, load_test_frags=will_test)
            self.scroll_vis = None
        
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

        scroll_ids = [s.scroll_id for s in self.c.data.scrolls]
        self._scroll_ids = scroll_ids
        self._scroll_train_sets = None   # {scroll_id: train_set} in multiscroll, for HM routing
        multi = len(scroll_ids) > 1

        if multi:

            ring_mode = getattr(self.c.data, 'ring_negatives', False)
            train_sets, valid_sets = [], []
            self._scroll_dms = {}
            self._scroll_train_sets = {}
            for sid in scroll_ids:
                dm = DataManager(self.c, scroll_id=sid)
                t_set, v_set = dm.get_datasets()
                train_sets.append(t_set)
                valid_sets.append(v_set)
                self._scroll_dms[sid] = dm
                self._scroll_train_sets[int(sid)] = t_set
                print(f"[multi-scroll] scroll {sid}: train_tiles={len(t_set)} valid_tiles={len(v_set)}")

            t_set_merged = MultiScrollIterableDataset(train_sets)
            v_set_merged = MultiScrollIterableDataset(valid_sets)
            t_loader, v_loader = get_dataloaders(t_set_merged, v_set_merged, self.c)
            if getattr(self.c.data, "dense_labels", False):
                pos_weight = calc_dense_pos_weight(t_set_merged)
            else:
                pos_weight = calc_class_wgts(t_set_merged, v_set_merged, scroll_id=None)
            self._t_set_full = t_set_merged
            self._t_set_ring = None
            print(f"[multi-scroll] merged train_tiles={len(t_set_merged)} valid_tiles={len(v_set_merged)}")
            print(f"Data setup done in {time.time() - start_time:.2f}s")
            return t_set_merged, t_loader, v_loader, pos_weight

        data_manager = DataManager(self.c, scroll_id=scroll_ids[0])
        self._scroll_dms = {scroll_ids[0]: data_manager}

        t_set_full, v_set = data_manager.get_datasets()
        t_loader, v_loader = get_dataloaders(t_set_full, v_set, self.c)
        if getattr(self.c.data, "dense_labels", False):
            pos_weight = calc_dense_pos_weight(t_set_full)
        else:
            pos_weight = calc_class_wgts(t_set_full, v_set, scroll_id=None)
        self._t_set_full = t_set_full
        self._t_set_ring = None

        print(f"Data setup done in {time.time() - start_time:.2f}s")
        return t_set_full, t_loader, v_loader, pos_weight

    def _setup_model_optim(self):
        """creates the model, optimizer, scheduler, and loss function"""
        print(f"Creating model and loss... l1 lambda {self.c.tra.l1_lambda}... ", end="")
        start_time = time.time()
        
        model, params = create_model(self.c)
        # optional warm-start: load MAE (or any) pretrained weights, ignoring
        # non-matching keys (e.g. the MAE 'recon' head vs the ink 'head').
        init_path = getattr(self.c, "init_weights", None)
        if init_path:
            sd = torch.load(init_path, map_location=self.c.device)
            # keep only keys that exist in the model AND share the same shape.
            # strict=False skips missing/unexpected keys but still errors on shape
            # mismatch, so filter those out for partial cross-arch transfer (e.g.
            # dense_unet MAE -> dense_unet_depth: per_slice matches, e1+ differ).
            model_sd = model.state_dict()
            compat = {k: v for k, v in sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
            skipped = len(sd) - len(compat)
            missing, unexpected = model.load_state_dict(compat, strict=False)
            print(f"[init-weights] loaded {len(compat)}/{len(sd)} tensors from {init_path} "
                  f"(shape-skipped={skipped} missing={len(missing)} unexpected={len(unexpected)})")
        optimizer, scheduler = create_optimizer_and_scheduler(model, self.c)
        # pos_weight (neg/pos) upweights positives -> biases outputs toward 1.0. optionally drop it.
        pw = self.pos_weight if getattr(self.c.tra, "pos_weight_enabled", True) else None
        if pw is None and self.pos_weight is not None:
            print("[loss] pos_weight DISABLED (pos_weight_enabled=False) -- using unweighted loss")
        criterion = create_loss_function(pw, self.c)
        
        print(f" done in {time.time() - start_time:.2f}s")
        return model, params, optimizer, scheduler, criterion

    def _train_batch(self, b_imgs, b_labels, mask):
        """trains the model on a single batch of data"""
        if getattr(self.c.data, "dense_labels", False):
            return self._train_batch_dense(b_imgs, b_labels, mask)
        b_imgs = b_imgs.to(self.c.device)
        b_labels = b_labels.to(self.c.device).view(-1, 1)
        # mask is (B, 32, 32) per-pixel; collapse to per-tile (B, 1): tile is valid if any pixel is valid
        mask = (mask.to(self.c.device).view(b_imgs.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)

        self.optimizer.zero_grad()

        # forward pass with automatic mixed precision
        with autocast(self.c.device):
            outputs = self.model(b_imgs)
            # per-voxel models return (B,1,H,W) heatmap; apply MIL max for tile-level BCE
            if outputs.dim() == 4:
                outputs = outputs.flatten(1).max(dim=1, keepdim=True).values
            # loss target: optional label smoothing (hard b_labels kept for metrics/ranking).
            # softening 0/1 -> ls/(1-ls) stops the model driving logits to +/-inf on possibly
            # mislabeled tiles -- a mild, targeted counter to memorizing label noise.
            target = b_labels.float()
            ls = float(getattr(self.c.tra, "label_smooth", 0.0))
            if ls > 0:
                target = target * (1.0 - 2.0 * ls) + ls
            raw_loss = self.criterion(outputs, target)
            
            # apply mask to zero out loss in irrelevant regions
            raw_loss = raw_loss * mask
            if mask.sum() <= 0:
                print("[ERROR] Mask sum is zero, skipping loss calculation.")
                return np.empty([]), np.empty([]), 0.0, 0.0
            
            # normalize loss and add l1 regularization
            raw_loss_val = (raw_loss.sum() / mask.sum()).item()
            l1_loss = sum(p.abs().sum() for p in self.model.parameters())
            loss = (raw_loss.sum() / mask.sum()) + self.c.tra.l1_lambda * l1_loss

            # optional pairwise-ranking term (AUC / partial-AUC surrogate). added on top of
            # BCE to directly optimize tile ordering -- the objective PR-AUC actually rewards.
            # well-matched to balanced ring data (unlike focal, which targets imbalance).
            ranking_lambda = float(getattr(self.c.tra, "ranking_lambda", 0.0))
            if ranking_lambda > 0:
                probs = torch.sigmoid(outputs).reshape(-1)
                rloss = pairwise_ranking_loss(
                    probs, b_labels.reshape(-1),
                    margin=float(getattr(self.c.tra, "ranking_margin", 0.3)),
                    neg_frac=float(getattr(self.c.tra, "ranking_neg_frac", 1.0)),
                )
                loss = loss + ranking_lambda * rloss

            # optional TTA-consistency regularizer: an extra FLIPPED view of the batch should
            # yield the SAME tile score (ink-present is flip-invariant). penalizing the two
            # predictions' disagreement makes the model invariant -> fewer holdout hallucinations
            # than augmentation alone (which never forces two views to AGREE). ~1 extra forward.
            tta_cons_lambda = float(getattr(self.c.tra, "tta_consistency_lambda", 0.0))
            if getattr(self.c.tra, "tta_consistency", False) and tta_cons_lambda > 0:
                # random flip among h / v / 180 over the spatial dims (last two axes)
                fd = ([-1], [-2], [-1, -2])[int(torch.randint(0, 3, (1,)).item())]
                out2 = self.model(torch.flip(b_imgs, dims=fd))
                if out2.dim() == 4:
                    out2 = out2.flatten(1).max(dim=1, keepdim=True).values
                p1 = torch.sigmoid(outputs)
                p2 = torch.sigmoid(out2)
                # stop-grad on the clean view (teacher); pull the flipped view toward it
                cons = ((p2 - p1.detach()) ** 2) * mask
                cons = cons.sum() / mask.sum().clamp(min=1)
                loss = loss + tta_cons_lambda * cons

        # backward pass and optimization step
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # calculate scores and labels for metrics
        scores = torch.sigmoid(outputs).cpu().detach().numpy().flatten()
        labels = b_labels.cpu().detach().numpy().flatten().astype(int)
        
        return scores, labels, loss.item(), raw_loss_val

    @staticmethod
    def _dense_pixel_sample(scores_map, label_map, pmask, max_px=4096):
        """flatten valid (mask>0) pixels and subsample for epoch-level metric accumulation.
        keeps per-pixel metric memory bounded across an epoch (a full tile is ~16k px)."""
        sel = pmask.reshape(-1) > 0
        s = scores_map.reshape(-1)[sel]
        l = (label_map.reshape(-1)[sel] > 0.5).astype(np.int64)
        if s.shape[0] > max_px:
            idx = np.random.default_rng(0).choice(s.shape[0], max_px, replace=False)
            s, l = s[idx], l[idx]
        return s, l

    def _train_batch_dense(self, b_imgs, b_labels, mask):
        """dense per-pixel training step: per-pixel masked BCE against the (B,1,T,T) label map.

        this is the non-binary path — the model returns a (B,1,H,W) logit map and every
        interior (mask>0) pixel contributes a BCE term. no MIL max-collapse, no tile scalar."""
        device = self.c.device
        b_imgs = b_imgs.to(device)
        b_labels = b_labels.to(device).float()                    # (B,1,T,T)
        pmask = (mask.to(device) > 0).float().unsqueeze(1)        # (B,1,T,T)

        self.optimizer.zero_grad()
        with autocast(device):
            outputs = self.model(b_imgs)                          # (B,1,H,W)
            raw = self.criterion(outputs, b_labels)              # per-pixel (reduction='none')
            denom = pmask.sum()
            if denom <= 0:
                return np.empty([]), np.empty([]), 0.0, 0.0
            raw_loss_val = ((raw * pmask).sum() / denom).item()
            l1_loss = sum(p.abs().sum() for p in self.model.parameters())
            loss = (raw * pmask).sum() / denom + self.c.tra.l1_lambda * l1_loss

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        scores_map = torch.sigmoid(outputs).detach().float().cpu().numpy()
        s, l = self._dense_pixel_sample(scores_map, b_labels.cpu().numpy(),
                                        pmask.cpu().numpy())
        return s, l, loss.item(), raw_loss_val

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

        for batch_idx, (b_imgs, b_labels, mask) in enumerate(tqdm(self.train_loader, desc="Training", mininterval=5, miniters=1, file=sys.stderr)):
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

        if getattr(self.c.data, "dense_labels", False):
            return self._validate_epoch_dense()

        with torch.no_grad(), autocast(self.c.device):
            for b_imgs, b_labels, mask in tqdm(self.valid_loader, desc="Validating", mininterval=5, miniters=1, file=sys.stderr):
                if mask.view(mask.size(0), -1).sum() <= 0:
                    print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")
                    continue

                b_imgs = b_imgs.to(self.c.device)
                b_labels = b_labels.to(self.c.device).view(-1, 1)
                # collapse per-pixel mask to per-tile (B, 1)
                mask = (mask.to(self.c.device).view(b_imgs.size(0), -1).sum(dim=1) > 0).float().unsqueeze(1)

                # forward pass
                outputs = self.model(b_imgs)
                if outputs.dim() == 4:
                    outputs = outputs.flatten(1).max(dim=1, keepdim=True).values
                
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

    def _validate_epoch_dense(self):
        """dense per-pixel validation: per-pixel masked BCE + subsampled per-pixel metrics."""
        device = self.c.device
        loss, nb = 0.0, 0
        labels, preds, scores = [], [], []
        with torch.no_grad(), autocast(device):
            for b_imgs, b_labels, mask in tqdm(self.valid_loader, desc="Validating", mininterval=5, miniters=1, file=sys.stderr):
                pmask = (mask.to(device) > 0).float().unsqueeze(1)     # (B,1,T,T)
                if pmask.sum() <= 0:
                    continue
                b_imgs = b_imgs.to(device)
                b_labels = b_labels.to(device).float()                # (B,1,T,T)
                outputs = self.model(b_imgs)                          # (B,1,H,W)
                raw = self.criterion(outputs, b_labels)
                loss += ((raw * pmask).sum() / pmask.sum()).item(); nb += 1
                s_map = torch.sigmoid(outputs).float().cpu().numpy()
                s, l = self._dense_pixel_sample(s_map, b_labels.cpu().numpy(),
                                                pmask.cpu().numpy())
                labels.extend(l.tolist())
                preds.extend((s > 0.5).astype(int).tolist())
                scores.extend(s.tolist())
        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        metrics['loss'] = loss / max(nb, 1)
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
            
            # load new samples from the previous epoch's per-scroll mining files.
            # works for one scroll (list len 1) or many; records are tagged with
            # scroll_id so the injector routes each to the right volume.
            new_samples = self.hard_manager.sample_for_epoch_scrolls(
                epoch - 1, target_hard, self._scroll_ids
            )
            
            if new_samples:
                self.hard_samples.extend(new_samples)
                print(f"[HARD][Epoch {epoch}] Added {len(new_samples)} new hard samples. Total is now {len(self.hard_samples)}.")
                self.vis.writer.add_scalar("HardMining/TotalSamplesInPool", len(self.hard_samples), epoch)
            else:
                print(f"[HARD][Epoch {epoch}] Mining file processed but no new samples were added.")

    def _log_epoch(self, epoch, train_metrics, val_metrics, time_elapsed):
        """logs metrics for the completed epoch to tensorboard"""
        current_lr = self.optimizer.param_groups[0]['lr']

        # parseable one-liner for external monitors (campaign early-stop wrapper reads this)
        print(f"[METRICS] epoch={epoch+1} train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f}")

        self.vis.log_epoch_metrics(epoch, self.model, train_metrics, val_metrics, current_lr, time_elapsed, self.params, self.pos_weight)
        if self.c.hm.enabled:
            self.vis.writer.add_scalar("HardMining/Injected", train_metrics.get('hard_injected', 0), epoch)

        # multi-scroll: the main visualizer logs scalars only; drive per-scroll
        # eval/test/probe figures here so each fragment gets its own visualizations
        if getattr(self, 'scroll_vis', None):
            eval_due  = (epoch + 1) % self.c.tra.eval_int  == 0
            test_due  = (epoch + 1) % self.c.tra.test_int  == 0
            probe_due = (epoch + 1) % self.c.tra.probe_int == 0
            # leave-one-out campaigns disable periodic tests (test_int=9999) but still want
            # the held-out fragment inferred once at the end -> force test on the final epoch.
            if getattr(self.c.tra, "test_on_final", False) and (epoch + 1) == self.c.tra.n_epochs:
                test_due = True
            dense = getattr(self.c.data, 'dense_labels', False)
            # eval figures are the slow part -> render only the first N scrolls (probes/test unaffected)
            max_eval_scrolls = getattr(self.c.tra, "eval_int_scrolls", 2)
            for idx, (sid, svis) in enumerate(self.scroll_vis.items()):
                if eval_due and idx < max_eval_scrolls and getattr(svis, 'eval_enabled', True):
                    try:
                        # dense models produce per-pixel maps -> dense figure; else tile figure.
                        # each per-scroll visualizer loaded its own volume/labels, so the two
                        # scrolls render as two separate figures namespaced by s<sid>/.
                        if dense:
                            svis.add_dense_evaluation_figure(epoch, self.model)
                        else:
                            svis.add_evaluation_figures(epoch, self.model)
                    except Exception as e:
                        print(f"[ERROR] eval figures failed for scroll {sid}: {e}")
                        import traceback; traceback.print_exc()
                if test_due and idx == 0:
                    try:
                        svis.add_test_figures(epoch, self.model)
                    except Exception as e:
                        print(f"[ERROR] test figures failed for scroll {sid}: {e}")
                # probe ROIs are fixed (scroll-independent); render once on the primary
                if probe_due and idx == 0:
                    try:
                        svis.add_probe_region_figures(epoch, self.model)
                    except Exception as e:
                        print(f"[ERROR] probe figures failed for scroll {sid}: {e}")
            for svis in self.scroll_vis.values():
                svis.writer.flush()

    def _pretrain_epoch(self):
        """one epoch of band-identity pretraining (self-supervised, no ink labels needed).

        the model learns to distinguish ink_band tiles (label=1) from flanking band tiles (label=0).
        because ink is the primary feature distinguishing these bands in ink regions, the
        backbone builds a representation that encodes differential absorption — useful for BCE fine-tuning.
        """
        import torch.nn.functional as F
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        tile = self.c.data.tile_size
        depth = self.c.data.depth
        device = self.c.device
        pre_z  = getattr(self.c.data, "pre_band_start", 20)
        post_z = getattr(self.c.data, "post_band_start", 40)
        ink_z  = getattr(self.c.data, "train_d_start", 32)
        criterion = nn.BCEWithLogitsLoss()

        for batch_idx, (b_imgs, b_labels, mask) in enumerate(tqdm(self.train_loader, desc="Pretrain", mininterval=5, miniters=1, file=sys.stderr)):
            # b_imgs is the normal ink-band batch; we also sample flanking batches
            # build: half ink-band (label=1), half flanking (label=0)
            bs = b_imgs.size(0)
            vol = self.train_dataset.vol
            y_start = self.train_dataset.y_start
            x_start = self.train_dataset.x_start

            # collect flanking band blocks for the same (y,x) positions
            flanking_blocks = []
            for i in range(bs):
                # pick pre or post band
                fz = pre_z if (i % 2 == 0) else post_z
                # recover approximate (y, x) from batch coords — just use random valid coords
                idx = self.train_dataset.block_coords[
                    np.random.randint(len(self.train_dataset.block_coords))
                ]
                _, y_off, x_off = idx
                y = y_start + y_off
                x = x_start + x_off
                blk = np.array(vol[fz:fz+depth, y:y+tile, x:x+tile]).astype(np.float32)
                norm = self.train_dataset._normalize_block(blk)
                flanking_blocks.append(norm)

            flanking = torch.from_numpy(np.stack(flanking_blocks)).float().unsqueeze(1).to(device)
            ink_imgs = b_imgs.to(device)

            combined = torch.cat([ink_imgs, flanking], dim=0)
            band_labels = torch.cat([
                torch.ones(bs, 1, device=device),
                torch.zeros(bs, 1, device=device)
            ], dim=0)

            self.optimizer.zero_grad()
            with autocast(device):
                logits = self.model(combined)
                loss = criterion(logits, band_labels)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _switch_to_epoch_dataset(self, epoch):
        """for alternating_ring mode: swap the training loader between full and ring sets.
        odd epochs use ring (closer to boundary, harder); even epochs use full set.
        hard mining is only enabled on ring epochs."""
        alternating = getattr(self.c.data, 'alternating_ring', False)
        if not alternating or self._t_set_ring is None:
            return
        use_ring = (epoch % 2 == 1)  # odd epochs → ring
        t_set = self._t_set_ring if use_ring else self._t_set_full
        self.train_dataset = t_set
        self.train_loader, _ = get_dataloaders(t_set, t_set, self.c)  # v_set unused here
        # suppress hard mining on full-set epochs to avoid off-boundary injections
        self._hm_active_this_epoch = use_ring and self.c.hm.enabled
        label = 'RING' if use_ring else 'FULL'
        print(f"[alternating_ring] epoch {epoch+1}: {label} set ({len(t_set)} tiles)")

    def run(self):
        """executes the main training loop, with optional pretraining phase"""
        pretrain_epochs = int(getattr(self.c.tra, "pretrain_epochs", 0))
        if pretrain_epochs > 0:
            print(f"\n=== PRETRAINING PHASE ({pretrain_epochs} epochs) ===")
            for ep in range(pretrain_epochs):
                pt_loss = self._pretrain_epoch()
                print(f"  Pretrain epoch {ep+1}/{pretrain_epochs}  loss={pt_loss:.4f}")
                self.vis.writer.add_scalar("Pretrain/Loss", pt_loss, ep)
            print("=== PRETRAINING COMPLETE — switching to BCE fine-tuning ===\n")
            # reset optimizer/scheduler for the fine-tuning phase
            self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.model, self.c)
        self._hm_active_this_epoch = self.c.hm.enabled  # default; overridden by alternating
        for epoch in range(self.c.tra.n_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.c.tra.n_epochs} ---")
            start_time = time.time()

            # swap dataset for alternating-ring mode before anything else
            self._switch_to_epoch_dataset(epoch)

            # activate data augmentation after a few initial epochs
            if epoch >= 5 and self.c.dl.data_aug:
                self.train_dataset.apply_transforms = True
            
            # check for new hard-mined samples to add to the pool
            self._update_hard_mining_samples(epoch)
            
            # create a new injector for the epoch if hard samples are available
            # in alternating-ring mode, only inject on ring epochs
            hard_injector = None
            if self.hard_samples and self._hm_active_this_epoch:
                # multiscroll: route by scroll_id to per-scroll train sets; single
                # scroll: wrap the current train dataset (handles alternating swap).
                if getattr(self, '_scroll_train_sets', None):
                    ds_arg = self._scroll_train_sets
                else:
                    ds_arg = {int(self._scroll_ids[0]): self.train_dataset}
                hard_injector = HardMiningInjector(self.hard_samples, ds_arg)
                if (epoch % self.c.tra.eval_int) == 0:
                    self.vis.writer.add_scalar("HardMining/InjectedSamplesPlanned", len(self.hard_samples), epoch)

            # run training and validation for the epoch
            train_metrics = self.train_epoch(hard_injector)

            # brief thermal cooldown between train and validation every epoch
            _val_cool = int(getattr(self.c.tra, 'val_cooldown_secs', 0))
            if _val_cool > 0:
                print(f"[COOLDOWN] train->val pause {_val_cool}s...")
                time.sleep(_val_cool)

            val_metrics = self.validate_epoch()
            
            # update learning rate scheduler and save models
            self.scheduler.step(val_metrics['loss'])
            self._periodic_model_save(epoch, val_metrics)
            
            # log results
            time_elapsed = time.time() - start_time
            self._log_epoch(epoch, train_metrics, val_metrics, time_elapsed)

            # cooldown after heavy inference epochs (probe / eval) to prevent overheating
            cooldown = int(getattr(self.c.tra, 'eval_cooldown_secs', 0))
            is_probe_epoch = (epoch + 1) % self.c.tra.probe_int == 0
            is_eval_epoch  = (epoch + 1) % self.c.tra.eval_int  == 0
            if cooldown > 0 and (is_probe_epoch or is_eval_epoch):
                kind = 'eval+probe' if is_eval_epoch and is_probe_epoch else ('eval' if is_eval_epoch else 'probe')
                print(f"[COOLDOWN] {kind} epoch — sleeping {cooldown}s for hardware to cool...")
                time.sleep(cooldown)

            # unconditional per-epoch thermal cooldown (fires after EVERY epoch).
            # avoid double-sleeping when the eval/probe cooldown above already ran.
            _ep_cool = int(getattr(self.c.tra, 'epoch_cooldown_secs', 0))
            if _ep_cool > 0 and not (cooldown > 0 and (is_probe_epoch or is_eval_epoch)):
                print(f"[COOLDOWN] end-of-epoch pause {_ep_cool}s for hardware to cool...")
                time.sleep(_ep_cool)

        # explicit final checkpoint save (deterministic path) for clean resume by the
        # campaign wrapper; independent of save_int arithmetic.
        _final = getattr(self.c, "save_final", None)
        if _final:
            os.makedirs(os.path.dirname(_final), exist_ok=True)
            save_model(self.model, _final)
            print(f"[save-final] wrote final model to {_final}")

        self.vis.close()
        if getattr(self, 'scroll_vis', None):
            for svis in self.scroll_vis.values():
                svis.close()
        print("Training completed.")


def main():
    """train.py takes only -n for the experiment name; all other config comes from config.py.
    campaign runners override config fields directly by instantiating Config() and mutating
    fields before passing to Trainer. see campaign_runner_p0139_triple.py for examples."""
    import argparse
    parser = argparse.ArgumentParser(description="Vesuvius ink-detection training")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="experiment name (used for TensorBoard log dir and checkpoint naming)")
    args = parser.parse_args()

    c = Config()
    if args.experiment_name:
        c.exp_name = args.experiment_name

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(c.tra.log_dir):
        c.tra.log_dir = os.path.normpath(os.path.join(repo_root, c.tra.log_dir))
    if not os.path.isabs(c.model_dir):
        c.model_dir = os.path.normpath(os.path.join(repo_root, c.model_dir))

    trainer = Trainer(c)
    trainer.run()


if __name__ == "__main__":
    main()
