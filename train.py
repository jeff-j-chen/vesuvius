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
    pairwise_ranking_loss,
    calculate_metrics,
    save_model
)

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from tqdm import tqdm
import sys
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
    if args.log_dir is not None:
        c.tra.log_dir = str(args.log_dir)

    if args.scroll_id is not None:
        c.data.tra_scroll_id = int(args.scroll_id)
    if getattr(args, "scroll_ids", None):
        ids = [int(s.strip()) for s in args.scroll_ids.split(",") if s.strip()]
        if ids:
            c.data.tra_scroll_ids = ids
            # keep tra_scroll_id as the primary (first) fragment for any single-scroll fallbacks
            c.data.tra_scroll_id = ids[0]
    else:
        # if only a single scroll id was given (or default), keep the list in sync
        c.data.tra_scroll_ids = [int(c.data.tra_scroll_id)]
    # resolve which scrolls render evaluation figures (subset of training scrolls).
    # must run AFTER tra_scroll_ids is finalized. None => all training scrolls.
    if getattr(args, "vis_scroll_ids", None):
        vids = [int(s.strip()) for s in args.vis_scroll_ids.split(",") if s.strip()]
        unknown = [v for v in vids if v not in c.data.tra_scroll_ids]
        if unknown:
            print(f"[warn] --vis-scroll-ids ids not in --scroll-ids are ignored: {unknown}")
        kept = [v for v in vids if v in c.data.tra_scroll_ids]
        c.data.vis_scroll_ids = kept or None
    else:
        c.data.vis_scroll_ids = None
    if args.scroll4_id is not None:
        c.data.scroll4_id = int(args.scroll4_id)
    if args.scroll3_id is not None:
        c.data.scroll3_id = int(args.scroll3_id)
    if getattr(args, 'test_scroll_id', None) is not None:
        c.data.test_scroll_id = int(args.test_scroll_id)
    for _tflag in ('test_show_train', 'test_show_scroll2', 'test_show_scroll3', 'test_show_scroll4'):
        if getattr(args, _tflag, False):
            setattr(c.data, _tflag, True)
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
    if args.focal_gamma is not None:
        c.tra.focal_gamma = float(args.focal_gamma)
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
    if getattr(args, 'noise_prob', None) is not None:
        c.dl.noise_prob = float(args.noise_prob)
    if getattr(args, 'rotation_prob', None) is not None:
        c.dl.rotation_prob = float(args.rotation_prob)

    if args.pooling is not None:
        c.model.pooling = str(args.pooling)
    if args.gem_p is not None:
        c.model.gem_p = float(args.gem_p)
    if args.conv3_dilation is not None:
        c.model.conv3_dilation = int(args.conv3_dilation)
    if args.arch is not None:
        c.model.arch = str(args.arch)
    if getattr(args, "init_weights", None) is not None:
        c.init_weights = str(args.init_weights)
    if getattr(args, "save_final", None) is not None:
        c.save_final = str(args.save_final)
    if args.smooth_sigma is not None:
        c.data.smooth_sigma = float(args.smooth_sigma)
    if args.input_mode is not None:
        c.data.input_mode = str(args.input_mode)
    if args.soft_label_prob is not None:
        c.data.soft_label_prob = float(args.soft_label_prob)
    if args.soft_label_value is not None:
        c.data.soft_label_value = float(args.soft_label_value)
    if args.ranking_lambda is not None:
        c.tra.ranking_lambda = float(args.ranking_lambda)
    if args.ranking_margin is not None:
        c.tra.ranking_margin = float(args.ranking_margin)
    if args.ranking_neg_frac is not None:
        c.tra.ranking_neg_frac = float(args.ranking_neg_frac)
    if getattr(args, "probe_rois", None) is not None:
        c.tra.probe_rois_enabled = bool(args.probe_rois)
    if args.pretrain_epochs is not None:
        c.tra.pretrain_epochs = int(args.pretrain_epochs)
    if args.preload_volume:
        c.data.preload_to_ram = True
    if getattr(args, "mask_memmap", False):
        c.data.mask_memmap = True
    if args.ring_negatives:
        c.data.ring_negatives = True
    if getattr(args, "dense_labels", False):
        c.data.dense_labels = True
    if getattr(args, "dense_soft_labels", False):
        c.data.dense_soft_labels = True
    if getattr(args, "test_scroll2_only", False):
        c.data.test_scroll2_only = True
    if args.ring_label_source is not None:
        c.data.ring_label_source = args.ring_label_source
    if getattr(args, "split_axis", None) is not None:
        c.data.split_axis = str(args.split_axis)
    if getattr(args, "train_split_frac", None) is not None:
        c.data.train_split_frac = float(args.train_split_frac)
    if getattr(args, "crop_x_frac", None) is not None:
        c.data.crop_x_frac = tuple(float(v) for v in args.crop_x_frac.split(","))
    if getattr(args, "crop_y_frac", None) is not None:
        c.data.crop_y_frac = tuple(float(v) for v in args.crop_y_frac.split(","))
    if args.alternating_ring:
        c.data.alternating_ring = True
    if args.eval_cooldown is not None:
        c.tra.eval_cooldown_secs = int(args.eval_cooldown)
    if getattr(args, 'val_cooldown', None) is not None:
        c.tra.val_cooldown_secs = int(args.val_cooldown)
    if getattr(args, 'fig_chunk_cooldown', None) is not None:
        c.tra.fig_chunk_cooldown_ms = int(args.fig_chunk_cooldown)
    if getattr(args, 'epoch_cooldown', None) is not None:
        c.tra.epoch_cooldown_secs = int(args.epoch_cooldown)
    if args.depth is not None:
        c.data.depth = int(args.depth)
    if getattr(args, "tile_size", None) is not None:
        c.data.tile_size = int(args.tile_size)
    if args.train_d_start is not None:
        c.data.train_d_start = int(args.train_d_start)
        c.data.train_d_end = c.data.train_d_start + c.data.depth  # default; overridden below if --train-d-end given
    if args.train_d_end is not None:
        c.data.train_d_end = int(args.train_d_end)
    # inference/eval depth window (drives the eval figure z_range + inference sweep). set to
    # 0/64 to visualize the whole sheet depth when the ink layer is unknown.
    if args.d_start is not None:
        c.data.d_start = int(args.d_start)
    if args.d_end is not None:
        c.data.d_end = int(args.d_end)
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
        scroll_ids = getattr(self, '_scroll_ids', None) or [self.c.data.tra_scroll_id]
        if len(scroll_ids) > 1:
            # merged training stream: the main visualizer owns the single tensorboard
            # run folder and logs scalar metrics; one figure-visualizer per scroll
            # renders its own eval/test figures into that SAME folder, namespacing its
            # tags with s<sid>/. this keeps the run list at one folder regardless of
            # scroll count (tag '/' is UI grouping only, not a folder on disk). probe
            # ROIs stay global (rendered once, unprefixed).
            self.vis = TensorboardVisualizer(self.c, mode='metrics')
            self.scroll_vis = {}
            for sid in scroll_ids:
                self.scroll_vis[sid] = TensorboardVisualizer(
                    self.c, mode='train', scroll_id=sid,
                    shared_writer=self.vis.writer, tag_prefix=f"s{sid}/"
                )
        else:
            self.vis = TensorboardVisualizer(self.c)
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

        # resolve the list of scroll fragments to train on. multiple fragments are
        # merged into a single stream so every epoch sees all of them (integrated
        # batches). defaults to the single primary scroll for backward compatibility.
        scroll_ids = [int(s) for s in (getattr(self.c.data, 'tra_scroll_ids', None) or [self.c.data.tra_scroll_id])]
        self._scroll_ids = scroll_ids
        self._scroll_train_sets = None   # {scroll_id: train_set} in multiscroll, for HM routing
        multi = len(scroll_ids) > 1

        alternating = getattr(self.c.data, 'alternating_ring', False)

        if multi:
            # multi-scroll merged training. alternating-ring is still unsupported here,
            # but hard mining IS supported: each scroll mines into its own dir and the
            # injector routes every mined record back to the right scroll volume via its
            # scroll_id. build a per-scroll train-set map so injection can resolve it.
            if alternating:
                print("[multi-scroll] alternating_ring not supported with multiple scrolls; ignoring")

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
            # pos_weight from the merged distribution across all fragments. dense per-pixel
            # BCE needs a PIXEL-level pos_weight (ink px fraction), not the tile class ratio.
            if getattr(self.c.data, 'dense_labels', False):
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

        if alternating:
            # build both datasets upfront: full-mask set and ring-mask set
            t_set_full, v_set = data_manager.get_datasets()  # ring_negatives=False path
            # temporarily enable ring for the ring dataset
            self.c.data.ring_negatives = True
            t_set_ring, _ = data_manager.get_datasets()
            self.c.data.ring_negatives = False
            # loader defaults to full set; switched per-epoch in run()
            t_loader, v_loader = get_dataloaders(t_set_full, v_set, self.c)
            # pos_weight from full scroll distribution (ring epochs re-weight internally)
            pos_weight = calc_class_wgts(t_set_full, v_set,
                                         scroll_id=self.c.data.tra_scroll_id)
            self._t_set_full = t_set_full
            self._t_set_ring = t_set_ring
            print(f"[alternating_ring] full_tiles={len(t_set_full)}  ring_tiles={len(t_set_ring)}")
        else:
            t_set_full, v_set = data_manager.get_datasets()
            t_loader, v_loader = get_dataloaders(t_set_full, v_set, self.c)
            ring_mode = getattr(self.c.data, 'ring_negatives', False)
            if getattr(self.c.data, 'dense_labels', False):
                # dense per-pixel BCE needs a PIXEL-level pos_weight (ink px fraction),
                # not the tile-level class ratio; tile-label sampling would also crash on
                # the (1,T,T) label maps.
                pos_weight = calc_dense_pos_weight(t_set_full)
            else:
                pos_weight = calc_class_wgts(
                    t_set_full, t_set_full if ring_mode else v_set,
                    scroll_id=None if ring_mode else self.c.data.tra_scroll_id
                )
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
        criterion = create_loss_function(self.pos_weight, self.c)
        
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

            # pairwise ranking loss: positive tiles must score above negatives by >= margin
            ranking_lambda = float(getattr(self.c.tra, 'ranking_lambda', 0.0))
            if ranking_lambda > 0:
                probs = torch.sigmoid(outputs).squeeze(1)
                labels_flat = b_labels.squeeze(1)
                rank_loss = pairwise_ranking_loss(
                    probs, labels_flat,
                    margin=float(getattr(self.c.tra, 'ranking_margin', 0.3)),
                    neg_frac=float(getattr(self.c.tra, 'ranking_neg_frac', 1.0))
                )
                loss = loss + ranking_lambda * rank_loss

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
            dense = getattr(self.c.data, 'dense_labels', False)
            for idx, (sid, svis) in enumerate(self.scroll_vis.items()):
                if eval_due and getattr(svis, 'eval_enabled', True):
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
                if test_due:
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
    """parses arguments, initializes the configuration, and starts training"""
    parser = argparse.ArgumentParser(description="Training script for Vesuvius model.")
    parser.add_argument("-n", "--experiment_name", type=str, default="", help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--eval-int", type=int, default=None, help="Override evaluation interval")
    parser.add_argument("--test-int", type=int, default=None, help="Override test figure interval")
    parser.add_argument("--probe-int", type=int, default=None, help="Override probe interval")
    parser.add_argument("--log-dir", type=str, default=None, help="Override TensorBoard log directory")

    parser.add_argument("--scroll-id", type=int, default=None, help="Scroll id for train/valid")
    parser.add_argument("--scroll-ids", type=str, default=None,
                        help="Comma-separated scroll fragment ids to train on simultaneously (merged batches), e.g. 20230827161847,20230702185753")
    parser.add_argument("--vis-scroll-ids", type=str, default=None,
                        help="Comma-separated subset of --scroll-ids that render EVALUATION figures each eval step (default: all training scrolls). Test figures still render for every scroll unless --test-scroll2-only.")
    parser.add_argument("--scroll4-id", type=int, default=None, help="Scroll id for scroll4 eval path")
    parser.add_argument("--scroll3-id", type=int, default=None, help="Scroll id for scroll3 goal-scroll test figure")
    parser.add_argument("--test-scroll-id", type=int, default=None, help="primary test fragment id; test figures render ONLY this by default (others opt-in via --test-show-*)")
    parser.add_argument("--test-show-train", action="store_true", default=False, help="also render the training-scroll Test figure")
    parser.add_argument("--test-show-scroll2", action="store_true", default=False, help="also render the scroll2 test figure")
    parser.add_argument("--test-show-scroll3", action="store_true", default=False, help="also render the scroll3 test figure")
    parser.add_argument("--test-show-scroll4", action="store_true", default=False, help="also render the scroll4 test figure")
    parser.add_argument("--zarr-path", type=str, default=None, help="Path to zarr root")

    parser.add_argument("--batch-size", type=int, default=None, help="Dataloader batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument("--data-aug", type=int, choices=[0, 1], default=None, help="Enable/disable data augmentation")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--l1-lambda", type=float, default=None, help="L1 regularization strength")

    parser.add_argument("--no-hard-mining", action="store_true", help="Disable hard mining entirely")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal loss gamma (0=BCE, 2.0=standard focal)")
    parser.add_argument("--hm-frac", type=float, default=None, help="Hard-mining sample fraction")
    parser.add_argument("--hn-cutoff", type=float, default=None, help="Hard-negative score cutoff")
    parser.add_argument("--hp-cutoff", type=float, default=None, help="Hard-positive score cutoff")
    parser.add_argument("--hm-dir", type=str, default=None, help="Hard-mining directory")

    parser.add_argument("--channel-mixing-prob", type=float, default=None, help="Depth channel permutation probability")
    parser.add_argument("--noise-prob", type=float, default=None, help="Gaussian noise augmentation probability (default 0.30; set low when signal is faint)")
    parser.add_argument("--rotation-prob", type=float, default=None, help="90/180/270 rotation augmentation probability")
    parser.add_argument("--pooling", type=str, choices=["avg", "max", "gem"], default=None, help="Pooling mode")
    parser.add_argument("--gem-p", type=float, default=None, help="Initial GeM pooling p")
    parser.add_argument("--conv3-dilation", type=int, default=None, help="Dilation for final conv stage")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture variant (v1, v2_slim_head, ...)")    
    parser.add_argument("--init-weights", type=str, default=None,
                        help="path to a checkpoint to warm-start the model (loaded strict=False; e.g. MAE-pretrained encoder)")
    parser.add_argument("--save-final", type=str, default=None,
                        help="path to save the final model at end of training (deterministic, for resume by campaign wrapper)")
    parser.add_argument("--smooth-sigma", type=float, default=None, help="Gaussian blur sigma applied to inference prediction maps (0=off)")
    parser.add_argument("--conv1-drop", type=float, default=None, help="Dropout after first conv block")
    parser.add_argument("--conv2-drop", type=float, default=None, help="Dropout after second conv block")
    parser.add_argument("--fc1-drop", type=float, default=None, help="Dropout on first FC layers")
    parser.add_argument("--fc2-drop", type=float, default=None, help="Dropout on final FC layer")
    # campaign 4 additions
    parser.add_argument("--input-mode", type=str, choices=["single","diff","triple","double","fulldepth"], default=None,
                        help="Input representation mode")
    parser.add_argument("--soft-label-prob", type=float, default=None,
                        help="Probability of sampling flanking band with soft label for ink tiles")
    parser.add_argument("--soft-label-value", type=float, default=None,
                        help="Label value assigned to flanking-band ink tiles (default 0.3)")
    parser.add_argument("--ranking-lambda", type=float, default=None,
                        help="Weight for pairwise ranking loss (0=off)")
    parser.add_argument("--ranking-margin", type=float, default=None,
                        help="Margin for pairwise ranking loss (default 0.3)")
    parser.add_argument("--ranking-neg-frac", type=float, default=None,
                        help="Partial-AUC: fraction of hardest negatives to rank against (1.0=all pairs, <1.0 focuses on low-FPR region)")
    parser.add_argument("--probe-rois", dest="probe_rois", action="store_true", default=None,
                        help="Enable fixed readability probe-ROI figures (default off)")
    parser.add_argument("--no-probe-rois", dest="probe_rois", action="store_false", default=None,
                        help="Disable probe-ROI figures")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="Epochs of self-supervised band-identity pretraining before BCE")
    parser.add_argument("--preload-volume", action="store_true", default=False,
                        help="load full zarr into RAM before training; only safe for small scrolls (~10GB free RAM needed)")
    parser.add_argument("--mask-memmap", action="store_true", default=False,
                        help="back each dataset's binary mask/labels with an on-disk memmap so they pickle as a path, not data; avoids per-worker RAM duplication at the 5-10 fragment scale (scratch dir via env VESUVIUS_MMAP_DIR)")
    parser.add_argument("--ring-negatives", action="store_true", default=False,
                        help="restrict training negatives to a ring around ink labels (~1:1 pos/neg ratio, no unlabeled-ink contamination)")
    parser.add_argument("--dense-labels", action="store_true", default=False,
                        help="DENSE per-pixel supervision: emit the (1,T,T) ink-label MAP per tile and train per-pixel masked BCE (switch away from binary tile labels). requires a dense arch, e.g. --arch dense_unet")
    parser.add_argument("--dense-soft-labels", action="store_true", default=False,
                        help="use continuous soft ink labels (soft_inklabels/<id>.png = eroded dilated+blurred) as the dense target instead of hard 0/1; calibrated soft edges. requires --dense-labels")
    parser.add_argument("--eval-cooldown", type=int, default=None,
                        help="seconds to sleep after probe/eval epochs to let hardware cool (default 0)")
    parser.add_argument("--val-cooldown", type=int, default=None,
                        help="seconds to sleep between training and validation each epoch (thermal relief, default 0)")
    parser.add_argument("--fig-chunk-cooldown", type=int, default=None,
                        help="milliseconds to sleep between spatial chunks during eval figure inference (default 0)")
    parser.add_argument("--epoch-cooldown", type=int, default=None,
                        help="seconds to sleep after EVERY epoch for thermal relief (default 0)")
    parser.add_argument("--ring-label-source", type=str, default=None, choices=["eroded","original","closed"],
                        help="which inklabels to use for ring boundary: 'original' (default, safest) or 'eroded'")
    parser.add_argument("--split-axis", type=str, default=None, choices=["x","y"],
                        help="train/val split axis: 'x' vertical (left/right, legacy) or 'y' horizontal (top train/bottom valid)")
    parser.add_argument("--train-split-frac", type=float, default=None,
                        help="fraction of the (cropped) split axis given to train (default 0.75)")
    parser.add_argument("--crop-x-frac", type=str, default=None,
                        help="region crop along x as 'start,end' fractions of the frame (e.g. 0.4,1.0 = right 60%%)")
    parser.add_argument("--crop-y-frac", type=str, default=None,
                        help="region crop along y as 'start,end' fractions of the frame (e.g. 0.0,0.75 = top 75%%)")
    parser.add_argument("--alternating-ring", action="store_true", default=False,
                        help="alternate between full dataset (even epochs) and ring dataset (odd epochs); hard mining only on ring epochs")
    parser.add_argument("--depth", type=int, default=None,
                        help="number of depth slices per tile (default 8; use 12 for ink band z=28-40)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="in-plane tile size (default 32; use 106 for the 2.4um teacher so a "
                             "106x106 teacher tile == a 32x32 7.91um tile physically)")
    parser.add_argument("--train-d-start", type=int, default=None,
                        help="start z-index of training depth window")
    parser.add_argument("--train-d-end", type=int, default=None,
                        help="end z-index of training depth window (exclusive); overrides the default train_d_start+depth")
    parser.add_argument("--d-start", type=int, default=None,
                        help="start z-index of the inference/eval depth window (eval figure z_range)")
    parser.add_argument("--d-end", type=int, default=None,
                        help="end z-index of the inference/eval depth window (exclusive); set 0/64 to sweep full depth")
    parser.add_argument("--test-scroll2-only", action="store_true", default=False,
                        help="test figure renders ONLY the full goal scroll2 fragment (skips the expensive training-scroll Test figure + scroll4); use for affordable end-of-training transfer checks")
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