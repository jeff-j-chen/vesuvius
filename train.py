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
from utils.dataloader import DataManager, MultiScrollIterableDataset, get_dataloaders, DotPositiveDataset, imread_gray, get_tile_pos_weight
from utils.hard_mining import HardMiningInjector, HardMiningManager
from utils.model import create_model, supcon_loss
from utils.surface import make_surface_targets_from_depth, surface_supervision_loss
from utils.training_utils import (
    calculate_character_metrics,
    calculate_metrics,
    create_loss_function,
    create_optimizer_and_scheduler,
    save_model,
)
from utils.visualizer import TensorboardVisualizer


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _character_group_losses(
    per_target_loss: torch.Tensor,
    mask: torch.Tensor,
    character_ids: torch.Tensor,
) -> tuple[list[int], list[torch.Tensor]]:
    """mean supervised loss for each character represented in a batch."""
    flat_loss = per_target_loss.reshape(-1)
    flat_mask = mask.reshape(-1) > 0
    flat_ids = character_ids.reshape(-1).to(device=flat_loss.device, dtype=torch.long)
    ids: list[int] = []
    losses: list[torch.Tensor] = []
    for value in torch.unique(flat_ids[flat_mask]):
        group_id = int(value.item())
        if group_id <= 0:
            continue
        selected = flat_mask & (flat_ids == value)
        if selected.any():
            ids.append(group_id)
            losses.append(flat_loss[selected].mean())
    return ids, losses


def character_bag_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    character_ids: torch.Tensor,
    margin: float = 0.5,
    topk_frac: float = 0.5,
) -> tuple[torch.Tensor, int]:
    """rank each character's positive bag above its assigned local-ring bag."""
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1) > 0
    flat_ids = character_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
    losses = []
    fraction = min(max(float(topk_frac), 1e-6), 1.0)
    for value in torch.unique(flat_ids[flat_mask]):
        if int(value.item()) <= 0:
            continue
        selected = flat_mask & (flat_ids == value)
        positives = flat_logits[selected & (flat_labels > 0.5)]
        negatives = flat_logits[selected & (flat_labels <= 0.5)]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        pos_k = max(1, int(math.ceil(positives.numel() * fraction)))
        neg_k = max(1, int(math.ceil(negatives.numel() * fraction)))
        positive_score = torch.topk(positives, pos_k).values.mean()
        negative_score = torch.topk(negatives, neg_k).values.mean()
        losses.append(F.softplus(float(margin) + negative_score - positive_score))
    if not losses:
        return logits.new_zeros(()), 0
    return torch.stack(losses).mean(), len(losses)


def character_cvar_loss(
    group_losses: list[torch.Tensor],
    alpha: float,
) -> tuple[torch.Tensor | None, int]:
    """mean loss over the worst alpha fraction of characters in the batch."""
    if not group_losses:
        return None, 0
    stacked = torch.stack(group_losses)
    tail_count = max(1, int(math.ceil(stacked.numel() * min(max(float(alpha), 1e-6), 1.0))))
    return torch.topk(stacked, tail_count).values.mean(), tail_count


def character_groupdro_loss(
    group_ids: list[int],
    group_losses: list[torch.Tensor],
    log_weights: dict[int, float],
    eta: float,
    max_ratio: float,
) -> tuple[torch.Tensor | None, float]:
    """exponentially upweight persistently difficult characters."""
    if not group_losses:
        return None, 1.0
    for group_id, group_loss in zip(group_ids, group_losses):
        log_weights[group_id] = log_weights.get(group_id, 0.0) + float(eta) * float(
            group_loss.detach().item()
        )
    maximum = max(log_weights.values())
    minimum = maximum - math.log(max(float(max_ratio), 1.0))
    for group_id in log_weights:
        log_weights[group_id] = max(log_weights[group_id], minimum) - maximum
    weights = torch.tensor(
        [math.exp(log_weights[group_id]) for group_id in group_ids],
        device=group_losses[0].device,
        dtype=group_losses[0].dtype,
    )
    weights = weights / weights.sum().clamp(min=1e-8)
    objective = sum(weight * value for weight, value in zip(weights, group_losses))
    ratio = float(weights.max().item() / weights.min().clamp(min=1e-8).item())
    return objective, ratio


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
        if bool(getattr(config.tra, "character_groupdro", False)) and bool(
            getattr(config.tra, "character_cvar", False)
        ):
            raise ValueError("character_groupdro and character_cvar are independent objectives")
        set_seed(
            int(getattr(config.tra, "seed", 41)),
            deterministic=bool(getattr(config.tra, "deterministic", False)),
        )
        if bool(getattr(config.tra, "context_consistency", False)):
            torch.backends.cudnn.benchmark = False
            print("[seed] context consistency uses variable batch sizes -- cudnn benchmark off")
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
        self.best_val_character = -1.0
        self._character_groupdro_log_weights: dict[int, float] = {}
        self._last_character_objectives = (0.0, 0.0, 0.0, 0.0)
        self._last_dann_accuracy = 0.0
        self._last_grl_scale = 0.0

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
            balance_scrolls = bool(getattr(self.c.data, "character_balance_scrolls", False))
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

            dot_dir = str(getattr(self.c.data, "dot_inklabel_dir", "") or "")
            # when a whitelist is set, only these scrolls have processed dots; the rest
            # of dots/*.png are unprocessed placeholders and must be skipped
            dot_whitelist = {int(s) for s in (getattr(self.c.data, "dot_scroll_whitelist", []) or [])}
            if dot_dir and balance_scrolls:
                print("[multi-scroll] dot-positive extras disabled for balanced character sampling")
            elif dot_dir:
                for scroll_id, dm in self._scroll_dms.items():
                    if dot_whitelist and int(scroll_id) not in dot_whitelist:
                        continue
                    dot_path = os.path.join(dot_dir, f"{scroll_id}.png")
                    if not os.path.exists(dot_path):
                        continue
                    dot_lbl = imread_gray(dot_path)
                    if dot_lbl is None:
                        continue
                    dot_ds = DotPositiveDataset(dm, dot_lbl)
                    if len(dot_ds) > 0:
                        train_sets.append(dot_ds)

            merged_train = MultiScrollIterableDataset(
                train_sets,
                balance_scrolls=balance_scrolls,
            )
            merged_valid = MultiScrollIterableDataset(valid_sets)
            train_loader, valid_loader = get_dataloaders(merged_train, merged_valid, self.c)
            # per-scroll InkVolumeDatasets (excludes DotPositiveDataset) for pos_weight sampling
            self._train_children = list(self._scroll_train_sets.values())
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
        self._train_children = [train_set]
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
        # multitile supervises at the 8px sub-tile level, whose class balance (~5:1 neg:pos) is
        # far worse than the ring-balanced single-tile grid; without pos_weight the rare positives
        # get abandoned (GCE/BCE alike) and the head collapses to all-negative.
        # precedence: explicit tile_pos_weight (>0) -> auto (compute+cache from data) -> None.
        _tpw = float(getattr(self.c.tra, "tile_pos_weight", 0.0) or 0.0)
        if _tpw > 0:
            _pw = torch.tensor([_tpw], dtype=torch.float32)
        elif bool(getattr(self.c.tra, "tile_pos_weight_auto", False)):
            _pw = get_tile_pos_weight(getattr(self, "_train_children", []), self.c)
        else:
            _pw = None
        criterion = create_loss_function(_pw, self.c)
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

    def _apply_fda(self, images: torch.Tensor) -> torch.Tensor:
        """within-batch FDA: swap low-freq amplitude between random tile pairs.
        targets fragment-specific amplitude texture; preserves phase (ink structure)."""
        fda_prob = float(getattr(self.c.dl, "fda_prob", 0.0))
        if fda_prob <= 0.0:
            return images
        fda_beta = float(getattr(self.c.dl, "fda_beta", 0.05))
        B = images.shape[0]
        H, W = images.shape[-2], images.shape[-1]
        apply_mask = torch.rand(B, device=images.device) < fda_prob
        if not apply_mask.any():
            return images
        perm = torch.randperm(B, device=images.device)
        fft = torch.fft.rfft2(images)           # (..., H, W//2+1) complex
        amp = fft.abs()
        phase = fft.angle()
        style_amp = amp[perm]
        # replace low-freq corners; ink strokes are high-freq and survive this
        h = max(1, int(fda_beta * H / 2))
        w = max(1, int(fda_beta * W / 2) + 1)
        new_amp = amp.clone()
        new_amp[..., :h, :w] = style_amp[..., :h, :w]
        if h > 1:
            new_amp[..., -h + 1:, :w] = style_amp[..., -h + 1:, :w]
        mask_shape = (B,) + (1,) * (images.dim() - 1)
        blended = torch.where(apply_mask.view(*mask_shape), new_amp, amp)
        out = torch.fft.irfft2(torch.polar(blended, phase), s=(H, W))
        return out.clamp(0.0, 1.0)

    def _train_batch(self, images, labels, mask, domain_ids=None, character_ids=None,
                     target_offsets=None, epoch: int = 0, unlabeled_images=None,
                     surface_depth=None, surface_confidence=None,
                     context_pair=None, context_pair_active=None):
        if mask.reshape(mask.size(0), -1).sum().item() <= 0:
            print("[ERROR] Mask sum is zero, skipping loss calculation.")
            return np.empty([]), np.empty([]), np.empty([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        images = images.to(self.c.device)
        surface_source = images
        images = self._apply_fda(images)
        B = images.size(0)
        labels = labels.to(self.c.device).view(B, -1)          # (B,1) single or (B,K) multitile
        mask = mask.to(self.c.device).view(B, -1)
        if mask.shape[1] == labels.shape[1]:
            mask = (mask > 0).float()                          # per-sub-tile validity (multitile)
        else:
            mask = (mask.sum(dim=1) > 0).float().unsqueeze(1)  # single-tile window gate
        # only supervised targets determine whether this split sees a positive window.
        # held-out ink cells can be present in the same multitile center with mask=0.
        sample_pos = ((labels * mask).amax(dim=1) > 0.5).float()
        if domain_ids is not None:
            domain_ids = domain_ids.to(self.c.device, non_blocking=True).view(-1)
        if target_offsets is not None:
            target_offsets = target_offsets.to(self.c.device, non_blocking=True).view(B, 2)
        character_ids_device = (
            character_ids.to(self.c.device, non_blocking=True).view(B, -1)
            if character_ids is not None else None
        )
        context_active_mask = (
            context_pair_active.view(-1) > 0
            if context_pair_active is not None else None
        )
        context_loss_value = images.new_zeros(())
        bag_rank_loss_value = images.new_zeros(())
        groupdro_loss_value = images.new_zeros(())
        cvar_loss_value = images.new_zeros(())
        dann_accuracy_value = images.new_zeros(())
        dann_grl_value = images.new_zeros(())

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.c.device, enabled=self.c.device == "cuda"):
            use_extras = hasattr(self.model, "forward_with_extras") and any([
                bool(getattr(self.c.tra, "supcon", False)),
                bool(getattr(self.c.tra, "dann", False)),
                bool(getattr(self.c.tra, "spill_reduction", False)),
                bool(getattr(self.c.tra, "spill_entropy", False)),
                bool(getattr(self.c.tra, "spill_prob", False)),
            ])
            if bool(getattr(self.c.tra, "dann_grl_anneal", False)):
                n_ep = float(getattr(self.c.tra, "n_epochs", 12))
                p = float(epoch) / max(1.0, n_ep)
                anneal = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            else:
                anneal = 1.0
            grl_scale = (
                float(getattr(self.c.tra, "dann_lambda", 0.0)) * anneal
                if bool(getattr(self.c.tra, "dann", False)) else 1.0
            )
            if use_extras:
                outputs, _, domain_logits, supcon_z = self.model.forward_with_extras(
                    images,
                    grl_scale=grl_scale,
                    target_offsets=target_offsets,
                )
            else:
                outputs = self.model(images, target_offsets=target_offsets)
                domain_logits = None
                supcon_z = None

            attn_entropy_loss = getattr(self.model, "last_attn_entropy_loss", None)
            attn_entropy_per_target = getattr(self.model, "last_attn_entropy_per_target", None)
            new_surface_logits = getattr(self.model, "last_new_surface_logits", None)

            if outputs.dim() == 4:
                outputs = outputs.flatten(1).max(dim=1, keepdim=True).values

            targets = labels.float()
            explicit_negative = labels < 0
            targets = torch.where(explicit_negative, torch.zeros_like(targets), targets)
            pos_smooth = float(getattr(self.c.tra, "label_smooth_pos", 0.0))
            neg_smooth = float(getattr(self.c.tra, "label_smooth_neg", 0.0))
            if pos_smooth > 0 or neg_smooth > 0:
                pos_mask = labels > 0.5
                targets = torch.where(
                    pos_mask,
                    torch.full_like(targets, 1.0 - pos_smooth),
                    torch.full_like(targets, neg_smooth),
                )
                targets = torch.where(explicit_negative, torch.zeros_like(targets), targets)

            per_target_loss = self.criterion(outputs, targets)
            raw_loss = per_target_loss * mask
            denom = mask.sum()
            raw_loss_value = raw_loss.sum() / denom
            primary_loss = raw_loss.sum() / denom
            group_ids: list[int] = []
            group_losses: list[torch.Tensor] = []
            if character_ids_device is not None and any([
                bool(getattr(self.c.tra, "character_groupdro", False)),
                bool(getattr(self.c.tra, "character_cvar", False)),
            ]):
                group_ids, group_losses = _character_group_losses(
                    per_target_loss,
                    mask,
                    character_ids_device,
                )
            if bool(getattr(self.c.tra, "character_groupdro", False)):
                robust_loss, _ = character_groupdro_loss(
                    group_ids,
                    group_losses,
                    self._character_groupdro_log_weights,
                    eta=float(getattr(self.c.tra, "character_groupdro_eta", 0.05)),
                    max_ratio=float(getattr(self.c.tra, "character_groupdro_max_ratio", 3.0)),
                )
                if robust_loss is not None:
                    primary_loss = robust_loss
                    groupdro_loss_value = robust_loss
            elif bool(getattr(self.c.tra, "character_cvar", False)):
                robust_loss, _ = character_cvar_loss(
                    group_losses,
                    alpha=float(getattr(self.c.tra, "character_cvar_alpha", 0.25)),
                )
                if robust_loss is not None:
                    primary_loss = robust_loss
                    cvar_loss_value = robust_loss
            loss = primary_loss
            if self.c.tra.l1_lambda > 0:
                l1_loss = sum(param.abs().sum() for param in self.model.parameters())
                loss = loss + self.c.tra.l1_lambda * l1_loss

            if (
                bool(getattr(self.c.tra, "character_bag_ranking", False))
                and character_ids_device is not None
            ):
                bag_rank_loss_value, _ = character_bag_ranking_loss(
                    outputs,
                    labels,
                    mask,
                    character_ids_device,
                    margin=float(getattr(self.c.tra, "character_bag_margin", 0.5)),
                    topk_frac=float(getattr(self.c.tra, "character_bag_topk_frac", 0.5)),
                )
                loss = loss + float(getattr(self.c.tra, "character_bag_lambda", 0.2)) * bag_rank_loss_value

            tta_lambda = float(getattr(self.c.tra, "tta_consistency_lambda", 0.0))
            _tta_on = getattr(self.c.tra, "tta_consistency", False) and tta_lambda > 0
            if _tta_on:
                # optional subsampling: skip the 2nd forward on (1-prob) of steps for speed.
                # only draw RNG when prob<1 so the default (1.0) stays byte-identical to before.
                tta_prob = float(getattr(self.c.tra, "tta_consistency_prob", 1.0))
                if tta_prob < 1.0 and torch.rand(1).item() >= tta_prob:
                    _tta_on = False
            if _tta_on:
                mode = str(getattr(self.c.tra, "tta_consistency_mode", "flips")).lower()
                # each entry pairs an INPUT transform with the inverse applied to the multitile
                # (B,n,n) sub-tile grid (grid dims -2,-1 == image H,W) so a flipped/rotated input
                # is compared cell-for-cell. single-tile output is a scalar -> inverse never runs.
                if mode == "dihedral":
                    choices = (
                        (lambda t: torch.flip(t, dims=[-1]),          lambda g: torch.flip(g, dims=[-1]),
                         lambda o: torch.stack((o[:, 0], -o[:, 1]), dim=1)),
                        (lambda t: torch.flip(t, dims=[-2]),          lambda g: torch.flip(g, dims=[-2]),
                         lambda o: torch.stack((-o[:, 0], o[:, 1]), dim=1)),
                        (lambda t: torch.flip(t, dims=[-1, -2]),      lambda g: torch.flip(g, dims=[-1, -2]),
                         lambda o: -o),
                        (lambda t: torch.rot90(t, 1, dims=[-2, -1]),  lambda g: torch.rot90(g, -1, dims=[-2, -1]),
                         lambda o: torch.stack((-o[:, 1], o[:, 0]), dim=1)),
                        (lambda t: torch.rot90(t, -1, dims=[-2, -1]), lambda g: torch.rot90(g, 1, dims=[-2, -1]),
                         lambda o: torch.stack((o[:, 1], -o[:, 0]), dim=1)),
                    )
                    tf_in, tf_grid_inv, tf_offset = choices[int(torch.randint(0, len(choices), (1,)).item())]
                else:
                    choices = (
                        (lambda t: torch.flip(t, dims=[-1]),     lambda g: torch.flip(g, dims=[-1]),
                         lambda o: torch.stack((o[:, 0], -o[:, 1]), dim=1)),
                        (lambda t: torch.flip(t, dims=[-2]),     lambda g: torch.flip(g, dims=[-2]),
                         lambda o: torch.stack((-o[:, 0], o[:, 1]), dim=1)),
                        (lambda t: torch.flip(t, dims=[-1, -2]), lambda g: torch.flip(g, dims=[-1, -2]),
                         lambda o: -o),
                    )
                    tf_in, tf_grid_inv, tf_offset = choices[int(torch.randint(0, 3, (1,)).item())]
                view = tf_in(images)
                other_offsets = tf_offset(target_offsets) if target_offsets is not None else None
                # eval mode for TTA second pass: BN uses running stats (no in-place updates)
                # which prevents the autograd version conflict with the compiled first-pass graph
                self.model.eval()
                other = self.model(view.contiguous(), target_offsets=other_offsets)
                self.model.train()
                if other.dim() == 4:
                    other = other.flatten(1).max(dim=1, keepdim=True).values
                # multitile: undo the input transform on the (B,n,n) grid so cells line up
                mt_n = int(getattr(self.model, "_mt_grid", 1)) if getattr(self.model, "_multitile", False) else 1
                if mt_n > 1 and other.dim() == 2 and other.shape[1] == mt_n * mt_n:
                    other = tf_grid_inv(other.view(-1, mt_n, mt_n)).reshape(other.shape[0], mt_n * mt_n)
                p1 = torch.sigmoid(outputs)
                p2 = torch.sigmoid(other)
                consistency = ((p2 - p1.detach()) ** 2) * mask
                loss = loss + tta_lambda * (consistency.sum() / denom.clamp(min=1))

            if (attn_entropy_per_target is not None
                    and attn_entropy_per_target.shape == mask.shape):
                loss = loss + (attn_entropy_per_target * mask).sum() / denom.clamp(min=1)
            elif attn_entropy_loss is not None:
                loss = loss + attn_entropy_loss

            dann_loss_value = outputs.new_zeros(())
            if (
                bool(getattr(self.c.tra, "dann", False))
                and domain_ids is not None
                and domain_logits is not None
            ):
                dann_loss_value = F.cross_entropy(domain_logits, domain_ids)
                # GRL scales only the adversarial gradient entering the backbone.
                # The domain head receives full CE gradients so weak lambdas still
                # train a meaningful domain classifier.
                loss = loss + dann_loss_value
                dann_accuracy_value = (
                    domain_logits.detach().argmax(dim=1) == domain_ids
                ).float().mean()
                dann_grl_value = outputs.new_tensor(grl_scale)

            spill_loss_value = outputs.new_zeros(())
            center_voxel_map = getattr(self.model, "last_center_voxel_map", None)

            def _positive_depth_profile(voxel_map):
                """mean each depth over supervised positive multitile cells only."""
                if labels.shape[1] <= 1:
                    return voxel_map.mean(dim=(3, 4)).squeeze(1)
                n = int(round(labels.shape[1] ** 0.5))
                h, w = voxel_map.shape[3], voxel_map.shape[4]
                if n * n != labels.shape[1] or h % n or w % n:
                    return voxel_map.mean(dim=(3, 4)).squeeze(1)
                positive = ((labels > 0.5) & (mask > 0)).float().view(B, n, n)
                spatial = positive.repeat_interleave(h // n, dim=1).repeat_interleave(
                    w // n, dim=2
                ).unsqueeze(1).unsqueeze(2)
                spatial_denom = spatial.sum(dim=(3, 4)).clamp(min=1.0)
                return ((voxel_map * spatial).sum(dim=(3, 4)) / spatial_denom).squeeze(1)

            if bool(getattr(self.c.tra, "spill_reduction", False)) and center_voxel_map is not None:
                pos_mask = sample_pos
                # variance of mean logit per depth slice: high var = depth-selective (good)
                # low var = uniform across all layers (spill); no cap on prediction confidence
                depth_logits = _positive_depth_profile(center_voxel_map)     # [B, D]
                depth_var = depth_logits.var(dim=1, unbiased=False)           # [B]
                min_var = float(getattr(self.c.tra, "spill_min_depth_var", 0.5))
                spill_loss_value = (
                    (F.relu(min_var - depth_var) * pos_mask).sum()
                    / pos_mask.sum().clamp(min=1.0)
                )
                loss = loss + float(getattr(self.c.tra, "spill_lambda", 0.0)) * spill_loss_value

            if bool(getattr(self.c.tra, "spill_prob", False)) and center_voxel_map is not None:
                pos_mask = sample_pos
                if pos_mask.sum() > 0:
                    # original prob-based spill: penalize active-depth-fraction > max_active_frac
                    center_probs = torch.sigmoid(center_voxel_map)
                    depth_profile = _positive_depth_profile(center_probs)  # [B, D]
                    depth_thresh = float(getattr(self.c.tra, "spill_depth_threshold", 0.35))
                    depth_tau = float(getattr(self.c.tra, "spill_active_depth_tau", 0.08))
                    max_active_frac = float(getattr(self.c.tra, "spill_max_active_depth_frac", 0.35))
                    active_depth_frac = torch.sigmoid(
                        (depth_profile - depth_thresh) / max(depth_tau, 1e-6)
                    ).mean(dim=1)
                    spill_loss_value = spill_loss_value + (
                        (F.relu(active_depth_frac - max_active_frac) * pos_mask).sum()
                        / pos_mask.sum().clamp(min=1.0)
                    )
                    loss = loss + float(getattr(self.c.tra, "spill_lambda", 0.0)) * spill_loss_value

            if bool(getattr(self.c.tra, "spill_entropy", False)):
                full_voxel_map = getattr(self.model, "last_voxel_map_full", None)
                if full_voxel_map is not None:
                    pos_mask_e = sample_pos
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

            surface_loss_value = outputs.new_zeros(())
            surface_alpha_value = outputs.new_zeros(())
            supcon_loss_value = outputs.new_zeros(())
            weighted_supcon_loss_value = outputs.new_zeros(())
            if bool(getattr(self.c.model, "new_learned_surface", False)) and new_surface_logits is not None:
                if surface_depth is None or surface_confidence is None:
                    raise RuntimeError("offline surface teacher is required for surface training")
                downsample = max(1, int(getattr(self.c.data, "context_downsample", 1)))
                if downsample > 1:
                    surface_volume = F.avg_pool3d(
                        surface_source,
                        kernel_size=(1, downsample, downsample),
                        stride=(1, downsample, downsample),
                    )
                else:
                    surface_volume = surface_source
                teacher_depth = surface_depth.to(self.c.device, non_blocking=True).float()
                teacher_confidence = surface_confidence.to(
                    self.c.device,
                    non_blocking=True,
                ).float()
                if teacher_depth.shape[-2:] != new_surface_logits.shape[-2:]:
                    teacher_depth = F.interpolate(
                        teacher_depth,
                        size=new_surface_logits.shape[-2:],
                        mode="nearest",
                    )
                    teacher_confidence = F.interpolate(
                        teacher_confidence,
                        size=new_surface_logits.shape[-2:],
                        mode="area",
                    )
                surface_target, surface_valid = make_surface_targets_from_depth(
                    teacher_depth,
                    teacher_confidence,
                    new_surface_logits.shape[2],
                    target_sigma=float(getattr(self.c.tra, "surface_target_sigma", 0.75)),
                )
                surface_total, surface_ce, surface_smooth, surface_valid = surface_supervision_loss(
                    new_surface_logits,
                    surface_volume,
                    smooth_weight=float(getattr(self.c.tra, "new_surface_smooth_lambda", 0.02)),
                    target=surface_target,
                    valid=surface_valid,
                )
                surface_loss_value = surface_total
                loss = loss + float(getattr(self.c.tra, "new_surface_lambda", 0.1)) * surface_total
            surface_alpha = getattr(self.model, "last_surface_guided_alpha", None)
            if surface_alpha is not None:
                surface_alpha_value = surface_alpha.float().mean()

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
                if supcon_z.dim() == 3 and supcon_z.shape[:2] == labels.shape:
                    supervised = mask > 0
                    supcon_input = supcon_z[supervised]
                    supcon_labels = labels.clamp(min=0)[supervised].long()
                    if domain_ids is not None and bool(getattr(self.c.tra, "supcon_cross_frag", False)):
                        cell_domains = domain_ids.unsqueeze(1).expand_as(labels)[supervised]
                    else:
                        cell_domains = None
                    supcon_loss_value = supcon_loss(
                        supcon_input,
                        supcon_labels,
                        temp=supcon_temp,
                        domain_ids=cell_domains,
                    )
                else:
                    cross_frag_ids = domain_ids if bool(getattr(self.c.tra, "supcon_cross_frag", False)) else None
                    supcon_loss_value = supcon_loss(
                        supcon_z,
                        sample_pos.long(),
                        temp=supcon_temp,
                        domain_ids=cross_frag_ids,
                    )
                weighted_supcon_loss_value = supcon_lambda * supcon_loss_value
                loss = loss + weighted_supcon_loss_value

            # entropy maximization on unlabeled (validation) tiles: rewards uncertainty outside
            # the labeled region, attacking the "predict not-ink everywhere" fixed point
            entropy_lambda = float(getattr(self.c.tra, "entropy_min_lambda", 0.0))
            if entropy_lambda > 0 and unlabeled_images is not None:
                cap = int(getattr(self.c.tra, "entropy_min_batch_size", 8))
                u_imgs = unlabeled_images[:cap].to(self.c.device)
                u_out = self.model(u_imgs)
                if u_out.dim() == 4:
                    u_out = u_out.flatten(1).max(dim=1, keepdim=True).values
                p = torch.sigmoid(u_out).clamp(1e-6, 1.0 - 1e-6)
                h = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
                loss = loss - entropy_lambda * h  # subtract to maximize H on unlabeled

        self.scaler.scale(loss).backward()
        if (
            bool(getattr(self.c.tra, "context_consistency", False))
            and context_pair is not None
            and context_active_mask is not None
            and context_active_mask.any()
        ):
            active_cpu = context_active_mask
            active_device = active_cpu.to(self.c.device, non_blocking=True)
            pair_images = context_pair[active_cpu].to(self.c.device, non_blocking=True)
            pair_offsets = target_offsets[active_device] if target_offsets is not None else None
            pair_target = torch.sigmoid(outputs[active_device]).detach()
            with autocast(self.c.device, enabled=self.c.device == "cuda"):
                self.model.eval()
                pair_outputs = self.model(pair_images, target_offsets=pair_offsets)
                self.model.train()
                pair_mask = mask[active_device]
                context_loss_value = (
                    (torch.sigmoid(pair_outputs) - pair_target).square() * pair_mask
                ).sum() / pair_mask.sum().clamp(min=1.0)
                weighted_context_loss = float(
                    getattr(self.c.tra, "context_consistency_lambda", 0.1)
                ) * context_loss_value
            self.scaler.scale(weighted_context_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        diagnostic_values = torch.stack((
            loss.detach().float(),
            raw_loss_value.detach().float(),
            dann_loss_value.detach().float(),
            spill_loss_value.detach().float(),
            surface_loss_value.detach().float(),
            surface_alpha_value.detach().float(),
            supcon_loss_value.detach().float(),
            weighted_supcon_loss_value.detach().float(),
            context_loss_value.detach().float(),
            bag_rank_loss_value.detach().float(),
            groupdro_loss_value.detach().float(),
            cvar_loss_value.detach().float(),
            dann_accuracy_value.detach().float(),
            dann_grl_value.detach().float(),
        )).cpu().tolist()
        self._last_character_objectives = tuple(diagnostic_values[8:12])
        self._last_dann_accuracy = diagnostic_values[12]
        self._last_grl_scale = diagnostic_values[13]

        if hasattr(self.model, "prototype_head") and self.model.prototype_head is not None:
            emb = getattr(self.model, "last_embedding_detached", None)
            if emb is not None:
                self.model.prototype_head.update(emb, sample_pos.view(-1, 1))

        scores = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        label_values = labels.clamp(min=0).detach().cpu().numpy().flatten().astype(int)
        if character_ids is not None:
            character_values = character_ids.view(B, -1).cpu().numpy().flatten().astype(np.int64)
        else:
            character_values = np.zeros_like(label_values, dtype=np.int64)
        if labels.shape[1] > 1:  # multitile: drop out-of-mask sub-tiles from the metric arrays
            keep = (mask > 0).detach().cpu().numpy().flatten()
            scores = scores[keep]
            label_values = label_values[keep]
            character_values = character_values[keep]
        return (
            scores,
            label_values,
            character_values,
            *diagnostic_values[:8],
        )

    def _ins_hard_samples(
        self,
        images,
        labels,
        mask,
        hard_injector,
        remaining_batches: int,
    ) -> tuple[int, list[int]]:
        if not hard_injector or not hard_injector.has_next():
            return 0, []

        if remaining_batches <= 0:
            inject_n = 0
        else:
            inject_n = min(images.size(0), (hard_injector.remaining() + remaining_batches - 1) // remaining_batches)

        injected = 0
        injected_indices = []
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
                injected_indices.append(int(replace_index))
            return injected, injected_indices

    def train_epoch(self, hard_injector, epoch: int = 0):
        self.model.train()
        loss_total = 0.0
        raw_loss_total = 0.0
        dann_loss_total = 0.0
        spill_loss_total = 0.0
        surface_loss_total = 0.0
        surface_alpha_total = 0.0
        supcon_loss_total = 0.0
        weighted_supcon_loss_total = 0.0
        labels = []
        preds = []
        scores = []
        character_ids_all = []
        total_injected = 0
        dann_accuracy_total = 0.0
        grl_scale_total = 0.0
        context_loss_total = 0.0
        bag_rank_loss_total = 0.0
        groupdro_loss_total = 0.0
        cvar_loss_total = 0.0

        entropy_lambda = float(getattr(self.c.tra, "entropy_min_lambda", 0.0))
        u_iter = iter(self.valid_loader) if entropy_lambda > 0 else None

        for batch_index, batch in enumerate(
            tqdm(self.train_loader, desc="Training", mininterval=5, miniters=1, file=sys.stderr)
        ):
            images, batch_labels, mask = batch[:3]
            with_domain = bool(getattr(self.c.tra, "dann", False)) or bool(
                getattr(self.c.tra, "supcon_cross_frag", False)
            )
            optional_index = 3
            domain_ids = batch[optional_index] if with_domain and len(batch) > optional_index else None
            optional_index += int(with_domain)
            batch_character_ids = (
                batch[optional_index]
                if bool(getattr(self.c.tra, "character_macro_metrics", False))
                and len(batch) > optional_index
                else None
            )
            optional_index += int(bool(getattr(self.c.tra, "character_macro_metrics", False)))
            batch_target_offsets = (
                batch[optional_index]
                if bool(getattr(self.c.data, "target_aware_ctx_jitter", False))
                and len(batch) > optional_index
                else None
            )
            optional_index += int(bool(getattr(self.c.data, "target_aware_ctx_jitter", False)))
            if bool(getattr(self.c.model, "new_learned_surface", False)):
                batch_surface_depth = batch[optional_index]
                batch_surface_confidence = batch[optional_index + 1]
                optional_index += 2
            else:
                batch_surface_depth = None
                batch_surface_confidence = None
            if bool(getattr(self.c.tra, "context_consistency", False)):
                batch_context_pair = batch[optional_index]
                batch_context_pair_active = batch[optional_index + 1]
            else:
                batch_context_pair = None
                batch_context_pair_active = None

            u_imgs = None
            if u_iter is not None:
                try:
                    u_batch = next(u_iter)
                except StopIteration:
                    u_iter = iter(self.valid_loader)
                    u_batch = next(u_iter)
                u_imgs = u_batch[0]
            injected, injected_indices = self._ins_hard_samples(
                images,
                batch_labels,
                mask,
                hard_injector,
                len(self.train_loader) - batch_index,
            )
            total_injected += injected
            if batch_surface_confidence is not None and injected_indices:
                batch_surface_confidence[injected_indices] = 0
            (
                batch_scores,
                batch_labels_out,
                batch_character_ids_out,
                batch_loss,
                batch_raw_loss,
                batch_dann_loss,
                batch_spill_loss,
                batch_surface_loss,
                batch_surface_alpha,
                batch_supcon_loss,
                batch_weighted_supcon_loss,
            ) = self._train_batch(
                images,
                batch_labels,
                mask,
                domain_ids=domain_ids,
                character_ids=batch_character_ids,
                target_offsets=batch_target_offsets,
                epoch=epoch,
                unlabeled_images=u_imgs,
                surface_depth=batch_surface_depth,
                surface_confidence=batch_surface_confidence,
                context_pair=batch_context_pair,
                context_pair_active=batch_context_pair_active,
            )
            if batch_scores.size == 0:
                continue
            loss_total += batch_loss
            raw_loss_total += batch_raw_loss
            dann_loss_total += batch_dann_loss
            dann_accuracy_total += self._last_dann_accuracy
            grl_scale_total += self._last_grl_scale
            spill_loss_total += batch_spill_loss
            surface_loss_total += batch_surface_loss
            surface_alpha_total += batch_surface_alpha
            supcon_loss_total += batch_supcon_loss
            weighted_supcon_loss_total += batch_weighted_supcon_loss
            labels.extend(batch_labels_out)
            preds.extend((batch_scores > 0.5).astype(int))
            scores.extend(batch_scores)
            character_ids_all.extend(batch_character_ids_out)
            context_value, bag_value, groupdro_value, cvar_value = self._last_character_objectives
            context_loss_total += context_value
            bag_rank_loss_total += bag_value
            groupdro_loss_total += groupdro_value
            cvar_loss_total += cvar_value

        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        if bool(getattr(self.c.tra, "character_macro_metrics", False)):
            metrics.update(calculate_character_metrics(
                labels,
                scores,
                character_ids_all,
                score_threshold=float(getattr(self.c.tra, "character_score_threshold", 0.5)),
                recall_target=float(getattr(self.c.tra, "character_recall_target", 0.5)),
                max_ring_fpr=float(getattr(self.c.tra, "character_max_ring_fpr", 0.1)),
            ))
        metrics["loss"] = loss_total / len(self.train_loader)
        metrics["raw_loss"] = raw_loss_total / len(self.train_loader)
        metrics["dann_loss"] = dann_loss_total / len(self.train_loader)
        metrics["dann_accuracy"] = dann_accuracy_total / len(self.train_loader)
        metrics["dann_grl_scale"] = grl_scale_total / len(self.train_loader)
        metrics["spill_loss"] = spill_loss_total / len(self.train_loader)
        metrics["surface_loss"] = surface_loss_total / len(self.train_loader)
        metrics["surface_alpha"] = surface_alpha_total / len(self.train_loader)
        metrics["supcon_loss"] = supcon_loss_total / len(self.train_loader)
        metrics["weighted_supcon_loss"] = weighted_supcon_loss_total / len(self.train_loader)
        metrics["context_consistency_loss"] = context_loss_total / len(self.train_loader)
        metrics["character_bag_ranking_loss"] = bag_rank_loss_total / len(self.train_loader)
        metrics["character_groupdro_loss"] = groupdro_loss_total / len(self.train_loader)
        metrics["character_cvar_loss"] = cvar_loss_total / len(self.train_loader)
        metrics["scores"] = scores
        metrics["hard_injected"] = total_injected

        # explicitly release the unlabeled iterator so its worker processes terminate
        # before predict_tiles and eval figures start reading zarr (prevents IO saturation)
        if u_iter is not None:
            del u_iter

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
        processed_batches = 0
        labels = []
        preds = []
        scores = []
        character_ids_all = []

        with torch.no_grad(), autocast(self.c.device, enabled=self.c.device == "cuda"):
            for batch in tqdm(
                self.valid_loader,
                desc="Validating",
                mininterval=5,
                miniters=1,
                file=sys.stderr,
            ):
                images, batch_labels, mask = batch[:3]
                with_domain = bool(getattr(self.c.tra, "dann", False)) or bool(
                    getattr(self.c.tra, "supcon_cross_frag", False)
                )
                optional_index = 3 + int(with_domain)
                batch_character_ids = (
                    batch[optional_index]
                    if bool(getattr(self.c.tra, "character_macro_metrics", False))
                    and len(batch) > optional_index
                    else None
                )
                optional_index += int(bool(getattr(self.c.tra, "character_macro_metrics", False)))
                batch_target_offsets = (
                    batch[optional_index]
                    if bool(getattr(self.c.data, "target_aware_ctx_jitter", False))
                    and len(batch) > optional_index
                    else None
                )
                optional_index += int(bool(getattr(self.c.data, "target_aware_ctx_jitter", False)))
                if bool(getattr(self.c.model, "new_learned_surface", False)):
                    optional_index += 2
                if mask.view(mask.size(0), -1).sum() <= 0:
                    print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")
                    continue

                images = images.to(self.c.device, non_blocking=True)
                B = images.size(0)
                batch_labels = batch_labels.to(self.c.device, non_blocking=True).view(B, -1)
                mask = mask.to(self.c.device).view(B, -1)
                if mask.shape[1] == batch_labels.shape[1]:
                    mask = (mask > 0).float()                          # per-sub-tile validity (multitile)
                else:
                    mask = (mask.sum(dim=1) > 0).float().unsqueeze(1)  # single-tile window gate

                outputs = self.model(images, target_offsets=batch_target_offsets)
                if outputs.dim() == 4:
                    outputs = outputs.flatten(1).max(dim=1, keepdim=True).values

                raw_loss = self.criterion(outputs, batch_labels)
                loss_total += ((raw_loss * mask).sum() / mask.sum()).item()
                processed_batches += 1

                batch_scores = torch.sigmoid(outputs).cpu().numpy().flatten()
                batch_lab = batch_labels.cpu().numpy().flatten().astype(int)
                if batch_character_ids is not None:
                    batch_chars = batch_character_ids.view(B, -1).numpy().flatten().astype(np.int64)
                else:
                    batch_chars = np.zeros_like(batch_lab, dtype=np.int64)
                if batch_labels.shape[1] > 1:  # multitile: exclude out-of-mask sub-tiles
                    keep = (mask > 0).cpu().numpy().flatten()
                    batch_scores = batch_scores[keep]
                    batch_lab = batch_lab[keep]
                    batch_chars = batch_chars[keep]
                labels.extend(batch_lab)
                preds.extend((batch_scores > 0.5).astype(int))
                scores.extend(batch_scores)
                character_ids_all.extend(batch_chars)

        metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(scores))
        if bool(getattr(self.c.tra, "character_macro_metrics", False)):
            metrics.update(calculate_character_metrics(
                labels,
                scores,
                character_ids_all,
                score_threshold=float(getattr(self.c.tra, "character_score_threshold", 0.5)),
                recall_target=float(getattr(self.c.tra, "character_recall_target", 0.5)),
                max_ring_fpr=float(getattr(self.c.tra, "character_max_ring_fpr", 0.1)),
            ))
        metrics["loss"] = loss_total / max(1, processed_batches)
        metrics["scores"] = scores
        return metrics

    def _periodic_model_save(self, epoch: int, val_metrics: dict) -> None:
        character_metric = str(getattr(
            self.c.tra,
            "character_checkpoint_metric",
            "character_ap_macro",
        ))
        character_score = val_metrics.get(character_metric)
        if character_score is not None and character_score > self.best_val_character:
            self.best_val_character = float(character_score)
            final_path = getattr(self.c, "save_final", None)
            if final_path:
                root, ext = os.path.splitext(final_path)
                if root.endswith("_final"):
                    root = root[:-len("_final")]
                character_path = f"{root}_best_character{ext or '.pth'}"
            else:
                character_path = f"{self.c.model_dir}/best_model_character.pth"
            save_model(self.model, character_path)
            print(
                f"New best character model saved! "
                f"Val {character_metric}: {self.best_val_character:.4f}"
            )

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

    def _shutdown_data_workers(self) -> None:
        """release persistent loader workers before full-scroll figure inference."""
        for loader in (self.train_loader, self.valid_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is None:
                continue
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            loader._iterator = None

    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, time_elapsed: float) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"[METRICS] epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f}"
            + (
                f" val_character_success={val_metrics['character_success_fraction']:.4f}"
                if "character_success_fraction" in val_metrics else ""
            )
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
        for key, tag in (
            ("context_consistency_loss", "Aux/ContextConsistency"),
            ("character_bag_ranking_loss", "Aux/CharacterBagRanking"),
            ("character_groupdro_loss", "Aux/CharacterGroupDRO"),
            ("character_cvar_loss", "Aux/CharacterCVaR"),
        ):
            if key in train_metrics:
                self.vis.writer.add_scalar(tag, train_metrics[key], epoch)
        if bool(getattr(self.c.tra, "dann", False)):
            self.vis.writer.add_scalar("DANN/DomainAccuracy", train_metrics["dann_accuracy"], epoch)
            self.vis.writer.add_scalar("DANN/GRLScale", train_metrics["dann_grl_scale"], epoch)
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

            self.train_dataset.apply_transforms = bool(
                epoch >= int(getattr(self.c.tra, "aug_start_epoch", 5)) and self.c.dl.data_aug
            )
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
            guard_epoch = int(getattr(self.c.tra, "sanity_guard_epoch", 0))
            if guard_epoch > 0 and (epoch + 1) == guard_epoch:
                character_ap = float(val_metrics.get("character_ap_macro", 0.0))
                specificity = float(val_metrics.get("specificity", 0.0))
                min_ap = float(getattr(self.c.tra, "sanity_min_character_ap", 0.0))
                min_specificity = float(getattr(self.c.tra, "sanity_min_specificity", 0.0))
                print(
                    f"[sanity] epoch={epoch + 1} character_ap={character_ap:.4f}"
                    f" specificity={specificity:.4f}"
                    f" required=({min_ap:.4f},{min_specificity:.4f})"
                )
                if character_ap < min_ap or specificity < min_specificity:
                    raise RuntimeError(
                        "sanity guard failed: early validation is below historical tolerance"
                    )
            self.scheduler.step(val_metrics["loss"])
            self._periodic_model_save(epoch, val_metrics)
            figure_due = any([
                (epoch + 1) % self.c.tra.eval_int == 0,
                (epoch + 1) % self.c.tra.test_int == 0,
                (epoch + 1) % self.c.tra.probe_int == 0,
                bool(getattr(self.c.tra, "test_on_final", False))
                and (epoch + 1) == self.c.tra.n_epochs,
            ])
            if figure_due:
                self._shutdown_data_workers()
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