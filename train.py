import math
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from utils.config import Config
from utils.dataloader import DataManager, MultiScrollIterableDataset, get_dataloaders, DotPositiveDataset, imread_gray, get_tile_pos_weight, needs_domain_ids
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


def _flatten_domain_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    domain_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """select supervised decoder embeddings and replicate sample domains per target."""
    valid = mask > 0
    if embeddings.dim() == 3 and embeddings.shape[:2] == labels.shape:
        domains = domain_ids.unsqueeze(1).expand_as(labels)
        return embeddings[valid], labels[valid].long(), domains[valid].long()
    return embeddings, labels[:, 0].long(), domain_ids.long()


def class_conditional_prototype_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """align per-domain class means and keep global class prototypes separated."""
    terms = []
    class_means = []
    for label in (0, 1):
        selected_class = labels == label
        if not selected_class.any():
            continue
        domain_means = []
        for domain in torch.unique(domain_ids[selected_class]):
            selected = selected_class & (domain_ids == domain)
            if selected.any():
                domain_means.append(F.normalize(embeddings[selected].mean(dim=0), dim=0))
        global_mean = F.normalize(torch.stack(domain_means).mean(dim=0), dim=0)
        class_means.append(global_mean)
        terms.extend(1.0 - torch.dot(domain_mean, global_mean) for domain_mean in domain_means)
    loss = torch.stack(terms).mean() if terms else embeddings.new_zeros(())
    if len(class_means) == 2:
        distance = 1.0 - torch.dot(class_means[0], class_means[1])
        loss = loss + F.relu(embeddings.new_tensor(float(margin)) - distance)
    return loss


def class_conditional_coral_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    mean_weight: float,
) -> torch.Tensor:
    """align class-conditional means and covariance matrices across physical domains."""
    terms = []
    for label in (0, 1):
        groups = []
        for domain in torch.unique(domain_ids[labels == label]):
            selected = embeddings[(labels == label) & (domain_ids == domain)]
            if selected.shape[0] >= 2:
                centered = selected - selected.mean(dim=0, keepdim=True)
                covariance = centered.T @ centered / float(selected.shape[0] - 1)
                groups.append((selected.mean(dim=0), covariance))
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                mean_loss = (groups[left][0] - groups[right][0]).square().mean()
                covariance_loss = (groups[left][1] - groups[right][1]).square().mean()
                terms.append(float(mean_weight) * mean_loss + covariance_loss)
    return torch.stack(terms).mean() if terms else embeddings.new_zeros(())


def _per_domain_pr_auc(labels, scores, domains) -> dict[int, float]:
    """calculate one ranking metric for every represented physical domain."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    domains = np.asarray(domains).reshape(-1)
    values = {}
    for domain in np.unique(domains):
        selected = domains == domain
        values[int(domain)] = float(
            calculate_metrics(
                labels[selected],
                (scores[selected] > 0.5).astype(int),
                scores[selected],
            )["pr_auc"]
        )
    return values


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


def domain_vrex_loss(
    domain_losses: list[torch.Tensor],
    penalty_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """average physical-domain risk plus a variance-equality penalty."""
    if not domain_losses:
        return None, None
    risks = torch.stack(domain_losses)
    penalty = risks.var(unbiased=False)
    return risks.mean() + float(penalty_weight) * penalty, penalty


def domain_cvar_loss(
    domain_losses: list[torch.Tensor],
    alpha: float,
) -> tuple[torch.Tensor | None, int]:
    """mean risk over the worst alpha fraction of physical domains in a batch."""
    if not domain_losses:
        return None, 0
    risks = torch.stack(domain_losses)
    tail_count = max(
        1,
        int(math.ceil(risks.numel() * min(max(float(alpha), 1e-6), 1.0))),
    )
    return torch.topk(risks, tail_count).values.mean(), tail_count


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


def _physical_domain_group_losses(
    per_target_loss: torch.Tensor,
    mask: torch.Tensor,
    domain_ids: torch.Tensor,
) -> tuple[list[int], list[torch.Tensor]]:
    """reduce supervised target losses separately for each physical scroll."""
    group_ids = []
    group_losses = []
    for domain_id in torch.unique(domain_ids):
        selected = (domain_ids == domain_id).unsqueeze(1) & (mask > 0)
        if selected.any():
            group_ids.append(int(domain_id.item()))
            group_losses.append(per_target_loss[selected].mean())
    return group_ids, group_losses


def _connected_domain_clusters(
    domain_ids: list[int],
    similarities: dict[tuple[int, int], float],
    threshold: float,
) -> list[list[int]]:
    """greedy complete-link clusters whose members are mutually compatible."""
    remaining = set(domain_ids)
    clusters = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cluster = [seed]
        for candidate in sorted(tuple(remaining)):
            if all(
                similarities.get(tuple(sorted((candidate, member))), -1.0) >= threshold
                for member in cluster
            ):
                cluster.append(candidate)
                remaining.remove(candidate)
        clusters.append(sorted(cluster))
    return clusters


def clam_instance_loss(
    instance_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """CLAM-lite: constrain representative positive/negative instances per bag."""
    k = min(max(1, int(k)), int(instance_logits.shape[-1]))
    top = torch.topk(instance_logits, k, dim=-1).values
    bottom = torch.topk(instance_logits, k, dim=-1, largest=False).values
    positive = labels > 0.5
    valid = mask > 0
    positive_loss = 0.5 * (
        F.softplus(-top).mean(dim=-1) + F.softplus(bottom).mean(dim=-1)
    )
    negative_loss = F.softplus(top).mean(dim=-1)
    per_bag = torch.where(positive, positive_loss, negative_loss)
    return (per_bag * valid).sum() / valid.sum().clamp(min=1.0)


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
        domain_gradient_mode = str(
            getattr(config.tra, "domain_gradient_mode", "") or ""
        )
        valid_domain_gradient_modes = {
            "",
            "cluster_balance",
            "cluster_worst",
            "conflict_weighted",
        }
        if domain_gradient_mode not in valid_domain_gradient_modes:
            raise ValueError(
                f"unknown domain_gradient_mode={domain_gradient_mode!r}; "
                f"valid={sorted(valid_domain_gradient_modes)}"
            )
        domain_robust_modes = [
            bool(domain_gradient_mode),
            bool(getattr(config.tra, "physical_domain_groupdro", False)),
            bool(getattr(config.tra, "domain_vrex", False)),
            bool(getattr(config.tra, "domain_cvar", False)),
            bool(getattr(config.tra, "pcgrad", False)),
        ]
        if sum(domain_robust_modes) > 1:
            raise ValueError(
                "domain-gradient, physical GroupDRO, V-REx, domain CVaR, and PCGrad "
                "are mutually exclusive"
            )
        if not 0.0 <= float(getattr(config.tra, "domain_gradient_blend", 1.0)) <= 1.0:
            raise ValueError("domain_gradient_blend must be in [0, 1]")
        if not 0.0 < float(getattr(config.tra, "domain_cvar_alpha", 0.25)) <= 1.0:
            raise ValueError("domain_cvar_alpha must be in (0, 1]")
        if not 0.0 <= float(getattr(config.tra, "model_ema_decay", 0.999)) < 1.0:
            raise ValueError("model_ema_decay must be in [0, 1)")
        if domain_gradient_mode and any([
            bool(getattr(config.tra, "mldg", False)),
            bool(getattr(config.tra, "character_groupdro", False)),
            bool(getattr(config.tra, "character_cvar", False)),
        ]):
            raise ValueError(
                "domain-gradient objectives cannot be combined with MLDG, GroupDRO, or CVaR"
            )
        if bool(getattr(config.tra, "mldg", False)) and any([
            bool(getattr(config.tra, "physical_domain_groupdro", False)),
            bool(getattr(config.tra, "character_groupdro", False)),
            bool(getattr(config.tra, "character_cvar", False)),
            bool(getattr(config.model, "sagnet", False)),
            bool(getattr(config.tra, "mae_reconstruction", False)),
        ]):
            raise ValueError(
                "MLDG replaces the ordinary robust/auxiliary objective path and cannot be "
                "combined with GroupDRO, CVaR, SagNet, or MAE continuation"
            )
        if float(getattr(config.tra, "sam_rho", 0.0)) > 0 and any([
            bool(getattr(config.tra, "tta_consistency", False)),
            bool(getattr(config.tra, "context_consistency", False)),
            bool(getattr(config.tra, "depth_view_consistency", False)),
            bool(getattr(config.tra, "spill_reduction", False)),
            bool(getattr(config.tra, "spill_prob", False)),
            bool(getattr(config.tra, "spill_entropy", False)),
        ]):
            raise ValueError("SAM currently supports primary, DANN, SupCon, CLAM, and ELR losses only")
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
        self._ema_state = (
            {
                name: value.detach().clone()
                for name, value in self.model.state_dict().items()
            }
            if bool(getattr(self.c.tra, "model_ema", False)) else None
        )
        self.hard_manager = HardMiningManager(self.c.hm.dir)
        self.hard_samples: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.best_val_character = -1.0
        self._character_groupdro_log_weights: dict[int, float] = {}
        self._physical_domain_groupdro_log_weights: dict[int, float] = {}
        self._domain_gradient_ema: dict[tuple[int, int], float] = {}
        self._last_character_objectives = (0.0, 0.0, 0.0, 0.0)
        self._last_depth_consistency = 0.0
        self._last_clam_loss = 0.0
        self._last_elr_loss = 0.0
        self._last_physical_domain_groupdro_loss = 0.0
        self._last_deep_supervision_loss = 0.0
        self._last_dual_scale_gate = 0.0
        self._last_domain_gradient_cosines: dict[tuple[int, int], float] = {}
        self._last_domain_gradient_clusters: list[list[int]] = []
        self._elr_targets: dict[int, float] = {}
        self._last_dann_accuracy = 0.0
        self._last_grl_scale = 0.0
        self._last_metric_domains = np.empty(0, dtype=np.int64)
        self._last_dg_losses = {
            "prototype_align_loss": 0.0,
            "coral_align_loss": 0.0,
            "cdan_loss": 0.0,
            "sagnet_loss": 0.0,
            "mae_reconstruction_loss": 0.0,
            "mldg_meta_train_loss": 0.0,
            "mldg_meta_test_loss": 0.0,
            "domain_vrex_loss": 0.0,
            "domain_cvar_loss": 0.0,
            "depth_shift_aux_loss": 0.0,
            "mae_anchor_loss": 0.0,
        }
        self._closed = False

    def _print_config(self) -> None:
        print("--- Configuration ---")
        for field in self.c.__dataclass_fields__:
            print(f"{field}: {getattr(self.c, field)}")
        print("---------------------")

    def _setup_data(self):
        print("Creating datasets...")
        start_time = time.time()

        scroll_ids = [int(scroll.scroll_id) for scroll in self.c.data.scrolls]
        scroll_dict = getattr(self.c.data, "train_scroll_dict", None)
        sampling_groups = None
        sampling_weights = None
        domain_by_scroll = {scroll_id: index for index, scroll_id in enumerate(scroll_ids)}
        if scroll_dict:
            group_names = list(scroll_dict)
            group_ids = [[int(scroll_id) for scroll_id in scroll_dict[name]] for name in group_names]
            flattened = [scroll_id for group in group_ids for scroll_id in group]
            if len(flattened) != len(set(flattened)):
                raise ValueError("train_scroll_dict assigns at least one segment more than once")
            if set(flattened) != set(scroll_ids):
                missing = sorted(set(scroll_ids) - set(flattened))
                extra = sorted(set(flattened) - set(scroll_ids))
                raise ValueError(
                    f"train_scroll_dict must exactly cover configured scrolls; "
                    f"missing={missing} extra={extra}"
                )
            requested_weights = getattr(self.c.data, "train_scroll_weights", None)
            sampling_weights = (
                [1] * len(group_names)
                if requested_weights is None else [int(value) for value in requested_weights]
            )
            if len(sampling_weights) != len(group_names) or any(value <= 0 for value in sampling_weights):
                raise ValueError("train_scroll_weights must provide one positive integer per dictionary key")
            domain_by_scroll = {
                scroll_id: domain_id
                for domain_id, group in enumerate(group_ids)
                for scroll_id in group
            }
            index_by_scroll = {scroll_id: index for index, scroll_id in enumerate(scroll_ids)}
            sampling_groups = [
                [index_by_scroll[scroll_id] for scroll_id in group]
                for group in group_ids
            ]
            print(
                f"[multi-scroll] physical groups={dict(zip(group_names, group_ids))} "
                f"weights={sampling_weights}"
            )
        n_domains = len(set(domain_by_scroll.values()))
        if bool(getattr(self.c.tra, "dann", False)):
            self.c.tra.dann_n_domains = n_domains
        self._scroll_ids = scroll_ids
        self._scroll_train_sets = None

        if len(scroll_ids) > 1:
            balance_scrolls = bool(getattr(self.c.data, "character_balance_scrolls", False))
            train_sets = []
            valid_sets = []
            self._scroll_dms = {}
            self._scroll_train_sets = {}
            for segment_index, scroll_id in enumerate(scroll_ids):
                domain_id = domain_by_scroll[scroll_id]
                data_manager = DataManager(
                    self.c,
                    scroll_id=scroll_id,
                    domain_id=domain_id,
                    character_namespace=segment_index,
                )
                train_set, valid_set = data_manager.get_datasets()
                if bool(getattr(self.c.data, "selective_chunk_preload", False)):
                    data_manager.enable_selective_chunk_preload(train_set, valid_set)
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
                sampling_groups=sampling_groups,
                sampling_weights=sampling_weights,
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
        if bool(getattr(self.c.data, "selective_chunk_preload", False)):
            data_manager.enable_selective_chunk_preload(train_set, valid_set)
        train_loader, valid_loader = get_dataloaders(train_set, valid_set, self.c)
        self._scroll_dms = {scroll_ids[0]: data_manager}
        self._train_children = [train_set]
        print(f"Data setup done in {time.time() - start_time:.2f}s")
        return train_set, train_loader, valid_loader

    def _setup_model_optim(self):
        print(f"Creating model and loss... l1 lambda {self.c.tra.l1_lambda}... ", end="")
        start_time = time.time()

        model, params = create_model(self.c)
        self._mae_anchor_reference: dict[str, torch.Tensor] = {}
        init_path = getattr(self.c, "init_weights", None)
        if init_path:
            state_dict = torch.load(init_path, map_location=self.c.device, weights_only=True)
            model_state = model.state_dict()
            compatible = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped = len(state_dict) - len(compatible)
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            if bool(getattr(self.c.model, "require_architecture_init", False)):
                active_prefixes = []
                if bool(getattr(self.c.model, "fiber_coordinate_branch", False)):
                    active_prefixes.append("fiber_coordinate_input.")
                if bool(getattr(self.c.model, "early_2d_unet", False)):
                    active_prefixes.extend(("early_depth_attn.", "early_depth_fuse.", "early2d_"))
                if bool(getattr(self.c.model, "mid_2d_unet", False)):
                    active_prefixes.extend(("mid_depth_attn.", "mid_depth_fuse.", "mid_skip1_fuse.", "mid2d_"))
                if bool(getattr(self.c.model, "gated_stems", False)):
                    active_prefixes.append("gated_cue_stem.")
                if bool(getattr(self.c.model, "overlapping_depth_windows", False)):
                    active_prefixes.extend(("overlap_bottleneck_fuse.", "overlap_decoded_fuse."))
                if bool(getattr(self.c.model, "divided_attention", False)):
                    active_prefixes.append("divided_attention.")
                if bool(getattr(self.c.model, "mednext_adapters", False)):
                    active_prefixes.extend(
                        ("mednext1.", "mednext2.", "mednext3.", "mednext_bottleneck.")
                    )
                if bool(getattr(self.c.model, "factorized_2plus1d", False)):
                    active_prefixes.extend(("enc1.", "enc2.", "enc3.", "bottleneck.", "dec3.", "dec2.", "dec1."))
                norm_mode = str(getattr(self.c.model, "norm_mode", "auto"))
                if norm_mode in ("instance", "ibn_full", "batch"):
                    active_prefixes.extend(("enc1.", "enc2.", "enc3.", "bottleneck.", "dec3.", "dec2.", "dec1."))
                required_architecture_keys = {
                    key
                    for key in model_state
                    if any(key.startswith(prefix) for prefix in active_prefixes)
                }
                missing_architecture_keys = sorted(required_architecture_keys - compatible.keys())
                if missing_architecture_keys:
                    raise RuntimeError(
                        f"[init-weights] checkpoint {init_path} is missing or incompatible with "
                        f"{len(missing_architecture_keys)} required architecture tensors: "
                        f"{missing_architecture_keys[:10]}"
                    )
            print(
                f"[init-weights] loaded {len(compatible)}/{len(state_dict)} tensors from {init_path} "
                f"(shape-skipped={skipped} missing={len(missing)} unexpected={len(unexpected)})"
            )
            anchor_lambda = float(getattr(self.c.tra, "mae_anchor_lambda", 0.0))
            if anchor_lambda > 0:
                anchor_prefixes = (
                    "enc1.",
                    "enc2.",
                    "gated_cue_stem.",
                    "mid_depth_attn.",
                    "mid_depth_fuse.",
                    "mid_skip1_fuse.",
                    "mid2d_enc3.",
                    "mid2d_bottleneck.",
                    "mid2d_up3.",
                    "mid2d_dec3.",
                    "mid2d_up2.",
                    "mid2d_dec2.",
                    "mid2d_up1.",
                    "mid2d_dec1.",
                )
                self._mae_anchor_reference = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if name in compatible and name.startswith(anchor_prefixes)
                }
                if not self._mae_anchor_reference:
                    raise RuntimeError("MAE anchoring found no loaded feature parameters")
                anchored = sum(value.numel() for value in self._mae_anchor_reference.values())
                print(
                    f"[mae-anchor] lambda={anchor_lambda:g} "
                    f"parameters={anchored:,} tensors={len(self._mae_anchor_reference)}"
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

    def _update_encoder_optimization(self, epoch: int) -> None:
        """keep the MAE encoder fixed initially, then enable its scaled learning rate."""
        freeze_epochs = int(getattr(self.c.tra, "encoder_freeze_epochs", 0))
        encoder_groups = [
            group for group in self.optimizer.param_groups
            if group.get("group_name") == "encoder"
        ]
        if not encoder_groups:
            return

        task_lr = next(
            group["lr"] for group in self.optimizer.param_groups
            if group.get("group_name") == "task"
        )
        enabled = epoch >= freeze_epochs
        for group in encoder_groups:
            was_enabled = bool(group.get("lr_enabled", True))
            group["lr_enabled"] = enabled
            group["lr"] = task_lr * float(group.get("lr_scale", 1.0)) if enabled else 0.0
            if enabled and not was_enabled:
                for parameter in group["params"]:
                    self.optimizer.state.pop(parameter, None)
            if enabled != was_enabled or epoch == 0:
                state = "enabled" if enabled else "frozen"
                print(f"[mae-preserve] encoder {state}: lr={group['lr']:.3e}")

    def _update_model_ema(self, epoch: int) -> None:
        if self._ema_state is None:
            return
        decay = float(getattr(self.c.tra, "model_ema_decay", 0.999))
        start_epoch = int(getattr(self.c.tra, "model_ema_start_epoch", 0))
        current = self.model.state_dict()
        with torch.no_grad():
            for name, value in current.items():
                if epoch < start_epoch or not value.is_floating_point():
                    self._ema_state[name].copy_(value)
                else:
                    self._ema_state[name].lerp_(value, 1.0 - decay)

    @contextmanager
    def _ema_weights(self):
        if self._ema_state is None:
            yield
            return
        current = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        self.model.load_state_dict(self._ema_state, strict=True)
        try:
            yield
        finally:
            self.model.load_state_dict(current, strict=True)

    def _pcgrad_backward(
        self,
        loss: torch.Tensor,
        primary_loss: torch.Tensor,
        domain_losses: list[torch.Tensor],
    ) -> None:
        if len(domain_losses) < 2:
            self.scaler.scale(loss).backward()
            return
        parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        auxiliary_loss = loss - primary_loss
        self.scaler.scale(auxiliary_loss).backward(retain_graph=True)
        task_gradients = []
        for index, domain_loss in enumerate(domain_losses):
            gradients = torch.autograd.grad(
                self.scaler.scale(domain_loss),
                parameters,
                retain_graph=index + 1 < len(domain_losses),
                allow_unused=True,
            )
            task_gradients.append([
                torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)
            ])
        projected = []
        for task_index, gradients in enumerate(task_gradients):
            adjusted = [gradient.clone() for gradient in gradients]
            order = torch.randperm(len(task_gradients), device=loss.device).tolist()
            for other_index in order:
                if other_index == task_index:
                    continue
                other = task_gradients[other_index]
                dot = sum((left * right).sum() for left, right in zip(adjusted, other))
                norm = sum((gradient * gradient).sum() for gradient in other).clamp(min=1e-12)
                if dot < 0:
                    scale = dot / norm
                    adjusted = [
                        left - scale * right
                        for left, right in zip(adjusted, other)
                    ]
            projected.append(adjusted)
        with torch.no_grad():
            for parameter_index, parameter in enumerate(parameters):
                gradient = torch.stack([
                    task[parameter_index] for task in projected
                ]).mean(dim=0)
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)

    def _init_visualizers(self) -> None:
        scroll_ids = self._scroll_ids
        visualizer_ids = list(
            dict.fromkeys(
                int(scroll_id)
                for scroll_id in (
                    getattr(self.c.data, "vis_scroll_ids", None) or scroll_ids
                )
            )
        )
        tra = self.c.tra
        will_test = (tra.test_int <= tra.n_epochs) or bool(getattr(tra, "test_on_final", False))
        if not will_test:
            print(
                f"[test] test_int={tra.test_int} > n_epochs={tra.n_epochs} and "
                f"test_on_final={getattr(tra, 'test_on_final', False)} -> skipping test-frag load"
            )

        if len(scroll_ids) > 1:
            self.vis = TensorboardVisualizer(self.c, mode="metrics")
            self.vis.writer.add_scalar("Run/Initializing", 1.0, 0)
            self.vis.writer.flush()
            self.scroll_vis = {}
            for index, scroll_id in enumerate(visualizer_ids):
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

    def _elr_regularizer(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        character_ids: torch.Tensor | None,
        epoch: int,
        update_targets: bool = True,
        temporal_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not bool(getattr(self.c.tra, "elr", False)):
            return outputs.new_zeros(()), None
        start_epoch = int(getattr(self.c.tra, "elr_start_epoch", 5))
        if epoch + 1 < start_epoch or character_ids is None:
            return outputs.new_zeros(()), None
        probabilities = torch.sigmoid(outputs)
        valid = (mask > 0) & (character_ids > 0)
        if not valid.any():
            return outputs.new_zeros(()), None
        if temporal_override is None:
            temporal = probabilities.detach().clone()
            beta = float(getattr(self.c.tra, "elr_beta", 0.7))
            with torch.no_grad():
                positive = labels > 0.5
                keys = character_ids.long() * 2 + positive.long()
                unique_keys = torch.unique(keys[valid])
                current_values = torch.stack([
                    probabilities.detach()[valid & (keys == key)].mean()
                    for key in unique_keys
                ]).cpu().tolist()
                key_values = unique_keys.cpu().tolist()
                for key, current in zip(key_values, current_values):
                    selected = valid & (keys == key)
                    previous = self._elr_targets.get(int(key), float(current))
                    updated = beta * previous + (1.0 - beta) * current
                    if update_targets:
                        self._elr_targets[int(key)] = updated
                    temporal[selected] = updated
        else:
            temporal = temporal_override
        agreement = probabilities * temporal + (1.0 - probabilities) * (1.0 - temporal)
        per_target = torch.log((1.0 - agreement).clamp(min=1e-4))
        loss = (per_target * valid).sum() / valid.sum().clamp(min=1.0)
        return loss, temporal.detach()

    def _sam_objective(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        sample_pos: torch.Tensor,
        domain_ids: torch.Tensor | None,
        character_ids: torch.Tensor | None,
        target_offsets: torch.Tensor | None,
        surface_depth: torch.Tensor | None,
        surface_confidence: torch.Tensor | None,
        epoch: int,
        grl_scale: float,
        elr_temporal: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs, _, domain_logits, supcon_z = self.model.forward_with_extras(
            images,
            grl_scale=grl_scale,
            target_offsets=target_offsets,
            teacher_surface_depth=surface_depth,
            teacher_surface_confidence=surface_confidence,
        )
        loss = (self.criterion(outputs, targets) * mask).sum() / mask.sum().clamp(min=1.0)
        if bool(getattr(self.c.tra, "dann", False)) and domain_logits is not None and domain_ids is not None:
            loss = loss + F.cross_entropy(domain_logits, domain_ids)
        if bool(getattr(self.c.tra, "supcon", False)) and supcon_z is not None:
            curriculum_epochs = int(getattr(self.c.tra, "supcon_curriculum_epochs", 15))
            progress = min(1.0, float(epoch) / max(1, curriculum_epochs - 1))
            supcon_lambda = (
                float(getattr(self.c.tra, "supcon_lambda_start", 0.1))
                + (
                    float(getattr(self.c.tra, "supcon_lambda_end", 0.5))
                    - float(getattr(self.c.tra, "supcon_lambda_start", 0.1))
                ) * progress
                if bool(getattr(self.c.tra, "supcon_curriculum", False))
                else float(getattr(self.c.tra, "supcon_lambda", 0.1))
            )
            supervised = mask > 0
            if supcon_z.dim() == 3 and supcon_z.shape[:2] == labels.shape:
                z_input = supcon_z[supervised]
                z_labels = labels.clamp(min=0)[supervised].long()
                z_domains = (
                    domain_ids.unsqueeze(1).expand_as(labels)[supervised]
                    if domain_ids is not None and bool(getattr(self.c.tra, "supcon_cross_frag", False))
                    else None
                )
            else:
                z_input = supcon_z
                z_labels = sample_pos.long()
                z_domains = domain_ids if bool(getattr(self.c.tra, "supcon_cross_frag", False)) else None
            loss = loss + supcon_lambda * supcon_loss(
                z_input,
                z_labels,
                temp=float(getattr(self.c.tra, "supcon_temp", 0.07)),
                domain_ids=z_domains,
            )
        if bool(getattr(self.c.tra, "clam_instance", False)):
            instances = getattr(self.model, "last_clam_instance_logits", None)
            if instances is None:
                raise RuntimeError("CLAM instance loss requires model instance logits")
            loss = loss + float(getattr(self.c.tra, "clam_instance_lambda", 0.1)) * clam_instance_loss(
                instances,
                labels,
                mask,
                int(getattr(self.c.tra, "clam_instance_k", 4)),
            )
        elr_loss, _ = self._elr_regularizer(
            outputs,
            labels,
            mask,
            character_ids,
            epoch,
            update_targets=False,
            temporal_override=elr_temporal,
        )
        loss = loss + float(getattr(self.c.tra, "elr_lambda", 0.1)) * elr_loss
        return loss

    @staticmethod
    def _slice_optional(value, selected):
        return value[selected] if value is not None else None

    def _mldg_forward_loss(
        self,
        images,
        labels,
        mask,
        target_offsets,
        surface_depth,
        surface_confidence,
    ):
        outputs = self.model(
            images,
            target_offsets=target_offsets,
            teacher_surface_depth=surface_depth,
            teacher_surface_confidence=surface_confidence,
        )
        if outputs.dim() == 4:
            outputs = outputs.flatten(1).max(dim=1, keepdim=True).values
        targets = torch.where(labels < 0, torch.zeros_like(labels), labels).float()
        loss = (self.criterion(outputs, targets) * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, outputs

    def _train_batch_mldg(
        self,
        images,
        labels,
        mask,
        domain_ids,
        character_ids,
        target_offsets,
        surface_depth,
        surface_confidence,
    ):
        """first-order MLDG: inner update on sources, outer gradient on one held domain."""
        held_domain = int(getattr(self.c.tra, "mldg_holdout_domain", -1))
        if bool(getattr(self.c.tra, "mldg_random_holdout", False)):
            represented = torch.unique(domain_ids)
            if represented.numel() < 2:
                raise RuntimeError("random-holdout MLDG requires at least two represented domains")
            held_domain = int(
                represented[
                    torch.randint(represented.numel(), (), device=represented.device)
                ].item()
            )
        held = domain_ids == held_domain
        source = ~held
        if not held.any() or not source.any():
            raise RuntimeError(
                f"MLDG batch must contain held domain {held_domain} and at least one source domain"
            )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        inner_lr = float(getattr(self.c.tra, "mldg_inner_lr", 1e-3))
        beta = float(getattr(self.c.tra, "mldg_beta", 1.0))
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.c.device, enabled=self.c.device == "cuda"):
            source_loss, source_outputs = self._mldg_forward_loss(
                images[source],
                labels[source],
                mask[source],
                self._slice_optional(target_offsets, source),
                self._slice_optional(surface_depth, source),
                self._slice_optional(surface_confidence, source),
            )
        source_grads = torch.autograd.grad(
            source_loss,
            parameters,
            allow_unused=True,
        )
        with torch.no_grad():
            for parameter, gradient in zip(parameters, source_grads):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-inner_lr)

        batch_norms = [
            module for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        training_states = [module.training for module in batch_norms]
        for module in batch_norms:
            module.eval()
        try:
            with autocast(self.c.device, enabled=self.c.device == "cuda"):
                held_loss, held_outputs = self._mldg_forward_loss(
                    images[held],
                    labels[held],
                    mask[held],
                    self._slice_optional(target_offsets, held),
                    self._slice_optional(surface_depth, held),
                    self._slice_optional(surface_confidence, held),
                )
            held_grads = torch.autograd.grad(
                held_loss,
                parameters,
                allow_unused=True,
            )
        finally:
            with torch.no_grad():
                for parameter, gradient in zip(parameters, source_grads):
                    if gradient is not None:
                        parameter.add_(gradient, alpha=inner_lr)
            for module, training in zip(batch_norms, training_states):
                module.train(training)

        for parameter, source_gradient, held_gradient in zip(
            parameters,
            source_grads,
            held_grads,
        ):
            if source_gradient is None and held_gradient is None:
                continue
            gradient = torch.zeros_like(parameter)
            if source_gradient is not None:
                gradient.add_(source_gradient)
            if held_gradient is not None:
                gradient.add_(held_gradient, alpha=beta)
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.c.tra.grad_norm)
        self.optimizer.step()

        outputs = torch.empty_like(labels, dtype=source_outputs.dtype)
        outputs[source] = source_outputs.detach()
        outputs[held] = held_outputs.detach()
        combined_loss = source_loss.detach() + beta * held_loss.detach()
        self._last_dg_losses["mldg_meta_train_loss"] = float(source_loss.detach())
        self._last_dg_losses["mldg_meta_test_loss"] = float(held_loss.detach())
        zeros = [0.0] * 9
        scores = torch.sigmoid(outputs).cpu().numpy().flatten()
        label_values = labels.clamp(min=0).cpu().numpy().flatten().astype(int)
        domains = domain_ids.unsqueeze(1).expand_as(labels).cpu().numpy().flatten()
        if labels.shape[1] > 1:
            keep = (mask > 0).cpu().numpy().flatten()
            scores = scores[keep]
            label_values = label_values[keep]
            domains = domains[keep]
        self._last_metric_domains = domains.astype(np.int64)
        if character_ids is not None:
            character_values = character_ids.cpu().numpy().flatten().astype(np.int64)
            if labels.shape[1] > 1:
                character_values = character_values[keep]
        else:
            character_values = np.zeros_like(label_values, dtype=np.int64)
        return (
            scores,
            label_values,
            character_values,
            float(combined_loss),
            float(combined_loss),
            *zeros,
        )

    def _train_batch(self, images, labels, mask, domain_ids=None, character_ids=None,
                     target_offsets=None, epoch: int = 0, unlabeled_images=None,
                     surface_depth=None, surface_confidence=None,
                     depth_shift=None,
                     context_pair=None, context_pair_active=None,
                     depth_pair=None, depth_pair_surface=None,
                     depth_pair_confidence=None, depth_pair_active=None):
        self._last_depth_consistency = 0.0
        self._last_clam_loss = 0.0
        self._last_elr_loss = 0.0
        self._last_physical_domain_groupdro_loss = 0.0
        self._last_deep_supervision_loss = 0.0
        self._last_dual_scale_gate = 0.0
        self._last_domain_gradient_cosines = {}
        self._last_domain_gradient_clusters = []
        self._last_metric_domains = np.empty(0, dtype=np.int64)
        for key in self._last_dg_losses:
            self._last_dg_losses[key] = 0.0
        if mask.reshape(mask.size(0), -1).sum().item() <= 0:
            print("[ERROR] Mask sum is zero, skipping loss calculation.")
            return (np.empty([]), np.empty([]), np.empty([]), *([0.0] * 11))
        images = images.to(self.c.device, non_blocking=True)
        surface_source = images
        if surface_depth is not None:
            surface_depth = surface_depth.to(self.c.device, non_blocking=True).float()
        if surface_confidence is not None:
            surface_confidence = surface_confidence.to(
                self.c.device,
                non_blocking=True,
            ).float()
        images = self._apply_fda(images)
        B = images.size(0)
        labels = labels.to(self.c.device, non_blocking=True).view(B, -1)  # (B,1) single or (B,K) multitile
        mask = mask.to(self.c.device, non_blocking=True).view(B, -1)
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
        if depth_shift is not None:
            depth_shift = depth_shift.to(self.c.device, non_blocking=True).view(-1)
        character_ids_device = (
            character_ids.to(self.c.device, non_blocking=True).view(B, -1)
            if character_ids is not None else None
        )
        context_active_mask = (
            context_pair_active.view(-1) > 0
            if context_pair_active is not None else None
        )
        context_loss_value = images.new_zeros(())
        depth_consistency_loss_value = images.new_zeros(())
        bag_rank_loss_value = images.new_zeros(())
        groupdro_loss_value = images.new_zeros(())
        cvar_loss_value = images.new_zeros(())
        clam_loss_value = images.new_zeros(())
        elr_loss_value = images.new_zeros(())
        prototype_loss_value = images.new_zeros(())
        coral_loss_value = images.new_zeros(())
        cdan_loss_value = images.new_zeros(())
        sagnet_loss_value = images.new_zeros(())
        mae_reconstruction_loss_value = images.new_zeros(())
        elr_temporal = None
        dann_accuracy_value = images.new_zeros(())
        dann_grl_value = images.new_zeros(())
        pcgrad_domain_losses: list[torch.Tensor] = []

        if bool(getattr(self.c.tra, "mldg", False)):
            if domain_ids is None:
                raise RuntimeError("MLDG requires physical-domain IDs")
            return self._train_batch_mldg(
                images,
                labels,
                mask,
                domain_ids,
                character_ids_device,
                target_offsets,
                surface_depth,
                surface_confidence,
            )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.c.device, enabled=self.c.device == "cuda"):
            use_extras = hasattr(self.model, "forward_with_extras") and any([
                bool(getattr(self.c.tra, "supcon", False)),
                bool(getattr(self.c.tra, "dann", False)),
                bool(getattr(self.c.tra, "prototype_align", False)),
                bool(getattr(self.c.tra, "coral_align", False)),
                bool(getattr(self.c.tra, "cdan", False)),
                bool(getattr(self.c.model, "mixstyle", False)),
                bool(getattr(self.c.model, "sagnet", False)),
                bool(getattr(self.c.model, "dual_scale", False)),
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
                    teacher_surface_depth=surface_depth,
                    teacher_surface_confidence=surface_confidence,
                    domain_ids=domain_ids,
                    sagnet_grl_scale=float(getattr(self.c.tra, "sagnet_lambda", 0.1)),
                )
            else:
                outputs = self.model(
                    images,
                    target_offsets=target_offsets,
                    teacher_surface_depth=surface_depth,
                    teacher_surface_confidence=surface_confidence,
                )
                domain_logits = None
                supcon_z = None

            attn_entropy_loss = getattr(self.model, "last_attn_entropy_loss", None)
            attn_entropy_per_target = getattr(self.model, "last_attn_entropy_per_target", None)
            new_surface_logits = getattr(self.model, "last_new_surface_logits", None)
            dual_scale_gate = getattr(self.model, "last_dual_scale_gate", None)
            if dual_scale_gate is not None:
                self._last_dual_scale_gate = float(dual_scale_gate.mean().detach())

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
            needs_domain_losses = any([
                bool(getattr(self.c.tra, "domain_gradient_mode", "")),
                bool(getattr(self.c.tra, "physical_domain_groupdro", False)),
                bool(getattr(self.c.tra, "domain_vrex", False)),
                bool(getattr(self.c.tra, "domain_cvar", False)),
                bool(getattr(self.c.tra, "pcgrad", False)),
            ])
            if needs_domain_losses:
                if domain_ids is None:
                    raise RuntimeError("physical-domain objectives require domain IDs")
                domain_group_ids, domain_group_losses = _physical_domain_group_losses(
                    per_target_loss,
                    mask,
                    domain_ids,
                )
            else:
                domain_group_ids, domain_group_losses = [], []
            domain_gradient_mode = str(
                getattr(self.c.tra, "domain_gradient_mode", "") or ""
            )
            if domain_gradient_mode:
                if domain_ids is None:
                    raise RuntimeError("domain-gradient objectives require physical-domain IDs")
                gradient_domain_ids = domain_group_ids
                gradient_domain_losses = domain_group_losses
                probe_names = (
                    "dec1.net.3.weight",
                    "out_head.weight",
                    "early2d_dec1.net.3.weight",
                    "early2d_head.weight",
                    "mid2d_dec1.net.3.weight",
                    "mid2d_head.weight",
                    "dual_scale_local.net.3.weight",
                    "dual_scale_head.weight",
                    "dual_scale_outer.net.3.weight",
                    "dual_scale_outer_head.weight",
                    "dual_scale_deep_local.dec1.net.3.weight",
                    "dual_scale_deep_local.head.weight",
                )
                probe_parameters = [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and any(token in name for token in probe_names)
                ]
                if not probe_parameters:
                    raise RuntimeError("domain-gradient objective found no prediction-head parameters")
                signatures = []
                for domain_loss in gradient_domain_losses:
                    gradients = torch.autograd.grad(
                        domain_loss,
                        probe_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    signatures.append(torch.cat([
                        (
                            torch.zeros_like(parameter).reshape(-1)
                            if gradient is None else gradient.detach().reshape(-1)
                        )
                        for parameter, gradient in zip(probe_parameters, gradients)
                    ]))
                current_cosines = {}
                ema = float(getattr(self.c.tra, "domain_gradient_ema", 0.9))
                for left in range(len(gradient_domain_ids)):
                    for right in range(left + 1, len(gradient_domain_ids)):
                        pair = tuple(sorted((gradient_domain_ids[left], gradient_domain_ids[right])))
                        cosine = float(F.cosine_similarity(
                            signatures[left].unsqueeze(0),
                            signatures[right].unsqueeze(0),
                            dim=1,
                            eps=1e-8,
                        ).item())
                        previous = self._domain_gradient_ema.get(pair, cosine)
                        smoothed = ema * previous + (1.0 - ema) * cosine
                        self._domain_gradient_ema[pair] = smoothed
                        current_cosines[pair] = smoothed
                self._last_domain_gradient_cosines = current_cosines
                threshold = float(getattr(self.c.tra, "domain_gradient_threshold", 0.0))
                clusters = _connected_domain_clusters(
                    gradient_domain_ids,
                    self._domain_gradient_ema,
                    threshold,
                )
                self._last_domain_gradient_clusters = clusters
                loss_by_domain = dict(zip(gradient_domain_ids, gradient_domain_losses))
                cluster_losses = [
                    torch.stack([loss_by_domain[domain] for domain in cluster]).mean()
                    for cluster in clusters
                ]
                if domain_gradient_mode == "cluster_balance":
                    primary_loss = torch.stack(cluster_losses).mean()
                elif domain_gradient_mode == "cluster_worst":
                    primary_loss = torch.stack(cluster_losses).max()
                elif domain_gradient_mode == "conflict_weighted":
                    conflict = []
                    for domain in gradient_domain_ids:
                        values = [
                            max(
                                0.0,
                                threshold
                                - self._domain_gradient_ema[tuple(sorted((domain, other)))],
                            )
                            for other in gradient_domain_ids
                            if other != domain
                        ]
                        conflict.append(sum(values) / max(len(values), 1))
                    weights = torch.softmax(
                        per_target_loss.new_tensor(conflict)
                        * float(getattr(self.c.tra, "domain_gradient_strength", 4.0)),
                        dim=0,
                    )
                    conflict_loss = sum(
                        weight * domain_loss
                        for weight, domain_loss in zip(weights, gradient_domain_losses)
                    )
                    blend = float(getattr(self.c.tra, "domain_gradient_blend", 1.0))
                    primary_loss = (1.0 - blend) * raw_loss_value + blend * conflict_loss
                else:
                    raise ValueError(
                        "domain_gradient_mode must be cluster_balance, cluster_worst, "
                        "or conflict_weighted"
                    )
            if bool(getattr(self.c.tra, "physical_domain_groupdro", False)):
                if domain_ids is None:
                    raise RuntimeError("physical-domain GroupDRO requires physical-domain IDs")
                robust_loss, _ = character_groupdro_loss(
                    domain_group_ids,
                    domain_group_losses,
                    self._physical_domain_groupdro_log_weights,
                    eta=float(getattr(self.c.tra, "physical_domain_groupdro_eta", 0.05)),
                    max_ratio=float(
                        getattr(self.c.tra, "physical_domain_groupdro_max_ratio", 3.0)
                    ),
                )
                if robust_loss is not None:
                    primary_loss = robust_loss
                    self._last_physical_domain_groupdro_loss = float(
                        robust_loss.detach()
                    )
            if bool(getattr(self.c.tra, "domain_vrex", False)) and domain_group_losses:
                vrex_lambda = (
                    float(getattr(self.c.tra, "domain_vrex_lambda", 1.0))
                    if epoch >= int(getattr(self.c.tra, "domain_vrex_warmup_epochs", 2))
                    else 0.0
                )
                primary_loss, vrex_penalty = domain_vrex_loss(
                    domain_group_losses,
                    vrex_lambda,
                )
                self._last_dg_losses["domain_vrex_loss"] = float(vrex_penalty.detach())
            if bool(getattr(self.c.tra, "domain_cvar", False)) and domain_group_losses:
                primary_loss, _ = domain_cvar_loss(
                    domain_group_losses,
                    float(getattr(self.c.tra, "domain_cvar_alpha", 0.25)),
                )
                self._last_dg_losses["domain_cvar_loss"] = float(primary_loss.detach())
            if bool(getattr(self.c.tra, "pcgrad", False)):
                pcgrad_domain_losses = domain_group_losses
                if domain_group_losses:
                    primary_loss = torch.stack(domain_group_losses).mean()
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
            anchor_lambda = float(getattr(self.c.tra, "mae_anchor_lambda", 0.0))
            if anchor_lambda > 0:
                named_parameters = dict(self.model.named_parameters())
                anchor_loss = sum(
                    (named_parameters[name] - reference).square().sum()
                    for name, reference in self._mae_anchor_reference.items()
                )
                loss = loss + anchor_lambda * anchor_loss
                self._last_dg_losses["mae_anchor_loss"] = float(anchor_loss.detach())
            if bool(getattr(self.c.tra, "depth_shift_aux", False)):
                depth_shift_logits = getattr(self.model, "last_depth_shift_logits", None)
                if depth_shift_logits is None or depth_shift is None:
                    raise RuntimeError("depth-shift auxiliary loss requires model logits and shifts")
                classes = int(getattr(self.c.tra, "depth_shift_aux_classes", 3))
                target_shift = depth_shift.clamp(
                    min=-(classes // 2),
                    max=classes // 2,
                ) + classes // 2
                depth_shift_loss = F.cross_entropy(depth_shift_logits, target_shift.long())
                loss = loss + float(
                    getattr(self.c.tra, "depth_shift_aux_lambda", 0.1)
                ) * depth_shift_loss
                self._last_dg_losses["depth_shift_aux_loss"] = float(
                    depth_shift_loss.detach()
                )
            deep_scores = getattr(self.model, "last_deep_supervision_scores", None)
            if deep_scores is not None:
                deep_weights = (
                    float(getattr(self.c.model, "sparse_deep_supervision_dec2_weight", 0.3)),
                    float(getattr(self.c.model, "sparse_deep_supervision_dec3_weight", 0.1)),
                )
                deep_loss = outputs.new_zeros(())
                for deep_output, deep_weight in zip(deep_scores, deep_weights):
                    deep_target_loss = self.criterion(deep_output, targets)
                    deep_loss = deep_loss + deep_weight * (
                        deep_target_loss * mask
                    ).sum() / denom
                loss = loss + deep_loss
                self._last_deep_supervision_loss = float(deep_loss.detach())
            if bool(getattr(self.c.tra, "clam_instance", False)):
                instances = getattr(self.model, "last_clam_instance_logits", None)
                if instances is None:
                    raise RuntimeError("CLAM instance loss requires model instance logits")
                clam_loss_value = clam_instance_loss(
                    instances,
                    labels,
                    mask,
                    int(getattr(self.c.tra, "clam_instance_k", 4)),
                )
                loss = loss + float(
                    getattr(self.c.tra, "clam_instance_lambda", 0.1)
                ) * clam_loss_value
            elr_loss_value, elr_temporal = self._elr_regularizer(
                outputs,
                labels,
                mask,
                character_ids_device,
                epoch,
            )
            loss = loss + float(getattr(self.c.tra, "elr_lambda", 0.1)) * elr_loss_value
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
                other_surface_depth = tf_in(surface_depth) if surface_depth is not None else None
                other_surface_confidence = (
                    tf_in(surface_confidence) if surface_confidence is not None else None
                )
                # eval mode for TTA second pass: BN uses running stats (no in-place updates)
                # which prevents the autograd version conflict with the compiled first-pass graph
                self.model.eval()
                other = self.model(
                    view.contiguous(),
                    target_offsets=other_offsets,
                    teacher_surface_depth=other_surface_depth,
                    teacher_surface_confidence=other_surface_confidence,
                )
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
            surface_ce_value = outputs.new_zeros(())
            surface_smooth_value = outputs.new_zeros(())
            surface_mae_value = outputs.new_zeros(())
            supcon_loss_value = outputs.new_zeros(())
            weighted_supcon_loss_value = outputs.new_zeros(())
            supervised_surface = bool(getattr(self.c.model, "new_learned_surface", False)) or bool(
                getattr(self.c.model, "better_surface", False)
            )
            if supervised_surface and new_surface_logits is not None:
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
                teacher_depth = surface_depth
                teacher_confidence = surface_confidence
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
                surface_total, surface_ce, surface_smooth, _ = surface_supervision_loss(
                    new_surface_logits,
                    surface_volume,
                    smooth_weight=float(getattr(self.c.tra, "new_surface_smooth_lambda", 0.02)),
                    target=surface_target,
                    valid=surface_valid,
                )
                surface_loss_value = surface_total
                surface_ce_value = surface_ce
                surface_smooth_value = surface_smooth
                surface_probs = F.softmax(new_surface_logits.float(), dim=2)
                depth_axis = torch.arange(
                    new_surface_logits.shape[2],
                    device=new_surface_logits.device,
                    dtype=surface_probs.dtype,
                ).view(1, 1, -1, 1, 1)
                expected_depth = (surface_probs * depth_axis).sum(dim=2)
                surface_mae_value = (
                    (expected_depth - teacher_depth).abs() * surface_valid
                ).sum() / surface_valid.sum().clamp(min=1.0)
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
                ignore_same_domain_same_class = bool(
                    getattr(self.c.tra, "supcon_ignore_same_domain_same_class", False)
                )
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
                        ignore_same_domain_same_class=ignore_same_domain_same_class,
                    )
                else:
                    cross_frag_ids = domain_ids if bool(getattr(self.c.tra, "supcon_cross_frag", False)) else None
                    supcon_loss_value = supcon_loss(
                        supcon_z,
                        sample_pos.long(),
                        temp=supcon_temp,
                        domain_ids=cross_frag_ids,
                        ignore_same_domain_same_class=ignore_same_domain_same_class,
                    )
                weighted_supcon_loss_value = supcon_lambda * supcon_loss_value
                loss = loss + weighted_supcon_loss_value

            if any([
                bool(getattr(self.c.tra, "prototype_align", False)),
                bool(getattr(self.c.tra, "coral_align", False)),
                bool(getattr(self.c.tra, "cdan", False)),
            ]):
                if supcon_z is None or domain_ids is None:
                    raise RuntimeError("domain alignment requires decoder embeddings and domain IDs")
                dg_embeddings, dg_labels, dg_domains = _flatten_domain_embeddings(
                    supcon_z,
                    labels.clamp(min=0),
                    mask,
                    domain_ids,
                )
                if bool(getattr(self.c.tra, "prototype_align", False)):
                    prototype_loss_value = class_conditional_prototype_loss(
                        dg_embeddings,
                        dg_labels,
                        dg_domains,
                        margin=float(getattr(self.c.tra, "prototype_margin", 0.5)),
                    )
                    loss = loss + float(
                        getattr(self.c.tra, "prototype_align_lambda", 0.1)
                    ) * prototype_loss_value
                if bool(getattr(self.c.tra, "coral_align", False)):
                    coral_loss_value = class_conditional_coral_loss(
                        dg_embeddings,
                        dg_labels,
                        dg_domains,
                        mean_weight=float(getattr(self.c.tra, "coral_mean_weight", 1.0)),
                    )
                    loss = loss + float(
                        getattr(self.c.tra, "coral_align_lambda", 0.1)
                    ) * coral_loss_value
                if bool(getattr(self.c.tra, "cdan", False)):
                    dg_probabilities = torch.sigmoid(outputs)[mask > 0]
                    cdan_logits = self.model.conditional_domain_logits(
                        dg_embeddings,
                        dg_probabilities,
                        grl_scale=float(getattr(self.c.tra, "cdan_lambda", 0.1)),
                    )
                    counts = torch.bincount(
                        dg_domains,
                        minlength=int(getattr(self.c.tra, "dann_n_domains", 0)),
                    ).float()
                    domain_weights = torch.where(
                        counts > 0,
                        counts.sum() / counts.clamp(min=1.0),
                        torch.zeros_like(counts),
                    )
                    cdan_loss_value = F.cross_entropy(
                        cdan_logits,
                        dg_domains,
                        weight=domain_weights,
                    )
                    loss = loss + cdan_loss_value

            if bool(getattr(self.c.model, "sagnet", False)):
                style_logits = getattr(self.model, "last_sagnet_logits", None)
                if style_logits is None:
                    raise RuntimeError("SagNet style logits were not produced")
                sagnet_loss_value = F.binary_cross_entropy_with_logits(
                    style_logits,
                    sample_pos,
                )
                loss = loss + sagnet_loss_value

            if (
                bool(getattr(self.c.tra, "mae_reconstruction", False))
                and epoch >= int(getattr(self.c.tra, "mae_reconstruction_start_epoch", 4))
            ):
                patch = int(getattr(self.c.tra, "mae_reconstruction_patch", 4))
                fraction = float(getattr(self.c.tra, "mae_reconstruction_mask_frac", 0.5))
                downsample = max(1, int(getattr(self.c.data, "context_downsample", 1)))
                target = (
                    F.avg_pool3d(images, kernel_size=(1, downsample, downsample), stride=(1, downsample, downsample))
                    if downsample > 1 else images
                )
                mask_height, mask_width = target.shape[-2:]
                coarse_height = max(1, math.ceil(mask_height / patch))
                coarse_width = max(1, math.ceil(mask_width / patch))
                coarse = (
                    torch.rand(
                        images.shape[0],
                        1,
                        1,
                        coarse_height,
                        coarse_width,
                        device=images.device,
                    ) < fraction
                ).to(images.dtype)
                reconstruction_mask = F.interpolate(
                    coarse.squeeze(2),
                    size=(mask_height, mask_width),
                    mode="nearest",
                ).unsqueeze(2)
                input_mask = F.interpolate(
                    reconstruction_mask.squeeze(2),
                    size=images.shape[-2:],
                    mode="nearest",
                ).unsqueeze(2)
                visible = 1.0 - input_mask
                visible_volume = visible.expand_as(images)
                fill = (images * visible_volume).sum(
                    dim=(2, 3, 4), keepdim=True
                ) / visible_volume.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
                masked_images = images * visible + fill * input_mask
                reconstruction = self.model.reconstruct(
                    masked_images,
                    teacher_surface_depth=surface_depth,
                    teacher_surface_confidence=surface_confidence,
                )
                error = (reconstruction - target).square()
                mae_reconstruction_loss_value = (
                    error * reconstruction_mask
                ).sum() / (
                    reconstruction_mask.sum() * target.shape[2]
                ).clamp(min=1.0)
                loss = loss + float(
                    getattr(self.c.tra, "mae_reconstruction_lambda", 0.1)
                ) * mae_reconstruction_loss_value

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

        if bool(getattr(self.c.tra, "pcgrad", False)):
            self._pcgrad_backward(loss, primary_loss, pcgrad_domain_losses)
        else:
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
                pair_outputs = self.model(
                    pair_images,
                    target_offsets=pair_offsets,
                    teacher_surface_depth=(
                        surface_depth[active_device] if surface_depth is not None else None
                    ),
                    teacher_surface_confidence=(
                        surface_confidence[active_device]
                        if surface_confidence is not None else None
                    ),
                )
                self.model.train()
                pair_mask = mask[active_device]
                context_loss_value = (
                    (torch.sigmoid(pair_outputs) - pair_target).square() * pair_mask
                ).sum() / pair_mask.sum().clamp(min=1.0)
                weighted_context_loss = float(
                    getattr(self.c.tra, "context_consistency_lambda", 0.1)
                ) * context_loss_value
            self.scaler.scale(weighted_context_loss).backward()
        if (
            bool(getattr(self.c.tra, "depth_view_consistency", False))
            and depth_pair is not None
            and depth_pair_active is not None
            and (depth_pair_active.view(-1) > 0).any()
        ):
            if depth_pair_surface is None or depth_pair_confidence is None:
                raise RuntimeError("depth-view consistency requires paired surface maps")
            active_cpu = depth_pair_active.view(-1) > 0
            active_device = active_cpu.to(self.c.device, non_blocking=True)
            pair_images = depth_pair[active_cpu].to(self.c.device, non_blocking=True)
            pair_offsets = target_offsets[active_device] if target_offsets is not None else None
            pair_surface = depth_pair_surface[active_cpu].to(
                self.c.device,
                non_blocking=True,
            ).float()
            pair_confidence = depth_pair_confidence[active_cpu].to(
                self.c.device,
                non_blocking=True,
            ).float()
            pair_target = torch.sigmoid(outputs[active_device]).detach()
            with autocast(self.c.device, enabled=self.c.device == "cuda"):
                self.model.eval()
                pair_outputs = self.model(
                    pair_images,
                    target_offsets=pair_offsets,
                    teacher_surface_depth=pair_surface,
                    teacher_surface_confidence=pair_confidence,
                )
                self.model.train()
                pair_mask = mask[active_device]
                depth_consistency_loss_value = (
                    (torch.sigmoid(pair_outputs) - pair_target).square() * pair_mask
                ).sum() / pair_mask.sum().clamp(min=1.0)
                weighted_depth_consistency = float(
                    getattr(self.c.tra, "depth_view_consistency_lambda", 0.2)
                ) * depth_consistency_loss_value
            self.scaler.scale(weighted_depth_consistency).backward()
        sam_rho = float(getattr(self.c.tra, "sam_rho", 0.0))
        if sam_rho > 0:
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            grad_norm = torch.norm(
                torch.stack([parameter.grad.detach().norm(2) for parameter in parameters]),
                2,
            )
            perturbations = []
            scale = sam_rho / grad_norm.clamp(min=1e-12)
            with torch.no_grad():
                for parameter in parameters:
                    perturbation = parameter.grad * scale.to(parameter)
                    parameter.add_(perturbation)
                    perturbations.append((parameter, perturbation))
            self.optimizer.zero_grad(set_to_none=True)
            try:
                with autocast(self.c.device, enabled=self.c.device == "cuda"):
                    sam_loss = self._sam_objective(
                        images,
                        targets,
                        labels,
                        mask,
                        sample_pos,
                        domain_ids,
                        character_ids_device,
                        target_offsets,
                        surface_depth,
                        surface_confidence,
                        epoch,
                        grl_scale,
                        elr_temporal,
                    )
                self.scaler.scale(sam_loss).backward()
            finally:
                with torch.no_grad():
                    for parameter, perturbation in perturbations:
                        parameter.sub_(perturbation)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.c.tra.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._update_model_ema(epoch)
        diagnostic_values = torch.stack((
            loss.detach().float(),
            raw_loss_value.detach().float(),
            dann_loss_value.detach().float(),
            spill_loss_value.detach().float(),
            surface_loss_value.detach().float(),
            surface_alpha_value.detach().float(),
            surface_ce_value.detach().float(),
            surface_smooth_value.detach().float(),
            surface_mae_value.detach().float(),
            supcon_loss_value.detach().float(),
            weighted_supcon_loss_value.detach().float(),
            context_loss_value.detach().float(),
            depth_consistency_loss_value.detach().float(),
            bag_rank_loss_value.detach().float(),
            groupdro_loss_value.detach().float(),
            cvar_loss_value.detach().float(),
            dann_accuracy_value.detach().float(),
            dann_grl_value.detach().float(),
            clam_loss_value.detach().float(),
            elr_loss_value.detach().float(),
        )).cpu().tolist()
        self._last_depth_consistency = diagnostic_values[12]
        self._last_character_objectives = (
            diagnostic_values[11],
            *diagnostic_values[13:16],
        )
        self._last_dann_accuracy = diagnostic_values[16]
        self._last_grl_scale = diagnostic_values[17]
        self._last_clam_loss = diagnostic_values[18]
        self._last_elr_loss = diagnostic_values[19]
        self._last_dg_losses.update({
            "prototype_align_loss": float(prototype_loss_value.detach()),
            "coral_align_loss": float(coral_loss_value.detach()),
            "cdan_loss": float(cdan_loss_value.detach()),
            "sagnet_loss": float(sagnet_loss_value.detach()),
            "mae_reconstruction_loss": float(mae_reconstruction_loss_value.detach()),
        })

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
        if domain_ids is not None:
            metric_domains = domain_ids.unsqueeze(1).expand_as(labels).detach().cpu().numpy().flatten()
            self._last_metric_domains = (
                metric_domains[keep] if labels.shape[1] > 1 else metric_domains
            ).astype(np.int64)
        return (
            scores,
            label_values,
            character_values,
            *diagnostic_values[:11],
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
        surface_ce_total = 0.0
        surface_smooth_total = 0.0
        surface_mae_total = 0.0
        supcon_loss_total = 0.0
        weighted_supcon_loss_total = 0.0
        labels = []
        preds = []
        scores = []
        character_ids_all = []
        domain_ids_all = []
        total_injected = 0
        dann_accuracy_total = 0.0
        grl_scale_total = 0.0
        context_loss_total = 0.0
        depth_consistency_loss_total = 0.0
        bag_rank_loss_total = 0.0
        groupdro_loss_total = 0.0
        physical_domain_groupdro_loss_total = 0.0
        deep_supervision_loss_total = 0.0
        dual_scale_gate_total = 0.0
        dual_scale_gate_batches = 0
        cvar_loss_total = 0.0
        clam_loss_total = 0.0
        elr_loss_total = 0.0
        dg_loss_totals = {key: 0.0 for key in self._last_dg_losses}
        domain_gradient_cosine_totals: dict[tuple[int, int], float] = {}
        domain_gradient_cosine_counts: dict[tuple[int, int], int] = {}
        domain_gradient_cluster_total = 0
        domain_gradient_cluster_batches = 0

        entropy_lambda = float(getattr(self.c.tra, "entropy_min_lambda", 0.0))
        u_iter = iter(self.valid_loader) if entropy_lambda > 0 else None

        for batch_index, batch in enumerate(
            tqdm(self.train_loader, desc="Training", mininterval=5, miniters=1, file=sys.stderr)
        ):
            images, batch_labels, mask = batch[:3]
            with_domain = needs_domain_ids(self.c)
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
            surface_maps_enabled = any([
                bool(getattr(self.c.model, "new_learned_surface", False)),
                bool(getattr(self.c.model, "better_surface", False)),
                bool(getattr(self.c.model, "surface_teacher_input", False)),
            ])
            if surface_maps_enabled:
                batch_surface_depth = batch[optional_index]
                batch_surface_confidence = batch[optional_index + 1]
                optional_index += 2
            else:
                batch_surface_depth = None
                batch_surface_confidence = None
            if bool(getattr(self.c.tra, "depth_shift_aux", False)):
                batch_depth_shift = batch[optional_index]
                optional_index += 1
            else:
                batch_depth_shift = None
            if bool(getattr(self.c.tra, "context_consistency", False)):
                batch_context_pair = batch[optional_index]
                batch_context_pair_active = batch[optional_index + 1]
                optional_index += 2
            else:
                batch_context_pair = None
                batch_context_pair_active = None
            if bool(getattr(self.c.tra, "depth_view_consistency", False)):
                batch_depth_pair = batch[optional_index]
                batch_depth_pair_surface = batch[optional_index + 1]
                batch_depth_pair_confidence = batch[optional_index + 2]
                batch_depth_pair_active = batch[optional_index + 3]
            else:
                batch_depth_pair = None
                batch_depth_pair_surface = None
                batch_depth_pair_confidence = None
                batch_depth_pair_active = None

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
            if batch_depth_pair_active is not None and injected_indices:
                batch_depth_pair_active[injected_indices] = 0
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
                batch_surface_ce,
                batch_surface_smooth,
                batch_surface_mae,
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
                depth_shift=batch_depth_shift,
                context_pair=batch_context_pair,
                context_pair_active=batch_context_pair_active,
                depth_pair=batch_depth_pair,
                depth_pair_surface=batch_depth_pair_surface,
                depth_pair_confidence=batch_depth_pair_confidence,
                depth_pair_active=batch_depth_pair_active,
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
            surface_ce_total += batch_surface_ce
            surface_smooth_total += batch_surface_smooth
            surface_mae_total += batch_surface_mae
            supcon_loss_total += batch_supcon_loss
            weighted_supcon_loss_total += batch_weighted_supcon_loss
            labels.extend(batch_labels_out)
            preds.extend((batch_scores > 0.5).astype(int))
            scores.extend(batch_scores)
            character_ids_all.extend(batch_character_ids_out)
            domain_ids_all.extend(self._last_metric_domains.tolist())
            context_value, bag_value, groupdro_value, cvar_value = self._last_character_objectives
            context_loss_total += context_value
            depth_consistency_loss_total += self._last_depth_consistency
            bag_rank_loss_total += bag_value
            groupdro_loss_total += groupdro_value
            physical_domain_groupdro_loss_total += self._last_physical_domain_groupdro_loss
            deep_supervision_loss_total += self._last_deep_supervision_loss
            if self._last_dual_scale_gate > 0:
                dual_scale_gate_total += self._last_dual_scale_gate
                dual_scale_gate_batches += 1
            cvar_loss_total += cvar_value
            clam_loss_total += self._last_clam_loss
            elr_loss_total += self._last_elr_loss
            for key, value in self._last_dg_losses.items():
                dg_loss_totals[key] += value
            for pair, value in self._last_domain_gradient_cosines.items():
                domain_gradient_cosine_totals[pair] = (
                    domain_gradient_cosine_totals.get(pair, 0.0) + value
                )
                domain_gradient_cosine_counts[pair] = (
                    domain_gradient_cosine_counts.get(pair, 0) + 1
                )
            if self._last_domain_gradient_clusters:
                domain_gradient_cluster_total += len(self._last_domain_gradient_clusters)
                domain_gradient_cluster_batches += 1

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
        metrics["surface_ce"] = surface_ce_total / len(self.train_loader)
        metrics["surface_smooth"] = surface_smooth_total / len(self.train_loader)
        metrics["surface_mae"] = surface_mae_total / len(self.train_loader)
        metrics["supcon_loss"] = supcon_loss_total / len(self.train_loader)
        metrics["weighted_supcon_loss"] = weighted_supcon_loss_total / len(self.train_loader)
        metrics["context_consistency_loss"] = context_loss_total / len(self.train_loader)
        metrics["depth_view_consistency_loss"] = (
            depth_consistency_loss_total / len(self.train_loader)
        )
        metrics["character_bag_ranking_loss"] = bag_rank_loss_total / len(self.train_loader)
        metrics["character_groupdro_loss"] = groupdro_loss_total / len(self.train_loader)
        metrics["physical_domain_groupdro_loss"] = (
            physical_domain_groupdro_loss_total / len(self.train_loader)
        )
        metrics["deep_supervision_loss"] = deep_supervision_loss_total / len(self.train_loader)
        metrics["dual_scale_gate"] = (
            dual_scale_gate_total / max(dual_scale_gate_batches, 1)
        )
        metrics["character_cvar_loss"] = cvar_loss_total / len(self.train_loader)
        metrics["clam_instance_loss"] = clam_loss_total / len(self.train_loader)
        metrics["elr_loss"] = elr_loss_total / len(self.train_loader)
        for key, value in dg_loss_totals.items():
            metrics[key] = value / len(self.train_loader)
        metrics["domain_gradient_cosines"] = {
            pair: value / domain_gradient_cosine_counts[pair]
            for pair, value in domain_gradient_cosine_totals.items()
        }
        metrics["domain_gradient_cluster_count"] = (
            domain_gradient_cluster_total / max(domain_gradient_cluster_batches, 1)
        )
        if bool(getattr(self.c.tra, "per_scroll_metrics", False)):
            metrics["per_scroll_pr_auc"] = _per_domain_pr_auc(
                labels,
                scores,
                domain_ids_all,
            )
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
        domain_ids_all = []

        with torch.no_grad(), autocast(self.c.device, enabled=self.c.device == "cuda"):
            for batch in tqdm(
                self.valid_loader,
                desc="Validating",
                mininterval=5,
                miniters=1,
                file=sys.stderr,
            ):
                images, batch_labels, mask = batch[:3]
                with_domain = needs_domain_ids(self.c)
                batch_domain_ids = batch[3] if with_domain and len(batch) > 3 else None
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
                surface_maps_enabled = any([
                    bool(getattr(self.c.model, "new_learned_surface", False)),
                    bool(getattr(self.c.model, "better_surface", False)),
                    bool(getattr(self.c.model, "surface_teacher_input", False)),
                ])
                if surface_maps_enabled:
                    batch_surface_depth = batch[optional_index]
                    batch_surface_confidence = batch[optional_index + 1]
                    optional_index += 2
                else:
                    batch_surface_depth = None
                    batch_surface_confidence = None
                if mask.view(mask.size(0), -1).sum() <= 0:
                    print("[ERROR] Encountered batch with mask sum == 0 in validation. This block should not be loaded!")
                    continue

                images = images.to(self.c.device, non_blocking=True)
                if batch_surface_depth is not None:
                    batch_surface_depth = batch_surface_depth.to(
                        self.c.device,
                        non_blocking=True,
                    ).float()
                    batch_surface_confidence = batch_surface_confidence.to(
                        self.c.device,
                        non_blocking=True,
                    ).float()
                B = images.size(0)
                batch_labels = batch_labels.to(self.c.device, non_blocking=True).view(B, -1)
                mask = mask.to(self.c.device, non_blocking=True).view(B, -1)
                if mask.shape[1] == batch_labels.shape[1]:
                    mask = (mask > 0).float()                          # per-sub-tile validity (multitile)
                else:
                    mask = (mask.sum(dim=1) > 0).float().unsqueeze(1)  # single-tile window gate

                outputs = self.model(
                    images,
                    target_offsets=batch_target_offsets,
                    teacher_surface_depth=batch_surface_depth,
                    teacher_surface_confidence=batch_surface_confidence,
                )
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
                if batch_domain_ids is not None:
                    batch_domains = batch_domain_ids.view(-1, 1).expand_as(batch_labels).numpy().flatten()
                    domain_ids_all.extend(
                        (batch_domains[keep] if batch_labels.shape[1] > 1 else batch_domains).tolist()
                    )

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
        if bool(getattr(self.c.tra, "per_scroll_metrics", False)):
            metrics["per_scroll_pr_auc"] = _per_domain_pr_auc(
                labels,
                scores,
                domain_ids_all,
            )
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
        for loader in (
            getattr(self, "train_loader", None),
            getattr(self, "valid_loader", None),
        ):
            if loader is None:
                continue
            iterator = getattr(loader, "_iterator", None)
            if iterator is None:
                continue
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            loader._iterator = None

    def close(self) -> None:
        """release workers, writers, and per-run references deterministically."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._shutdown_data_workers()
        scroll_visualizers = getattr(self, "scroll_vis", None) or {}
        for visualizer in scroll_visualizers.values():
            release = getattr(visualizer, "release_visualization_volume", None)
            if release is not None:
                release()
        visualizer = getattr(self, "vis", None)
        if visualizer is not None:
            visualizer.close()
        self.train_loader = None
        self.valid_loader = None
        self.train_dataset = None
        self.valid_dataset = None
        self._scroll_train_sets = None
        self._scroll_dms = {}
        self.scroll_vis = {}

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
        encoder_group = next(
            (
                group for group in self.optimizer.param_groups
                if group.get("group_name") == "encoder"
            ),
            None,
        )
        if encoder_group is not None:
            self.vis.writer.add_scalar(
                "Learning_Rate/Encoder",
                float(encoder_group["lr"]),
                epoch,
            )
        for key, tag in (
            ("context_consistency_loss", "Aux/ContextConsistency"),
            ("depth_view_consistency_loss", "Aux/DepthViewConsistency"),
            ("character_bag_ranking_loss", "Aux/CharacterBagRanking"),
            ("character_groupdro_loss", "Aux/CharacterGroupDRO"),
            ("physical_domain_groupdro_loss", "Aux/PhysicalDomainGroupDRO"),
            ("deep_supervision_loss", "Aux/SparseDeepSupervision"),
            ("dual_scale_gate", "Architecture/DualScaleGate"),
            ("character_cvar_loss", "Aux/CharacterCVaR"),
            ("clam_instance_loss", "Aux/CLAMInstance"),
            ("elr_loss", "Aux/ELR"),
            ("prototype_align_loss", "Aux/PrototypeAlign"),
            ("coral_align_loss", "Aux/ConditionalCORAL"),
            ("cdan_loss", "Aux/CDAN"),
            ("sagnet_loss", "Aux/SagNetStyle"),
            ("mae_reconstruction_loss", "Aux/MAEReconstruction"),
            ("mldg_meta_train_loss", "Aux/MLDGMetaTrain"),
            ("mldg_meta_test_loss", "Aux/MLDGMetaTest"),
            ("domain_vrex_loss", "Aux/DomainVREx"),
            ("domain_cvar_loss", "Aux/DomainCVaR"),
            ("depth_shift_aux_loss", "Aux/DepthShiftClassification"),
            ("mae_anchor_loss", "Aux/MAEAnchor"),
        ):
            if key in train_metrics:
                self.vis.writer.add_scalar(tag, train_metrics[key], epoch)
        if bool(getattr(self.c.tra, "dann", False)):
            self.vis.writer.add_scalar("DANN/DomainAccuracy", train_metrics["dann_accuracy"], epoch)
            self.vis.writer.add_scalar("DANN/GRLScale", train_metrics["dann_grl_scale"], epoch)
        if bool(getattr(self.c.tra, "per_scroll_metrics", False)):
            domain_names = list(getattr(self.c.data, "train_scroll_dict", {}) or {})
            for split, metrics in (("Train", train_metrics), ("Valid", val_metrics)):
                for domain, value in metrics.get("per_scroll_pr_auc", {}).items():
                    name = domain_names[domain] if domain < len(domain_names) else f"domain_{domain}"
                    self.vis.writer.add_scalar(
                        f"Per_Scroll/PR_AUC_{split}/{name}",
                        value,
                        epoch,
                    )
        for (left, right), value in train_metrics.get(
            "domain_gradient_cosines", {}
        ).items():
            self.vis.writer.add_scalar(
                f"DomainGradient/Cosine_{left}_{right}",
                value,
                epoch,
            )
        if bool(getattr(self.c.tra, "domain_gradient_mode", "")):
            self.vis.writer.add_scalar(
                "DomainGradient/ClusterCount",
                train_metrics.get("domain_gradient_cluster_count", 0.0),
                epoch,
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
                    visualizer.load_visualization_volume()
                    visualizer.add_evaluation_figures(epoch, self.model)
                    eval_rendered += 1
                except Exception as exc:
                    print(f"[ERROR] eval figures failed for scroll {scroll_id}: {exc}")
                    import traceback

                    traceback.print_exc()
                finally:
                    visualizer.release_visualization_volume()
            if test_due and index == 0:
                try:
                    visualizer.add_test_figures(epoch, self.model)
                except Exception as exc:
                    print(f"[ERROR] test figures failed for scroll {scroll_id}: {exc}")
            if probe_due and index == 0:
                try:
                    visualizer.load_visualization_volume()
                    visualizer.add_probe_region_figures(epoch, self.model)
                except Exception as exc:
                    print(f"[ERROR] probe figures failed for scroll {scroll_id}: {exc}")
                finally:
                    visualizer.release_visualization_volume()
        for visualizer in self.scroll_vis.values():
            visualizer.writer.flush()

    def run(self) -> None:
        try:
            self._run()
        finally:
            self.close()

    def _run(self) -> None:
        self._hm_active_this_epoch = self.c.hm.enabled
        for epoch in range(self.c.tra.n_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.c.tra.n_epochs} ---")
            start_time = time.time()

            self._update_encoder_optimization(epoch)

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

            with self._ema_weights():
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
            with self._ema_weights():
                save_model(self.model, final_path)
            print(f"[save-final] wrote final model to {final_path}")

        self.close()
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