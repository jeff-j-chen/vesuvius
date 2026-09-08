"""3D JEPA pretraining for the nnunet3d_lcndz backbone.

The student receives a volume with masked 3D cuboids and predicts EMA-teacher
feature vectors at masked decoder locations. Unlike MAE, no raw voxels are
reconstructed. A lightweight predictor creates student/teacher asymmetry;
variance and covariance regularization provide an explicit collapse guard.

Fine-tune output is a plain nnU-Net state dict:
  models/jepa_nnunet_192_ibn.pth

Example:
  python jepa_pretrain_nnunet.py --name jepa_nnunet_192_ibn --ctx 192 --ds 2 \
    --batch-size 8 --accum-steps 4 --require-all-scrolls
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from mae_pretrain_nnunet import CropSampler, MultiSampler, _apply_mask
from utils.config import Config, DEFAULT_SCROLLS

try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast_impl

    def _autocast(device):
        return _autocast_impl(str(device))

    def _scaler(device):
        return _GradScaler(str(device))
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast_impl

    def _autocast(_device):
        return _autocast_impl()

    def _scaler(_device):
        return _GradScaler()


class Jepa3D(nn.Module):
    """student predictor and stop-gradient EMA teacher."""

    def __init__(self, backbone: nn.Module, projection_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        channels = int(backbone.out_head.in_channels)
        self.student = backbone
        self.student_projector = nn.Conv3d(channels, projection_dim, kernel_size=1)
        self.predictor = nn.Sequential(
            nn.Conv3d(projection_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, projection_dim, kernel_size=1),
        )
        self.teacher = copy.deepcopy(backbone)
        self.teacher_projector = copy.deepcopy(self.student_projector)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_projector.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def student_features(self, masked_volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, decoded = self.student._encode_decode(masked_volume)
        projected = self.student_projector(decoded)
        return self.predictor(projected), projected

    @torch.no_grad()
    def teacher_features(self, volume: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        _, decoded = self.teacher._encode_decode(volume)
        return self.teacher_projector(decoded)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.lerp_(student.detach(), 1.0 - float(momentum))
        for teacher, student in zip(
            self.teacher_projector.parameters(), self.student_projector.parameters()
        ):
            teacher.lerp_(student.detach(), 1.0 - float(momentum))
        for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
            if teacher.dtype.is_floating_point:
                teacher.lerp_(student.detach(), 1.0 - float(momentum))
            else:
                teacher.copy_(student)
        for teacher, student in zip(
            self.teacher_projector.buffers(), self.student_projector.buffers()
        ):
            teacher.copy_(student)


def make_3d_block_mask(
    batch: int,
    depth: int,
    height: int,
    width: int,
    depth_patch: int,
    spatial_patch: int,
    fraction: float,
    device: torch.device | str,
    rng: np.random.Generator,
) -> torch.Tensor:
    """return a decoder-resolution mask with contiguous 3D cuboids."""
    if depth % depth_patch or height % spatial_patch or width % spatial_patch:
        raise ValueError("feature dimensions must be divisible by JEPA block dimensions")
    gd, gh, gw = depth // depth_patch, height // spatial_patch, width // spatial_patch
    blocks = gd * gh * gw
    masked = max(1, min(blocks - 1, int(round(float(fraction) * blocks))))
    coarse = torch.zeros(batch, blocks, device=device)
    for index in range(batch):
        chosen = torch.from_numpy(rng.choice(blocks, size=masked, replace=False)).to(device)
        coarse[index, chosen] = 1.0
    coarse = coarse.view(batch, 1, gd, gh, gw)
    return F.interpolate(coarse, size=(depth, height, width), mode="nearest")


def masked_jepa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.detach().float(), dim=1)
    squared = (prediction - target).square() * mask
    return squared.sum() / (mask.sum() * prediction.shape[1]).clamp(min=1.0)


def variance_covariance_loss(
    features: torch.Tensor,
    max_points: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style collapse guard over sampled 3D feature locations."""
    points = features.permute(0, 2, 3, 4, 1).reshape(-1, features.shape[1]).float()
    if points.shape[0] > max_points:
        indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points = points[indices]
    points = points - points.mean(dim=0, keepdim=True)
    std = torch.sqrt(points.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance = (points.T @ points) / max(points.shape[0] - 1, 1)
    covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = covariance.square().sum() / features.shape[1]
    return variance, covariance_loss


def momentum_at(step: int, total_steps: int, base: float, final: float) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return final - (final - base) * (math.cos(math.pi * progress) + 1.0) * 0.5


def build_samplers(args, cfg: Config):
    import zarr

    scroll_ids = args.scroll_ids or [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
    split_by_id = {
        int(scroll.scroll_id): (scroll.split_axis, float(scroll.train_split_frac))
        for scroll in DEFAULT_SCROLLS
    }
    train_samplers = []
    monitor_samplers = []
    missing = []
    for scroll_id in scroll_ids:
        try:
            volume = zarr.open(
                os.path.join(cfg.data.zarr_path, f"{scroll_id}.zarr"),
                mode="r",
            )
        except Exception as error:
            print(f"[jepa] skip {scroll_id}: {error}")
            missing.append(scroll_id)
            continue
        height, width = int(volume.shape[1]), int(volume.shape[2])
        axis, fraction = split_by_id.get(scroll_id, ("x", 0.75))
        if axis == "y":
            y1 = (int(height * fraction) // args.ctx) * args.ctx
            x1 = (width // args.ctx) * args.ctx
        else:
            y1 = (height // args.ctx) * args.ctx
            x1 = (int(width * fraction) // args.ctx) * args.ctx
        if y1 < args.ctx or x1 < args.ctx:
            print(f"[jepa] skip {scroll_id}: train region too small")
            missing.append(scroll_id)
            continue
        print(
            f"[jepa] {scroll_id} ({height}x{width}) axis={axis} frac={fraction} "
            f"-> y[0,{y1}] x[0,{x1}]"
        )
        train_samplers.append(
            CropSampler(
                scroll_id,
                cfg.data.zarr_path,
                cfg,
                0,
                y1,
                0,
                x1,
                "train",
                holdout_frac=args.holdout_frac,
            )
        )
        monitor_samplers.append(
            CropSampler(
                scroll_id,
                cfg.data.zarr_path,
                cfg,
                0,
                y1,
                0,
                x1,
                "monitor",
                holdout_frac=args.holdout_frac,
            )
        )
    if args.require_all_scrolls and missing:
        raise RuntimeError(
            f"required all {len(scroll_ids)} scrolls, but {len(missing)} are unavailable: {missing}"
        )
    if not train_samplers:
        raise RuntimeError("no usable scrolls")
    print(f"[jepa] balanced round-robin across {len(train_samplers)}/{len(scroll_ids)} scrolls")
    return MultiSampler(train_samplers), MultiSampler(monitor_samplers)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D JEPA pretraining for nnunet3d_lcndz")
    parser.add_argument("-n", "--name", default="jepa_nnunet_192_ibn")
    parser.add_argument("--scroll-ids", type=int, nargs="+", default=None)
    parser.add_argument("--require-all-scrolls", action="store_true")
    parser.add_argument("--ctx", type=int, default=192)
    parser.add_argument("--ds", type=int, default=2)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--d-start", type=int, default=4)
    parser.add_argument("--d-end", type=int, default=28)
    parser.add_argument("--spatial-patch", type=int, default=8)
    parser.add_argument("--depth-patch", type=int, default=4)
    parser.add_argument("--mask-frac", type=float, default=0.60)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.04)
    parser.add_argument("--ema-base", type=float, default=0.996)
    parser.add_argument("--ema-final", type=float, default=1.0)
    parser.add_argument("--no-ibn", action="store_false", dest="ibn", default=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--min-lr-frac", type=float, default=0.02)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--log-int", type=int, default=50)
    parser.add_argument("--save-int", type=int, default=1000)
    parser.add_argument("--log-dir", default="runs_jepa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="resume checkpoint written by this script")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.ctx <= 0 or args.ds <= 0 or args.ctx % args.ds:
        parser.error("ctx must be positive and divisible by ds")
    if args.depth <= 0 or args.depth_patch <= 0 or args.depth % args.depth_patch:
        parser.error("depth must be positive and divisible by depth-patch")
    feature_size = args.ctx // args.ds
    if args.spatial_patch <= 0 or feature_size % args.spatial_patch:
        parser.error("ctx/ds must be divisible by spatial-patch")
    if not 0.0 < args.mask_frac < 1.0:
        parser.error("mask-frac must be between zero and one")
    if args.batch_size <= 0 or args.accum_steps <= 0:
        parser.error("batch-size and accum-steps must be positive")

    cfg = Config()
    cfg.model.arch = "nnunet3d_lcndz"
    cfg.model.compile_model = False
    cfg.model.use_ibn = args.ibn
    cfg.model.attn_mil = False
    cfg.model.feature_attn_mil = False
    cfg.model.learned_surface = False
    cfg.model.new_learned_surface = False
    cfg.model.conv1_drop = 0.0
    cfg.model.conv2_drop = 0.0
    cfg.model.head_drop = 0.0
    cfg.model.skip_drop = 0.0
    cfg.tra.supcon = False
    cfg.tra.dann = False
    cfg.data.tile_size = 16
    cfg.data.depth = args.depth
    cfg.data.train_d_start = args.d_start
    cfg.data.train_d_end = args.d_end
    cfg.data.context_size = args.ctx
    cfg.data.context_downsample = args.ds
    device = cfg.device
    rng = np.random.default_rng(args.seed)

    train_sampler, monitor_sampler = build_samplers(args, cfg)

    from utils.model import create_model

    backbone, _ = create_model(cfg)
    network = Jepa3D(
        backbone,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    network.train()
    network.teacher.eval()

    if args.dry_run:
        sample_count = min(args.batch_size, 2)
        batch = train_sampler.sample(sample_count, rng).to(device)
        feature_height = args.ctx // args.ds
        mask = make_3d_block_mask(
            sample_count,
            args.depth,
            feature_height,
            feature_height,
            args.depth_patch,
            args.spatial_patch,
            args.mask_frac,
            device,
            rng,
        )
        input_mask = F.interpolate(mask, size=(args.depth, args.ctx, args.ctx), mode="nearest")
        with torch.no_grad(), _autocast(device):
            prediction, projected = network.student_features(_apply_mask(batch, input_mask))
            target = network.teacher_features(batch)
            loss = masked_jepa_loss(prediction, target, mask)
            variance, covariance = variance_covariance_loss(projected)
        monitor = monitor_sampler.sample(sample_count, rng)
        print(
            f"[jepa] dry-run OK train={tuple(batch.shape)} monitor={tuple(monitor.shape)} "
            f"features={tuple(prediction.shape)} loss={float(loss):.5f} "
            f"var={float(variance):.5f} cov={float(covariance):.5f}"
        )
        return

    parameters = list(network.student.parameters())
    parameters += list(network.student_projector.parameters())
    parameters += list(network.predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scaler = _scaler(device)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    start_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        network.student.load_state_dict(resume["student"])
        network.student_projector.load_state_dict(resume["student_projector"])
        network.predictor.load_state_dict(resume["predictor"])
        network.teacher.load_state_dict(resume["teacher"])
        network.teacher_projector.load_state_dict(resume["teacher_projector"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_step = int(resume["step"])
        print(f"[jepa] resumed {args.resume} at step {start_step}")
    from torch.utils.tensorboard import SummaryWriter

    run_dir = os.path.join(args.log_dir, f"{args.name}_{time.strftime('%m%d_%H-%M-%S')}")
    writer = SummaryWriter(run_dir)
    os.makedirs("models", exist_ok=True)
    fine_tune_path = os.path.join("models", f"{args.name}.pth")
    resume_path = os.path.join("models", f"{args.name}_resume.pth")
    print(
        f"[jepa] ctx={args.ctx}/ds{args.ds} depth={args.depth} "
        f"effective_batch={args.batch_size * args.accum_steps} mask={args.mask_frac} "
        f"steps={args.steps} log={run_dir}"
    )
    print(f"[jepa] fine-tune checkpoint -> {fine_tune_path}")

    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = variance_value = covariance_value = 0.0
        completed = 0
        for _ in range(args.accum_steps):
            batch = train_sampler.sample(args.batch_size, rng).to(device)
            feature_height = args.ctx // args.ds
            mask = make_3d_block_mask(
                batch.shape[0],
                args.depth,
                feature_height,
                feature_height,
                args.depth_patch,
                args.spatial_patch,
                args.mask_frac,
                device,
                rng,
            )
            input_mask = F.interpolate(mask, size=(args.depth, args.ctx, args.ctx), mode="nearest")
            masked = _apply_mask(batch, input_mask)
            with _autocast(device):
                prediction, projected = network.student_features(masked)
                with torch.no_grad():
                    target = network.teacher_features(batch)
                prediction_loss = masked_jepa_loss(prediction, target, mask)
                variance_loss, covariance_loss = variance_covariance_loss(projected)
                micro_loss = (
                    prediction_loss
                    + args.variance_weight * variance_loss
                    + args.covariance_weight * covariance_loss
                )
                scaled_loss = micro_loss / args.accum_steps
            scaler.scale(scaled_loss).backward()
            train_loss += float(prediction_loss.detach().item())
            variance_value += float(variance_loss.detach().item())
            covariance_value += float(covariance_loss.detach().item())
            completed += 1
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        momentum = momentum_at(step, args.steps, args.ema_base, args.ema_final)
        network.update_teacher(momentum)

        if step % args.log_int == 0:
            divisor = max(completed, 1)
            writer.add_scalar("JEPA/prediction_loss_train", train_loss / divisor, step)
            writer.add_scalar("JEPA/variance_loss_train", variance_value / divisor, step)
            writer.add_scalar("JEPA/covariance_loss_train", covariance_value / divisor, step)
            writer.add_scalar("JEPA/ema_momentum", momentum, step)
            writer.add_scalar("JEPA/lr", optimizer.param_groups[0]["lr"], step)
            network.eval()
            network.teacher.eval()
            with torch.no_grad():
                monitor = monitor_sampler.sample(args.batch_size, rng).to(device)
                mask = make_3d_block_mask(
                    monitor.shape[0],
                    args.depth,
                    args.ctx // args.ds,
                    args.ctx // args.ds,
                    args.depth_patch,
                    args.spatial_patch,
                    args.mask_frac,
                    device,
                    rng,
                )
                input_mask = F.interpolate(mask, size=(args.depth, args.ctx, args.ctx), mode="nearest")
                with _autocast(device):
                    prediction, _ = network.student_features(_apply_mask(monitor, input_mask))
                    target = network.teacher_features(monitor)
                    monitor_loss = masked_jepa_loss(prediction, target, mask)
                writer.add_scalar("JEPA/prediction_loss_monitor", float(monitor_loss.item()), step)
            network.train()
            network.teacher.eval()
            print(
                f"[jepa] step {step}/{args.steps} pred={train_loss/divisor:.5f} "
                f"monitor={float(monitor_loss):.5f} var={variance_value/divisor:.5f} "
                f"cov={covariance_value/divisor:.5f} ({time.time()-started:.0f}s)",
                flush=True,
            )

        if step % args.save_int == 0 or step == args.steps:
            torch.save(network.student.state_dict(), fine_tune_path)
            torch.save(
                {
                    "student": network.student.state_dict(),
                    "student_projector": network.student_projector.state_dict(),
                    "predictor": network.predictor.state_dict(),
                    "teacher": network.teacher.state_dict(),
                    "teacher_projector": network.teacher_projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                resume_path,
            )
            print(f"[jepa] saved {fine_tune_path} @ step {step}", flush=True)

    writer.close()
    print(f"[jepa] done. c.init_weights = '{fine_tune_path}'")


if __name__ == "__main__":
    main()
