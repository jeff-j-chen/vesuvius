"""model.py -- current nnU-Net-style ink detector.

The repo has converged on one backbone family:
  nnunet3d_lcndz

This file keeps only the integrations still exercised by the current sweep:
  - raw + lcn + dz stem
  - optional learned surface attention
  - optional attention-MIL with entropy regularization
    - fixed minimum-support, WELDON, and CLAM-lite MIL evidence
    - optional structure-tensor fiber coordinates
  - optional spatial SupCon projection head
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, float(scale))


def _mil_lse(voxel_map: torch.Tensor, lse_r: torch.Tensor) -> torch.Tensor:
    """aggregate voxel logits into one tile logit with learnable log-sum-exp."""
    r = lse_r.clamp(min=0.5, max=10.0)
    flat = voxel_map.flatten(1)
    n_voxels = flat.new_tensor(float(flat.shape[1]))
    return (1.0 / r) * (torch.logsumexp(r * flat, dim=1, keepdim=True) - torch.log(n_voxels))


def _lcn2d(x5: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """per-slice local contrast normalization for (B, 1, D, H, W)."""
    batch, channels, depth, height, width = x5.shape
    flat = x5.reshape(batch * depth, channels, height, width)
    mean = F.avg_pool2d(flat, kernel_size, stride=1, padding=kernel_size // 2)
    var = F.avg_pool2d(flat * flat, kernel_size, stride=1, padding=kernel_size // 2) - mean * mean
    norm = (flat - mean) / torch.sqrt(var.clamp(min=1e-4))
    return norm.reshape(batch, channels, depth, height, width)


class DepthSurfaceAttn(nn.Module):
    """tiny depth-only conv stack that learns surface-proximal slices."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
        )
        self.reset_output_layer()

    def reset_output_layer(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class NewDepthSurfaceHead(nn.Module):
    """predict one papyrus-air boundary using local and dilated spatial context."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=True),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=(1, 2, 2),
                dilation=(1, 2, 2),
                bias=True,
            ),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=(1, 4, 4),
                dilation=(1, 4, 4),
                bias=True,
            ),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetterDepthSurfaceHead(nn.Module):
    """predict a coherent surface from physical edge evidence and broad context."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv3d(2, hidden, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=True),
            nn.LeakyReLU(0.01, inplace=False),
        )
        self.local = nn.Sequential(
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=(1, 2, 2),
                dilation=(1, 2, 2),
                bias=True,
            ),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=(1, 4, 4),
                dilation=(1, 4, 4),
                bias=True,
            ),
            nn.LeakyReLU(0.01, inplace=False),
        )
        self.coarse = nn.Sequential(
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=(1, 2, 2),
                dilation=(1, 2, 2),
                bias=True,
            ),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.01, inplace=False),
        )
        self.output = nn.Conv3d(2 * hidden, 1, kernel_size=1, bias=True)
        self.prior_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _transition_evidence(x: torch.Tensor) -> torch.Tensor:
        smooth = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        smooth = (smooth[:, :, :-2] + 2.0 * smooth[:, :, 1:-1] + smooth[:, :, 2:]) * 0.25
        low = torch.quantile(smooth.float(), 0.10, dim=2, keepdim=True).to(smooth.dtype)
        high = torch.quantile(smooth.float(), 0.90, dim=2, keepdim=True).to(smooth.dtype)
        contrast = high - low
        threshold = low + 0.35 * contrast
        occupancy = torch.sigmoid((smooth - threshold) / (0.08 * contrast).clamp(min=0.01))
        transition = F.relu(occupancy[:, :, :-1] - occupancy[:, :, 1:])
        fine = F.avg_pool3d(transition, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        coarse = F.avg_pool3d(
            transition,
            kernel_size=(1, 17, 17),
            stride=1,
            padding=(0, 8, 8),
        )
        evidence = 0.35 * fine + 0.65 * coarse
        return F.pad(evidence, (0, 0, 0, 0, 0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        evidence = self._transition_evidence(x)
        features = self.input(torch.cat((x, evidence), dim=1))
        local = self.local(features) + features
        coarse = F.avg_pool3d(local, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        coarse = self.coarse(coarse)
        coarse = F.interpolate(
            coarse,
            size=local.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        residual = self.output(torch.cat((local, coarse), dim=1))
        prior = torch.log(evidence.clamp(min=1e-4))
        return residual + F.softplus(self.prior_gain) * prior


class SupConHead(nn.Module):
    """projection head for supervised contrastive learning."""

    def __init__(self, in_features: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        z = self.net(embedding)
        return F.normalize(z, dim=-1)


class DomainClassifier(nn.Module):
    """small MLP domain head used by DANN over fragment embeddings."""

    def __init__(self, in_features: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, embedding: torch.Tensor, grl_scale: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(embedding, grl_scale))


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.07, domain_ids: torch.Tensor | None = None) -> torch.Tensor:
    """supervised contrastive loss; domain_ids restricts positives to cross-fragment pairs only."""
    batch = z.shape[0]
    if batch < 2:
        return z.new_zeros(())

    sim = torch.mm(z, z.T) / temp
    labels = labels.view(-1)
    eye = torch.eye(batch, dtype=torch.bool, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye)
    if domain_ids is not None:
        d = domain_ids.view(-1)
        pos_mask = pos_mask & (d.unsqueeze(0) != d.unsqueeze(1))

    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
    pos_count = pos_mask.float().sum(dim=1)
    valid = (pos_count > 0).float()
    per_row = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1.0)
    return (per_row * valid).sum() / valid.sum().clamp(min=1.0)


class GatedAttentionMIL(nn.Module):
    """gated attention-MIL aggregator over voxel logits."""

    def __init__(self, feat_dim: int = 1, att_dim: int = 32):
        super().__init__()
        self.v = nn.Linear(feat_dim, att_dim, bias=False)
        self.u = nn.Linear(feat_dim, att_dim, bias=False)
        self.w = nn.Linear(att_dim, 1, bias=False)
        self.out = nn.Linear(feat_dim, 1, bias=True)
        self.last_attn_weights: torch.Tensor | None = None
        self.last_entropy_loss_per_bag: torch.Tensor | None = None

    def forward(
        self,
        voxel_map: torch.Tensor,
        entropy_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = voxel_map.flatten(2).permute(0, 2, 1)
        gate = torch.tanh(self.v(features)) * torch.sigmoid(self.u(features))
        weights = torch.softmax(self.w(gate).squeeze(-1), dim=-1)
        self.last_attn_weights = weights.detach()
        score = (weights.unsqueeze(-1) * self.out(features)).sum(dim=1)

        self.last_entropy_loss_per_bag = voxel_map.new_zeros((voxel_map.shape[0],))
        entropy_loss = voxel_map.new_zeros(())
        if entropy_weight > 0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
            self.last_entropy_loss_per_bag = -entropy_weight * entropy
            entropy_loss = self.last_entropy_loss_per_bag.mean()
        return score, entropy_loss


def _init_norm(norm: nn.Module) -> None:
    if hasattr(norm, "weight") and norm.weight is not None:
        nn.init.constant_(norm.weight, 1.0)
    if hasattr(norm, "bias") and norm.bias is not None:
        nn.init.constant_(norm.bias, 0.0)


class PrototypeHead(nn.Module):
    """online ink/papyrus prototypes updated via EMA; classifies by cosine distance."""

    def __init__(self, feat_dim: int = 256, ema: float = 0.99):
        super().__init__()
        self.ema = ema
        self.register_buffer("proto_ink", F.normalize(torch.ones(feat_dim), dim=0))
        self.register_buffer("proto_pap", F.normalize(-torch.ones(feat_dim), dim=0))

    @torch.no_grad()
    def update(self, embedding: torch.Tensor, labels: torch.Tensor) -> None:
        for z, key in [
            (embedding[labels.view(-1) > 0.5], "proto_ink"),
            (embedding[labels.view(-1) <= 0.5], "proto_pap"),
        ]:
            if z.shape[0] == 0:
                continue
            z_mean = F.normalize(z.mean(dim=0), dim=0)
            proto = getattr(self, key)
            proto.copy_(F.normalize(self.ema * proto + (1 - self.ema) * z_mean, dim=0))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """logit: sim(z, ink_proto) - sim(z, pap_proto), shaped (B, 1)."""
        z = F.normalize(embedding, dim=-1)
        return (z @ self.proto_ink - z @ self.proto_pap).unsqueeze(-1)


class DepthProfileHead(nn.Module):
    """spatial-free classifier: averages center voxel map over H,W then classifies depth profile.
    has zero spatial capacity -- cannot memorize tile coordinates, only depth signal."""

    def __init__(self, depth: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(depth, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
        )

    def forward(self, center_voxels: torch.Tensor) -> torch.Tensor:
        """center_voxels: (B, 1, D, H, W) → (B, 1) logit via depth profile only."""
        profile = center_voxels.mean(dim=(1, 3, 4))  # (B, D): collapse spatial, keep depth
        return self.net(profile)


class IBN3d(nn.Module):
    """IBN-a: instance norm on first half of channels, batch norm on second (Pan et al. 2018).
    IN strips fragment-specific style; BN preserves discriminative content statistics."""

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.in_norm = nn.InstanceNorm3d(half, affine=True)
        self.bn_norm = nn.BatchNorm3d(channels - half, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[1] // 2
        return torch.cat([self.in_norm(x[:, :half]), self.bn_norm(x[:, half:])], dim=1)


class ConvBlock3d(nn.Module):
    """two-conv nnU-Net block with instance norm and leaky relu."""

    def __init__(self, in_channels: int, out_channels: int, use_ibn: bool = False):
        super().__init__()
        # IBN only on the first conv's norm; second conv always uses pure IN
        norm1: nn.Module = IBN3d(out_channels) if use_ibn else nn.InstanceNorm3d(out_channels, affine=True)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock2d(nn.Module):
    """two-conv 2D U-Net block used after early surface-normal fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DividedSpaceDepthAttention3d(nn.Module):
    """TimeSformer-style depth attention followed by optional windowed XY attention."""

    def __init__(self, channels: int, heads: int = 4, window: int = 8, spatial: bool = True):
        super().__init__()
        if channels % heads:
            raise ValueError("divided-attention channels must be divisible by heads")
        self.channels = int(channels)
        self.window = max(1, int(window))
        self.spatial = bool(spatial)
        self.depth_norm = nn.LayerNorm(channels)
        self.depth_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        if self.spatial:
            self.spatial_norm = nn.LayerNorm(channels)
            self.spatial_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        else:
            self.spatial_norm = None
            self.spatial_attn = None
        self.depth_gain = nn.Parameter(torch.tensor(0.0))
        self.spatial_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        depth_tokens = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, depth, channels)
        normalized = self.depth_norm(depth_tokens)
        attended, _ = self.depth_attn(normalized, normalized, normalized, need_weights=False)
        depth_tokens = depth_tokens + self.depth_gain * attended
        x = depth_tokens.reshape(batch, height, width, depth, channels).permute(0, 4, 3, 1, 2)
        if not self.spatial:
            return x

        window = self.window
        pad_h = (-height) % window
        pad_w = (-width) % window
        padded = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = padded.shape[-2:]
        tokens = padded.permute(0, 2, 3, 4, 1).reshape(
            batch,
            depth,
            padded_h // window,
            window,
            padded_w // window,
            window,
            channels,
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, window * window, channels)
        normalized = self.spatial_norm(tokens)
        attended, _ = self.spatial_attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.spatial_gain * attended
        padded = tokens.reshape(
            batch,
            depth,
            padded_h // window,
            padded_w // window,
            window,
            window,
            channels,
        ).permute(0, 6, 1, 2, 4, 3, 5).reshape(
            batch, channels, depth, padded_h, padded_w
        )
        return padded[:, :, :, :height, :width]


class MedNeXtAdapter3d(nn.Module):
    """zero-initialized anisotropic large-kernel ConvNeXt residual adapter."""

    def __init__(self, channels: int, spatial_kernel: int = 5, expansion: int = 2):
        super().__init__()
        if spatial_kernel < 3 or spatial_kernel % 2 == 0:
            raise ValueError("MedNeXt spatial kernel must be an odd integer >=3")
        hidden = channels * max(1, int(expansion))
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, spatial_kernel, spatial_kernel),
            padding=(1, spatial_kernel // 2, spatial_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.expand = nn.Conv3d(channels, hidden, kernel_size=1)
        self.project = nn.Conv3d(hidden, channels, kernel_size=1)

    def reset_output(self) -> None:
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(x)
        residual = self.norm(residual)
        residual = F.gelu(self.expand(residual))
        return x + self.project(residual)


class NnUnet3dLcndz(nn.Module):
    """current production backbone: nnU-Net with raw + lcn + dz stem."""

    def __init__(self, config: Config):
        super().__init__()
        self._downsample = max(1, int(getattr(config.data, "context_downsample", 1)))
        self._tile_size = int(getattr(config.data, "tile_size", 16))
        self._context_size = int(getattr(config.data, "context_size", 0) or 0)
        self._attn_entropy_weight = float(getattr(config.model, "attn_entropy_weight", 0.0))

        self.last_voxel_map: torch.Tensor | None = None
        self.last_voxel_map_full: torch.Tensor | None = None
        self.last_center_voxel_map: torch.Tensor | None = None
        self.last_attn_entropy_loss: torch.Tensor | None = None
        self.last_attn_entropy_per_target: torch.Tensor | None = None
        self.last_surface_attn: torch.Tensor | None = None
        self.last_new_surface_logits: torch.Tensor | None = None
        self.last_new_surface_probs: torch.Tensor | None = None
        self.last_surface_guided_alpha: torch.Tensor | None = None
        self.last_surface_target: torch.Tensor | None = None
        self.last_surface_valid: torch.Tensor | None = None
        self.last_clam_instance_logits: torch.Tensor | None = None

        self.lse_r = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        input_depth = int(getattr(config.data, "depth", 24))
        self._allow_depth4 = bool(getattr(config.model, "allow_depth4", False))
        if self._allow_depth4 and input_depth != 4:
            raise ValueError("allow_depth4=True is only valid with depth=4")
        if input_depth < 8 and not (self._allow_depth4 and input_depth == 4):
            raise ValueError("depth below 8 requires allow_depth4=True and depth=4")
        self.pool = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d((1, 2, 2)) if self._allow_depth4 else self.pool

        use_ibn = bool(getattr(config.model, "use_ibn", False))
        # width multiplier on the 32/64/128/256 channel ladder (0.5 = half -> ~4x fewer conv FLOPs)
        _m = float(getattr(config.model, "channels_mult", 1.0) or 1.0)
        c1, c2, c3, c4 = (max(1, int(round(ch * _m))) for ch in (32, 64, 128, 256))
        self.enc1 = ConvBlock3d(3, c1, use_ibn=use_ibn)
        self.enc2 = ConvBlock3d(c1, c2, use_ibn=use_ibn)
        self.enc3 = ConvBlock3d(c2, c3)
        self.bottleneck = ConvBlock3d(c3, c4)

        # spatial channel dropout after early encoder stages and before classification head
        _d1 = float(getattr(config.model, "conv1_drop", 0.0))
        _d2 = float(getattr(config.model, "conv2_drop", 0.0))
        _dh = float(getattr(config.model, "head_drop", 0.0))
        self._enc1_drop = nn.Dropout3d(p=_d1) if _d1 > 0 else None
        self._enc2_drop = nn.Dropout3d(p=_d2) if _d2 > 0 else None
        self._head_drop = nn.Dropout3d(p=_dh) if _dh > 0 else None

        up3_kernel = (1, 2, 2) if self._allow_depth4 else 2
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=up3_kernel, stride=up3_kernel)
        self.dec3 = ConvBlock3d(c3 * 2, c3)   # cat(up3, enc3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c1 * 2, c1)
        self.out_head = nn.Conv3d(c1, 1, kernel_size=1, bias=True)

        self._early_2d_unet = bool(getattr(config.model, "early_2d_unet", False))
        if self._early_2d_unet:
            if not bool(getattr(config.model, "multitile", False)):
                raise ValueError("early_2d_unet requires multitile=True")
            self.early_depth_attn = nn.Conv3d(c1, 1, kernel_size=1)
            self.early_depth_fuse = nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=False)
            self.early2d_enc2 = ConvBlock2d(c1, c2)
            self.early2d_enc3 = ConvBlock2d(c2, c3)
            self.early2d_bottleneck = ConvBlock2d(c3, c4)
            self.early2d_up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
            self.early2d_dec3 = ConvBlock2d(c3 * 2, c3)
            self.early2d_up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.early2d_dec2 = ConvBlock2d(c2 * 2, c2)
            self.early2d_up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.early2d_dec1 = ConvBlock2d(c1 * 2, c1)
            self.early2d_head = nn.Conv2d(c1, 1, kernel_size=1)
        else:
            self.early_depth_attn = None
            self.early_depth_fuse = None
            self.early2d_enc2 = None
            self.early2d_enc3 = None
            self.early2d_bottleneck = None
            self.early2d_up3 = None
            self.early2d_dec3 = None
            self.early2d_up2 = None
            self.early2d_dec2 = None
            self.early2d_up1 = None
            self.early2d_dec1 = None
            self.early2d_head = None

        self._divided_attention = bool(getattr(config.model, "divided_attention", False))
        self.divided_attention = (
            DividedSpaceDepthAttention3d(
                c3,
                heads=int(getattr(config.model, "divided_attention_heads", 4)),
                window=int(getattr(config.model, "divided_attention_window", 8)),
                spatial=bool(getattr(config.model, "divided_attention_spatial", False)),
            )
            if self._divided_attention else None
        )

        self._mednext_adapters = bool(getattr(config.model, "mednext_adapters", False))
        if self._mednext_adapters:
            kernel = int(getattr(config.model, "mednext_kernel", 5))
            expansion = int(getattr(config.model, "mednext_expansion", 2))
            self.mednext1 = MedNeXtAdapter3d(c1, kernel, expansion)
            self.mednext2 = MedNeXtAdapter3d(c2, kernel, expansion)
            self.mednext3 = MedNeXtAdapter3d(c3, kernel, expansion)
            self.mednext_bottleneck = MedNeXtAdapter3d(c4, kernel, expansion)
        else:
            self.mednext1 = None
            self.mednext2 = None
            self.mednext3 = None
            self.mednext_bottleneck = None
        self._fiber_coordinate_branch = bool(
            getattr(config.model, "fiber_coordinate_branch", False)
        )
        self.fiber_coordinate_input = (
            nn.Conv3d(3, c1, kernel_size=1, bias=False)
            if self._fiber_coordinate_branch else None
        )

        use_better_surface = bool(getattr(config.model, "better_surface", False))
        use_new_surface = bool(getattr(config.model, "new_learned_surface", False))
        self._surface_teacher_input = bool(getattr(config.model, "surface_teacher_input", False))
        if use_better_surface and use_new_surface:
            raise ValueError("better_surface and new_learned_surface are mutually exclusive")
        if use_better_surface:
            self.new_surface_head: nn.Module | None = BetterDepthSurfaceHead(hidden=8)
        elif use_new_surface:
            self.new_surface_head = NewDepthSurfaceHead(hidden=8)
        else:
            self.new_surface_head = None
        if self._surface_teacher_input:
            # literal local depth, confidence, and signed offset from each z slice
            self.new_surface_input = nn.Conv3d(3, c1, kernel_size=1, bias=False)
        elif self.new_surface_head is not None:
            # keep the pretrained three-channel stem intact and inject the new map residually
            self.new_surface_input = nn.Conv3d(1, c1, kernel_size=1, bias=False)
        else:
            self.new_surface_input = None

        if bool(getattr(config.model, "learned_surface", False)):
            self.depth_surface_attn: DepthSurfaceAttn | None = DepthSurfaceAttn(hidden=8)
        else:
            self.depth_surface_attn = None

        if bool(getattr(config.model, "attn_mil", False)):
            self.attn_mil: GatedAttentionMIL | None = GatedAttentionMIL(feat_dim=1, att_dim=32)
        else:
            self.attn_mil = None

        if bool(getattr(config.model, "feature_attn_mil", False)):
            self.feature_attn_mil: GatedAttentionMIL | None = GatedAttentionMIL(
                feat_dim=c1,
                att_dim=32,
            )
        else:
            self.feature_attn_mil = None

        self._feature_depth_fusion = bool(
            getattr(config.model, "feature_depth_fusion", False)
        )
        if self._feature_depth_fusion:
            if not bool(getattr(config.model, "multitile", False)):
                raise ValueError("feature_depth_fusion requires multitile=True")
            if self.attn_mil is not None or self.feature_attn_mil is not None:
                raise ValueError("feature_depth_fusion is mutually exclusive with attention-MIL")
            self.depth_fusion_attn = nn.Conv3d(c1, 1, kernel_size=1, bias=True)
            self.depth_fusion_head = nn.Sequential(
                nn.Conv2d(c1 * 2, c1, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(c1, affine=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.Conv2d(c1, 1, kernel_size=1, bias=True),
            )
        else:
            self.depth_fusion_attn = None
            self.depth_fusion_head = None

        self._minimum_support_k = int(getattr(config.model, "minimum_support_k", 0))
        self._minimum_support_kernel = int(
            getattr(config.model, "minimum_support_kernel", 3)
        )
        self._weldon_k = int(getattr(config.model, "weldon_k", 0))
        self._clam_instance = bool(getattr(config.tra, "clam_instance", False))
        if self._minimum_support_k < 0 or self._weldon_k < 0:
            raise ValueError("minimum-support and WELDON k must be non-negative")
        if self._minimum_support_k and self._weldon_k:
            raise ValueError("minimum-support and WELDON aggregators are mutually exclusive")
        if self._minimum_support_kernel < 1 or self._minimum_support_kernel % 2 == 0:
            raise ValueError("minimum_support_kernel must be a positive odd integer")

        self._surface_guided_mil = bool(getattr(config.model, "surface_guided_mil", False))
        self._surface_guided_mix = float(getattr(config.model, "surface_guided_mix", 0.5))
        self._surface_band_sigma = float(getattr(config.model, "surface_band_sigma", 1.5))
        self._surface_canonicalize = bool(
            getattr(config.model, "surface_canonicalize", False)
        )
        self._surface_canonical_depth = int(
            getattr(config.model, "surface_canonical_depth", 24)
        )
        if self._surface_canonicalize and self._surface_canonical_depth < 8:
            raise ValueError("surface_canonical_depth must be at least 8 for three pooling levels")

        self.supcon_head: SupConHead | None = None
        if bool(getattr(config.tra, "supcon", False)):
            self.supcon_head = SupConHead(
                in_features=c1 if bool(getattr(config.model, "multitile", False)) else c4,
                proj_dim=int(getattr(config.tra, "supcon_proj_dim", 128)),
                hidden=int(getattr(config.tra, "supcon_hidden_dim", 256)),
            )

        self.domain_head: DomainClassifier | None = None
        if bool(getattr(config.tra, "dann", False)):
            n_domains = int(getattr(config.tra, "dann_n_domains", 0))
            if n_domains > 1:
                self.domain_head = DomainClassifier(in_features=c4, n_domains=n_domains)

        self.prototype_head: PrototypeHead | None = None
        if bool(getattr(config.model, "use_prototype", False)):
            self.prototype_head = PrototypeHead(
                feat_dim=c4,
                ema=float(getattr(config.model, "prototype_ema", 0.99)),
            )
        self._skip_drop = float(getattr(config.model, "skip_drop", 0.0))
        self._no_dz = bool(getattr(config.model, "no_dz", False))
        # multitile head: predict a grid x grid map of sub-tile logits over a (grid*subtile)px
        # center, instead of one scalar. sub-tile size is in feature-map px (subtile // downsample).
        self._multitile = bool(getattr(config.model, "multitile", False))
        self._mt_grid = max(1, int(getattr(config.model, "multitile_grid", 4)))
        self._mt_sub_feat = max(1, int(getattr(config.model, "multitile_subtile", 8)) // self._downsample)
        self._mt_center_feat = self._mt_grid * self._mt_sub_feat

        self.depth_profile_head: DepthProfileHead | None = None
        if bool(getattr(config.model, "use_depth_profile", False)):
            self.depth_profile_head = DepthProfileHead(depth=int(getattr(config.data, "depth", 24)))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if self._downsample > 1:
            x = F.avg_pool3d(
                x,
                kernel_size=(1, self._downsample, self._downsample),
                stride=(1, self._downsample, self._downsample),
            )
        return x

    def _stem_in(self, x: torch.Tensor) -> torch.Tensor:
        dz = torch.zeros_like(x)
        if not self._no_dz:
            dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return torch.cat([x, _lcn2d(x, 5), dz], dim=1)

    @staticmethod
    def _fiber_coordinates(x: torch.Tensor) -> torch.Tensor:
        """local tangent orientation and anisotropy from a per-slice structure tensor."""
        lcn = _lcn2d(x, 5)
        batch, channels, depth, height, width = lcn.shape
        flat = lcn.reshape(batch * depth, channels, height, width)
        gx = F.pad(flat, (1, 1, 0, 0), mode="replicate")[:, :, :, 2:] \
            - F.pad(flat, (1, 1, 0, 0), mode="replicate")[:, :, :, :-2]
        gy = F.pad(flat, (0, 0, 1, 1), mode="replicate")[:, :, 2:, :] \
            - F.pad(flat, (0, 0, 1, 1), mode="replicate")[:, :, :-2, :]
        jxx = F.avg_pool2d(gx.square(), 7, stride=1, padding=3)
        jyy = F.avg_pool2d(gy.square(), 7, stride=1, padding=3)
        jxy = F.avg_pool2d(gx * gy, 7, stride=1, padding=3)
        delta = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-6)
        trace = (jxx + jyy).clamp(min=1e-4)
        coherence = delta / trace
        cos2theta = (jxx - jyy) / delta
        sin2theta = 2.0 * jxy / delta
        features = torch.cat((coherence, cos2theta, sin2theta), dim=1)
        return features.reshape(batch, 3, depth, height, width)

    def _merge_skip(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="trilinear", align_corners=False)
        if self.training and self._skip_drop > 0:
            # zero whole skip connection with probability skip_drop; forces decoder bottleneck reliance
            mask = torch.bernoulli(torch.full((1,), 1.0 - self._skip_drop, device=skip.device))
            skip = skip * mask
        return torch.cat([upsampled, skip], dim=1)

    def _merge_skip_2d(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
        if self.training and self._skip_drop > 0:
            mask = torch.bernoulli(torch.full((1,), 1.0 - self._skip_drop, device=skip.device))
            skip = skip * mask
        return torch.cat([upsampled, skip], dim=1)

    def _apply_learned_surface(self, raw_x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.depth_surface_attn is None:
            self.last_surface_attn = None
            return features
        attn = self.depth_surface_attn(raw_x)
        self.last_surface_attn = attn.detach()
        return features * (1.0 + attn)

    @staticmethod
    def _surface_relative_resample(
        volume: torch.Tensor,
        center_depth: torch.Tensor,
        output_depth: int,
    ) -> torch.Tensor:
        """sample a per-column depth window centered on the estimated surface."""
        batch, _, source_depth, height, width = volume.shape
        output_depth = int(output_depth)
        relative = torch.linspace(
            -(output_depth - 1) / 2.0,
            (output_depth - 1) / 2.0,
            output_depth,
            device=volume.device,
            dtype=volume.dtype,
        ).view(1, output_depth, 1, 1)
        source_z = center_depth.squeeze(1).to(volume.dtype).unsqueeze(1) + relative
        z_grid = 2.0 * source_z / max(source_depth - 1, 1) - 1.0
        y_grid = torch.linspace(-1.0, 1.0, height, device=volume.device, dtype=volume.dtype)
        x_grid = torch.linspace(-1.0, 1.0, width, device=volume.device, dtype=volume.dtype)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        xx = xx.view(1, 1, height, width).expand(batch, output_depth, -1, -1)
        yy = yy.view(1, 1, height, width).expand(batch, output_depth, -1, -1)
        grid = torch.stack((xx, yy, z_grid), dim=-1)
        return F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    @staticmethod
    def _teacher_surface_features(
        depth: torch.Tensor,
        confidence: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """broadcast literal local depth and confidence into auxiliary 3D features."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(1)
        if depth.shape[-2:] != reference.shape[-2:]:
            depth = F.interpolate(depth.float(), size=reference.shape[-2:], mode="nearest")
            confidence = F.interpolate(
                confidence.float(),
                size=reference.shape[-2:],
                mode="nearest",
            )
        depth = depth.to(device=reference.device, dtype=reference.dtype)
        confidence = confidence.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        valid = ((depth >= 0) & (depth <= reference.shape[2] - 1)).to(reference.dtype)
        confidence = confidence * valid
        depth_scale = float(max(reference.shape[2] - 1, 1))
        depth_value = torch.where(valid > 0, depth / depth_scale, torch.zeros_like(depth))
        depth_axis = torch.arange(
            reference.shape[2],
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, 1, -1, 1, 1)
        depth_value = depth_value.unsqueeze(2).expand(-1, -1, reference.shape[2], -1, -1)
        confidence_volume = confidence.unsqueeze(2).expand_as(depth_value)
        signed_offset = (depth_axis - depth.unsqueeze(2)) / depth_scale
        signed_offset = signed_offset * valid.unsqueeze(2)
        return torch.cat((depth_value, confidence_volume, signed_offset), dim=1)

    def _encode_decode(
        self,
        x: torch.Tensor,
        teacher_surface_depth: torch.Tensor | None = None,
        teacher_surface_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_x = self._prepare_input(x)
        if self._surface_teacher_input:
            if teacher_surface_depth is None or teacher_surface_confidence is None:
                raise RuntimeError("surface_teacher_input requires depth and confidence maps")
            surface_for_backbone = self._teacher_surface_features(
                teacher_surface_depth,
                teacher_surface_confidence,
                raw_x,
            )
            if self._surface_canonicalize:
                center_depth = surface_for_backbone[:, 0:1, 0] * float(
                    max(raw_x.shape[2] - 1, 1)
                )
                valid = surface_for_backbone[:, 1:2, 0] > 0
                fallback = torch.full_like(center_depth, (raw_x.shape[2] - 1) / 2.0)
                center_depth = torch.where(valid, center_depth, fallback)
                raw_for_backbone = self._surface_relative_resample(
                    raw_x,
                    center_depth,
                    self._surface_canonical_depth,
                )
                surface_for_backbone = self._surface_relative_resample(
                    surface_for_backbone,
                    center_depth,
                    self._surface_canonical_depth,
                )
            else:
                raw_for_backbone = raw_x
            self.last_new_surface_logits = None
            self.last_new_surface_probs = None
            self.last_surface_target = None
            self.last_surface_valid = None
        elif self.new_surface_head is not None and self.new_surface_input is not None:
            surface_logits = self.new_surface_head(raw_x)
            surface_probs = F.softmax(surface_logits, dim=2)
            self.last_new_surface_logits = surface_logits
            if self._surface_canonicalize:
                depth_axis = torch.arange(
                    raw_x.shape[2],
                    device=raw_x.device,
                    dtype=surface_probs.dtype,
                ).view(1, 1, -1, 1, 1)
                center_depth = (surface_probs * depth_axis).sum(dim=2)
                self.last_surface_target = None
                self.last_surface_valid = None
                raw_for_backbone = self._surface_relative_resample(
                    raw_x,
                    center_depth,
                    self._surface_canonical_depth,
                )
                surface_for_backbone = self._surface_relative_resample(
                    surface_probs.detach(),
                    center_depth,
                    self._surface_canonical_depth,
                )
            else:
                self.last_surface_target = None
                self.last_surface_valid = None
                raw_for_backbone = raw_x
                surface_for_backbone = surface_probs
            self.last_new_surface_probs = surface_for_backbone.detach()
        else:
            self.last_new_surface_logits = None
            self.last_new_surface_probs = None
            self.last_surface_target = None
            self.last_surface_valid = None
            raw_for_backbone = raw_x
            surface_for_backbone = None

        stem_x = self._stem_in(raw_for_backbone)
        enc1 = self.enc1(stem_x)
        if self.fiber_coordinate_input is not None:
            enc1 = enc1 + self.fiber_coordinate_input(
                self._fiber_coordinates(raw_for_backbone)
            )
        if surface_for_backbone is not None:
            enc1 = enc1 + self.new_surface_input(surface_for_backbone)
        enc1 = self._apply_learned_surface(raw_for_backbone, enc1)
        if self.mednext1 is not None:
            enc1 = self.mednext1(enc1)
        if self._enc1_drop is not None:
            enc1 = self._enc1_drop(enc1)
        enc2 = self.enc2(self.pool(enc1))
        if self.mednext2 is not None:
            enc2 = self.mednext2(enc2)
        if self._enc2_drop is not None:
            enc2 = self._enc2_drop(enc2)
        enc3 = self.enc3(self.pool(enc2))
        if self.mednext3 is not None:
            enc3 = self.mednext3(enc3)
        if self.divided_attention is not None:
            enc3 = self.divided_attention(enc3)
        bottleneck = self.bottleneck(self.pool3(enc3))
        if self.mednext_bottleneck is not None:
            bottleneck = self.mednext_bottleneck(bottleneck)

        dec3 = self.dec3(self._merge_skip(self.up3(bottleneck), enc3))
        dec2 = self.dec2(self._merge_skip(self.up2(dec3), enc2))
        dec1 = self.dec1(self._merge_skip(self.up1(dec2), enc1))
        if self._head_drop is not None:
            dec1 = self._head_drop(dec1)
        return bottleneck, dec1

    def _encode_decode_early_2d(
        self,
        x: torch.Tensor,
        teacher_surface_depth: torch.Tensor | None,
        teacher_surface_confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """use a local 3D stem, then spend the remaining U-Net capacity on sheet-tangent XY."""
        if self.early_depth_attn is None or self.early_depth_fuse is None:
            raise RuntimeError("early 2D U-Net modules are not initialized")
        raw = self._prepare_input(x)
        surface = None
        if self._surface_teacher_input:
            if teacher_surface_depth is None or teacher_surface_confidence is None:
                raise RuntimeError("surface_teacher_input requires depth and confidence maps")
            surface = self._teacher_surface_features(
                teacher_surface_depth,
                teacher_surface_confidence,
                raw,
            )
        elif self.new_surface_head is not None:
            raise ValueError("early_2d_unet currently requires literal or disabled surface input")
        features3d = self.enc1(self._stem_in(raw))
        if surface is not None and self.new_surface_input is not None:
            features3d = features3d + self.new_surface_input(surface)
        if self.fiber_coordinate_input is not None:
            features3d = features3d + self.fiber_coordinate_input(self._fiber_coordinates(raw))
        if self.mednext1 is not None:
            features3d = self.mednext1(features3d)
        if self._enc1_drop is not None:
            features3d = self._enc1_drop(features3d)
        weights = torch.softmax(self.early_depth_attn(features3d), dim=2)
        weighted = (features3d * weights).sum(dim=2)
        strongest = features3d.amax(dim=2)
        enc1 = self.early_depth_fuse(torch.cat((weighted, strongest), dim=1))
        enc2 = self.early2d_enc2(F.max_pool2d(enc1, 2))
        if self._enc2_drop is not None:
            enc2 = F.dropout2d(enc2, p=self._enc2_drop.p, training=self.training)
        enc3 = self.early2d_enc3(F.max_pool2d(enc2, 2))
        bottleneck = self.early2d_bottleneck(F.max_pool2d(enc3, 2))
        dec3 = self.early2d_dec3(self._merge_skip_2d(self.early2d_up3(bottleneck), enc3))
        dec2 = self.early2d_dec2(self._merge_skip_2d(self.early2d_up2(dec3), enc2))
        dec1 = self.early2d_dec1(self._merge_skip_2d(self.early2d_up1(dec2), enc1))
        if self._head_drop is not None:
            dec1 = F.dropout2d(dec1, p=self._head_drop.p, training=self.training)
        return bottleneck, dec1

    @staticmethod
    def _embedding(bottleneck: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(bottleneck, output_size=1).flatten(1)

    def _crop_to_center_tile(self, voxel_map: torch.Tensor) -> torch.Tensor:
        """crop voxel map to the center tile region before bag aggregation.

        when context_size > tile_size, the decoded map covers the full context window
        (ctx/ds per side) but the label only covers the center tile (tile/ds per side).
        aggregating the full map makes scores context-dominated, causing ~ctx-sized blobs.
        cropping to the center tile anchors the score to the labeled footprint.
        """
        if self._context_size <= self._tile_size:
            return voxel_map
        H = voxel_map.shape[3]
        t = self._tile_size // self._downsample   # tile extent in feature-map pixels
        t = max(1, t)
        cs = (H - t) // 2                        # top-left of center crop
        return voxel_map[:, :, :, cs:cs + t, cs:cs + t]

    def _crop_center_feat(
        self,
        voxel_map: torch.Tensor,
        feat: int,
        target_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """crop a target-aligned feat x feat region, optionally displaced from center."""
        H, W = voxel_map.shape[3], voxel_map.shape[4]
        cy = max(0, (H - feat) // 2)
        cx = max(0, (W - feat) // 2)
        if target_offsets is None:
            return voxel_map[:, :, :, cy:cy + feat, cx:cx + feat]

        offsets = torch.div(
            target_offsets.to(device=voxel_map.device, dtype=torch.long),
            self._downsample,
            rounding_mode="trunc",
        )
        rows = cy + offsets[:, 0:1] + torch.arange(feat, device=voxel_map.device)
        cols = cx + offsets[:, 1:2] + torch.arange(feat, device=voxel_map.device)
        rows = rows.clamp(0, H - 1)
        cols = cols.clamp(0, W - 1)
        B, C, D = voxel_map.shape[:3]
        row_index = rows[:, None, None, :, None].expand(B, C, D, feat, W)
        cropped_rows = torch.gather(voxel_map, dim=3, index=row_index)
        col_index = cols[:, None, None, None, :].expand(B, C, D, feat, feat)
        return torch.gather(cropped_rows, dim=4, index=col_index)

    def _multitile_aggregate(self, center: torch.Tensor) -> torch.Tensor:
        """aggregate the (B,1,D,cf,cf) center crop into a (B, grid*grid) map of per-sub-tile
        logits via per-cell log-sum-exp. cell (iy, ix) pools its D*sub*sub voxel bag; the flat
        output index is iy*grid + ix (row-major, iy indexes y) to match the label ordering."""
        n, sub = self._mt_grid, self._mt_sub_feat
        c = center.squeeze(1)                              # (B, D, cf, cf)
        B, D, cf, _ = c.shape
        c = c.reshape(B, D, n, sub, n, sub).permute(0, 2, 4, 1, 3, 5).reshape(B, n * n, D * sub * sub)
        r = self.lse_r.clamp(min=0.5, max=10.0)
        m = c.new_tensor(float(c.shape[-1]))
        return (1.0 / r) * (torch.logsumexp(r * c, dim=2) - torch.log(m))   # (B, grid*grid)

    def _multitile_spatial_instances(self, center: torch.Tensor) -> torch.Tensor:
        """depth-collapse voxel logits and return locally supported XY instances per cell."""
        n, sub = self._mt_grid, self._mt_sub_feat
        values = center.squeeze(1)
        r = self.lse_r.clamp(min=0.5, max=10.0)
        depth_count = values.new_tensor(float(values.shape[1]))
        spatial = (
            torch.logsumexp(r * values, dim=1) - torch.log(depth_count)
        ) / r
        kernel = self._minimum_support_kernel
        if kernel > 1:
            padded = F.pad(
                spatial.unsqueeze(1),
                (kernel // 2,) * 4,
                mode="replicate",
            )
            spatial = F.avg_pool2d(padded, kernel, stride=1).squeeze(1)
        batch = spatial.shape[0]
        return spatial.reshape(batch, n, sub, n, sub).permute(
            0, 1, 3, 2, 4
        ).reshape(batch, n * n, sub * sub)

    def _multitile_minimum_support(self, center: torch.Tensor, k: int) -> torch.Tensor:
        instances = self._multitile_spatial_instances(center)
        k = min(max(1, int(k)), instances.shape[-1])
        return torch.topk(instances, k, dim=-1).values.mean(dim=-1)

    def _multitile_weldon(self, center: torch.Tensor, k: int) -> torch.Tensor:
        instances = self._multitile_spatial_instances(center)
        k = min(max(1, int(k)), instances.shape[-1])
        positive = torch.topk(instances, k, dim=-1).values.mean(dim=-1)
        negative = torch.topk(instances, k, dim=-1, largest=False).values.mean(dim=-1)
        return 0.5 * (positive + negative)

    def _multitile_feature_depth_fusion(
        self,
        decoded: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fuse the short surface-normal profile before spatial multitile LSE."""
        if self.depth_fusion_attn is None or self.depth_fusion_head is None:
            raise RuntimeError("feature depth fusion modules are not initialized")
        depth_weights = torch.softmax(self.depth_fusion_attn(decoded), dim=2)
        weighted = (decoded * depth_weights).sum(dim=2)
        strongest = decoded.amax(dim=2)
        fused = self.depth_fusion_head(torch.cat((weighted, strongest), dim=1))
        center = self._crop_center_feat(
            fused.unsqueeze(2),
            self._mt_center_feat,
            target_offsets,
        ).squeeze(1).squeeze(1)
        n, sub = self._mt_grid, self._mt_sub_feat
        batch = center.shape[0]
        cells = center.reshape(batch, n, sub, n, sub).permute(
            0, 1, 3, 2, 4
        ).reshape(batch, n * n, sub * sub)
        r = self.lse_r.clamp(min=0.5, max=10.0)
        count = cells.new_tensor(float(cells.shape[-1]))
        return (torch.logsumexp(r * cells, dim=2) - torch.log(count)) / r

    def _multitile_attn_aggregate(self, center: torch.Tensor) -> torch.Tensor:
        """per-sub-tile gated attention-MIL: fold the grid into the batch dim so one attn_mil
        call pools every 8px sub-tile bag (D*sub*sub voxels) independently. returns (B, grid*grid)
        in iy*grid+ix order and sets last_attn_entropy_loss (averaged over all sub-tile bags)."""
        n, sub = self._mt_grid, self._mt_sub_feat
        c = center.squeeze(1)                              # (B, D, cf, cf)
        B, D, cf, _ = c.shape
        c = c.reshape(B, D, n, sub, n, sub).permute(0, 2, 4, 1, 3, 5).reshape(B * n * n, 1, D, sub, sub)
        score, entropy_loss = self.attn_mil(c, entropy_weight=self._attn_entropy_weight)
        self.last_attn_entropy_loss = entropy_loss
        per_bag = self.attn_mil.last_entropy_loss_per_bag
        self.last_attn_entropy_per_target = (
            per_bag.view(B, n * n) if per_bag is not None else None
        )
        return score.view(B, n * n)

    def _multitile_feature_bags(
        self,
        decoded: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """fold centered decoder features into independent multitile bags."""
        n, sub = self._mt_grid, self._mt_sub_feat
        center = self._crop_center_feat(decoded, self._mt_center_feat, target_offsets)
        B, C, D, _, _ = center.shape
        bags = center.reshape(B, C, D, n, sub, n, sub).permute(
            0, 3, 5, 1, 2, 4, 6
        ).reshape(B * n * n, C, D, sub, sub)
        return bags, B

    def _multitile_feature_attn_aggregate(
        self,
        decoded: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """attention-MIL over decoder feature vectors rather than scalar voxel logits."""
        bags, batch = self._multitile_feature_bags(decoded, target_offsets)
        score, entropy_loss = self.feature_attn_mil(
            bags,
            entropy_weight=self._attn_entropy_weight,
        )
        self.last_attn_entropy_loss = entropy_loss
        per_bag = self.feature_attn_mil.last_entropy_loss_per_bag
        self.last_attn_entropy_per_target = (
            per_bag.view(batch, self._mt_grid * self._mt_grid) if per_bag is not None else None
        )
        return score.view(batch, self._mt_grid * self._mt_grid)

    def _multitile_embeddings(
        self,
        decoded: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """one decoder embedding per multitile target for target-aligned SupCon."""
        bags, batch = self._multitile_feature_bags(decoded, target_offsets)
        pooled = bags.mean(dim=(2, 3, 4))
        return pooled.view(batch, self._mt_grid * self._mt_grid, -1)

    def _multitile_aggregate_2d(
        self,
        voxel_map: torch.Tensor,
        target_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        center = self._crop_center_feat(
            voxel_map.unsqueeze(2),
            self._mt_center_feat,
            target_offsets,
        ).squeeze(1).squeeze(1)
        n, sub = self._mt_grid, self._mt_sub_feat
        batch = center.shape[0]
        cells = center.reshape(batch, n, sub, n, sub).permute(
            0, 1, 3, 2, 4
        ).reshape(batch, n * n, sub * sub)
        r = self.lse_r.clamp(min=0.5, max=10.0)
        count = cells.new_tensor(float(cells.shape[-1]))
        return (torch.logsumexp(r * cells, dim=-1) - torch.log(count)) / r

    def _multitile_embeddings_2d(
        self,
        decoded: torch.Tensor,
        target_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        center = self._crop_center_feat(
            decoded.unsqueeze(2),
            self._mt_center_feat,
            target_offsets,
        ).squeeze(2)
        n, sub = self._mt_grid, self._mt_sub_feat
        batch, channels = center.shape[:2]
        bags = center.reshape(batch, channels, n, sub, n, sub).permute(
            0, 2, 4, 1, 3, 5
        ).reshape(batch, n * n, channels, sub * sub)
        return bags.mean(dim=-1)

    def _surface_guided_aggregate(
        self,
        center_voxels: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """pool a soft surface band and return an entropy-derived blend weight."""
        if self.last_new_surface_probs is None:
            return None, None

        n, sub = self._mt_grid, self._mt_sub_feat
        surface = self._crop_center_feat(
            self.last_new_surface_probs,
            self._mt_center_feat,
            target_offsets,
        )
        depth_axis = torch.arange(
            surface.shape[2],
            device=surface.device,
            dtype=surface.dtype,
        ).view(1, 1, -1, 1, 1)
        expected_depth = (surface * depth_axis).sum(dim=2, keepdim=True)
        sigma = max(self._surface_band_sigma, 0.25)
        band = torch.exp(-0.5 * ((depth_axis - expected_depth) / sigma) ** 2)
        band = band / band.sum(dim=2, keepdim=True).clamp(min=1e-8)

        surface_map = (center_voxels * band).sum(dim=2).squeeze(1)
        B, cf, _ = surface_map.shape
        cells = surface_map.reshape(B, n, sub, n, sub).permute(
            0, 1, 3, 2, 4
        ).reshape(B, n * n, sub * sub)
        r = self.lse_r.clamp(min=0.5, max=10.0)
        count = cells.new_tensor(float(cells.shape[-1]))
        guided = (torch.logsumexp(r * cells, dim=2) - torch.log(count)) / r

        entropy = -(surface * surface.clamp(min=1e-8).log()).sum(dim=2).squeeze(1)
        confidence = 1.0 - entropy / math.log(float(surface.shape[2]))
        confidence = confidence.reshape(B, n, sub, n, sub).permute(
            0, 1, 3, 2, 4
        ).reshape(B, n * n, sub * sub).mean(dim=2)
        alpha = (self._surface_guided_mix * confidence).clamp(min=0.0, max=1.0)
        return guided, alpha

    def _bag_score(self, voxel_map: torch.Tensor) -> torch.Tensor:
        self.last_attn_entropy_per_target = None
        if self.attn_mil is not None:
            score, entropy_loss = self.attn_mil(voxel_map, entropy_weight=self._attn_entropy_weight)
            self.last_attn_entropy_loss = entropy_loss
            return score
        self.last_attn_entropy_loss = voxel_map.new_zeros(())
        return _mil_lse(voxel_map, self.lse_r)

    def forward_with_extras(
        self,
        x: torch.Tensor,
        grl_scale: float = 1.0,
        target_offsets: torch.Tensor | None = None,
        teacher_surface_depth: torch.Tensor | None = None,
        teacher_surface_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._early_2d_unet:
            bottleneck2d, decoded2d = self._encode_decode_early_2d(
                x,
                teacher_surface_depth,
                teacher_surface_confidence,
            )
            voxel2d = self.early2d_head(decoded2d)
            center2d = self._crop_center_feat(
                voxel2d.unsqueeze(2),
                self._mt_center_feat,
                target_offsets,
            ).clone()
            self.last_voxel_map = None if self.training else voxel2d.unsqueeze(2).detach().clone()
            self.last_voxel_map_full = voxel2d.unsqueeze(2)
            self.last_center_voxel_map = center2d
            self.last_clam_instance_logits = None
            embedding = F.adaptive_avg_pool2d(bottleneck2d, 1).flatten(1)
            self.last_embedding_detached = embedding.detach().clone()
            domain_logits = (
                self.domain_head(embedding, grl_scale=grl_scale)
                if self.domain_head is not None else None
            )
            supcon_z = (
                self.supcon_head(self._multitile_embeddings_2d(decoded2d, target_offsets))
                if self.supcon_head is not None else None
            )
            score = self._multitile_aggregate_2d(voxel2d, target_offsets)
            self.last_attn_entropy_loss = voxel2d.new_zeros(())
            self.last_attn_entropy_per_target = None
            self.last_surface_guided_alpha = None
            return score, embedding, domain_logits, supcon_z

        bottleneck, decoded = self._encode_decode(
            x,
            teacher_surface_depth=teacher_surface_depth,
            teacher_surface_confidence=teacher_surface_confidence,
        )
        voxel_map = self.out_head(decoded)
        # break output aliasing so torch.compile's AOTAutograd doesn't hit its alias-regen bug
        # ('TensorAlias' has no attribute 'is_complex'). last_voxel_map is vis-only (read at eval),
        # so skip the full-size copy during training; clone (not view) otherwise.
        self.last_voxel_map = None if self.training else voxel_map.detach().clone()
        self.last_voxel_map_full = voxel_map  # non-detached; needed for spill_entropy gradient flow
        # clone (grad-preserving) so the center crop owns its storage: it is stored for the spill
        # loss (needs grad) and must not alias voxel_map / last_voxel_map_full
        if self._multitile:
            center_voxels = self._crop_center_feat(
                voxel_map,
                self._mt_center_feat,
                target_offsets,
            ).clone()
        else:
            center_voxels = self._crop_to_center_tile(voxel_map).clone()
        self.last_center_voxel_map = center_voxels
        self.last_clam_instance_logits = (
            self._multitile_spatial_instances(center_voxels)
            if self._multitile and self._clam_instance else None
        )
        embedding = self._embedding(bottleneck)
        self.last_embedding_detached = embedding.detach().clone()
        domain_logits = self.domain_head(embedding, grl_scale=grl_scale) if self.domain_head is not None else None
        if self.supcon_head is not None:
            supcon_input = (
                self._multitile_embeddings(decoded, target_offsets)
                if self._multitile else embedding
            )
            supcon_z = self.supcon_head(supcon_input)
        else:
            supcon_z = None
        if self._multitile:
            if self._feature_depth_fusion:
                score = self._multitile_feature_depth_fusion(decoded, target_offsets)
            elif self._minimum_support_k > 0:
                score = self._multitile_minimum_support(
                    center_voxels,
                    self._minimum_support_k,
                )
            elif self._weldon_k > 0:
                score = self._multitile_weldon(center_voxels, self._weldon_k)
            elif self.feature_attn_mil is not None:
                score = self._multitile_feature_attn_aggregate(decoded, target_offsets)
            elif self.attn_mil is not None:
                score = self._multitile_attn_aggregate(center_voxels)
            else:
                score = self._multitile_aggregate(center_voxels)
            if self._surface_guided_mil:
                guided, alpha = self._surface_guided_aggregate(center_voxels, target_offsets)
                if guided is not None and alpha is not None:
                    score = (1.0 - alpha) * score + alpha * guided
                    self.last_surface_guided_alpha = alpha.detach()
                else:
                    self.last_surface_guided_alpha = None
            else:
                self.last_surface_guided_alpha = None
        elif self.depth_profile_head is not None:
            score = self.depth_profile_head(center_voxels)
        elif self.prototype_head is not None:
            score = self.prototype_head(embedding)
        else:
            score = self._bag_score(center_voxels)
        return score, embedding, domain_logits, supcon_z

    def forward(
        self,
        x: torch.Tensor,
        target_offsets: torch.Tensor | None = None,
        teacher_surface_depth: torch.Tensor | None = None,
        teacher_surface_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score, _, _, _ = self.forward_with_extras(
            x,
            target_offsets=target_offsets,
            teacher_surface_depth=teacher_surface_depth,
            teacher_surface_confidence=teacher_surface_confidence,
        )
        return score


InkDetectorArch = NnUnet3dLcndz


_ARCH_MAP = {
    "nnunet3d_lcndz": NnUnet3dLcndz,
    "nnunet3d_lcndz_attn": NnUnet3dLcndz,
    "v16_arch_ctx": NnUnet3dLcndz,
}


def create_model(config: Config):
    """instantiate and initialize the current production model."""
    arch = str(getattr(config.model, "arch", "nnunet3d_lcndz")).lower()
    if arch not in _ARCH_MAP:
        valid = ", ".join(sorted(_ARCH_MAP))
        raise ValueError(f"unknown arch '{arch}'; supported: {valid}")

    model = _ARCH_MAP[arch](config).to(config.device)

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=0.8)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.InstanceNorm2d, nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            _init_norm(module)

    model.apply(init_weights)
    if model.new_surface_input is not None:
        # begin as the pretrained baseline while the auxiliary loss trains the new head
        nn.init.zeros_(model.new_surface_input.weight)
    if model.fiber_coordinate_input is not None:
        nn.init.zeros_(model.fiber_coordinate_input.weight)
    if model.depth_fusion_attn is not None:
        nn.init.zeros_(model.depth_fusion_attn.weight)
        nn.init.zeros_(model.depth_fusion_attn.bias)
    if model.early_depth_attn is not None:
        nn.init.zeros_(model.early_depth_attn.weight)
        nn.init.zeros_(model.early_depth_attn.bias)
    for module in model.modules():
        if isinstance(module, DepthSurfaceAttn):
            module.reset_output_layer()
        elif isinstance(module, MedNeXtAdapter3d):
            module.reset_output()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({arch}): {params:,}")
    # keep an EAGER handle for the figure/probe predict path: it feeds variable batch/tile
    # shapes that make torch.compile recompile on every chunk (dynamo/inductor runs on CPU,
    # GPU idles ~0%). training uses one fixed shape, so it keeps the compiled hot path below.
    # torch.compile requires pytorch >= 2.0; skip silently on older installs.
    model._eager_forward_with_extras = model.forward_with_extras
    if bool(getattr(config.model, "compile_model", True)) and hasattr(torch, "compile"):
        model.forward_with_extras = torch.compile(model.forward_with_extras)
    return model, params