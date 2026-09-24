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


class ContinuousStyleFiLM3d(nn.Module):
    """condition early features on continuous per-depth acquisition statistics."""

    def __init__(self, depth: int, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * depth, hidden),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(hidden, 2 * channels),
        )
        self.channels = channels

    def forward(self, features: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        mean = raw.mean(dim=(1, 3, 4))
        std = raw.std(dim=(1, 3, 4), unbiased=False)
        gamma, beta = self.net(torch.cat((mean, std), dim=1)).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma).view(-1, self.channels, 1, 1, 1)
        beta = 0.25 * torch.tanh(beta).view(-1, self.channels, 1, 1, 1)
        return features * (1.0 + gamma) + beta


class SagNetStyleHead(nn.Module):
    """predict target presence from early feature style through gradient reversal."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(hidden, 1),
        )

    def forward(self, style: torch.Tensor, grl_scale: float) -> torch.Tensor:
        return self.net(grad_reverse(style, grl_scale)).squeeze(1)


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temp: float = 0.07,
    domain_ids: torch.Tensor | None = None,
    ignore_same_domain_same_class: bool = False,
) -> torch.Tensor:
    """supervised contrastive loss with optional cross-domain positive filtering."""
    batch = z.shape[0]
    if batch < 2:
        return z.new_zeros(())

    sim = torch.mm(z, z.T) / temp
    labels = labels.view(-1)
    eye = torch.eye(batch, dtype=torch.bool, device=z.device)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_class & (~eye)
    denominator_mask = ~eye
    if domain_ids is not None:
        d = domain_ids.view(-1)
        same_domain = d.unsqueeze(0) == d.unsqueeze(1)
        pos_mask = pos_mask & (~same_domain)
        if ignore_same_domain_same_class:
            denominator_mask = denominator_mask & (~(same_domain & same_class))

    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(~denominator_mask, 0.0)
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


def _norm3d(channels: int, mode: str, allow_ibn: bool = False) -> nn.Module:
    if mode == "batch":
        return nn.BatchNorm3d(channels, affine=True)
    if mode == "ibn_full":
        return IBN3d(channels)
    if mode == "ibn" and allow_ibn:
        return IBN3d(channels)
    if mode in ("instance", "ibn"):
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"unknown 3D normalization mode: {mode!r}")


class ConvBlock3d(nn.Module):
    """two-conv nnU-Net block with instance norm and leaky relu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_ibn: bool = False,
        norm_mode: str | None = None,
    ):
        super().__init__()
        mode = str(norm_mode or ("ibn" if use_ibn else "instance"))
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(out_channels, mode, allow_ibn=use_ibn),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(out_channels, mode),
            nn.LeakyReLU(0.01, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock3d(nn.Module):
    """two-conv residual block that preserves baseline convolution key names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_ibn: bool = False,
        norm_mode: str | None = None,
    ):
        super().__init__()
        self.net = ConvBlock3d(
            in_channels,
            out_channels,
            use_ibn=use_ibn,
            norm_mode=norm_mode,
        ).net
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.net(x) + self.shortcut(x), 0.01, inplace=False)


def _planarize_depth_convs(module: nn.Module) -> None:
    """swap depth-spanning conv kernels for per-slice (1,k,k) kernels."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d) and child.kernel_size[0] > 1:
            setattr(module, name, nn.Conv3d(
                child.in_channels,
                child.out_channels,
                kernel_size=(1, *child.kernel_size[1:]),
                stride=(1, *child.stride[1:]),
                padding=(0, *child.padding[1:]),
                dilation=(1, *child.dilation[1:]),
                groups=child.groups,
                bias=child.bias is not None,
            ))
        else:
            _planarize_depth_convs(child)


class GatedCueStem3d(nn.Module):
    """independently refine raw, LCN, and dZ cues before content-adaptive fusion."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, hidden, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(hidden, affine=True),
                nn.GELU(),
                nn.Conv3d(hidden, 1, kernel_size=1),
            )
            for _ in range(3)
        ])
        self.gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def reset_identity(self) -> None:
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        statistics = torch.cat(
            (cues.mean(dim=(2, 3, 4)), cues.std(dim=(2, 3, 4), unbiased=False)),
            dim=1,
        )
        weights = 3.0 * torch.softmax(self.gate(statistics), dim=1)
        outputs = []
        for index, branch in enumerate(self.branches):
            cue = cues[:, index:index + 1]
            weight = weights[:, index:index + 1, None, None, None]
            outputs.append(weight * (cue + branch(cue)))
        return torch.cat(outputs, dim=1)


class FactorizedConvBlock3d(nn.Module):
    """two spatial-then-depth (2+1)D stages with explicit nonlinear separation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_mode: str,
        use_ibn: bool = False,
    ):
        super().__init__()
        self.spatial1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False,
        )
        self.spatial1_norm = _norm3d(out_channels, norm_mode, allow_ibn=use_ibn)
        self.depth1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.depth1_norm = _norm3d(out_channels, norm_mode)
        self.spatial2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False,
        )
        self.spatial2_norm = _norm3d(out_channels, norm_mode)
        self.depth2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.depth2_norm = _norm3d(out_channels, norm_mode)

    @staticmethod
    def _activate(value: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(value, negative_slope=0.01, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._activate(self.spatial1_norm(self.spatial1(x)))
        x = self._activate(self.depth1_norm(self.depth1(x)))
        x = self._activate(self.spatial2_norm(self.spatial2(x)))
        return self._activate(self.depth2_norm(self.depth2(x)))


class ConvBlock2d(nn.Module):
    """two-conv 2D U-Net block used after early surface-normal fusion."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("2D block depth must be positive")
        layers = []
        for index in range(depth):
            layers.extend((
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.01, inplace=False),
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock2d(nn.Module):
    """configurable-depth residual block for the collapsed 2D U-Net."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2):
        super().__init__()
        if depth < 1:
            raise ValueError("2D block depth must be positive")
        layers = []
        for index in range(depth):
            layers.append(nn.Conv2d(
                in_channels if index == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ))
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            if index + 1 < depth:
                layers.append(nn.LeakyReLU(0.01, inplace=False))
        self.net = nn.Sequential(*layers)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.net(x) + self.shortcut(x), 0.01, inplace=False)


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


class LocalUNet3d(nn.Module):
    """independent full U-Net expert over the supervised local field."""

    def __init__(self, in_channels: int, channels: tuple[int, int, int, int], norm_mode: str):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.pool = nn.MaxPool3d(2)
        self.enc1 = ConvBlock3d(in_channels, c1, norm_mode=norm_mode)
        self.enc2 = ConvBlock3d(c1, c2, norm_mode=norm_mode)
        self.enc3 = ConvBlock3d(c2, c3, norm_mode=norm_mode)
        self.bottleneck = ConvBlock3d(c3, c4, norm_mode=norm_mode)
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(c3 * 2, c3, norm_mode=norm_mode)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(c2 * 2, c2, norm_mode=norm_mode)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c1 * 2, c1, norm_mode=norm_mode)
        self.head = nn.Conv3d(c1, 1, kernel_size=1)

    @staticmethod
    def _merge(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return torch.cat((upsampled, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(self._merge(self.up3(bottleneck), enc3))
        dec2 = self.dec2(self._merge(self.up2(dec3), enc2))
        dec1 = self.dec1(self._merge(self.up1(dec2), enc1))
        return self.head(dec1)


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
        self.last_sagnet_logits: torch.Tensor | None = None
        self.last_depth_shift_logits: torch.Tensor | None = None

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
        norm_mode = str(getattr(config.model, "norm_mode", "auto")).lower()
        if norm_mode == "auto":
            norm_mode = "ibn" if use_ibn else "instance"
        if norm_mode not in ("instance", "ibn", "ibn_full", "batch"):
            raise ValueError("norm_mode must be one of: auto, instance, ibn, ibn_full, batch")
        self._norm_mode = norm_mode
        self._factorized_2plus1d = bool(
            getattr(config.model, "factorized_2plus1d", False)
        )
        self._residual_unet = bool(getattr(config.model, "residual_unet", False))
        self._cue_dropout = float(getattr(config.model, "cue_dropout", 0.0))
        self._raw_only_stem = bool(getattr(config.model, "raw_only_stem", False))
        self._explicit_depth_channels = bool(
            getattr(config.model, "explicit_depth_channels", False)
        )
        self._depth_antialias = bool(getattr(config.model, "depth_antialias", False))
        self._overlapping_depth_windows = bool(
            getattr(config.model, "overlapping_depth_windows", False)
        )
        self._overlap_window_size = int(
            getattr(config.model, "overlapping_depth_window_size", 4)
        )
        self._overlap_window_stride = int(
            getattr(config.model, "overlapping_depth_window_stride", 2)
        )
        if not 0.0 <= self._cue_dropout <= 1.0:
            raise ValueError("cue_dropout must be in [0, 1]")
        if self._raw_only_stem and bool(getattr(config.model, "gated_stems", False)):
            raise ValueError("raw_only_stem and gated_stems are mutually exclusive")
        if self._raw_only_stem and self._explicit_depth_channels:
            raise ValueError("raw_only_stem and explicit_depth_channels are mutually exclusive")
        if self._factorized_2plus1d and self._residual_unet:
            raise ValueError("factorized_2plus1d and residual_unet are mutually exclusive")
        # width multiplier on the 32/64/128/256 channel ladder (0.5 = half -> ~4x fewer conv FLOPs)
        _m = float(getattr(config.model, "channels_mult", 1.0) or 1.0)
        c1, c2, c3, c4 = (max(1, int(round(ch * _m))) for ch in (32, 64, 128, 256))

        def block3d(in_channels: int, out_channels: int, shallow: bool = False):
            if self._factorized_2plus1d:
                return FactorizedConvBlock3d(
                    in_channels,
                    out_channels,
                    norm_mode,
                    use_ibn=shallow and norm_mode == "ibn",
                )
            if self._residual_unet:
                return ResidualConvBlock3d(
                    in_channels,
                    out_channels,
                    use_ibn=shallow and norm_mode == "ibn",
                    norm_mode=norm_mode,
                )
            return ConvBlock3d(
                in_channels,
                out_channels,
                use_ibn=shallow and norm_mode == "ibn",
                norm_mode=norm_mode,
            )

        stem_channels = 1 if self._raw_only_stem else (5 if self._explicit_depth_channels else 3)
        self.enc1 = block3d(stem_channels, c1, shallow=True)
        self.gated_cue_stem = (
            GatedCueStem3d()
            if bool(getattr(config.model, "gated_stems", False)) else None
        )
        if bool(getattr(config.model, "planar_early_convs", False)):
            if not bool(getattr(config.model, "early_2d_unet", False)):
                raise ValueError("planar_early_convs requires early_2d_unet")
            _planarize_depth_convs(self.enc1)
            if self.gated_cue_stem is not None:
                _planarize_depth_convs(self.gated_cue_stem)
        self.enc2 = block3d(c1, c2, shallow=True)
        self.enc3 = block3d(c2, c3)
        self.bottleneck = block3d(c3, c4)

        self._mixstyle = bool(getattr(config.model, "mixstyle", False))
        self._mixstyle_prob = float(getattr(config.model, "mixstyle_prob", 0.8))
        self._mixstyle_alpha = float(getattr(config.model, "mixstyle_alpha", 0.1))
        self._sagnet = bool(getattr(config.model, "sagnet", False))
        if self._mixstyle and self._sagnet:
            raise ValueError("SagNet already applies style mixing and cannot combine with MixStyle")
        self.sagnet_style_head = SagNetStyleHead(c1) if self._sagnet else None
        self.style_film = (
            ContinuousStyleFiLM3d(
                input_depth,
                c1,
                int(getattr(config.model, "style_film_hidden", 64)),
            )
            if bool(getattr(config.model, "style_film", False)) else None
        )

        # spatial channel dropout after early encoder stages and before classification head
        _d1 = float(getattr(config.model, "conv1_drop", 0.0))
        _d2 = float(getattr(config.model, "conv2_drop", 0.0))
        _dh = float(getattr(config.model, "head_drop", 0.0))
        self._enc1_drop = nn.Dropout3d(p=_d1) if _d1 > 0 else None
        self._enc2_drop = nn.Dropout3d(p=_d2) if _d2 > 0 else None
        self._head_drop = nn.Dropout3d(p=_dh) if _dh > 0 else None

        up3_kernel = (1, 2, 2) if self._allow_depth4 else 2
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=up3_kernel, stride=up3_kernel)
        self.dec3 = block3d(c3 * 2, c3)   # cat(up3, enc3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = block3d(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = block3d(c1 * 2, c1)
        self.out_head = nn.Conv3d(c1, 1, kernel_size=1, bias=True)
        self._sparse_deep_supervision = bool(
            getattr(config.model, "sparse_deep_supervision", False)
        )
        self.deep_supervision_dec2 = (
            nn.Conv3d(c2, 1, kernel_size=1)
            if self._sparse_deep_supervision else None
        )
        self.deep_supervision_dec3 = (
            nn.Conv3d(c3, 1, kernel_size=1)
            if self._sparse_deep_supervision else None
        )
        self.last_deep_supervision_scores: tuple[torch.Tensor, torch.Tensor] | None = None

        self._dual_scale = bool(getattr(config.model, "dual_scale", False))
        self._dual_scale_local_size = int(
            getattr(config.model, "dual_scale_local_size", 64)
        )
        self._dual_scale_mix = float(getattr(config.model, "dual_scale_mix", 0.25))
        self._dual_scale_deep = bool(getattr(config.model, "dual_scale_deep", False))
        self._dual_scale_local_only = bool(
            getattr(config.model, "dual_scale_local_only", False)
        )
        self._dual_scale_outer_size = int(
            getattr(config.model, "dual_scale_outer_size", 0)
        )
        self._dual_scale_outer_mix = float(
            getattr(config.model, "dual_scale_outer_mix", 0.0)
        )
        self._dual_scale_adaptive_gate = bool(
            getattr(config.model, "dual_scale_adaptive_gate", False)
        )
        self._dual_scale_gate_max = float(
            getattr(config.model, "dual_scale_gate_max", 1.0)
        )
        if self._dual_scale and self._dual_scale_local_size <= 0:
            raise ValueError("dual_scale_local_size must be positive")
        if self._dual_scale_outer_size < 0:
            raise ValueError("dual_scale_outer_size must be non-negative")
        if self._dual_scale_outer_size and not self._dual_scale:
            raise ValueError("dual_scale_outer_size requires dual_scale=True")
        if self._dual_scale_adaptive_gate and not self._dual_scale:
            raise ValueError("dual_scale_adaptive_gate requires dual_scale=True")
        if self._dual_scale_gate_max <= 0:
            raise ValueError("dual_scale_gate_max must be positive")
        self.dual_scale_deep_local = (
            LocalUNet3d(3, (c1, c2, c3, c4), norm_mode)
            if self._dual_scale and self._dual_scale_deep else None
        )
        self.dual_scale_local = (
            ConvBlock3d(3, c1)
            if self._dual_scale and not self._dual_scale_deep else None
        )
        self.dual_scale_head = (
            nn.Conv3d(c1, 1, kernel_size=1, bias=True)
            if self._dual_scale and not self._dual_scale_deep else None
        )
        self.dual_scale_outer = (
            ConvBlock3d(3, c1)
            if self._dual_scale_outer_size > 0 else None
        )
        self.dual_scale_outer_head = (
            nn.Conv3d(c1, 1, kernel_size=1, bias=True)
            if self._dual_scale_outer_size > 0 else None
        )
        self.dual_scale_gate = (
            nn.Sequential(
                nn.Linear(2 * input_depth, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            if self._dual_scale_adaptive_gate else None
        )
        self.last_dual_scale_gate: torch.Tensor | None = None

        self._early_2d_unet = bool(getattr(config.model, "early_2d_unet", False))
        self._mid_2d_unet = bool(getattr(config.model, "mid_2d_unet", False))
        self._mid_depth_entropy_floor = float(
            getattr(config.model, "mid_depth_entropy_floor", 0.0)
        )
        self._mid_depth_max_mode = str(getattr(config.model, "mid_depth_max_mode", "amax"))
        self._mid_depth_topk_frac = float(getattr(config.model, "mid_depth_topk_frac", 0.5))
        self._mt_lse_r_max = float(getattr(config.model, "mt_lse_r_max", 10.0))
        if not 0.0 <= self._mid_depth_entropy_floor <= 1.0:
            raise ValueError("mid_depth_entropy_floor must be a fraction of log(depth) in [0, 1]")
        if self._mid_depth_max_mode not in ("amax", "topk"):
            raise ValueError("mid_depth_max_mode must be 'amax' or 'topk'")
        if not 0.0 < self._mid_depth_topk_frac <= 1.0:
            raise ValueError("mid_depth_topk_frac must be in (0, 1]")
        if self._mt_lse_r_max < 0.5:
            raise ValueError("mt_lse_r_max must be at least the 0.5 lse floor")
        self.last_mid_depth_entropy_penalty: torch.Tensor | None = None
        self.last_mid_depth_entropy: torch.Tensor | None = None
        self._residual_2d_unet = bool(
            getattr(config.model, "residual_2d_unet", False)
        )
        self._two_d_block_depth = int(getattr(config.model, "two_d_block_depth", 2))
        self._two_d_bottleneck_channels = int(
            getattr(config.model, "two_d_bottleneck_channels", 0)
        )
        self._two_d_extra_levels = int(getattr(config.model, "two_d_extra_levels", 0))
        self._two_d_extra_channels = tuple(
            int(value) for value in getattr(config.model, "two_d_extra_channels", ())
        )
        if self._two_d_block_depth < 1:
            raise ValueError("two_d_block_depth must be positive")
        if self._two_d_bottleneck_channels < 0:
            raise ValueError("two_d_bottleneck_channels must be non-negative")
        if self._two_d_extra_levels < 0:
            raise ValueError("two_d_extra_levels must be non-negative")
        if any(value <= 0 for value in self._two_d_extra_channels):
            raise ValueError("two_d_extra_channels must contain positive values")
        if self._two_d_extra_channels:
            if self._two_d_extra_levels not in (0, len(self._two_d_extra_channels)):
                raise ValueError("two_d_extra_levels must match two_d_extra_channels")
            self._two_d_extra_levels = len(self._two_d_extra_channels)

        def block2d(in_channels: int, out_channels: int):
            block_type = ResidualConvBlock2d if self._residual_2d_unet else ConvBlock2d
            return block_type(in_channels, out_channels, self._two_d_block_depth)

        if self._early_2d_unet and self._mid_2d_unet:
            raise ValueError("early_2d_unet and mid_2d_unet are mutually exclusive")
        text_region_gate = bool(getattr(config.model, "text_region_gate", False))
        if text_region_gate and not self._early_2d_unet:
            raise ValueError("text_region_gate requires early_2d_unet")
        self.text_region_head = None
        self.last_text_region_logits: torch.Tensor | None = None
        if self._early_2d_unet:
            if not bool(getattr(config.model, "multitile", False)):
                raise ValueError("early_2d_unet requires multitile=True")
            early_mult = float(getattr(config.model, "early_2d_channels_mult", 1.0))
            e1, e2, e3, e4 = (
                max(1, int(round(ch * early_mult)))
                for ch in (c1, c2, c3, c4)
            )
            if self._two_d_bottleneck_channels:
                e4 = self._two_d_bottleneck_channels
            self.early_depth_attn = nn.Conv3d(c1, 1, kernel_size=1)
            self.early_depth_fuse = nn.Conv2d(c1 * 2, e1, kernel_size=1, bias=False)
            self.early2d_enc2 = block2d(e1, e2)
            self.early2d_enc3 = block2d(e2, e3)
            self.early2d_bottleneck = block2d(e3, e4)
            self.early2d_up3 = nn.ConvTranspose2d(e4, e3, kernel_size=2, stride=2)
            self.early2d_dec3 = block2d(e3 * 2, e3)
            self.early2d_up2 = nn.ConvTranspose2d(e3, e2, kernel_size=2, stride=2)
            self.early2d_dec2 = block2d(e2 * 2, e2)
            self.early2d_up1 = nn.ConvTranspose2d(e2, e1, kernel_size=2, stride=2)
            self.early2d_dec1 = block2d(e1 * 2, e1)
            self.early2d_head = nn.Conv2d(e1, 1, kernel_size=1)
            extra_channels = self._two_d_extra_channels or (e4,) * self._two_d_extra_levels
            encoder_channels = (e4,) + extra_channels
            self.early2d_extra_encoders = nn.ModuleList([
                block2d(in_channels, out_channels)
                for in_channels, out_channels in zip(encoder_channels, extra_channels)
            ])
            self.early2d_extra_ups = nn.ModuleList([
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
                for in_channels, out_channels in zip(encoder_channels, extra_channels)
            ])
            self.early2d_extra_decoders = nn.ModuleList([
                block2d(in_channels * 2, in_channels)
                for in_channels in encoder_channels[:-1]
            ])
            overlap_channels = (encoder_channels[-1], e1)
            if text_region_gate:
                self.text_region_head = nn.Conv2d(encoder_channels[-1], 1, kernel_size=1)
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
            self.early2d_extra_encoders = nn.ModuleList()
            self.early2d_extra_ups = nn.ModuleList()
            self.early2d_extra_decoders = nn.ModuleList()

        if self._mid_2d_unet:
            if not bool(getattr(config.model, "multitile", False)):
                raise ValueError("mid_2d_unet requires multitile=True")
            mid_mult = float(getattr(config.model, "mid_2d_channels_mult", 1.0))
            if mid_mult <= 0:
                raise ValueError("mid_2d_channels_mult must be positive")
            m1, m2, m3, m4 = (
                max(1, int(round(ch * mid_mult)))
                for ch in (c1, c2, c3, c4)
            )
            if self._two_d_bottleneck_channels:
                m4 = self._two_d_bottleneck_channels
            self.mid_depth_attn = nn.Conv3d(c2, 1, kernel_size=1)
            self.mid_depth_fuse = nn.Conv2d(c2 * 2, m2, kernel_size=1, bias=False)
            self.mid_skip1_fuse = nn.Conv2d(c1 * 2, m1, kernel_size=1, bias=False)
            self.mid2d_enc3 = block2d(m2, m3)
            self.mid2d_bottleneck = block2d(m3, m4)
            self.mid2d_up3 = nn.ConvTranspose2d(m4, m3, kernel_size=2, stride=2)
            self.mid2d_dec3 = block2d(m3 * 2, m3)
            self.mid2d_up2 = nn.ConvTranspose2d(m3, m2, kernel_size=2, stride=2)
            self.mid2d_dec2 = block2d(m2 * 2, m2)
            self.mid2d_up1 = nn.ConvTranspose2d(m2, m1, kernel_size=2, stride=2)
            self.mid2d_dec1 = block2d(m1 * 2, m1)
            self.mid2d_head = nn.Conv2d(m1, 1, kernel_size=1)
            extra_channels = self._two_d_extra_channels or (m4,) * self._two_d_extra_levels
            encoder_channels = (m4,) + extra_channels
            self.mid2d_extra_encoders = nn.ModuleList([
                block2d(in_channels, out_channels)
                for in_channels, out_channels in zip(encoder_channels, extra_channels)
            ])
            self.mid2d_extra_ups = nn.ModuleList([
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
                for in_channels, out_channels in zip(encoder_channels, extra_channels)
            ])
            self.mid2d_extra_decoders = nn.ModuleList([
                block2d(in_channels * 2, in_channels)
                for in_channels in encoder_channels[:-1]
            ])
            overlap_channels = (c4, c1)
        else:
            self.mid_depth_attn = None
            self.mid_depth_fuse = None
            self.mid_skip1_fuse = None
            self.mid2d_enc3 = None
            self.mid2d_bottleneck = None
            self.mid2d_up3 = None
            self.mid2d_dec3 = None
            self.mid2d_up2 = None
            self.mid2d_dec2 = None
            self.mid2d_up1 = None
            self.mid2d_dec1 = None
            self.mid2d_head = None
            self.mid2d_extra_encoders = nn.ModuleList()
            self.mid2d_extra_ups = nn.ModuleList()
            self.mid2d_extra_decoders = nn.ModuleList()

        if self._overlapping_depth_windows and (self._early_2d_unet or self._mid_2d_unet):
            if self._overlap_window_size < 2 or self._overlap_window_stride < 1:
                raise ValueError("overlapping depth window size/stride must be positive")
            if input_depth < self._overlap_window_size:
                raise ValueError("input depth is smaller than the overlapping window")
            self._overlap_window_count = (
                (input_depth - self._overlap_window_size)
                // self._overlap_window_stride
                + 1
            )
            bottleneck_channels, decoded_channels = overlap_channels
            self.overlap_bottleneck_fuse = nn.Conv2d(
                self._overlap_window_count * bottleneck_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            )
            self.overlap_decoded_fuse = nn.Conv2d(
                self._overlap_window_count * decoded_channels,
                decoded_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self._overlap_window_count = 0
            self.overlap_bottleneck_fuse = None
            self.overlap_decoded_fuse = None

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
            nn.Conv3d(4, c1, kernel_size=1, bias=False)
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
        legacy_weldon_k = int(getattr(config.model, "weldon_k", 0))
        self._weldon_top_k = int(
            getattr(config.model, "weldon_top_k", 0) or legacy_weldon_k
        )
        self._weldon_bottom_k = int(
            getattr(config.model, "weldon_bottom_k", 0) or legacy_weldon_k
        )
        self._weldon_top_weight = float(
            getattr(config.model, "weldon_top_weight", 0.5)
        )
        self._weldon_multi_k = bool(getattr(config.model, "weldon_multi_k", False))
        self._weldon_top_k2 = int(getattr(config.model, "weldon_top_k2", 0))
        self._weldon_bottom_k2 = int(getattr(config.model, "weldon_bottom_k2", 0))
        self._weldon_multi_mix = float(getattr(config.model, "weldon_multi_mix", 0.5))
        self._weldon_depth_support_k = int(
            getattr(config.model, "weldon_depth_support_k", 0)
        )
        self._clam_instance = bool(getattr(config.tra, "clam_instance", False))
        if self._minimum_support_k < 0 or min(self._weldon_top_k, self._weldon_bottom_k) < 0:
            raise ValueError("minimum-support and WELDON k must be non-negative")
        if self._minimum_support_k and self._weldon_top_k:
            raise ValueError("minimum-support and WELDON aggregators are mutually exclusive")
        if self._weldon_top_k and not (0.0 <= self._weldon_top_weight <= 1.0):
            raise ValueError("weldon_top_weight must be in [0, 1]")
        if self._weldon_multi_k and min(self._weldon_top_k2, self._weldon_bottom_k2) <= 0:
            raise ValueError("multi-k WELDON requires positive secondary top and bottom k")
        if not 0.0 <= self._weldon_multi_mix <= 1.0:
            raise ValueError("weldon_multi_mix must be in [0, 1]")
        if self._minimum_support_kernel < 1 or self._minimum_support_kernel % 2 == 0:
            raise ValueError("minimum_support_kernel must be a positive odd integer")
        if self._early_2d_unet or self._mid_2d_unet:
            bypassed = []
            if self._dual_scale and self._early_2d_unet:
                bypassed.append("dual_scale")
            if self.style_film is not None:
                bypassed.append("style_film")
            if self._mixstyle:
                bypassed.append("mixstyle")
            if self._sagnet:
                bypassed.append("sagnet")
            if self._weldon_top_k > 0:
                bypassed.append("weldon")
            if self._sparse_deep_supervision:
                bypassed.append("sparse_deep_supervision")
            if bypassed:
                raise ValueError(
                    "early/mid 2D U-Net path bypasses incompatible features: "
                    + ", ".join(bypassed)
                )

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

        use_dg_embeddings = any([
            bool(getattr(config.tra, "prototype_align", False)),
            bool(getattr(config.tra, "coral_align", False)),
            bool(getattr(config.tra, "cdan", False)),
        ])
        self.dg_head: SupConHead | None = None
        if use_dg_embeddings and self.supcon_head is None:
            self.dg_head = SupConHead(
                in_features=c1,
                proj_dim=int(getattr(config.tra, "supcon_proj_dim", 128)),
                hidden=int(getattr(config.tra, "supcon_hidden_dim", 256)),
            )
        self.cdan_head: DomainClassifier | None = None
        if bool(getattr(config.tra, "cdan", False)):
            n_domains = int(getattr(config.tra, "dann_n_domains", 0))
            if n_domains <= 1:
                raise ValueError("CDAN requires at least two physical domains")
            self.cdan_head = DomainClassifier(
                in_features=2 * int(getattr(config.tra, "supcon_proj_dim", 128)),
                n_domains=n_domains,
            )

        self.mae_reconstruction_head: nn.Module | None = None
        if bool(getattr(config.model, "mae_reconstruction_head", False)):
            self.mae_reconstruction_head = nn.Conv3d(c1, 1, kernel_size=1, bias=True)

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

        self._depth_attention_2d_head = bool(
            getattr(config.model, "depth_attention_2d_head", False)
        )
        self.depth_2d_attention = (
            nn.Conv3d(c1, 1, kernel_size=1, bias=True)
            if self._depth_attention_2d_head else None
        )
        self.depth_2d_bias = (
            nn.Parameter(torch.zeros(1, 1, input_depth, 1, 1))
            if self._depth_attention_2d_head else None
        )
        self.depth_shift_head = (
            nn.Linear(
                c4,
                int(getattr(config.tra, "depth_shift_aux_classes", 3)),
            )
            if bool(getattr(config.tra, "depth_shift_aux", False)) else None
        )

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

    def _mix_feature_style(
        self,
        features: torch.Tensor,
        domain_ids: torch.Tensor | None,
        probability: float,
    ) -> torch.Tensor:
        """mix channel statistics with a preferably different-domain peer."""
        if not self.training or features.shape[0] < 2 or probability <= 0:
            return features
        batch = features.shape[0]
        permutation = torch.randperm(batch, device=features.device)
        if domain_ids is not None:
            domains = domain_ids.view(-1)
            allowed = domains.unsqueeze(0) != domains.unsqueeze(1)
            random_scores = torch.rand(batch, batch, device=features.device)
            choices = random_scores.masked_fill(~allowed, -1.0).argmax(dim=1)
            permutation = torch.where(allowed.any(dim=1), choices, permutation)
        mean = features.mean(dim=(2, 3, 4), keepdim=True)
        std = features.var(dim=(2, 3, 4), keepdim=True, unbiased=False).add(1e-6).sqrt()
        normalized = (features - mean) / std
        concentration = torch.full(
            (batch,),
            max(self._mixstyle_alpha, 1e-3),
            device=features.device,
            dtype=torch.float32,
        )
        lam = torch.distributions.Beta(concentration, concentration).sample()
        lam = lam.to(dtype=features.dtype).view(batch, 1, 1, 1, 1)
        mixed_mean = lam * mean + (1.0 - lam) * mean[permutation]
        mixed_std = lam * std + (1.0 - lam) * std[permutation]
        mixed = normalized * mixed_std.detach() + mixed_mean.detach()
        apply = (torch.rand(batch, device=features.device) < probability).view(
            batch, 1, 1, 1, 1
        )
        return torch.where(apply, mixed, features)

    def _stem_in(
        self,
        x: torch.Tensor,
        surface_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = x
        if self._depth_antialias and x.shape[2] > 1:
            padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
            raw = (
                padded[:, :, :-2]
                + 2.0 * padded[:, :, 1:-1]
                + padded[:, :, 2:]
            ) * 0.25
        if self._raw_only_stem:
            return raw
        dz = torch.zeros_like(raw)
        if not self._no_dz:
            dz[:, :, 1:] = raw[:, :, 1:] - raw[:, :, :-1]
        cues = torch.cat([raw, _lcn2d(raw, 5), dz], dim=1)
        if self.training and self._cue_dropout > 0:
            apply = torch.rand(cues.shape[0], device=cues.device) < self._cue_dropout
            cue_index = torch.randint(0, 3, (cues.shape[0],), device=cues.device)
            keep = torch.ones(
                cues.shape[0], 3, 1, 1, 1,
                device=cues.device,
                dtype=cues.dtype,
            )
            keep[torch.arange(cues.shape[0], device=cues.device), cue_index] = (~apply).to(
                cues.dtype
            )
            cues = cues * keep
        if not self._explicit_depth_channels:
            return cues
        relative = torch.linspace(
            -1.0,
            1.0,
            raw.shape[2],
            device=raw.device,
            dtype=raw.dtype,
        ).view(1, 1, -1, 1, 1).expand(raw.shape[0], 1, -1, raw.shape[3], raw.shape[4])
        signed_surface = (
            surface_features[:, 2:3]
            if surface_features is not None else torch.zeros_like(relative)
        )
        return torch.cat((cues, relative, signed_surface), dim=1)

    def _apply_gated_cues(self, stem: torch.Tensor) -> torch.Tensor:
        if self.gated_cue_stem is None:
            return stem
        gated = self.gated_cue_stem(stem[:, :3])
        return torch.cat((gated, stem[:, 3:]), dim=1)

    @staticmethod
    def _fiber_coordinates(x: torch.Tensor) -> torch.Tensor:
        """local tangent orientation and anisotropy from a per-slice structure tensor, plus how
        far the local orientation departs from the surrounding fibre direction (strokes cross
        fibres; crackle and fibre texture follow them)."""
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
        wide_xx = F.avg_pool2d(jxx, 31, stride=1, padding=15, count_include_pad=False)
        wide_yy = F.avg_pool2d(jyy, 31, stride=1, padding=15, count_include_pad=False)
        wide_xy = F.avg_pool2d(jxy, 31, stride=1, padding=15, count_include_pad=False)
        wide_delta = torch.sqrt((wide_xx - wide_yy).square() + 4.0 * wide_xy.square() + 1e-6)
        alignment = (cos2theta * (wide_xx - wide_yy) + sin2theta * 2.0 * wide_xy) / wide_delta
        deviation = coherence * (1.0 - alignment)
        features = torch.cat((coherence, cos2theta, sin2theta, deviation), dim=1)
        return features.reshape(batch, 4, depth, height, width)

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
        domain_ids: torch.Tensor | None = None,
        sagnet_grl_scale: float = 0.0,
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

        stem_x = self._apply_gated_cues(
            self._stem_in(raw_for_backbone, surface_for_backbone)
        )
        enc1 = self.enc1(stem_x)
        if self.style_film is not None:
            enc1 = self.style_film(enc1, raw_for_backbone)
        if self.sagnet_style_head is not None:
            style_mean = enc1.mean(dim=(2, 3, 4))
            style_std = enc1.std(dim=(2, 3, 4), unbiased=False)
            self.last_sagnet_logits = self.sagnet_style_head(
                torch.cat((style_mean, style_std), dim=1),
                sagnet_grl_scale,
            )
            enc1 = self._mix_feature_style(enc1, domain_ids, 1.0)
        elif self._mixstyle:
            self.last_sagnet_logits = None
            enc1 = self._mix_feature_style(enc1, domain_ids, self._mixstyle_prob)
        else:
            self.last_sagnet_logits = None
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
        self._last_decoder_scales = (dec2, dec3)
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
        features3d = self.enc1(self._apply_gated_cues(self._stem_in(raw, surface)))
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
        deepest = bottleneck
        extra_skips = []
        for encoder in self.early2d_extra_encoders:
            extra_skips.append(deepest)
            deepest = encoder(F.max_pool2d(deepest, 2))
        decoded = deepest
        for upsample, decoder, skip in zip(
            reversed(self.early2d_extra_ups),
            reversed(self.early2d_extra_decoders),
            reversed(extra_skips),
        ):
            decoded = decoder(self._merge_skip_2d(upsample(decoded), skip))
        dec3 = self.early2d_dec3(self._merge_skip_2d(self.early2d_up3(decoded), enc3))
        dec2 = self.early2d_dec2(self._merge_skip_2d(self.early2d_up2(dec3), enc2))
        dec1 = self.early2d_dec1(self._merge_skip_2d(self.early2d_up1(dec2), enc1))
        if self._head_drop is not None:
            dec1 = F.dropout2d(dec1, p=self._head_drop.p, training=self.training)
        return deepest, dec1

    def _encode_decode_mid_2d(
        self,
        x: torch.Tensor,
        teacher_surface_depth: torch.Tensor | None,
        teacher_surface_confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """retain two volumetric encoder stages, then spend deeper capacity in XY."""
        if self.mid_depth_attn is None or self.mid_depth_fuse is None:
            raise RuntimeError("mid 2D U-Net modules are not initialized")
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
            raise ValueError("mid_2d_unet currently requires literal or disabled surface input")
        stem = self._apply_gated_cues(self._stem_in(raw, surface))
        enc1 = self.enc1(stem)
        if surface is not None and self.new_surface_input is not None:
            enc1 = enc1 + self.new_surface_input(surface)
        if self.fiber_coordinate_input is not None:
            enc1 = enc1 + self.fiber_coordinate_input(self._fiber_coordinates(raw))
        if self._enc1_drop is not None:
            enc1 = self._enc1_drop(enc1)
        enc2 = self.enc2(self.pool(enc1))
        if self._enc2_drop is not None:
            enc2 = self._enc2_drop(enc2)

        depth_weights = torch.softmax(self.mid_depth_attn(enc2), dim=2)
        self._record_mid_depth_entropy(depth_weights)
        enc2_2d = self.mid_depth_fuse(torch.cat(
            ((enc2 * depth_weights).sum(dim=2), self._mid_depth_peak(enc2)),
            dim=1,
        ))
        enc1_weights = F.interpolate(
            depth_weights,
            size=enc1.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        enc1_weights = enc1_weights / enc1_weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        enc1_2d = self.mid_skip1_fuse(torch.cat(
            ((enc1 * enc1_weights).sum(dim=2), self._mid_depth_peak(enc1)),
            dim=1,
        ))

        enc3 = self.mid2d_enc3(F.max_pool2d(enc2_2d, 2))
        bottleneck = self.mid2d_bottleneck(F.max_pool2d(enc3, 2))
        deepest = bottleneck
        extra_skips = []
        for encoder in self.mid2d_extra_encoders:
            extra_skips.append(deepest)
            deepest = encoder(F.max_pool2d(deepest, 2))
        decoded = deepest
        for upsample, decoder, skip in zip(
            reversed(self.mid2d_extra_ups),
            reversed(self.mid2d_extra_decoders),
            reversed(extra_skips),
        ):
            decoded = decoder(self._merge_skip_2d(upsample(decoded), skip))
        dec3 = self.mid2d_dec3(self._merge_skip_2d(self.mid2d_up3(decoded), enc3))
        dec2 = self.mid2d_dec2(self._merge_skip_2d(self.mid2d_up2(dec3), enc2_2d))
        dec1 = self.mid2d_dec1(self._merge_skip_2d(self.mid2d_up1(dec2), enc1_2d))
        if self._head_drop is not None:
            dec1 = F.dropout2d(dec1, p=self._head_drop.p, training=self.training)
        return deepest, dec1

    def _mid_depth_peak(self, features: torch.Tensor) -> torch.Tensor:
        if self._mid_depth_max_mode == "amax":
            return features.amax(dim=2)
        k = max(1, int(round(features.shape[2] * self._mid_depth_topk_frac)))
        return features.topk(k, dim=2).values.mean(dim=2)

    def _record_mid_depth_entropy(self, depth_weights: torch.Tensor) -> None:
        weights = depth_weights.float().clamp(min=1e-8)
        entropy = -(weights * weights.log()).sum(dim=2)
        self.last_mid_depth_entropy = entropy.mean()
        if self._mid_depth_entropy_floor > 0:
            target = self._mid_depth_entropy_floor * math.log(max(depth_weights.shape[2], 2))
            self.last_mid_depth_entropy_penalty = F.relu(target - entropy).mean()
        else:
            self.last_mid_depth_entropy_penalty = None

    def _encode_decode_overlapping_depth(
        self,
        x: torch.Tensor,
        teacher_surface_depth: torch.Tensor | None,
        teacher_surface_confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.overlap_bottleneck_fuse is None or self.overlap_decoded_fuse is None:
            raise RuntimeError("overlapping depth fusion is not initialized")
        encode = (
            self._encode_decode_early_2d if self._early_2d_unet else self._encode_decode_mid_2d
        )
        bottlenecks = []
        decoded = []
        penalties = []
        for index in range(self._overlap_window_count):
            start = index * self._overlap_window_stride
            end = start + self._overlap_window_size
            window_depth = None
            window_confidence = None
            if teacher_surface_depth is not None and teacher_surface_confidence is not None:
                window_depth = teacher_surface_depth - float(start)
                valid = (window_depth >= 0) & (window_depth <= self._overlap_window_size - 1)
                window_depth = torch.where(valid, window_depth, torch.full_like(window_depth, -1.0))
                window_confidence = torch.where(
                    valid,
                    teacher_surface_confidence,
                    torch.zeros_like(teacher_surface_confidence),
                )
            bottleneck, features = encode(
                x[:, :, start:end],
                window_depth,
                window_confidence,
            )
            bottlenecks.append(bottleneck)
            decoded.append(features)
            if self.last_mid_depth_entropy_penalty is not None:
                penalties.append(self.last_mid_depth_entropy_penalty)
        if penalties:
            self.last_mid_depth_entropy_penalty = torch.stack(penalties).mean()
        return (
            self.overlap_bottleneck_fuse(torch.cat(bottlenecks, dim=1)),
            self.overlap_decoded_fuse(torch.cat(decoded, dim=1)),
        )

    @staticmethod
    def _embedding(bottleneck: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(bottleneck, output_size=1).flatten(1)

    def conditional_domain_logits(
        self,
        embeddings: torch.Tensor,
        probabilities: torch.Tensor,
        grl_scale: float,
    ) -> torch.Tensor:
        if self.cdan_head is None:
            raise RuntimeError("conditional domain head is not initialized")
        probabilities = probabilities.unsqueeze(-1)
        conditioned = torch.cat(
            (embeddings * probabilities, embeddings * (1.0 - probabilities)),
            dim=-1,
        )
        return self.cdan_head(conditioned, grl_scale=grl_scale)

    def reconstruct(
        self,
        x: torch.Tensor,
        teacher_surface_depth: torch.Tensor | None = None,
        teacher_surface_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mae_reconstruction_head is None:
            raise RuntimeError("MAE reconstruction head is not initialized")
        if self._early_2d_unet:
            raise RuntimeError("supervised MAE continuation currently requires the 3D decoder")
        _, decoded = self._encode_decode(
            x,
            teacher_surface_depth,
            teacher_surface_confidence,
        )
        return self.mae_reconstruction_head(decoded)

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

    def _multitile_spatial_instances(
        self,
        center: torch.Tensor,
        depth_support_k: int = 0,
    ) -> torch.Tensor:
        """depth-collapse voxel logits and return locally supported XY instances per cell."""
        n, sub = self._mt_grid, self._mt_sub_feat
        values = center.squeeze(1)
        if depth_support_k > 0:
            support = min(max(1, int(depth_support_k)), values.shape[1])
            spatial = torch.topk(values, support, dim=1).values.mean(dim=1)
        else:
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

    def _multitile_weldon_once(
        self,
        center: torch.Tensor,
        top_k: int,
        bottom_k: int,
    ) -> torch.Tensor:
        instances = self._multitile_spatial_instances(
            center,
            self._weldon_depth_support_k,
        )
        top_k = min(max(1, int(top_k)), instances.shape[-1])
        bottom_k = min(max(1, int(bottom_k)), instances.shape[-1])
        positive = torch.topk(instances, top_k, dim=-1).values.mean(dim=-1)
        negative = torch.topk(
            instances,
            bottom_k,
            dim=-1,
            largest=False,
        ).values.mean(dim=-1)
        return self._weldon_top_weight * positive + (
            1.0 - self._weldon_top_weight
        ) * negative

    def _multitile_weldon(self, center: torch.Tensor) -> torch.Tensor:
        primary = self._multitile_weldon_once(
            center,
            self._weldon_top_k,
            self._weldon_bottom_k,
        )
        if not self._weldon_multi_k:
            return primary
        secondary = self._multitile_weldon_once(
            center,
            self._weldon_top_k2,
            self._weldon_bottom_k2,
        )
        return (
            self._weldon_multi_mix * primary
            + (1.0 - self._weldon_multi_mix) * secondary
        )

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
        r = self.lse_r.clamp(min=0.5, max=self._mt_lse_r_max)
        count = cells.new_tensor(float(cells.shape[-1]))
        return (torch.logsumexp(r * cells, dim=-1) - torch.log(count)) / r

    def _multitile_mean_2d(
        self,
        feature_map: torch.Tensor,
        target_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        """mean of a (B,1,H,W) map over each multitile cell, in iy*grid+ix order."""
        center = self._crop_center_feat(
            feature_map.unsqueeze(2),
            self._mt_center_feat,
            target_offsets,
        ).squeeze(1).squeeze(1)
        n, sub = self._mt_grid, self._mt_sub_feat
        return center.reshape(center.shape[0], n, sub, n, sub).mean(dim=(2, 4)).flatten(1)

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

    def _apply_dual_scale_score(
        self,
        score: torch.Tensor,
        x: torch.Tensor,
        target_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._dual_scale:
            return score
        prepared = self._prepare_input(x)
        local_size = max(1, self._dual_scale_local_size // self._downsample)
        local_input = self._crop_center_feat(prepared, local_size, target_offsets)
        local_stem = self._stem_in(local_input)
        if self.dual_scale_deep_local is not None:
            local_voxels = self.dual_scale_deep_local(local_stem)
        elif self.dual_scale_local is not None and self.dual_scale_head is not None:
            local_voxels = self.dual_scale_head(self.dual_scale_local(local_stem))
        else:
            raise RuntimeError("dual-scale local expert is not initialized")
        local_center = self._crop_center_feat(local_voxels, self._mt_center_feat)
        local_score = self._multitile_aggregate(local_center)
        if self._dual_scale_local_only:
            score = local_score
            self.last_dual_scale_gate = None
        elif self.dual_scale_gate is not None:
            statistics = torch.cat(
                (
                    prepared.mean(dim=(1, 3, 4)),
                    prepared.std(dim=(1, 3, 4), unbiased=False),
                ),
                dim=1,
            )
            gate = self._dual_scale_gate_max * torch.sigmoid(
                self.dual_scale_gate(statistics)
            )
            score = score + gate * local_score
            self.last_dual_scale_gate = gate.detach()
        else:
            score = score + self._dual_scale_mix * local_score
            self.last_dual_scale_gate = None
        if self.dual_scale_outer is not None and self.dual_scale_outer_head is not None:
            outer_size = max(1, self._dual_scale_outer_size // self._downsample)
            outer_input = self._crop_center_feat(prepared, outer_size, target_offsets)
            outer_voxels = self.dual_scale_outer_head(
                self.dual_scale_outer(self._stem_in(outer_input))
            )
            outer_center = self._crop_center_feat(outer_voxels, self._mt_center_feat)
            outer_score = self._multitile_aggregate(outer_center)
            score = score + self._dual_scale_outer_mix * outer_score
        return score

    def forward_with_extras(
        self,
        x: torch.Tensor,
        grl_scale: float = 1.0,
        target_offsets: torch.Tensor | None = None,
        teacher_surface_depth: torch.Tensor | None = None,
        teacher_surface_confidence: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        sagnet_grl_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._early_2d_unet:
            encode = (
                self._encode_decode_overlapping_depth
                if self._overlapping_depth_windows else self._encode_decode_early_2d
            )
            bottleneck2d, decoded2d = encode(
                x,
                teacher_surface_depth,
                teacher_surface_confidence,
            )
            voxel2d = self.early2d_head(decoded2d)
            if self.text_region_head is not None:
                region = F.interpolate(
                    self.text_region_head(bottleneck2d),
                    size=voxel2d.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                self.last_text_region_logits = self._multitile_mean_2d(region, target_offsets)
                voxel2d = voxel2d + F.logsigmoid(region)
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

        if self._mid_2d_unet:
            if self._overlapping_depth_windows:
                bottleneck2d, decoded2d = self._encode_decode_overlapping_depth(
                    x,
                    teacher_surface_depth,
                    teacher_surface_confidence,
                )
            else:
                bottleneck2d, decoded2d = self._encode_decode_mid_2d(
                    x,
                    teacher_surface_depth,
                    teacher_surface_confidence,
                )
            voxel2d = self.mid2d_head(decoded2d)
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
            self.last_depth_shift_logits = (
                self.depth_shift_head(embedding)
                if self.depth_shift_head is not None else None
            )
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
            score = self._apply_dual_scale_score(score, x, target_offsets)
            self.last_attn_entropy_loss = voxel2d.new_zeros(())
            self.last_attn_entropy_per_target = None
            self.last_surface_guided_alpha = None
            return score, embedding, domain_logits, supcon_z

        bottleneck, decoded = self._encode_decode(
            x,
            teacher_surface_depth=teacher_surface_depth,
            teacher_surface_confidence=teacher_surface_confidence,
            domain_ids=domain_ids,
            sagnet_grl_scale=sagnet_grl_scale,
        )
        voxel_map = self.out_head(decoded)
        if self._sparse_deep_supervision:
            dec2, dec3 = self._last_decoder_scales
            aux2 = F.interpolate(
                self.deep_supervision_dec2(dec2),
                size=voxel_map.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            aux3 = F.interpolate(
                self.deep_supervision_dec3(dec3),
                size=voxel_map.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            self.last_deep_supervision_scores = (
                self._multitile_aggregate(
                    self._crop_center_feat(aux2, self._mt_center_feat, target_offsets)
                ),
                self._multitile_aggregate(
                    self._crop_center_feat(aux3, self._mt_center_feat, target_offsets)
                ),
            )
        else:
            self.last_deep_supervision_scores = None
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
        projection_head = self.supcon_head or self.dg_head
        if projection_head is not None:
            supcon_input = (
                self._multitile_embeddings(decoded, target_offsets)
                if self._multitile else embedding
            )
            supcon_z = projection_head(supcon_input)
        else:
            supcon_z = None
        if self._multitile:
            if self._depth_attention_2d_head:
                attention_logits = self.depth_2d_attention(decoded)
                depth_bias = self.depth_2d_bias[:, :, :attention_logits.shape[2]]
                depth_weights = torch.softmax(attention_logits + depth_bias, dim=2)
                logits2d = (voxel_map * depth_weights).sum(dim=2)
                score = self._multitile_aggregate_2d(logits2d, target_offsets)
                self.last_voxel_map = (
                    None if self.training else logits2d.unsqueeze(2).detach().clone()
                )
                self.last_voxel_map_full = logits2d.unsqueeze(2)
            elif self._feature_depth_fusion:
                score = self._multitile_feature_depth_fusion(decoded, target_offsets)
            elif self._minimum_support_k > 0:
                score = self._multitile_minimum_support(
                    center_voxels,
                    self._minimum_support_k,
                )
            elif self._weldon_top_k > 0:
                score = self._multitile_weldon(center_voxels)
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
        score = self._apply_dual_scale_score(score, x, target_offsets)
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
    if model.mid_depth_attn is not None:
        nn.init.zeros_(model.mid_depth_attn.weight)
        nn.init.zeros_(model.mid_depth_attn.bias)
    if model.depth_2d_attention is not None:
        nn.init.zeros_(model.depth_2d_attention.weight)
        nn.init.zeros_(model.depth_2d_attention.bias)
    if model.style_film is not None:
        final = model.style_film.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    if model.dual_scale_head is not None:
        nn.init.zeros_(model.dual_scale_head.weight)
        nn.init.zeros_(model.dual_scale_head.bias)
    if model.dual_scale_outer_head is not None:
        nn.init.zeros_(model.dual_scale_outer_head.weight)
        nn.init.zeros_(model.dual_scale_outer_head.bias)
    if model.dual_scale_gate is not None:
        final = model.dual_scale_gate[-1]
        nn.init.zeros_(final.weight)
        initial_fraction = min(
            max(model._dual_scale_mix / model._dual_scale_gate_max, 1e-4),
            1.0 - 1e-4,
        )
        nn.init.constant_(
            final.bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )
    if model.dual_scale_deep_local is not None:
        nn.init.zeros_(model.dual_scale_deep_local.head.weight)
        nn.init.zeros_(model.dual_scale_deep_local.head.bias)
    if model.text_region_head is not None:
        # logsigmoid(4) ~ -0.02: the gate starts open so pretrained ink logits pass unchanged
        nn.init.zeros_(model.text_region_head.weight)
        nn.init.constant_(model.text_region_head.bias, 4.0)
    if model.gated_cue_stem is not None:
        model.gated_cue_stem.reset_identity()
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