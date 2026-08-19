"""model.py -- current nnU-Net-style ink detector.

The repo has converged on one backbone family:
  nnunet3d_lcndz

This file keeps only the integrations still exercised by the current sweep:
  - raw + lcn + dz stem
  - optional learned surface attention
  - optional attention-MIL with entropy regularization
  - optional spatial SupCon projection head
"""
from __future__ import annotations

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
    if not pos_mask.any():
        return z.new_zeros(())

    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    return (-(log_prob * pos_mask.float()).sum(dim=1) / pos_count).mean()


class GatedAttentionMIL(nn.Module):
    """gated attention-MIL aggregator over voxel logits."""

    def __init__(self, feat_dim: int = 1, att_dim: int = 32):
        super().__init__()
        self.v = nn.Linear(feat_dim, att_dim, bias=False)
        self.u = nn.Linear(feat_dim, att_dim, bias=False)
        self.w = nn.Linear(att_dim, 1, bias=False)
        self.out = nn.Linear(feat_dim, 1, bias=True)
        self.last_attn_weights: torch.Tensor | None = None

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

        entropy_loss = voxel_map.new_zeros(())
        if entropy_weight > 0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -entropy_weight * entropy
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
        self.last_surface_attn: torch.Tensor | None = None

        self.lse_r = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.pool = nn.MaxPool3d(2)

        use_ibn = bool(getattr(config.model, "use_ibn", False))
        self.enc1 = ConvBlock3d(3, 32, use_ibn=use_ibn)
        self.enc2 = ConvBlock3d(32, 64, use_ibn=use_ibn)
        self.enc3 = ConvBlock3d(64, 128)
        self.bottleneck = ConvBlock3d(128, 256)

        # spatial channel dropout after early encoder stages and before classification head
        _d1 = float(getattr(config.model, "conv1_drop", 0.0))
        _d2 = float(getattr(config.model, "conv2_drop", 0.0))
        _dh = float(getattr(config.model, "head_drop", 0.0))
        self._enc1_drop = nn.Dropout3d(p=_d1) if _d1 > 0 else None
        self._enc2_drop = nn.Dropout3d(p=_d2) if _d2 > 0 else None
        self._head_drop = nn.Dropout3d(p=_dh) if _dh > 0 else None

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(64, 32)
        self.out_head = nn.Conv3d(32, 1, kernel_size=1, bias=True)

        if bool(getattr(config.model, "learned_surface", False)):
            self.depth_surface_attn: DepthSurfaceAttn | None = DepthSurfaceAttn(hidden=8)
        else:
            self.depth_surface_attn = None

        if bool(getattr(config.model, "attn_mil", False)):
            self.attn_mil: GatedAttentionMIL | None = GatedAttentionMIL(feat_dim=1, att_dim=32)
        else:
            self.attn_mil = None

        self.supcon_head: SupConHead | None = None
        if bool(getattr(config.tra, "supcon", False)):
            self.supcon_head = SupConHead(
                in_features=256,
                proj_dim=int(getattr(config.tra, "supcon_proj_dim", 128)),
                hidden=int(getattr(config.tra, "supcon_hidden_dim", 256)),
            )

        self.domain_head: DomainClassifier | None = None
        if bool(getattr(config.tra, "dann", False)):
            n_domains = int(getattr(config.tra, "dann_n_domains", 0))
            if n_domains > 1:
                self.domain_head = DomainClassifier(in_features=256, n_domains=n_domains)

        self.prototype_head: PrototypeHead | None = None
        if bool(getattr(config.model, "use_prototype", False)):
            self.prototype_head = PrototypeHead(
                feat_dim=256,
                ema=float(getattr(config.model, "prototype_ema", 0.99)),
            )
        self._skip_drop = float(getattr(config.model, "skip_drop", 0.0))
        self._no_dz = bool(getattr(config.model, "no_dz", False))

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

    def _merge_skip(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="trilinear", align_corners=False)
        if self.training and self._skip_drop > 0:
            # zero whole skip connection with probability skip_drop; forces decoder bottleneck reliance
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

    def _encode_decode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_x = self._prepare_input(x)
        stem_x = self._stem_in(raw_x)

        enc1 = self._apply_learned_surface(raw_x, self.enc1(stem_x))
        if self._enc1_drop is not None:
            enc1 = self._enc1_drop(enc1)
        enc2 = self.enc2(self.pool(enc1))
        if self._enc2_drop is not None:
            enc2 = self._enc2_drop(enc2)
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.dec3(self._merge_skip(self.up3(bottleneck), enc3))
        dec2 = self.dec2(self._merge_skip(self.up2(dec3), enc2))
        dec1 = self.dec1(self._merge_skip(self.up1(dec2), enc1))
        if self._head_drop is not None:
            dec1 = self._head_drop(dec1)
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

    def _bag_score(self, voxel_map: torch.Tensor) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bottleneck, decoded = self._encode_decode(x)
        voxel_map = self.out_head(decoded)
        self.last_voxel_map = voxel_map.detach()
        self.last_voxel_map_full = voxel_map  # non-detached; needed for spill_entropy gradient flow
        # aggregate only the center tile region so the score is anchored to the label footprint
        center_voxels = self._crop_to_center_tile(voxel_map)
        self.last_center_voxel_map = center_voxels
        embedding = self._embedding(bottleneck)
        self.last_embedding_detached = embedding.detach()
        domain_logits = self.domain_head(embedding, grl_scale=grl_scale) if self.domain_head is not None else None
        supcon_z = self.supcon_head(embedding) if self.supcon_head is not None else None
        if self.depth_profile_head is not None:
            score = self.depth_profile_head(center_voxels)
        elif self.prototype_head is not None:
            score = self.prototype_head(embedding)
        else:
            score = self._bag_score(center_voxels)
        return score, embedding, domain_logits, supcon_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score, _, _, _ = self.forward_with_extras(x)
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
        elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            _init_norm(module)

    model.apply(init_weights)
    for module in model.modules():
        if isinstance(module, DepthSurfaceAttn):
            module.reset_output_layer()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({arch}): {params:,}")
    # compile the hot path; graph breaks at self.lastXxx assignments are fine
    model.forward_with_extras = torch.compile(model.forward_with_extras)
    return model, params