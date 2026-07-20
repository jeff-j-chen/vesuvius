"""model.py -- ink detection models.

current production architectures:
  v14_mil_deep   -- the winner: per-slice stem + 3D depth-mix + per-voxel logit head + LSE bag aggregation.
                    tile BCE (32x32->1). no global spatial pooling; sparse ink survives the aggregation.
  v14b_mil_zgrad -- variation: adds a depth-gradient input channel [raw, dI/dz]. ink between layers is
                    a DISCONTINUITY in the depth profile; dI/dz peaks at the interface and is baseline-invariant.
  v14c_mil_lcn   -- variation: local contrast normalization front-end [raw, lcn] + learnable depth
                    positional encoding. LCN removes 113keV bulk-density baseline; depth-PE lets the
                    model key on the absolute depth band where ink sits.

all architectures output a single (B, 1) tile logit aggregated via log-sum-exp over all voxel positions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class CBAM3D(nn.Module):
    """channel + spatial attention block used inside the depth-mix stage."""
    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spatial_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        attn = self.sigmoid_channel(self.channel_mlp(x_flat))
        attn = attn.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        x = x * (1 + self.channel_scale * (attn - 1)).float()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * (1 + self.spatial_scale * (sp - 1)).float()
        return x


def _mil_lse(vmap, lse_r, device):
    """MIL log-sum-exp: (B,1,D,H,W) voxel logits -> (B,1) tile logit.
    learnable r interpolates between mean (r->0) and max (r->inf)."""
    r = lse_r.clamp(min=0.5, max=10.0)
    flat = vmap.flatten(1)
    N = flat.shape[1]
    return (1.0 / r) * (
        torch.logsumexp(r * flat, dim=1, keepdim=True)
        - torch.log(torch.tensor(float(N), device=device))
    )


def _lcn2d(x5, k=5):
    """per-slice local contrast normalization: (B,1,D,H,W) -> (B,1,D,H,W)."""
    B, C, D, H, W = x5.shape
    xf = x5.reshape(B * D, C, H, W)
    mu = F.avg_pool2d(xf, k, stride=1, padding=k // 2)
    var = F.avg_pool2d(xf * xf, k, stride=1, padding=k // 2) - mu * mu
    return ((xf - mu) / torch.sqrt(var.clamp(min=1e-4))).reshape(B, C, D, H, W)


def _depth_mix(drop1=0.0, drop2=0.05):
    """shared depth-aware mixing stage: (B,64,D,H,W) -> (B,256,D,H/2,W/2)."""
    return nn.Sequential(
        nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
        CBAM3D(128), nn.MaxPool3d(kernel_size=(1, 2, 2)), nn.Dropout3d(drop1),
        nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        CBAM3D(256), nn.Dropout3d(drop2),
    )


class InkDetectorMILDeep(nn.Module):
    """v14_mil_deep: two-stage MIL detector.

    stage A -- per-slice (depth-kernel=1) texture stem: learns 2D features per depth slice
              independently (no cross-slice interaction).
    stage B -- full 3D depth-mix: learns which depth slice holds the ink interface.
    head    -- per-voxel logit + LSE aggregation: tile label = soft-max over voxel scores.

    the MIL framing is the key: because the tile label is driven by the MAXIMUM-evidence
    voxels (not the mean), sparse ink signal from a thin stroke survives aggregation.
    global spatial averaging -- used by v1 and asym_pool -- dilutes that signal ~1000x.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_mix = _depth_mix(d1, d2)
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(x)
        f = self.depth_mix(f)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


class InkDetectorMILDeepZGrad(nn.Module):
    """v14b_mil_zgrad: v14_mil_deep + depth-gradient input channel.

    physics: ink between papyrus layers is a DISCONTINUITY in the depth profile.
    the finite difference dI/dz peaks at the interface and is invariant to the
    slowly-varying bulk papyrus baseline that dominates absolute intensity at 113keV.
    the stem receives [raw, dz] so the interface signal is explicit from the start.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_mix = _depth_mix(d1, d2)
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        f = self.per_slice(torch.cat([x, dz], dim=1))
        f = self.depth_mix(f)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


class InkDetectorMILDeepLCN(nn.Module):
    """v14c_mil_lcn: v14_mil_deep + local contrast normalization + depth positional encoding.

    physics: at 113keV absolute voxel intensity is dominated by bulk papyrus density,
    not ink. LCN (subtracts local mean, divides by local std per slice) removes that
    baseline and exposes the faint local contrast where ink perturbs fiber structure.
    depth-PE lets the model key on the absolute depth band where ink sits.
    stem receives [raw, lcn]; positional encoding is added before depth-mixing.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_pe = nn.Parameter(torch.zeros(1, 64, 32, 1, 1))
        self.depth_mix = _depth_mix(d1, d2)
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(torch.cat([x, _lcn2d(x, 5)], dim=1))
        f = f + self.depth_pe[:, :, :f.shape[2]]
        f = self.depth_mix(f)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


_ARCH_MAP = {
    "v14_mil_deep":   InkDetectorMILDeep,
    "v14b_mil_zgrad": InkDetectorMILDeepZGrad,
    "v14c_mil_lcn":   InkDetectorMILDeepLCN,
}


def create_model(config: Config):
    """instantiate and weight-initialize the model specified in config.model.arch."""
    arch = str(getattr(config.model, "arch", "v14_mil_deep")).lower()
    if arch not in _ARCH_MAP:
        raise ValueError(f"unknown arch '{arch}'; valid: {sorted(_ARCH_MAP)}")
    model = _ARCH_MAP[arch](config).to(config.device)

    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({arch}): {params:,}")
    return model, params
