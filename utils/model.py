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


def _gauss1d(sigma: float, device=None) -> torch.Tensor:
    """1D Gaussian kernel for separable per-slice 2D Gaussian filtering."""
    k = int(2 * math.ceil(3.0 * sigma) + 1) | 1  # ensure odd
    x = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    g = torch.exp(-x**2 / (2.0 * sigma**2))
    return g / g.sum()


def _gauss2d_blur(xf: torch.Tensor, sigma: float) -> torch.Tensor:
    """separable 2D Gaussian blur on (N, C, H, W). operates per-channel."""
    g = _gauss1d(sigma, xf.device)
    k = g.shape[0]
    pad = k // 2
    C = xf.shape[1]
    kh = g.view(1, 1, 1, k).expand(C, 1, 1, k)   # (C, 1, 1, k)
    kv = g.view(1, 1, k, 1).expand(C, 1, k, 1)   # (C, 1, k, 1)
    xf = F.conv2d(xf, kh, padding=(0, pad), groups=C)
    xf = F.conv2d(xf, kv, padding=(pad, 0), groups=C)
    return xf


def _dog2d(x5, sigma1: float = 1.0, sigma2: float = 2.5) -> torch.Tensor:
    """per-slice Difference of Gaussians: (B,C,D,H,W) -> (B,C,D,H,W).
    fires positive on bright rings with radius ~(sigma2+sigma1)/2 px.
    physics: the ink-papyrus boundary creates a bright annular region at the
    paper surface that contracts/expands through depth layers as the X-ray
    beam crosses the ink deposit at different oblique angles."""
    B, C, D, H, W = x5.shape
    xf = x5.reshape(B * D, C, H, W)
    return (_gauss2d_blur(xf, sigma1) - _gauss2d_blur(xf, sigma2)).reshape(B, C, D, H, W)


# fixed Sobel kernels registered once (avoids re-allocation every forward pass)
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0


def _grad_mag2d(x5) -> torch.Tensor:
    """per-slice spatial gradient magnitude: (B,C,D,H,W) -> (B,C,D,H,W)."""
    B, C, D, H, W = x5.shape
    xf = x5.reshape(B * D, C, H, W)
    sx = _SOBEL_X.to(xf.device).expand(C, 1, 3, 3)
    sy = _SOBEL_Y.to(xf.device).expand(C, 1, 3, 3)
    gx = F.conv2d(xf, sx, padding=1, groups=C)
    gy = F.conv2d(xf, sy, padding=1, groups=C)
    return torch.sqrt(gx**2 + gy**2 + 1e-8).reshape(B, C, D, H, W)


def _surface_attn(x5, temp: float = 8.0) -> torch.Tensor:
    """soft depth attention peaked at the papyrus surface: (B,C,D,H,W) -> (B,C,D,H,W).
    surface = per-(y,x) depth where |dI/dz| is largest (papyrus-air intensity boundary).
    output is softmax(|dz|*temp) over depth, summing to 1 across D at each (y,x).

    physics: ink sits at the papyrus surface. this channel tells the stem 'how close is
    this voxel to the surface?' for each spatial position independently, compensating for
    papyrus undulation that shifts the surface depth across the tile."""
    dz = torch.zeros_like(x5)
    dz[:, :, 1:] = x5[:, :, 1:] - x5[:, :, :-1]
    return torch.softmax(dz.abs() * temp, dim=2)


def _surface_dist(x5, temp: float = 8.0) -> torch.Tensor:
    """signed normalized distance from the detected papyrus surface: (B,C,D,H,W).
    surface detected as soft-argmax of |dI/dz| over depth per (y,x) position.

    returns (z - z_surface(y,x)) / (D/2) in range [-1, +1].
    voxels above the surface (air side): negative. below (papyrus interior): positive.
    voxels at the surface: ~0.

    with this channel the stem's depth coordinate becomes RELATIVE to the surface rather
    than absolute, making ink features appear at depth_relative ~= 0 regardless of which
    absolute depth slice the wavy papyrus surface happens to be at for this tile."""
    B, C, D, H, W = x5.shape
    dz = torch.zeros_like(x5)
    dz[:, :, 1:] = x5[:, :, 1:] - x5[:, :, :-1]
    # soft-argmax: differentiable surface localization
    attn = torch.softmax(dz.abs() * temp, dim=2)   # (B, C, D, H, W)
    depth_idx = torch.arange(D, dtype=x5.dtype, device=x5.device).view(1, 1, D, 1, 1)
    surf_z = (attn * depth_idx).sum(dim=2, keepdim=True)  # (B, C, 1, H, W)
    return (depth_idx - surf_z) / max(D / 2.0, 1.0)  # (B, C, D, H, W)


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


def _dog_depth_max(x5, sigma1: float = 1.0, sigma2: float = 2.5) -> torch.Tensor:
    """depth-max of DoG: (B,C,D,H,W) -> (B,C,D,H,W) with same value broadcast.
    computes per-slice DoG then takes the max across ALL depth slices in the window
    and broadcasts it back. this tells the stem: 'at this spatial position, was there
    EVER a strong ring response in any depth of this 8-slice window?'
    addresses the wavy-papyrus problem: ink may only appear in 1-2 of the 8 slices,
    so per-slice DoG produces mostly zeros; the depth-max collapses this to a reliable
    presence map that doesn't depend on knowing which depth the ring is at."""
    dog = _dog2d(x5, sigma1, sigma2)              # (B, C, D, H, W)
    dmax = dog.max(dim=2, keepdim=True).values     # (B, C, 1, H, W) - peak ring response
    return dmax.expand_as(dog)                     # broadcast back to (B, C, D, H, W)
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
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_mix = _depth_mix(d1, d2)
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(x)
        f = self.depth_mix(f)
        vmap = self.voxel_head(self.head_drop(f))
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
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.per_slice = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_mix = _depth_mix(d1, d2)
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        f = self.per_slice(torch.cat([x, dz], dim=1))
        f = self.depth_mix(f)
        vmap = self.voxel_head(self.head_drop(f))
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
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.per_slice = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_pe = nn.Parameter(torch.zeros(1, 64, 32, 1, 1))
        self.depth_mix = _depth_mix(d1, d2)
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def get_voxel_logits(self, x, depth_offset=0):
        """extract per-voxel logit map (B,1,D,H/2,W/2) before MIL aggregation.
        depth_offset: absolute start slice index in the zarr, selects the correct
        depth_pe slice so the model receives accurate absolute-depth context."""
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(torch.cat([x, _lcn2d(x, 5)], dim=1))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        f = self.depth_mix(f)
        return self.voxel_head(self.head_drop(f))

    def encode(self, x, depth_offset=0):
        """feature extractor up to depth_mix (before the voxel head): (B,256,D,H/2,W/2).
        used by MAE pretraining so the shared stage-1 backbone can be warm-started."""
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(torch.cat([x, _lcn2d(x, 5)], dim=1))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        return self.depth_mix(f)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(torch.cat([x, _lcn2d(x, 5)], dim=1))
        f = f + self.depth_pe[:, :, :f.shape[2]]
        f = self.depth_mix(f)
        vmap = self.voxel_head(self.head_drop(f))
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


class InkDetectorTwoStageLCN(nn.Module):
    """v15_twostage_lcn: two-stage MIL detector with learned cross-window depth fusion.

    stage 1 (shared backbone, tied weights):
      v14c_mil_lcn applied to each of 3 non-overlapping 8-slice depth windows
      covering depth 4->28 (windows: 4-12, 12-20, 20-28).
      weights are TIED across windows -- same feature extractor, but each window
      receives its correct ABSOLUTE depth positional encoding (offset 4, 12, 20),
      so the model can genuinely distinguish depth bands (unlike single-window
      training where depth_pe is always indexed 0-7 regardless of absolute depth).

    stage 2 (small 3D CNN):
      fuses the 3 per-voxel logit maps (shape B,3,8,H/2,W/2) via learned convolutions.
      can learn cross-window patterns, e.g. "strong in window 2 but silent in 1+3"
      -- a signature the single-window model cannot detect.
      final MIL-LSE aggregates the fused map to a tile logit.

    why this differs from the old dense_unet:
      dense_unet used hard depth-max -> 2D U-Net decoder with dense pixel labels.
      this model uses soft MIL-LSE per window -> learned cross-window fusion -> tile
      label, staying fully within the MIL tile-label framing.

    input: (B, 1, 24, H, W) raw CT  (single channel; LCN computed per-window internally)
    depth layout within the 24-slice block:
      slices  0-7  -> absolute depth  4-12  (pe_offset=4)
      slices  8-15 -> absolute depth 12-20  (pe_offset=12)
      slices 16-23 -> absolute depth 20-28  (pe_offset=20)
    config: set depth=24, train_d_start=4, train_d_end=28, d_start=4, d_end=28
    """

    # (slice_start, slice_end, absolute_pe_offset) for each window
    WINDOWS = [(0, 8, 4), (8, 16, 12), (16, 24, 20)]

    def __init__(self, config: Config):
        super().__init__()
        # shared stage-1 backbone (tied weights across all 3 windows)
        self.stage1 = InkDetectorMILDeepLCN(config)

        # stage-2 fusion CNN: (B,3,D,H/2,W/2) -> (B,1,D,H/2,W/2)
        self.stage2 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1, bias=True),
        )
        # independent MIL-LSE temperature for stage 2
        self.lse_r2 = nn.Parameter(torch.tensor(2.0))
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        # stage 1: per-window voxel logit extraction with correct absolute depth PE
        voxel_maps = [
            self.stage1.get_voxel_logits(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in self.WINDOWS
        ]  # each (B, 1, 8, H/2, W/2)

        # stage 2: cross-window fusion
        v_cat = torch.cat(voxel_maps, dim=1)      # (B, 3, 8, H/2, W/2)
        fused = self.stage2(v_cat)                 # (B, 1, 8, H/2, W/2)
        self.last_voxel_map = fused.detach()

        # MIL-LSE aggregation on stage-2 output
        r = self.lse_r2.clamp(min=0.5, max=4.0)
        flat = fused.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=fused.device))
        )


class InkDetectorTwoStageWide(InkDetectorTwoStageLCN):
    """v15_twostage_wide: v15 with a higher-capacity stage-2 fusion CNN.

    the original fusion (3->16->8->1) was only ~4.8k params -- a tiny bottleneck sitting
    on top of three backbones. peak TRAIN PR-AUC plateaued ~0.66, i.e. the model underfits.
    this widens/deepens the fusion (3->32->32->16->1) so cross-window patterns have more
    capacity to be learned. backbone (stage 1) is unchanged / still tied across windows.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage2 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1, bias=True),
        )


class InkDetectorMILDeepLCNZGrad(InkDetectorMILDeepLCN):
    """v14c backbone + depth-gradient input channel: stem sees [raw, lcn, dI/dz].

    combines the LCN baseline-removal with the explicit ink-interface feature (dI/dz peaks
    at the sheet boundary and is invariant to the slow 113keV bulk baseline).
    """
    def __init__(self, config: Config):
        super().__init__(config)
        # override the stem to accept 3 input channels: [raw, lcn, dz]
        self.per_slice = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )

    def _stem_in(self, x):
        """build the [raw, lcn, dz] stem input for a (B,1,D,H,W) block."""
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return torch.cat([x, _lcn2d(x, 5), dz], dim=1)

    def get_voxel_logits(self, x, depth_offset=0):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        f = self.depth_mix(f)
        return self.voxel_head(self.head_drop(f))

    def encode(self, x, depth_offset=0):
        """[raw, lcn, dz] feature extractor up to depth_mix (before the voxel head).
        used by MAE pretraining so the shared stage-1 backbone can be warm-started."""
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        return self.depth_mix(f)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        f = f + self.depth_pe[:, :, :f.shape[2]]
        f = self.depth_mix(f)
        vmap = self.voxel_head(self.head_drop(f))
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


class InkDetectorMILDeepPhysics(InkDetectorMILDeepLCNZGrad):
    """v14d_physics: extends zgrad+lcn stem with ring-detector and sharpness channels.

    stem sees [raw, lcn, dz, dog, grad_mag] -> 5 channels.

    physics motivation (from direct scroll inspection):
    - dog (Difference of Gaussians): detects the bright RING pattern at the
      ink-papyrus boundary. Scrubbing through depth layers reveals an annular
      brightness that contracts/expands: cross-sections of the 3D ink deposit.
      DoG fires positive on bright rings at the scale set by (sigma1, sigma2).
    - grad_mag (|grad I_spatial|): captures the fuzz/clarity contrast. Ink regions
      are locally SHARPER than surrounding papyrus (hypothetically: carbonization
      compresses fibers). High |grad I| = sharp = ink; low = fuzzy = surrounding.

    both channels are fixed (not learned) operations on the raw CT input, so they
    cannot overfit -- they either expose more signal or they're ignored.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        # 5-channel stem: raw, lcn, dz, dog, grad_mag
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()

    def _stem_in(self, x):
        """5-channel input: [raw, lcn, dz, dog(ring-boundary scale), grad_mag(fiber scale)].
        dog sigma=(8,20): rings span ~100px, tile is 48px, so the tile sees only a
        ~10-20px wide ring-edge transition. DoG(8,20) fires on ~14px features = ring edge.
        sigma=(1,2.5) was wrong -- that detects papyrus fiber texture, not ring edges.
        grad_mag 3px Sobel is kept for fuzz/clarity (fiber-level contrast, correct scale)."""
        lcn = _lcn2d(x, 5)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        dog = _dog2d(x, sigma1=8.0, sigma2=20.0)
        grad = _grad_mag2d(x)
        return torch.cat([x, lcn, dz, dog, grad], dim=1)


class InkDetectorMILDeepPhysicsDepthMax(InkDetectorMILDeepPhysics):
    """physics stem variant: replaces per-slice DoG with depth-max DoG.

    the standard physics stem computes DoG independently per depth slice, so if ink
    only appears in 2 of the 8 window slices the other 6 get near-zero DoG responses
    (noise). this variant takes the MAX of DoG across all 8 slices and broadcasts it
    back -- making the 'was there a ring at this (y,x) position?' answer available at
    EVERY depth slice, regardless of which specific slice the ink is in.

    this directly addresses the wavy-papyrus problem: the papyrus surface undulates,
    so ink appears at different absolute depths across the tile. the depth-max DoG
    collapses that uncertainty into a single presence map per spatial position.

    stem sees [raw, lcn, dz, dog_depthmax, grad_mag] (5 channels, same structure).
    the dog channel is semantically different: not 'ring strength here at this depth'
    but 'ring strength here anywhere in this 8-slice window'.
    """

    def _stem_in(self, x):
        """5-channel input with depth-max DoG at ring-boundary scale."""
        lcn = _lcn2d(x, 5)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        dog_dmax = _dog_depth_max(x, sigma1=8.0, sigma2=20.0)
        grad = _grad_mag2d(x)
        return torch.cat([x, lcn, dz, dog_dmax, grad], dim=1)


class InkDetectorMILDeepSurface(InkDetectorMILDeepLCNZGrad):
    """surface-aware physics stem: [raw, lcn, dz, surface_dist, surface_attn] = 5 channels.

    surface_dist: per-(y,x) signed distance from the detected papyrus surface.
      makes the stem's depth coordinate RELATIVE to the surface, not absolute.
      voxels on the air side: negative. papyrus interior: positive. at surface: ~0.
      with this channel, ink features always appear near surface_dist=0 regardless
      of where in the 8-slice window the wavy papyrus surface actually is.

    surface_attn: softmax(|dI/dz| * temp) over depth, peaked at the surface.
      equivalent to 'surface proximity probability' per voxel -- high where the
      depth profile transitions (the surface boundary), low deep inside papyrus or air.

    both are differentiable (use soft-argmax, not hard argmax) so gradients flow
    through them if needed. currently used as fixed preprocessing.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()

    def _stem_in(self, x):
        """[raw, lcn, dz, surface_dist, surface_attn]."""
        lcn = _lcn2d(x, 5)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        sd = _surface_dist(x)
        sa = _surface_attn(x)
        return torch.cat([x, lcn, dz, sd, sa], dim=1)


class InkDetectorMILDeepSurfaceDog(InkDetectorMILDeepLCNZGrad):
    """combined physics stem: surface alignment + ring detection = 6 channels.
    stem sees [raw, lcn, dz, dog, surface_dist, surface_attn].

    this is the 'everything known' fixed-physics variant:
    - dog: spatial ring detector (bright annulus at ink-papyrus boundary)
    - surface_dist/surface_attn: depth alignment (compensates for wavy papyrus)
    together they encode both WHERE in-plane the ink ring is and WHERE in depth
    the papyrus surface is for each (y,x) position independently.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        d1 = float(getattr(config.model, "conv1_drop", 0.0))
        d2 = float(getattr(config.model, "conv2_drop", 0.05))
        self.per_slice = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        dh = float(getattr(config.model, "head_drop", 0.0))
        self.head_drop = nn.Dropout3d(dh) if dh > 0 else nn.Identity()

    def _stem_in(self, x):
        """[raw, lcn, dz, dog, surface_dist, surface_attn]."""
        lcn = _lcn2d(x, 5)
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        dog = _dog2d(x, sigma1=8.0, sigma2=20.0)  # ring-boundary scale
        sd = _surface_dist(x)
        sa = _surface_attn(x)
        return torch.cat([x, lcn, dz, dog, sd, sa], dim=1)


class DepthSurfaceAttn(nn.Module):
    """lightweight 1D-depth conv that learns which depth slices are surface-proximal.

    processes only the depth axis (kernel=(k,1,1)), so each (y,x) position in the tile
    is analyzed independently. learns a 'surface score' per depth slice that is used as
    a residual multiplicative amplifier on the per-slice backbone features:
        features *= (1 + sigmoid(depth_surface_attn(raw)))
    so surface-proximal slices can get up to 2x amplification; other slices unchanged.

    key capabilities (all learned, not hardcoded):
    - find the papyrus-air boundary (sharp |dz| transition)
    - distinguish which SIDE the ink is on (above vs below surface)
    - handle flaked/delaminated papyrus (two surfaces: two peaks in |dz|)
    - zero-suppress slices that are entirely in air or deep papyrus

    ~320 parameters total. initialized to near-zero output (no initial amplification).
    """
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=(1, 1, 1), bias=True),
        )
        # start near-zero (sigmoid(-2) ≈ 0.12): gentle initial amplification
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x):
        """x: (B, 1, D, H, W). returns (B, 1, D, H, W) in (0,1) via sigmoid."""
        return torch.sigmoid(self.net(x))


class InkDetectorMILDeepLearnedSurface(InkDetectorMILDeepLCNZGrad):
    """learned surface-aware backbone: adds a DepthSurfaceAttn module that learns
    which depth slices are surface-proximal.

    architecture change: after the per-slice stem extracts features (B, 64, D, H, W),
    a learned residual amplification is applied:
        f = f * (1.0 + depth_surface_attn(raw))
    where depth_surface_attn has only ~320 parameters (3 small 1D conv layers).

    vs the fixed physics alternatives (InkDetectorMILDeepSurface):
    - FIXED: surface_dist uses the |dz| peak directly. correct for clean papyrus.
      wrong if the surface boundary is not the sharpest transition in the window
      (e.g., if a fiber bundle creates a larger |dz| than the actual surface).
    - LEARNED: DepthSurfaceAttn learns what 'surface' means from the training signal.
      can adapt to multi-sheet papyrus, different surface sharpness across scrolls,
      and which SIDE of the surface has ink (not knowable from geometry alone).

    the depth PE (absolute position) and the surface attention (relative/learned)
    are complementary -- the model has both signals available simultaneously.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.depth_surface_attn = DepthSurfaceAttn(hidden=8)

    def get_voxel_logits(self, x, depth_offset=0):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        # learned surface attention: amplify surface-proximal slices by up to 2x
        attn = self.depth_surface_attn(x)   # (B, 1, D, H, W)
        f = f * (1.0 + attn)
        f = self.depth_mix(f)
        return self.voxel_head(self.head_drop(f))

    def encode(self, x, depth_offset=0):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        D = f.shape[2]
        f = f + self.depth_pe[:, :, depth_offset:depth_offset + D]
        attn = self.depth_surface_attn(x)
        f = f * (1.0 + attn)
        return self.depth_mix(f)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(self._stem_in(x))
        f = f + self.depth_pe[:, :, :f.shape[2]]
        attn = self.depth_surface_attn(x)
        f = f * (1.0 + attn)
        f = self.depth_mix(f)
        vmap = self.voxel_head(self.head_drop(f))
        self.last_voxel_map = vmap.detach()
        return _mil_lse(vmap, self.lse_r, x.device)


class InkDetectorTwoStageZGrad(InkDetectorTwoStageLCN):
    """v15_twostage_zgrad: v15 whose shared backbone also ingests dI/dz ([raw, lcn, dz])."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage1 = InkDetectorMILDeepLCNZGrad(config)


class InkDetectorTwoStageDense(InkDetectorTwoStageLCN):
    """v15_twostage_dense: emits a per-pixel (B,1,T,T) ink-prob logit map for DENSE BCE
    supervision instead of a single tile logit.

    dense supervision gives ~64x more gradient signal per tile (an 8x8 spatial grid vs one
    scalar), directly targeting the underfitting seen with tile-scalar MIL. the fused voxel
    map (B,1,D,H/2,W/2) is collapsed over depth (max = MIL over depth) then bilinearly
    upsampled back to tile resolution. requires config.data.dense_labels=True.
    """
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        voxel_maps = [
            self.stage1.get_voxel_logits(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in self.WINDOWS
        ]
        v_cat = torch.cat(voxel_maps, dim=1)      # (B,3,8,H/2,W/2)
        fused = self.stage2(v_cat)                 # (B,1,8,H/2,W/2)
        self.last_voxel_map = fused.detach()
        dmax = fused.max(dim=2).values             # collapse depth -> (B,1,H/2,W/2)
        T = x.shape[-1]
        return F.interpolate(dmax, size=(T, T), mode="bilinear", align_corners=False)


class InkDetectorTwoStageWideZGrad(InkDetectorTwoStageWide):
    """v15_twostage_wide_zgrad: wide stage-2 fusion (C) + zgrad backbone (E) combined.

    inherits the wide 3->32->32->16->1 fusion from InkDetectorTwoStageWide, then swaps the
    shared stage-1 backbone for the zgrad variant ([raw, lcn, dI/dz]). pair with the ranking
    loss (config.tra.ranking_lambda) to get the full C+D+E combo.
    """
    def __init__(self, config: Config):
        super().__init__(config)              # wide stage2 + default lcn stage1
        self.stage1 = InkDetectorMILDeepLCNZGrad(config)   # swap in zgrad backbone


class InkDetectorTwoStageWideZGradCtx(InkDetectorTwoStageWideZGrad):
    """context-window variant of v15_twostage_wide_zgrad. the input crop is LARGER than the
    label tile (config.data.context_size, e.g. 48px) and centered on the tile. the backbone
    sees the surround, but the final MIL-LSE pools ONLY the central tile_size region of the
    fused voxel map -- so the tile label / ring supervision is unchanged; context enters
    purely via the conv receptive field. degrades gracefully to the plain model if fed a
    tile-sized input (the center crop becomes the whole map).

    config.data.context_downsample (>1) avg-pools the input crop at the stem, keeping the full
    context EXTENT but at a coarser resolution -- cheaper + less overfit than shrinking the crop."""
    def __init__(self, config: Config):
        super().__init__(config)
        # coarse-context option: avg-pool the input by this factor at the stem so the model keeps
        # the FULL context extent but at a coarser resolution (fewer activations -> ~plain compute,
        # smaller overfit surface, no big-fragment inference OOM). 1 = off (full-res context).
        self._ds = max(1, int(getattr(config.data, "context_downsample", 1)))
        # tile region in POOLED coords: depth_mix pools H,W by 2 once, times the optional input
        # downsample -> total spatial reduction is (2 * self._ds).
        self._center = max(1, int(config.data.tile_size) // (2 * self._ds))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        # coarsen the whole context crop (depth preserved) before the backbone
        if self._ds > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self._ds, self._ds), stride=(1, self._ds, self._ds))
        voxel_maps = [
            self.stage1.get_voxel_logits(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in self.WINDOWS
        ]
        v_cat = torch.cat(voxel_maps, dim=1)               # (B,3,8,Hf,Wf)
        fused = self.stage2(v_cat)                          # (B,1,8,Hf,Wf), Hf=context/2
        # center-crop the fused map to the tile region so MIL reflects the CENTER tile only
        Hf, Wf = fused.shape[-2], fused.shape[-1]
        ch, cw = min(self._center, Hf), min(self._center, Wf)
        oy, ox = (Hf - ch) // 2, (Wf - cw) // 2
        center = fused[:, :, :, oy:oy + ch, ox:ox + cw]
        self.last_voxel_map = center.detach()
        r = self.lse_r2.clamp(min=0.5, max=10.0)
        flat = center.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=fused.device))
        )


class InkDetectorTwoStageWideZGradFovea(InkDetectorTwoStageWideZGrad):
    """foveated-context variant of v15_twostage_wide_zgrad.

    two tied-backbone streams, fused before MIL:
      center   -- the FULL-RESOLUTION central tile_size crop. preserves fine detail, which
                  matters at ~10um where ink is already near the resolution limit (prior models
                  resolved ink at 1-2um); coarsening the middle throws that away.
      surround -- the WHOLE context_size crop avg-pooled down to tile_size (coarse, but carries
                  the wider context the convs need to see a stroke continue past the tile).
    the surround's center-tile region is upsampled and fused with the center stream via a small
    1x1x1 head, then MIL-LSE aggregates the fused center-tile map.

    cost is ~2x a plain tile pass (two tile-sized backbone passes) vs the ~4x of full-res
    context, and unlike context_downsample the MIDDLE stays full res. adds a tiny fovea-fusion
    head (fresh weights); stage1 still warm-starts from MAE. requires context_size a multiple of
    tile_size (surround pools exactly to tile res)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._ctx = int(config.data.context_size)
        self._tile = int(config.data.tile_size)
        self._ds = max(2, self._ctx // self._tile)   # surround downsample -> lands at tile res
        # fuse [center_fused, surround_center_fused] (2 ch) -> 1 (fresh init, small)
        self.fovea_fuse = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=1, bias=False),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1, bias=True),
        )

    def _fuse_windows(self, xin):
        """stage1 (tied) over the 3 depth windows + wide stage2 fusion -> (B,1,8,h,w)."""
        vms = [self.stage1.get_voxel_logits(xin[:, :, z0:z1], depth_offset=pe)
               for z0, z1, pe in self.WINDOWS]
        return self.stage2(torch.cat(vms, dim=1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        C, T, ds = self._ctx, self._tile, self._ds
        off = (C - T) // 2
        # center stream: full-resolution central tile
        xc = x[:, :, :, off:off + T, off:off + T]
        fused_c = self._fuse_windows(xc)                       # (B,1,8,T/2,T/2)
        # surround stream: full extent, coarsened to tile res
        xs = F.avg_pool3d(x, kernel_size=(1, ds, ds), stride=(1, ds, ds))
        fused_s = self._fuse_windows(xs)                       # (B,1,8,T/2,T/2), covers full extent
        # pull the center-tile region out of the surround map and upsample to the center grid
        Hc, Wc = fused_c.shape[-2], fused_c.shape[-1]
        hs = fused_s.shape[-1]
        cc = max(1, hs // ds)
        o = (hs - cc) // 2
        s_c = fused_s[:, :, :, o:o + cc, o:o + cc]             # coarse center-tile
        B_, Ch_, D_, h_, w_ = s_c.shape
        s_c = s_c.permute(0, 2, 1, 3, 4).reshape(B_ * D_, Ch_, h_, w_)
        s_c = F.interpolate(s_c, size=(Hc, Wc), mode="bilinear", align_corners=False)
        s_c = s_c.reshape(B_, D_, Ch_, Hc, Wc).permute(0, 2, 1, 3, 4)
        # fuse the two resolutions over the center tile, then MIL-LSE
        fused = self.fovea_fuse(torch.cat([fused_c, s_c], dim=1))   # (B,1,8,T/2,T/2)
        self.last_voxel_map = fused.detach()
        r = self.lse_r2.clamp(min=0.5, max=10.0)
        flat = fused.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=fused.device))
        )


_ARCH_MAP = {
    "v14_mil_deep":       InkDetectorMILDeep,
    "v14b_mil_zgrad":     InkDetectorMILDeepZGrad,
    "v14c_mil_lcn":       InkDetectorMILDeepLCN,
    "v15_twostage_lcn":   InkDetectorTwoStageLCN,
    "v15_twostage_wide":  InkDetectorTwoStageWide,
    "v15_twostage_zgrad": InkDetectorTwoStageZGrad,
    "v15_twostage_dense": InkDetectorTwoStageDense,
    "v15_twostage_wide_zgrad": InkDetectorTwoStageWideZGrad,
    "v15_twostage_wide_zgrad_ctx": InkDetectorTwoStageWideZGradCtx,
    "v15_twostage_wide_zgrad_fovea": InkDetectorTwoStageWideZGradFovea,
    # v16_arch_ctx registered below AFTER InkDetectorArch is defined
}


# ============================================================================
# CAMPAIGN ARCHS: DANN, SupCon projection head, Attention-MIL
# ============================================================================

class _GradientReversal(torch.autograd.Function):
    """forward = identity, backward = multiply gradient by -lambda.
    lets the backbone be trained to FOOL the domain classifier (scroll id head)."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        lam, = ctx.saved_tensors
        return -lam * grad, None

def _grad_reversal(x, lam: float):
    return _GradientReversal.apply(x, lam)


class DomainHead(nn.Module):
    """2-layer MLP domain classifier (n_domains-way): takes the flattened tile embedding
    (the pre-aggregation features from stage-2 output, after center-crop) and predicts
    which scroll it came from. trained via a gradient-reversal layer so the backbone learns
    SCROLL-INVARIANT features. n_in = embedding dim = D*H*W from the center voxel map."""
    def __init__(self, n_in: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_domains),
        )
    def forward(self, emb, lam: float):
        return self.net(_grad_reversal(emb, lam))


class SupConHead(nn.Module):
    """2-layer projection head for supervised contrastive learning. takes the flattened tile
    embedding and projects to a unit-norm vector for the contrastive loss. follows Khosla 2020:
    linear -> relu -> linear -> l2-norm. proj_dim usually 128."""
    def __init__(self, n_in: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )
    def forward(self, emb):
        z = self.net(emb)
        return F.normalize(z, dim=-1)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """supervised contrastive loss (Khosla et al., 2020) for binary labels.
    z: (B, D) L2-normalized projections; labels: (B,) binary {0,1} long tensor.
    positives = same class, negatives = different class. returns scalar loss mean per anchor."""
    B = z.shape[0]
    if B < 2:
        return z.new_zeros(())
    # cosine sim matrix / temperature
    sim = torch.mm(z, z.T) / temp          # (B, B)
    # mask: same label (pos pairs), excluding self
    labels = labels.view(-1)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~torch.eye(B, dtype=torch.bool, device=z.device))
    # if no positive pair for any anchor, skip gracefully
    if not pos_mask.any():
        return z.new_zeros(())
    # log-softmax denominator: all j != i
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12))
    log_prob = sim - sim.max(dim=1, keepdim=True).values.detach() - log_denom
    # mean log-prob over POSITIVE pairs for each anchor
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    return loss_per_anchor.mean()


class DepthProfileHead(nn.Module):
    """projects the mean tile depth profile to a normalized embedding for contrastive learning.

    the spatial SupCon head learns a contrastive embedding from the spatial backbone features
    (after all convolutions). this head operates on the RAW mean depth profile at the center
    tile before any spatial processing -- a completely separate signal.

    motivation: ds=2 == ds=1 in performance, and surface_dist was the strongest learner.
    this confirms the ink signal is primarily in the DEPTH profile (how CT intensity varies
    with depth), not in the spatial texture within a slice. contrastive learning on depth
    profiles directly pulls ink depth signatures together across scrolls.

    input: (B, D_total) mean depth profile (averaged over tile center spatial positions).
    D_total = 24 = 3 windows x 8 slices, the full scan depth used for training.
    output: (B, proj_dim) L2-normalized embedding.
    """
    def __init__(self, n_depth: int = 24, proj_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_depth, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, depth_profile):
        """depth_profile: (B, D). returns (B, proj_dim) L2-normalized."""
        return F.normalize(self.net(depth_profile), dim=-1)


class GatedAttentionMIL(nn.Module):
    """gated attention-MIL aggregation (Ilse, Tomczak & Welling 2018).

    replaces the fixed LSE pooling with a LEARNED per-voxel attention: the tile score is
    the attention-weighted sum of projected voxel features.

    a_i = softmax(wᵀ (tanh(V hᵢ) ⊙ σ(U hᵢ)))
    score = Σ aᵢ · (Wout hᵢ)

    benefits over LSE:
      (1) discovers WHICH voxels carry ink signal -> better SNR on faint strokes.
      (2) attention map is a free sub-tile soft segmentation (shape improvement).
      (3) still returns ONE scalar per tile (MIL ban respected).
    """
    def __init__(self, feat_dim: int = 1, att_dim: int = 32):
        super().__init__()
        self.V = nn.Linear(feat_dim, att_dim, bias=False)
        self.U = nn.Linear(feat_dim, att_dim, bias=False)
        self.w = nn.Linear(att_dim, 1, bias=False)
        self.out = nn.Linear(feat_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.zeros_(self.w.weight)
        self.last_attn_weights = None   # (B, N) saved for visualization

    def forward(self, vmap: torch.Tensor, entropy_weight: float = 0.0) -> tuple:
        """vmap: (B, 1, D, H, W) or (B, C, D, H, W) voxel feature map.
        returns (score, entropy_loss) where entropy_loss is 0 if entropy_weight==0."""
        B = vmap.shape[0]
        h = vmap.flatten(2).permute(0, 2, 1)   # (B, N, C)
        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))   # (B, N, att_dim)
        a = self.w(gate).squeeze(-1)            # (B, N) raw logits
        a = torch.softmax(a, dim=-1)            # (B, N) attention weights
        self.last_attn_weights = a.detach()
        score = (a.unsqueeze(-1) * self.out(h)).sum(dim=1)  # (B, 1)
        
        # entropy regularization: prevent attention collapse
        entropy_loss = torch.tensor(0.0, device=vmap.device)
        if entropy_weight > 0:
            # entropy = -sum(p * log(p)), we want to maximize it (high entropy = spread out)
            # so we minimize -entropy (equivalent to maximizing entropy)
            entropy = -(a * torch.log(a + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -entropy_weight * entropy  # negative because we maximize entropy
        
        return score, entropy_loss


class InkDetectorArch(InkDetectorTwoStageWideZGradCtx):
    """v16_arch_ctx: campaign_archs baseline — ctx48/ds2 + optional DANN, SupCon,
    and Attention-MIL. all three features are gated by config flags so each can be
    tested individually and in combination. architecture core is UNCHANGED from
    InkDetectorTwoStageWideZGradCtx (same backbone / stage2 / center-crop / MIL-LSE).

    new config flags (all in config.tra or config.model):
      tra.dann              bool    False   DANN domain-adversarial head
      tra.dann_n_domains    int     15      number of scroll-id classes
      tra.supcon            bool    False   supervised contrastive projection head
      model.attn_mil        bool    False   replace LSE with gated attention-MIL
    """
    def __init__(self, config: Config):
        super().__init__(config)
        # pre-MIL center embedding dim (D * ch * cw after center-crop in pooled coords)
        # for ctx48/ds2: input 48/2=24 -> depth_mix maxpool -> Hf=12; center=tile//(2*ds)=4
        # so D=8, ch=cw=4 -> emb_dim = 8*4*4 = 128
        _ds = max(1, int(getattr(config.data, "context_downsample", 1)))
        _tile = int(config.data.tile_size)
        _ctr = max(1, _tile // (2 * _ds))
        self._emb_dim = 8 * _ctr * _ctr   # always 8 depth slices per window (after depth_mix)

        # physics stem: swap stage1 backbone for the appropriate variant
        # priority order: depthmax > physics > surfacedog > surface > learned
        if bool(getattr(config.model, "physics_stem_depthmax", False)):
            self.stage1 = InkDetectorMILDeepPhysicsDepthMax(config)
        elif bool(getattr(config.model, "physics_stem", False)):
            self.stage1 = InkDetectorMILDeepPhysics(config)
        elif bool(getattr(config.model, "surface_stem_withdog", False)):
            self.stage1 = InkDetectorMILDeepSurfaceDog(config)
        elif bool(getattr(config.model, "surface_stem", False)):
            self.stage1 = InkDetectorMILDeepSurface(config)
        elif bool(getattr(config.model, "learned_surface", False)):
            self.stage1 = InkDetectorMILDeepLearnedSurface(config)

        # DANN domain head
        self._use_dann = bool(getattr(config.tra, "dann", False))
        if self._use_dann:
            n_dom = int(getattr(config.tra, "dann_n_domains", 15))
            self.domain_head = DomainHead(self._emb_dim, n_dom, hidden=64)

        # SupCon projection head
        self._use_supcon = bool(getattr(config.tra, "supcon", False))
        if self._use_supcon:
            proj_dim = int(getattr(config.tra, "supcon_proj_dim", 128))
            hidden_dim = int(getattr(config.tra, "supcon_hidden_dim", 256))
            self.supcon_head = SupConHead(self._emb_dim, proj_dim=proj_dim, hidden=hidden_dim)

        # Attention-MIL (replaces LSE)
        self._use_attn_mil = bool(getattr(config.model, "attn_mil", False))
        self._attn_entropy_weight = float(getattr(config.model, "attn_entropy_weight", 0.0))
        if self._use_attn_mil:
            self.attn_mil = GatedAttentionMIL(feat_dim=1, att_dim=32)

        # depth profile SupCon head: contrastive on raw depth profiles (independent of spatial)
        self._use_depth_supcon = bool(getattr(config.tra, "depth_supcon", False))
        if self._use_depth_supcon:
            # n_depth=24 = 3 windows x 8 slices; proj_dim small (depth profiles are 1D, low info)
            self.depth_profile_head = DepthProfileHead(n_depth=24, proj_dim=32, hidden=64)

    def forward_with_extras(self, x):
        """like forward() but also returns (embedding, domain_logits, supcon_proj).
        used by train.py when DANN or SupCon is active to compute aux losses.
        non-active outputs are None. embedding is always returned (for DANN/SupCon)."""
        if x.dim() == 4: x = x.unsqueeze(1)
        if self._ds > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self._ds, self._ds), stride=(1, self._ds, self._ds))
        voxel_maps = [
            self.stage1.get_voxel_logits(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in self.WINDOWS
        ]
        v_cat = torch.cat(voxel_maps, dim=1)
        fused = self.stage2(v_cat)
        Hf, Wf = fused.shape[-2], fused.shape[-1]
        ch, cw = min(self._center, Hf), min(self._center, Wf)
        oy, ox = (Hf - ch) // 2, (Wf - cw) // 2
        center = fused[:, :, :, oy:oy + ch, ox:ox + cw]
        self.last_voxel_map = center.detach()

        # embedding: flatten center features for DANN / SupCon
        emb = center.flatten(1)   # (B, emb_dim)

        # depth profile at center tile: mean over spatial dims across all 24 depth slices.
        # captured BEFORE spatial convolutions -- this is the raw CT depth signature.
        # used by DepthProfileHead for depth-profile contrastive learning.
        if self._use_depth_supcon:
            # x is (B, 1, 24, H_eff, W_eff) after ds downsampling
            # extract center spatial region and average over it
            H_eff, W_eff = x.shape[-2], x.shape[-1]
            csize = min(self._center * 2, H_eff)   # center in effective (pre-pool) space
            oy_c = (H_eff - csize) // 2
            ox_c = (W_eff - csize) // 2
            depth_profile = x[:, 0, :, oy_c:oy_c+csize, ox_c:ox_c+csize].mean(dim=[-2, -1])  # (B, 24)
        else:
            depth_profile = None
        self.last_depth_profile = depth_profile

        # tile score: attention-MIL or LSE
        if self._use_attn_mil:
            # use stored entropy_weight from initialization
            tile_score, attn_entropy_loss = self.attn_mil(center, entropy_weight=self._attn_entropy_weight)
            # store for train.py to add to total loss
            self.last_attn_entropy_loss = attn_entropy_loss
        else:
            self.last_attn_entropy_loss = torch.tensor(0.0, device=center.device)
            r = self.lse_r2.clamp(min=0.5, max=10.0)
            flat = center.flatten(1)
            N = flat.shape[1]
            tile_score = (1.0 / r) * (
                torch.logsumexp(r * flat, dim=1, keepdim=True)
                - torch.log(torch.tensor(float(N), device=fused.device))
            )

        dom_logits = None
        supcon_z = None
        return tile_score, emb, dom_logits, supcon_z

    def forward(self, x):
        score, _, _, _ = self.forward_with_extras(x)
        return score


# register v16_arch_ctx AFTER the class is defined
_ARCH_MAP["v16_arch_ctx"] = InkDetectorArch


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
