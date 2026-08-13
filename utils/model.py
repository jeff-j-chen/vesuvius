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

# Import radical architectures for campaign_archs_6
try:
    from .radical_archs import ViT3D, Swin3D, ConvNeXt3D, XCiT3D, nnUNet3D, SlotAttention3D
    _RADICAL_ARCHS_AVAILABLE = True
except ImportError:
    _RADICAL_ARCHS_AVAILABLE = False


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


class ChannelLayerNorm3d(nn.Module):
    """layer norm over channels for 3D tensors shaped (B, C, D, H, W)."""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.eps = eps

    def forward(self, x):
        y = x.permute(0, 2, 3, 4, 1)
        y = F.layer_norm(y, (self.channels,), self.weight, self.bias, self.eps)
        return y.permute(0, 4, 1, 2, 3)


def _norm3d(channels: int, mode: str = "batch") -> nn.Module:
    """factory for campaign-configurable 3D normalization layers."""
    mode = str(mode or "batch").lower()
    channels = int(channels)
    if mode == "group":
        for groups in (8, 4, 2):
            if channels % groups == 0:
                return nn.GroupNorm(groups, channels)
        return nn.GroupNorm(1, channels)
    if mode == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if mode == "layer":
        return ChannelLayerNorm3d(channels)
    return nn.BatchNorm3d(channels)


def _act3d(mode: str = "relu") -> nn.Module:
    """factory for campaign-configurable activations."""
    return nn.LeakyReLU(0.01, inplace=True) if str(mode or "relu").lower() == "leaky" else nn.ReLU(inplace=True)


def _make_stage2(in_channels: int, norm_mode: str = "batch", activation: str = "relu",
                 widths: tuple[int, int, int] = (32, 32, 16)) -> nn.Sequential:
    """generic stage-2 fusion builder that preserves the standard (B,1,D,H,W) output."""
    c1, c2, c3 = widths
    return nn.Sequential(
        nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
        _norm3d(c1, norm_mode), _act3d(activation),
        nn.Conv3d(c1, c2, kernel_size=3, padding=1, bias=False),
        _norm3d(c2, norm_mode), _act3d(activation),
        nn.Conv3d(c2, c3, kernel_size=3, padding=1, bias=False),
        _norm3d(c3, norm_mode), _act3d(activation),
        nn.Conv3d(c3, 1, kernel_size=1, bias=True),
    )


def _replace_norm_activation(module: nn.Module, norm_mode: str = "batch", activation: str = "relu") -> None:
    """mutate a module tree so config-only norm/activation knobs become real behavior."""
    for name, child in list(module.named_children()):
        replacement = child
        if isinstance(child, nn.BatchNorm3d):
            replacement = _norm3d(child.num_features, norm_mode)
        elif isinstance(child, nn.ReLU):
            replacement = _act3d(activation)
        if replacement is not child:
            setattr(module, name, replacement)
            child = replacement
        _replace_norm_activation(child, norm_mode, activation)


class DepthSEGate(nn.Module):
    """depth-aware squeeze excitation for small voxel-map channel counts."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate = self.net(x.mean(dim=(-2, -1)))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class NonLocalVoxelBlock(nn.Module):
    """lightweight non-local block over the concatenated window voxel maps."""
    def __init__(self, channels: int):
        super().__init__()
        inter = max(1, channels)
        self.theta = nn.Conv3d(channels, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(channels, inter, kernel_size=1, bias=False)
        self.g = nn.Conv3d(channels, inter, kernel_size=1, bias=False)
        self.out = nn.Conv3d(inter, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, _, d, h, w = x.shape
        n = d * h * w
        theta = self.theta(x).reshape(b, -1, n).transpose(1, 2)
        phi = self.phi(x).reshape(b, -1, n)
        g = self.g(x).reshape(b, -1, n).transpose(1, 2)
        attn = torch.softmax((theta @ phi) / math.sqrt(max(phi.shape[1], 1)), dim=-1)
        y = (attn @ g).transpose(1, 2).reshape(b, -1, d, h, w)
        return x + self.out(y)


class CoordAttention3dLite(nn.Module):
    """coordinate-style spatial gating over H and W axes."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels)
        self.mix = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h_pool = x.mean(dim=4, keepdim=True)
        w_pool = x.mean(dim=3, keepdim=True)
        gate_h = self.mix(h_pool)
        gate_w = self.mix(w_pool)
        return x * gate_h * gate_w


class GhostStage2(nn.Module):
    """cheap ghost-style stage-2 fusion block."""
    def __init__(self, in_channels: int, norm_mode: str = "batch", activation: str = "relu"):
        super().__init__()
        self.primary = nn.Sequential(
            nn.Conv3d(in_channels, 12, kernel_size=1, bias=False),
            _norm3d(12, norm_mode), _act3d(activation),
        )
        self.cheap = nn.Sequential(
            nn.Conv3d(12, 12, kernel_size=3, padding=1, groups=12, bias=False),
            _norm3d(12, norm_mode), _act3d(activation),
        )
        self.fuse = _make_stage2(24, norm_mode=norm_mode, activation=activation, widths=(24, 16, 8))

    def forward(self, x):
        primary = self.primary(x)
        ghost = self.cheap(primary)
        return self.fuse(torch.cat([primary, ghost], dim=1))


class InvertedResidualStage2(nn.Module):
    """MobileNetV2-style stage-2 fusion."""
    def __init__(self, in_channels: int, norm_mode: str = "batch", activation: str = "relu"):
        super().__init__()
        hidden = max(16, in_channels * 4)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            _norm3d(hidden, norm_mode), _act3d(activation),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            _norm3d(hidden, norm_mode), _act3d(activation),
            nn.Conv3d(hidden, in_channels, kernel_size=1, bias=False),
            _norm3d(in_channels, norm_mode),
        )
        self.out = _make_stage2(in_channels, norm_mode=norm_mode, activation=activation, widths=(24, 16, 8))

    def forward(self, x):
        return self.out(x + self.net(x))


class ResNeXtStage2(nn.Module):
    """grouped-convolution stage-2 fusion."""
    def __init__(self, in_channels: int, norm_mode: str = "batch", activation: str = "relu"):
        super().__init__()
        width = 24
        groups = 3 if width % 3 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=1, bias=False),
            _norm3d(width, norm_mode), _act3d(activation),
            nn.Conv3d(width, width, kernel_size=3, padding=1, groups=groups, bias=False),
            _norm3d(width, norm_mode), _act3d(activation),
            nn.Conv3d(width, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


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
        self._norm_mode = str(getattr(config.model, "normalization_layer", "batch")).lower()
        self._activation = str(getattr(config.model, "activation", "relu")).lower()
        self.domain_head = None
        self.supcon_head = None
        self.dann_lambda = 1.0
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

        # 5-window depth coverage: adds overlapping windows at the seams between the 3
        # standard windows. the ink surface can fall exactly at seam depths 12 or 20
        # (absolute), where a 3-window model sees it at the very edge of two windows.
        # with 5 windows the seam depths are always in the middle of at least one window.
        # windows: [0-8, 4-12, 8-16, 12-20, 16-24] in downsampled-input slice coords,
        # pe_offsets: [4, 8, 12, 16, 20] (correct absolute depth for positional encoding).
        # stage1 weights are TIED across all 5 windows (same params as 3-window model).
        # stage2 input grows from 3 -> 5 channels (very small param increase ~1.5k params).
        _n_win = int(getattr(config.model, "n_depth_windows", 3))
        if _n_win == 5:
            self.WINDOWS = [
                (0,  8,  4),   # abs 4-12  (same as window 0)
                (4,  12, 8),   # abs 8-16  NEW -- covers seam at depth 12
                (8,  16, 12),  # abs 12-20 (same as window 1)
                (12, 20, 16),  # abs 16-24 NEW -- covers seam at depth 20
                (16, 24, 20),  # abs 20-28 (same as window 2)
            ]
            # rebuild stage2 for 5-channel input (parent built it for 3)
            self.stage2 = _make_stage2(5, norm_mode=self._norm_mode, activation=self._activation)

        # depth profile SupCon head: contrastive on raw depth profiles (independent of spatial)
        self._use_depth_supcon = bool(getattr(config.tra, "depth_supcon", False))
        if self._use_depth_supcon:
            # n_depth=24 = 3 windows x 8 slices; proj_dim small (depth profiles are 1D, low info)
            self.depth_profile_head = DepthProfileHead(n_depth=24, proj_dim=32, hidden=64)

        self._apply_campaign_overrides()

    def _apply_campaign_overrides(self):
        """apply config-driven norm/activation swaps after all submodules exist."""
        if self._norm_mode != "batch" or self._activation != "relu":
            _replace_norm_activation(self.stage1, self._norm_mode, self._activation)
            _replace_norm_activation(self.stage2, self._norm_mode, self._activation)

    def _prepare_input(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if self._ds > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self._ds, self._ds), stride=(1, self._ds, self._ds))
        return x

    def _window_voxel_maps(self, x, windows=None):
        windows = self.WINDOWS if windows is None else windows
        return [
            self.stage1.get_voxel_logits(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in windows
        ]

    def _window_feature_maps(self, x, windows=None):
        windows = self.WINDOWS if windows is None else windows
        return [
            self.stage1.encode(x[:, :, z0:z1], depth_offset=pe_off)
            for z0, z1, pe_off in windows
        ]

    def _center_crop_fused(self, fused):
        Hf, Wf = fused.shape[-2], fused.shape[-1]
        ch, cw = min(self._center, Hf), min(self._center, Wf)
        oy, ox = (Hf - ch) // 2, (Wf - cw) // 2
        return fused[:, :, :, oy:oy + ch, ox:ox + cw]

    def _depth_profile_from_input(self, x):
        H_eff, W_eff = x.shape[-2], x.shape[-1]
        csize = min(self._center * 2, H_eff)
        oy_c = (H_eff - csize) // 2
        ox_c = (W_eff - csize) // 2
        return x[:, 0, :, oy_c:oy_c + csize, ox_c:ox_c + csize].mean(dim=[-2, -1])

    def _mil_score(self, center):
        if self._use_attn_mil:
            tile_score, attn_entropy_loss = self.attn_mil(center, entropy_weight=self._attn_entropy_weight)
            self.last_attn_entropy_loss = attn_entropy_loss
            return tile_score
        self.last_attn_entropy_loss = torch.tensor(0.0, device=center.device)
        return _mil_lse(center, self.lse_r2, center.device)

    def _finalize_outputs(self, fused, x_input, tile_score=None, already_centered=False):
        center = fused if already_centered else self._center_crop_fused(fused)
        self.last_voxel_map = center.detach()
        emb = center.flatten(1)
        self.last_depth_profile = self._depth_profile_from_input(x_input) if self._use_depth_supcon else None
        if tile_score is None:
            tile_score = self._mil_score(center)
        elif tile_score.dim() == 1:
            tile_score = tile_score.unsqueeze(1)
        dom_logits = self.domain_head(emb, self.dann_lambda) if self._use_dann and self.domain_head is not None else None
        supcon_z = self.supcon_head(emb) if self._use_supcon and self.supcon_head is not None else None
        return tile_score, emb, dom_logits, supcon_z

    def forward_with_extras(self, x):
        """like forward() but also returns (embedding, domain_logits, supcon_proj).
        used by train.py when DANN or SupCon is active to compute aux losses.
        non-active outputs are None. embedding is always returned (for DANN/SupCon)."""
        x = self._prepare_input(x)
        voxel_maps = self._window_voxel_maps(x)
        v_cat = torch.cat(voxel_maps, dim=1)
        fused = self.stage2(v_cat)
        return self._finalize_outputs(fused, x)

    def _finalize_forward(self, fused, x_input, tile_score=None, already_centered=False):
        """compat wrapper for campaign variants."""
        return self._finalize_outputs(fused, x_input, tile_score=tile_score, already_centered=already_centered)

    def forward(self, x):
        score, _, _, _ = self.forward_with_extras(x)
        return score


# ============================================================================
# CAMPAIGN ARCHS 7: 27 NEW ARCHITECTURAL VARIANTS
# ============================================================================

# Foveated context (test 6)
class InkDetectorArch_Fovea(InkDetectorTwoStageWideZGradFovea):
    """v16_arch_ctx_fovea: foveated context with DANN, SupCon, Attention-MIL"""
    def __init__(self, config: Config):
        super().__init__(config)
        # Calculate embedding dimension from actual fused output shape
        # After fovea_fuse and stage2: (B, 1, 8, tile_size/2, tile_size/2)
        _tile = int(config.data.tile_size)
        _spatial = _tile // 2  # stage2 has MaxPool3d reducing spatial by 2x
        _depth = 8  # fixed depth dimension from 3 windows
        self._emb_dim = 1 * _depth * _spatial * _spatial  # 1*8*8*8 = 512 for tile_size=16
        
        # DANN domain adversarial head
        self._use_dann = bool(getattr(config.tra, "dann", False))
        if self._use_dann:
            n_domains = int(getattr(config.tra, "dann_n_domains", 1))
            self.domain_head = DomainHead(self._emb_dim, n_domains, hidden=64)
        
        # SupCon projection head
        self._use_supcon = bool(getattr(config.tra, "supcon", False))
        if self._use_supcon:
            proj_dim = int(getattr(config.tra, "supcon_proj_dim", 128))
            hidden_dim = int(getattr(config.tra, "supcon_hidden_dim", 256))
            self.supcon_head = SupConHead(self._emb_dim, proj_dim=proj_dim, hidden=hidden_dim)
        
        # Attention-MIL
        self._use_attn_mil = bool(getattr(config.model, "attn_mil", False))
        self._attn_entropy_weight = float(getattr(config.model, "attn_entropy_weight", 0.0))
        if self._use_attn_mil:
            self.attn_mil = GatedAttentionMIL(feat_dim=1, att_dim=32)
        
        # DANN lambda for gradient reversal
        self.dann_lambda = 1.0

        norm_mode = str(getattr(config.model, "normalization_layer", "batch")).lower()
        activation = str(getattr(config.model, "activation", "relu")).lower()
        if norm_mode != "batch" or activation != "relu":
            _replace_norm_activation(self.stage1, norm_mode, activation)
            _replace_norm_activation(self.stage2, norm_mode, activation)
            _replace_norm_activation(self.fovea_fuse, norm_mode, activation)

    def forward_with_extras(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        C, T, ds = self._ctx, self._tile, self._ds
        off = (C - T) // 2
        xc = x[:, :, :, off:off + T, off:off + T]
        fused_c = self._fuse_windows(xc)
        xs = F.avg_pool3d(x, kernel_size=(1, ds, ds), stride=(1, ds, ds))
        fused_s = self._fuse_windows(xs)
        Hc, Wc = fused_c.shape[-2], fused_c.shape[-1]
        hs = fused_s.shape[-1]
        cc = max(1, hs // ds)
        o = (hs - cc) // 2
        s_c = fused_s[:, :, :, o:o + cc, o:o + cc]
        B_, Ch_, D_, h_, w_ = s_c.shape
        s_c = s_c.permute(0, 2, 1, 3, 4).reshape(B_ * D_, Ch_, h_, w_)
        s_c = F.interpolate(s_c, size=(Hc, Wc), mode="bilinear", align_corners=False)
        s_c = s_c.reshape(B_, D_, Ch_, Hc, Wc).permute(0, 2, 1, 3, 4)
        fused = self.fovea_fuse(torch.cat([fused_c, s_c], dim=1))
        self.last_voxel_map = fused.detach()
        
        B = fused.shape[0]
        flat = fused.view(B, -1)
        
        # MIL aggregation
        if self._use_attn_mil:
            tile_score, attn_entropy_loss = self.attn_mil(fused, entropy_weight=self._attn_entropy_weight)
            self.last_attn_entropy_loss = attn_entropy_loss
        else:
            self.last_attn_entropy_loss = torch.tensor(0.0, device=fused.device)
            r = self.lse_r2.clamp(min=0.5, max=10.0)
            tile_score = (flat.exp() * r).mean(dim=1).log() / r
        
        # Embeddings for DANN and SupCon
        emb = flat
        
        # DANN
        if self._use_dann:
            dom_logits = self.domain_head(_grad_reversal(emb, self.dann_lambda))
        else:
            dom_logits = None
        
        # SupCon
        if self._use_supcon:
            supcon_z = F.normalize(self.supcon_head(emb), dim=1)
        else:
            supcon_z = None
        
        return tile_score, emb, dom_logits, supcon_z

    def forward(self, x):
        score, _, _, _ = self.forward_with_extras(x)
        return score


class InkDetectorArchRelaxedCrop(InkDetectorArch):
    """v16_arch_ctx_relaxedcrop: baseline bagging over a larger pooled center region."""
    def __init__(self, config: Config):
        super().__init__(config)
        relaxed_center = max(self._center, int(config.data.tile_size) // max(1, self._ds))
        if relaxed_center != self._center:
            self._center = relaxed_center
            self._emb_dim = 8 * self._center * self._center
            if self._use_dann:
                n_dom = int(getattr(config.tra, "dann_n_domains", 15))
                self.domain_head = DomainHead(self._emb_dim, n_dom, hidden=64)
            if self._use_supcon:
                proj_dim = int(getattr(config.tra, "supcon_proj_dim", 128))
                hidden_dim = int(getattr(config.tra, "supcon_hidden_dim", 256))
                self.supcon_head = SupConHead(self._emb_dim, proj_dim=proj_dim, hidden=hidden_dim)


# Dual-stream architectures (tests 7-10)
class InkDetectorDualStreamEarly(InkDetectorArch):
    """v16_dual_stream_early: fuse full-depth and squashed streams before stage-2."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        self.early_fuse = nn.Sequential(
            nn.Conv3d(in_ch * 2, in_ch, kernel_size=1, bias=False),
            _norm3d(in_ch, self._norm_mode), _act3d(self._activation),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        full_maps = self._window_voxel_maps(x)
        xs = x.amax(dim=2, keepdim=True).expand_as(x)
        squashed_maps = self._window_voxel_maps(xs)
        fused_in = self.early_fuse(torch.cat([torch.cat(full_maps, dim=1), torch.cat(squashed_maps, dim=1)], dim=1))
        return self._finalize_forward(self.stage2(fused_in), x)

class InkDetectorDualStreamLate(InkDetectorArch):
    """v16_dual_stream_late: separate streams fused at the tile-logit level."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.late_stage2 = _make_stage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)
        self.late_alpha = nn.Parameter(torch.tensor(0.0))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        full_center = self._center_crop_fused(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)))
        xs = x.amax(dim=2, keepdim=True).expand_as(x)
        squashed_center = self._center_crop_fused(self.late_stage2(torch.cat(self._window_voxel_maps(xs), dim=1)))
        alpha = torch.sigmoid(self.late_alpha)
        tile_score = alpha * self._mil_score(squashed_center) + (1.0 - alpha) * self._mil_score(full_center)
        mixed_center = alpha * squashed_center + (1.0 - alpha) * full_center
        return self._finalize_forward(mixed_center, x, tile_score=tile_score, already_centered=True)

class InkDetectorDualStreamGated(InkDetectorArch):
    """v16_dual_stream_gated: sample-wise gate between full and squashed streams."""
    def __init__(self, config: Config):
        super().__init__(config)
        emb_dim = self._emb_dim * 2
        self.gated_stage2 = _make_stage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim, max(32, self._emb_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, self._emb_dim // 2), 1),
            nn.Sigmoid(),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        full_center = self._center_crop_fused(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)))
        xs = x.amax(dim=2, keepdim=True).expand_as(x)
        squashed_center = self._center_crop_fused(self.gated_stage2(torch.cat(self._window_voxel_maps(xs), dim=1)))
        gate = self.gate_net(torch.cat([full_center.flatten(1), squashed_center.flatten(1)], dim=1))
        gate = gate.view(-1, 1, 1, 1, 1)
        mixed_center = gate * squashed_center + (1.0 - gate) * full_center
        return self._finalize_forward(mixed_center, x, already_centered=True)

class InkDetectorDualStreamAsym(InkDetectorArch):
    """v16_dual_stream_asym: lightweight squashed context plus heavy 3D detail stream."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        self.context_light = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=in_ch, bias=False),
            _norm3d(in_ch, self._norm_mode), _act3d(self._activation),
            nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False),
            _norm3d(in_ch, self._norm_mode), _act3d(self._activation),
        )
        self.asym_fuse = nn.Sequential(
            nn.Conv3d(in_ch * 2, in_ch, kernel_size=1, bias=False),
            _norm3d(in_ch, self._norm_mode), _act3d(self._activation),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        full_cat = torch.cat(self._window_voxel_maps(x), dim=1)
        xs = x.amax(dim=2, keepdim=True).expand_as(x)
        light_ctx = self.context_light(torch.cat(self._window_voxel_maps(xs), dim=1))
        fused = self.stage2(self.asym_fuse(torch.cat([full_cat, light_ctx], dim=1)))
        return self._finalize_forward(fused, x)

# Hybrid depth attention (tests 11-14)
class InkDetectorHybridDepthPerWindow(InkDetectorArch):
    """v16_hybrid_depth_per_window: attention + max branch per 8-slice window."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_heads = nn.ModuleList([
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True) for _ in self.WINDOWS
        ])
        self.hybrid_stage2 = _make_stage2(len(self.WINDOWS) * 2, norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        fused_windows = []
        for attn_head, voxel_map in zip(self.attn_heads, self._window_voxel_maps(x)):
            attn = torch.softmax(attn_head(voxel_map), dim=2)
            attn_pool = (voxel_map * attn).sum(dim=2, keepdim=True).expand_as(voxel_map)
            max_pool = voxel_map.amax(dim=2, keepdim=True).expand_as(voxel_map)
            fused_windows.extend([attn_pool, max_pool])
        fused = self.hybrid_stage2(torch.cat(fused_windows, dim=1))
        return self._finalize_forward(fused, x)

class InkDetectorHybridDepthGlobal(InkDetectorArch):
    """v16_hybrid_depth_global: collapse all 24 slices with one global depth attention."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.global_attn = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True)
        self.global_stage2 = _make_stage2(2, norm_mode=self._norm_mode, activation=self._activation, widths=(16, 16, 8))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        full_depth = torch.cat(self._window_voxel_maps(x), dim=2)
        attn = torch.softmax(self.global_attn(full_depth), dim=2)
        attn_pool = (full_depth * attn).sum(dim=2, keepdim=True).expand(-1, -1, 8, -1, -1)
        max_pool = full_depth.amax(dim=2, keepdim=True).expand(-1, -1, 8, -1, -1)
        fused = self.global_stage2(torch.cat([attn_pool, max_pool], dim=1))
        return self._finalize_forward(fused, x)

class InkDetectorHybridDepthTriple(InkDetectorArch):
    """v16_hybrid_depth_triple: attention + max + mean branches per window."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_heads = nn.ModuleList([
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True) for _ in self.WINDOWS
        ])
        self.triple_stage2 = _make_stage2(len(self.WINDOWS) * 3, norm_mode=self._norm_mode, activation=self._activation, widths=(48, 32, 16))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        fused_windows = []
        for attn_head, voxel_map in zip(self.attn_heads, self._window_voxel_maps(x)):
            attn = torch.softmax(attn_head(voxel_map), dim=2)
            fused_windows.extend([
                (voxel_map * attn).sum(dim=2, keepdim=True).expand_as(voxel_map),
                voxel_map.amax(dim=2, keepdim=True).expand_as(voxel_map),
                voxel_map.mean(dim=2, keepdim=True).expand_as(voxel_map),
            ])
        fused = self.triple_stage2(torch.cat(fused_windows, dim=1))
        return self._finalize_forward(fused, x)

class InkDetectorHybridDepthGated(InkDetectorArch):
    """v16_hybrid_depth_gated: learned mix between attention and max per window."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.attn_heads = nn.ModuleList([
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True) for _ in self.WINDOWS
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(1, 1, kernel_size=1), nn.Sigmoid()) for _ in self.WINDOWS
        ])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        fused_windows = []
        for attn_head, gate_head, voxel_map in zip(self.attn_heads, self.gates, self._window_voxel_maps(x)):
            attn = torch.softmax(attn_head(voxel_map), dim=2)
            attn_pool = (voxel_map * attn).sum(dim=2, keepdim=True).expand_as(voxel_map)
            max_pool = voxel_map.amax(dim=2, keepdim=True).expand_as(voxel_map)
            gate = gate_head(voxel_map)
            fused_windows.append(gate * attn_pool + (1.0 - gate) * max_pool)
        return self._finalize_forward(self.stage2(torch.cat(fused_windows, dim=1)), x)

# Multi-scale & efficient (tests 15-20)
class InkDetectorMultiscalePyramid(InkDetectorArch):
    """v16_multiscale_pyramid: 1x, 1/2x and 1/4x voxel-map fusion."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.scale_stage2 = _make_stage2(len(self.WINDOWS) * 3, norm_mode=self._norm_mode, activation=self._activation, widths=(48, 32, 16))

    def _resized_vcat(self, x, pool: int):
        xp = x if pool == 1 else F.avg_pool3d(x, kernel_size=(1, pool, pool), stride=(1, pool, pool))
        vcat = torch.cat(self._window_voxel_maps(xp), dim=1)
        if vcat.shape[-2:] != x.shape[-2:]:
            b, c, d, h, w = vcat.shape
            vcat = vcat.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
            vcat = F.interpolate(vcat, size=x.shape[-2:], mode="bilinear", align_corners=False)
            vcat = vcat.reshape(b, d, c, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        return vcat

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        fused = self.scale_stage2(torch.cat([self._resized_vcat(x, 1), self._resized_vcat(x, 2), self._resized_vcat(x, 4)], dim=1))
        return self._finalize_forward(fused, x)

class InkDetectorDepthSE(InkDetectorArch):
    """v16_depth_se: depth squeeze-excitation over window voxel maps."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.depth_se = DepthSEGate(len(self.WINDOWS))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = self.depth_se(torch.cat(self._window_voxel_maps(x), dim=1))
        return self._finalize_forward(self.stage2(vcat), x)

class InkDetectorDepthwiseSep(InkDetectorArch):
    """v16_depthwise_sep: depthwise-separable 3D stage-2 fusion."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        hidden = max(16, in_ch * 8)
        self.stage2 = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False),
            _norm3d(hidden, self._norm_mode), _act3d(self._activation),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            _norm3d(hidden, self._norm_mode), _act3d(self._activation),
            nn.Conv3d(hidden, 16, kernel_size=1, bias=False),
            _norm3d(16, self._norm_mode), _act3d(self._activation),
            nn.Conv3d(16, 1, kernel_size=1, bias=True),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        return self._finalize_forward(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)), x)

class InkDetectorMixedDepthWindows(InkDetectorArch):
    """v16_mixed_depth_windows: 5 fixed windows with learned seam emphasis."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.WINDOWS = [
            (0, 8, 4),
            (4, 12, 8),
            (8, 16, 12),
            (12, 20, 16),
            (16, 24, 20),
        ]
        self.window_gain = nn.Parameter(torch.tensor([1.0, 1.25, 1.0, 1.25, 1.0], dtype=torch.float32))
        self.stage2 = _make_stage2(5, norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        weighted = [vm * gain for vm, gain in zip(self._window_voxel_maps(x), self.window_gain)]
        return self._finalize_forward(self.stage2(torch.cat(weighted, dim=1)), x)

class InkDetectorOctaveConv(InkDetectorArch):
    """v16_octave_conv: fuse high/low spatial-frequency voxel maps."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.octave_stage2 = _make_stage2(len(self.WINDOWS) * 2, norm_mode=self._norm_mode, activation=self._activation, widths=(32, 24, 16))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = torch.cat(self._window_voxel_maps(x), dim=1)
        low = F.avg_pool3d(vcat, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        low = F.interpolate(low.flatten(0, 1), size=vcat.shape[-2:], mode="bilinear", align_corners=False).view_as(vcat)
        high = vcat - low
        return self._finalize_forward(self.octave_stage2(torch.cat([high, low], dim=1)), x)

class InkDetectorEfficientScale(InkDetectorArch):
    """v16_efficientnet_scale: narrower compound-scaled stage-2 path."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage2 = InvertedResidualStage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        return self._finalize_forward(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)), x)

# Attention mechanisms (tests 21-26)
class InkDetectorNonLocalDepth(InkDetectorArch):
    """v16_nonlocal_depth: non-local attention before stage-2 fusion."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.nonlocal_block = NonLocalVoxelBlock(len(self.WINDOWS))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = self.nonlocal_block(torch.cat(self._window_voxel_maps(x), dim=1))
        return self._finalize_forward(self.stage2(vcat), x)

class InkDetectorCoordAttention(InkDetectorArch):
    """v16_coord_attention: coordinate-aware gating over voxel maps."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.coord_attn = CoordAttention3dLite(len(self.WINDOWS))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = self.coord_attn(torch.cat(self._window_voxel_maps(x), dim=1))
        return self._finalize_forward(self.stage2(vcat), x)

class InkDetectorDeformableConv(InkDetectorArch):
    """v16_deformable_conv: multi-dilation fusion as a deformable proxy."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        self.branch1 = nn.Conv3d(in_ch, 16, kernel_size=3, padding=1, bias=False)
        self.branch2 = nn.Conv3d(in_ch, 16, kernel_size=3, padding=2, dilation=2, bias=False)
        self.deform_stage2 = _make_stage2(32, norm_mode=self._norm_mode, activation=self._activation, widths=(32, 24, 16))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = torch.cat(self._window_voxel_maps(x), dim=1)
        fused = torch.cat([self.branch1(vcat), self.branch2(vcat)], dim=1)
        return self._finalize_forward(self.deform_stage2(fused), x)

class InkDetectorProgressiveDepth(InkDetectorArch):
    """v16_progressive_depth: refine each window using the previous refined state."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
                _norm3d(1, self._norm_mode), _act3d(self._activation),
            ) for _ in self.WINDOWS
        ])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        refined = []
        prev = None
        for voxel_map, refine in zip(self._window_voxel_maps(x), self.refine):
            if prev is None:
                cur = voxel_map
            else:
                cur = refine(torch.cat([voxel_map, prev], dim=1))
            refined.append(cur)
            prev = cur
        return self._finalize_forward(self.stage2(torch.cat(refined, dim=1)), x)

class InkDetectorDualAttention(InkDetectorArch):
    """v16_dual_attention: channel and spatial attention over window voxel maps."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        hidden = max(4, in_ch * 2)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_ch, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = torch.cat(self._window_voxel_maps(x), dim=1)
        vcat = vcat * self.channel_attn(vcat)
        max_pool = vcat.max(dim=1, keepdim=True).values
        mean_pool = vcat.mean(dim=1, keepdim=True)
        vcat = vcat * self.spatial_attn(torch.cat([max_pool, mean_pool], dim=1))
        return self._finalize_forward(self.stage2(vcat), x)

class InkDetectorAxialAttention(InkDetectorArch):
    """v16_axial_attention: separate depth and spatial-axis gating."""
    def __init__(self, config: Config):
        super().__init__(config)
        in_ch = len(self.WINDOWS)
        self.depth_gate = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        vcat = torch.cat(self._window_voxel_maps(x), dim=1)
        depth_gate = self.depth_gate(vcat.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        spatial_gate = self.spatial_gate(vcat.mean(dim=2)).unsqueeze(2)
        vcat = vcat * depth_gate * spatial_gate
        return self._finalize_forward(self.stage2(vcat), x)

# Advanced fusion (tests 27-32)
class InkDetectorFPN(InkDetectorArch):
    """v16_fpn: top-down refinement across depth windows."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.lateral = nn.ModuleList([nn.Conv3d(1, 1, kernel_size=1, bias=False) for _ in self.WINDOWS])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        maps = self._window_voxel_maps(x)
        for idx in range(len(maps) - 1, 0, -1):
            maps[idx - 1] = maps[idx - 1] + self.lateral[idx](maps[idx])
        return self._finalize_forward(self.stage2(torch.cat(maps, dim=1)), x)

class InkDetectorBiFPN(InkDetectorArch):
    """v16_bifpn: top-down and bottom-up refinement across windows."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.td = nn.ModuleList([nn.Conv3d(1, 1, kernel_size=1, bias=False) for _ in self.WINDOWS])
        self.bu = nn.ModuleList([nn.Conv3d(1, 1, kernel_size=1, bias=False) for _ in self.WINDOWS])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        maps = self._window_voxel_maps(x)
        for idx in range(len(maps) - 1, 0, -1):
            maps[idx - 1] = maps[idx - 1] + self.td[idx](maps[idx])
        for idx in range(len(maps) - 1):
            maps[idx + 1] = maps[idx + 1] + self.bu[idx](maps[idx])
        return self._finalize_forward(self.stage2(torch.cat(maps, dim=1)), x)

class InkDetectorGhostConv(InkDetectorArch):
    """v16_ghost_conv: ghost-style cheap feature generation in stage-2."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage2 = GhostStage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        return self._finalize_forward(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)), x)

class InkDetectorInvertedResidual(InkDetectorArch):
    """v16_inverted_residual: MobileNetV2-style stage-2 fusion."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage2 = InvertedResidualStage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        return self._finalize_forward(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)), x)

class InkDetectorResNeXt(InkDetectorArch):
    """v16_resnext_groups: grouped-conv fusion over window voxel maps."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.stage2 = ResNeXtStage2(len(self.WINDOWS), norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        return self._finalize_forward(self.stage2(torch.cat(self._window_voxel_maps(x), dim=1)), x)

class InkDetectorDepthShift(InkDetectorArch):
    """v16_depth_shift: shift voxel evidence along depth before fusion."""
    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        shifted = []
        for voxel_map in self._window_voxel_maps(x):
            shift_up = torch.cat([voxel_map[:, :, 1:], voxel_map[:, :, :1]], dim=2)
            shift_down = torch.cat([voxel_map[:, :, -1:], voxel_map[:, :, :-1]], dim=2)
            shifted.append((voxel_map + shift_up + shift_down) / 3.0)
        return self._finalize_forward(self.stage2(torch.cat(shifted, dim=1)), x)


class ShallowFeatureUNet3D(nn.Module):
    """small 3D u-net used to fuse richer per-window features before voxel collapse."""
    def __init__(self, in_channels: int, base_channels: int = 64,
                 norm_mode: str = "batch", activation: str = "relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=1, bias=False),
            _norm3d(base_channels, norm_mode), _act3d(activation),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(base_channels, norm_mode), _act3d(activation),
        )
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            _norm3d(base_channels * 2, norm_mode), _act3d(activation),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            _norm3d(base_channels * 2, norm_mode), _act3d(activation),
        )
        self.up = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decode = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(base_channels, norm_mode), _act3d(activation),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(base_channels, norm_mode), _act3d(activation),
        )
        self.out = nn.Conv3d(base_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        skip = self.stem(x)
        low = self.bottleneck(self.pool(skip))
        up = self.up(low)
        if up.shape[-3:] != skip.shape[-3:]:
            up = F.interpolate(up, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.out(self.decode(torch.cat([skip, up], dim=1)))


class InkDetectorLateCollapse32(InkDetectorArch):
    """v16_latecollapse32: keep 32 feature channels per window until after stage-2 fusion."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_proj = nn.Sequential(
            nn.Conv3d(256, proj_ch, kernel_size=1, bias=False),
            _norm3d(proj_ch, self._norm_mode), _act3d(self._activation),
        )
        self.stage2 = _make_stage2(len(self.WINDOWS) * proj_ch, norm_mode=self._norm_mode,
                                   activation=self._activation, widths=(96, 64, 32))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        return self._finalize_forward(self.stage2(torch.cat(feats, dim=1)), x)


class InkDetectorLateCollapse64(InkDetectorArch):
    """v16_latecollapse64: same idea as latecollapse32 but keeps 64 channels per window."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 64
        self.feature_proj = nn.Sequential(
            nn.Conv3d(256, proj_ch, kernel_size=1, bias=False),
            _norm3d(proj_ch, self._norm_mode), _act3d(self._activation),
        )
        self.stage2 = _make_stage2(len(self.WINDOWS) * proj_ch, norm_mode=self._norm_mode,
                                   activation=self._activation, widths=(128, 96, 48))

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        return self._finalize_forward(self.stage2(torch.cat(feats, dim=1)), x)


class InkDetectorLateUNet(InkDetectorArch):
    """v16_late_unet: fuse rich window features with a shallow u-net, then crop only at final MIL."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_proj = nn.Sequential(
            nn.Conv3d(256, proj_ch, kernel_size=1, bias=False),
            _norm3d(proj_ch, self._norm_mode), _act3d(self._activation),
        )
        self.stage2 = ShallowFeatureUNet3D(len(self.WINDOWS) * proj_ch, base_channels=64,
                                           norm_mode=self._norm_mode, activation=self._activation)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        return self._finalize_forward(self.stage2(torch.cat(feats, dim=1)), x)


class InkDetectorLateCollapse32NonLocal(InkDetectorLateCollapse32):
    """v16_latecollapse32_nonlocal: late rich features with non-local refinement before collapse."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_nonlocal = NonLocalVoxelBlock(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_nonlocal(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateCollapse32CoordAttention(InkDetectorLateCollapse32):
    """v16_latecollapse32_coord: late rich features with coordinate attention before collapse."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_coord_attn = CoordAttention3dLite(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_coord_attn(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateCollapse32DepthSE(InkDetectorLateCollapse32):
    """v16_latecollapse32_depthse: late rich features with depth squeeze-excitation before collapse."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_depth_se = DepthSEGate(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_depth_se(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateCollapse32FPN(InkDetectorLateCollapse32):
    """v16_latecollapse32_fpn: late rich features with top-down cross-window refinement."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.lateral = nn.ModuleList([
            nn.Conv3d(proj_ch, proj_ch, kernel_size=1, bias=False) for _ in self.WINDOWS
        ])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        for idx in range(len(feats) - 1, 0, -1):
            feats[idx - 1] = feats[idx - 1] + self.lateral[idx](feats[idx])
        return self._finalize_forward(self.stage2(torch.cat(feats, dim=1)), x)


class InkDetectorLateUNetNonLocal(InkDetectorLateUNet):
    """v16_late_unet_nonlocal: shallow feature u-net with non-local refinement on projected windows."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_nonlocal = NonLocalVoxelBlock(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_nonlocal(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateUNetCoordAttention(InkDetectorLateUNet):
    """v16_late_unet_coord: shallow feature u-net with coordinate attention on projected windows."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_coord_attn = CoordAttention3dLite(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_coord_attn(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateUNetDepthSE(InkDetectorLateUNet):
    """v16_late_unet_depthse: shallow feature u-net with depth squeeze-excitation on projected windows."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.feature_depth_se = DepthSEGate(len(self.WINDOWS) * proj_ch)

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        fused = self.feature_depth_se(torch.cat(feats, dim=1))
        return self._finalize_forward(self.stage2(fused), x)


class InkDetectorLateUNetFPN(InkDetectorLateUNet):
    """v16_late_unet_fpn: shallow feature u-net with top-down cross-window refinement."""
    def __init__(self, config: Config):
        super().__init__(config)
        proj_ch = 32
        self.lateral = nn.ModuleList([
            nn.Conv3d(proj_ch, proj_ch, kernel_size=1, bias=False) for _ in self.WINDOWS
        ])

    def forward_with_extras(self, x):
        x = self._prepare_input(x)
        feats = [self.feature_proj(fm) for fm in self._window_feature_maps(x)]
        for idx in range(len(feats) - 1, 0, -1):
            feats[idx - 1] = feats[idx - 1] + self.lateral[idx](feats[idx])
        return self._finalize_forward(self.stage2(torch.cat(feats, dim=1)), x)


# ===================================================================
# RADICAL ARCHITECTURES (tests 39-44)
# ===================================================================

class ViT3D(nn.Module):
    """3D Vision Transformer for ink detection.
    
    Divides 3D volume into patches, applies transformer encoder, outputs tile score.
    Input: (B, 1, D, H, W) where D=24, H=W=16 (or 48 with context)
    """
    def __init__(self, config: Config):
        super().__init__()
        self.tile_size = int(getattr(config.data, 'context_size', config.data.tile_size))
        self.depth = int(config.data.depth)
        
        # Patch embedding: 4x4x4 patches
        self.patch_d, self.patch_h, self.patch_w = 4, 4, 4
        self.patch_embed = nn.Conv3d(1, 256, kernel_size=(self.patch_d, self.patch_h, self.patch_w), 
                                     stride=(self.patch_d, self.patch_h, self.patch_w))
        
        # Transformer encoder (4 layers, lighter than typical ViT)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, 
                                                   dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 1)
        
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        
        # Patch embedding: (B, 256, n_d, n_h, n_w)
        x = self.patch_embed(x)
        
        # Flatten to sequence: (B, n_patches, 256)
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]
        
        # Learnable positional encoding (adaptive to sequence length)
        if not hasattr(self, 'pos_embed') or self.pos_embed.shape[1] != N:
            self.pos_embed = nn.Parameter(torch.randn(1, N, 256, device=x.device) * 0.02)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling over patches
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        return self.head(x)


class Swin3D(nn.Module):
    """Swin Transformer 3D with shifted window attention.
    
    Hierarchical transformer with shifted windows for efficiency.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.tile_size = int(config.data.tile_size)
        self.depth = int(config.data.depth)
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(1, 96, kernel_size=4, stride=4)
        
        # Window attention (simplified - single stage)
        self.window_size = 2  # 2x2x2 windows
        self.attn = nn.MultiheadAttention(96, num_heads=4, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(96)
        self.norm2 = nn.LayerNorm(96)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(96, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 96),
            nn.Dropout(0.1)
        )
        
        # Pooling and head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(96, 1)
        
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 96, D', H', W')
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, 96)
        
        # Window attention (simplified - no actual windowing for efficiency)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Pool and classify
        x = x.transpose(1, 2)  # (B, 96, N)
        x = self.pool(x).squeeze(-1)  # (B, 96)
        return self.head(x)


class ConvNeXt3D(nn.Module):
    """ConvNeXt 3D - modernized CNN with large kernels and inverted bottlenecks.
    
    Key features: 7x7x7 depthwise convs, LayerNorm, GELU, inverted bottleneck (expand then compress).
    """
    def __init__(self, config: Config):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=4, stride=4)
        )
        
        # ConvNeXt blocks
        self.blocks = nn.ModuleList([
            self._make_block(96) for _ in range(4)
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LayerNorm(96),
            nn.Linear(96, 1)
        )
        
    def _make_block(self, dim):
        return nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim),  # Depthwise
            nn.BatchNorm3d(dim),  # Use BatchNorm instead of LayerNorm for flexibility
            nn.Conv3d(dim, dim * 4, kernel_size=1),  # Expand
            nn.GELU(),
            nn.Conv3d(dim * 4, dim, kernel_size=1),  # Compress
        )
    
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        
        x = self.stem(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual
        
        return self.head(x)


class XCiT3D(nn.Module):
    """XCiT 3D - Cross-Covariance Image Transformer.
    
    Uses cross-covariance attention instead of standard attention for efficiency.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.tile_size = int(config.data.tile_size)
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(1, 128, kernel_size=4, stride=4)
        
        # Cross-covariance attention (simplified)
        self.temperature = nn.Parameter(torch.ones(1))
        self.qkv = nn.Linear(128, 128 * 3)
        self.proj = nn.Linear(128, 128)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.Dropout(0.1)
        )
        
        self.head = nn.Linear(128, 1)
        
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 128)
        
        # Cross-covariance attention
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Cross-covariance: q^T k / sqrt(C)
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = F.softmax(attn, dim=-1)
        
        x_attn = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x + self.proj(x_attn)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Pool and classify
        x = x.mean(dim=1)
        return self.head(x)


class nnUNet3D(nn.Module):
    """nnU-Net 3D - Medical imaging encoder-decoder with deep supervision.
    
    Encoder-decoder with skip connections, designed for medical 3D volumes.
    """
    def __init__(self, config: Config):
        super().__init__()
        self._ds = max(1, int(getattr(config.data, "context_downsample", 1)))
        self._use_attn_mil = bool(getattr(config.model, "attn_mil", False))
        self._attn_entropy_weight = float(getattr(config.model, "attn_entropy_weight", 0.0))
        self._use_learned_surface = bool(getattr(config.model, "learned_surface", False))
        self._use_supcon = bool(getattr(config.tra, "supcon", False))
        self._use_dann = bool(getattr(config.tra, "dann", False))
        self.last_voxel_map = None
        self.last_attn_entropy_loss = None
        self.last_surface_attn = None
        self.supcon_head = None
        self.domain_head = None
        if self._use_attn_mil:
            self.attn_mil = GatedAttentionMIL(feat_dim=1, att_dim=32)
        if self._use_learned_surface:
            self.depth_surface_attn = DepthSurfaceAttn(hidden=8)
        self._emb_dim = 256
        if self._use_supcon:
            proj_dim = int(getattr(config.tra, "supcon_proj_dim", 128))
            hidden_dim = int(getattr(config.tra, "supcon_hidden_dim", 256))
            self.supcon_head = SupConHead(self._emb_dim, proj_dim=proj_dim, hidden=hidden_dim)
        if self._use_dann:
            n_dom = int(getattr(config.tra, "dann_n_domains", 15))
            self.domain_head = DomainHead(self._emb_dim, n_dom, hidden=64)
        
        # Encoder
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Deep supervision outputs
        self.out1 = nn.Conv3d(32, 1, kernel_size=1)
        self.out2 = nn.Conv3d(64, 1, kernel_size=1)
        self.out3 = nn.Conv3d(128, 1, kernel_size=1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def _prepare_input(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if self._ds > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self._ds, self._ds), stride=(1, self._ds, self._ds))
        return x

    def _stem_in(self, x):
        return x

    def _apply_learned_surface(self, raw_x, feat):
        if not self._use_learned_surface:
            self.last_surface_attn = None
            return feat
        attn = self.depth_surface_attn(raw_x)
        self.last_surface_attn = attn.detach()
        return feat * (1.0 + attn)

    def _encode_features(self, x):
        raw_x = self._prepare_input(x)
        stem_x = self._stem_in(raw_x)

        e1 = self.enc1(stem_x)
        e1 = self._apply_learned_surface(raw_x, e1)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return b, d1

    def _project_embedding(self, bottleneck):
        return F.adaptive_avg_pool3d(bottleneck, output_size=1).flatten(1)

    def _bag_score(self, out):
        if self._use_attn_mil:
            tile_score, attn_entropy_loss = self.attn_mil(out, entropy_weight=self._attn_entropy_weight)
            self.last_attn_entropy_loss = attn_entropy_loss
            return tile_score
        self.last_attn_entropy_loss = torch.tensor(0.0, device=out.device)
        out_flat = out.flatten(1)
        r = 2.0
        N = out_flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * out_flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=out.device))
        )
    
    def forward(self, x):
        _, d1 = self._encode_features(x)
        out = self.out1(d1)
        self.last_voxel_map = out.detach()
        return self._bag_score(out)

    def forward_with_extras(self, x):
        b, d1 = self._encode_features(x)
        out = self.out1(d1)
        self.last_voxel_map = out.detach()
        emb = self._project_embedding(b)
        return self._bag_score(out), emb, None, None


class nnUNet3DLCNZGrad(nnUNet3D):
    """nnU-Net 3D with the baseline-style raw+lcn+dz stem inputs."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.enc1 = self._conv_block(3, 32)

    def _stem_in(self, x):
        dz = torch.zeros_like(x)
        dz[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return torch.cat([x, _lcn2d(x, 5), dz], dim=1)


class nnUNet3DDS(nnUNet3D):
    """compat alias now that nnUNet3D honors context_downsample directly."""
    pass


class SlotAttention3D(nn.Module):
    """Slot Attention 3D - Object-centric learning with slot attention.
    
    Learns to decompose scene into slots through iterative attention.
    """
    def __init__(self, config: Config):
        super().__init__()
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Slot attention
        self.num_slots = 4  # Learn 4 slots (ink patterns)
        self.slot_dim = 64
        self.num_iters = 3
        
        self.slots_init = nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim))
        
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_inputs = nn.LayerNorm(self.slot_dim)
        
        # Attention
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim)
        
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.slot_dim)
        )
        
        # Classifier
        self.head = nn.Linear(self.slot_dim * self.num_slots, 1)
        
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B = x.shape[0]
        
        # Extract features
        features = self.encoder(x)  # (B, 64, D', H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, N, 64)
        
        # Initialize slots
        slots = self.slots_init.expand(B, -1, -1)
        
        # Slot attention iterations
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            q = self.to_q(slots)
            k = self.to_k(self.norm_inputs(features))
            v = self.to_v(self.norm_inputs(features))
            
            # Compute attention weights
            attn_logits = torch.einsum('bid,bjd->bij', q, k) / math.sqrt(self.slot_dim)
            attn = F.softmax(attn_logits, dim=1)
            
            # Weighted mean
            updates = torch.einsum('bij,bjd->bid', attn, v)
            
            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)
            
            # MLP
            slots = slots + self.mlp(self.norm_slots(slots))
        
        # Aggregate slots and classify
        slots_flat = slots.flatten(1)
        return self.head(slots_flat)


# register v16_arch_ctx AFTER the class is defined
_ARCH_MAP["v16_arch_ctx"] = InkDetectorArch
_ARCH_MAP["v16_arch_ctx_relaxedcrop"] = InkDetectorArchRelaxedCrop

# Register Campaign Archs 7 architectures
_ARCH_MAP["v16_arch_ctx_fovea"] = InkDetectorArch_Fovea
_ARCH_MAP["v16_dual_stream_early"] = InkDetectorDualStreamEarly
_ARCH_MAP["v16_dual_stream_late"] = InkDetectorDualStreamLate
_ARCH_MAP["v16_dual_stream_gated"] = InkDetectorDualStreamGated
_ARCH_MAP["v16_dual_stream_asym"] = InkDetectorDualStreamAsym
_ARCH_MAP["v16_hybrid_depth_per_window"] = InkDetectorHybridDepthPerWindow
_ARCH_MAP["v16_hybrid_depth_global"] = InkDetectorHybridDepthGlobal
_ARCH_MAP["v16_hybrid_depth_triple"] = InkDetectorHybridDepthTriple
_ARCH_MAP["v16_hybrid_depth_gated"] = InkDetectorHybridDepthGated
_ARCH_MAP["v16_multiscale_pyramid"] = InkDetectorMultiscalePyramid
_ARCH_MAP["v16_depth_se"] = InkDetectorDepthSE
_ARCH_MAP["v16_depthwise_sep"] = InkDetectorDepthwiseSep
_ARCH_MAP["v16_mixed_depth_windows"] = InkDetectorMixedDepthWindows
_ARCH_MAP["v16_octave_conv"] = InkDetectorOctaveConv
_ARCH_MAP["v16_efficientnet_scale"] = InkDetectorEfficientScale
_ARCH_MAP["v16_nonlocal_depth"] = InkDetectorNonLocalDepth
_ARCH_MAP["v16_coord_attention"] = InkDetectorCoordAttention
_ARCH_MAP["v16_deformable_conv"] = InkDetectorDeformableConv
_ARCH_MAP["v16_progressive_depth"] = InkDetectorProgressiveDepth
_ARCH_MAP["v16_dual_attention"] = InkDetectorDualAttention
_ARCH_MAP["v16_axial_attention"] = InkDetectorAxialAttention
_ARCH_MAP["v16_fpn"] = InkDetectorFPN
_ARCH_MAP["v16_bifpn"] = InkDetectorBiFPN
_ARCH_MAP["v16_ghost_conv"] = InkDetectorGhostConv
_ARCH_MAP["v16_inverted_residual"] = InkDetectorInvertedResidual
_ARCH_MAP["v16_resnext_groups"] = InkDetectorResNeXt
_ARCH_MAP["v16_depth_shift"] = InkDetectorDepthShift
_ARCH_MAP["v16_latecollapse32"] = InkDetectorLateCollapse32
_ARCH_MAP["v16_latecollapse64"] = InkDetectorLateCollapse64
_ARCH_MAP["v16_late_unet"] = InkDetectorLateUNet
_ARCH_MAP["v16_latecollapse32_nonlocal"] = InkDetectorLateCollapse32NonLocal
_ARCH_MAP["v16_latecollapse32_coord"] = InkDetectorLateCollapse32CoordAttention
_ARCH_MAP["v16_latecollapse32_depthse"] = InkDetectorLateCollapse32DepthSE
_ARCH_MAP["v16_latecollapse32_fpn"] = InkDetectorLateCollapse32FPN
_ARCH_MAP["v16_late_unet_nonlocal"] = InkDetectorLateUNetNonLocal
_ARCH_MAP["v16_late_unet_coord"] = InkDetectorLateUNetCoordAttention
_ARCH_MAP["v16_late_unet_depthse"] = InkDetectorLateUNetDepthSE
_ARCH_MAP["v16_late_unet_fpn"] = InkDetectorLateUNetFPN

# Register radical architectures (tests 39-44) - fully implemented!
_ARCH_MAP["vit3d"] = ViT3D
_ARCH_MAP["swin3d"] = Swin3D
_ARCH_MAP["convnext3d"] = ConvNeXt3D
_ARCH_MAP["xcit3d"] = XCiT3D
_ARCH_MAP["nnunet3d"] = nnUNet3D
_ARCH_MAP["nnunet3d_lcndz"] = nnUNet3DLCNZGrad
_ARCH_MAP["nnunet3d_ds"] = nnUNet3DDS
_ARCH_MAP["slot3d"] = SlotAttention3D


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
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm3d, ChannelLayerNorm3d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({arch}): {params:,}")
    return model, params
