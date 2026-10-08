import torch
import torch.nn as nn
from .config import Config


class GeMPool3d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        return nn.functional.adaptive_avg_pool3d(x.pow(self.p), 1).pow(1.0 / self.p)

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super(CBAM3D, self).__init__()
        self.channel_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.spatial_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Permutation-Invariant Channel Attention (shared MLP across spatial)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention (unchanged)
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, D, H, W)

        # --- Permutation-Invariant Channel Attention ---
        b, c, d, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x_flat = x_perm.view(-1, c)                     # (B*D*H*W, C)
        attn = self.sigmoid_channel(self.channel_mlp(x_flat))  # (B*D*H*W, C)
        attn = attn.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        scale = (1 + self.channel_scale * (attn - 1)).float()
        x = x * scale

        # --- Spatial Attention (same as before) ---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))

        scale = (1 + self.spatial_scale * (spatial_attn - 1)).float()
        x = x * scale

        return x

class InkDetector(nn.Module):
    def __init__(self, config: Config):
        super(InkDetector, self).__init__()

        conv3_dilation = max(1, int(getattr(config.model, "conv3_dilation", 1)))

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 4, 4), padding=1, bias=False),  # (B, 32, 8, 31, 31)
            nn.BatchNorm3d(32).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(32),

            nn.Conv3d(32, 128, kernel_size=(3, 3, 3), padding=1, bias=False),  # (B, 96, 8, 31, 31)
            nn.BatchNorm3d(128).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (B, 96, 4, 15, 15)
            nn.Dropout3d(config.model.conv1_drop),

            nn.Conv3d(
                128,
                256,
                kernel_size=(3, 3, 3),
                padding=conv3_dilation,
                dilation=conv3_dilation,
                bias=False,
            ),  # (B, 128, 4, 15, 15)
            nn.BatchNorm3d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (B, 128, 2, 7, 7)
            nn.Dropout3d(config.model.conv2_drop),
        )

        pool_mode = str(getattr(config.model, "pooling", "avg")).lower()
        if pool_mode == "avg":
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pool_mode == "max":
            self.pool = nn.AdaptiveMaxPool3d(1)
        elif pool_mode == "gem":
            self.pool = GeMPool3d(p=float(getattr(config.model, "gem_p", 3.0)))
        else:
            raise ValueError(f"unsupported pooling mode: {pool_mode}")


        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512).to(dtype=torch.float32),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc2_drop),

            nn.Linear(32, 1)  # Output: (B, 1)
        )
        self.activations = {}
        # hooks are NOT registered at startup; call _register_hooks() explicitly
        # if activation logging is needed, as storing layer outputs on every forward
        # pass wastes significant GPU memory (100s of MB at batch_size=96)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def _register_hooks(self):
        def hook(module, input, output):
            self.activations[module] = output.detach()

        for layer in self.features:
            if not isinstance(layer, (nn.Dropout3d, nn.BatchNorm3d)):
                layer.register_forward_hook(hook)
        for layer in self.classifier:
            if not isinstance(layer, (nn.Dropout, nn.BatchNorm1d)):
                layer.register_forward_hook(hook)
        
        

# ──────────────────────────────────────────────────────────────────────────────
# shared helpers for v2 architecture variants
# ──────────────────────────────────────────────────────────────────────────────

def _pool_layer(config):
    """factory for global 3D pooling based on config.model.pooling"""
    mode = str(getattr(config.model, "pooling", "avg")).lower()
    if mode == "avg":
        return nn.AdaptiveAvgPool3d(1)
    if mode == "max":
        return nn.AdaptiveMaxPool3d(1)
    if mode == "gem":
        return GeMPool3d(p=float(getattr(config.model, "gem_p", 3.0)))
    raise ValueError(f"unknown pooling mode: {mode}")


def _slim_head(in_dim, drop=0.2):
    """2-layer MLP head: in_dim → 64 → 1
    hypothesis: the 5-layer head in v1 memorizes rather than generalizes"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 64, bias=False),
        nn.BatchNorm1d(64).to(dtype=torch.float32),
        nn.ReLU(inplace=True),
        nn.Dropout(drop),
        nn.Linear(64, 1),
    )


class SE3D(nn.Module):
    """squeeze-excitation for 3D: global avg pool → 2-layer MLP → per-channel scale
    lighter than CBAM3D (no spatial attention, no per-position MLP)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x)).view(b, c, 1, 1, 1)


class ECA3D(nn.Module):
    """efficient channel attention: 1D conv over the channel axis, zero FC overhead
    avoids the full MLP of SE while still doing cross-channel recalibration"""
    def __init__(self, channels, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, 1, c)
        return x * self.sigmoid(self.conv(y)).view(b, c, 1, 1, 1)


class ResBlock3D(nn.Module):
    """post-activation residual: conv → BN → ReLU → conv → BN, then add skip
    skip connection lets model bypass transformation if identity is better"""
    def __init__(self, channels, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(drop),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels).to(dtype=torch.float32),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class PreActResBlock3D(nn.Module):
    """pre-activation residual: BN → ReLU → conv → BN → ReLU → conv, add skip
    ResNet-v2 style; pre-activation allows cleaner gradient flow through the skip path"""
    def __init__(self, channels, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(channels).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.Dropout3d(drop),
            nn.BatchNorm3d(channels).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x) + x


class BottleneckBlock3D(nn.Module):
    """bottleneck residual: 1×1 reduce → 3×3 → 1×1 expand + skip (ResNet-50 style)
    reduces parameter count for deep networks while maintaining capacity"""
    def __init__(self, channels, drop=0.0):
        super().__init__()
        mid = max(1, channels // 4)
        self.block = nn.Sequential(
            nn.Conv3d(channels, mid, 1, bias=False),
            nn.BatchNorm3d(mid).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(drop),
            nn.Conv3d(mid, channels, 1, bias=False),
            nn.BatchNorm3d(channels).to(dtype=torch.float32),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


# ──────────────────────────────────────────────────────────────────────────────
# v2 architecture variants  (input: B × 1 × D × 32 × 32, output: B × 1)
# all use (3,3,3) padding=1 convs so spatial dims stay clean powers of 2
# ──────────────────────────────────────────────────────────────────────────────

class InkDetectorSlimHead(nn.Module):
    """v2_slim_head: same 3-block CBAM backbone, 2-layer head instead of 5
    tests if the deep head memorizes training patterns rather than generalizing"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorNoCBAM(nn.Module):
    """v2_no_cbam: same backbone, all CBAM blocks removed
    tests whether attention actually helps on small 32×32 tiles or just adds noise"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorSEOnly(nn.Module):
    """v2_se_only: SE blocks (channel-only attention) instead of CBAM
    removes spatial attention overhead; channel recalibration may suffice at tile scale"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), SE3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), SE3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), SE3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorECA(nn.Module):
    """v2_eca: efficient channel attention (1D conv over channels, no FC)
    even cheaper than SE; tests whether minimal channel recalibration is sufficient"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), ECA3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), ECA3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), ECA3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorResidual(nn.Module):
    """v2_residual: CBAM backbone + ResBlock3D after each conv stage
    residual connections allow identity bypass when transformation is harmful"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            ResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            ResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            ResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorResidualNoCBAM(nn.Module):
    """v2_residual_no_cbam: residual blocks only, no attention modules
    isolates the residual connection benefit from attention confounders"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorWiderShallow(nn.Module):
    """v2_wider_shallow: 2 conv blocks, wider (1→64→256), only 1 MaxPool
    fewer abstraction levels; 32×32 input may not need deep compression"""
    def __init__(self, config):
        super().__init__()
        d1 = config.model.conv1_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(64),
            nn.Conv3d(64, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorSlimAll(nn.Module):
    """v2_slim_all: narrow backbone (1→16→64→128) + 2-layer head
    tests the overparameterization hypothesis for binary tile classification"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorFactorized(nn.Module):
    """v2_factorized_depth: each conv block is (3,1,1) depth-conv then (1,3,3) spatial-conv
    explicitly models depth and spatial axes independently; matches the depth-ordering insight
    from campaign 1: channel structure fidelity matters more than pooling"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop

        def _fact(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, (3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.BatchNorm3d(out_ch).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, (1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(out_ch).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        self.block1 = _fact(1, 32)
        self.block2 = _fact(32, 128)
        self.pool1 = nn.Sequential(nn.MaxPool3d(2), nn.Dropout3d(d1))
        self.block3 = _fact(128, 256)
        self.pool2 = nn.Sequential(nn.MaxPool3d(2), nn.Dropout3d(d2))
        self.global_pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(self.block2(x))
        x = self.pool2(self.block3(x))
        return self.classifier(self.global_pool(x))


class InkDetectorAsymFirst(nn.Module):
    """v2_asymmetric_first: first conv is (1,3,3) - learns spatial before mixing depth
    the original (3,4,4) first kernel mixes depth from layer 1; this delays that coupling
    allowing spatial patterns to form independently before depth integration begins"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            # spatial-only first conv: each depth slice processed independently
            nn.Conv3d(1, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            # full 3D mixing from here onward
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorStridedConv(nn.Module):
    """v2_strided_conv: strided Conv3d replaces MaxPool3d for downsampling
    max pool always keeps the maximum activation; strided conv learns what to preserve
    may better retain weak ink signals that maxpool would discard"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            # strided conv instead of maxpool: (B,128,D,H,W) → (B,128,D//2,H//2,W//2)
            nn.Conv3d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.Conv3d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDualPool(nn.Module):
    """v2_dual_pool: concat global avg + global max pool before head (512-dim input)
    avg captures mean activation level; max captures peak ink evidence;
    both carry complementary information for binary tile classification"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # 256 avg + 256 max = 512 input to head
        self.classifier = _slim_head(512, config.model.fc1_drop)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1))


class InkDetectorGroupNorm(nn.Module):
    """v2_group_norm: GroupNorm(8, ch) throughout instead of BatchNorm3d
    GroupNorm statistics are batch-size independent; may be more stable across
    varied batch compositions (mixed ink/no-ink ratios during training)"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        # 32, 128, 256 are all divisible by 8 groups
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        # use LayerNorm in head to stay consistent with GN philosophy
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDepthProject(nn.Module):
    """v2_depth_project: reshape (B,1,D,H,W)→(B,D,H,W), process with 2D CNN
    treats 8 depth slices as independent input channels (analogous to RGB)
    removes depth-spatial entanglement; 2D conv learns to combine depth freely"""
    def __init__(self, config):
        super().__init__()
        depth = config.data.depth
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv2d(depth, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d1),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        # (B, 1, D, H, W) → (B, D, H, W)
        x = x.squeeze(1)
        return self.classifier(self.pool(self.features(x)))


class InkDetectorTwoStream(nn.Module):
    """v2_two_stream: parallel depth-stream (1D conv) + spatial-stream (2D conv), merged
    depth stream: how does voxel absorption change along Z? (depth profile)
    spatial stream: what does the average spatial texture look like? (ink shape)
    merged at the end so each stream can specialize independently"""
    def __init__(self, config):
        super().__init__()
        drop = config.model.conv1_drop

        # depth stream: spatial avg → 1D conv along depth axis
        self.depth_stream = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # spatial stream: depth avg → 2D CNN
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(drop),
        )

        # merge: 64 (depth) + 64 (spatial) = 128
        self.classifier = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        # depth stream: average over H and W → (B, 1, D) → 1D conv → (B, 64)
        d_feat = x.mean(dim=[3, 4])                        # (B, 1, D)
        d_feat = self.depth_stream(d_feat).squeeze(-1)      # (B, 64)

        # spatial stream: average over D → (B, 1, H, W) → 2D conv → (B, 64)
        s_feat = x.mean(dim=2)                              # (B, 1, H, W)
        s_feat = self.spatial_stream(s_feat).flatten(1)     # (B, 64)

        return self.classifier(torch.cat([d_feat, s_feat], dim=1))


class InkDetectorInceptionFirst(nn.Module):
    """v2_inception_first: parallel (1,3,3) + (3,1,1) + (1,1,1) branches at entry
    explicit multi-scale: captures spatial texture, depth profiles, and pointwise mixing
    branches concatenated (48ch) then fed into standard backbone"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        dil = max(1, getattr(config.model, "conv3_dilation", 1))

        # 3 parallel entry branches (16 channels each → 48 total)
        self.branch_spatial = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.branch_depth = nn.Sequential(
            nn.Conv3d(1, 16, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.branch_point = nn.Sequential(
            nn.Conv3d(1, 16, 1, bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )

        # main backbone takes 48 channels from concatenated branches
        self.backbone = nn.Sequential(
            nn.Conv3d(48, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        x = torch.cat([self.branch_spatial(x), self.branch_depth(x), self.branch_point(x)], dim=1)
        return self.classifier(self.pool(self.backbone(x)))


class InkDetectorDeeper(nn.Module):
    """v2_deeper: 4-block backbone (32→128→256→384), 3 MaxPool stages
    depth after 3 pools: 8→4→2→1; spatial: 32→16→8→4, then adaptive pool
    tests whether 3 abstraction levels is insufficient for this task"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True), CBAM3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.Conv3d(256, 384, 3, padding=1, bias=False),
            nn.BatchNorm3d(384).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(384, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorBottleneck(nn.Module):
    """v2_bottleneck: bottleneck residual blocks (1×1 reduce → 3×3 → 1×1 expand + skip)
    ResNet-50 style; keeps channel capacity while reducing 3×3 conv cost"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            BottleneckBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            BottleneckBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            BottleneckBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActRes(nn.Module):
    """v2_preact_res: pre-activation residual blocks (BN→ReLU→conv + skip)
    ResNet-v2 style: the skip path carries unmodified signal all the way through,
    enabling cleaner gradient flow and better generalization in deeper networks"""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            # final BN+ReLU to activate the last pre-act block's output
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorNoNorm(nn.Module):
    """v2_no_norm_drop: no BatchNorm anywhere, heavier dropout instead
    BatchNorm creates statistical coupling between samples in a batch;
    this tests if BN introduces dependencies that hurt cross-scroll generalization"""
    def __init__(self, config):
        super().__init__()
        d1 = max(0.15, config.model.conv1_drop)
        d2 = max(0.15, config.model.conv2_drop)
        dil = max(1, getattr(config.model, "conv3_dilation", 1))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True), nn.Dropout3d(0.1),
            nn.Conv3d(32, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        # no BN in head either
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ──────────────────────────────────────────────────────────────────────────────
# v3 architectures — campaign 3
# design principles: build on preact_res + residual_no_cbam as proven base;
# explore depth-axis specialization, multi-scale pooling, and attention pooling
# to improve sensitivity to faint/subtle ink signals in hard regions
#
# NEW for revised campaign 3:
#   v3_linear_head      — maximum simplification: pool + single linear layer
#   v3_depth_project_deep — deeper 2D CNN (builds on t18 which was 2nd visually)
# ──────────────────────────────────────────────────────────────────────────────

class InkDetectorLinearHead(nn.Module):
    """v3_linear_head: preact backbone + single linear layer (pool → Linear(256,1)).
    most aggressive simplification of the head possible while keeping the backbone.
    if t01_slim_head (2-layer) was visually better than 5-layer, does 1-layer go further?
    coarser head = fewer degrees of freedom = spatially smoother, more coherent outputs."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        # single linear: no intermediate projections, no non-linearity
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDepthProjectDeep(nn.Module):
    """v3_depth_project_deep: deeper 2D CNN treating depth as channels.
    t18_depth_project (64→256→512, 2 maxpool) was 2nd best visually in campaign 2.
    this version adds a 3rd conv block for more spatial abstraction depth.
    depth-as-channels = decoupled depth selection + rich spatial processing."""
    def __init__(self, config):
        super().__init__()
        depth = config.data.depth
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv2d(depth, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d1),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d2),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _slim_head(512, config.model.fc1_drop)

    def forward(self, x):
        x = x.squeeze(1)    # (B, 1, D, H, W) → (B, D, H, W)
        return self.classifier(self.pool(self.features(x)))

class DepthAttention1D(nn.Module):
    """1D attention over the depth axis: learn which depth slices matter most.
    applied after spatial global-avg-pool, before 3D pool — selects the ink-bearing depth."""
    def __init__(self, depth):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(depth, depth, bias=False),
            nn.Tanh(),
            nn.Linear(depth, depth, bias=False),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # x: (B, C, D, H, W) → spatial avg → (B, C, D) → per-depth weights
        b, c, d, h, w = x.shape
        spatial_avg = x.mean(dim=[3, 4])             # (B, C, D)
        channel_avg = spatial_avg.mean(dim=1)        # (B, D)
        weights = self.attn(channel_avg)             # (B, D)
        weights = weights.view(b, 1, d, 1, 1)
        return x * weights


class SpatialAttnPool3d(nn.Module):
    """learn a spatial attention map over (H,W), then use it to weight the global pool.
    instead of uniform average, focuses pool on the most ink-relevant spatial locations."""
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # (B, C, D, H, W) → attention weights → weighted sum → (B, C, 1, 1, 1)
        w = self.score(x)              # (B, 1, D, H, W)
        return (x * w).sum(dim=[2, 3, 4], keepdim=True) / (w.sum(dim=[2, 3, 4], keepdim=True) + 1e-8)


class NonLocal3D(nn.Module):
    """non-local means block: each position attends to all others (embedded gaussian).
    captures long-range spatial context — ink tiles near other ink tiles get stronger signal."""
    def __init__(self, channels):
        super().__init__()
        mid = max(1, channels // 2)
        self.theta = nn.Conv3d(channels, mid, 1, bias=False)
        self.phi   = nn.Conv3d(channels, mid, 1, bias=False)
        self.g     = nn.Conv3d(channels, mid, 1, bias=False)
        self.out   = nn.Conv3d(mid, channels, 1, bias=False)
        self.bn    = nn.BatchNorm3d(channels).to(dtype=torch.float32)

    def forward(self, x):
        b, c, d, h, w = x.shape
        n = d * h * w
        theta = self.theta(x).view(b, -1, n).permute(0, 2, 1)   # (B, N, C/2)
        phi   = self.phi(x).view(b, -1, n)                       # (B, C/2, N)
        attn  = torch.softmax(torch.bmm(theta, phi), dim=-1)     # (B, N, N)
        g     = self.g(x).view(b, -1, n).permute(0, 2, 1)       # (B, N, C/2)
        y     = torch.bmm(attn, g).permute(0, 2, 1).view(b, -1, d, h, w)
        return x + self.bn(self.out(y))


class InkDetectorPreActBaseline(nn.Module):
    """v3_preact_baseline: clean re-run of v2_preact_res with all bug fixes applied.
    establishes a fair control for campaign 3 without hook/cuDNN/OOM confounds."""
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv1_drop),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv2_drop),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActDeep(nn.Module):
    """v3_preact_deep: 5 preact residual blocks (32→128→256→256, 3 maxpools).
    hypothesis: t11_deeper had best hard probe; deeper preact may combine both strengths."""
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv1_drop),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv2_drop),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActWide(nn.Module):
    """v3_preact_wide: preact residuals with 1→64→256→512 channels.
    more capacity in each layer; tests if width or depth is the key factor."""
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1, bias=False),
            PreActResBlock3D(64),
            nn.Conv3d(64, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv1_drop),
            nn.Conv3d(256, 512, 3, padding=1, bias=False),
            PreActResBlock3D(512),
            nn.MaxPool3d(2), nn.Dropout3d(config.model.conv2_drop),
            nn.BatchNorm3d(512).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(512, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorResNoCBAMDeep(nn.Module):
    """v3_res_no_cbam_deep: residual_no_cbam (campaign 2 readability winner) made deeper.
    t06 had the best readability; add a 4th block to push further."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.Conv3d(256, 384, 3, padding=1, bias=False),
            nn.BatchNorm3d(384).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(384, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDepthAttn(nn.Module):
    """v3_depth_attn: preact backbone + explicit 1D attention over depth slices.
    ink appears at specific depth windows; learning which depths matter most should
    help generalize from easy (clear-depth) to hard (ambiguous-depth) regions."""
    def __init__(self, config):
        super().__init__()
        depth = config.data.depth
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
        )
        # depth attention before the second maxpool compresses the depth axis
        self.depth_attn = DepthAttention1D(depth // 2)
        self.late = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.depth_attn(x)
        x = self.late(x)
        return self.classifier(self.pool(x))


class InkDetectorSpatialAttnPool(nn.Module):
    """v3_spatial_attn_pool: preact backbone, replace global avg pool with spatial attention pool.
    instead of uniform averaging, learns to weight spatial positions by ink relevance.
    hypothesis: in hard regions the ink is spatially localized; uniform avg dilutes it."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = SpatialAttnPool3d(256)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorNonLocal(nn.Module):
    """v3_nonlocal: preact backbone with non-local means block at mid-level features.
    long-range context: an ink tile surrounded by other ink tiles should score higher;
    standard conv only has local receptive field at 32x32 scale."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.early = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
        )
        self.nonlocal_block = NonLocal3D(128)
        self.late = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        x = self.early(x)
        x = self.nonlocal_block(x)
        x = self.late(x)
        return self.classifier(self.pool(x))


class InkDetectorFPN(nn.Module):
    """v3_fpn: feature pyramid network — merge features from stride-1, stride-2, stride-4.
    coarser scales capture global context; finer scales capture local texture.
    multi-scale fusion may help when ink signal strength varies across spatial frequency."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop

        self.p1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(32),
        )
        self.p2 = nn.Sequential(
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
        )
        self.p3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )

        # lateral 1×1 projections to 64-channel for all levels
        self.lat1 = nn.Conv3d(32, 64, 1, bias=False)
        self.lat2 = nn.Conv3d(128, 64, 1, bias=False)
        self.lat3 = nn.Conv3d(256, 64, 1, bias=False)

        self.pool = nn.AdaptiveAvgPool3d(1)
        # 3 levels × 64 ch = 192
        self.classifier = _slim_head(192, config.model.fc1_drop)

    def forward(self, x):
        f1 = self.p1(x)
        f2 = self.p2(f1)
        f3 = self.p3(f2)
        # pool each level to scalar then concat
        v1 = self.pool(self.lat1(f1)).flatten(1)
        v2 = self.pool(self.lat2(f2)).flatten(1)
        v3 = self.pool(self.lat3(f3)).flatten(1)
        return self.classifier(torch.cat([v1, v2, v3], dim=1))


class InkDetectorMultiScalePool(nn.Module):
    """v3_multiscale_pool: adaptive pool to 1×1, 2×2, 4×4 grids, then concat and classify.
    spatial pyramid pooling captures both global and local response patterns.
    the 4×4 grid preserves some spatial layout info lost by global pooling."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        # spatial dims after 2×maxpool on 32→8; depth 8→2
        # adaptive_avg_pool3d output sizes: (1,1,1), (1,2,2), (1,4,4)
        # → 256 + 256*4 + 256*16 = 256+1024+4096 too big; use depth-collapsed 2D pools instead
        self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool2 = nn.AdaptiveAvgPool3d((1, 2, 2))
        self.pool4 = nn.AdaptiveAvgPool3d((1, 4, 4))
        # 256*(1 + 4 + 16) = 5376 — project down first
        self.proj = nn.Linear(256 * (1 + 4 + 16), 256, bias=False)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        x = self.features(x)
        v1 = self.pool1(x).flatten(1)    # 256
        v2 = self.pool2(x).flatten(1)    # 256*4
        v4 = self.pool4(x).flatten(1)    # 256*16
        v = torch.relu(self.proj(torch.cat([v1, v2, v4], dim=1)))
        return self.classifier(v)


class InkDetectorInstanceNorm(nn.Module):
    """v3_instance_norm: instance norm throughout instead of batch norm.
    each sample normalized independently — no coupling between easy and hard tiles in a batch.
    hypothesis: batch norm's mean is dominated by easy tiles, suppressing hard tile gradients."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=True),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            _block(1, 32),
            _block(32, 128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            _block(128, 256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActECA(nn.Module):
    """v3_preact_eca: preact residuals + ECA channel attention after each block.
    ECA was the least harmful attention in campaign 2; combining with preact skip paths
    may let attention help without corrupting gradient flow."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32), ECA3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128), ECA3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256), ECA3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActGeMPool(nn.Module):
    """v3_preact_gem: preact backbone + GeM pooling (learnable p).
    geometric mean pool emphasizes peak responses; may better capture sparse ink signal
    vs global avg which is diluted by surrounding background voxels."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = GeMPool3d(p=float(getattr(config.model, "gem_p", 3.0)))
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActDualPool(nn.Module):
    """v3_preact_dual_pool: preact backbone + concat(avg_pool, max_pool).
    avg captures background level; max captures peak ink signal.
    both signals together may better separate faint ink from background than either alone."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = _slim_head(512, config.model.fc1_drop)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.cat([self.avg_pool(x).flatten(1),
                                          self.max_pool(x).flatten(1)], dim=1))


class InkDetectorPreActAsym(nn.Module):
    """v3_preact_asym_first: preact backbone + asymmetric (1,3,3) first conv.
    learn spatial features before coupling depth; campaign 2 showed this helps slightly.
    combines the t13 structural insight with the proven preact backbone."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActBottleneck(nn.Module):
    """v3_preact_bottleneck: preact backbone with bottleneck residuals (1×1→3×3→1×1).
    fewer 3×3 operations → more layers at same compute cost → richer feature hierarchy."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            BottleneckBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            BottleneckBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            BottleneckBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDeeperNoCBAM(nn.Module):
    """v3_deeper_no_cbam: 4-block plain residual backbone (no attention anywhere).
    t11_deeper had best hard probe; t06 (no CBAM) had best readability; combine both."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            ResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.Conv3d(256, 384, 3, padding=1, bias=False),
            nn.BatchNorm3d(384).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(384, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorPreActDeep3Pool(nn.Module):
    """v3_preact_deep_3pool: preact + 3 maxpool stages (32→128→256→384).
    matches t11_deeper topology but with preact residuals instead of plain CBAM conv.
    direct test of whether preact gradient flow helps in the 4-block deep regime."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.Conv3d(256, 384, 3, padding=1, bias=False),
            PreActResBlock3D(384),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(384).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(384, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorDepthSqueeze(nn.Module):
    """v3_depth_squeeze: collapse depth axis first via learned 1D conv → 2D spatial CNN.
    explicitly separates depth selection from spatial pattern recognition.
    if ink appears only at specific depths, learning which depth to select first is optimal."""
    def __init__(self, config):
        super().__init__()
        depth = config.data.depth
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop

        # learn to compress depth from D slices to 1 per channel (B,1,D,H,W) → (B,8,1,H,W)
        self.depth_conv = nn.Sequential(
            nn.Conv3d(1, depth, (depth, 1, 1), bias=False),
            nn.BatchNorm3d(depth).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        # now treat result as (B, depth, 1, H, W); squeeze depth dim → 2D
        # use 2D convolutions on (B, depth, H, W)
        self.spatial = nn.Sequential(
            nn.Conv2d(depth, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d1),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _slim_head(512, config.model.fc1_drop)

    def forward(self, x):
        x = self.depth_conv(x).squeeze(2)   # (B, depth, H, W)
        x = self.spatial(x)
        return self.classifier(self.pool(x))


class InkDetectorDilatedPreAct(nn.Module):
    """v3_dilated_preact: preact backbone with dilation=2 in the 3rd conv block.
    larger receptive field at the deepest level captures more spatial context
    without adding parameters; may detect diffuse/faint ink patterns better."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            # dilation=2 at this stage: receptive field doubled in each spatial dim
            nn.Conv3d(128, 256, 3, padding=2, dilation=2, bias=False),
            PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


_ARCH_MAP = {
    "v1":                       InkDetector,
    "v2_slim_head":             InkDetectorSlimHead,
    "v2_no_cbam":               InkDetectorNoCBAM,
    "v2_se_only":               InkDetectorSEOnly,
    "v2_eca":                   InkDetectorECA,
    "v2_residual":              InkDetectorResidual,
    "v2_residual_no_cbam":      InkDetectorResidualNoCBAM,
    "v2_wider_shallow":         InkDetectorWiderShallow,
    "v2_slim_all":              InkDetectorSlimAll,
    "v2_factorized_depth":      InkDetectorFactorized,
    "v2_asymmetric_first":      InkDetectorAsymFirst,
    "v2_strided_conv":          InkDetectorStridedConv,
    "v2_dual_pool":             InkDetectorDualPool,
    "v2_group_norm":            InkDetectorGroupNorm,
    "v2_depth_project":         InkDetectorDepthProject,
    "v2_two_stream":            InkDetectorTwoStream,
    "v2_inception_first":       InkDetectorInceptionFirst,
    "v2_deeper":                InkDetectorDeeper,
    "v2_bottleneck":            InkDetectorBottleneck,
    "v2_preact_res":            InkDetectorPreActRes,
    "v2_no_norm_drop":          InkDetectorNoNorm,
    # v3 — campaign 3
    "v3_preact_baseline":       InkDetectorPreActBaseline,
    "v3_linear_head":           InkDetectorLinearHead,
    "v3_depth_project_deep":    InkDetectorDepthProjectDeep,
    "v3_preact_deep":           InkDetectorPreActDeep,
    "v3_preact_wide":           InkDetectorPreActWide,
    "v3_res_no_cbam_deep":      InkDetectorResNoCBAMDeep,
    "v3_depth_attn":            InkDetectorDepthAttn,
    "v3_spatial_attn_pool":     InkDetectorSpatialAttnPool,
    "v3_nonlocal":              InkDetectorNonLocal,
    "v3_fpn":                   InkDetectorFPN,
    "v3_multiscale_pool":       InkDetectorMultiScalePool,
    "v3_instance_norm":         InkDetectorInstanceNorm,
    "v3_preact_eca":            InkDetectorPreActECA,
    "v3_preact_gem":            InkDetectorPreActGeMPool,
    "v3_preact_dual_pool":      InkDetectorPreActDualPool,
    "v3_preact_asym":           InkDetectorPreActAsym,
    "v3_preact_bottleneck":     InkDetectorPreActBottleneck,
    "v3_deeper_no_cbam":        InkDetectorDeeperNoCBAM,
    "v3_preact_deep_3pool":     InkDetectorPreActDeep3Pool,
    "v3_depth_squeeze":         InkDetectorDepthSqueeze,
    "v3_dilated_preact":        InkDetectorDilatedPreAct,
    "v3_preact_eca_deep_3pool": InkDetectorPreActDeep3Pool,  # eca variant shares topology; run with --arch v3_preact_deep_3pool + separate eca run
}

# ──────────────────────────────────────────────────────────────────────────────
# v5 architectures — campaign 5
# core insight: global average pool dilutes sparse ink signal by 160-1600×
# solutions: MIL attention, local normalization, depth profile, spectral, per-voxel
# ──────────────────────────────────────────────────────────────────────────────

class MILAttentionPool(nn.Module):
    """learn which spatial positions carry ink via softmax attention over instances"""
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x):
        B, C, *dims = x.shape
        instances = x.flatten(2).permute(0, 2, 1)          # (B, N, C)
        a = torch.softmax(self.attn(instances), dim=1)      # (B, N, 1)
        return (a * instances).sum(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class GatedMILAttentionPool(nn.Module):
    """gated attention (Ilse 2018): two-branch gate prevents attention collapse
    under class imbalance — more stable than vanilla softmax attention"""
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(feature_dim, hidden, bias=False)
        self.U = nn.Linear(feature_dim, hidden, bias=False)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        B, C, *dims = x.shape
        h = x.flatten(2).permute(0, 2, 1)                  # (B, N, C)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a, dim=1)                         # (B, N, 1)
        return (a * h).sum(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class LocalNorm3D(nn.Module):
    """normalize each tile by its own mean/std, removing the scroll-body baseline
    so ink appears as a local positive deviation rather than an absolute value"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B = x.shape[0]
        flat = x.reshape(B, -1)
        mu  = flat.mean(1).view(B, 1, 1, 1, 1)
        std = flat.std(1).clamp(min=self.eps).view(B, 1, 1, 1, 1)
        return (x - mu) / std


class InkDetectorMILAttention(nn.Module):
    """v5_mil_attention: preact backbone → attention-weighted MIL pool.
    the only pooling that can detect a signal covering <1% of the tile volume;
    global avg pool mathematically dilutes it to noise floor."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = MILAttentionPool(256)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorMILGated(nn.Module):
    """v5_mil_gated: same backbone, gated attention (Ilse 2018) — more stable."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = GatedMILAttentionPool(256)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class InkDetectorLocalNormPreact(nn.Module):
    """v5_local_norm_preact: ablation — local normalization alone with preact.
    removes scroll body baseline before any learned feature extraction."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.local_norm = LocalNorm3D()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.local_norm(x))))


class InkDetectorLocalNormMIL(nn.Module):
    """v5_local_norm_mil: local normalization (amplify small ink signal) + MIL attention (find it)."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.local_norm = LocalNorm3D()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = MILAttentionPool(256)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.local_norm(x))))


class InkDetectorLocalNormMILGated(nn.Module):
    """v5_local_norm_mil_gated: local normalization + gated MIL — most stable combination."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.local_norm = LocalNorm3D()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = GatedMILAttentionPool(256)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.local_norm(x))))


class InkDetectorDepthProfile1D(nn.Module):
    """v5_depth_profile_1d: treat depth slices as absorption time-series.
    spatial average → 1D CNN along depth axis.
    ink creates a characteristic bell-shaped absorption peak across depth."""
    def __init__(self, config):
        super().__init__()
        self.depth_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        profile = x.mean(dim=[3, 4])             # (B, 1, D) — spatial average per depth
        h = self.depth_cnn(profile)              # (B, 128, D)
        return self.classifier(self.pool(h))     # scalar


class InkDetectorDepthTransformer(nn.Module):
    """v5_depth_profile_transformer: transformer over depth positions.
    self-attention learns inter-depth relationships (e.g. absorption rise before drop = ink).
    more expressive than 1D CNN for non-local depth patterns."""
    def __init__(self, config):
        super().__init__()
        depth   = config.data.depth
        d_model = 64
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, depth, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.classifier = _slim_head(d_model, config.model.fc1_drop)

    def forward(self, x):
        profile = x.mean(dim=[3, 4]).squeeze(1).unsqueeze(-1)  # (B, D, 1)
        h = self.input_proj(profile) + self.pos_emb            # (B, D, d_model)
        h = self.encoder(h).mean(dim=1)                        # (B, d_model)
        return self.classifier(h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorDepthVariance2D(nn.Module):
    """v5_depth_variance_2d: depth variance map as primary signal.
    at ink positions, absorption varies strongly across depth (rise then fall);
    background has flat depth profiles (low variance). zero-parameter physics feature."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            # 2 channels: depth variance + depth mean
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d1),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(d2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        sq = x.squeeze(1)                               # (B, D, H, W)
        var_map  = sq.var(dim=1, keepdim=True)          # (B, 1, H, W)
        mean_map = sq.mean(dim=1, keepdim=True)
        return self.classifier(self.pool(self.features(torch.cat([var_map, mean_map], dim=1))))


class InkDetectorSpectral3D(nn.Module):
    """v5_spectral_3d: classify on FFT magnitude spectrum per depth slice.
    ink creates characteristic high-frequency absorption edges that survive into
    the log-magnitude spectrum even when sub-voxel in size."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        spec = torch.fft.rfft2(x.squeeze(1), norm='ortho')  # (B, D, H, W//2+1)
        mag  = torch.log1p(spec.abs()).unsqueeze(1)           # (B, 1, D, H, W//2+1)
        pad  = W - mag.shape[-1]
        if pad > 0:
            mag = torch.nn.functional.pad(mag, (0, pad))
        return self.classifier(self.pool(self.features(mag)))


class InkDetectorPerVoxelMIL(nn.Module):
    """v5_per_voxel_mil: output a 32×32 spatial heatmap; trained with MIL max loss.
    the model is forced to locate WHICH positions in the tile have ink,
    not just whether the tile-average signal exceeds a threshold.
    training: max(heatmap) → BCE.  inference: max(heatmap) = tile ink score."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        # pool only depth axis so spatial dims (32×32) are preserved
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d((2, 1, 1)), nn.Dropout3d(d1),    # depth: 8→4, spatial: 32→32
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d((2, 1, 1)), nn.Dropout3d(d2),    # depth: 4→2
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        # collapse remaining depth dim, output per-position logit map
        self.spatial_head = nn.Sequential(
            nn.Conv3d(256, 64, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, 1, bias=True),
            nn.AdaptiveMaxPool3d((1, None, None)),  # collapse depth → (B,1,1,H,W)
        )

    def forward(self, x):
        h    = self.encoder(x)          # (B, 256, 2, 32, 32)
        heat = self.spatial_head(h)     # (B, 1, 1, 32, 32)
        return heat.squeeze(2)          # (B, 1, 32, 32) — spatial heatmap


class InkDetectorSiamese(nn.Module):
    """v5_siamese: compare ink_band and pre_band embeddings explicitly.
    requires input_mode='double' — input is (B, 1, 16, H, W): ink(8) + pre(8).
    learns WHAT MAKES INK DIFFERENT from its non-ink depth neighbors,
    bypassing global-avg dilution by computing a differential embedding."""
    def __init__(self, config):
        super().__init__()
        d1, d2 = config.model.conv1_drop, config.model.conv2_drop
        # shared encoder: processes each 8-depth branch identically
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False), PreActResBlock3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False), PreActResBlock3D(128),
            nn.MaxPool3d(2), nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=1, bias=False), PreActResBlock3D(256),
            nn.MaxPool3d(2), nn.Dropout3d(d2),
            nn.BatchNorm3d(256).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        # compare: difference + hadamard product (Bromley 1993 siamese features)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 128, bias=False),
            nn.BatchNorm1d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, 1, 16, H, W) → split ink (first 8) and pre (last 8)
        ink = x[:, :, :8, :, :]
        pre = x[:, :, 8:, :, :]
        e_ink = self.pool(self.encoder(ink)).flatten(1)  # (B, 256)
        e_pre = self.pool(self.encoder(pre)).flatten(1)
        feat = torch.cat([e_ink - e_pre, e_ink * e_pre], dim=1)  # (B, 512)
        return self.classifier(feat)


class InkDetectorAEAnomaly(nn.Module):
    """v5_ae_anomaly: autoencoder where reconstruction error is the ink score.
    the autoencoder learns to reconstruct normal scroll patterns perfectly;
    ink creates anomalous voxel patterns that reconstruct poorly.
    high reconstruction error → high ink probability.
    trained end-to-end with BCE on the learned error-to-logit mapping."""
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        # learnable mapping: reconstruction_error → logit
        # log_scale=0 → scale=1; bias=-3 → starts near-negative (correct for imbalanced data)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        h = self.encoder(x)
        recon = self.decoder(h)
        if recon.shape != x.shape:
            recon = torch.nn.functional.interpolate(
                recon, size=x.shape[2:], mode='trilinear', align_corners=False
            )
        error = (x - recon).pow(2).mean(dim=[1, 2, 3, 4], keepdim=True)  # (B, 1, 1, 1, 1)
        return (error * self.log_scale.exp() + self.bias).view(-1, 1)     # (B, 1)


# add v5 entries to the existing map
_ARCH_MAP.update({
    "v5_mil_attention":          InkDetectorMILAttention,
    "v5_mil_gated":              InkDetectorMILGated,
    "v5_local_norm_preact":      InkDetectorLocalNormPreact,
    "v5_local_norm_mil":         InkDetectorLocalNormMIL,
    "v5_local_norm_mil_gated":   InkDetectorLocalNormMILGated,
    "v5_depth_profile_1d":       InkDetectorDepthProfile1D,
    "v5_depth_profile_transformer": InkDetectorDepthTransformer,
    "v5_depth_variance_2d":      InkDetectorDepthVariance2D,
    "v5_spectral_3d":            InkDetectorSpectral3D,
    "v5_per_voxel_mil":          InkDetectorPerVoxelMIL,
    "v5_siamese":                InkDetectorSiamese,
    "v5_ae_anomaly":             InkDetectorAEAnomaly,
})

# ──────────────────────────────────────────────────────────────────────────────
# v6 architectures — campaign 6
# key insight from C5: depth profile 1D CNN (0.360) and Transformer (0.372)
# are the best hard-probe performers across ALL campaigns.
# the absorption curve shape through depth IS the ink signal.
# campaign 6: push this to its physical limits.
#   - full 64-depth profiles (8x more info, requires fulldepth input mode)
#   - per-pixel profiles (no spatial averaging — find the exact ink pixels)
#   - sequential models (LSTM/GRU/attention) designed to capture curve shape
#   - physics transforms (derivative, Beer-Lambert)
#   - multi-scale spatial aggregation
# ──────────────────────────────────────────────────────────────────────────────

class _PixelDepthCNN(nn.Module):
    """shared 1D CNN for independent per-pixel depth profile analysis"""
    def __init__(self, in_depth, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.cnn(x)).squeeze(-1)  # (N, out_dim)


class InkDetectorPerPixel1D(nn.Module):
    """v6_perpixel_1d: process each of 32x32=1024 pixels' 8-depth profile independently
    via a shared 1D CNN, then aggregate with MIL attention across pixel predictions.
    unlike C5 depth profile models which average spatially first, this preserves WHERE
    in the tile the ink absorption signal is strongest."""
    def __init__(self, config):
        super().__init__()
        ch = 32
        self.pixel_cnn = _PixelDepthCNN(config.data.depth, ch)
        self.attn = nn.Sequential(nn.Linear(ch, 16), nn.Tanh(), nn.Linear(16, 1))
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = x.squeeze(1).permute(0,2,3,1).reshape(B*H*W, 1, D)
        f = self.pixel_cnn(p).view(B, H*W, -1)
        a = torch.softmax(self.attn(f), dim=1)
        return self.head((a * f).sum(1))


class InkDetectorPerPixelGated(nn.Module):
    """v6_perpixel_gated: per-pixel 1D profiles + gated MIL (Ilse 2018).
    gated attention is more stable than vanilla softmax attention under class imbalance."""
    def __init__(self, config):
        super().__init__()
        ch = 32
        self.pixel_cnn = _PixelDepthCNN(config.data.depth, ch)
        self.V = nn.Linear(ch, 16, bias=False)
        self.U = nn.Linear(ch, 16, bias=False)
        self.w = nn.Linear(16, 1, bias=False)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = x.squeeze(1).permute(0,2,3,1).reshape(B*H*W, 1, D)
        f = self.pixel_cnn(p).view(B, H*W, -1)
        a = torch.softmax(self.w(torch.tanh(self.V(f)) * torch.sigmoid(self.U(f))), dim=1)
        return self.head((a * f).sum(1))


class InkDetectorPerPixelMax(nn.Module):
    """v6_perpixel_max: per-pixel 1D profiles + hard max pooling.
    if even ONE pixel has a strongly ink-shaped absorption profile, tile = ink.
    unlike attention which averages, max is uncompromising about the best pixel."""
    def __init__(self, config):
        super().__init__()
        ch = 32
        self.pixel_cnn = _PixelDepthCNN(config.data.depth, ch)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = x.squeeze(1).permute(0,2,3,1).reshape(B*H*W, 1, D)
        f = self.pixel_cnn(p).view(B, H*W, -1)
        return self.head(f.max(dim=1).values)


class InkDetectorFullDepth1D(nn.Module):
    """v6_fulldepth_1d: full 64-depth spatial-mean profile + deep 1D CNN.
    requires fulldepth input mode. uses ALL 64 zarr depth slices (not just 8).
    the complete absorption curve reveals the ink peak shape far more clearly:
    the bell-shaped absorption profile of carbon ink spans many depth slices."""
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile = x.squeeze(1).mean(dim=[2,3]).unsqueeze(1)  # (B, 1, D)
        return self.head(self.pool(self.cnn(profile)))


class InkDetectorFullDepthTransformer(nn.Module):
    """v6_fulldepth_transformer: full 64-depth spatial-mean profile + Transformer.
    each depth position is a token; self-attention captures the full absorption curve.
    64 tokens gives the transformer enough context to identify the characteristic
    ink absorption peak shape (baseline → rise → peak → fall → baseline)."""
    def __init__(self, config):
        super().__init__()
        d_model = 64
        self.proj = nn.Linear(1, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=256,
                                          dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=4)
        self.head = _slim_head(d_model, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile = x.squeeze(1).mean(dim=[2,3]).unsqueeze(-1)  # (B, D, 1)
        h = self.proj(profile)                                  # (B, D, d_model)
        h = self.enc(h).mean(dim=1)                            # (B, d_model)
        return self.head(h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorFullDepthGRU(nn.Module):
    """v6_fulldepth_gru: full 64-depth spatial-mean profile + bidirectional GRU.
    bidirectional: captures BOTH the entry edge (absorption rising into ink layer)
    AND the exit edge (falling out). ink absorption is symmetric — BiGRU exploits this."""
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(1, 64, num_layers=3, batch_first=True,
                           bidirectional=True, dropout=0.0)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile = x.squeeze(1).mean(dim=[2,3]).unsqueeze(-1)  # (B, D, 1)
        out, _ = self.gru(profile)
        return self.head(out.mean(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorFullDepthPerPixel(nn.Module):
    """v6_fulldepth_perpixel: per-pixel FULL 64-depth profiles + gated MIL.
    the most information-rich approach in the campaign: every pixel's complete
    absorption curve through all 64 depths, with attention learning which pixels
    show the characteristic ink absorption signature. if ink is at even 1 pixel,
    its full 64-depth absorption profile should be unambiguously distinguishable."""
    def __init__(self, config):
        super().__init__()
        ch = 64
        self.pixel_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, ch, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.V = nn.Linear(ch, 32, bias=False)
        self.U = nn.Linear(ch, 32, bias=False)
        self.w = nn.Linear(32, 1, bias=False)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = x.squeeze(1).permute(0,2,3,1).reshape(B*H*W, 1, D)
        f = self.pixel_cnn(p).squeeze(-1).view(B, H*W, -1)
        a = torch.softmax(self.w(torch.tanh(self.V(f)) * torch.sigmoid(self.U(f))), dim=1)
        return self.head((a * f).sum(1))


class InkDetectorDepthDeriv(nn.Module):
    """v6_depth_derivative: 1D CNN on depth DERIVATIVE profile (dI/dz).
    ink creates an absorption edge perpendicular to the scroll surface:
    the derivative shows a positive peak entering the ink layer, negative exiting.
    this edge pattern is more discriminative than the raw absorption value."""
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile = x.squeeze(1).mean(dim=[2,3])      # (B, D)
        deriv = profile[:, 1:] - profile[:, :-1]    # (B, D-1) — first derivative
        return self.head(self.pool(self.cnn(deriv.unsqueeze(1))))


class InkDetectorBeerLambert(nn.Module):
    """v6_beer_lambert: -log(I+eps) transform before depth profile classification.
    Beer-Lambert law: CT intensity I is related to attenuation by I = I0 * exp(-mu*x).
    taking -log(I) converts to linear attenuation coefficients where ink and papyrus
    have different material constants. the depth profile in log-space may be more
    discriminative for detecting the small ink absorption above papyrus baseline."""
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        bl = -torch.log(x.squeeze(1).mean(dim=[2,3]).clamp(min=1e-6))  # (B, D)
        return self.head(self.pool(self.cnn(bl.unsqueeze(1))))


class InkDetectorRobustStats(nn.Module):
    """v6_robust_stats: spatial MEAN + STANDARD DEVIATION depth profile (2-channel).
    the mean profile is the C5 depth signal. the std profile is new: at ink positions,
    absorption is spatially variable (some voxels have carbon, others don't within the tile).
    the combination of mean (signal) + std (spatial heterogeneity) may be more discriminative."""
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        flat = x.squeeze(1).view(B, D, H*W)
        mean = flat.mean(dim=2)  # (B, D)
        std  = flat.std(dim=2)   # (B, D)
        profile = torch.stack([mean, std], dim=1)  # (B, 2, D)
        return self.head(self.pool(self.cnn(profile)))


class InkDetectorLSTMSlices(nn.Module):
    """v6_lstm_slices: LSTM over 8 depth slices, each slice encoded by a 2D conv.
    the LSTM learns: 'what changes as we go deeper through the scroll?'
    ink creates a systematic absorption increase-then-decrease pattern across slices.
    the LSTM state captures this accumulating pattern as it processes each slice."""
    def __init__(self, config):
        super().__init__()
        self.slice_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        sq = x.squeeze(1)
        feats = [self.slice_enc(sq[:, d:d+1]).view(B, -1) for d in range(D)]
        seq = torch.stack(feats, dim=1)  # (B, D, 64)
        out, _ = self.lstm(seq)
        return self.head(out.mean(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorBiGRUSlices(nn.Module):
    """v6_bigru_slices: bidirectional GRU over depth slices, each = 2D conv features.
    bidirectional: captures BOTH the forward (entering ink) and backward (exiting ink) context.
    lighter than LSTM but captures the same cross-depth sequential dependencies."""
    def __init__(self, config):
        super().__init__()
        self.slice_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        sq = x.squeeze(1)
        feats = [self.slice_enc(sq[:, d:d+1]).view(B, -1) for d in range(D)]
        seq = torch.stack(feats, dim=1)
        out, _ = self.gru(seq)
        return self.head(out.mean(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorSliceAttention(nn.Module):
    """v6_slice_attention: multi-head self-attention over depth slices.
    each depth slice is encoded as a feature vector; transformer attends across slices.
    unlike LSTM which processes sequentially, attention can directly compare any two
    depths — e.g. 'this depth has high absorption compared to depth 2 slices earlier.'"""
    def __init__(self, config):
        super().__init__()
        d_model = 64
        self.slice_enc = nn.Sequential(
            nn.Conv2d(1, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        enc = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=128,
                                          dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.head = _slim_head(d_model, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        sq = x.squeeze(1)
        feats = [self.slice_enc(sq[:, d:d+1]).view(B, -1) for d in range(D)]
        seq = torch.stack(feats, dim=1)
        h = self.enc(seq).mean(dim=1)
        return self.head(h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorPerPixelLocalSub(nn.Module):
    """v6_perpixel_local_sub: per-pixel RESIDUAL profiles (pixel - spatial mean).
    subtracts the tile-level mean depth profile from each pixel before classifying.
    removes tile-level background variation; the residual encodes WHERE ink deviates
    from the local baseline — amplifying the small ink signal above the scroll body."""
    def __init__(self, config):
        super().__init__()
        ch = 32
        self.pixel_cnn = _PixelDepthCNN(config.data.depth, ch)
        self.V = nn.Linear(ch, 16, bias=False)
        self.U = nn.Linear(ch, 16, bias=False)
        self.w = nn.Linear(16, 1, bias=False)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        spatial_mean = x.mean(dim=[3,4], keepdim=True)  # (B,1,D,1,1)
        residual = x - spatial_mean                       # per-pixel residual profiles
        p = residual.squeeze(1).permute(0,2,3,1).reshape(B*H*W, 1, D)
        f = self.pixel_cnn(p).view(B, H*W, -1)
        a = torch.softmax(self.w(torch.tanh(self.V(f)) * torch.sigmoid(self.U(f))), dim=1)
        return self.head((a * f).sum(1))


class InkDetectorTripleScale(nn.Module):
    """v6_triple_scale: multi-resolution depth profile analysis.
    computes depth profiles at 3 spatial granularities simultaneously:
      - whole-tile mean (1 profile): global absorption
      - 2x2 quadrant means (4 profiles): coarse spatial structure
      - 4x4 block means (16 profiles): fine spatial structure
    if ink covers only part of a tile, coarser/finer scales reveal it differently."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        def _cnn(in_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, 32, 3, padding=1, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        self.cnn1  = _cnn(1)
        self.cnn4  = _cnn(4)
        self.cnn16 = _cnn(16)
        self.head  = _slim_head(64*3, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        sq = x.squeeze(1)  # (B, D, H, W)
        p1  = sq.mean(dim=[2,3]).unsqueeze(1)  # (B, 1, D)
        H2, W2 = H//2, W//2
        q4  = torch.stack([sq[:,:,i*H2:(i+1)*H2,j*W2:(j+1)*W2].mean(dim=[2,3])
                           for i in range(2) for j in range(2)], dim=1)  # (B,4,D)
        H4, W4 = H//4, W//4
        q16 = torch.stack([sq[:,:,i*H4:(i+1)*H4,j*W4:(j+1)*W4].mean(dim=[2,3])
                           for i in range(4) for j in range(4)], dim=1)  # (B,16,D)
        f1  = self.cnn1(p1).squeeze(-1)
        f4  = self.cnn4(q4).squeeze(-1)
        f16 = self.cnn16(q16).squeeze(-1)
        return self.head(torch.cat([f1, f4, f16], dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorProfilePCA(nn.Module):
    """v6_profile_pca: learnable PCA-like depth profile decomposition.
    learns N basis vectors for 1D depth profiles. classification is on projection
    coefficients. ink profiles should project onto different basis vectors than
    background, even when the raw profile difference is tiny in L2 terms."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        n = 16
        self.basis = nn.Parameter(torch.randn(n, D) * 0.1)
        self.head   = _slim_head(n, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile   = x.squeeze(1).mean(dim=[2,3])                       # (B, D)
        basis_n   = nn.functional.normalize(self.basis, dim=1)          # (n, D)
        proj      = torch.matmul(profile, basis_n.T)                    # (B, n)
        return self.head(proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class InkDetectorFullDepthDeriv(nn.Module):
    """v6_fulldepth_deriv: first derivative of full 64-depth spatial-mean profile.
    with 64 depth values, the ink absorption edge (rise then fall) is clearly resolved.
    the derivative converts the bell curve into a biphasic signal (positive + negative peak)
    which may be more discriminative since background profiles have near-zero derivative."""
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2, stride=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(128, config.model.fc1_drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        profile = x.squeeze(1).mean(dim=[2,3])      # (B, D)
        deriv   = profile[:, 1:] - profile[:, :-1]  # (B, D-1)
        return self.head(self.pool(self.cnn(deriv.unsqueeze(1))))


# ---- spatial self-attention architectures (v6_pixel_spatial_attn, v6_pixel_local_attn) ----
#
# motivation: ink at 7.91um is sub-voxel (ink particles 1-5um), but ink STROKES are
# 100-500um wide = 12-63 voxels. this means many adjacent pixels all carry a weak
# sub-voxel absorption signal. self-attention lets each pixel ask "do my neighbors
# also look anomalous?" — exactly the right inductive bias for detecting correlated
# partial-voxel ink absorption across a letter stroke.
#
# unlike MIL (which scores pixels independently), these models can learn that a
# CLUSTER of mildly-elevated pixels is more indicative of ink than a single outlier.

class InkDetectorPixelSpatialAttn(nn.Module):
    """
    full spatial self-attention over per-pixel depth profiles.

    each of the 1024 pixels in a 32x32 tile gets its depth profile
    encoded by a weight-shared MLP. an 8-layer, 8-head transformer then
    lets all pixels compare with each other to find spatially correlated
    absorption anomalies (ink letter strokes spanning many adjacent pixels).

    ink strokes are 100-500um wide = 12-63 voxels at 7.91um/vox. a cluster
    of mildly-elevated pixels is more indicative of ink than a single outlier;
    self-attention learns exactly this distinction.

    d=512, 8 layers, nhead=8 with flash attention: ~8-9 GB VRAM at batch=64.
    uses z=28-40 (12 slices) covering the full ink band rather than just the peak.
    """
class InkDetectorPixelSpatialAttn(nn.Module):
    """
    full spatial self-attention over per-pixel depth profiles.

    each of the 1024 pixels in a 32x32 tile gets its 8-depth profile encoded
    by a weight-shared MLP. a transformer then lets all pixels compare with
    each other to find spatially correlated absorption anomalies.

    d=256 keeps VRAM at ~4-5GB with batch=64, leaving headroom on 25GB GPU.
    norm_first=False enables pytorch's fused flash-attention kernel and nested
    tensor fast path, giving real GPU utilization instead of 1%.
    gradient checkpointing trades recomputation for ~half the activation VRAM.
    """
    def __init__(self, config):
        super().__init__()
        in_ch = config.data.depth
        d = 512

        self.profile_enc = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.GELU(),
            nn.Linear(256, d),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # norm_first=False enables pytorch's fast fused path and nested tensors
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=8,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5:
            x = x.squeeze(1)
        profiles = x.permute(0, 2, 3, 1).reshape(B * 1024, -1)
        feats = self.profile_enc(profiles)
        tokens = feats.reshape(B, 1024, -1) + self.pos_embed
        tokens = self.transformer(tokens)
        out = self.norm(tokens.mean(dim=1))
        return self.head(out)


class InkDetectorPixelLocalAttn(nn.Module):
    """
    local window attention over per-pixel depth profiles, then global cross-window.

    splits 32x32 into 16 non-overlapping 8x8 windows (64 tokens each, ~63um/window).
    local attention captures sub-stroke structure; global attention captures
    stroke-level structure. d=256, norm_first=False for fast path.
    gradient checkpointing on local stage.
    """
    def __init__(self, config):
        super().__init__()
        in_ch = config.data.depth
        d = 512
        W = 8

        self.profile_enc = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.GELU(),
            nn.Linear(256, d),
            nn.GELU(),
        )
        self.local_pos = nn.Parameter(torch.zeros(1, W * W, d))
        nn.init.trunc_normal_(self.local_pos, std=0.02)

        local_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.local_attn = nn.TransformerEncoder(
            local_layer, num_layers=4, enable_nested_tensor=False
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.global_attn = nn.TransformerEncoder(
            global_layer, num_layers=4, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.W = W

    def _run_local(self, feats):
        return self.local_attn(feats + self.local_pos)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5:
            x = x.squeeze(1)
        W = self.W
        n_w = 32 // W

        feats = self.profile_enc(x.permute(0, 2, 3, 1))
        feats = feats.reshape(B, n_w, W, n_w, W, -1)
        feats = feats.permute(0, 1, 3, 2, 4, 5).reshape(B * n_w * n_w, W * W, -1)
        feats = self._run_local(feats)
        window_reps = feats.mean(dim=1).reshape(B, n_w * n_w, -1)

        # global attention across 16 windows
        window_reps = self.global_attn(window_reps)
        out = self.norm(window_reps.mean(dim=1))
        return self.head(out)


_ARCH_MAP.update({
    "v6_perpixel_1d":           InkDetectorPerPixel1D,
    "v6_perpixel_gated":        InkDetectorPerPixelGated,
    "v6_perpixel_max":          InkDetectorPerPixelMax,
    "v6_fulldepth_1d":          InkDetectorFullDepth1D,
    "v6_fulldepth_transformer": InkDetectorFullDepthTransformer,
    "v6_fulldepth_gru":         InkDetectorFullDepthGRU,
    "v6_fulldepth_perpixel":    InkDetectorFullDepthPerPixel,
    "v6_depth_derivative":      InkDetectorDepthDeriv,
    "v6_beer_lambert":          InkDetectorBeerLambert,
    "v6_robust_stats":          InkDetectorRobustStats,
    "v6_lstm_slices":           InkDetectorLSTMSlices,
    "v6_bigru_slices":          InkDetectorBiGRUSlices,
    "v6_slice_attention":       InkDetectorSliceAttention,
    "v6_perpixel_local_sub":    InkDetectorPerPixelLocalSub,
    "v6_triple_scale":          InkDetectorTripleScale,
    "v6_profile_pca":           InkDetectorProfilePCA,
    "v6_fulldepth_deriv":       InkDetectorFullDepthDeriv,
    "v6_pixel_spatial_attn":    InkDetectorPixelSpatialAttn,
    "v6_pixel_local_attn":      InkDetectorPixelLocalAttn,
})


# ==== campaign 7 architectures ====
#
# C6 finding: depth-sequential models (LSTM/BiGRU/Transformer over depth positions)
# outperformed all spatial approaches. the hard probe still shows nothing visible.
# C7 hypothesis: the signal is sub-voxel; we need more sensitive feature extraction.
# key directions: robust statistics (percentiles), anomaly detection, multi-scale depth,
# deeper sequential models, and re-enabling hard mining + focal loss.

class InkDetectorPercentileDepth(nn.Module):
    """
    5 percentiles (10/25/50/75/90) at each of D depth slices → 5D-dim deep MLP.
    if ink occupies 5-10% of pixels in a tile, the 90th percentile at ink depths
    will be elevated even when the mean is masked by noise. more robust than mean+std.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        self.mlp = nn.Sequential(
            nn.Linear(D * 5, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        pixels = x.reshape(B, x.size(1), -1)           # (B, D, 1024)
        q = torch.quantile(pixels, self.qs, dim=-1)     # (5, B, D)
        return self.mlp(q.permute(1, 2, 0).reshape(B, -1))


class InkDetectorCenteredDepth(nn.Module):
    """
    subtract the per-tile mean absorption from each depth slice.
    the centered profile encodes RELATIVE depth variation — the ink absorption
    shape — removing scroll-wide brightness that drowns out the weak ink signal.
    a flat non-ink profile becomes all-zeros; ink creates a positive bump at specific depths.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])                          # (B, D)
        centered = profile - profile.mean(dim=1, keepdim=True)  # remove tile baseline
        return self.head(self.pool(self.cnn(centered.unsqueeze(1))))


class InkDetectorPairwiseDepth(nn.Module):
    """
    all C(D,2) pairwise differences between depth positions → deep MLP.
    for D=8: 28 features. explicitly encodes which depths are relatively brighter.
    ink creates a systematic pattern where D_ink > D_pre and D_ink > D_post.
    scale-invariant to global brightness shifts across tiles.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        pairs = [(i, j) for i in range(D) for j in range(i+1, D)]
        self.register_buffer('pi', torch.tensor([p[0] for p in pairs]))
        self.register_buffer('pj', torch.tensor([p[1] for p in pairs]))
        n = len(pairs)
        self.mlp = nn.Sequential(
            nn.Linear(n, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        diffs = profile[:, self.pi] - profile[:, self.pj]  # (B, n_pairs)
        return self.mlp(diffs)


class InkDetectorPrototypeDepth(nn.Module):
    """
    K learned prototype depth profiles; score each tile by cosine similarity
    to each prototype. like a matched filter bank — the model learns what
    ink-like and non-ink-like depth profile shapes look like, then scores
    by similarity. K=32 prototypes → MLP classifier.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        K = 32
        self.prototypes = nn.Parameter(torch.randn(K, D) * 0.1)
        self.head = nn.Sequential(
            nn.Linear(K, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        p_n = torch.nn.functional.normalize(profile, dim=1)
        k_n = torch.nn.functional.normalize(self.prototypes, dim=1)
        sim = p_n @ k_n.T  # (B, K) cosine similarity
        return self.head(sim)


class InkDetectorSuperpixelAttn(nn.Module):
    """
    split 32x32 tile into 4×4=16 non-overlapping 8x8 superpixels.
    each superpixel gets its mean depth profile as a token.
    transformer self-attention over 16 tokens: superpixels ask 'are my neighbors
    also anomalous?' — the right scale for ink stroke detection (8×8 = 63um, matching
    typical letter stroke width). compromise between per-pixel noise and global averaging.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        d = 128
        self.proj = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d*4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.attn = nn.TransformerEncoder(enc, num_layers=6, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        D = x.size(1)
        # 4×4 superpixels of 8×8 each — average over spatial region
        sp = x.reshape(B, D, 4, 8, 4, 8).mean(dim=[3, 5])  # (B, D, 4, 4)
        tokens = sp.permute(0, 2, 3, 1).reshape(B, 16, D)   # (B, 16, D)
        tokens = self.proj(tokens) + self.pos
        tokens = self.attn(tokens)
        return self.head(self.norm(tokens.mean(1)))


class _ResBlock1D(nn.Module):
    """residual block for 1D sequences: 2×conv + batchnorm + skip"""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class InkDetectorDeepResnet1D(nn.Module):
    """
    12-block 1D ResNet on the depth profile.
    each residual block has 2 conv layers + skip connection.
    far deeper than any depth-profile model tried before — 24 conv layers total.
    the depth profile is only 8 or 64 values, so the model learns extremely fine
    pattern distinctions that shallower models miss.
    used for both 8-depth (v7_deep_resnet_depth) and 64-depth (v7_full64_deep_resnet).
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        ch = 256
        self.embed = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock1D(ch) for _ in range(12)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(ch, config.model.fc1_drop)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])              # (B, D)
        h = self.embed(profile.unsqueeze(1))         # (B, ch, D)
        h = self.blocks(h)
        return self.head(self.pool(h))


class InkDetectorMultiscaleDepth(nn.Module):
    """
    parallel dilated conv1d at rates 1, 2, 4 on depth profile — concatenated.
    rate-1: adjacent depths differ (local ink peak edge).
    rate-2: 2-step differences (broader ink band shape).
    rate-4: global profile shape (full bell curve).
    captures local and global depth patterns simultaneously, unlike single-scale 1D CNN.
    """
    def __init__(self, config):
        super().__init__()
        ch = 128
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
            )
            for d in [1, 2, 4]
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(ch * 3, config.model.fc1_drop)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1]).unsqueeze(1)  # (B, 1, D)
        feats = [self.pool(b(profile)).squeeze(-1) for b in self.branches]
        return self.head(torch.cat(feats, dim=1))


class InkDetectorInceptionDepth(nn.Module):
    """
    inception-style: parallel conv1d with kernel sizes 1, 3, 5, 7 concatenated.
    kernel-1: pointwise (depth value directly).
    kernel-3: adjacent depth context.
    kernel-5/7: broader depth shape.
    then a second stage mixes features from all scales.
    """
    def __init__(self, config):
        super().__init__()
        ch = 64
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(ch), nn.GELU(),
            )
            for k in [1, 3, 5, 7]
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(ch * 4, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(256, config.model.fc1_drop)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1]).unsqueeze(1)  # (B, 1, D)
        feats = torch.cat([b(profile) for b in self.branches], dim=1)
        return self.head(self.pool(self.mix(feats)))


class InkDetectorDeepTransformerDepth(nn.Module):
    """
    12-layer transformer on D=8 depth positions with d=256.
    when N=8, each layer has only 64 attention values per head — trivially cheap.
    this is 3-6x deeper than any C5/C6 depth transformer.
    deep chains of attention can learn complex non-local relationships between
    depth positions that shallow transformers miss.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        d = 256
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, D, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=12, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])               # (B, D)
        tokens = self.embed(profile.unsqueeze(-1)) + self.pos  # (B, D, d)
        tokens = self.transformer(tokens)
        return self.head(self.norm(tokens.mean(1)))


class InkDetectorFull64DeepBiGRU(nn.Module):
    """
    4-layer bidirectional GRU on the full 64-depth spatial-mean profile.
    C6 t09 (1-layer full64 bigru) got hard=0.436 — best depth-sequential result.
    4 layers with 256 hidden should capture far more complex sequential patterns
    across the complete 64-depth absorption bell curve.
    requires input_mode='fulldepth'.
    """
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(
            1, 256, num_layers=4, batch_first=True,
            bidirectional=True, dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])       # (B, D=64)
        out, _ = self.gru(profile.unsqueeze(-1))  # (B, D, 512)
        return self.head(out[:, -1])         # last timestep


class InkDetectorPixelDeviation(nn.Module):
    """
    per-pixel depth profile minus tile mean depth profile → gated MIL.
    the residual encodes "how does THIS pixel deviate from the tile average?"
    background: all pixels similar → near-zero residuals.
    ink pixel: anomalous absorption at ink depths → large residual there.
    gated MIL selects the most deviant (most anomalous) pixels as evidence for ink.
    explicitly designed to find sub-voxel ink particles as spatial outliers.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        ch = 32
        self.pixel_cnn = _PixelDepthCNN(D, ch)
        # gated MIL
        self.attn_V = nn.Linear(ch, ch)
        self.attn_U = nn.Linear(ch, ch)
        self.attn_w = nn.Linear(ch, 1)
        self.head = nn.Sequential(nn.Linear(ch, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        tile_mean = x.mean(dim=[-2, -1], keepdim=True)  # (B, D, 1, 1)
        residual = x - tile_mean                          # per-pixel deviation
        pixels = residual.permute(0, 2, 3, 1).reshape(B * 1024, -1)
        feats = self.pixel_cnn(pixels.unsqueeze(1)).reshape(B, 1024, -1)
        V = torch.tanh(self.attn_V(feats))
        U = torch.sigmoid(self.attn_U(feats))
        a = torch.softmax(self.attn_w(V * U).squeeze(-1), dim=-1)
        z = (a.unsqueeze(-1) * feats).sum(1)
        return self.head(z)


class InkDetectorDiffPercentile(nn.Module):
    """
    percentile features on the DIFFERENTIAL signal (ink_band - pre_band).
    combines two independently effective ideas:
    - differential input removes scroll baseline, leaving ink absorption shape
    - percentile statistics are robust to per-pixel noise; the 90th percentile of
      (ink-pre) is elevated even when mean is masked by noise
    use with input_mode='diff'.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        self.mlp = nn.Sequential(
            nn.Linear(D * 5, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        pixels = x.reshape(B, x.size(1), -1)         # (B, D, 1024)
        q = torch.quantile(pixels, self.qs, dim=-1)  # (5, B, D)
        return self.mlp(q.permute(1, 2, 0).reshape(B, -1))


class InkDetectorSpectralDepth(nn.Module):
    """
    real FFT of depth profile: D values → D//2+1 complex → real+imag concatenated.
    different materials have different characteristic frequency spectra in their
    depth absorption pattern. ink creates a specific frequency component (the bell
    curve shape concentrates energy at low frequencies but with a characteristic ratio).
    scale-invariant to overall brightness; the frequency spectrum encodes shape only.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        n_fft = D // 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(n_fft * 2, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])             # (B, D)
        fft = torch.fft.rfft(profile, dim=1)       # (B, D//2+1) complex
        feats = torch.cat([fft.real, fft.imag], dim=1)
        return self.mlp(feats)


class InkDetectorBiGRUPercentile(nn.Module):
    """
    BiGRU on the sequence of percentile vectors across depth positions.
    at each depth d, we have 5 percentiles (10/25/50/75/90) describing the
    spatial distribution. the BiGRU processes this as a time series of
    distribution-snapshots through depth. captures how the spatial distribution
    EVOLVES through depth — ink should cause an anomalous distribution shift at specific depths.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        self.gru = nn.GRU(
            5, 256, num_layers=3, batch_first=True,
            bidirectional=True, dropout=0.1
        )
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        pixels = x.reshape(B, x.size(1), -1)         # (B, D, 1024)
        q = torch.quantile(pixels, self.qs, dim=-1)  # (5, B, D)
        seq = q.permute(1, 2, 0)                     # (B, D, 5) — time series of percentile vectors
        out, _ = self.gru(seq)                       # (B, D, 512)
        return self.head(out[:, -1])


class InkDetectorAEBottleneck(nn.Module):
    """
    joint autoencoder + classifier on depth profiles.
    the encoder compresses D→16 bottleneck. decoder reconstructs D.
    the classifier sees: bottleneck features + per-dimension reconstruction errors.
    ink tiles with unusual depth profiles should have high reconstruction error
    under the model trained mostly on background profiles. both the compressed
    representation AND the reconstruction difficulty are informative.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.encoder = nn.Sequential(
            nn.Linear(D, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 16), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, D),
        )
        self.head = nn.Sequential(
            nn.Linear(16 + D, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])           # (B, D)
        bottleneck = self.encoder(profile)
        recon = self.decoder(bottleneck)
        error = (profile - recon).abs()          # per-dim reconstruction error
        feats = torch.cat([bottleneck, error], dim=1)
        return self.head(feats)


_ARCH_MAP.update({
    "v7_percentile_depth":       InkDetectorPercentileDepth,
    "v7_centered_depth":         InkDetectorCenteredDepth,
    "v7_pairwise_depth":         InkDetectorPairwiseDepth,
    "v7_prototype_depth":        InkDetectorPrototypeDepth,
    "v7_superpixel_attn":        InkDetectorSuperpixelAttn,
    "v7_deep_resnet_depth":      InkDetectorDeepResnet1D,
    "v7_multiscale_depth":       InkDetectorMultiscaleDepth,
    "v7_inception_depth":        InkDetectorInceptionDepth,
    "v7_deep_transformer_depth": InkDetectorDeepTransformerDepth,
    "v7_full64_deep_bigru":      InkDetectorFull64DeepBiGRU,
    "v7_pixel_deviation":        InkDetectorPixelDeviation,
    "v7_diff_percentile":        InkDetectorDiffPercentile,
    "v7_spectral_depth":         InkDetectorSpectralDepth,
    "v7_bigru_percentile":       InkDetectorBiGRUPercentile,
    "v7_ae_bottleneck":          InkDetectorAEBottleneck,
})


# ==== campaign 8: sub-voxel ink sensitivity push ====
# ring negatives give clean 1:1 training. no focal loss needed.
# all architectures use ring negatives by default in campaign 8.
# key challenge: ink at 7.91um is sub-voxel (1-5um particles in 7.91um voxel).
# the 3.7um scan immediately revealed ink — the signal exists but is tiny.
# strategies: matched filters, deep sequential, spatial contrast, physics-based.

class InkDetectorMatchedFilter(nn.Module):
    """
    K=64 learned depth-profile templates. at test time each tile's profile
    is compared to all templates by cosine similarity. the similarity vector
    → MLP classifier. this is the matched filter principle: the optimal
    detector for a known signal shape in Gaussian noise learns what ink
    absorption curves look like and scores by how well the input matches.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        K = 64
        self.templates = nn.Parameter(torch.randn(K, D) * 0.1)
        self.head = nn.Sequential(
            nn.Linear(K, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])                             # (B, D)
        p_n = torch.nn.functional.normalize(profile, dim=1)
        t_n = torch.nn.functional.normalize(self.templates, dim=1)
        sim = p_n @ t_n.T                                          # (B, K)
        return self.head(sim)


class InkDetectorPercentileBiGRU(nn.Module):
    """
    percentile-feature sequence → deep BiGRU.
    at each depth d: 5 percentiles [10,25,50,75,90] across 32x32 pixels.
    a 4-layer BiGRU processes the (D, 5) sequence through depth.
    combines: (1) percentile robustness to sparse ink pixels,
              (2) sequential depth modeling of the absorption curve shape,
              (3) depth of recurrent model to capture complex patterns.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        self.gru = nn.GRU(5, 256, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        pixels = x.reshape(B, x.size(1), -1)         # (B, D, 1024)
        q = torch.quantile(pixels, self.qs, dim=-1)  # (5, B, D)
        seq = q.permute(1, 2, 0)                     # (B, D, 5)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class InkDetectorDiffOfGaussians(nn.Module):
    """
    difference of Gaussians (DoG) filter applied to depth profile.
    DoG = Gaussian(sigma1) - Gaussian(sigma2) is the optimal blob detector.
    ink creates a bump (bell curve) in the depth absorption profile.
    DoG explicitly detects bumps at specific depth scales.
    multiple DoG pairs at different scales capture bumps of different widths.
    output = [DoG(1,2), DoG(2,4), DoG(4,8)] at all depth positions → MLP.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # 3 DoG scales: fine, medium, coarse
        sigmas = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
        self._kernels = []
        for s1, s2 in sigmas:
            # create 1D Gaussian kernels and store as buffers
            k_size = min(D, 7)
            coords = torch.arange(k_size, dtype=torch.float32) - k_size // 2
            g1 = torch.exp(-0.5 * (coords / s1) ** 2)
            g2 = torch.exp(-0.5 * (coords / s2) ** 2)
            g1 = g1 / g1.sum(); g2 = g2 / g2.sum()
            self._kernels.append((g1 - g2).unsqueeze(0).unsqueeze(0))  # (1,1,k)
        self.head = nn.Sequential(
            nn.Linear(D * 3, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        # register kernels as buffers
        for i, k in enumerate(self._kernels):
            self.register_buffer(f'dog_kernel_{i}', k)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1]).unsqueeze(1)  # (B, 1, D)
        feats = []
        for i in range(3):
            k = getattr(self, f'dog_kernel_{i}')
            p = k.size(-1) // 2
            feats.append(torch.nn.functional.conv1d(profile, k, padding=p).squeeze(1))
        return self.head(torch.cat(feats, dim=1))


class InkDetectorAbsorptionRatio(nn.Module):
    """
    physics-based: explicit ink-band vs pre-band absorption ratio.
    ink absorbs more X-rays at depths 32-40 than at baseline depths 20-28.
    ratio = mean(ink_band) / (mean(pre_band) + eps) is scale-invariant.
    plus: difference = ink_band - pre_band (differential absorption).
    these 2×8 features are the most direct physical measurement of ink.
    fed to a deep MLP to learn the exact pattern.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.mlp = nn.Sequential(
            nn.Linear(D + D, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        ink = x.mean(dim=[-2, -1])            # (B, D) spatial mean at each depth
        tile_mean = ink.mean(dim=1, keepdim=True)  # (B, 1)
        # centered profile: removes tile-wide brightness, keeps shape
        centered = ink - tile_mean            # (B, D)
        # ratio: each depth vs tile mean
        ratio = ink / (tile_mean.abs().clamp(min=1e-6))  # (B, D)
        feats = torch.cat([centered, ratio], dim=1)       # (B, D*2)
        return self.mlp(feats)


class InkDetectorSpatialContrast(nn.Module):
    """
    spatial contrast: how does this tile's depth profile compare to its
    spatial neighborhood? ink is spatially localized — a few tiles in a
    letter stroke will differ from the surrounding non-ink tiles.

    implemented as: for each depth, compute the deviation of this tile's
    mean from the tile's own local spatial mean profile (across 4 quadrants).
    then apply BiGRU to the contrast sequence through depth.

    note: uses only the 32x32 spatial data — no wider spatial context.
    the 4 quadrants (16x16 each) serve as the local reference.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # 4 quadrant profiles + full-tile profile → 5D features per depth
        self.gru = nn.GRU(5, 256, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        D, H, W = x.size(1), x.size(2), x.size(3)
        H2, W2 = H // 2, W // 2
        full = x.mean(dim=[-2, -1])                               # (B, D)
        q00  = x[:, :, :H2, :W2].mean(dim=[-2, -1])
        q01  = x[:, :, :H2, W2:].mean(dim=[-2, -1])
        q10  = x[:, :, H2:, :W2].mean(dim=[-2, -1])
        q11  = x[:, :, H2:, W2:].mean(dim=[-2, -1])
        # contrast: each quadrant relative to tile mean
        seq = torch.stack([full - full.mean(dim=1, keepdim=True),
                           q00 - full, q01 - full,
                           q10 - full, q11 - full], dim=-1)  # (B, D, 5)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class InkDetectorDeepBiGRU(nn.Module):
    """
    6-layer bidirectional GRU with 512 hidden on 8-depth spatial mean.
    C6's 1-layer BiGRU (hidden=256) was the best non-ring model (0.419).
    ring+BiGRU got 0.436. scaling depth: 6 layers × 512 hidden = 8× capacity.
    deep recurrent models can represent exponentially more complex sequential
    patterns. the depth absorption profile has at most ~8 meaningful values
    so depth receptive field is complete by layer 2; layers 3-6 refine.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.gru = nn.GRU(1, 512, num_layers=6, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(1024, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        out, _ = self.gru(profile.unsqueeze(-1))
        return self.head(out[:, -1])


class InkDetectorWaveletDepth(nn.Module):
    """
    discrete Haar wavelet transform of depth profile → approximation + details → MLP.
    wavelets decompose the signal into scale-specific components.
    low-frequency (approximation): overall absorption level.
    high-frequency (details): edges, sharp transitions = ink absorption edges.
    unlike FFT (global sinusoids), wavelets detect localized features.
    the ink absorption edge (rise at depth 32, fall at depth 40) is exactly
    the kind of localized transient feature wavelets are designed to detect.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # haar wavelet: averaging and difference filter
        h = [0.7071067811865476, 0.7071067811865476]
        g = [0.7071067811865476, -0.7071067811865476]
        self.register_buffer('h', torch.tensor(h).float().view(1, 1, -1))
        self.register_buffer('g', torch.tensor(g).float().view(1, 1, -1))
        # 2 levels of decomposition
        n_features = D // 2 + D // 4 + D // 4  # approx L2 + detail L2 + detail L1
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def _dwt1d(self, x):
        # x: (B, 1, L); returns (approx, detail) each (B, 1, L//2)
        a = torch.nn.functional.conv1d(x, self.h, stride=2, padding=0)
        d = torch.nn.functional.conv1d(x, self.g, stride=2, padding=0)
        return a, d

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        p = x.mean(dim=[-2, -1]).unsqueeze(1)  # (B, 1, D)
        # level 1
        a1, d1 = self._dwt1d(p)
        # level 2
        a2, d2 = self._dwt1d(a1)
        feats = torch.cat([a2.squeeze(1), d2.squeeze(1), d1.squeeze(1)], dim=1)
        return self.mlp(feats)


class InkDetectorPairwiseBiGRU(nn.Module):
    """
    pairwise depth differences → BiGRU.
    the C(D,2) pairwise differences (v7_pairwise_depth, hard=0.414) encode
    relative absorption across all depth pairs — scale-invariant.
    BiGRU processes these as a sequence (ordered by depth index pair).
    combines the noise-robustness of pairwise comparisons with the
    sequential pattern-matching capacity of recurrent models.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        n_pairs = D * (D - 1) // 2  # 28 for D=8
        self.gru = nn.GRU(1, 128, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
        self.register_buffer('pi', torch.tensor([p[0] for p in pairs]))
        self.register_buffer('pj', torch.tensor([p[1] for p in pairs]))

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])                          # (B, D)
        diffs = profile[:, self.pi] - profile[:, self.pj]       # (B, n_pairs)
        out, _ = self.gru(diffs.unsqueeze(-1))
        return self.head(out[:, -1])


class InkDetectorFullDepthTransformerDeep(nn.Module):
    """
    16-layer transformer on full 64-depth spatial-mean profile.
    C6 t08 (4-layer, hard=0.339) and C7 t14 (4-layer deep bigru, hard=0.399).
    16 layers of self-attention over 64 tokens: each depth position attends
    to all others simultaneously. much deeper than any prior transformer.
    with ring negatives (1:1 balanced), training should be stable.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        d = 256
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, D, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=16,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        tokens = self.embed(profile.unsqueeze(-1)) + self.pos
        tokens = self.transformer(tokens)
        return self.head(self.norm(tokens.mean(1)))


class InkDetectorTileEntropyDepth(nn.Module):
    """
    spatial entropy at each depth slice: H(d) = -sum(p * log(p)) of pixel
    value histogram. ink tiles are spatially heterogeneous (some pixels hit
    ink, others don't) → HIGH entropy. pure background is spatially uniform
    → LOW entropy. entropy explicitly measures the spatial heterogeneity
    that sub-voxel ink should create: only a few pixels are ink-like.
    entropy profile through depth → BiGRU.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.n_bins = 32
        self.gru = nn.GRU(2, 128, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def _soft_entropy(self, x_slice):
        # x_slice: (B, H, W) in [0, 1] → soft histogram entropy
        B = x_slice.size(0)
        pixels = x_slice.reshape(B, -1)  # (B, 1024)
        # soft binning: each bin center
        centers = torch.linspace(0, 1, self.n_bins, device=x_slice.device)
        width = 1.0 / self.n_bins
        dists = (pixels.unsqueeze(-1) - centers.unsqueeze(0).unsqueeze(0)).abs()
        weights = torch.clamp(1.0 - dists / width, min=0)
        hist = weights.sum(dim=1) / weights.sum(dim=[1, 2]).unsqueeze(-1).clamp(min=1e-6)
        p = hist.clamp(min=1e-6)
        return -(p * p.log()).sum(dim=-1)  # (B,)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        D = x.size(1)
        entropy = torch.stack([self._soft_entropy(x[:, d]) for d in range(D)], dim=1)  # (B, D)
        mean_profile = x.mean(dim=[-2, -1])  # (B, D)
        seq = torch.stack([mean_profile, entropy], dim=-1)  # (B, D, 2)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class InkDetectorRobustZScore(nn.Module):
    """
    z-score the depth profile against the PER-TILE robust statistics:
    z(d) = (x(d) - median(x)) / (IQR(x) + eps)
    where x is the 8-depth spatial-mean profile.
    this makes the model scale- and shift-invariant to scroll-wide brightness.
    the z-scored profile encodes the anomalous depth pattern ONLY.
    ink: z-score peaks at depths 32-40. background: flat z-score near zero.
    z-profile → deep 1D CNN → classifier.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(256, 0.1)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        # robust z-score: median and IQR
        med = profile.median(dim=1, keepdim=True).values
        q75 = torch.quantile(profile, 0.75, dim=1, keepdim=True)
        q25 = torch.quantile(profile, 0.25, dim=1, keepdim=True)
        iqr = (q75 - q25).clamp(min=1e-6)
        z = (profile - med) / iqr
        return self.head(self.pool(self.cnn(z.unsqueeze(1))))


class InkDetectorLaplacianDepth(nn.Module):
    """
    Laplacian of the depth profile: d2f/dx2 at each depth position.
    the second derivative is maximally sensitive to curvature peaks —
    exactly the shape that ink absorption creates (a bump).
    flat baseline → near-zero Laplacian. ink bump → large negative Laplacian
    at the peak. biphasic Laplacian flanking the ink peak.
    Laplacian + raw profile → 2-channel 1D CNN → classifier.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm1d(256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _slim_head(256, 0.1)

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        profile = x.mean(dim=[-2, -1])  # (B, D)
        # finite-difference Laplacian: pad endpoints
        p = profile.unsqueeze(1)  # (B, 1, D)
        p_pad = torch.nn.functional.pad(p, (1, 1), mode='reflect')
        lap = p_pad[:, :, 2:] - 2 * p_pad[:, :, 1:-1] + p_pad[:, :, :-2]
        feats = torch.cat([p, lap], dim=1)  # (B, 2, D)
        return self.head(self.pool(self.cnn(feats)))


class InkDetectorSuperPixelBiGRU(nn.Module):
    """
    4×4 superpixel mean profiles → BiGRU that processes the 16 superpixels
    as a sequence (raster order). each superpixel = 8×8 pixels = D-dim profile.
    the BiGRU learns spatial patterns in the superpixel arrangement:
    a letter stroke spanning ~2-4 superpixels creates a contiguous run of
    elevated profiles. the recurrent state accumulates this spatial pattern.
    different from t12 (superpixel transformer): recurrent instead of attention.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.proj = nn.Linear(D, 128)
        self.gru = nn.GRU(128, 256, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        D = x.size(1)
        # 4×4 superpixels of 8×8 each
        sp = x.reshape(B, D, 4, 8, 4, 8).mean(dim=[3, 5])  # (B, D, 4, 4)
        tokens = sp.permute(0, 2, 3, 1).reshape(B, 16, D)   # (B, 16, D)
        tokens = self.proj(tokens)                            # (B, 16, 128)
        out, _ = self.gru(tokens)
        return self.head(out[:, -1])


class InkDetectorMultiScalePercentile(nn.Module):
    """
    percentile features at 3 spatial scales: full 32×32, 2×2 quadrants, 4×4 cells.
    each scale provides 5 percentiles per depth position.
    scales: 1 full tile (5×D), 4 quadrants (4×5×D), 16 cells (16×5×D).
    total: (1+4+16) × 5 × D = 105D features.
    different scales are robust to different ink patch sizes:
    small patches → visible at 4×4 cell scale; large strokes → full tile scale.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        n = (1 + 4 + 16) * 5 * D
        self.mlp = nn.Sequential(
            nn.Linear(n, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def _pct(self, x):
        # x: (B, D, H, W) → (B, D*5) percentile features
        B, D = x.size(0), x.size(1)
        px = x.reshape(B, D, -1)
        q = torch.quantile(px, self.qs, dim=-1)  # (5, B, D)
        return q.permute(1, 2, 0).reshape(B, -1)  # (B, D*5)

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        D, H, W = x.size(1), x.size(2), x.size(3)
        # full tile
        f_full = self._pct(x)
        # 2×2 quadrants
        H2, W2 = H // 2, W // 2
        f_q = torch.cat([self._pct(x[:, :, :H2, :W2]),
                         self._pct(x[:, :, :H2, W2:]),
                         self._pct(x[:, :, H2:, :W2]),
                         self._pct(x[:, :, H2:, W2:])], dim=1)
        # 4×4 cells
        H4, W4 = H // 4, W // 4
        f_c = torch.cat([self._pct(x[:, :, i*H4:(i+1)*H4, j*W4:(j+1)*W4])
                         for i in range(4) for j in range(4)], dim=1)
        return self.mlp(torch.cat([f_full, f_q, f_c], dim=1))


class InkDetectorResidualSpatialDepth(nn.Module):
    """
    hybrid 3D CNN with residual blocks, specifically designed for spatial-depth
    joint features. unlike v1 (no skip connections), this has:
    - depth-separable convolutions (depthwise in space, pointwise in depth)
    - skip connections with learned projection
    - squeeze-excitation on depth dimension
    the idea: spatial features at each depth slice, then combine across depth
    with attention-weighted depth aggregation. more parameter-efficient than
    full 3D conv while capturing depth-varying spatial patterns.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # depth embedding: each z-slice → 64 spatial features
        self.slice_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),  # (B, 64, 4, 4)
        )
        # temporal (depth) attention over the D slice features
        self.depth_attn = nn.Sequential(
            nn.Linear(64 * 16, 256), nn.GELU(),
            nn.Linear(256, D), nn.Softmax(dim=-1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 16, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)  # (B, D, 32, 32)
        D = x.size(1)
        # encode each depth slice independently
        slices = []
        for d in range(D):
            s = self.slice_enc(x[:, d:d+1])    # (B, 64, 4, 4)
            slices.append(s.reshape(B, -1))     # (B, 1024)
        slices = torch.stack(slices, dim=1)     # (B, D, 1024)
        # depth attention
        attn = self.depth_attn(slices.mean(dim=1))  # (B, D)
        weighted = (attn.unsqueeze(-1) * slices).sum(dim=1)  # (B, 1024)
        return self.head(weighted)


class InkDetectorFull64PercentileBiGRU(nn.Module):
    """
    percentile-sequence BiGRU on full 64-depth profile.
    the 64-depth profile (fulldepth mode) has the complete ink absorption curve
    from baseline through peak through return. 5 percentiles per depth position
    gives the spatial distribution at each depth. 3-layer BiGRU processes this
    sequence. combines: (1) full absorption curve context, (2) percentile
    robustness, (3) recurrent depth modeling.
    requires input_mode='fulldepth'.
    """
    def __init__(self, config):
        super().__init__()
        D = config.data.depth  # 64 when fulldepth
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        self.gru = nn.GRU(5, 256, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.size(0)
        if x.dim() == 5: x = x.squeeze(1)
        pixels = x.reshape(B, x.size(1), -1)
        q = torch.quantile(pixels, self.qs, dim=-1)  # (5, B, D)
        seq = q.permute(1, 2, 0)                     # (B, D, 5)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


# ==== campaign 10: v10 helper classes ====

def _Conv3dBnRelu(in_c, out_c, k=3, **kw):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, bias=False, **kw),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )

def _Conv2dBnRelu(in_c, out_c, k=3, **kw):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, bias=False, **kw),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class _ResBlock1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
    def forward(self, x): return torch.relu(x + self.net(x))

class _CBAM3d(nn.Module):
    """channel + spatial attention for 3D feature maps"""
    def __init__(self, c, r=4):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(c, max(c // r, 4), bias=False), nn.ReLU(),
            nn.Linear(max(c // r, 4), c, bias=False), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel(x).view(x.size(0), x.size(1), 1, 1, 1)
        x = x * ca
        sa_in = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * self.spatial(sa_in)


# ==== campaign 10: original v10 architectures ====

class InkDetectorV1FullReg(InkDetector):
    """alias for v1 InkDetector — used as the campaign 10 baseline control."""
    pass


class InkDetector3DUNet(nn.Module):
    """3D U-Net: encoder downsamples spatially only (preserves depth via pool stride=(1,2,2)).
    skip connections. classification from bottleneck global avg pool."""
    def __init__(self, config):
        super().__init__()
        def dconv(i, o): return nn.Sequential(_Conv3dBnRelu(i, o, padding=1), _Conv3dBnRelu(o, o, padding=1))
        self.e1 = dconv(1, 32);   self.p1 = nn.MaxPool3d((1, 2, 2))
        self.e2 = dconv(32, 64);  self.p2 = nn.MaxPool3d((1, 2, 2))
        self.e3 = dconv(64, 128); self.p3 = nn.MaxPool3d((1, 2, 2))
        self.bn = dconv(128, 256)
        # decoder (representation quality; classification comes from bottleneck)
        self.up3 = nn.ConvTranspose3d(256, 128, (1,2,2), stride=(1,2,2))
        self.d3  = dconv(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, (1,2,2), stride=(1,2,2))
        self.d2  = dconv(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, (1,2,2), stride=(1,2,2))
        self.d1  = dconv(64, 32)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(256, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        e1 = self.e1(x); e2 = self.e2(self.p1(e1)); e3 = self.e3(self.p2(e2))
        bn = self.bn(self.p3(e3))
        d3 = self.d3(torch.cat([self.up3(bn), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        self.d1(torch.cat([self.up1(d2), e1], 1))  # decoder runs for gradient quality
        return self.head(bn)


class InkDetector3DUNetClassify(nn.Module):
    """3D U-Net where BOTH bottleneck AND full-res decoder output contribute to classification.
    bottleneck sees global context (ink stroke shape); full-res sees local voxel anomalies."""
    def __init__(self, config):
        super().__init__()
        def dconv(i, o): return nn.Sequential(_Conv3dBnRelu(i, o, padding=1), _Conv3dBnRelu(o, o, padding=1))
        self.e1 = dconv(1, 16);  self.p1 = nn.MaxPool3d((1, 2, 2))
        self.e2 = dconv(16, 32); self.p2 = nn.MaxPool3d((1, 2, 2))
        self.bn = dconv(32, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, (1,2,2), stride=(1,2,2))
        self.d2  = dconv(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, (1,2,2), stride=(1,2,2))
        self.d1  = dconv(32, 16)
        self.head_bn  = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, 32))
        self.head_dec = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(16, 32))
        self.cls = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        e1 = self.e1(x); e2 = self.e2(self.p1(e1))
        bn = self.bn(self.p2(e2))
        d2 = self.d2(torch.cat([self.up2(bn), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.cls(torch.cat([self.head_bn(bn), self.head_dec(d1)], 1))


class InkDetectorDepthSlice2DGRU(nn.Module):
    """2D CNN per depth slice (weight-shared across depth) → BiGRU across depth.
    preserves full 32x32 spatial resolution at each depth before fusion."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, padding=1),
            _Conv2dBnRelu(32, 64, padding=1),
            nn.AdaptiveAvgPool2d(2),   # (B, 64, 2, 2) = 256 features
        )
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]
        seq = torch.stack(slices, dim=1)   # (B, D, 256)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class InkDetectorDeep3DCBAM(nn.Module):
    """4-block deep 3D ResNet with CBAM at every block. v1 only applies CBAM at the end.
    CBAM at every block maintains sharp spatial focus throughout the network."""
    def __init__(self, config):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                _Conv3dBnRelu(i, o, padding=1),
                _Conv3dBnRelu(o, o, padding=1),
                _CBAM3d(o),
            )
        self.b1 = block(1,  32);  self.p1 = nn.MaxPool3d(2, padding=0, ceil_mode=True)
        self.b2 = block(32, 64);  self.p2 = nn.MaxPool3d(2, padding=0, ceil_mode=True)
        self.b3 = block(64, 128); self.p3 = nn.MaxPool3d(2, padding=0, ceil_mode=True)
        self.b4 = block(128, 256)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        x = self.p1(self.b1(x)); x = self.p2(self.b2(x))
        x = self.p3(self.b3(x)); x = self.b4(x)
        return self.head(x)


class InkDetectorLocalStatPool(nn.Module):
    """3D CBAM CNN with AdaptiveAvgPool3d(2,4,4) instead of global avg pool.
    preserves local spatial structure — each pooling cell covers ~8x8 pixels (~63um)."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, padding=1),
            _Conv3dBnRelu(32, 64, padding=1),
            _CBAM3d(64),
            _Conv3dBnRelu(64, 64, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool3d((2, 4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 4 * 4, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        return self.head(self.pool(self.net(x)))


class InkDetectorSpatialDepthCrossAttn(nn.Module):
    """2D CNN per slice → cross-depth Transformer at each spatial position → MIL.
    each (x,y) attends across D=8 depth tokens to detect its depth absorption signature."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 16, padding=1),
            nn.AdaptiveAvgPool2d(8),   # 8×8 spatial positions
        )  # (B, 16, 8, 8) per slice
        self.pos = nn.Embedding(D, 16)
        enc = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=64,
                                         dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]) for d in range(D)]  # (B, 16, 8, 8)
        # (B, 16, 8, 8) → stack to (B, D, 16, 8, 8) → reshape to (B*64, D, 16)
        feats = torch.stack(slices, dim=1)  # (B, D, 16, 8, 8)
        B2, D2, C, h, w = feats.shape
        feats = feats.permute(0, 3, 4, 1, 2).reshape(B2 * h * w, D2, C)
        pos = self.pos(torch.arange(D2, device=x.device))
        feats = feats + pos.unsqueeze(0)
        out = self.transformer(feats)       # (B*64, D, 16)
        out = out.mean(1)                   # (B*64, 16)
        scores = self.head(out)             # (B*64, 1)
        return scores.reshape(B2, h * w).max(dim=1, keepdim=True).values  # MIL max


class InkDetectorMultiScale3D(nn.Module):
    """3 parallel 3D CNN branches (full 32x32, 16x16, 8x8) + cross-scale attention."""
    def __init__(self, config):
        super().__init__()
        def branch(c): return nn.Sequential(
            _Conv3dBnRelu(1, c, padding=1),
            _Conv3dBnRelu(c, c, padding=1),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.br_full    = branch(32)
        self.br_half    = branch(32)
        self.br_quarter = branch(32)
        self.attn   = nn.Linear(32 * 3, 3)
        self.head   = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, C, D, H, W = x.shape
        f_full    = self.br_full(x)
        f_half    = self.br_half(torch.nn.functional.avg_pool3d(x, (1, 2, 2), stride=(1, 2, 2)))
        f_quarter = self.br_quarter(torch.nn.functional.avg_pool3d(x, (1, 4, 4), stride=(1, 4, 4)))
        combined  = torch.stack([f_full, f_half, f_quarter], dim=1)  # (B, 3, 32)
        weights   = torch.softmax(self.attn(combined.flatten(1)), dim=1).unsqueeze(-1)
        fused     = (weights * combined).sum(dim=1)                   # (B, 32)
        return self.head(fused)


class InkDetectorPerPixelDepthAttn(nn.Module):
    """1024 per-pixel depth profiles → lightweight encoder → Transformer spatial attention.
    each pixel's depth profile is encoded; then spatial attention discovers correlations."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # encode each pixel's depth profile D→32→64
        self.pixel_enc = nn.Sequential(
            _ResBlock1D(D),
            nn.Linear(D, 32), nn.GELU(),
            nn.Linear(32, 64),
        )
        # transformer over 1024 spatial tokens (one per pixel)
        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                         dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        pixels = x.squeeze(1).reshape(B, D, H * W).permute(0, 2, 1)  # (B, 1024, D)
        enc = self.pixel_enc(pixels)        # (B, 1024, 64)
        out = self.transformer(enc)         # (B, 1024, 64)
        return self.head(out.mean(dim=1))   # MIL mean → (B, 1)


_ARCH_MAP.update({
    "v10_v1_full_reg":          InkDetectorV1FullReg,
    "v10_3d_unet":              InkDetector3DUNet,
    "v10_3d_unet_classify":     InkDetector3DUNetClassify,
    "v10_depth_slice_2d_gru":   InkDetectorDepthSlice2DGRU,
    "v10_deep_3d_cbam":         InkDetectorDeep3DCBAM,
    "v10_local_stat_pool":      InkDetectorLocalStatPool,
    "v10_spatial_depth_xattn":  InkDetectorSpatialDepthCrossAttn,
    "v10_multiscale_3d":        InkDetectorMultiScale3D,
    "v10_perpixel_depth_attn":  InkDetectorPerPixelDepthAttn,
})


# ==== campaign 10 extended: sub-voxel + depth sequential, individual and in tandem ====

class InkDetectorMaxDepthPool(nn.Module):
    """3D CBAM CNN with global MAX pool instead of avg across the depth dim.
    avg collapses a 1-voxel ink spike to 1/8 its value. max preserves it.
    separates spatial avg (correct for noise) from depth max (correct for sub-voxel)."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # avg over spatial, keep depth
        self.depth_max    = nn.AdaptiveMaxPool1d(1)              # max over depth
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.net(x)                                 # (B, 128, D, 1, 1) after spatial_pool
        f = self.spatial_pool(f).squeeze(-1).squeeze(-1)  # (B, 128, D)
        f = self.depth_max(f).squeeze(-1)               # (B, 128)
        return self.head(f)


class InkDetectorTopKDepth(nn.Module):
    """3D CNN + top-k mean pool over depth. less extreme than hard max,
    more robust to noise than avg. k=3 averages the 3 highest-absorbing depth slices."""
    def __init__(self, config):
        super().__init__()
        self.k = 3
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.net(x)
        f = self.spatial_pool(f).squeeze(-1).squeeze(-1)  # (B, 128, D)
        topk = torch.topk(f, min(self.k, f.size(-1)), dim=-1).values  # (B, 128, k)
        f = topk.mean(dim=-1)                             # (B, 128)
        return self.head(f)


class InkDetectorDeepBiGRUSlice(nn.Module):
    """2D CNN per depth slice → stacked 3-layer BiGRU across depth.
    deeper than t04 (1-layer) — more capacity to model complex depth profiles.
    wider hidden dim (384) to capture richer sequential patterns."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(4),
        )
        self.gru = nn.GRU(64 * 16, 384, num_layers=3, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]  # list of (B, 1024)
        seq = torch.stack(slices, dim=1)  # (B, D, 1024)
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


class InkDetectorTCNDepth(nn.Module):
    """dilated temporal conv network (TCN) across depth per spatial position.
    per-slice 2D CNN → spatial mean → (B, D, C) → TCN with exponentially growing dilation.
    TCN has receptive field 2^n vs GRU's O(n) — captures global depth context in O(log n) layers."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        # dilated 1D convs: dilation 1, 2, 4 → receptive field = 7 (> D=8)
        self.tcn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1,  dilation=1), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]
        seq = torch.stack(slices, dim=1).transpose(1, 2)  # (B, 64, D)
        out = self.tcn(seq)                                # (B, 128, D)
        return self.head(out.max(dim=-1).values)           # max over depth → sub-voxel


class InkDetectorDepthTransformerMax(nn.Module):
    """per-slice 2D CNN spatial mean → full D-token transformer → max-pool over tokens.
    attention lets every depth slice attend to every other. max-pool over tokens
    preserves the single most anomalous depth slice rather than averaging all."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pos = nn.Embedding(D, 64)
        enc_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                                               dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]
        seq = torch.stack(slices, dim=1)                      # (B, D, 64)
        pos = self.pos(torch.arange(D, device=x.device))
        seq = seq + pos.unsqueeze(0)
        out = self.transformer(seq)                           # (B, D, 64)
        return self.head(out.max(dim=1).values)               # max over D tokens


class InkDetectorGRUMaxPool(nn.Module):
    """slice 2D CNN → BiGRU across depth → MAX over spatial positions for final score.
    the 'tandem' core: BiGRU captures the depth profile pattern,
    max-over-space preserves the single hottest spatial position."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        T = config.data.tile_size
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
        )  # output: (B, 64, H, W) per slice
        self.gru = nn.GRU(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        # process each spatial position independently across depth
        slices = [self.slice_cnn(x[:, :, d]) for d in range(D)]  # list of (B, 64, H, W)
        # flatten spatial: each position gets a depth sequence
        feats = torch.stack(slices, dim=2)  # (B, 64, D, H, W)
        feats = feats.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, 64)  # (B*H*W, D, 64)
        out, _ = self.gru(feats)            # (B*H*W, D, 256)
        out = out[:, -1]                    # (B*H*W, 256)
        out = out.reshape(B, H * W, 256)
        out = out.max(dim=1).values         # (B, 256) — max over spatial positions
        return self.head(out)


class InkDetectorPercentileTCN(nn.Module):
    """percentile features (sub-voxel robust) → TCN (depth sequential).
    tandem: percentile at each depth captures sparse ink better than mean;
    TCN models the depth profile shape with dilated convolutions."""
    def __init__(self, config):
        super().__init__()
        self.register_buffer('qs', torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        self.tcn = nn.Sequential(
            nn.Conv1d(7, 64,  kernel_size=3, padding=1,  dilation=1), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        pixels = x.squeeze(1).reshape(B, D, -1)           # (B, D, H*W)
        q = torch.quantile(pixels, self.qs, dim=-1)       # (7, B, D)
        seq = q.permute(1, 0, 2)                           # (B, 7, D) — channels=percentiles, len=depth
        out = self.tcn(seq)                                # (B, 128, D)
        return self.head(out.max(dim=-1).values)


class InkDetectorAsymmetricPool(nn.Module):
    """3D CBAM CNN with asymmetric pooling: avg over spatial (H,W), MAX over depth (D).
    the two axes have different optimal pooling strategies:
    - spatial: avg is correct (ink texture is spread, not a single hot pixel spatially)
    - depth:   max is correct (ink absorption spike at 1-2 depth slices, not all 8)"""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
            _Conv3dBnRelu(128, 256, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, 256, D, 1, 1)
        self.depth_max   = nn.AdaptiveMaxPool1d(1)             # (B, 256, 1)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.spatial_avg(self.net(x)).squeeze(-1).squeeze(-1)  # (B, 256, D)
        f = self.depth_max(f).squeeze(-1)                          # (B, 256)
        return self.head(f)


class InkDetectorSparseDepthAttn(nn.Module):
    """sparse top-k attention over depth: only the k most anomalous depth positions
    contribute to the classification. soft-top-k via temperature-scaled softmax.
    per-slice 2D CNN → (B, D, C) → anomaly score per depth → weighted sum.
    explicitly learns to ignore normal depths and focus on the ink spike."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.k = 3
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        # produces an anomaly score per depth slice
        self.scorer = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]
        seq = torch.stack(slices, dim=1)                   # (B, D, 64)
        scores = self.scorer(seq).squeeze(-1)              # (B, D)
        # soft-top-k: sharpen attention with temperature, keep top k
        topk_mask = torch.zeros_like(scores)
        idx = scores.topk(min(self.k, D), dim=-1).indices
        topk_mask.scatter_(-1, idx, 1.0)
        attn = torch.softmax(scores / self.temp.clamp(min=0.1), dim=-1) * topk_mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        fused = (attn.unsqueeze(-1) * seq).sum(dim=1)     # (B, 64)
        return self.head(fused)


class InkDetectorHierarchicalGRU(nn.Module):
    """hierarchical BiGRU: local (2-slice window) → global (across windows).
    local BiGRU detects short-range depth transitions (ink onset/offset).
    global BiGRU detects the overall depth profile shape.
    two-level temporal reasoning captures both fine and coarse depth structure."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        self.window = 2
        self.n_windows = D // self.window
        self.slice_cnn = nn.Sequential(
            _Conv2dBnRelu(1, 32, 3, padding=1),
            _Conv2dBnRelu(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.local_gru  = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.global_gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        slices = [self.slice_cnn(x[:, :, d]).flatten(1) for d in range(D)]
        seq = torch.stack(slices, dim=1)  # (B, D, 64)
        # local BiGRU over windows of size 2
        windows = seq.reshape(B * self.n_windows, self.window, 64)
        local_out, _ = self.local_gru(windows)  # (B*n_win, window, 128)
        local_summary = local_out[:, -1].reshape(B, self.n_windows, 128)  # (B, n_win, 128)
        # global BiGRU over window summaries
        global_out, _ = self.global_gru(local_summary)  # (B, n_win, 256)
        return self.head(global_out[:, -1])              # (B, 256) → (B, 1)


_ARCH_MAP.update({
    "v10_max_depth_pool":        InkDetectorMaxDepthPool,
    "v10_topk_depth":            InkDetectorTopKDepth,
    "v10_deep_bigru_slice":      InkDetectorDeepBiGRUSlice,
    "v10_tcn_depth":             InkDetectorTCNDepth,
    "v10_depth_transformer_max": InkDetectorDepthTransformerMax,
    "v10_gru_maxpool":           InkDetectorGRUMaxPool,
    "v10_percentile_tcn":        InkDetectorPercentileTCN,
    "v10_asymmetric_pool":       InkDetectorAsymmetricPool,
    "v10_sparse_depth_attn":     InkDetectorSparseDepthAttn,
    "v10_hierarchical_gru":      InkDetectorHierarchicalGRU,
})



# ==== campaign 11: conv-stem + attention hybrids ====

class InkDetectorConvStemDepthAttn(nn.Module):
    """3D conv stem (v1 blocks 1-2) → single thin transformer over depth tokens.
    convolution builds spatially grounded features first; attention then resolves
    cross-depth relationships in feature space where S/N is already high.
    the attention window is post-pool (D=2, H=4, W=4) so N=32 tokens — tractable."""
    def __init__(self, config):
        super().__init__()
        # identical to v1 blocks 1-2 (conv→CBAM→conv→CBAM→pool)
        dil = max(1, int(getattr(config.model, "conv3_dilation", 1)))
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, (3, 4, 4), padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(2),  # (B, 128, ~4, 15, 15)
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.MaxPool3d(2),  # (B, 256, ~2, 7, 7)
        )
        # project each spatial-depth voxel to a compact token
        self.proj = nn.Conv3d(256, 64, 1, bias=False)   # (B, 64, D', H', W')
        d_model = 64
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=0.0, batch_first=True)
        self.attn = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.proj(self.stem(x))                         # (B, 64, D', H', W')
        B, C, D, H, W = f.shape
        tokens = f.flatten(2).permute(0, 2, 1)             # (B, D'*H'*W', 64)
        out = self.attn(tokens)                             # (B, N, 64)
        return self.head(out.mean(dim=1))                   # mean over tokens → (B, 1)


class InkDetectorConvStemSE(nn.Module):
    """v1 3D conv backbone with SE (squeeze-excitation) blocks instead of CBAM.
    SE does channel recalibration only — lighter than CBAM's spatial attention,
    more stable under noisy labels. tests whether spatial attention is net-negative."""
    def __init__(self, config):
        super().__init__()
        d1 = config.model.conv1_drop
        d2 = config.model.conv2_drop
        dil = max(1, int(getattr(config.model, "conv3_dilation", 1)))
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, (3, 4, 4), padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            SE3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            SE3D(128),
            nn.MaxPool3d(2),
            nn.Dropout3d(d1),
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            SE3D(256),
            nn.MaxPool3d(2),
            nn.Dropout3d(d2),
        )
        self.pool = _pool_layer(config)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc2_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        return self.classifier(self.pool(self.features(x)))


class InkDetectorConvStemNonLocal(nn.Module):
    """v1 3D conv stem + single non-local block after block 2, before the final pool.
    non-local block lets each voxel in the feature volume attend to every other —
    capturing long-range spatial-depth correlations in learned feature space, not raw voxels.
    inserted at (B, 256, ~2, 7, 7) so N=98 tokens, manageable quadratic attention."""
    def __init__(self, config):
        super().__init__()
        d1 = config.model.conv1_drop
        d2 = config.model.conv2_drop
        dil = max(1, int(getattr(config.model, "conv3_dilation", 1)))
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 4, 4), padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            CBAM3D(32),
            nn.Conv3d(32, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(2),
            nn.Dropout3d(d1),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.MaxPool3d(2),
            nn.Dropout3d(d2),
        )
        # non-local after the last conv stage, on compact (B, 256, ~2, 7, 7) feature maps
        self.nonlocal_block = NonLocal3D(256)
        self.pool = _pool_layer(config)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),
            nn.Linear(128, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc2_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        x = self.nonlocal_block(x)
        return self.classifier(self.pool(x))


_ARCH_MAP.update({
    "v11_conv_stem_depth_attn": InkDetectorConvStemDepthAttn,
    "v11_conv_stem_se":         InkDetectorConvStemSE,
    "v11_conv_stem_nonlocal":   InkDetectorConvStemNonLocal,
})


# ==== campaign 12: soft depth attention pooling ====

class InkDetectorAsymAttnPool(nn.Module):
    """asym_pool with learned soft depth attention instead of hard depth-max.
    backbone identical to v10_asymmetric_pool (4x conv3d, spatial avg → (B,256,D));
    a small 1D conv attention head replaces the hard AdaptiveMaxPool1d with
    a softmax-weighted sum over depth positions.
    motivation: ink absorption may be a smooth 2-3 slice bump rather than a sharp single
    spike; soft attention can learn the exact aggregation shape from data rather than
    committing to the hardest single slice upfront."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
            _Conv3dBnRelu(128, 256, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, 256, D, 1, 1)
        # 1D attention: (B,256,D) → (B,1,D) attention logits via 1D conv
        self.depth_attn = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.spatial_avg(self.net(x)).squeeze(-1).squeeze(-1)  # (B, 256, D)
        attn = torch.softmax(self.depth_attn(f), dim=-1)           # (B, 1, D)
        fused = (f * attn).sum(dim=-1)                             # (B, 256)
        return self.head(fused)


_ARCH_MAP.update({
    "v12_asym_attn_pool": InkDetectorAsymAttnPool,
})


# ==== campaign 13: depth-profile focus, ring training, diverse input modes ====


class InkDetectorFullDepthConv1DRing(nn.Module):
    """v13_fulldepth_conv1d_ring: full depth profile (fulldepth mode) → deep 1D CNN.
    unlike v6_fulldepth_1d which averages spatial first, this conv3d spatially
    reduces first (preserving per-pixel depth resolution) then applies 1D conv on
    the remaining depth axis. uses conv3d with (D,1,1) kernels — purely depth-axis
    convolutions — forcing the model to learn depth profiles without spatial mixing.
    designed for ring training: ring provides boundary diversity, depth axis is signal."""
    def __init__(self, config):
        super().__init__()
        # reduce spatial with pointwise 3D convs, leaving full depth intact
        self.spatial_reduce = nn.Sequential(
            nn.Conv3d(1, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # (B, 32, D, 1, 1)
        )
        # 1D CNN along the full depth axis
        self.depth_cnn = nn.Sequential(
            nn.Conv1d(32, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),  # max over depth: takes the spike
        )
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.spatial_reduce(x).squeeze(-1).squeeze(-1)  # (B, 32, D)
        f = self.depth_cnn(f)                               # (B, 128, 1)
        return self.head(f)


class InkDetectorDepthDiffConv(nn.Module):
    """v13_depth_diff_conv: explicit differential input (diff mode) + asym_pool backbone.
    diff mode computes clip(ink_band - pre_band, 0) — the positive absorption anomaly.
    a normal (non-inked) pixel should have ~zero diff; ink pixels show a spike.
    asym_pool backbone (spatial-avg → depth-max) then isolates the strongest depth
    position in the already-differential signal. tests if the subtraction is enough
    to bring hard ink above the detection threshold."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
            _Conv3dBnRelu(128, 256, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.depth_max   = nn.AdaptiveMaxPool1d(1)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.spatial_avg(self.net(x)).squeeze(-1).squeeze(-1)  # (B, 256, D)
        f = self.depth_max(f).squeeze(-1)                          # (B, 256)
        return self.head(f)


class InkDetectorTripleBandStem(nn.Module):
    """v13_triple_band_stem: triple input (pre+ink+post, 24 slices) → factorized 3D conv.
    triple mode provides the full absorption context: background level (pre),
    ink band (ink), and recovery (post). factorized depth-then-spatial convs
    process the 24-slice volume so the model can explicitly learn cross-band
    relationships (how does absorption change from pre → ink → post?).
    if ink is present, the ink band should stand out against both flanking bands."""
    def __init__(self, config):
        super().__init__()
        # 24 input channels = 3 bands × 8 depth slices each
        depth_in = config.data.depth * 3

        def _fact(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, (3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                nn.Conv3d(o, o, (1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(o), nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            _fact(1, 32),
            _fact(32, 64),
            nn.MaxPool3d((2, 2, 2)),
            _fact(64, 128),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.depth_max   = nn.AdaptiveMaxPool1d(1)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        # triple input is (B, 24, H, W) — reshape to (B, 1, 24, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 24, H, W)
        f = self.spatial_avg(self.stem(x)).squeeze(-1).squeeze(-1)  # (B, 128, D')
        f = self.depth_max(f).squeeze(-1)                           # (B, 128)
        return self.head(f)


class InkDetectorDepthDeltaTransformer(nn.Module):
    """v13_depth_delta_transformer: per-slice CNN → token sequence → transformer.
    unlike spatial transformers (which fail), this operates over DEPTH tokens only.
    each depth slice is embedded by a shared tiny 2D CNN into a 64-d token;
    tokens are fed to a transformer that learns which depth positions are anomalous.
    key: the transformer sees the full sequence of depth embeddings, not just raw voxels.
    conv pre-processing ensures features are meaningful before attention is applied
    (C11 lesson: attention on raw voxels fails; on conv features it may succeed)."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        d_model = 64
        # shared 2D CNN embeds each depth slice to a vector
        self.slice_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # (B, d_model)
        )
        drop = config.model.fc1_drop
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=drop, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        # depth position embedding: allocated for up to 128 slices so fulldepth
        # mode (64 slices) works without re-init; forward slices to actual D
        _max_d = 128
        self.pos = nn.Parameter(torch.zeros(1, _max_d, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)          # (B, 1, D, H, W)
        B, _, D, H, W = x.shape
        slices = x[:, 0]                               # (B, D, H, W)
        # embed each slice independently: (B*D, 1, H, W) → (B*D, d_model)
        s = slices.reshape(B * D, 1, H, W)
        tok = self.slice_embed(s).reshape(B, D, -1)   # (B, D, d_model)
        tok = tok + self.pos[:, :D]
        out = self.transformer(tok)                    # (B, D, d_model)
        # use max over depth instead of mean: picks the most anomalous slice
        logit = self.head(out.max(dim=1).values)       # (B, 1)
        return logit


class InkDetectorDepthProfileMIL(nn.Module):
    """v13_depth_profile_mil: per-pixel depth profiles → gated MIL over spatial positions.
    each pixel has an 8-slice depth profile; a shared 1D CNN embeds it to a feature.
    gated MIL (Ilse 2018) then aggregates: the gate learns which spatial positions
    show ink-like depth profiles and weights them accordingly.
    explicitly models: 'which pixels in this 32x32 tile have the ink depth signature?'
    key difference from old per-pixel models: the MIL gate is depth-profile-driven,
    not spatial-texture-driven."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        feat_dim = 64
        # 1D CNN per pixel's depth profile
        self.profile_enc = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),  # max = spike detection
        )
        # gated MIL attention
        self.gate_V = nn.Linear(feat_dim, 32, bias=False)
        self.gate_U = nn.Linear(feat_dim, 32, bias=False)
        self.gate_w = nn.Linear(32, 1, bias=False)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(feat_dim, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        N = H * W
        # reshape: each pixel's depth profile → (B*N, 1, D)
        pix = x[:, 0].permute(0, 2, 3, 1).reshape(B * N, 1, D)  # (B*N, 1, D)
        feat = self.profile_enc(pix).squeeze(-1)                  # (B*N, feat_dim)
        feat = feat.reshape(B, N, -1)                             # (B, N, feat_dim)
        # gated MIL: a = softmax(w^T tanh(V h) ⊙ sigmoid(U h))
        a = torch.tanh(self.gate_V(feat)) * torch.sigmoid(self.gate_U(feat))  # (B, N, 32)
        a = torch.softmax(self.gate_w(a), dim=1)                  # (B, N, 1)
        pooled = (a * feat).sum(dim=1)                             # (B, feat_dim)
        return self.head(pooled)


class InkDetectorDepthContrastConv(nn.Module):
    """v13_depth_contrast_conv: double input (ink+pre concatenated depth-wise) → asym_pool.
    the model receives 16 slices: first 8 = ink band, next 8 = pre-ink background.
    a 3D conv processes them jointly; the network can learn to compare the two bands
    directly (difference, ratio, anomaly detection). the depth-max pool then finds
    the single most discriminative ink-vs-background depth position.
    'double' input_mode = concat([ink, pre], axis=0) = (16, H, W)."""
    def __init__(self, config):
        super().__init__()
        # input: (B, 1, 16, H, W) — 16-slice volume: [ink | pre]
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
            _Conv3dBnRelu(128, 256, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.depth_max   = nn.AdaptiveMaxPool1d(1)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)  # (B, 1, 16, H, W)
        f = self.spatial_avg(self.net(x)).squeeze(-1).squeeze(-1)  # (B, 256, D)
        f = self.depth_max(f).squeeze(-1)                          # (B, 256)
        return self.head(f)


class InkDetectorCrossDepthAttn(nn.Module):
    """v13_cross_depth_attn: 3D conv reduces spatial, then cross-depth attention.
    key difference from v11_conv_stem_depth_attn: the query is the depth-max position
    (the strongest response), which attends to all other depth positions.
    this implements: 'given the most activated depth slice, what do neighboring slices
    confirm or deny about its being ink?' — a targeted cross-depth verification step.
    targets the C11 finding that inter-layer attention on conv features (not raw voxels)
    is where the signal lives."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # 3D conv stem (same as asym_pool backbone)
        self.stem = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, 128, D, 1, 1)
        d_feat = 128
        # cross-depth attention: Q from max-depth position, K/V from all positions
        self.q_proj = nn.Linear(d_feat, 32, bias=False)
        self.k_proj = nn.Linear(d_feat, 32, bias=False)
        self.v_proj = nn.Linear(d_feat, 64, bias=False)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))
        self.scale = 32 ** -0.5

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.spatial_avg(self.stem(x)).squeeze(-1).squeeze(-1)  # (B, 128, D)
        f = f.permute(0, 2, 1)                                      # (B, D, 128)
        # query = the depth position with maximum L2 norm (most activated)
        norms = f.norm(dim=-1)                                       # (B, D)
        max_idx = norms.argmax(dim=1, keepdim=True)                  # (B, 1)
        q = f.gather(1, max_idx.unsqueeze(-1).expand(-1, -1, f.shape[-1]))  # (B, 1, 128)
        q = self.q_proj(q)                                           # (B, 1, 32)
        k = self.k_proj(f)                                           # (B, D, 32)
        v = self.v_proj(f)                                           # (B, D, 64)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)  # (B, 1, D)
        out = (attn @ v).squeeze(1)                                  # (B, 64)
        return self.head(out)


class InkDetectorDepthGradient(nn.Module):
    """v13_depth_gradient: explicit depth derivative (first-order gradient along D) as input.
    ink absorption creates a sharp transition in the depth profile: a steep rise at the
    ink layer boundary and steep fall at exit. the gradient of the depth profile captures
    these transitions directly — turning a soft peak into a sharp positive-then-negative
    step function that should be easier to detect than the peak itself.
    applied to each pixel independently, then spatially reduced via asym_pool."""
    def __init__(self, config):
        super().__init__()
        # process gradient volume: (B, 1, D-1, H, W)
        self.net = nn.Sequential(
            _Conv3dBnRelu(1, 32, 3, padding=1),
            _Conv3dBnRelu(32, 64, 3, padding=1),
            _Conv3dBnRelu(64, 128, 3, padding=1),
        )
        self.spatial_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.depth_max   = nn.AdaptiveMaxPool1d(1)
        drop = config.model.fc1_drop
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        # compute depth gradient: x[:, :, 1:] - x[:, :, :-1]
        grad = x[:, :, 1:] - x[:, :, :-1]              # (B, 1, D-1, H, W)
        f = self.spatial_avg(self.net(grad)).squeeze(-1).squeeze(-1)  # (B, 128, D-1)
        f = self.depth_max(f).squeeze(-1)               # (B, 128)
        return self.head(f)


class InkDetectorSliceContrastBiGRU(nn.Module):
    """v13_slice_contrast_bigru: per-slice spatial-max → contrast against running mean → BiGRU.
    for each depth slice d: feature = max_pool(slice_d) - running_mean(slices_0..d-1).
    this creates a sequence of 'how much does this slice deviate from the local baseline?'
    the BiGRU then learns the temporal pattern of those deviations — ink should produce
    a clear spike-deviation pattern at specific depth positions.
    spatial max (not mean) ensures a single ink pixel can dominate the slice feature."""
    def __init__(self, config):
        super().__init__()
        D = config.data.depth
        # per-slice 2D processing
        self.slice_embed = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),  # (B, 32)
        )
        # BiGRU processes the deviation sequence
        drop = config.model.fc1_drop
        self.gru = nn.GRU(32, 64, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=drop if drop > 0 else 0.0)
        self.head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape
        # embed each slice
        slices = x[:, 0]  # (B, D, H, W)
        feats = []
        for d in range(D):
            feats.append(self.slice_embed(slices[:, d:d+1]))  # (B, 32)
        seq = torch.stack(feats, dim=1)  # (B, D, 32)
        # running mean contrast: subtract cumulative mean up to slice d
        cum_sum = seq.cumsum(dim=1)
        counts = torch.arange(1, D + 1, device=x.device).float().view(1, D, 1)
        running_mean = cum_sum / counts
        # shift by 1 so we subtract the mean *before* slice d
        baseline = torch.cat([torch.zeros(B, 1, seq.shape[-1], device=x.device),
                               running_mean[:, :-1]], dim=1)
        contrast_seq = seq - baseline                          # (B, D, 32)
        out, _ = self.gru(contrast_seq)                        # (B, D, 128)
        # take the max-norm hidden state (most deviant depth position)
        norms = out.norm(dim=-1)                               # (B, D)
        best = out[torch.arange(B), norms.argmax(dim=1)]      # (B, 128)
        return self.head(best)


_ARCH_MAP.update({
    "v13_fulldepth_conv1d_ring":  InkDetectorFullDepthConv1DRing,
    "v13_depth_diff_conv":        InkDetectorDepthDiffConv,
    "v13_triple_band_stem":       InkDetectorTripleBandStem,
    "v13_depth_delta_transformer":InkDetectorDepthDeltaTransformer,
    "v13_depth_profile_mil":      InkDetectorDepthProfileMIL,
    "v13_depth_contrast_conv":    InkDetectorDepthContrastConv,
    "v13_cross_depth_attn":       InkDetectorCrossDepthAttn,
    "v13_depth_gradient":         InkDetectorDepthGradient,
    "v13_slice_contrast_bigru":   InkDetectorSliceContrastBiGRU,
})


_ARCH_MAP.update({
    "v8_matched_filter":          InkDetectorMatchedFilter,
    "v8_percentile_bigru":        InkDetectorPercentileBiGRU,
    "v8_diff_of_gaussians":       InkDetectorDiffOfGaussians,
    "v8_absorption_ratio":        InkDetectorAbsorptionRatio,
    "v8_spatial_contrast":        InkDetectorSpatialContrast,
    "v8_deep_bigru":              InkDetectorDeepBiGRU,
    "v8_wavelet_depth":           InkDetectorWaveletDepth,
    "v8_pairwise_bigru":          InkDetectorPairwiseBiGRU,
    "v8_fulldepth_transformer16": InkDetectorFullDepthTransformerDeep,
    "v8_tile_entropy":            InkDetectorTileEntropyDepth,
    "v8_robust_zscore":           InkDetectorRobustZScore,
    "v8_laplacian_depth":         InkDetectorLaplacianDepth,
    "v8_superpixel_bigru":        InkDetectorSuperPixelBiGRU,
    "v8_multiscale_percentile":   InkDetectorMultiScalePercentile,
    "v8_residual_spatial_depth":  InkDetectorResidualSpatialDepth,
    "v8_full64_pct_bigru":        InkDetectorFull64PercentileBiGRU,
})


def create_model(config: Config):
    """create and initialize the model, dispatching on config.model.arch"""
    arch = str(getattr(config.model, "arch", "v1")).lower()
    if arch not in _ARCH_MAP:
        raise ValueError(f"unknown arch '{arch}'; valid options: {sorted(_ARCH_MAP)}")
    model = _ARCH_MAP[arch](config).to(config.device)

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.numel(p) for p in model_parameters])
    print(f"Model parameters ({arch}): {params:,}")

    return model, params