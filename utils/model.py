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