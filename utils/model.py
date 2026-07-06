import torch
import torch.nn as nn
from .config import Config


# ==============================================================================
# production architectures
# only two archs are kept: the v1 baseline and the v12 asymmetric attention pool.
# both were the top performers across the campaign search. all other exploratory
# variants were pruned; see model.py.bak / git history if any are needed again.
# ==============================================================================


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
    """v1 baseline: 3x (conv3d + CBAM) backbone with global pooling and a deep MLP
    head. conv3_dilation lets the final conv stage widen its receptive field while
    keeping the input grid (the 'dilated' campaign variant is just v1 with
    conv3_dilation=2)."""
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


def _Conv3dBnRelu(in_c, out_c, k=3, **kw):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, bias=False, **kw),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class InkDetectorAsymAttnPool(nn.Module):
    """v12_asym_attn_pool: asym_pool with learned soft depth attention instead of
    hard depth-max. backbone is 4x conv3d with spatial avg -> (B,256,D); a small 1D
    conv attention head replaces the hard AdaptiveMaxPool1d with a softmax-weighted
    sum over depth positions.
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


_ARCH_MAP = {
    "v1":                 InkDetector,
    "v12_asym_attn_pool": InkDetectorAsymAttnPool,
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
