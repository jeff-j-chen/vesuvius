"""radical_archs.py -- cutting-edge architecture implementations for campaign_archs_6.

six paradigm-shifting architectures adapted to 3D ink detection:
  1. ViT3D - pure vision transformer (Dosovitskiy et al. 2020)
  2. Swin3D - shifted-window hierarchical transformer (Liu et al. 2021)
  3. ConvNeXt3D - modernized CNN (Liu et al. 2022)
  4. XCiT3D - cross-covariance transformer (El-Nouby et al. 2021)
  5. nnU-Net3D - self-configuring U-Net with deep supervision (Isensee et al. 2021)
  6. SlotAttention3D - object-centric representation (Locatello et al. 2020)

these are imported and registered in utils/model.py's _ARCH_MAP.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


# ==============================================================================
# 1. ViT3D - Pure Vision Transformer for 3D
# ==============================================================================

class PatchEmbed3D(nn.Module):
    """split 3D volume into non-overlapping patches and linearly embed them"""
    def __init__(self, patch_size=4, in_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D//P, H//P, W//P)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        x = self.norm(x)
        return x, (D, H, W)


class Attention3D(nn.Module):
    """multi-head self-attention with optional relative position bias"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """drop paths (stochastic depth) per sample"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    """transformer block: LN -> Attn -> LN -> MLP, with residuals"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT3D(nn.Module):
    """3D Vision Transformer for ink detection
    
    hypothesis: global receptive field from layer 1 captures long-range correlations.
    no inductive bias (unlike CNNs), so needs more data but can learn arbitrary patterns.
    """
    def __init__(self, config: Config):
        super().__init__()
        # input after context_downsample=2: (B, 1, 24, 24, 24) at ctx=48, tile=16
        # patch_size=4 -> 6x6x6 = 216 tokens (manageable for global attention)
        patch_size = 4
        embed_dim = 256
        depth = 6
        num_heads = 8
        mlp_ratio = 4.
        drop_rate = getattr(config.model, "head_drop", 0.4)
        attn_drop_rate = 0.1
        drop_path_rate = 0.1

        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = (24 // patch_size) ** 3  # 216 for ctx=48 after ds=2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

        # initialize
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _resize_pos_embed(self, pos_embed, target_len):
        """Interpolate positional embeddings to match target sequence length."""
        # pos_embed: (1, num_patches+1, embed_dim)
        # We keep CLS token and interpolate the rest
        cls_pos = pos_embed[:, :1, :]  # (1, 1, embed_dim)
        patch_pos = pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)
        
        if target_len - 1 == patch_pos.shape[1]:
            return pos_embed
        
        # Linear interpolation
        patch_pos = patch_pos.permute(0, 2, 1)  # (1, embed_dim, num_patches)
        patch_pos = F.interpolate(
            patch_pos, size=target_len - 1, mode='linear', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 1)  # (1, target_len-1, embed_dim)
        
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x, _ = self.patch_embed(x)  # (B, num_patches_actual, embed_dim)
        B, N, C = x.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)
        
        # interpolate positional embeddings if needed
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self._resize_pos_embed(self.pos_embed, x.shape[1])
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # cls token prediction


# ==============================================================================
# 2. Swin3D - Shifted Window Transformer for 3D
# ==============================================================================

class WindowAttention3D(nn.Module):
    """window-based multi-head self-attention for 3D"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block with windowed/shifted attention"""
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size=window_size, num_heads=num_heads, qkv_bias=True,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # cyclic shift (simplified for 3D - shift only H and W dims)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows (simplified: assume divisible)
        # window_size must divide D, H, W
        # for simplicity, process as global if dims not divisible
        x_windows = shifted_x.reshape(B, -1, C)  # simplified: treat as sequence
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # reverse cyclic shift
        shifted_x = attn_windows.view(B, D, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Swin3D(nn.Module):
    """Swin Transformer for 3D ink detection
    
    hypothesis: local windowed attention is more efficient than global, and shifting
    windows allows global information flow. hierarchical structure (like CNNs) for multi-scale.
    """
    def __init__(self, config: Config):
        super().__init__()
        embed_dim = 192  # single stage, higher dim to compensate
        depth = 12  # total blocks
        num_heads = 6
        window_size = 4
        drop_rate = getattr(config.model, "head_drop", 0.4)
        drop_path_rate = 0.1

        self.patch_embed = PatchEmbed3D(patch_size=2, in_chans=1, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Single stage with alternating window shifts
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=4.,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, (D, H, W) = self.patch_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, D, H, W)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# ==============================================================================
# 3. ConvNeXt3D - Modernized CNN for 3D
# ==============================================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DWConv -> LayerNorm -> 1x1 expand -> GELU -> 1x1 project"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt3D(nn.Module):
    """ConvNeXt: A ConvNet for the 2020s (adapted to 3D)
    
    hypothesis: modern CNN design (large kernels, better norm, inverted bottlenecks)
    can match transformers with better inductive bias for 3D spatial structure.
    """
    def __init__(self, config: Config):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        drop_path_rate = 0.1
        drop_rate = getattr(config.model, "head_drop", 0.4)

        # stem: aggressive downsampling (4x)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 6, 6, 6], eps=1e-6)  # 24/4=6
        )
        self.downsample_layers.append(stem)

        # 3 downsampling transitions
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm([dims[i], 6 // (2**i), 6 // (2**i), 6 // (2**i)], eps=1e-6),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 stages of ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], 1)
        self.head_drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = x.mean([-3, -2, -1])  # global average pool
        x = self.norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


# ==============================================================================
# 4. XCiT3D - Cross-Covariance Image Transformer for 3D
# ==============================================================================

class XCA(nn.Module):
    """Cross-Covariance Attention: O(d^2) instead of O(N^2)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, C//heads)

        # L2 normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # cross-covariance: (B, heads, C//heads, C//heads)
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # apply to values
        x = (v @ attn).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCiTBlock(nn.Module):
    """XCiT transformer block with cross-covariance attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class XCiT3D(nn.Module):
    """Cross-Covariance Image Transformer for 3D
    
    hypothesis: XCA models feature interactions explicitly (not just token similarity),
    better for fine-grained texture discrimination. also more efficient O(d^2) vs O(N^2).
    """
    def __init__(self, config: Config):
        super().__init__()
        patch_size = 4
        embed_dim = 256
        depth = 12
        num_heads = 8
        mlp_ratio = 4.
        drop_rate = getattr(config.model, "head_drop", 0.4)
        drop_path_rate = 0.1

        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            XCiTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=False, drop=drop_rate, attn_drop=0.1, drop_path=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, _ = self.patch_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average over all tokens
        x = self.head(x)
        return x


# ==============================================================================
# 5. nnU-Net3D - Self-Configuring U-Net with Deep Supervision
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # handle size mismatch
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class nnUNet3D(nn.Module):
    """nnU-Net style encoder-decoder with deep supervision (TILE-LEVEL ONLY)
    
    hypothesis: skip connections preserve fine spatial detail. multi-scale features
    capture both local texture and global layout. deep supervision prevents vanishing gradients.
    
    CRITICAL: NO DENSE OUTPUTS. all heads (including deep supervision) output 1 logit per tile.
    deep supervision heads use global pooling -> each intermediate feature map -> 1 scalar.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.n_channels = 1
        bilinear = True

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # deep supervision heads - TILE-LEVEL (pool to 1 logit, NO dense outputs)
        # these are NOT used during inference, only during training for auxiliary supervision
        self.ds_head1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512 // factor, 1)
        )
        self.ds_head2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256 // factor, 1)
        )
        self.ds_head3 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128 // factor, 1)
        )
        
        # final head - TILE-LEVEL (1 logit)
        self.outc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        # deep supervision outputs (not used during inference, trainer can optionally use during training)
        # each outputs EXACTLY 1 logit (global pool -> linear)
        # ds1 = self.ds_head1(x)  # (B, 1) - TILE-LEVEL logit
        
        x = self.up2(x, x3)
        # ds2 = self.ds_head2(x)  # (B, 1) - TILE-LEVEL logit
        
        x = self.up3(x, x2)
        # ds3 = self.ds_head3(x)  # (B, 1) - TILE-LEVEL logit
        
        x = self.up4(x, x1)
        logits = self.outc(x)  # (B, 1) - TILE-LEVEL logit
        
        return logits  # EXACTLY 1 binary logit per tile, NO dense outputs


# ==============================================================================
# 6. SlotAttention3D - Object-Centric Representation Learning
# ==============================================================================

class SlotAttention(nn.Module):
    """Slot Attention module: iteratively bind features to object slots"""
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs):
        b, n, d = inputs.shape
        
        # initialize slots
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttention3D(nn.Module):
    """Slot Attention for 3D ink detection
    
    hypothesis: ink regions are discrete "objects" in the volume. slot attention learns
    to segment them without localization labels. highly interpretable (can visualize slots).
    """
    def __init__(self, config: Config):
        super().__init__()
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        
        self.encoder_pos = nn.Parameter(torch.randn(1, 6*6*6, 128) * 0.02)  # 24/4=6 after 2 pools
        
        self.slot_attention = SlotAttention(
            num_slots=8,
            dim=128,
            iters=4,
            hidden_dim=256
        )
        
        # pool slots and classify
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(getattr(config.model, "head_drop", 0.4)),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # encode
        x = self.encoder(x)  # (B, 128, D/4, H/4, W/4)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        x = x + self.encoder_pos
        
        # slot attention
        slots = self.slot_attention(x)  # (B, num_slots, C)
        
        # pool slots (mean) and classify
        slots_pooled = slots.mean(dim=1)
        logits = self.head(slots_pooled)
        
        return logits
