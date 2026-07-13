import torch
import torch.nn as nn
import torch.nn.functional as F
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


class InkDetectorMIL(nn.Module):
    """v13_mil: multiple-instance learning with log-sum-exp tile aggregation.

    THE CORE PROBLEM WITH v1/v12 FOR FINE TEXTURE SIGNALS:
    both architectures collapse spatial dimensions to a single vector via
    AdaptiveAvgPool3d before classifying. for scroll1 (strong absorption signal
    in every voxel touched by ink), averaging is fine. for scroll4 at 7.9um or
    any carbon-ink scroll, ink leaves a fine morphological texture trace in a
    small fraction of voxels within each tile. averaging dilutes that signal by
    the tile area (~1000x for a 32px tile), making it undetectable.

    THE FIX — multiple-instance learning (MIL):
    treat each voxel position within the tile as a separate 'instance'. the tile
    label is 1 if ANY instance contains ink. we train with this assumption by
    aggregating per-voxel logits via log-sum-exp, which is a smooth differentiable
    approximation to max. gradient flows back primarily to the highest-confidence
    voxels, teaching the model WHERE ink is rather than just WHETHER a tile average
    is slightly elevated. same BCEWithLogitsLoss, same dataloaders, different signal.

    LSE AGGREGATION:
        tile_logit = (1/r) * log( (1/N) * sum_i exp(r * voxel_logit_i) )
    as r → 0: approaches mean (== global avg pool on logits)
    as r → ∞: approaches max (hard MIL)
    r is a learnable parameter initialized to 2, allowing the model to choose
    the right aggregation hardness for the signal strength it finds.

    ARCHITECTURE:
    - no global spatial downsampling until after the voxel head; the backbone
      retains (D, H/2, W/2) spatial resolution so the model sees individual
      voxel neighborhoods rather than heavily pooled summaries
    - CBAM3D attention at every stage remains (channel + spatial weighting)
    - a lightweight 1x1x1 conv produces per-voxel logits — this map is available
      as a spatial ink heatmap for visualization without running a separate pass

    VISUALIZATION: after any forward pass, model.last_voxel_map holds the
    (B, 1, D', H', W') per-voxel logit map. upsampled to input resolution it
    shows WHERE the model thinks ink is within each tile.
    """

    def __init__(self, config: Config):
        super().__init__()
        # learnable LSE temperature r: clipped to [0.5, 10] during forward
        # r=2 starts between mean and max; the model will find the right value
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        drop1 = float(getattr(config.model, "conv1_drop", 0.0))
        drop2 = float(getattr(config.model, "conv2_drop", 0.05))

        # stage 1: local feature extraction — no spatial reduction
        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(32),
        )  # -> (B, 32, D, H, W)

        # stage 2: richer features — no spatial reduction
        self.stage2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(64),
        )  # -> (B, 64, D, H, W)

        # stage 3: first spatial reduction (spatial only, keep full depth)
        # halving H and W, NOT D, lets the depth attention work over the original
        # depth grid rather than a compressed one. ink has a specific depth profile
        # we don't want to destroy with a cubic pool.
        self.stage3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),   # (B, 128, D, H/2, W/2)
            nn.Dropout3d(drop1),
        )

        # stage 4: deep features at reduced spatial resolution
        self.stage4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.Dropout3d(drop2),
        )  # -> (B, 256, D, H/2, W/2)

        # per-voxel logit head: 1x1x1 conv maps each feature vector to a scalar
        # no bias — the LSE aggregation will shift the output appropriately
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)

        # store the last per-voxel map for visualization (populated in forward)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)   # ensure (B, 1, D, H, W)

        f = self.stage1(x)
        f = self.stage2(f)
        f = self.stage3(f)
        f = self.stage4(f)

        # per-voxel logits: (B, 1, D, H', W')
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()   # saved for visualization

        # log-sum-exp aggregation over all voxel positions
        # tile_logit = (1/r) * log( mean_i exp(r * vmap_i) )
        #            = (1/r) * (logsumexp(r*vmap) - log(N))
        r = self.lse_r.clamp(min=0.5, max=10.0)
        flat = vmap.flatten(1)                          # (B, N)
        N = flat.shape[1]
        tile_logit = (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=x.device))
        )  # (B, 1)

        return tile_logit


class InkDetectorMILDeep(nn.Module):
    """v14_mil_deep: v13_mil with a deeper backbone and explicit depth separation.

    MOTIVATION:
    v13_mil uses a single 3D conv backbone with one spatial downsampling step.
    for fine ink texture at 2.4um, the useful signal may be confined to 1-3 depth
    slices (where the papyrus surface and ink sit) while neighboring slices are
    noise. v14 separates this into two stages:
      1. a per-depth-slice 2D feature extractor (via Conv3d with depth kernel=1)
         that learns SPATIAL texture INDEPENDENTLY per slice — no depth mixing yet
      2. a depth-mixing stage that learns WHICH slices to attend to
    the two-stage design mirrors what a human analyst does: look at each slice for
    texture, then decide which slices show it.
    the MIL aggregation is identical to v13: LSE over all voxel positions.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        drop1 = float(getattr(config.model, "conv1_drop", 0.0))
        drop2 = float(getattr(config.model, "conv2_drop", 0.05))

        # stage A: per-slice spatial texture (depth kernel=1, no depth mixing)
        self.per_slice = nn.Sequential(
            nn.Conv3d(1,  32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
        )  # -> (B, 64, D, H, W) — per-slice features, no depth interaction

        # stage B: depth-aware mixing (full 3D convs)
        self.depth_mix = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(drop1),
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.Dropout3d(drop2),
        )  # -> (B, 256, D, H/2, W/2)

        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(x)
        f = self.depth_mix(f)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()
        r = self.lse_r.clamp(min=0.5, max=10.0)
        flat = vmap.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=x.device))
        )


class InkDetectorMILContrast(nn.Module):
    """v15_mil_contrast: MIL with differential (pre-band subtraction) input.

    MOTIVATION:
    ink at 78keV is a contrast agent relative to the local papyrus baseline.
    the absolute intensity at any spatial position is dominated by papyrus density
    and scan exposure, not ink. the DIFFERENTIAL — ink-band minus a reference
    background band — suppresses the spatially-varying papyrus signal and amplifies
    ink-specific absorption or texture changes.

    INPUT FORMAT: expects depth=16 with a 'diff' convention applied INTERNALLY.
    the first 8 slices are the target band; the last 8 are the reference band.
    the first conv layer sees (target - reference) per voxel, which is roughly
    zero everywhere except where ink causes a brightness shift.

    this is a differentiable version of the 'look at the diff of two depth windows'
    idea from the depth profile: z~20-32 and z~48-60 showed opposite signs in some
    regions, suggesting ink shifts the profile in a direction the diff would amplify.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        drop1 = float(getattr(config.model, "conv1_drop", 0.0))
        drop2 = float(getattr(config.model, "conv2_drop", 0.05))
        # 1x1x1 learned blend: takes (target, ref) per-depth-slice -> weighted diff
        # produces 1 channel from 2 inputs; initialized to +1/-1 for pure subtraction
        self.diff_blend = nn.Conv3d(2, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.diff_blend.weight[:, 0], 1.0)   # target: +1
        nn.init.constant_(self.diff_blend.weight[:, 1], -1.0)  # ref:    -1

        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(64),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(128),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(drop1),
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.Dropout3d(drop2),
        )
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        D = x.shape[2]
        half = D // 2
        # split into target (first half) and reference (last half) depth bands
        target = x[:, :, :half]     # (B, 1, D/2, H, W)
        ref    = x[:, :, half:]     # (B, 1, D/2, H, W)
        if target.shape[2] != ref.shape[2]:
            ref = ref[:, :, :target.shape[2]]
        diff = self.diff_blend(torch.cat([target, ref], dim=1))  # (B, 1, D/2, H, W)
        f = self.backbone(diff)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()
        r = self.lse_r.clamp(min=0.5, max=10.0)
        flat = vmap.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=x.device))
        )


class InkDetectorMILMultiscale(nn.Module):
    """v16_mil_multiscale: parallel multi-scale feature extraction before MIL pooling.

    HYPOTHESIS BEING TESTED:
    ink texture at 2.4um lives at a SPECIFIC spatial scale — the fiber gap scale
    (~10-50px). our other models use a single-scale backbone that may stride through
    the critical scale without explicitly representing it. this model runs THREE
    parallel branches at different spatial scales simultaneously and lets the network
    decide which scale carries signal.

    WHAT IS RADICALLY DIFFERENT:
    - v13_mil: single backbone, LSE finds best voxel anywhere
    - v14_mil_deep: sequential (per-slice then depth mixing)
    - v12: single backbone, depth attention
    - v16: PARALLEL paths: fine (1x), medium (2x downsampled), coarse (4x downsampled)
      each processes at its native resolution then upsamples back for fusion.
      the fusion weights are learned, so if the signal is at fiber scale (~fine),
      the model loads that branch; if it's at stroke scale (~coarse), it uses that.

    SPECIFICALLY USEFUL FOR TEXTURE BECAUSE:
    - papyrus fiber gaps are ~4-20px: captured by fine branch
    - ink stroke interior pattern (~50-100px): captured by medium branch
    - stroke-level ink/non-ink boundary (~100-200px): captured by coarse branch
    - fusing all three means the model is not blind to any of these scales
    """

    def __init__(self, config: Config):
        super().__init__()
        self.lse_r = nn.Parameter(torch.tensor(2.0))
        drop = float(getattr(config.model, "conv2_drop", 0.05))

        def _branch(out_c):
            return nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(32).to(dtype=torch.float32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c).to(dtype=torch.float32),
                nn.ReLU(inplace=True),
            )

        # three branches: fine (no pool), medium (pool 1x2x2), coarse (pool 1x4x4)
        self.fine   = _branch(64)
        self.medium = _branch(64)
        self.coarse = _branch(64)

        # learned scale weights: which branch matters?
        # a 3-way gating via global average then softmax — one set of weights per tile
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64 * 3, 3),
            nn.Softmax(dim=-1),
        )

        # fusion backbone on combined 192-channel features -> voxel logit
        self.fuse = nn.Sequential(
            nn.Conv3d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256).to(dtype=torch.float32),
            nn.ReLU(inplace=True),
            CBAM3D(256),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(drop),
        )
        self.voxel_head = nn.Conv3d(256, 1, kernel_size=1, bias=True)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, _, D, H, W = x.shape

        # fine branch: full spatial resolution
        f_fine = self.fine(x)                                        # (B, 64, D, H, W)

        # medium branch: 2x spatial downsampling
        x_med = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        f_med = self.medium(x_med)
        f_med = torch.nn.functional.interpolate(f_med, size=(D, H, W), mode='trilinear', align_corners=False)

        # coarse branch: 4x spatial downsampling
        x_crs = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        f_crs = self.coarse(x_crs)
        f_crs = torch.nn.functional.interpolate(f_crs, size=(D, H, W), mode='trilinear', align_corners=False)

        # learned scale gating: compute weights from combined pooled features
        cat_pool = torch.cat([
            f_fine.mean(dim=[2, 3, 4]),     # (B, 64)
            f_med.mean(dim=[2, 3, 4]),
            f_crs.mean(dim=[2, 3, 4]),
        ], dim=1)                            # (B, 192)
        w = self.scale_gate[1:](cat_pool)   # (B, 3) softmax weights
        w_fine  = w[:, 0].view(B, 1, 1, 1, 1)
        w_med   = w[:, 1].view(B, 1, 1, 1, 1)
        w_crs   = w[:, 2].view(B, 1, 1, 1, 1)

        # weighted fusion — each branch contributes according to learned importance
        fused = torch.cat([
            f_fine  * (3 * w_fine),    # scale back so sum = 1 doesn't shrink features
            f_med   * (3 * w_med),
            f_crs   * (3 * w_crs),
        ], dim=1)                           # (B, 192, D, H, W)

        f = self.fuse(fused)
        vmap = self.voxel_head(f)
        self.last_voxel_map = vmap.detach()

        r = self.lse_r.clamp(min=0.5, max=10.0)
        flat = vmap.flatten(1)
        N = flat.shape[1]
        return (1.0 / r) * (
            torch.logsumexp(r * flat, dim=1, keepdim=True)
            - torch.log(torch.tensor(float(N), device=x.device))
        )


class InkDetector2p1dMaxAttn(nn.Module):
    """v17_2p1d_maxattn: strict 2+1D — per-slice texture → spatial max → depth attention.

    MOTIVATION:
    v14_mil_deep showed the most promise across all runs. Its key property: the first stage
    uses kernel=(1,3,3) — extracting spatial texture features from EACH DEPTH SLICE
    INDEPENDENTLY, with zero cross-depth information mixing. The remaining stages then
    mix depth and space with 3D convs, which partially undoes this advantage.

    v17 keeps the per-slice insight but makes the subsequent stages also strictly 1D:
      1. Per-slice backbone: 4× Conv3d(kernel=(1,3,3)) — pure 2D texture per slice
      2. AdaptiveMaxPool3d((None,1,1)): BEST texture response per depth slice, not mean.
         Max is better than avg here because we're looking for local ink texture events,
         not a global elevation. This is how v13_mil's LSE behaves but cleaner.
      3. 1D conv depth attention (like v12's winning component): learns which depth
         slices carry the ink texture and downweights uninformative ones.
      4. Softmax-weighted sum over depth → global ink score.

    PAST IDEAS USED:
    - Depth attention is v12_asym_attn_pool's core win across campaigns 14/15.
    - Max aggregation was explored in earlier campaigns for its sensitivity to local peaks.
    - Strict 2+1D separation is a principled video architecture technique that preserves
      spatial structure better than isotropic 3D convs when the spatial and temporal/depth
      signals are independent (which they are here: texture in each slice, depth selection).
    """

    def __init__(self, config: Config):
        super().__init__()
        drop = float(getattr(config.model, "conv2_drop", 0.05))

        # per-slice texture extraction — strictly no depth mixing
        self.per_slice = nn.Sequential(
            nn.Conv3d(1,   32,  kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32),  nn.ReLU(inplace=True),
            nn.Conv3d(32,  64,  kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32),  nn.ReLU(inplace=True),
            nn.Conv3d(64,  128, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(drop),
        )  # -> (B, 128, D, H, W)

        # spatial max per slice — strongest local texture response per depth level
        self.spatial_max = nn.AdaptiveMaxPool3d((None, 1, 1))  # -> (B, 128, D, 1, 1)

        # 1D depth attention (v12's winning component, now applied to per-slice descriptors)
        self.depth_attn = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1, bias=False),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64).to(dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(float(getattr(config.model, "fc1_drop", 0.2))),
            nn.Linear(64, 1),
        )
        self.last_voxel_map = None   # visualizer compatibility

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(x)                                        # (B, 128, D, H, W)
        f_max = self.spatial_max(f).squeeze(-1).squeeze(-1)          # (B, 128, D)
        attn = torch.softmax(self.depth_attn(f_max), dim=-1)         # (B, 1, D)
        fused = (f_max * attn).sum(dim=-1)                           # (B, 128)
        return self.head(fused)                                       # (B, 1)


class InkDetector2p1dLV(nn.Module):
    """v18_2p1d_lv: 2+1D with local variance as an explicit input channel.

    MOTIVATION — FUNDAMENTAL MECHANICS OF THE INK SIGNAL AT 2.4UM:
    At 2.4um/78keV, carbon ink and carbonized papyrus absorb X-rays almost identically.
    The detectable difference is MORPHOLOGICAL: papyrus alone shows alternating high-density
    cellulose fibers and low-density air gaps at the 4-20 voxel scale. Ink FILLS these gaps
    with carbon particles, making the local density MORE UNIFORM — lower local variance.

    A bare papyrus region: fiber(high) gap(low) fiber(high) gap(low) → HIGH local variance
    An ink-covered region: ink(mid) ink(mid) ink(mid) ink(mid)       → LOW local variance

    This model explicitly computes the LOCAL VARIANCE of each depth slice in a 3×3 spatial
    neighborhood and presents it as a second input channel alongside the raw intensity.
    The per-slice backbone then sees BOTH what the mean intensity is AND how much local
    texture variation is present at each voxel — the exact statistic theory predicts is
    discriminative for carbon ink on papyrus at this resolution.

    The rest of the architecture is identical to v17: per-slice backbone → spatial max →
    depth attention → linear head. The only difference is the 2-channel input.
    """

    def __init__(self, config: Config):
        super().__init__()
        drop = float(getattr(config.model, "conv2_drop", 0.05))

        # local variance pooling kernel (fixed, non-learned)
        # computes local std in each depth slice independently (kernel_d=1)
        self.lv_pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        # per-slice backbone — 2 input channels (intensity + local std)
        self.per_slice = nn.Sequential(
            nn.Conv3d(2,   32,  kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32),  nn.ReLU(inplace=True),
            nn.Conv3d(32,  64,  kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(64).to(dtype=torch.float32),  nn.ReLU(inplace=True),
            nn.Conv3d(64,  128, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Dropout3d(drop),
        )

        self.spatial_max = nn.AdaptiveMaxPool3d((None, 1, 1))

        self.depth_attn = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1, bias=False),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64).to(dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(float(getattr(config.model, "fc1_drop", 0.2))),
            nn.Linear(64, 1),
        )
        self.last_voxel_map = None

    def _local_std(self, x):
        """compute local std per voxel in a 3×3 spatial window, per depth slice."""
        mean   = self.lv_pool(x)
        sq_mean = self.lv_pool(x * x)
        var    = (sq_mean - mean.pow(2)).clamp(min=0)
        return var.sqrt()

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        lv = self._local_std(x)                                      # (B, 1, D, H, W)
        x_aug = torch.cat([x, lv], dim=1)                           # (B, 2, D, H, W)
        f = self.per_slice(x_aug)                                    # (B, 128, D, H, W)
        f_max = self.spatial_max(f).squeeze(-1).squeeze(-1)          # (B, 128, D)
        attn = torch.softmax(self.depth_attn(f_max), dim=-1)         # (B, 1, D)
        fused = (f_max * attn).sum(dim=-1)                           # (B, 128)
        return self.head(fused)                                       # (B, 1)


class InkDetectorDenseUNet(nn.Module):
    """dense_unet: per-pixel ink prediction (fully-convolutional 2.5D U-Net).

    ORIGINAL CONFIGURATION — BatchNorm throughout.
    BatchNorm was the configuration that produced the best visual results
    (valid AUC 0.5548, 15 epochs, hard labels, confirmed 2026-07-09).
    InstanceNorm was tested in v4 and killed learning across all 11 archs:
    it normalises each tile independently to zero mean, destroying the
    batch-level relative intensity signal that distinguishes ink from papyrus.
    BatchNorm uses batch-level statistics (1024 tiles) so the ordering is preserved.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        # per-slice texture stem — no depth mixing (kernel depth = 1)
        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1 = conv2(16, 32)
        self.e2 = conv2(32, 64)
        self.e3 = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = conv2(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = conv2(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None   # visualizer compatibility (unused by dense path)

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)      # (B,1,D,H,W)
        f = self.per_slice(x)                    # (B,16,D,H,W)
        f = f.max(dim=2).values                  # depth-max -> (B,16,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)                     # (B,1,H,W) logits


class InkDetectorDenseUNetDepth(nn.Module):
    """dense_unet_depth: dense per-pixel U-Net that MODELS DEPTH instead of discarding it.

    uses InstanceNorm throughout for the same texture-preservation reason as dense_unet.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_mix = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.depth_score = nn.Conv3d(32, 1, kernel_size=1, bias=True)

        self.e1 = conv2(32, 32)
        self.e2 = conv2(32, 64)
        self.e3 = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = conv2(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = conv2(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_depth_attn = None
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f = self.per_slice(x)
        f = self.depth_mix(f)
        score = self.depth_score(f)
        attn = torch.softmax(score, dim=2)
        self.last_depth_attn = attn.detach()
        f2d = (f * attn).sum(dim=2)
        s1 = self.e1(f2d)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)


class _ResBlock2d(nn.Module):
    """residual double-conv block with BatchNorm. skip projection added when ci != co."""
    def __init__(self, ci, co):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(co)
        self.proj  = (nn.Sequential(nn.Conv2d(ci, co, 1, bias=False),
                                    nn.BatchNorm2d(co))
                      if ci != co else nn.Identity())

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)), inplace=True)
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.proj(x), inplace=True)


class _AttnGate2d(nn.Module):
    """additive attention gate (Oktay et al. 2018) for U-Net skip connections.

    computes a spatial attention map from the skip feature s and the gating signal g
    (decoder path at same resolution), then gates the skip: output = s * sigmoid(attn).
    focuses the decoder on ink-boundary locations while suppressing blank papyrus texture.
    inter: intermediate channel dimension (halved relative to skip, standard practice).
    """
    def __init__(self, f_skip, f_gate, inter=None):
        super().__init__()
        inter = inter or max(f_skip // 2, 1)
        self.theta = nn.Conv2d(f_skip, inter, 1, bias=False)   # from skip
        self.phi   = nn.Conv2d(f_gate,  inter, 1, bias=False)  # from gate
        self.psi   = nn.Conv2d(inter,   1,     1, bias=True)   # scalar attn map

    def forward(self, s, g):
        """s: skip connection (B, f_skip, H, W); g: gating signal same spatial size"""
        attn = torch.sigmoid(self.psi(F.relu(self.theta(s) + self.phi(g), inplace=True)))
        return s * attn


class InkDetectorDenseUNetResAttn(nn.Module):
    """dense_unet_res_attn: per-slice stem + hard depth-max + residual encoder + attention gates.

    builds on dense_unet's proven hard depth-max collapse (depth is not a signal source
    on 7.9um scrolls) and adds:
      - residual double-conv blocks in encoder AND decoder: better gradient flow to the
        per-slice texture stem which is the primary ink-detection layer
      - attention gates on all three skip connections: suppresses blank-papyrus encoder
        features before concatenation so the decoder focuses on ink-boundary regions
      - InstanceNorm throughout: preserves fine within-tile texture that BatchNorm
        normalises away across the ink-minority batch
    output: (B,1,H,W) logits; H,W divisible by 8.
    """
    def __init__(self, config: Config):
        super().__init__()

        # per-slice texture stem (no depth mixing)
        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )

        # residual encoder
        self.e1   = _ResBlock2d(16,  32)
        self.e2   = _ResBlock2d(32,  64)
        self.e3   = _ResBlock2d(64,  128)
        self.bott = _ResBlock2d(128, 256)
        self.pool = nn.MaxPool2d(2)

        # upsamplers + attention gates + residual decoder
        self.u3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag3 = _AttnGate2d(128, 128)
        self.d3  = _ResBlock2d(256, 128)

        self.u2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ag2 = _AttnGate2d(64, 64)
        self.d2  = _ResBlock2d(128, 64)

        self.u1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ag1 = _AttnGate2d(32, 32)
        self.d1  = _ResBlock2d(64, 32)

        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)          # (B,1,D,H,W)
        f  = self.per_slice(x).max(dim=2).values      # depth-max -> (B,16,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        g3 = self.u3(b)
        d3 = self.d3(torch.cat([g3, self.ag3(s3, g3)], 1))
        g2 = self.u2(d3)
        d2 = self.d2(torch.cat([g2, self.ag2(s2, g2)], 1))
        g1 = self.u1(d2)
        d1 = self.d1(torch.cat([g1, self.ag1(s1, g1)], 1))
        return self.head(d1)                           # (B,1,H,W) logits


class InkDetectorDenseUNetAsym(nn.Module):
    """dense_unet_asym: deep 4-level encoder, shallow 2-level decoder.

    the ink-papyrus boundary signal is weak and spatially fine-grained — more
    encoder depth extracts richer representations before the signal gets diffused.
    the decoder is intentionally shallow (only 2 transposed-conv levels) because the
    final heatmap resolution only needs to resolve at tile scale, not sub-pixel scale.
    the two 'missing' decoder levels are replaced by bilinear upsample which is faster
    and avoids checkerboard artifacts. skip connections retained at all 4 encoder levels.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        # 4-level encoder (16 → 32 → 64 → 128 → 256)
        self.e1 = conv2(16,  32)
        self.e2 = conv2(32,  64)
        self.e3 = conv2(64,  128)
        self.e4 = conv2(128, 256)
        self.bott = conv2(256, 512)
        # 2-level transposed-conv decoder (512 → 256 → 128)
        self.u4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = conv2(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = conv2(256, 128)
        # remaining 2 levels: bilinear upsample + 1×1 proj (no checkerboard)
        self.proj2 = nn.Conv2d(128, 64, 1, bias=False)
        self.proj1 = nn.Conv2d(64,  32, 1, bias=False)
        self.head  = nn.Conv2d(32,   1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f  = self.per_slice(x).max(dim=2).values        # (B,16,H,W)
        s1 = self.e1(f)                                  # (B,32,H,W)
        s2 = self.e2(self.pool(s1))                      # (B,64,H/2,W/2)
        s3 = self.e3(self.pool(s2))                      # (B,128,H/4,W/4)
        s4 = self.e4(self.pool(s3))                      # (B,256,H/8,W/8)
        b  = self.bott(self.pool(s4))                    # (B,512,H/16,W/16)
        d4 = self.d4(torch.cat([self.u4(b), s4], 1))    # (B,256,H/8,W/8)
        d3 = self.d3(torch.cat([self.u3(d4), s3], 1))   # (B,128,H/4,W/4)
        # bilinear upsample for remaining two levels (no learned skip, just size match)
        up2 = F.relu(self.proj2(F.interpolate(d3, scale_factor=2, mode="bilinear",
                                               align_corners=False)), inplace=True)
        up1 = F.relu(self.proj1(F.interpolate(up2, scale_factor=2, mode="bilinear",
                                               align_corners=False)), inplace=True)
        return self.head(up1)                            # (B,1,H,W)


class InkDetectorDenseUNetLap(nn.Module):
    """dense_unet_lap: fixed Laplacian edge filter prepended to the per-slice stem.

    the ink-papyrus boundary is a morphological transition (pen-pressure groove or
    fiber disruption) that should produce a local edge response in the CT scan even
    if the mean intensity is similar. a fixed (non-learned) Laplacian filter detects
    second-derivative edges per slice and is concatenated with the raw slice, giving
    the learned stem access to BOTH intensity and local edge structure.
    the Laplacian is applied per-depth slice via a 3D conv with kernel depth=1 and
    frozen weights so it never changes during training.
    """
    def __init__(self, config: Config):
        super().__init__()

        # fixed Laplacian kernel applied per slice: (1,1,1,3,3) shape
        lap = torch.tensor([[[[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]]]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer("lap_kernel", lap)    # (1,1,1,3,3)

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        # stem receives 2 channels per slice: raw intensity + Laplacian edge map
        self.per_slice = nn.Sequential(
            nn.Conv3d(2, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1   = conv2(16, 32)
        self.e2   = conv2(32, 64)
        self.e3   = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3   = conv2(256, 128)
        self.u2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2   = conv2(128, 64)
        self.u1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1   = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)          # (B,1,D,H,W)
        B, _, D, H, W = x.shape
        # apply fixed Laplacian to each slice independently: treat (B*D) as batch
        x_flat = x.squeeze(1).reshape(B * D, 1, H, W)
        lap_k  = self.lap_kernel.squeeze(0)           # (1,1,3,3) for conv2d
        edges  = F.conv2d(x_flat, lap_k, padding=1)  # (B*D,1,H,W)
        edges  = edges.reshape(B, D, H, W).unsqueeze(1)   # (B,1,D,H,W)
        x_in   = torch.cat([x, edges], dim=1)         # (B,2,D,H,W) raw + edges
        f  = self.per_slice(x_in).max(dim=2).values   # (B,16,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)


class InkDetectorDenseUNetSpatialDrop(nn.Module):
    """dense_unet_sdrop: dense_unet with spatial dropout on stem output.

    channel-wise spatial dropout (drops entire feature maps) after depth-max
    forces the decoder to be robust to missing texture channels — acts as a
    strong regularizer that prevents the model from over-relying on any single
    per-slice filter response. combined with light rotation-only augmentation.
    dropout rate 0.2: drops 20% of the 16 feature channels each forward pass.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.stem_drop = nn.Dropout2d(0.2)   # channel-wise spatial dropout after depth-max
        self.e1   = conv2(16, 32)
        self.e2   = conv2(32, 64)
        self.e3   = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3   = conv2(256, 128)
        self.u2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2   = conv2(128, 64)
        self.u1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1   = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f  = self.per_slice(x).max(dim=2).values   # (B,16,H,W)
        f  = self.stem_drop(f)                      # channel spatial dropout
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)


class InkDetectorDenseUNetDeep(nn.Module):
    """dense_unet_deep: deeper per-slice stem (4 conv layers instead of 2).

    rationale: the per-slice texture stem is the primary ink-detection layer.
    every other arch improvement (wide, multiscale, residual) adds breadth.
    this adds DEPTH to the stem: 4 stacked (1,3,3) Conv3d layers build a
    deeper per-slice feature hierarchy before depth-max collapse.
    the decoder is the same standard 3-level U-Net as dense_unet.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        # 4-layer per-slice stem: 1→16→32→32→32 channels
        self.per_slice = nn.Sequential(
            nn.Conv3d(1,  16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1   = conv2(32, 32)
        self.e2   = conv2(32, 64)
        self.e3   = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3   = conv2(256, 128)
        self.u2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2   = conv2(128, 64)
        self.u1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1   = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f  = self.per_slice(x).max(dim=2).values   # (B,32,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)




class InkDetectorDenseUNetWide(nn.Module):
    """dense_unet_wide: same as dense_unet but with 32 stem channels instead of 16.

    the per-slice texture stem is the most important single component — it's where
    the model learns to detect the ink-papyrus boundary texture from individual slices.
    doubling its channel capacity from 16→32 gives it more 'texture detectors' before
    the depth-max collapse. the decoder is wider too (e1 starts at 32 channels).
    total params roughly 2× dense_unet; still ~4M — well within 24GB VRAM.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        self.per_slice = nn.Sequential(
            nn.Conv3d(1, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1   = conv2(32,  64)
        self.e2   = conv2(64,  128)
        self.e3   = conv2(128, 256)
        self.bott = conv2(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3   = conv2(512, 256)
        self.u2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2   = conv2(256, 128)
        self.u1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1   = conv2(128, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        f  = self.per_slice(x).max(dim=2).values   # depth-max -> (B,32,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)


class InkDetectorDenseUNetMultiscale(nn.Module):
    """dense_unet_multiscale: dual-scale per-slice stem.

    each slice is processed at TWO spatial scales simultaneously:
      - fine scale:   3×3 conv per slice — local boundary texture
      - coarse scale: 3×3 dilated (dilation=2, equiv receptive field 5×5) per slice
    both produce 16-channel maps, which are concatenated to 32 channels before depth-max.
    the decoder then has the same 32-channel starting width as dense_unet_wide, but the
    features contain BOTH fine boundary texture AND broader morphological context from each
    slice without any parameter overhead from pooling/unpooling.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        # fine-scale branch: standard 3×3 per slice
        self.stem_fine = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        # coarse-scale branch: dilated 3×3 (dilation 2 in spatial, same output size)
        self.stem_coarse = nn.Sequential(
            nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        # fuse: one more per-slice conv on the concatenated 32-channel features
        self.stem_fuse = nn.Sequential(
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1   = conv2(32,  64)
        self.e2   = conv2(64,  128)
        self.e3   = conv2(128, 256)
        self.bott = conv2(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3   = conv2(512, 256)
        self.u2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2   = conv2(256, 128)
        self.u1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1   = conv2(128, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)           # (B,1,D,H,W)
        fine   = self.stem_fine(x)                     # (B,16,D,H,W)
        coarse = self.stem_coarse(x)                   # (B,16,D,H,W)
        fused  = self.stem_fuse(torch.cat([fine, coarse], dim=1))  # (B,32,D,H,W)
        f  = fused.max(dim=2).values                   # depth-max -> (B,32,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)


class InkDetectorDenseUNetCoord(nn.Module):
    """dense_unet_coord: CoordConv per-slice stem.

    appends normalized (y, x) coordinate channels to each depth slice before
    the learned per-slice convolutions. the stem receives 3 channels per slice:
    [raw_intensity, y_norm, x_norm] where y/x are in [0,1] within the tile.

    rationale: the U-Net decoder loses exact spatial position during the three
    rounds of 2× downsampling. re-feeding position explicitly at the stem lets
    the decoder always know where it is within the tile, helping it produce
    spatially coherent boundary heatmaps. additionally, position-dependent
    textures (if any) become easier to detect when position is an explicit input.

    the coordinate maps are computed once per forward pass and broadcast to
    the batch; no learnable parameters added by CoordConv itself.
    """
    def __init__(self, config: Config):
        super().__init__()

        def conv2(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co).to(dtype=torch.float32), nn.ReLU(inplace=True),
            )

        # stem takes 3 channels per slice: intensity + y_coord + x_coord
        self.per_slice = nn.Sequential(
            nn.Conv3d(3, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16).to(dtype=torch.float32), nn.ReLU(inplace=True),
        )
        self.e1   = conv2(16, 32)
        self.e2   = conv2(32, 64)
        self.e3   = conv2(64, 128)
        self.bott = conv2(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.u3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3   = conv2(256, 128)
        self.u2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2   = conv2(128, 64)
        self.u1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1   = conv2(64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.last_voxel_map = None

    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)          # (B,1,D,H,W)
        B, _, D, H, W = x.shape
        # build coordinate maps in [0,1]: same for every item in batch
        yy = torch.linspace(0, 1, H, device=x.device).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        xx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
        x_in = torch.cat([x, yy, xx], dim=1)          # (B,3,D,H,W)
        f  = self.per_slice(x_in).max(dim=2).values   # depth-max -> (B,16,H,W)
        s1 = self.e1(f)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.head(d1)                           # (B,1,H,W)


_ARCH_MAP = {
    "v1":                 InkDetector,
    "v12_asym_attn_pool": InkDetectorAsymAttnPool,
    "v13_mil":            InkDetectorMIL,
    "v14_mil_deep":       InkDetectorMILDeep,
    "v15_mil_contrast":   InkDetectorMILContrast,
    "v16_mil_multiscale": InkDetectorMILMultiscale,
    "v17_2p1d_maxattn":   InkDetector2p1dMaxAttn,
    "v18_2p1d_lv":        InkDetector2p1dLV,
    "dense_unet":              InkDetectorDenseUNet,
    "dense_unet_depth":        InkDetectorDenseUNetDepth,
    "dense_unet_res_attn":     InkDetectorDenseUNetResAttn,
    "dense_unet_wide":         InkDetectorDenseUNetWide,
    "dense_unet_multiscale":   InkDetectorDenseUNetMultiscale,
    "dense_unet_asym":         InkDetectorDenseUNetAsym,
    "dense_unet_lap":          InkDetectorDenseUNetLap,
    "dense_unet_sdrop":        InkDetectorDenseUNetSpatialDrop,
    "dense_unet_deep":         InkDetectorDenseUNetDeep,
    "dense_unet_coord":        InkDetectorDenseUNetCoord,
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
