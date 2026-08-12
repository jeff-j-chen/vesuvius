# Campaign Archs 7: Architecture Variants Guide
**Date**: 2026-08-11  
**Total Tests**: 44  
**Scroll**: w044 (20260115000000) single-scroll isolation

---

## Design Philosophy

**What Works** (from campaigns 1-6):
- ✅ **Depth preservation**: keeping 3D representation through network
- ✅ **Large context**: 48×48 or larger receptive fields
- ✅ **Wide head**: more channels in final layers
- ✅ **Z-gradient**: dI/dz as input channel
- ✅ **LCN**: local contrast normalization
- ✅ **Attention-MIL + entropy**: gated attention with coverage regularization
- ✅ **SupCon**: contrastive learning (both ink and depth profiles)

**What Fails** (empirically proven):
- ❌ **Early depth squashing**: "smashing depth early removes all ability to learn"
- ❌ **Dense pixel-level**: requires dense labels which we don't have

---

## Test Groups

### Group 1-2: Sanity Checks
Verify infrastructure works, establish baseline performance.

#### 1. w044_sanity
- Standard depsc_high configuration
- All proven components: depth SupCon λ=0.3, attn-MIL, entropy=0.03
- Purpose: verify infrastructure on known-good scroll

#### 2. w044_noaug
- Same as sanity but **NO augmentation**
- Purpose: can the model overfit? If not, data/label issue

---

### Group 3-6: Context Ablation
Test receptive field size effects (Villa uses 128×128).

#### 3. ctx96_ds2
- 96×96 context, downsample 2× → 48×48 effective
- Memory: ~2.25× baseline (still fits 24GB VRAM)
- Hypothesis: wider context helps spatial coherence

#### 4. ctx64_ds2
- 64×64 context, downsample 2× → 32×32 effective
- Memory: baseline-like
- Hypothesis: smaller than 48×48 baseline, test if context too small hurts

#### 5. ctx128_ds2
- 128×128 context, downsample 2× → 64×64 effective
- Memory: ~4× baseline (may OOM on 24GB, reduce batch if needed)
- Hypothesis: maximum context we can fit, test ceiling performance

#### 6. ctx96_fovea
- Architecture: `v16_arch_ctx_fovea`
- Center 16×16 at full resolution
- Surround 96×96 downsampled 3×
- Hypothesis: sharp center + wide context without memory explosion

---

### Group 7-10: Dual-Stream Depth
Parallel processing of squashed (2D) + non-squashed (3D) representations.

**Core Idea**: Combine Villa's depth fusion benefits with our proven 3D approach.

#### 6. dual_early (Early Fusion)
```python
# Architecture flow:
Squashed stream: 3D stem → depth fusion → 2D features (128ch)
Non-squashed stream: stage1 → 3-window → 3D features (256ch)

# Fusion AFTER STEMS, before deep processing:
concat([squashed_2d_broadcast, non_squashed_3d])
→ shared deep layers
→ MIL aggregation
```
**Hypothesis**: Early fusion lets gradients flow through both paths equally.

#### 7. dual_late (Late Fusion / Ensemble)
```python
# Each stream processes independently to logits:
Squashed: stem → depth fusion → 2D head → tile_logit_2d
Non-squashed: stage1+2 → 3D MIL → tile_logit_3d

# Fusion at output:
tile_logit_final = α * tile_logit_2d + (1-α) * tile_logit_3d
# α learned during training
```
**Hypothesis**: Ensemble-like, each stream specializes independently.

#### 8. dual_gated (Gated Fusion)
```python
# Fusion with learned attention:
gate = σ(FC([squashed_feat, nonsquashed_feat]))
fused = gate * squashed + (1-gate) * nonsquashed
→ MIL aggregation
```
**Hypothesis**: Model learns which stream to trust per-sample.

#### 9. dual_asym (Asymmetric Capacity)
```python
# Lightweight squashed (context only):
Squashed: 1 conv layer → depth fusion → 32ch

# Heavy non-squashed (detail):
Non-squashed: full stage1+2 → 256ch

# Fusion:
concat([32ch_2d, 256ch_3d]) → MIL
```
**Hypothesis**: Squashed provides global context cheaply, non-squashed does heavy lifting.

---

### Group 10-13: Hybrid Depth Attention
Villa's dual-branch attention adapted to our multi-window needs.

#### 10. hybrid_win (Per-Window - Recommended)
```python
# For EACH 8-slice window:
3D stem (k=3 in depth) → (B, 32, 8, H, W)

# Dual-branch depth fusion:
attn_branch = softmax(Conv3d(32→1)) → weighted_sum → 32ch
max_branch = max_over_depth → 32ch
concat → 64ch per window (now 2D)

# Fuse 3 windows:
concat(3 × 64ch) → 2D head → MIL
```
**Hypothesis**: Villa's robustness + our multi-window coverage. Best of both.

#### 11. hybrid_global (Global - Risky)
```python
# Single attention over ALL 24 slices:
3D stem → (B, 32, 24, H, W)
attn_branch = softmax_over_24 → weighted_sum → 32ch
max_branch = max_over_24 → 32ch
→ single 2D representation

# This is exactly Villa's approach
```
**Hypothesis**: Will likely fail (our empirical finding), but test to confirm.

#### 12. hybrid_triple (Triple-Branch)
```python
# THREE fusion branches per window:
attn_branch = softmax → weighted_sum
max_branch = max_over_depth
mean_branch = mean_over_depth
concat → 96ch per window

# More robust than dual-branch
```
**Hypothesis**: Mean adds stability when attention/max both fail.

#### 13. hybrid_gated (Gated Mix)
```python
# Learnable mix of attention vs max:
gate = σ(FC(global_pool(features)))
fused = gate * attn_pooled + (1-gate) * max_pooled

# Per-sample decision: trust attention or max?
```
**Hypothesis**: Adapts fusion strategy to sample difficulty.

---

### Group 14-19: Multi-Scale & Efficient

#### 14. multiscale (Multi-Scale Pyramid)
```python
# Process at 3 scales:
scale1 = forward(x)              # 1.0x
scale2 = forward(downsample_2x(x))  # 0.5x (coarser context)
scale3 = forward(downsample_4x(x))  # 0.25x (global context)

# Fuse all scales:
upsample + concat → fusion head → MIL
```
**Rationale**: Ink features visible at multiple scales (2px at 9µm = ~18µm stroke width).

#### 15. depth_se (Depth Squeeze-Excitation)
```python
# Channel attention on DEPTH dimension:
global_pool_over_HW → FC(D→D/r→D) → σ → scale depth channels
# Learns which depth slices are important globally
```
**Rationale**: Lightweight (few params), proven in 2D (SENet), adapted to depth.

#### 16. depthsep (Depthwise Separable 3D)
```python
# Factorize 3D conv:
Conv3d(C_in, C_out, k=3) 
→ DepthwiseConv3d(C_in, k=3) + PointwiseConv3d(C_in→C_out, k=1)
# Cheaper, forces depth features to stay separate
```
**Rationale**: Aligns with "depth separation" principle, fewer params.

#### 17. mixed_win (Mixed Depth Windows)
```python
# Process 5 windows SIMULTANEOUSLY (not sequentially):
non_overlap = [0-8, 8-16, 16-24]  # 3 standard
overlap = [4-12, 12-20]           # 2 at seams

# All 5 fused together:
concat(5 windows) → stage2 fusion
```
**Rationale**: Builds on 5-window success from archs5, processes in parallel for efficiency.

#### 18. octave (Octave Convolutions)
```python
# Separate high/low frequency paths:
high_freq = input[:, :, high_channels]  # fine detail
low_freq = downsample(input[:, :, low_channels])  # global structure

# Process separately, exchange info:
high_to_low = downsample(high)
low_to_high = upsample(low)
high' = f_high(high, low_to_high)
low' = f_low(low, high_to_low)
```
**Rationale**: Ink (high-freq edges) + papyrus (low-freq density) separated naturally.

#### 19. efficient (EfficientNet Compound Scaling)
```python
# Balanced scaling of width/depth/resolution:
width_mult = 1.2    # 20% more channels
depth_mult = 1.1    # 10% more layers
resolution = 1.15   # 15% larger input (48 → 55)

# All scaled together maintains balance
```
**Rationale**: EfficientNet paper showed compound scaling > single-dimension scaling.

---

### Group 20-25: Attention Mechanisms

#### 20. nonlocal (Non-Local Blocks)
```python
# Self-attention WITHIN depth dimension:
Q = Conv(x, k=1)  # queries
K = Conv(x, k=1)  # keys  
V = Conv(x, k=1)  # values

# Attention over depth dimension only (not H,W):
attn = softmax(Q @ K.T / √d)
out = attn @ V  # each depth attends to all other depths
```
**Rationale**: Long-range depth dependencies (ink at multiple depths correlates).

#### 21. coord_attn (Coordinate Attention, CVPR 2021)
```python
# Factorize spatial attention:
pool_H = avg_pool_width(x)   # (B,C,H,1)
pool_W = avg_pool_height(x)  # (B,C,1,W)

attn_H = sigmoid(Conv1d(pool_H))
attn_W = sigmoid(Conv1d(pool_W))

# Apply coordinate-wise:
out = x * attn_H * attn_W
```
**Rationale**: Cheaper than full spatial attention, captures H/W correlations separately.

#### 22. deform (Deformable Convolutions)
```python
# Learn offset patterns:
offsets = Conv(x)  # (B, 2*k*k, D, H, W)
# Sample input at offset positions:
out = grid_sample(x, base_grid + offsets)
```
**Rationale**: Ink isn't always at grid-aligned positions; deformable kernels adapt.

#### 23. progressive (Progressive Depth Refinement)
```python
# Coarse-to-fine processing:
coarse = process(downsample_depth_4x(x))  # 6 slices
medium = process(downsample_depth_2x(x))  # 12 slices
fine = process(x)                          # 24 slices

# Each stage refines the previous:
medium' = medium + upsample(coarse)
fine' = fine + upsample(medium')
```
**Rationale**: Curriculum learning - learn global depth structure first, refine details.

#### 24. dual_attn (Dual Channel + Spatial)
```python
# CBAM-style but applied together:
channel_attn = σ(FC(global_pool(x)))
spatial_attn = σ(Conv(concat([max_pool_C(x), avg_pool_C(x)])))

out = x * channel_attn * spatial_attn
```
**Rationale**: Channel tells "what" features, spatial tells "where". Both needed.

#### 25. axial (Axial Attention)
```python
# Attention along each axis separately:
attn_H = self_attention_over_H_axis(x)
attn_W = self_attention_over_W_axis(x)
attn_D = self_attention_over_D_axis(x)

out = attn_H + attn_W + attn_D
# Complexity: O(D*H + D*W + H*W) vs O(D*H*W) for full 3D attention
```
**Rationale**: Cheaper than full 3D attention, still captures long-range dependencies.

---

### Group 26-31: Advanced Fusion & Refinement

#### 26. fpn (Feature Pyramid Network)
```python
# Build pyramid top-down:
C5 = stage5(x)  # coarsest, deepest features
C4 = stage4(x)
C3 = stage3(x)

# Top-down pathway with lateral connections:
P5 = C5
P4 = upsample(P5) + lateral(C4)
P3 = upsample(P4) + lateral(C3)

# Predict from all levels:
predictions = [head(P3), head(P4), head(P5)]
final = fuse(predictions)
```
**Rationale**: Multi-scale features proven for object detection, adapt to ink detection.

#### 27. bifpn (Bi-directional FPN)
```python
# FPN + bottom-up pathway:
# Top-down (FPN):
P5 = C5
P4 = upsample(P5) + C4

# Bottom-up (new):
P4' = P4 + downsample(P3)
P5' = P5 + downsample(P4')

# Iterative refinement
```
**Rationale**: EfficientDet showed bi-directional > uni-directional for multi-scale.

#### 28. ghost (Ghost Convolutions, Huawei 2020)
```python
# Generate features cheaply:
intrinsic = Conv(x, C_out/2)  # half the channels normally
cheap = DepthwiseConv(intrinsic)  # cheap transforms

out = concat([intrinsic, cheap])  # same output channels, less compute
```
**Rationale**: 2× speedup with minimal accuracy loss (proven in MobileNet successors).

#### 29. inverted (Inverted Residuals, MobileNetV2)
```python
# Expand → Depthwise → Project:
expanded = PointwiseConv(x, expand_ratio * C)  # 6× channels
dw = DepthwiseConv(expanded)
projected = PointwiseConv(dw, C)  # back to original

out = x + projected  # residual
```
**Rationale**: Efficient feature mixing, proven in mobile architectures.

#### 30. resnext (ResNeXt Grouped Convolutions)
```python
# Split channels into groups:
groups = split(x, cardinality=32)
group_convs = [Conv(g) for g in groups]
out = concat(group_convs)

# Cardinality (# groups) > width (# channels per group)
```
**Rationale**: ResNeXt showed cardinality > depth or width for same compute.

#### 31. depth_shift (Depth Shift Module)
```python
# Temporal shift adapted for depth:
shift_up = shift(x, +1, dim=depth)    # shift depth slices up
shift_down = shift(x, -1, dim=depth)  # shift depth slices down
no_shift = x

out = concat([shift_up, no_shift, shift_down]) → Conv(3C → C)
# Zero-param way to mix depth information
```
**Rationale**: TSM (temporal shift module) proven for video, adapt to depth dimension.

---

### Group 32-33: Archs5 Proven

#### 32. gce_noise (GCE Loss q=0.9)
- Generalized Cross-Entropy with q=0.9 (vs q=0.7 baseline)
- More robust to label noise
- Proven effective in archs5

#### 33. five_win (5 Depth Windows)
- Windows: [0-8, 4-12, 8-16, 12-20, 16-24]
- Seam depths (12, 20) now in center of a window
- Proven effective in archs5

---

### Group 34-37: Villa Normalization

#### 34. villa_mad_norm (Robust-MAD)
```python
# Per-patch normalization:
clip = np.clip(patch, percentile(1), percentile(99))
median = np.median(clip)
mad = np.median(np.abs(clip - median))
normalized = (patch - median) / (mad * 1.4826)
```
**Rationale**: Outlier-robust, scroll-adaptive, proven in Villa.

#### 35. villa_groupnorm (GroupNorm + LeakyReLU)
- GroupNorm: batch-independent (stable across scroll diversity)
- LeakyReLU: prevents dead neurons (slope=0.01)
**Rationale**: Villa's normalization choice, batch-size independent.

#### 36. villa_instancenorm (InstanceNorm + LeakyReLU)
- InstanceNorm: per-sample statistics
- **Risk**: unstable for small tiles (16×16×24 = 6k voxels)
**Rationale**: Villa's choice for 128×128 patches, test on our smaller tiles.

#### 37. villa_layernorm (LayerNorm + LeakyReLU)
- LayerNorm: normalize over channels (transformer-style)
**Rationale**: Alternative to GroupNorm, proven in transformers.

---

### Group 38-43: Radical Architectures (from archs6)

#### 38. vit3d (3D Vision Transformer)
- Pure self-attention, no convolutions
- Patchify 3D input → transformer blocks
**Expected**: Likely to fail (needs dense labels), but test for completeness.

#### 39. swin3d (Swin Transformer 3D)
- Hierarchical shifted-window attention
- Local windows + shifting for global propagation
**Expected**: Better than ViT (local bias), still risky without dense labels.

#### 40. convnext3d (ConvNeXt 3D)
- Modernized CNN: 7×7 kernels, LayerNorm, GELU
**Expected**: Most promising of transformers (still fundamentally conv-based).

#### 41. xcit3d (XCiT 3D)
- Cross-covariance attention (cheaper than dot-product)
**Expected**: Marginal vs ViT, both likely struggle.

#### 42. nnunet3d (nnU-Net 3D)
- Encoder-decoder with skip connections
- Deep supervision at multiple scales
**Expected**: Will fail (needs dense labels for decoder supervision).

#### 43. slot3d (Slot Attention 3D)
- Object-centric iterative attention
- Each "slot" binds to one ink region
**Expected**: Interesting but needs object-level reasoning we don't have labels for.

---

## Expected Winners (Predictions)

**Tier 1 - Highly Likely to Win**:
1. **dual_asym** (asymmetric dual-stream): Best of 2D context + 3D detail
2. **hybrid_win** (per-window hybrid): Villa robustness + our multi-window
3. **multiscale** (multi-scale pyramid): Proven across domains
4. **mixed_win** (mixed depth windows): Builds on proven 5-window
5. **depth_se** (depth squeeze-excitation): Lightweight, effective

**Tier 2 - Moderate Chance**:
6. **ctx96_fovea**: Sharp center + wide context balance
7. **coord_attn**: Modern, proven attention mechanism
8. **progressive**: Curriculum learning aligns with our training
9. **fpn**: Multi-scale fusion proven in detection
10. **depthsep**: Aligns with depth-separation principle

**Tier 3 - Experimental**:
11. **nonlocal**: May help depth correlation learning
12. **bifpn**: Iterative refinement could help
13. **convnext3d**: Most CNN-like of the transformers

**Expected to Fail**:
- **hybrid_global**: Conflicts with "no early squashing" finding
- **vit3d, swin3d, xcit3d**: Need dense labels
- **nnunet3d, slot3d**: Need dense/object-level labels
- **dual_late**: Ensemble usually needs diversity (both streams similar)

---

## Implementation Priority

**Phase 1 (Implement First)**: Tests 1-13
- Core infrastructure (sanity, context, dual-stream, hybrid)
- Highest expected ROI

**Phase 2 (Implement Second)**: Tests 14-25
- Multi-scale and attention (proven effective elsewhere)
- Medium expected ROI

**Phase 3 (Implement Last)**: Tests 26-43
- Advanced fusion and transformers
- Lower expected ROI but comprehensive coverage

---

## Architecture Flags

New config flags to implement:

```python
# Model config:
model.dual_stream_mode: str = "none"  # "early" | "late" | "gated" | "asym"
model.hybrid_depth_mode: str = "none"  # "per_window" | "global" | "triple" | "gated"
model.multiscale: bool = False
model.depth_se: bool = False
model.depthwise_separable: bool = False
model.mixed_windows: bool = False
model.octave_conv: bool = False
model.nonlocal_depth: bool = False
model.coord_attention: bool = False
model.deformable_conv: bool = False
model.progressive_depth: bool = False
model.axial_attention: bool = False
model.fpn: bool = False
model.bifpn: bool = False
model.ghost_conv: bool = False
model.inverted_residual: bool = False
model.resnext_groups: int = 1  # 1 = disabled, 32 = ResNeXt
model.depth_shift: bool = False
```

Note: Most will be implemented as separate architecture classes (e.g., `v16_dual_stream_early`) rather than flags to keep code clean.
