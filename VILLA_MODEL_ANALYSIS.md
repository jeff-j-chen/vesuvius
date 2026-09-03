# VILLA vs Current Model Analysis
**Date**: 2026-08-11  
**Last Updated**: 2026-09-03
**Models Compared**:  
- **VILLA**: aligned21_hybrid_3d2d (3D stem → 2D U-Net)  
- **OURS AT ORIGINAL WRITING**: v16_arch_ctx (2-stage MIL with context window)

## 2026-09-03 status correction

This document below is a historical comparison against a retired model. Its VILLA notes remain
useful, but recommendations and descriptions of the current model are superseded by this section
and the repository README.

The active model is now `nnunet3d_lcndz`:

- full 3D nnU-Net encoder/decoder with raw + LCN + dI/dz inputs
- 192x192 context with spatial downsampling 2
- IBN in shallow encoder blocks
- 16x16 prediction center split into four 8x8 multitile targets
- gated attention-MIL with entropy regularization
- closed-ring, `pos_only` supervision and a hand-authored train/validation split on w013
- legacy learned-surface attention plus the campaign-17 supervised depth-softmax surface branch
- MAE warm start, variance spill, SupCon, TTA consistency, and weak feature dropout

Corrections to the historical recommendations:

1. Dense labels exist, but dense training repeatedly failed because the transferred labels are
  uncertain. The project intentionally uses sparse multitile supervision rather than competing
  with VILLA's dense recipe.
2. Instance normalization and LeakyReLU are already used throughout the nnU-Net; IBN-a is used in
  the first normalization of the two shallow blocks.
3. Depth jitter is already active at +/-4 slices in campaign 17.
4. The current campaign uses weighted BCE, not GCE, Dice, or label smoothing.
5. DANN cannot help the one-scroll campaign because `dann_n_domains=1` makes it a no-op. Earlier
  multi-domain DANN was too strong and damaged low-contrast regions.
6. Context has been swept through 256px. The measured optimum is 192px/ds2; 256px adds substantial
  cost without useful improvement.
7. Multitile flips and rotations now transform the image and 2x2 target/mask grids together.
  Elastic and context jitter remain disabled until exact dense-target warping is implemented.

Do not use the numerical gain projections at the bottom of this historical document for planning;
they were speculative and predate campaigns 9-17.

---

## VERIFIED FACTS FROM HUGGINGFACE (2026-08-11)

**Source**: https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um

### Model Architecture
- **Config**: `configs/aligned21_hybrid_3d2d.json` (in Villa repo)
- **Architecture**: "small local 3D stem feeding a 2D U-Net"
- **Patch Size**: **128×128 pixels** (NOT hand-drawn rectangles!)
- **Sampling**: Uniform grid, **stride 32px** (dense overlap: 75% of each patch overlaps neighbors)
- **Depth Window**: 17 of 21 slices, jittered ±2 slices during training
- **Input**: ~9µm **isotropic surface volumes** (explained below)

### What "Isotropic Surface" Means
**Isotropic** = voxels are roughly cubic (similar resolution in X, Y, Z dimensions)

Villa trains on TWO data sources at the same effective scale:
1. **Aligned 2.4µm → 9.6µm isotropic**:
   - Start: 2.399µm XY surface volumes (anisotropic: fine XY, coarse Z)
   - Process: Level-2 XY pyramid (4× downsample) + 4× z mean-pooling
   - Result: ~9.6µm isotropic (cubic voxels)
   
2. **Native 9.362µm volumes** (PHerc0139 w035, w039, w040, w041, w044):
   - Already roughly isotropic at native resolution
   - Used directly, no pooling needed

**Why isotropic matters**: 3D convolutions assume roughly cubic receptive fields. Anisotropic data (2.4µm XY, 9.6µm Z) would make depth kernels "see" 4× coarser than XY kernels, breaking the 3D inductive bias.

### Training Details
- **Batch Size**: 64 patches total
- **Fixed Scroll Prior** (stratified sampling):
  - PHerc0139: 29 patches/batch
  - Scroll 1667: 22 patches/batch
  - PHerc Paris4: 11 patches/batch
  - PHerc0814: 2 patches/batch
- **Loss**: BCE (label_smoothing=0.5) + Dice
- **Training Data**: 24 aligned segments + 5 native 9.362µm segments = 29 total
- **Labels**: Single-slice annotation (Z=10 for aligned, Z=14 for native)
- **Checkpoints**: 2 training runs (seed42, seed43), 7 checkpoints each (step 10k-75k)

### Output Calibration
**Important**: Due to label smoothing 0.5, model outputs are shifted:
- No-ink confident prediction: **~0.25** (not 0.0)
- Ink confident prediction: **~0.75** (not 1.0)
- Display rescale: `(pred - 0.25) / 0.5` to map to [0, 1]

---

## CRITICAL DIFFERENCES THAT MAKE VILLA BETTER

### 1. **ARCHITECTURE: 3D Stem → 2D U-Net vs 2-Stage MIL**

#### VILLA Model (`Local3DStem2DUNet`)
```python
# 3D Stem: LocalDepthFusionStem
Conv3d(1, 16, k=3) → InstanceNorm3d → LeakyReLU
Conv3d(16, 16, k=3) → InstanceNorm3d → LeakyReLU

# Depth Fusion: TWO parallel branches
1. Attention-pooled: softmax(Conv3d(16,1,k=1)) → weighted sum → 16 channels
2. Max-pooled: direct max → 16 channels
Output: Concatenate [attention, max] → 32 channels (2D)

# 2D U-Net backbone processes the fused 2D representation
```

**Why this works**:
- **Depth attention learns where ink is** (soft argmax over depth slices)
- **Max pooling preserves strong local features** even if depth attention fails
- **TWO branches = robustness**: attention can fail on ambiguous tiles, max acts as fallback
- **InstanceNorm3d** instead of BatchNorm3d: handles per-tile statistics, robust to batch composition
- **LeakyReLU (slope=0.01)** instead of ReLU: prevents dead neurons

#### OUR Model (`InkDetectorTwoStageWideZGradCtx`)
```python
# Stage 1: Per-slice 2D convs (depth kernel=1, NO cross-depth interaction in stem)
Conv3d(3, 32, k=(1,3,3))  # [raw, lcn, dI/dz]
Conv3d(32, 64, k=(1,3,3))

# Depth mixing happens LATER in stage 1 (after 2D features extracted)
Conv3d(64, 128, k=3) → CBAM → MaxPool(1,2,2)
Conv3d(128, 256, k=3) → CBAM

# Stage 2: 3-window fusion (3 separate 8-slice windows, tied weights)
# MIL-LSE aggregation → single scalar tile logit
```

**Why ours is weaker**:
- **No early depth attention**: stem treats each slice independently
- **Depth mixing comes too late**: 2D features already extracted before seeing cross-depth patterns
- **Single pooling strategy** (CBAM): no fallback if attention fails
- **BatchNorm3d**: less robust to varying batch composition (scroll diversity)
- **ReLU**: risk of dead neurons during training
- **MIL-LSE aggregation throws away spatial structure**: reduces to single scalar, loses fine-grained localization

---

### 2. **NORMALIZATION: Robust-MAD vs BatchNorm + Manual LCN**

#### VILLA Normalization
```json
"image_normalization": {
  "mode": "robust_mad",
  "percentile_lower": 1.0,
  "percentile_upper": 99.0
}
```
**Robust-MAD** (Median Absolute Deviation):
```python
# Pseudo-code
clip = np.clip(image, percentile(1), percentile(99))
median = np.median(clip)
mad = np.median(np.abs(clip - median))
normalized = (image - median) / (mad * 1.4826)  # scale to match std
```

**Why this is superior**:
- **Outlier-robust**: MAD is resistant to extreme voxels (scan artifacts, metal fragments)
- **Scroll-adaptive**: median/MAD computed per-patch, adapts to scroll energy level
- **Consistent across scrolls**: different keV energies (113 vs 88 vs 70) automatically normalized
- **No batch dependency**: works identically in train/val/inference (unlike BatchNorm)

#### OUR Normalization
```python
# Global z-score normalization (precomputed scroll-level mean/std)
x = (x - mean_scroll) / std_scroll

# Then manual LCN per-tile (in forward pass)
lcn = _lcn2d(x, k=5)  # local mean/std per 5x5 window
```

**Why ours is weaker**:
- **Fixed scroll statistics**: fails when test scroll has different energy level (e.g., PHerc0139 9µm at 88keV)
- **Not outlier-robust**: mean/std sensitive to bright artifacts
- **Two-stage normalization is redundant**: global z-score THEN local LCN double-processing
- **BatchNorm in model**: train/val statistics differ, inference may drift

---

### 3. **LOSS: Smoothed BCE + Dice vs GCE + Ranking**

#### VILLA Loss
```json
"loss": {
  "bce_label_smoothing": 0.5,    // VERY AGGRESSIVE smoothing
  "dice_label_smoothing": 0.0
}
```
**Combined BCE + Dice**:
```python
# BCE with heavy smoothing (0→0.5, 1→0.5)
target_smooth = target * 0.5 + 0.5 * (1 - target)
bce = F.binary_cross_entropy_with_logits(pred, target_smooth)

# Dice loss (IoU-based, class-balanced)
dice = 1 - (2 * intersection + ε) / (pred_sum + target_sum + ε)

loss = bce + dice
```

**Why this is better for DENSE outputs**:
- **Dice directly optimizes IoU** (the metric that matters for segmentation)
- **Class-balanced**: Dice inherently handles ink/papyrus imbalance (unlike BCE)
- **Label smoothing = 0.5**: treats all labels as "50% confident" → prevents overfitting to noisy labels
- **Dense supervision**: every pixel contributes gradient (vs our single scalar per tile)

#### OUR Loss
```python
# GCE (Generalized Cross Entropy, q=0.7)
gce = (1 - (p_t ** q)) / q

# + Pairwise ranking loss
ranking = max(0, margin - (p_pos - p_neg))

# + Optional SupCon, DANN, etc.
loss = gce + λ_rank * ranking + λ_sc * supcon + ...
```

**Why ours is weaker for dense learning**:
- **GCE is for TILE-level labels**: designed for noisy scalar labels, not dense pixel supervision
- **Ranking loss needs tile-level ordering**: doesn't apply to dense pixel predictions
- **No Dice term**: missing the IoU-optimizing component that villa has
- **MIL aggregation loses spatial info**: can't do dense supervision with scalar tile labels

---

### 4. **TRAINING: Fixed Scroll Prior vs Random Sampling**

#### VILLA Sampling
```json
"sampling_strategy": "fixed_scroll_prior_stratified",
"fixed_scroll_prior": {
  "target_batch_counts": {
    "0139": 29,   // PHerc0139 (9 segments, native 9µm + 4x pooled 2.4µm)
    "1667": 22,   // Scroll 1667 (6 segments)
    "Paris4": 11, // PHerc Paris 4 (8 segments)
    "0814": 2     // PHerc0814 (1 segment)
  }
}
```
**Every batch**: 29+22+11+2 = 64 patches, fixed scroll distribution

**Why this is critical**:
- **Balanced scroll representation**: prevents model from overfitting to dominant scroll
- **Scroll diversity per batch**: every gradient update sees all scroll domains
- **Stratified sampling**: rare scrolls (0814) guaranteed representation
- **Stable training**: consistent batch composition → stable BatchNorm statistics

#### OUR Sampling
```python
# Random sampling with scroll weights (config.data.scrolls)
# Batch composition varies randomly each iteration
```

**Why ours is weaker**:
- **Imbalanced batches**: some batches may be 100% one scroll (PHerc0139)
- **No scroll diversity guarantee**: model can overfit to dominant scroll's characteristics
- **Unstable BatchNorm**: varying scroll mix → noisy running statistics
- **Rare scrolls underrepresented**: small scrolls (w044) rarely sampled

---

### 5. **DATA AUGMENTATION: Depth Jitter**

#### VILLA Depth Jitter
```json
"flat_z_window_jitter": {
  "enabled": true,
  "window_depth": 17,      // use 17 of 21 available slices
  "max_offset": 2,         // jitter ±2 slices
  "probability": 1.0,
  "padding": "forbidden"   // must fit in 21-slice volume
}
```

**Why this prevents depth overfitting**:
- **Forces depth-invariant features**: can't memorize "ink is always at absolute slice 12"
- **Augments depth dimension**: effectively 5× more training data (offset ∈ {0,1,2,3,4})
- **Models papyrus undulation**: ink depth varies ±2 slices due to surface waviness
- **Prevents absolute-depth shortcuts**: model must recognize ink texture, not depth coordinate

#### OUR Augmentation
```python
# Spatial augmentations only: flip, rotate, noise, brightness, cutout
# Depth is FIXED: always slices 4-28, no jitter
# Depth positional encoding is LEARNED but fixed per absolute depth
```

**Why ours is weaker**:
- **Absolute depth memorization**: model learns "ink appears at depth 12" (papyrus surface)
- **No depth augmentation**: only spatial diversity
- **Fixed depth PE**: positional encoding locks onto absolute depths

---

## WHAT WE CAN ADOPT (Binary-Compatible)

### ✅ **1. Depth Attention Stem** (HIGH PRIORITY)
Replace our per-slice stem with villa's `LocalDepthFusionStem`:
```python
class LocalDepthFusionStem(nn.Module):
    def __init__(self, channels=16):
        self.features = nn.Sequential(
            nn.Conv3d(1, channels, k=3, p=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01),
            nn.Conv3d(channels, channels, k=3, p=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.01),
        )
        self.attention_logits = nn.Conv3d(channels, 1, k=1)
    
    def forward(self, x):
        f = self.features(x)  # (B, C, D, H, W)
        # Depth attention: where is the ink?
        attn = torch.softmax(self.attention_logits(f), dim=2)  # (B, 1, D, H, W)
        attn_pooled = (f * attn).sum(dim=2)  # (B, C, H, W)
        max_pooled = f.amax(dim=2)           # (B, C, H, W)
        return torch.cat([attn_pooled, max_pooled], dim=1)  # (B, 2C, H, W)
```

**Integration**: Replace `self.per_slice` in Stage 1 with this stem, output feeds into our existing `depth_mix` stage.

---

### ✅ **2. Robust-MAD Normalization** (HIGH PRIORITY)
```python
def robust_mad_normalize(patch, p_low=1, p_high=99):
    """Per-patch robust normalization using MAD."""
    clip = torch.clamp(patch, 
                       torch.quantile(patch, p_low/100), 
                       torch.quantile(patch, p_high/100))
    median = torch.median(clip)
    mad = torch.median(torch.abs(clip - median))
    return (patch - median) / (mad * 1.4826 + 1e-8)
```

**Integration**: Apply in `Dataset.__getitem__` BEFORE the model sees data. Remove global scroll-level normalization.

---

### ✅ **3. Depth Window Jitter** (MEDIUM PRIORITY)
```python
class DepthJitterAugmentation:
    def __init__(self, total_depth=24, window_depth=17, max_offset=2):
        self.total = total_depth
        self.window = window_depth
        self.max_offset = max_offset
    
    def __call__(self, volume):
        """volume: (C, D=24, H, W)"""
        if self.training:
            offset = random.randint(0, self.max_offset)
            return volume[:, offset:offset+self.window, :, :]
        else:
            # Inference: center window
            offset = (self.total - self.window) // 2
            return volume[:, offset:offset+self.window, :, :]
```

**Integration**: Add to `Transform` augmentation pipeline. Change `depth=24` → `depth=17` in config during training.

---

### ✅ **4. InstanceNorm3d + LeakyReLU** (LOW PRIORITY, EASY WIN)
```python
# REPLACE everywhere in model.py:
nn.BatchNorm3d(C) → nn.InstanceNorm3d(C, affine=True)
nn.ReLU() → nn.LeakyReLU(negative_slope=0.01, inplace=True)
```

**Why**: More stable, fewer dead neurons, better batch-invariance.

---

### ✅ **5. Fixed Scroll Prior Sampling** (MEDIUM PRIORITY)
```python
class FixedScrollPriorSampler:
    def __init__(self, datasets, batch_size=64, scroll_counts=None):
        # scroll_counts: {'0139': 29, '1667': 22, ...}
        self.scroll_groups = {sid: [] for sid in scroll_counts}
        for i, ds in enumerate(datasets):
            self.scroll_groups[ds.scroll_id].append(i)
        self.counts = scroll_counts
    
    def sample_batch(self):
        batch = []
        for scroll_id, count in self.counts.items():
            indices = self.scroll_groups[scroll_id]
            batch.extend(random.choices(indices, k=count))
        return batch
```

**Integration**: Replace default DataLoader sampler.

---

## LABEL SUPERVISION: SINGLE-SLICE DENSE vs TILE-LEVEL SPARSE

**Critical Discovery from HuggingFace README**:

### VILLA Labels (Single-Slice Dense)
```
labels/aligned-scrollprizeorg-21slices/<segment>/<segment>_inklabels.zarr
Shape: (21, H, W)  — but only Z=10 is annotated, other slices are zeros
```

**What this means**:
- Labels are **2D pixel masks** at a single depth plane (Z=10 for aligned, Z=14 for native)
- Model predicts **128×128 pixel segmentation** at that depth
- Loss is **per-pixel BCE + Dice** over the annotated slice
- Other 20 depth slices are IGNORED during training (supervision_mask filters them out)

**Why this works for Villa**:
- **Dense spatial supervision**: 128×128 = 16,384 pixels per patch vs our 1 scalar per tile
- **2D U-Net matches label structure**: outputs (B, 1, H, W) prediction map at the annotated slice
- **Depth fusion happens in stem**: 3D stem collapses 17 slices → single 2D representation
- **Single-slice annotation is practical**: humans annotate 2D slices, not full 3D volumes

### OUR Labels (Tile-Level Sparse)
```
eroded_inklabels/<scroll_id>.png  (2D binary map, 1 value per tile_size×tile_size region)
Shape: (H/tile_size, W/tile_size)  — typically (H/16, W/16)
```

**What this means**:
- Labels are **tile-level binary scalars** (16×16 tile → 1 or 0)
- Model predicts **1 scalar** per tile via MIL aggregation over 24×16×16 voxels
- Loss is **scalar BCE** per tile (256× fewer supervised signals than Villa)
- **NO depth annotation**: tile label = 1 if ANY depth slice in that tile has ink

**Why this is fundamentally different**:
- **Cannot use U-Net decoder**: no dense pixel targets to supervise (H, W) output
- **Must use MIL aggregation**: reduce 24×16×16 voxels → 1 scalar somehow
- **Depth ambiguity**: tile label=1 doesn't tell us WHICH of 24 slices has ink
- **Coarser spatial resolution**: 16×16 tiles vs 128×128 dense pixels (64× fewer labels)

### Architectural Implications

✅ **CAN ADOPT from Villa**:
- LocalDepthFusionStem (3D stem with attention+max branches)
- Robust-MAD normalization (per-patch, depth-independent)
- InstanceNorm3d + LeakyReLU (architecture details)
- Depth jitter augmentation (prevents depth overfitting)
- Fixed scroll sampling (training stability)

❌ **CANNOT ADOPT from Villa**:
- 2D U-Net decoder (requires dense pixel labels at fixed depth)
- Dice loss (requires spatial overlap between prediction and ground truth maps)
- 128×128 dense pixel predictions (we have only tile-level scalars)

🔄 **HYBRID APPROACH (What We Should Do)**:
Villa's stem architecture is COMPATIBLE with our MIL head:
```
[Input: 24×48×48] 
  → LocalDepthFusionStem (Villa's 3D stem)
  → [Output: 32×48×48 2D features]
  → Spatial downsampling + depth re-expansion (NEW adapter layer)
  → [Output: D'×H'×W' voxel features]
  → MIL-LSE aggregation (our existing head)
  → [Output: 1 scalar tile logit]
```

This gives us Villa's depth fusion benefits while preserving tile-level supervision compatibility.

---

## WHAT WE CANNOT ADOPT (Requires Dense Labels)

## WHAT WE CANNOT ADOPT (Incompatible with Tile-Scalar Labels)

### ❌ **1. 2D U-Net Decoder**
Villa's decoder produces dense segmentation maps (B, 1, H, W) for single-slice supervision.  
We have tile-scalar labels (B,) requiring MIL aggregation.

**Why incompatible**: U-Net decoder expects per-pixel targets. Our eroded inklabels are binary per-tile, not dense masks.

---

### ❌ **2. Dice Loss**
Dice = `2×(pred∩target) / (pred+target)` requires spatial overlap computation.  
We have scalar predictions (1 value per tile), not segmentation maps.

**Why incompatible**: Dice operates on 2D masks. Our BCE loss is already optimal for binary scalars.

---

### ❌ **3. 128×128 Patch Size at 9µm** (Storage/Memory Constraint)
Villa uses 128×128×17 patches (278k voxels) with stride 32 (75% overlap).  
Our 48×48×24 context (55k voxels) already pushes memory limits on 24GB VRAM.

**Why risky to adopt**:
- 5× more voxels per sample → 5× GPU memory (would need batch_size=1 or smaller tiles)
- Dense 32px stride sampling → 16× more training samples per epoch (4× longer training)
- Tile-level labels don't benefit from dense spatial sampling (no sub-tile detail to learn)

**Verdict**: Keep context=48, ds=2 (effective 96×96 voxel grid at 2× coarser resolution).

---

## RECOMMENDED IMPLEMENTATION PRIORITY

### Phase 1: **Core Architecture** (1-2 weeks)
1. ✅ Implement `LocalDepthFusionStem` (dual attention+max branches)
2. ✅ Replace BatchNorm3d → InstanceNorm3d
3. ✅ Replace ReLU → LeakyReLU(0.01)
4. Test on baseline (ctx=48, ds=2) and compare PR-AUC

### Phase 2: **Normalization** (3-5 days)
1. ✅ Implement `robust_mad_normalize` in dataset pipeline
2. Remove global scroll-level normalization
3. Verify on multi-scroll eval (PHerc0139, 1667, etc.)

### Phase 3: **Training Dynamics** (1 week)
1. ✅ Implement depth jitter augmentation (17 of 24 slices, ±2 offset)
2. ✅ Implement fixed scroll prior sampler
3. Retrain baseline and compare convergence speed

### Phase 4: **Experiment** (2-3 weeks)
1. Run full ablation: each component vs baseline
2. Combine all winning components
3. Test on held-out scrolls (Scroll1, PHerc0051, etc.)

---

## EXPECTED GAINS

Based on villa's performance vs ours:
- **Depth attention stem**: +3-5% PR-AUC (eliminates per-slice processing weakness)
- **Robust-MAD normalization**: +2-4% PR-AUC (cross-scroll generalization)
- **Depth jitter**: +1-2% PR-AUC (prevents depth overfitting)
- **Fixed scroll sampling**: +1-2% PR-AUC (stable training, better scroll balance)
- **InstanceNorm + LeakyReLU**: +0.5-1% PR-AUC (training stability)

**Total expected gain**: +7-14% PR-AUC  
**Current best (depsc_high)**: 0.60507 PR-AUC  
**Projected with villa components**: **0.65-0.69 PR-AUC**

---

## RISKS & MITIGATIONS

### Risk 1: **Depth Attention May Fail on Mushy Scrolls**
Villa's depth attention learns to find the surface, but mushy/compressed regions have no clear surface.  
**Mitigation**: The max-pooling branch acts as fallback. If attention fails, max preserves strong features.

### Risk 2: **Robust-MAD Sensitive to Outliers in Tiny Patches**
MAD computed on 17×128×128 patches (280k voxels) is stable, but edge cases (metal fragments) may skew.  
**Mitigation**: Add clipping at 99th percentile before MAD calculation (villa does this).

### Risk 3: **Depth Jitter Reduces Effective Context**
17 slices instead of 24 means less depth coverage.  
**Mitigation**: Villa's results prove 17 is sufficient. Our current model uses 8 slices per window; 17 is already 2× more.

### Risk 4: **Fixed Scroll Prior May Underfit Rare Scrolls**
PHerc0814 only gets 2/64 samples per batch.  
**Mitigation**: Villa's performance proves this is acceptable. Rare scrolls still contribute, just less frequently.

---

## CONCLUSION

Villa's model is better because:
1. **3D stem with depth attention** finds ink depth automatically
2. **Robust-MAD normalization** handles multi-energy scrolls
3. **Dense supervision** gives 64× more gradient signal (we can't adopt this)
4. **Depth jitter** prevents overfitting to absolute depth
5. **Fixed scroll sampling** balances training across scrolls

**We can adopt 1, 2, 4, 5** without changing our MIL/binary framework.

**Start with Phase 1 (depth attention stem).** This is the biggest architectural win and compatible with our current pipeline.
