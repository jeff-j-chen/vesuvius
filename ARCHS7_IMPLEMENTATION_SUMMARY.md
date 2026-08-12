# Campaign Archs 7: Implementation Summary
**Date**: 2026-08-11  
**Status**: Configuration complete, ready for architecture implementation  
**Total Tests**: 44

---

## Campaign Overview

Campaign 7 is a comprehensive architectural exploration testing **44 distinct architectural variants** on a single scroll (w044) to isolate architecture effects from multi-scroll domain diversity.

### Design Philosophy
- **Builds on proven patterns**: depth preservation, large context, wide head, zgrad, lcn, attn-MIL, supcon
- **Avoids proven failures**: early depth squashing, dense pixel-level architectures
- **Explores new frontiers**: dual-stream processing, Villa-inspired depth attention, modern attention mechanisms, advanced fusion strategies

### Configuration
- **Scroll**: w044 (20260115000000)
- **Epochs**: 12 (fast iteration on small scroll)
- **Augmentation**: NONE (overfit test - can the model learn at all?)
- **Evaluation**: End-of-training only (eval_int=12)
- **Fast eval**: Bottom-left 40%×40% = 16% area (6.25× speedup)
- **Baseline**: depsc_high from archs5 (depth SupCon λ=0.3, ctx=48 ds=2, PR-AUC=0.605)

---

## Test Organization (44 tests)

### Group 1-2: Sanity Checks (2 tests)
Verify infrastructure works, establish baseline.

| Test ID | Tag | Purpose |
|---------|-----|---------|
| 1 | w044_sanity | Standard depsc_high config verification |
| 2 | w044_noaug | Overfit test (same as #1, confirms data loading) |

### Group 3-6: Context Ablation (4 tests)
Test receptive field size (Villa uses 128×128, ours is 48×48).

| Test ID | Tag | Context | Downsample | Effective Resolution |
|---------|-----|---------|------------|---------------------|
| 3 | ctx96_ds2 | 96×96 | 2× | 48×48 (same as baseline) |
| 4 | ctx64_ds2 | 64×64 | 2× | 32×32 (smaller than baseline) |
| 5 | ctx128_ds2 | 128×128 | 2× | 64×64 (larger, may OOM) |
| 6 | ctx96_fovea | 96×96 | 3× surround, 1× center | Sharp center + wide context |

### Group 7-10: Dual-Stream Depth (4 tests)
Parallel squashed (2D) + non-squashed (3D) processing.

| Test ID | Tag | Fusion Strategy | Architecture |
|---------|-----|----------------|--------------|
| 7 | dual_early | Fuse after stems | v16_dual_stream_early |
| 8 | dual_late | Fuse at logits (ensemble) | v16_dual_stream_late |
| 9 | dual_gated | Learnable gate weights | v16_dual_stream_gated |
| 10 | dual_asym | Lightweight squashed + heavy 3D | v16_dual_stream_asym |

**Expected Winner**: dual_asym (best of 2D context + 3D detail)

### Group 11-14: Hybrid Depth Attention (4 tests)
Villa's dual-branch attention adapted to multi-window needs.

| Test ID | Tag | Attention Scope | Architecture |
|---------|-----|----------------|--------------|
| 11 | hybrid_win | Per-window (3 windows) | v16_hybrid_depth_per_window |
| 12 | hybrid_global | Global (all 24 slices, risky) | v16_hybrid_depth_global |
| 13 | hybrid_triple | Triple-branch (attn+max+mean) | v16_hybrid_depth_triple |
| 14 | hybrid_gated | Learnable mix attn vs max | v16_hybrid_depth_gated |

**Expected Winner**: hybrid_win (Villa robustness + our multi-window coverage)  
**Expected Failure**: hybrid_global (conflicts with "no early squashing")

### Group 15-20: Multi-Scale & Efficient (6 tests)

| Test ID | Tag | Technique | Architecture |
|---------|-----|-----------|--------------|
| 15 | multiscale | Multi-scale pyramid (1x, 0.5x, 0.25x) | v16_multiscale_pyramid |
| 16 | depth_se | Depth squeeze-excitation | v16_depth_se |
| 17 | depthsep | Depthwise-separable 3D convs | v16_depthwise_sep |
| 18 | mixed_win | 5 windows (3 non-overlap + 2 at seams) | v16_mixed_depth_windows |
| 19 | octave | Octave convs (high/low freq) | v16_octave_conv |
| 20 | efficient | EfficientNet compound scaling | v16_efficientnet_scale |

**Expected Winners**: multiscale (proven across domains), depth_se (lightweight), mixed_win (builds on 5-window success)

### Group 21-26: Attention Mechanisms (6 tests)

| Test ID | Tag | Mechanism | Architecture |
|---------|-----|-----------|--------------|
| 21 | nonlocal | Non-local depth self-attention | v16_nonlocal_depth |
| 22 | coord_attn | Coordinate attention (CVPR 2021) | v16_coord_attention |
| 23 | deform | Deformable 3D convolutions | v16_deformable_conv |
| 24 | progressive | Progressive depth refinement | v16_progressive_depth |
| 25 | dual_attn | Dual channel + spatial | v16_dual_attention |
| 26 | axial | Axial attention (H, W, D axes) | v16_axial_attention |

**Expected Winners**: coord_attn (modern, proven), progressive (aligns with curriculum learning)

### Group 27-32: Advanced Fusion (6 tests)

| Test ID | Tag | Technique | Architecture |
|---------|-----|-----------|--------------|
| 27 | fpn | Feature Pyramid Network | v16_fpn |
| 28 | bifpn | Bi-directional FPN | v16_bifpn |
| 29 | ghost | Ghost convolutions | v16_ghost_conv |
| 30 | inverted | Inverted residuals (MobileNetV2) | v16_inverted_residual |
| 31 | resnext | ResNeXt grouped convs | v16_resnext_groups |
| 32 | depth_shift | Depth shift module (TSM adapted) | v16_depth_shift |

**Expected Winners**: fpn (proven for multi-scale), bifpn (iterative refinement)

### Group 33-34: Archs5 Proven (2 tests)
Quick sanity checks from previous campaign.

| Test ID | Tag | Technique | Config Change |
|---------|-----|-----------|---------------|
| 33 | gce_noise | GCE loss q=0.9 | gce_q=0.9 |
| 34 | five_win | 5 depth windows | n_depth_windows=5 |

### Group 35-38: Villa Normalization (4 tests)
Villa's normalization strategies (depth attention covered in group 11-14).

| Test ID | Tag | Normalization | Config |
|---------|-----|---------------|--------|
| 35 | villa_mad_norm | Robust-MAD | normalization_mode="robust_mad" |
| 36 | villa_groupnorm | GroupNorm + LeakyReLU | normalization_layer="group", activation="leaky" |
| 37 | villa_instancenorm | InstanceNorm + LeakyReLU (risky for small tiles) | normalization_layer="instance", activation="leaky" |
| 38 | villa_layernorm | LayerNorm + LeakyReLU | normalization_layer="layer", activation="leaky" |

**Expected Winner**: villa_groupnorm (batch-independent, stable)

### Group 39-44: Radical Architectures (6 tests)
Transformer-based and novel architectures from archs6.

| Test ID | Tag | Architecture | Expected Outcome |
|---------|-----|--------------|------------------|
| 39 | vit3d | 3D Vision Transformer | Likely fail (needs dense labels) |
| 40 | swin3d | Swin Transformer 3D | Better than ViT (local bias) |
| 41 | convnext3d | ConvNeXt 3D | Most promising transformer |
| 42 | xcit3d | XCiT 3D | Marginal vs ViT |
| 43 | nnunet3d | nnU-Net 3D | Fail (needs dense labels) |
| 44 | slot3d | Slot Attention 3D | Fail (needs object-level labels) |

**Expected Winner**: convnext3d (still fundamentally CNN-based)

---

## Implementation Status

### ✅ COMPLETE: Configuration & Planning
- [x] campaign_archs_7.py created with all 44 test configurations
- [x] _BASE7 dict with depsc_high baseline settings
- [x] _mk7() helper function for test creation
- [x] TESTS list populated with all 44 variants
- [x] Proper test numbering and organization
- [x] Documentation (ARCHS7_ARCHITECTURE_GUIDE.md)

### ⚠️ PENDING: Architecture Implementation
The following architecture classes need to be implemented in `utils/model.py`:

#### Dual-Stream (tests 7-10) - 4 architectures
- [ ] `InkDetectorDualStreamEarly` (v16_dual_stream_early)
- [ ] `InkDetectorDualStreamLate` (v16_dual_stream_late)
- [ ] `InkDetectorDualStreamGated` (v16_dual_stream_gated)
- [ ] `InkDetectorDualStreamAsym` (v16_dual_stream_asym)

#### Hybrid Depth Attention (tests 11-14) - 4 architectures
- [ ] `InkDetectorHybridDepthPerWindow` (v16_hybrid_depth_per_window)
- [ ] `InkDetectorHybridDepthGlobal` (v16_hybrid_depth_global)
- [ ] `InkDetectorHybridDepthTriple` (v16_hybrid_depth_triple)
- [ ] `InkDetectorHybridDepthGated` (v16_hybrid_depth_gated)

#### Multi-Scale & Efficient (tests 15-20) - 6 architectures
- [ ] `InkDetectorMultiscalePyramid` (v16_multiscale_pyramid)
- [ ] `InkDetectorDepthSE` (v16_depth_se)
- [ ] `InkDetectorDepthwiseSep` (v16_depthwise_sep)
- [ ] `InkDetectorMixedDepthWindows` (v16_mixed_depth_windows)
- [ ] `InkDetectorOctaveConv` (v16_octave_conv)
- [ ] `InkDetectorEfficientScale` (v16_efficientnet_scale)

#### Attention Mechanisms (tests 21-26) - 6 architectures
- [ ] `InkDetectorNonLocalDepth` (v16_nonlocal_depth)
- [ ] `InkDetectorCoordAttention` (v16_coord_attention)
- [ ] `InkDetectorDeformableConv` (v16_deformable_conv)
- [ ] `InkDetectorProgressiveDepth` (v16_progressive_depth)
- [ ] `InkDetectorDualAttention` (v16_dual_attention)
- [ ] `InkDetectorAxialAttention` (v16_axial_attention)

#### Advanced Fusion (tests 27-32) - 6 architectures
- [ ] `InkDetectorFPN` (v16_fpn)
- [ ] `InkDetectorBiFPN` (v16_bifpn)
- [ ] `InkDetectorGhostConv` (v16_ghost_conv)
- [ ] `InkDetectorInvertedResidual` (v16_inverted_residual)
- [ ] `InkDetectorResNeXt` (v16_resnext_groups)
- [ ] `InkDetectorDepthShift` (v16_depth_shift)

#### Foveated (test 6) - 1 architecture
- [ ] `InkDetectorTwoStageWideZGradCtxFovea` (v16_arch_ctx_fovea)

#### Existing Architectures (tests 1-5, 33-44) - 0 new
All other tests use existing architectures:
- Tests 1-5: v16_arch_ctx (InkDetectorTwoStageWideZGradCtx)
- Tests 33-34: v16_arch_ctx with config changes only
- Tests 35-38: v16_arch_ctx with normalization flags (needs utils/config.py support)
- Tests 39-44: vit3d, swin3d, convnext3d, xcit3d, nnunet3d, slot3d (from archs6)

**Total new architectures needed**: 27

---

## Configuration Support Needed

### utils/config.py additions
The following new config parameters need to be added:

```python
# Normalization (tests 35-38)
normalization_mode: str = "zscore"  # "zscore" | "robust_mad"
normalization_layer: str = "batch"  # "batch" | "group" | "instance" | "layer"
activation: str = "relu"  # "relu" | "leaky"
```

These were partially added in the Villa analysis phase, but need full integration into DataManager and model forward passes.

---

## Next Steps (Implementation Priority)

### Phase 1: High-Expected-ROI (tests 7-14)
1. Implement dual-stream architectures (tests 7-10)
2. Implement hybrid depth attention (tests 11-14)
3. Test on w044_sanity first to verify infrastructure
4. **Rationale**: These directly address Villa's competitive advantage while preserving our proven depth separation

### Phase 2: Proven Techniques (tests 15-20, 27-32)
1. Implement multi-scale pyramids (test 15, 27-28)
2. Implement efficient modules (tests 16-17)
3. Implement attention mechanisms (tests 21-26)
4. **Rationale**: Well-established techniques from other domains, high success probability

### Phase 3: Experimental (tests 21-26, remaining)
1. Implement remaining attention variants
2. Implement foveated context (test 6)
3. **Rationale**: Novel but less proven, complete comprehensive coverage

### Phase 4: Low-Expected-ROI (tests 39-44)
1. Verify transformer architectures from archs6 still work
2. **Rationale**: Expected to fail but included for completeness

---

## Running the Campaign

```bash
# Verify configuration (will fail without PyTorch in base env)
python campaign_archs_7.py --dry-run

# Run single test for infrastructure verification
python campaign_archs_7.py --only w044_sanity

# Run dual-stream group
python campaign_archs_7.py --only dual_early,dual_late,dual_gated,dual_asym

# Run hybrid depth attention group
python campaign_archs_7.py --only hybrid_win,hybrid_global,hybrid_triple,hybrid_gated

# Run all 44 tests (after architecture implementation)
python campaign_archs_7.py
```

---

## Expected Outcomes

### Top 5 Predicted Winners
1. **dual_asym** (test 10): Asymmetric dual-stream - best of 2D context + 3D detail
2. **hybrid_win** (test 11): Per-window hybrid - Villa robustness + our multi-window coverage
3. **multiscale** (test 15): Multi-scale pyramid - proven across many domains
4. **mixed_win** (test 18): Mixed depth windows - builds on proven 5-window success
5. **depth_se** (test 16): Depth squeeze-excitation - lightweight, effective

### Predicted Top 10 (in order)
1. dual_asym
2. hybrid_win
3. multiscale
4. mixed_win
5. depth_se
6. ctx96_fovea
7. coord_attn
8. progressive
9. fpn
10. depthsep

### Predicted Failures
- **hybrid_global** (test 12): Conflicts with "no early squashing" empirical finding
- **vit3d, swin3d, xcit3d** (tests 39, 40, 42): Need dense labels
- **nnunet3d, slot3d** (tests 43, 44): Need dense/object-level labels
- **dual_late** (test 8): Ensemble needs diversity, both streams too similar

---

## Success Criteria

### Minimum Viable Success
- At least 1 test beats baseline depsc_high (PR-AUC > 0.605)
- w044_noaug successfully overfits (confirms data loading works)

### Good Success
- Top 3 tests beat baseline by >2% relative (PR-AUC > 0.617)
- Dual-stream or hybrid approaches show clear advantage

### Exceptional Success
- Top test beats baseline by >5% relative (PR-AUC > 0.635)
- Multiple architecture families successful (not just one family)
- Clear insights on what architectural patterns work for sparse tile-level ink detection

---

## Notes

### Design Decisions
- **No augmentation**: Makes training deterministic, easier to isolate architecture effects
- **Fast eval (16% area)**: 6.25× speedup, sufficient for architecture ranking
- **Single scroll**: Removes multi-scroll domain diversity as confounding factor
- **12 epochs only**: Fast iteration, w044 is small enough to converge quickly

### Risks & Mitigations
- **Risk**: 27 new architectures is ambitious
  - **Mitigation**: Phased implementation, test infrastructure first with w044_sanity
- **Risk**: Some architectures may OOM (128×128 context, transformers)
  - **Mitigation**: Reduce batch size if needed (batch=32 → 16 → 8)
- **Risk**: No augmentation may lead to all tests failing
  - **Mitigation**: w044_noaug overfit test confirms data loading works

### User Preferences (from memory)
- Prefers minimal, targeted changes
- Wants to be asked before broader system-level refactors
- **Implication**: Implement architectures incrementally, test each group before proceeding
