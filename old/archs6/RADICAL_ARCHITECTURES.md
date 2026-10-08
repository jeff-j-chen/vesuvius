# Radical Architectures Implementation Summary

## Overview
Successfully implemented 6 cutting-edge deep learning architectures for 3D ink detection in the Vesuvius Challenge. All architectures have been smoke-tested and validated for integration with the campaign pipeline.

## Architectures Implemented

### 1. ViT3D - Vision Transformer 3D (test 39)
**Parameters**: ~2.1M  
**Key Innovation**: Pure self-attention architecture adapted for 3D volumes

**Architecture**:
- Divides 3D volume into 4×4×4 patches
- Patch embedding: Conv3d stride=4 → 256 channels
- 4-layer Transformer encoder (8 attention heads)
- Global average pooling + classification head

**Why It Matters**: Tests whether transformer's global receptive field can capture long-range ink patterns better than CNNs. Transformers excel at modeling global dependencies.

**Expected Behavior**: May struggle initially (transformers need more data), but could discover novel depth-spatial correlations.

---

### 2. Swin3D - Swin Transformer 3D (test 40)
**Parameters**: ~118K  
**Key Innovation**: Shifted window attention for computational efficiency

**Architecture**:
- Patch embedding with stride-4
- Window-based multi-head attention (simplified)
- Hierarchical structure with MLP blocks
- Efficient O(n) complexity vs O(n²) for standard ViT

**Why It Matters**: Tests whether local-then-global attention (like CNN inductive bias) works better than pure global attention for scroll data.

**Expected Behavior**: More parameter-efficient than ViT. May converge faster with better local feature learning.

---

### 3. ConvNeXt3D - Modernized CNN (test 41)
**Parameters**: ~436K  
**Key Innovation**: CNN with transformer-era design choices

**Architecture**:
- Large 7×7×7 depthwise convolutions
- Inverted bottleneck (expand 4× then compress)
- BatchNorm + GELU activation
- Residual connections

**Why It Matters**: Tests whether "pure CNN can still compete" - combines CNN's strong inductive bias with modern training techniques.

**Expected Behavior**: Should perform well - combines locality bias with modern design. May be best baseline.

---

### 4. XCiT3D - Cross-Covariance Attention (test 42)
**Parameters**: ~207K  
**Key Innovation**: Cross-covariance attention instead of standard Q-K-V

**Architecture**:
- Patch embedding
- Cross-covariance attention: (Q^T @ K) instead of (Q @ K^T)
- Operates on channel dimension rather than spatial
- More efficient than standard attention

**Why It Matters**: Tests whether feature-to-feature attention (rather than patch-to-patch) better captures ink's chemical signature.

**Expected Behavior**: Lightweight and efficient. May excel if ink detection is more about "which features co-occur" than spatial layout.

---

### 5. nnU-Net3D - Medical Imaging Standard (test 43)
**Parameters**: ~5.6M (largest)  
**Key Innovation**: Encoder-decoder with deep supervision, designed for medical 3D

**Architecture**:
- U-Net encoder-decoder with skip connections
- Instance normalization (robust to intensity variations)
- Deep supervision at multiple scales
- LeakyReLU activation

**Why It Matters**: Medical imaging gold standard - scroll CT is fundamentally a medical imaging task. Deep supervision encourages multi-scale features.

**Expected Behavior**: Likely strong performer - medical imaging is closest domain. High capacity may overfit on small w044.

---

### 6. SlotAttention3D - Object-Centric Learning (test 44)
**Parameters**: ~1.1M  
**Key Innovation**: Decomposes scene into "slots" (object-centric representations)

**Architecture**:
- CNN feature extractor
- 4 learnable "slots" (ink pattern prototypes)
- Iterative attention refinement (3 iterations)
- GRU updates for slot evolution

**Why It Matters**: Tests whether ink detection benefits from discovering compositional structure - e.g., "ink = carbon blob + papyrus boundary + depth signature".

**Expected Behavior**: May discover interpretable ink prototypes. Could fail if ink isn't compositional. High risk, high reward.

---

## Smoke Test Results

### Standard Input (24×16×16)
✅ All 6 architectures pass  
- Forward pass works
- Output shape correct (B, 1)
- No runtime errors

### Context Input (24×48×48)
✅ All 6 architectures pass  
- Adaptive to input size
- No hardcoded dimensions
- Ready for campaign deployment

### Campaign Integration
✅ All 6 architectures registered  
- Dry-run successful for all
- Config properly parsed
- Ready for overnight training

---

## Testing Strategy

Each architecture will train on **w044 single-scroll** for **12 epochs** with:
- No augmentation (pure overfitting test)
- Fast eval (16% area for speed)
- Depth SupCon disabled (incompatible with some archs)
- Same data/optimization for fair comparison

**Success Metrics**:
1. **Training converges** - model learns something
2. **PR-AUC > 0.5** - better than random
3. **Overfits well** - high train PR-AUC (no aug = should memorize)

**Learning Goals**:
- Which inductive biases help? (CNN locality vs Transformer global)
- Do transformers need more data? (ViT vs Swin vs ConvNeXt)
- Does object-centric help? (SlotAttention decomposition)
- Is U-Net domain transfer useful? (medical → scrolls)

---

## Expected Outcomes

**Likely Winners**:
1. **ConvNeXt3D** - Modern CNN, proven on image tasks
2. **nnU-Net3D** - Medical imaging gold standard
3. **Swin3D** - Efficient transformer with locality

**Interesting Experiments**:
4. **ViT3D** - May need more data, test pure attention
5. **XCiT3D** - Feature-wise attention, novel approach
6. **SlotAttention3D** - Compositional learning, high variance

**Key Comparisons**:
- ViT3D vs Swin3D: Global vs local-first attention
- ConvNeXt3D vs nnU-Net3D: Modern CNN vs medical U-Net
- All vs baseline v16_arch_ctx: Do modern architectures beat proven depth-aware CNN?

---

## Files Created
- `utils/model.py`: All 6 architectures implemented (lines 1408-1806)
- `_test_radical_archs.py`: Standard smoke test
- `_test_radical_context.py`: Context input validation
- `RADICAL_ARCHITECTURES.md`: This summary

**Campaign Command**:
```bash
python campaign_archs_7.py --only vit3d,swin3d,convnext3d,xcit3d,nnunet3d,slot3d
```

---

## Technical Notes

**All architectures**:
- Accept (B, 1, D, H, W) input where D=24, H=W=16 or 48
- Output (B, 1) tile score for MIL
- Support both tile_size=16 and context_size=48
- Initialized with Xavier/He initialization
- Compatible with existing training pipeline

**Key Design Decisions**:
- Lightweight models (100K-5M params) for fast training
- Simplified implementations capturing core innovations
- No pretrained weights (train from scratch)
- Adaptive to input size (no hardcoded shapes)

Ready for overnight experiments! 🚀
