# Campaign Archs 6: Radical Architecture Search (2026-08-08)

## Platform Configuration ✓

**NEW**: Multi-machine support via `utils/platform.py`

- **Windows (laptop)**: `C:\Users\ChenJeff\Documents\ves_zarrs2`
- **Linux Desktop (EPYC 7702P)**: `/media/jeff/Seagate/ves_zarrs2` ← YOU ARE HERE
- **Linux Runpod (remote)**: `/vesuvius/ves_zarrs2`

Detection: checks for `/media/jeff/Seagate/` existence → auto-configures paths, workers, batch sizes.

Files updated:
- `utils/platform.py` (NEW)
- `utils/config.py` (uses platform-aware defaults)
- `assemble_training_segments.py` (uses `get_zarr_dir()`)
- `assemble_test_segments.py` (uses `get_zarr_dir()`)

Downloads now automatically go to the correct location per machine. No more manual path edits!

---

## Six Radical Architectures (campaign_archs_6.py)

### What We're Testing

Moving BEYOND incremental CNN tweaks. Six paradigm-shifting architectures from cutting-edge vision papers, all adapted to 3D ink detection. Each outputs **EXACTLY 1 binary logit** (tile-level, NOT dense).

### The Architectures

#### 1. **ViT3D** - Pure Vision Transformer
**Paper**: Dosovitskiy et al. 2020 (An Image is Worth 16x16 Words)  
**Hypothesis**: Global receptive field from layer 1 captures long-range correlations CNNs miss.  
**Architecture**: 
- Patch size 4 → 6×6×6 = 216 tokens
- 6 transformer blocks, 8 heads, dim=256
- MLP ratio 4×, droppath 0.1
- CLS token → 1 logit

**Parameters**: ~8.7M  
**Why it might win**: No inductive bias; can learn arbitrary spatial patterns. Global attention sees entire 3D context.

---

#### 2. **Swin3D** - Shifted-Window Hierarchical Transformer
**Paper**: Liu et al. 2021 (Swin Transformer: Hierarchical Vision Transformer using Shifted Windows)  
**Hypothesis**: Local attention windows (efficient) + shifting (global info flow) + hierarchical pooling = best of CNNs + transformers.  
**Architecture**:
- 4 stages: [2,2,6,2] blocks, dims [96,192,384,768]
- Window size 4, shift size 2
- Local attention within windows, shifted every other layer

**Parameters**: ~22M  
**Why it might win**: Hierarchical multi-scale features (like successful CNNs) but with attention's flexibility.

---

#### 3. **ConvNeXt3D** - Modernized CNN
**Paper**: Liu et al. 2022 (A ConvNet for the 2020s)  
**Hypothesis**: CNNs can match transformers IF designed correctly. Larger kernels capture more context.  
**Architecture**:
- Depthwise separable 7×7×7 kernels
- LayerNorm (not BatchNorm), GELU (not ReLU)
- Inverted bottlenecks, layer scale
- 4 stages: [3,3,9,3] blocks, dims [96,192,384,768]

**Parameters**: ~28M  
**Why it might win**: Better inductive bias for 3D spatial structure. Large kernels capture the ring signature directly.

---

#### 4. **XCiT3D** - Cross-Covariance Transformer
**Paper**: El-Nouby et al. 2021 (XCiT: Cross-Covariance Image Transformers)  
**Hypothesis**: XCA models feature interactions explicitly (not just token similarity). O(d²) not O(N²) → efficient.  
**Architecture**:
- Cross-covariance attention: (Q^T K) instead of (Q K^T)
- 12 layers, 8 heads, dim=256
- L2-normalized Q,K with learned temperature

**Parameters**: ~11M  
**Why it might win**: Fine-grained texture discrimination (ink vs papyrus fibers). More efficient than standard attention.

---

#### 5. **nnU-Net3D** - Encoder-Decoder U-Net
**Paper**: Isensee et al. 2021 (nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation)  
**Hypothesis**: Skip connections preserve fine spatial detail. Multi-scale features see both local texture AND global layout.  
**Architecture**:
- 5 encoder stages (64→128→256→512→512)
- 4 decoder stages with skip connections
- Deep supervision heads (ALL output 1 logit, NOT dense)
- Bilinear upsampling

**Parameters**: ~14M  
**Why it might win**: Skip connections bridge fine/coarse scales. Proven for medical imaging (similar domain).  
**CRITICAL**: Deep supervision heads use `AdaptiveAvgPool3d(1)` → each outputs **1 tile-level logit**, NO dense outputs.

---

#### 6. **SlotAttention3D** - Object-Centric Learning
**Paper**: Locatello et al. 2020 (Object-Centric Learning with Slot Attention)  
**Hypothesis**: Ink regions are discrete "objects". Slot attention learns to segment them without localization labels. Highly interpretable.  
**Architecture**:
- CNN encoder: 64→128, 2× MaxPool
- 8 slots, dim=128, 4 iterations
- Iterative attention binding (GRU refinement)
- Pool slots → 1 logit

**Parameters**: ~2.3M  
**Why it might win**: Interpretable (can visualize what each slot binds to). Natural fit for discrete ink regions.

---

## Training Configuration

**Common to all 6**:
- 15 epochs (proven baseline stack)
- MAE pretraining warmstart (where compatible)
- SupCon curriculum (λ 0.05→0.5 over 10 epochs)
- Attention MIL + entropy regularization (0.03)
- Ring negatives (closed label source)
- Context 48px, depth 24 slices, tile 16px
- Depth window 4-28 (full training range)
- LR 1.5e-4 (desktop EPYC), weight decay 0.3
- Data aug: flip 0.6, rotate 0.6, noise 0.3, brightness/contrast 0.6, cutout 0.4

**Platform-aware defaults** (auto-detected):
- Batch size: 96 (desktop) / 32 (laptop)
- Workers: 12 (desktop) / 0 (laptop)
- Eval BS: 256 (desktop) / 32 (laptop)

---

## Running the Campaign

```bash
# dry-run (verify setup)
python campaign_archs_6.py --dry-run

# single test (pilot)
python campaign_archs_6.py --only vit3d

# full campaign (6 tests × 15 epochs ≈ 12-18 hours on EPYC)
python campaign_archs_6.py
```

Logs: `./runs_archs6/`

---

## What We Know Works (carried forward)

- **MAE pretraining** on papyrus texture (unlabeled)
- **LCN preprocessing** (removes 113 keV bulk-density baseline)
- **zgrad** (dI/dz highlights ink-layer interfaces)
- **Context 48px** (competition limit, spatial context helps)
- **Depth 24 slices** (captures full ring signature)
- **SupCon curriculum** (0.05→0.5, stabilizes training)
- **Attention MIL** (LSE aggregation, sparse signal concentration)
- **Entropy regularization** (0.03, prevents overconfident collapse)
- **Ring negatives** (closed source, eliminates unlabeled ink contamination)

---

## Success Metrics

**Primary**: Valid F1, Valid AUC  
**Secondary**: Interpretability (SlotAttention), parameter efficiency (XCiT), training stability

**What we're looking for**:
1. ANY architecture breaks the plateau (F1 > current best)
2. Transformer vs CNN comparison (which paradigm fits this problem?)
3. Evidence for/against global attention (ViT vs Swin vs ConvNeXt)
4. Object-centric hypothesis test (does SlotAttention find discrete ink regions?)

---

## Files Created/Modified

**NEW**:
- `utils/platform.py` - multi-machine path detection
- `utils/radical_archs.py` - 6 new architectures (1021 lines)
- `campaign_archs_6.py` - campaign runner (369 lines)

**MODIFIED**:
- `utils/config.py` - platform-aware defaults
- `utils/model.py` - register radical archs in `_ARCH_MAP`
- `assemble_training_segments.py` - use `get_zarr_dir()`
- `assemble_test_segments.py` - use `get_zarr_dir()`

---

## Next Steps

1. ✓ Platform detection implemented
2. ✓ 6 architectures implemented (all output 1 logit, NO dense)
3. ✓ Campaign runner ready
4. ⏳ Training data download in progress
5. ⏳ Run campaign: `python campaign_archs_6.py`

---

## Architecture Comparison Table

| Arch | Paradigm | Params | Key Innovation | Complexity | Interpretability |
|------|----------|--------|----------------|------------|------------------|
| ViT3D | Pure Transformer | ~8.7M | Global attention layer 1 | O(N²) | Medium (attention maps) |
| Swin3D | Hierarchical Transformer | ~22M | Shifted windows + multi-scale | O(N) per window | Medium (local attention) |
| ConvNeXt3D | Modern CNN | ~28M | Large kernels + LayerNorm | O(N) | Low (CNN black box) |
| XCiT3D | Cross-Cov Transformer | ~11M | Feature interaction (not token sim) | O(d²) | Medium (covariance) |
| nnUNet3D | U-Net (encoder-decoder) | ~14M | Skip connections + multi-scale | O(N) | Low (CNN) |
| SlotAttention3D | Object-centric | ~2.3M | Iterative slot binding | O(K·N) | **HIGH** (slot visualization) |

**Philosophy**:
- NO PESSIMISM. This is a super hard problem → we need radical solutions.
- Test paradigms, not params. These aren't tuned for equal FLOPs.
- Verdict = does ANY break the plateau?

---

**2026-08-08**: Ready to break through the barrier. 🚀
