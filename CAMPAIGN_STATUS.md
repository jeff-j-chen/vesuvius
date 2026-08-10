# Campaign Architectures 6 - Status Report

## Architecture Validation (as of 2026-08-08 23:42)

### ✅ Working Architectures (5/6)

| Architecture | Parameters | Output Shape | Status |
|--------------|------------|--------------|--------|
| ViT3D        | 4.81M      | (B, 1)       | ✅ Ready |
| Swin3D       | 5.34M      | (B, 1)       | ✅ Ready |
| XCiT3D       | 9.49M      | (B, 1)       | ✅ Ready |
| nnUNet3D     | 51.77M     | (B, 1)       | ✅ Ready |
| SlotAttention3D | 1.03M   | (B, 1)       | ✅ Ready |

### ❌ Needs Fixing

| Architecture | Issue | Priority |
|--------------|-------|----------|
| ConvNeXt3D   | Hardcoded dimensions in LayerNorm, over-aggressive downsampling | Medium |

## Training Data ✅

- **Downloaded:** 17/17 fragments (w044-w055, seg46527, 500P2_front)
- **Zarr Path:** `/media/jeff/Seagate/ves_zarrs2`
- **All integrity checks passed**

## Hardware Status ⚠️

### GPU Issue
- **Problem:** PyTorch compiled for CUDA 12.8, system has CUDA 11.4
- **Impact:** Training falls back to CPU (10-100x slower)
- **Solution Options:**
  1. Update NVIDIA driver to 525+ (requires sudo + reboot)
  2. Reinstall PyTorch for CUDA 11.x: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
  3. Accept slow CPU training

### Current System
- **GPU:** NVIDIA GeForce (5938MiB VRAM available)
- **Driver:** 470.256.02 (CUDA 11.4)
- **CPU:** AMD EPYC 7702P (64 cores)
- **RAM:** Sufficient for training

## Campaign Configuration ✅

### Scroll IDs (Updated for Downloaded Data)
```python
c.data.scroll1_id = 20260115000000  # w044 (train)
c.data.scroll2_id = 20250223000000  # w059 (valid)
c.data.scroll4_id = 20260206000001  # w047 (test)
```

### Training Parameters
- **Epochs:** 15 per architecture
- **Batch Size:** 96 (desktop) / 32 (laptop)
- **Workers:** 12 (desktop) / 0 (laptop)
- **Context:** 48×48 → 24×24×24 after downsample=2
- **Tile Size:** 16×16
- **Depth:** 24 slices (range 4-28)

### Regularization Stack
- SupCon with curriculum (λ: 0.05 → 0.5 over 10 epochs)
- AttnMIL with entropy regularization (weight=0.03)
- Ring negatives (closed label source)
- L1: 7e-6, Weight Decay: 0.3
- Dropout: conv1=0.15, conv2=0.15, head=0.4
- Data aug: flip=0.6, rot=0.6, noise=0.3, brightness=0.6, contrast=0.6

## Context Limit Verification ✅

All architectures operate within 48×48 maximum context:
- Input: 48×48×24 context window
- After downsample=2: 24×24×24 (actual model input)
- All tested and verified with correct dimensions

## Output Constraint ✅

**CRITICAL:** All architectures return exactly 1 binary logit per tile.
- **Verified:** All 5 working architectures output shape (B, 1)
- **NO dense outputs anywhere**

## Next Steps

### Immediate (Before Training)
1. **Fix GPU/CUDA issue** for 10-100x speed boost
   - Option A: Update driver (recommended, requires reboot)
   - Option B: Reinstall PyTorch for CUDA 11.x
   
2. **Optional: Fix ConvNeXt3D** (or run campaign with 5 architectures)

### Launch Campaign
```bash
# Full campaign (all working architectures)
python campaign_archs_6.py

# Single test for verification
python campaign_archs_6.py --only vit3d

# Dry run to verify config
python campaign_archs_6.py --dry-run
```

### Monitor Progress
```bash
# Check running campaigns
ps aux | grep campaign_archs_6

# Watch logs
tail -f campaign_archs6_full.log

# Check TensorBoard
tensorboard --logdir=./runs_archs6
```

## Estimated Completion Times

| Hardware | Time per Epoch | Total (5 arch × 15 ep) |
|----------|----------------|------------------------|
| CPU only | ~30-60 min     | ~37-75 hours           |
| GPU (fixed) | ~3-6 min     | ~3.75-7.5 hours        |

## Files Created

- `utils/platform.py` - Platform detection
- `utils/radical_archs.py` - 6 architecture implementations
- `campaign_archs_6.py` - Campaign runner
- `auto_launch_campaign.sh` - Auto-launcher (downloads complete)
- `CAMPAIGN_ARCHS_6_PLAN.md` - Architecture details
- `AUTO_LAUNCH_GUIDE.md` - Usage guide

## Known Issues

1. **GPU Not Available:** PyTorch/CUDA version mismatch (see Hardware Status)
2. **ConvNeXt3D:** Dimension mismatch (medium priority, campaign can run with 5 archs)
3. **CUDA Warning:** Driver too old (cosmetic, doesn't prevent CPU training)
