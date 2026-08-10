# Auto-Launch Campaign Setup

## Context Size Verification ✓

**All 6 architectures are WITHIN the 48×48 context limit:**

| Architecture | Input Size | After Downsample | Patch/Kernel Size | Tokens/Features |
|--------------|------------|------------------|-------------------|-----------------|
| ViT3D | 48×48×24 | 24×24×24 | patch_size=4 | 6×6×6 = 216 tokens |
| Swin3D | 48×48×24 | 24×24×24 | patch_size=2 | 12×12×12 patches |
| ConvNeXt3D | 48×48×24 | 24×24×24 | stem kernel=4 | 6×6×6 after stem |
| XCiT3D | 48×48×24 | 24×24×24 | patch_size=4 | 6×6×6 = 216 tokens |
| nnUNet3D | 48×48×24 | 24×24×24 | standard CNN | adaptive pooling |
| SlotAttention3D | 48×48×24 | 24×24×24 | MaxPool | 6×6×6 after 2 pools |

**All architectures receive 24×24×24 input** (48px context with downsample=2), well within the 48×48 limit.

---

## Auto-Launch Script

**Location**: `auto_launch_campaign.sh`

### What It Does

1. **Monitors** training data download progress (polls every 30s)
2. **Detects** when all 17 zarrs are present
3. **Waits** 10s for file writes to complete
4. **Auto-launches** `campaign_archs_6.py`

### Usage

```bash
# Full campaign (6 tests × 15 epochs, auto-starts when download completes)
bash auto_launch_campaign.sh

# Pilot mode (only vit3d test, for quick validation)
bash auto_launch_campaign.sh --pilot
```

### Run in Background (Recommended)

```bash
# Full campaign with logging
nohup bash auto_launch_campaign.sh > campaign_archs6_auto.log 2>&1 &

# Check progress
tail -f campaign_archs6_auto.log

# Or pilot mode
nohup bash auto_launch_campaign.sh --pilot > campaign_vit3d_pilot.log 2>&1 &
```

### Manual Launch (Alternative)

If you prefer to launch manually after download completes:

```bash
# Wait for download to complete (check with)
find /media/jeff/Seagate/ves_zarrs2/ -maxdepth 1 -name "*.zarr" -type d | wc -l
# Should show 17 when complete

# Then launch campaign
python campaign_archs_6.py              # full campaign
python campaign_archs_6.py --only vit3d  # pilot test
```

---

## Current Status

**Download Progress**: 1/17 zarrs complete (as of 09:08)

- Downloading to: `/media/jeff/Seagate/ves_zarrs2/`
- Parallel downloads: 5 fragments at a time
- Estimated time: ~30-60 minutes (depends on network speed)

**Active Downloads**:
```bash
# Check progress manually
ps aux | grep aria2c | grep -v grep | wc -l  # active download processes
find /media/jeff/Seagate/ves_zarrs2/ -maxdepth 1 -name "*.zarr" -type d | wc -l  # completed zarrs
```

---

## Auto-Launch Features

✓ **Progress monitoring** - polls every 30s, shows completion percentage  
✓ **Error detection** - alerts if download stops prematurely  
✓ **Graceful waiting** - 10s buffer after download before campaign start  
✓ **Exit code handling** - preserves campaign exit code for CI/CD  
✓ **Pilot mode** - test single architecture first (--pilot flag)  

**Example Output**:
```
[2026-08-08 09:15:42] Progress: 5/17 zarrs (29.4%)  |  Active downloads: 5
[2026-08-08 09:16:12] Progress: 8/17 zarrs (47.1%)  |  Active downloads: 5
[2026-08-08 09:16:42] Progress: 12/17 zarrs (70.6%)  |  Active downloads: 3
...
[2026-08-08 09:18:30] ✓ Download complete! 17/17 zarrs present
[auto-launch] Waiting 10s for file writes to complete...
[auto-launch] LAUNCHING CAMPAIGN_ARCHS_6
```

---

## Recommended Workflow

```bash
# 1. Start auto-launcher in background with logging
nohup bash auto_launch_campaign.sh > campaign_archs6_auto.log 2>&1 &

# 2. Monitor progress (Ctrl+C to stop tailing, doesn't kill the job)
tail -f campaign_archs6_auto.log

# 3. Detach and come back later
# The campaign will run for ~12-18 hours on EPYC hardware
# Results will be in ./runs_archs6/

# 4. Check results when complete
ls -ltr ./runs_archs6/
```

---

## Files

- `auto_launch_campaign.sh` - auto-launch script (executable)
- `campaign_archs_6.py` - campaign runner
- `utils/radical_archs.py` - 6 architecture implementations
- `utils/platform.py` - multi-machine path detection
- `CAMPAIGN_ARCHS_6_PLAN.md` - full architecture details

**Ready to launch!** 🚀
