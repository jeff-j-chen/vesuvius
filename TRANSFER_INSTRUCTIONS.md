## File Transfer and Assembly Instructions for PHerc0211 Merged Segment

This document describes how to transfer and assemble the new merged PHerc0211 test segment that replaces the two previous patches (`auto_grown_20260717193517520` and `auto_grown_20260719202304218`).

### Step 1: Transfer the zip file from Windows to Linux

From your Windows PowerShell or Command Prompt, run:

```powershell
scp -P 20700 -i C:\Users\ChenJeff\.ssh\id_ed25519 C:\Users\ChenJeff\Downloads\auto_grown_20260717193517520_0_1_2_3_4_merged.zip root@157.157.221.29:/vesuvius/tifxyz/
```

### Step 2: Extract the zip file on the Linux server

After transfer completes, on the Linux server run:

```bash
cd /vesuvius/tifxyz
unzip auto_grown_20260717193517520_0_1_2_3_4_merged.zip
rm auto_grown_20260717193517520_0_1_2_3_4_merged.zip

# verify extraction
ls -la auto_grown_20260717193517520_0_1_2_3_4_merged/
# should see x.tif, y.tif, z.tif files
```

### Step 3: Assemble the test segment zarr

The segment will be assembled from the tifxyz mesh using the updated `assemble_test_segments.py` script:

```bash
cd /vesuvius

# assemble just the new PHerc0211 merged segment (recommended to verify first)
# note: this will download chunks from S3 and can take 15-30 minutes
python assemble_test_segments.py --workers 32

# the script now only processes:
#   - 20260716083545 (PHerc0813)
#   - 20260717193517 (PHerc0211 merged - NEW)
#   - 20260720090842 (PHerc1203)
#   - 20250703034159 (PHerc1447)
```

### Step 4: Verify assembly

After assembly completes:

```bash
# check zarr exists
ls -lh ves_zarrs2/20260717193517.zarr/

# check mask exists
ls -lh masks/20260717193517.png

# zarr metadata
python -c "import zarr; z=zarr.open('ves_zarrs2/20260717193517.zarr', 'r'); print(f'shape: {z.shape}, dtype: {z.dtype}')"
```

### Step 5: (Optional) Clean up old segments

Once the new segment is verified working, you can optionally remove the old PHerc0211 segments:

```bash
# ONLY do this after verifying the new merged segment works!
cd /vesuvius/tifxyz
rm -rf auto_grown_20260717193517520
rm -rf auto_grown_20260719202304218

# note: the zarrs for these old segments are NOT removed automatically
# if you want to clean them up:
# rm -rf ves_zarrs2/20260719202304.zarr
# rm masks/20260719202304.png
```

### Notes

- The assembly script has been updated to use the new merged segment automatically
- `utils/config.py` has been updated to only reference the merged segment in `test_scroll_ids`
- The README has been updated to document the change
- A training run is currently in progress - the assembly can be done safely as it only reads from S3 and writes to new zarr files
