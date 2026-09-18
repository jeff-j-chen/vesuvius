#!/usr/bin/env bash
# verify_new_segment.sh - verify the new PHerc0211 merged segment is properly set up

set -euo pipefail

echo "======================================================================"
echo "Verification Script for New PHerc0211 Merged Segment"
echo "======================================================================"
echo ""

# Check if tifxyz mesh exists
MESH_DIR="/vesuvius/tifxyz/auto_grown_20260717193517520_0_1_2_3_4_merged"
echo "[1/5] Checking tifxyz mesh directory..."
if [[ -d "$MESH_DIR" ]]; then
    echo "  ✓ Mesh directory exists: $MESH_DIR"
    if [[ -f "$MESH_DIR/x.tif" && -f "$MESH_DIR/y.tif" && -f "$MESH_DIR/z.tif" ]]; then
        echo "  ✓ Required tif files present (x.tif, y.tif, z.tif)"
    else
        echo "  ✗ ERROR: Missing required tif files!"
        exit 1
    fi
else
    echo "  ✗ ERROR: Mesh directory not found: $MESH_DIR"
    echo "  Please extract the zip file first:"
    echo "    cd /vesuvius/tifxyz"
    echo "    unzip auto_grown_20260717193517520_0_1_2_3_4_merged.zip"
    exit 1
fi
echo ""

# Check if zarr exists (might not if not assembled yet)
ZARR_PATH="/vesuvius/ves_zarrs2/20260717193517.zarr"
echo "[2/5] Checking zarr assembly..."
if [[ -d "$ZARR_PATH" ]]; then
    echo "  ✓ Zarr exists: $ZARR_PATH"
    # Check shape using Python
    python3 -c "import zarr; z=zarr.open('$ZARR_PATH', 'r'); print(f'  ✓ Zarr shape: {z.shape}, dtype: {z.dtype}')"
else
    echo "  ⚠ Zarr not yet assembled: $ZARR_PATH"
    echo "  Run: python assemble_test_segments.py --workers 32"
fi
echo ""

# Check mask
MASK_PATH="/vesuvius/masks/20260717193517.png"
echo "[3/5] Checking mask..."
if [[ -f "$MASK_PATH" ]]; then
    echo "  ✓ Mask exists: $MASK_PATH"
else
    echo "  ⚠ Mask not found (created during zarr assembly)"
fi
echo ""

# Check config
echo "[4/5] Checking config.py test_scroll_ids..."
TEST_SCROLLS=$(python3 -c "from utils.config import Config; c=Config(); print(','.join(map(str, c.data.test_scroll_ids)))")
if [[ "$TEST_SCROLLS" == "20260716083545,20260717193517,20260720090842,20250703034159" ]]; then
    echo "  ✓ Config correct: test_scroll_ids = [$TEST_SCROLLS]"
else
    echo "  ✗ ERROR: Unexpected test_scroll_ids: [$TEST_SCROLLS]"
    exit 1
fi
echo ""

# Check old segments cleanup
echo "[5/5] Checking for obsolete segments..."
OLD1="/vesuvius/tifxyz/auto_grown_20260719202304218"
if [[ -d "$OLD1" ]]; then
    echo "  ⚠ Old segment still present: $OLD1"
    echo "    (Optional) Remove with: rm -rf $OLD1"
else
    echo "  ✓ Old segment auto_grown_20260719202304218 not present"
fi
echo ""

# Summary
echo "======================================================================"
echo "Verification Summary"
echo "======================================================================"
if [[ -d "$MESH_DIR" ]] && [[ -f "$MESH_DIR/x.tif" ]]; then
    if [[ -d "$ZARR_PATH" ]] && [[ -f "$MASK_PATH" ]]; then
        echo "✓ All checks passed! New segment is fully set up."
    else
        echo "⚠ Mesh extracted but zarr not assembled yet."
        echo "  Next step: python assemble_test_segments.py --workers 32"
    fi
else
    echo "✗ Setup incomplete. See error messages above."
    exit 1
fi
