#!/usr/bin/env bash
# assemble_test_zarr.sh -- render the PHerc0813 test segment zarr from its tifxyz mesh.
#
# the tifxyz mesh is stored in:
#   tifxyz/auto_grown_20260716083545968/   <- root level = latest state (max_gen=179)
#   tifxyz/auto_grown_20260716083545968/N/ <- numbered subdirs = historical autosaves
#
# HOW IT WORKS
# The tifxyz does NOT contain intensity values -- it contains the SURFACE COORDINATES.
# For each (u,v) cell in the flattened 2D papyrus grid, x.tif/y.tif/z.tif store the
# 3D (x,y,z) voxel coordinate of that surface point in the raw CT scan. Intensity values
# live in the raw CT volume on S3. The render script:
#   1. reads x.tif/y.tif/z.tif -> knows WHERE on the raw CT each surface pixel came from
#   2. fetches those voxels (+ depth neighbors) from the S3 raw volume
#   3. writes them as a local zarr with shape (layers, H, W)
# This is why the tifxyz is tiny (~0.6 MB per file) even for a large segment.
#
# requires: python + project venv, curl, ~2 GB disk
# usage:  bash assemble_test_zarr.sh [--workers N]
#
# output:
#   ves_zarrs2/20260716083545.zarr   (28, 4421, 4421) uint16
#   masks/20260716083545.png

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/Scripts/python.exe"
if [[ "$OSTYPE" != "msys"* && "$OSTYPE" != "win"* ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
RENDERER="${SCRIPT_DIR}/old/render_9um_surface.py"
WORKERS=24
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done
cd "$SCRIPT_DIR"
mkdir -p C:/Users/ChenJeff/Documents/ves_zarrs2 masks _ves_tmp/pherc0813_test_chunks

# the tifxyz root level = the most grown/latest version of the segment
MESH_DIR="${SCRIPT_DIR}/tifxyz/auto_grown_20260716083545968"

# PHerc0813 raw CT volume on S3 (single available resolution: 9.362um / 113keV)
# shape (16993, 7947, 7947) -- z,y,x in voxel coordinates
PHERC0813_RAW="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/0"

echo "=== rendering PHerc0813 test segment 20260716083545 from tifxyz ==="
echo "  mesh: $MESH_DIR  (max_gen=179, area~2.82cm2)"
echo "  raw vol: $PHERC0813_RAW"
echo "  this downloads chunks on demand from S3 -- may take 10-30 min depending on speed"

"$PYTHON" "$RENDERER" \
    --mesh-dir "$MESH_DIR" \
    --cache-dir "_ves_tmp/pherc0813_test_chunks" \
    --vol-base "$PHERC0813_RAW" \
    --vol-shape "16993,7947,7947" \
    --layers 28 \
    --workers "$WORKERS" \
    --out-zarr "C:/Users/ChenJeff/Documents/ves_zarrs2/20260716083545.zarr" \
    --out-id 20260716083545

echo "=== precomputing normalization stats ==="
"$PYTHON" precompute_norm.py --scroll-id 20260716083545

echo "=== done ==="
"$PYTHON" -c "
import zarr
z = zarr.open('C:/Users/ChenJeff/Documents/ves_zarrs2/20260716083545.zarr')
print(f'  20260716083545  shape={z.shape}  dtype={z.dtype}')
"
