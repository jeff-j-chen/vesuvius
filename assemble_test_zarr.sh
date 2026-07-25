#!/usr/bin/env bash
# assemble_test_zarr.sh -- render the 5 competition test-segment zarrs from their tifxyz meshes
# (PHerc0813, PHerc0211 x2, PHerc1203, PHerc1447). the w055 HOLDOUT is a PHerc0139 segment and is
# assembled by assemble_training_segments.py (download path), not here.
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
# python: default to 'python' on PATH (the pod's docker provides it); override via $PYTHON.
# output dir: linux -> $VESUVIUS_ZARR_PATH or /vesuvius/ves_zarrs2 (matches precompute_norm/config),
# windows -> the local documents path; override either with --out-dir DIR.
PYTHON="${PYTHON:-python}"
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win"* ]]; then
    OUT_DIR="C:/Users/ChenJeff/Documents/ves_zarrs2"
else
    OUT_DIR="${VESUVIUS_ZARR_PATH:-/vesuvius/ves_zarrs2}"
fi
RENDERER="${SCRIPT_DIR}/old/render_9um_surface.py"
# default workers = cpu count; override with --workers N
if command -v nproc >/dev/null 2>&1; then WORKERS="$(nproc)"; else WORKERS="${NUMBER_OF_PROCESSORS:-24}"; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done
cd "$SCRIPT_DIR"
mkdir -p "$OUT_DIR" masks _ves_tmp
echo "[assemble] python=$PYTHON  out_dir=$OUT_DIR  workers=$WORKERS"

BUCKET="https://vesuvius-challenge-open-data.s3.amazonaws.com"
# each entry: out_id | tifxyz mesh subdir | raw-volume base url (.zarr/0) | vol shape z,y,x.
# the mesh voxel coords live in the listed volume's space, so vol-base + vol-shape MUST match the
# mesh (verified against each mesh bbox). NOTE: 1447's only volume is 8.640um (not 9.362um) -- that
# IS the volume its mesh was built on (vc3d folder 20250521151220_editable).
FRAGMENTS=(
    "20260716083545|auto_grown_20260716083545968|$BUCKET/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/0|16993,7947,7947"
    "20260717193517|auto_grown_20260717193517520|$BUCKET/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/0|19416,7948,7948"
    "20260719202304|auto_grown_20260719202304218|$BUCKET/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/0|19416,7948,7948"
    "20260720090842|auto_grown_20260720090842117|$BUCKET/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr/0|18977,6844,6844"
    "20250703034159|20250703034159|$BUCKET/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr/0|24297,8343,8343"
)

n=${#FRAGMENTS[@]}
i=0
for entry in "${FRAGMENTS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r ZID MESHSUB VOLBASE VOLSHAPE <<< "$entry"
    MESH_DIR="${SCRIPT_DIR}/tifxyz/${MESHSUB}"
    OUTZ="$OUT_DIR/${ZID}.zarr"
    echo "=== ${i}/${n}  ${ZID}  (mesh ${MESHSUB}) ==="
    # idempotent: skip if this zarr + mask already exist
    if [[ -d "$OUTZ" && -f "masks/${ZID}.png" ]]; then
        echo "  zarr + mask exist -> skip"
        continue
    fi
    if [[ ! -d "$MESH_DIR" ]]; then
        echo "  [WARN] mesh dir missing: $MESH_DIR -- skipping"
        continue
    fi
    echo "  raw vol: $VOLBASE  shape=$VOLSHAPE"
    echo "  (renders on-demand from S3 -- can take 10-30 min per fragment depending on size/speed)"
    if ! "$PYTHON" "$RENDERER" \
        --mesh-dir "$MESH_DIR" \
        --cache-dir "_ves_tmp/render_${ZID}" \
        --vol-base "$VOLBASE" \
        --vol-shape "$VOLSHAPE" \
        --layers 28 \
        --workers "$WORKERS" \
        --out-zarr "$OUTZ" \
        --out-id "$ZID"; then
        echo "  [WARN] render failed for ${ZID} -- continuing to next fragment"
        continue
    fi
done

echo "=== precomputing normalization stats for all test fragments ==="
"$PYTHON" precompute_norm.py \
    --scroll-id 20260716083545 \
    --scroll-id 20260717193517 \
    --scroll-id 20260719202304 \
    --scroll-id 20260720090842 \
    --scroll-id 20250703034159 \
    --zarr-path "$OUT_DIR" || echo "[WARN] norm precompute reported errors (a fragment may be missing)"

echo "=== done ==="
for entry in "${FRAGMENTS[@]}"; do
    IFS='|' read -r ZID _REST <<< "$entry"
    "$PYTHON" -c "import zarr; z=zarr.open('$OUT_DIR/${ZID}.zarr'); print(f'  ${ZID}  shape={z.shape}  dtype={z.dtype}')" 2>/dev/null || echo "  ${ZID}  (not assembled)"
done
