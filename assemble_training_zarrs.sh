#!/usr/bin/env bash
# assemble_training_zarrs.sh -- download and assemble the 3 PHerc0139 9.362um surface
# volumes used for training (w044, w059, w047). each is a pre-rendered OME-Zarr on S3;
# this script fetches the chunks and writes them as local training zarrs + masks.
#
# requires: python + the project venv, curl, ~15 GB disk (3 x ~2-5 GB each)
# usage:  bash assemble_training_zarrs.sh [--workers N]
#
# output (written relative to the repo root):
#   ves_zarrs2/20260115000000.zarr   w044 (28, 6021, 8141)
#   ves_zarrs2/20250223000000.zarr   w059 (28, 7220, 10020)
#   ves_zarrs2/20260206000001.zarr   w047 (28, 5821, 8421)
#   masks/20260115000000.png
#   masks/20250223000000.png
#   masks/20260206000001.png        <- already restricted to 1.1um overlap band
#
# SEGMENT PROVENANCE
# All three fragments come from PHerc0139 (Herculaneum scroll, 9.362um / 113keV / 1.2m
# detector distance). the raw volume ID is 20250728140407.
#   w044  segment 20260115000000   ~52 cm^2  (the primary labelled training fragment)
#   w059  segment 20250223000000   (narrow band -- the 1.1um ROI2 overlap area)
#   w047  segment 20260206000001   ~43 cm^2  (the larger wing patch, also ROI2-masked)
#
# TEST SEGMENT
# The test/inference fragment is from PHerc0813 (9.362um / 113keV / 1.2m):
#   raw volume: 20250821151723   rendered -> ves_zarrs2/20260716083545.zarr (28, 4421, 4421)
#   tifxyz: auto_grown_20260716083545968  max_gen=179  area=2.84 cm^2
#   the tifxyz encodes for each (u,v) in the flattened sheet the raw-volume (x,y,z)
#   voxel coordinate. bbox z 9208-11370 y 3261-4570 x 4176-5730 (9.362um voxel coords).
#   id=20260716083545 = timestamp of the segment creation, traceable to the tifxyz.
#
# NOTE: masks for w059 and w047 are INTERSECTED with the 1.1um ink-detection footprint
# (only the portion covered by the high-res scan carries reliable ink labels). the full
# 9.4um footprint is preserved as masks/<id>_full9um.png if re-rendering is needed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/Scripts/python.exe"
if [[ "$OSTYPE" != "msys"* && "$OSTYPE" != "win"* ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi
DOWNLOADER="${SCRIPT_DIR}/old/download_surface_zarr.py"
RENDERER="${SCRIPT_DIR}/old/render_9um_surface.py"
WORKERS=24
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done
cd "$SCRIPT_DIR"
mkdir -p C:/Users/ChenJeff/Documents/ves_zarrs2 masks

PHERC0139_RAW="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0"
SEG_BASE="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments"

echo "=== 1/3  w044  20260115000000  (primary labelled fragment) ==="
# w044 has a pre-rendered surface volume on S3 -- use the downloader (fast, no mesh needed)
"$PYTHON" "$DOWNLOADER" \
    --mode volume --level 0 \
    --url "${SEG_BASE}/20250115000000-w044_2025011500/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr" \
    --out-id 20260115000000 --workers "$WORKERS"
# fallback: render from tifxyz if the surface volume URL differs on your VC3D version
# "$PYTHON" "$RENDERER" \
#   --mesh-dir ~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260115000000 \
#   --cache-dir /tmp/w044_chunks \
#   --vol-base "$PHERC0139_RAW" --vol-shape "20974,6621,6621" \
#   --layers 28 --workers "$WORKERS" \
#   --out-zarr C:/Users/ChenJeff/Documents/ves_zarrs2/20260115000000.zarr --out-id 20260115000000

echo "=== 2/3  w059  20250223000000  (1.1um overlap band) ==="
"$PYTHON" "$DOWNLOADER" \
    --mode volume --level 0 \
    --url "${SEG_BASE}/20250223000000-w059_2025022312/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr" \
    --out-id 20250223000000 --workers "$WORKERS"

echo "=== 3/3  w047  20260206000001  (larger wing patch) ==="
# w047 is rendered from its tifxyz mesh (mesh file: 20260206000001-on-20250728140407-9.362um.tifxyz)
"$PYTHON" "$RENDERER" \
    --mesh-dir _ves_tmp/w047_9um_mesh \
    --cache-dir _ves_tmp/p0139_w047_9um_chunks \
    --vol-base "$PHERC0139_RAW" \
    --vol-shape "20974,6621,6621" \
    --layers 28 --workers "$WORKERS" \
    --out-zarr "C:/Users/ChenJeff/Documents/ves_zarrs2/20260206000001.zarr" \
    --out-id 20260206000001
# NOTE: after render, restrict the w047 mask to the 1.1um overlap band:
#   python - <<'EOF'
#   import numpy as np, cv2; from PIL import Image; Image.MAX_IMAGE_PIXELS=None
#   m9=np.array(Image.open("masks/20260206000001.png").convert("L"))
#   ink=np.array(Image.open("_ves_tmp/w047_ink_1p1um_CORRECT_fullres.png").convert("L"))
#   H,W=m9.shape; ink9=cv2.resize(ink,(W,H),cv2.INTER_AREA)
#   foot=cv2.morphologyEx((ink9>0).astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
#   Image.fromarray(((m9>0)&(foot>0)).astype(np.uint8)*255).save("masks/20260206000001.png")
#   EOF

echo "=== precomputing normalization stats for all 3 scrolls ==="
"$PYTHON" precompute_norm.py \
    --scroll-id 20260115000000 \
    --scroll-id 20250223000000 \
    --scroll-id 20260206000001

echo "=== done ==="
echo "zarrs:"
for id in 20260115000000 20250223000000 20260206000001; do
    python -c "import zarr; z=zarr.open('C:/Users/ChenJeff/Documents/ves_zarrs2/$id.zarr'); print(f'  $id  {z.shape}')"
done
