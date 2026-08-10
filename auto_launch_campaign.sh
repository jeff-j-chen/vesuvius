#!/bin/bash
# auto_launch_campaign.sh -- monitor training data download and auto-launch campaign_archs_6
#
# usage:
#   bash auto_launch_campaign.sh          # monitor and launch full campaign
#   bash auto_launch_campaign.sh --pilot  # launch only vit3d test first

set -e

ZARR_DIR="/media/jeff/Seagate/ves_zarrs2"
EXPECTED_ZARRS=17  # 16 training fragments + 1 existing (20230709155141.zarr)
POLL_INTERVAL=30   # seconds between checks
PILOT_MODE=0

# parse args
if [[ "$1" == "--pilot" ]]; then
    PILOT_MODE=1
    echo "[auto-launch] PILOT MODE: will run --only vit3d after download completes"
fi

echo "================================================================================"
echo "[auto-launch] Monitoring training data download"
echo "================================================================================"
echo "  Expected zarrs: $EXPECTED_ZARRS"
echo "  Zarr directory: $ZARR_DIR"
echo "  Poll interval:  ${POLL_INTERVAL}s"
echo "  Campaign mode:  $([ $PILOT_MODE -eq 1 ] && echo "PILOT (vit3d only)" || echo "FULL (all 6 tests)")"
echo "================================================================================"

while true; do
    # count completed zarrs
    CURRENT=$(find "$ZARR_DIR" -maxdepth 1 -name "*.zarr" -type d 2>/dev/null | wc -l)
    
    # count active download processes
    ACTIVE_DL=$(ps aux | grep -E "assemble_training|aria2c" | grep -v grep | wc -l)
    
    NOW=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$CURRENT" -ge "$EXPECTED_ZARRS" ]; then
        echo ""
        echo "================================================================================"
        echo "[$NOW] ✓ Download complete! $CURRENT/$EXPECTED_ZARRS zarrs present"
        echo "================================================================================"
        
        # wait a bit for any stragglers to finish writing
        echo "[auto-launch] Waiting 10s for file writes to complete..."
        sleep 10
        
        # verify norm cache exists (if not, precompute will run during training startup)
        if [ -f "./norm_cache.json" ]; then
            echo "[auto-launch] ✓ norm_cache.json exists"
        else
            echo "[auto-launch] WARNING: norm_cache.json missing (will be computed on first epoch)"
        fi
        
        # launch campaign
        echo ""
        echo "================================================================================"
        echo "[$NOW] LAUNCHING CAMPAIGN_ARCHS_6"
        echo "================================================================================"
        
        if [ $PILOT_MODE -eq 1 ]; then
            echo "[auto-launch] Running pilot test: vit3d only"
            python campaign_archs_6.py --only vit3d
        else
            echo "[auto-launch] Running full campaign: 6 tests × 15 epochs"
            python campaign_archs_6.py
        fi
        
        EXIT_CODE=$?
        echo ""
        echo "================================================================================"
        echo "[$NOW] Campaign completed with exit code: $EXIT_CODE"
        echo "================================================================================"
        
        exit $EXIT_CODE
        
    elif [ "$ACTIVE_DL" -eq 0 ] && [ "$CURRENT" -lt "$EXPECTED_ZARRS" ]; then
        echo ""
        echo "================================================================================"
        echo "[$NOW] ERROR: Download appears to have stopped"
        echo "================================================================================"
        echo "  Current zarrs: $CURRENT/$EXPECTED_ZARRS"
        echo "  Active downloads: $ACTIVE_DL"
        echo ""
        echo "Possible issues:"
        echo "  - Download script terminated early"
        echo "  - Network failure"
        echo "  - Disk full"
        echo ""
        echo "Please check manually and restart assemble_training_segments.py if needed."
        exit 1
        
    else
        # still downloading
        PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$EXPECTED_ZARRS)*100}")
        echo "[$NOW] Progress: $CURRENT/$EXPECTED_ZARRS zarrs (${PROGRESS}%)  |  Active downloads: $ACTIVE_DL"
        sleep $POLL_INTERVAL
    fi
done
