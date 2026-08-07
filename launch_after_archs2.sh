#!/bin/bash
# launch_after_archs2.sh: poll for campaign_archs_2 completion then start campaign_archs_3.
# run this in a separate tmux pane while archs_2 is running:
#   bash launch_after_archs2.sh
# it will sleep until archs_2 exits, then immediately start archs_3.

cd "$(dirname "$0")"

echo "[chain] watching for campaign_archs_2 to finish..."
while pgrep -f "campaign_archs_2.py" > /dev/null; do
    sleep 15
done

echo "[chain] campaign_archs_2 finished. starting campaign_archs_3..."
python campaign_archs_3.py
