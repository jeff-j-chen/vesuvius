#!/usr/bin/env bash
# plan R (crossres/PLAN.md section 0.7): after the other queues, train the degrader
# batch 4: the default 16 x 28 x 128^2 needs more than the 24 GB A5000
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs_crossres
mkdir -p "$LOG/done"
while pgrep -f "crossres/run_queue.sh|crossres/run_queue_v2.sh|mae_pretrain_crossres.py" > /dev/null; do sleep 60; done
while pgrep -f "build_pairs.py --zarr-dir ves_zarrs2 --plan degrade" > /dev/null; do sleep 60; done
echo "[queue_r] $(date +%T) degrade tiles: $(ls crossres/pairs/degrade/*/meta.json | wc -l) segments"
if [[ ! -f "$LOG/done/degrader" ]]; then
    echo "[queue_r] $(date +%T) start degrader"
    python crossres/train_degrader.py --dry-run --batch-size 4 > "$LOG/train_degrader_dryrun.log" 2>&1
    python crossres/train_degrader.py --batch-size 4 > "$LOG/train_degrader.log" 2>&1
    touch "$LOG/done/degrader"
    echo "[queue_r] $(date +%T) done degrader"
fi
