#!/usr/bin/env bash
# D and X redone with the deep SR head, after run_queue.sh and run_queue_r.sh have finished
set -euo pipefail
cd "$(dirname "$0")/.."
INIT=models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth
LOG=logs_crossres
DONE=$LOG/done
mkdir -p "$DONE"
while pgrep -f "crossres/run_queue.sh|crossres/run_queue_r.sh" > /dev/null; do sleep 60; done

for plan in depth xyz; do
    name=mae_crossres_${plan}_v2
    if [[ -f "$DONE/$name" ]]; then echo "[queue_v2] $(date +%T) skip $name (done)"; continue; fi
    echo "[queue_v2] $(date +%T) start $name"
    python crossres/mae_pretrain_crossres.py --plan "$plan" --name "$name" --init-weights "$INIT" \
        --exclude-holdout-pairs --sr-head deep --head-lr-mult 5 > "$LOG/train_$name.log" 2>&1
    touch "$DONE/$name"
    echo "[queue_v2] $(date +%T) done $name"
done
