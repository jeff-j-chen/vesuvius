#!/usr/bin/env bash
# sequential GPU queue for the cross-resolution pretrains (crossres/PLAN.md sections 0.2 and 0.6)
set -euo pipefail
cd "$(dirname "$0")/.."
INIT=models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth
LOG=logs_crossres

wait_for_plan() {
    while pgrep -f "build_pairs.py --zarr-dir ves_zarrs2 --plan $1" > /dev/null; do sleep 60; done
    echo "[queue] $(date +%T) plan $1 tiles ready: $(ls crossres/pairs/$1/*/meta.json | wc -l) segments"
}

pretrain() {  # name, then extra args
    local name=$1; shift
    echo "[queue] $(date +%T) start $name"
    python crossres/mae_pretrain_crossres.py --name "$name" --init-weights "$INIT" --exclude-holdout-pairs "$@" \
        > "$LOG/train_$name.log" 2>&1
    echo "[queue] $(date +%T) done $name"
}

wait_for_plan depth
pretrain mae_crossres_depth --plan depth
wait_for_plan xyz
pretrain mae_crossres_xyz --plan xyz
echo "[queue] $(date +%T) start upsampler_learned_xy2_d4"
python crossres/train_upsampler.py > "$LOG/train_upsampler.log" 2>&1
echo "[queue] $(date +%T) done upsampler_learned_xy2_d4"
# micro-batch 16 peaks at ~9 GB on the 24 GB A5000; effective batch stays 32
pretrain mae_upsampled_learned_native96 --plan none --upsampler models/upsampler_learned_xy2_d4.pth --accum-steps 2
pretrain mae_upsampled_trilinear_native96 --plan none --upsampler trilinear --accum-steps 2
echo "[queue] $(date +%T) all done"
