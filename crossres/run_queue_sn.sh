#!/usr/bin/env bash
# plan U retrained with surface-anchored normalisation (campaign 39 fine-tunes with it), after the degrader
set -euo pipefail
cd "$(dirname "$0")/.."
INIT=models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth
UPSAMPLER=models/upsampler_learned_xy2_d4_surfacenorm.pth
LOG=logs_crossres
DONE=$LOG/done
mkdir -p "$DONE"
while pgrep -f "crossres/run_queue_r[.]sh" > /dev/null; do sleep 60; done

run() {  # name, then the command
    local name=$1; shift
    if [[ -f "$DONE/$name" ]]; then echo "[queue_sn] $(date +%T) skip $name (done)"; return; fi
    echo "[queue_sn] $(date +%T) start $name"
    "$@" > "$LOG/train_$name.log" 2>&1
    touch "$DONE/$name"
    echo "[queue_sn] $(date +%T) done $name"
}

run upsampler_learned_xy2_d4_surfacenorm python crossres/train_upsampler.py \
    --name upsampler_learned_xy2_d4_surfacenorm --norm-mode surface_anchor
for kind in learned trilinear; do
    spec=$([[ $kind == learned ]] && echo "$UPSAMPLER" || echo trilinear)
    run "mae_upsampled_${kind}_surfacenorm_native96" python crossres/mae_pretrain_crossres.py \
        --name "mae_upsampled_${kind}_surfacenorm_native96" --init-weights "$INIT" --exclude-holdout-pairs \
        --plan none --upsampler "$spec" --accum-steps 2 --data-norm-mode surface_anchor
done
echo "[queue_sn] $(date +%T) all done"
