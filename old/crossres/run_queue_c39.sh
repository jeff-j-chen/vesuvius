#!/bin/bash
# campaign 39 overnight pretraining queue: fiber branch + surface-anchored normalisation everywhere, on all
# 37 campaign-33 pretraining volumes (24 campaign-39 training/holdout segments + the 13 official tests).
# Resumable: a finished step leaves logs_c39/done/<name>; rerunning skips it.
set -uo pipefail
cd /vesuvius
LOG=logs_c39
mkdir -p "$LOG/done"
IDS=$(python -c "import campaign_archs_33 as c; print(' '.join(str(int(s)) for s in c.PRETRAIN_SCROLL_IDS))" 2>/dev/null)
BASE=models/c39_mae_base.pth
FIBER_SN=(--fiber-coordinate-branch --data-norm-mode surface_anchor)

run() {  # run <name> <command...>: skip if done, stop the queue on failure
    local name=$1; shift
    [[ -e "$LOG/done/$name" ]] && { echo "[queue] skip $name (done)"; return 0; }
    echo "[queue] $(date +%H:%M) start $name"
    if "$@" > "$LOG/train_$name.log" 2>&1; then
        touch "$LOG/done/$name"; echo "[queue] $(date +%H:%M) done $name"
    else
        echo "[queue] $(date +%H:%M) FAILED $name (see $LOG/train_$name.log)"; exit 1
    fi
}

# 1. every pretraining volume assembled (downloads started separately; [.] keeps pgrep off this script)
while pgrep -f "assemble_(training|test)_segment[s].py" > /dev/null || pgrep -f "retry_test[s].sh" > /dev/null; do
    sleep 60
done
# anything killed (the first 13-way parallel start ran out of memory) is redone one at a time
for z in $(python -c "import assemble_test_segments as a; print(' '.join(str(f[0]) for f in a.FRAGMENTS))" 2>/dev/null); do
    grep -q "$z: OK" "$LOG/assemble/test_$z.log" 2>/dev/null && continue
    echo "[queue] re-assembling test $z"
    python assemble_test_segments.py --workers 12 --only "$z" > "$LOG/assemble/test_$z.log" 2>&1
    grep -q "$z: OK" "$LOG/assemble/test_$z.log" || { echo "[queue] test $z failed"; exit 1; }
done
for n in w068 w087 w013 w018 cr1fr3 paris4 paris2_fr143 paris2_fr47 scroll6_fr8 paris1_fr34 p343; do
    grep -q "^\[done\] $n " "$LOG/assemble/train_$n.log" 2>/dev/null && continue
    echo "[queue] re-assembling training $n"
    python assemble_training_segments.py --no-r2 --skip-norm --concurrent-fragments 1 --workers 16 --only "$n" \
        > "$LOG/assemble/train_$n.log" 2>&1
    grep -q "^\[done\] $n " "$LOG/assemble/train_$n.log" || { echo "[queue] training $n failed"; exit 1; }
done
missing=$(for s in $IDS; do [[ -d "ves_zarrs2/$s.zarr/0" || -f "ves_zarrs2/$s.zarr/.zarray" || -f "ves_zarrs2/$s.zarr/zarr.json" ]] || echo -n "$s "; done)
[[ -n "$missing" ]] && { echo "[queue] missing zarrs: $missing"; exit 1; }
python - <<'EOF' || exit 1
import campaign_archs_33 as c33
from utils.norm import ensure_surface_anchors
ensure_surface_anchors([int(s) for s in c33.PRETRAIN_SCROLL_IDS], "ves_zarrs2")
print("[queue] surface anchors ready for all", len(c33.PRETRAIN_SCROLL_IDS))
EOF

# 2. base MAE from scratch: the production recipe (campaign34.preflight_pretraining) + fiber + surface norm
run c39_mae_base python mae_pretrain_nnunet.py --name c39_mae_base --scroll-ids $IDS \
    --require-all-scrolls --physical-round-robin --ctx 96 --ds 1 --depth 8 --d-start 10 --d-end 18 \
    --steps 2000 --batch-size 32 --accum-steps 1 --lr 3e-4 --no-figures \
    --early-2d-unet --gated-stems --norm-mode ibn_full "${FIBER_SN[@]}"

# 3. cross-resolution continuations of the base (replay branch on all 37 volumes)
crossres() {  # crossres <name> <args...>
    local name=$1; shift
    run "$name" python crossres/mae_pretrain_crossres.py --name "$name" --init-weights "$BASE" \
        --scroll-ids $IDS --exclude-holdout-pairs "${FIBER_SN[@]}" "$@"
}
crossres c39_mae_crossres_depth --plan depth --sr-head deep --head-lr-mult 5
crossres c39_mae_crossres_xyz --plan xyz --sr-head deep --head-lr-mult 5
crossres c39_mae_slab --plan none

# 4. plan U: the upsampler sees the same pairs as before, but in the new (surface-anchored) normalisation
run c39_upsampler_learned python crossres/train_upsampler.py --name c39_upsampler_learned --norm-mode surface_anchor
crossres c39_mae_upsampled_learned --plan none --upsampler models/c39_upsampler_learned.pth --accum-steps 2
crossres c39_mae_upsampled_trilinear --plan none --upsampler trilinear --accum-steps 2

# 5. plan R: degrader (pooled 2.4 um -> native 9.36 um), then the translated fine-scan siblings
run c39_degrader python crossres/train_degrader.py --name c39_degrader --batch-size 4
run c39_translate python assemble_training_segments.py --degrader models/c39_degrader.pth
run c39_noise_bank python crossres/build_noise_bank.py

echo "[queue] $(date +%H:%M) all campaign-39 pretrains finished"
