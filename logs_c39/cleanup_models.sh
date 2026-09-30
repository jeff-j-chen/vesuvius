#!/bin/bash
# after crossres/run_queue_c39.sh finishes every step, delete the superseded pretrains in ./models (top level only).
# kept: models/c39_* (the new source of truth), the N2V denoiser (the noise bank and campaign 38's denoiser arm
# need it) and the fine-tuned campaign runs in models/archs*/.
cd /vesuvius
while pgrep -f "run_queue_c3[9].sh|queue_superviso[r].sh" > /dev/null; do sleep 120; done
steps="c39_mae_base c39_mae_crossres_depth c39_mae_crossres_xyz c39_mae_slab c39_upsampler_learned
       c39_mae_upsampled_learned c39_mae_upsampled_trilinear c39_degrader c39_translate c39_noise_bank"
for s in $steps; do
    [[ -e logs_c39/done/$s ]] || { echo "[cleanup] $s not done; nothing deleted"; exit 1; }
done
while pgrep -f "eval_baseline[s].py" > /dev/null; do sleep 120; done
for f in models/*; do
    [[ -f "$f" ]] || continue
    case "$(basename "$f")" in
        c39_*|denoiser_n2v_block3_4k.*) ;;
        *) echo "[cleanup] rm $f"; rm -f -- "$f" ;;
    esac
done
echo "[cleanup] done; kept: $(ls models | tr '\n' ' ')"
