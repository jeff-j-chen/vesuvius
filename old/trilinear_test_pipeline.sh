#!/usr/bin/env bash
# re-render every test fragment (main, then quick_patches_from_D, then atlas) with trilinear 9.362um sampling in
# tmux 'dl'; each finished fragment is queued for and evaluated by the and_mask + new_letters_10 ensemble (test_inference_and_letters.ipynb) while assembly continues.
set -u
cd /vesuvius
STATE=_ves_tmp/test_eval

rm -rf "$STATE" && mkdir -p "$STATE"
HOOK="python3 old/test_eval.py tidy; python3 old/test_eval.py enqueue {zid}"
tmux send-keys -t dl "cd /vesuvius && python3 assemble_test_segments.py --include-extras --parallel 3 \
--extras-order quick_patches_from_D,atlas --min-free-gb 25 --on-complete '$HOOK' 2>&1 | tee _ves_tmp/test_assembly.log; \
python3 old/test_eval.py end; echo TEST_ASSEMBLY_DONE" Enter
echo "[pipeline] $(date '+%F %T') assembly started in tmux 'dl' (log _ves_tmp/test_assembly.log)"

nohup python3 old/test_eval.py serve-ensemble > _ves_tmp/test_eval_ensemble.log 2>&1 &
echo "[pipeline] ensemble evaluator pid=$! (log _ves_tmp/test_eval_ensemble.log)"
