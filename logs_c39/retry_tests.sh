#!/bin/bash
# rerun the test assemblies OOM-killed by the 13-way parallel start, keeping at most two alive (~15 GB peak each)
cd /vesuvius
for z in 20260814140748 20260720090842 20260723112922 20260921094413 20260918132724 20260922073234 \
         20260925085345 20260928000001; do
    while [[ $(pgrep -fc "[a]ssemble_test_segments.py") -ge 2 ]]; do sleep 30; done
    nohup python assemble_test_segments.py --workers 12 --only $z > logs_c39/assemble/test_$z.log 2>&1 &
    sleep 60
done
wait
echo "retry_tests done"
