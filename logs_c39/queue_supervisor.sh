#!/bin/bash
# restart the resumable campaign-39 queue when it stops on a transient download failure (timeouts, http 429)
cd /vesuvius
for attempt in 1 2 3; do
    while pgrep -f "run_queue_c3[9].sh" > /dev/null; do sleep 60; done
    [[ -e logs_c39/done/c39_noise_bank ]] && { echo "[supervisor] queue finished"; exit 0; }
    tail -n 3 logs_c39/queue.log | grep -qE "\[queue\] (test|training) .* failed|missing zarrs" \
        || { echo "[supervisor] queue stopped on a non-download error; not restarting"; exit 1; }
    echo "[supervisor] $(date +%H:%M) download failure; restart $attempt after 10 min"
    sleep 600
    nohup crossres/run_queue_c39.sh >> logs_c39/queue.log 2>&1 &
    sleep 30
done
