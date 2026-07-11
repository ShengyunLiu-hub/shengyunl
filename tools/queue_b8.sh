#!/bin/bash
# B8 validation queue: waits for the pid given in $1 to exit, then runs the
# given configs sequentially on the GPU given in $2.
# Usage: nohup bash tools/queue_b8.sh <wait_pid> <gpu> <cfg1> [cfg2 ...] &
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
WAIT_PID=$1; GPU=$2; shift 2
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for cfg in "$@"; do
    echo "$(date) launching ${cfg} on GPU ${GPU}" >> logs/queue_b8.status
    CUDA_VISIBLE_DEVICES=$GPU /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/${cfg}.json" \
        > "logs/nohup_${cfg}.out" 2>&1
    echo "$(date) finished ${cfg} (exit $?)" >> logs/queue_b8.status
done
echo "$(date) b8 queue on GPU ${GPU} done" >> logs/queue_b8.status
