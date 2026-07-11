#!/bin/bash
# 40-task ImageNet-R (5cls x 40) reruns, queued behind the gfinal queue
# daemon (pid passed as $1). Runs sequentially on GPU 3:
#   1. inr_base        (calib off, ridge off)
#   2. inr_ridge_only  (calib off, ridge w0.2, dump_eval on -> 40-task sweep)
# Status appended to logs/queue_inr40.status.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
GFINAL_DAEMON=$1
while kill -0 "$GFINAL_DAEMON" 2>/dev/null; do
    sleep 300
done
sleep 60
for cfg in inr_base inr_ridge_only; do
    echo "$(date) launching ${cfg}" >> logs/queue_inr40.status
    CUDA_VISIBLE_DEVICES=3 /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/${cfg}.json" \
        > "logs/nohup_${cfg}_40t.out" 2>&1
    echo "$(date) finished ${cfg} (exit $?)" >> logs/queue_inr40.status
done
echo "$(date) inr40 queue done" >> logs/queue_inr40.status
