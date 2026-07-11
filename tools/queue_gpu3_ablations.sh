#!/bin/bash
# Wait for the inr_b1b3 run (pid 2455664) to finish, then run the ablation
# queue sequentially on GPU 3:
#   1. vtab_ridge_only  (B1 off, ridge on, dump_eval on)  ~30 min
#   2. ina_ridge_only   (B1 off, ridge on, dump_eval on)  ~50 min
#   3. ina_b1b3_noewc   (full b1b3, EWC off)              ~50 min
# Status appended to logs/queue_gpu3_ablations.status.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
while kill -0 2455664 2>/dev/null; do
    sleep 300
done
sleep 60
for cfg in vtab_ridge_only ina_ridge_only ina_b1b3_noewc; do
    echo "$(date) launching ${cfg}" >> logs/queue_gpu3_ablations.status
    CUDA_VISIBLE_DEVICES=3 /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/${cfg}.json" \
        > "logs/nohup_${cfg}.out" 2>&1
    echo "$(date) finished ${cfg} (exit $?)" >> logs/queue_gpu3_ablations.status
done
echo "$(date) gpu3 ablation queue done" >> logs/queue_gpu3_ablations.status
# appended 2026-07-08: this file is re-read by bash as it executes line by line,
# but the running daemon may have already cached past this point — so the inr
# follow-up gets its own daemon (tools/queue_inr_dump.sh) instead.
