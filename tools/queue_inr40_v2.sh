#!/bin/bash
# 40-task ImageNet-R rerun v2 (ewc.gamma=0.9 after the gamma=1.0 Fisher
# blow-up at task 37). Sequential on GPU 2:
#   1. inr_gfinal      (pos_shift + ridge w0.15)   -- redo, v1 NaN'd
#   2. inr_base        (calib off, ridge off)
#   3. inr_ridge_only  (ridge w0.2, dump_eval on -> 40-task sweep)
# Status appended to logs/queue_inr40_v2.status.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
for cfg in inr_gfinal inr_base inr_ridge_only; do
    echo "$(date) launching ${cfg}" >> logs/queue_inr40_v2.status
    CUDA_VISIBLE_DEVICES=2 /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/${cfg}.json" \
        > "logs/nohup_${cfg}_40t_v2.out" 2>&1
    rc=$?
    echo "$(date) finished ${cfg} (exit ${rc})" >> logs/queue_inr40_v2.status
    # NaN guard: stop the queue if the run died or collapsed again
    log=$(ls -t logs/cllora/imagenetr/0/5/${cfg}_[0-9]*.log 2>/dev/null | head -1)
    if [ "$rc" -ne 0 ] || { [ -n "$log" ] && grep -q "Loss nan" "$log"; }; then
        echo "$(date) ABORT: ${cfg} failed or went NaN" >> logs/queue_inr40_v2.status
        exit 1
    fi
done
echo "$(date) inr40 v2 queue done" >> logs/queue_inr40_v2.status
