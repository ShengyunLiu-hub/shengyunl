#!/bin/bash
# Confirmation runs for the chosen global inference config
# (branch_calibration pos_shift + ridge ensemble_weight 0.15),
# sequentially on GPU 3: vtab (~30min) -> ina (~50min) -> inr (~2.5h).
# Status appended to logs/queue_gfinal.status.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
for cfg in vtab_gfinal ina_gfinal inr_gfinal; do
    echo "$(date) launching ${cfg}" >> logs/queue_gfinal.status
    CUDA_VISIBLE_DEVICES=3 /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/${cfg}.json" \
        > "logs/nohup_${cfg}.out" 2>&1
    echo "$(date) finished ${cfg} (exit $?)" >> logs/queue_gfinal.status
done
echo "$(date) gfinal queue done" >> logs/queue_gfinal.status
