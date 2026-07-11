#!/bin/bash
# Session-independent queue: wait for the b1_shift full run to finish,
# then launch the B1+B3 (pos_shift calibration + ridge ensemble) full run on GPU 2.
# Launched with nohup so it survives terminal / Claude session exit.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
while pgrep -f "main.py exps/sd_expD_rr_tc_ewc_b1shift.json" >/dev/null; do
    sleep 300
done
sleep 60
CUDA_VISIBLE_DEVICES=2 nohup /home/lsy2025/.conda/envs/cl_lora/bin/python main.py exps/sd_expD_rr_tc_ewc_b1b3.json \
    > logs/nohup_b1b3_ridge_full.out 2>&1 &
echo "$(date) launched b1b3, pid $!" >> logs/queue_b1b3.status
