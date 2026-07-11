#!/bin/bash
# Wait for the GPU-3 ablation queue to finish, then run inr_ridge_only
# (B1 off, ridge on, dump_eval on) on GPU 3 for the ImageNet-R offline sweep.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
until grep -q "gpu3 ablation queue done" logs/queue_gpu3_ablations.status 2>/dev/null; do
    sleep 300
done
echo "$(date) launching inr_ridge_only" >> logs/queue_gpu3_ablations.status
CUDA_VISIBLE_DEVICES=3 /home/lsy2025/.conda/envs/cl_lora/bin/python main.py exps/inr_ridge_only.json \
    > logs/nohup_inr_ridge_only.out 2>&1
echo "$(date) finished inr_ridge_only (exit $?)" >> logs/queue_gpu3_ablations.status
