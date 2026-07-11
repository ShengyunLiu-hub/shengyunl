#!/bin/bash
# Wait for the two ImageNet-R runs to finish, then measure peak VRAM of the
# b1b3 config on cifar/ina/vtab via 1-epoch-per-task smoke runs on GPU 2.
# VRAM peaks land in logs/vram_smokes.log. Accuracy of these smokes is meaningless.
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
while kill -0 2455663 2>/dev/null || kill -0 2455664 2>/dev/null; do
    sleep 300
done
sleep 60
for name in vtab ina cifar; do
    CUDA_VISIBLE_DEVICES=2 nohup /home/lsy2025/.conda/envs/cl_lora/bin/python main.py "exps/vram_smoke_${name}.json" \
        > "logs/nohup_vram_smoke_${name}.out" 2>&1 &
    pid=$!
    echo "$(date) launched vram_smoke_${name}, pid $pid" >> logs/queue_vram_smokes.status
    bash tools/vram_sampler.sh logs/vram_smokes.log "${pid}:${name}"
    echo "$(date) finished vram_smoke_${name}" >> logs/queue_vram_smokes.status
done
echo "$(date) all smokes done" >> logs/queue_vram_smokes.status
