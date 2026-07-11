#!/bin/bash
# Attach vram_sampler to each 40-task inr run as it appears.
# Peaks land in logs/vram_inr40_v2.log. Gives up on a run if it hasn't
# appeared within 24h (queue died upstream).
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1
for p in inr_gfinal inr_base inr_ridge_only; do
    waited=0
    until pid=$(pgrep -f "main.py exps/${p}.json" | head -1) && [ -n "$pid" ]; do
        sleep 60
        waited=$((waited+60))
        if [ "$waited" -ge 86400 ]; then
            echo "GAVE UP waiting for $p" >> logs/vram_inr40_v2.log
            continue 2
        fi
    done
    bash tools/vram_sampler.sh logs/vram_inr40_v2.log "${pid}:${p}"
done
echo "$(date) all inr40 vram watches done" >> logs/vram_inr40_v2.log
