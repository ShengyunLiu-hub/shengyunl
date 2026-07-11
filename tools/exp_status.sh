#!/bin/bash
# One-shot experiment status summary. Claude runs this when the user says "查看".
cd /media/hdd2/lsy2025/CL-LoRA_SD_RR_EWC || exit 1

echo "===== running processes ====="
ps aux | grep "[m]ain.py exps" | awk '{printf "pid %s  cpu %s%%  elapsed %s  %s\n", $2, $3, $10, $NF}' | sort -u

echo ""
echo "===== GPU ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "===== queue daemons ====="
for q in queue_gfinal queue_inr40_v2; do
    if pgrep -f "tools/${q}.sh" >/dev/null; then
        echo "$q: waiting/running"
    fi
    [ -f "logs/${q}.status" ] && tail -2 "logs/${q}.status" | sed "s/^/  [$q] /"
done

# prefix:logdir pairs — newest full runs per dataset
RUNS="
b1b3_ridge_full:logs/cllora/cifar224/0/5
ina4:logs/cllora/imageneta/0/20
ina_b1b3:logs/cllora/imageneta/0/20
ina_ridge_only:logs/cllora/imageneta/0/20
ina_b1b3_noewc:logs/cllora/imageneta/0/20
vtab7:logs/cllora/vtab/0/10
vtab_b1b3:logs/cllora/vtab/0/10
vtab_ridge_only:logs/cllora/vtab/0/10
vtab_gfinal:logs/cllora/vtab/0/10
ina_gfinal:logs/cllora/imageneta/0/20
inr_gfinal:logs/cllora/imagenetr/0/5
inr_base:logs/cllora/imagenetr/0/5
inr_ridge_only:logs/cllora/imagenetr/0/5
"

echo ""
for pair in $RUNS; do
    prefix=${pair%%:*}; dir=${pair##*:}
    log=$(ls -t "$dir/${prefix}"_[0-9]*.log 2>/dev/null | head -1)
    [ -z "$log" ] && continue
    echo "===== $prefix ====="
    grep -oE "Task [0-9]+, Epoch [0-9]+/[0-9]+ => Loss [0-9.a-z]+" "$log" | tail -1
    grep "Average Accuracy (CNN)" "$log" | tail -1 | grep -oE "Average Accuracy.*"
    grep -E "Task correct" "$log" | tail -2 | grep -oE "Task correct.*"
    grep "Forgetting (CNN)" "$log" | tail -1 | grep -oE "Forgetting.*"
    echo ""
done

echo "===== cifar baseline reference ====="
echo "baseline : avg 92.39 | final 87.49 | forgetting 6.09 | task-correct 87.57"
echo "B4 fixed : avg 92.30 | final 87.02 | forgetting 6.07 | task-correct 87.12"
echo "b1b3     : avg 93.98 | final 90.08 | forgetting 4.04"
