#!/bin/bash
# Track peak VRAM of given processes via nvidia-smi polling.
# usage: vram_sampler.sh <outfile> <pid:name> [pid:name ...]
# Appends a line on each new peak; writes "FINAL <name> peak=<MiB>" when all pids exit.
out=$1; shift
declare -A max
while true; do
    alive=0
    for spec in "$@"; do
        pid=${spec%%:*}; name=${spec#*:}
        kill -0 "$pid" 2>/dev/null && alive=1
        m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
            | awk -F', ' -v p="$pid" '$1==p{print $2}')
        if [ -n "$m" ] && [ "$m" -gt "${max[$name]:-0}" ]; then
            max[$name]=$m
            echo "$(date '+%F %T') $name pid=$pid new_peak=${m}MiB" >> "$out"
        fi
    done
    [ "$alive" -eq 0 ] && break
    sleep 30
done
for spec in "$@"; do
    name=${spec#*:}
    echo "FINAL $name peak=${max[$name]:-NA}MiB" >> "$out"
done
