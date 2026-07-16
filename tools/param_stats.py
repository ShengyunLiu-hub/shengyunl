"""#Params columns for the paper tables, parsed from the ms_* training logs.

Per dataset (averaged over seeds):
  - trainable/task: mean over tasks of the logged 'Trainable params' (what gets
    gradient updates each task; declines over tasks via rank reduction)
  - stored total:   logged 'All params' at task 0 and at the final task, and the
    per-task storage growth (the CL-LoRA-inherited per-task adapter cost)
  - ridge head:     768 x C fp32 weights, stored but never gradient-trained
    (closed-form); report separately for the storage footnote.

Usage: python tools/param_stats.py [--logs 'logs/nohup_ms_*.out']
"""
import argparse
import glob
import re

import numpy as np

BACKBONE = 86.0  # M, ViT-B/16 reference base used for the % column
CLASSES = {"cifar": 100, "cifar10t": 100, "inr": 200, "inr10t": 200, "ina": 200, "vtab": 50}


def parse_log(path):
    trainable, allp = [], []
    for line in open(path, errors="ignore"):
        m = re.search(r"\[trainer\.py\] => Trainable params: (\d+)", line)
        if not m:
            m = re.search(r"^Trainable params: (\d+)", line)
        if m:
            trainable.append(int(m.group(1)))
            continue
        m = re.search(r"All params: (\d+)", line)
        if m:
            allp.append(int(m.group(1)))
    return trainable, allp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/nohup_ms_*.out")
    args = ap.parse_args()

    per_ds = {}
    for f in sorted(glob.glob(args.logs)):
        m = re.match(r"nohup_ms_([a-z0-9]+)_(\d+)\.out", f.split("/")[-1])
        if not m:
            continue
        ds, seed = m.group(1), int(m.group(2))
        trainable, allp = parse_log(f)
        if not trainable:
            continue
        per_ds.setdefault(ds, []).append((seed, trainable, allp))

    hdr = ("%-9s %2s %6s | %-18s | %-10s %-10s %-12s | %s" % (
        "dataset", "n", "tasks", "trainable/task M(%)",
        "stored@t0", "stored@T", "growth/task", "ridge head"))
    print(hdr)
    print("-" * len(hdr))
    for ds in ["cifar", "cifar10t", "inr", "inr10t", "ina", "vtab"]:
        if ds not in per_ds:
            continue
        runs = per_ds[ds]
        ntasks = len(runs[0][1])
        tr_mean = np.mean([np.mean(t) for _, t, _ in runs]) / 1e6
        a0 = np.mean([a[0] for _, _, a in runs if a]) / 1e6
        aT = np.mean([a[-1] for _, _, a in runs if a]) / 1e6
        growth = (aT - a0) / max(ntasks - 1, 1)
        ridge = 768 * CLASSES.get(ds, 0) / 1e6
        print("%-9s %2d %6d | %6.3fM (%.2f%%)     | %8.2fM %8.2fM %8.3fM/task | %.3fM (no grad)" % (
            ds, len(runs), ntasks, tr_mean, tr_mean / BACKBONE * 100,
            a0, aT, growth, ridge))
    print("\nNote: trainable/task = gradient-updated params per task (rank declines over tasks);")
    print("stored = logged 'All params' (backbone + accumulated adapters + heads);")
    print("ridge head is stored, closed-form solved, never gradient-trained; Fisher is train-time only.")


if __name__ == "__main__":
    main()
