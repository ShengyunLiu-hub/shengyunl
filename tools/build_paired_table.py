"""Aggregate multi-seed sweep outputs into the paired base-vs-gfinal table.

Reads logs/multiseed/sweeps/sweep_analysis_ms_<ds>_<seed>.txt (produced by
tools/sweep_ensemble.py --weights 0.0 0.15) and reports, per dataset:
mean±std of base (scheme none, w=0), gfinal (pos_shift, w=0.15), and the
per-seed paired gain. Rerun as new seeds land — it picks up whatever exists.

Usage: python tools/build_paired_table.py [--metric avg|final]
"""
import argparse
import glob
import os
import re

import numpy as np

SWEEP_DIR = "logs/multiseed/sweeps"
PAPER = {  # CL-LoRA CVPR25 Table 1 (arXiv 2505.24816): avg, final
    "cifar": (91.02, 85.32),
    "inr": (81.58, 74.51),
    "ina": (70.15, 60.54),
    "vtab": (94.57, 94.29),
}


def parse(path):
    rows = {}
    for line in open(path):
        m = re.match(r"(none|pos_shift)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows[(m.group(1), float(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["avg", "final"], default="avg")
    args = ap.parse_args()
    idx = 0 if args.metric == "avg" else 1

    data = {}
    for f in sorted(glob.glob(os.path.join(SWEEP_DIR, "sweep_analysis_ms_*.txt"))):
        m = re.match(r"sweep_analysis_ms_([a-z0-9]+)_(\d+)\.txt", os.path.basename(f))
        ds, seed = m.group(1), int(m.group(2))
        rows = parse(f)
        if ("none", 0.0) not in rows or ("pos_shift", 0.15) not in rows:
            continue
        data.setdefault(ds, {})[seed] = (
            rows[("none", 0.0)][idx], rows[("pos_shift", 0.15)][idx])

    print("metric = %s accuracy;  base = none w0, gfinal = pos_shift w0.15" % args.metric)
    hdr = "%-8s %2s  %-16s %-16s %-14s %s" % (
        "dataset", "n", "base mean±std", "gfinal mean±std", "paired gain", "paper CL-LoRA")
    print(hdr)
    print("-" * len(hdr))
    for ds in ["cifar", "inr", "ina", "vtab"]:
        if ds not in data:
            continue
        seeds = sorted(data[ds])
        base = np.array([data[ds][s][0] for s in seeds])
        gf = np.array([data[ds][s][1] for s in seeds])
        gain = gf - base
        print("%-8s %2d  %6.2f ±%-8.2f %6.2f ±%-8.2f %+5.2f ±%-6.2f %6.2f" % (
            ds, len(seeds), base.mean(), base.std(ddof=1), gf.mean(), gf.std(ddof=1),
            gain.mean(), gain.std(ddof=1), PAPER[ds][idx]))
        per_seed = "  ".join("%d: %.2f->%.2f (%+.2f)" % (s, b, g, g - b)
                             for s, (b, g) in sorted(data[ds].items()))
        print("         seeds  %s" % per_seed)


if __name__ == "__main__":
    main()
