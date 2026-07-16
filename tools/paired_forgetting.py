"""Paired forgetting (base vs gfinal) recovered offline from dump_eval dumps.

Replicates the scoring in tools/sweep_ensemble.py for exactly two configs —
base (scheme none, w=0) and gfinal (pos_shift, w=0.15) — plus the grouped
accuracy matrix + forgetting definition from trainer.py/utils.toolkit
(per-group accuracies rounded to 2 decimals, F = mean over groups<T of
max-over-steps minus final-step).

Per-dump results are cached as sweeps/forgetting_<name>.txt so reruns are
incremental; the aggregate table is rebuilt from whatever caches exist.

Usage:
    python tools/paired_forgetting.py <dump_dir> [<dump_dir> ...]   # compute + cache
    python tools/paired_forgetting.py --table                       # aggregate table only
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

OUT_DIM = 768
CACHE_DIR = "logs/multiseed/sweeps"
CONFIGS = [("none", 0.0), ("pos_shift", 0.15)]  # base, gfinal


def zscore(m, eps=1e-8):
    return (m - m.mean()) / (m.std() + eps)


def l2norm(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def grouped_acc(pred, targets, init_cls, inc):
    """toolkit.accuracy grouped values (2-decimal rounded), in task order."""
    groups = [(0, init_cls)]
    for c in range(init_cls, int(targets.max()) + 1, inc):
        if c == 0:
            continue
        groups.append((c, c + inc))
    vals = []
    for s, e in groups:
        idx = np.where((targets >= s) & (targets < e))[0]
        if len(idx) == 0:
            break
        vals.append(np.around((pred[idx] == targets[idx]).mean() * 100, 2))
    return vals


def forgetting(matrix):
    """matrix[step] = list of per-group acc at that step (trainer.py:131-141)."""
    task = len(matrix) - 1
    if task == 0:
        return 0.0
    tbl = np.zeros((task + 1, task + 1))
    for i, line in enumerate(matrix):
        tbl[i, :len(line)] = line
    tbl = tbl.T
    return float(np.mean((np.max(tbl, axis=1) - tbl[:, task])[:task]))


def process(dump_dir, lam=1.0):
    eval_files = sorted(
        glob.glob(os.path.join(dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)),
    )
    if not eval_files:
        raise RuntimeError("no eval dumps in " + dump_dir)

    matrices = {cfg: [] for cfg in CONFIGS}
    curves = {cfg: [] for cfg in CONFIGS}
    G = np.zeros((OUT_DIM, OUT_DIM), dtype=np.float64)
    C = None

    for path in eval_files:
        t = int(re.search(r"eval_task(\d+)\.npz", path).group(1))
        d = np.load(path)
        init_cls, inc = int(d["init_cls"]), int(d["increment"])
        total_cls = init_cls + t * inc
        targets = d["targets"].astype(np.int64)
        raw = d["raw_logits"].astype(np.float32)

        tr = np.load(os.path.join(dump_dir, "train_emb_task{}_branch0.npz".format(t)))
        x = l2norm(tr["embeddings"].astype(np.float32))
        y = tr["labels"].astype(np.int64)
        G += x.T @ x
        if C is None:
            C = np.zeros((OUT_DIM, total_cls), dtype=np.float64)
        elif C.shape[1] < total_cls:
            C = np.pad(C, ((0, 0), (0, total_cls - C.shape[1])))
        onehot = np.zeros((x.shape[0], total_cls), dtype=np.float64)
        onehot[np.arange(x.shape[0]), y] = 1.0
        C += x.astype(np.float64).T @ onehot

        W = np.linalg.solve(G + lam * np.eye(OUT_DIM), C).astype(np.float32)
        x0 = l2norm(d["features"].astype(np.float32)[:, :OUT_DIM])
        ridge_logits = x0 @ W

        calib = raw.copy()
        pos_b = d["pos_stats_branches"].astype(int)
        pos_mu = d["pos_stats"][:, 0]
        for b, mu in zip(pos_b, pos_mu):
            if b > t:
                continue
            s = b * inc if init_cls == inc else (0 if b == 0 else init_cls + (b - 1) * inc)
            e = s + (init_cls if (b == 0 and init_cls != inc) else inc)
            calib[:, s:e] = raw[:, s:e] - mu

        diag = {"none": raw, "pos_shift": calib}
        zr = zscore(ridge_logits)
        for scheme, w in CONFIGS:
            zd = zscore(diag[scheme])
            scores = w * zr + (1.0 - w) * zd
            pred = scores.argmax(1)
            matrices[(scheme, w)].append(grouped_acc(pred, targets, init_cls, inc))
            curves[(scheme, w)].append((pred == targets).mean() * 100)

    out = {}
    for cfg in CONFIGS:
        out[cfg] = (forgetting(matrices[cfg]),
                    float(np.mean(curves[cfg])), curves[cfg][-1])
    return out


def cache_path(dump_dir):
    return os.path.join(CACHE_DIR, "forgetting_%s.txt" % os.path.basename(dump_dir).replace("analysis_", ""))


def write_cache(dump_dir, res):
    with open(cache_path(dump_dir), "w") as f:
        f.write("scheme          w  forgetting      avg    final\n")
        for (scheme, w), (fg, avg, fin) in res.items():
            f.write("%-10s %6.2f %11.4f %8.2f %8.2f\n" % (scheme, w, fg, avg, fin))


def parse_cache(path):
    rows = {}
    for line in open(path):
        m = re.match(r"(none|pos_shift)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows[(m.group(1), float(m.group(2)))] = float(m.group(3))
    return rows


def table():
    data = {}
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "forgetting_ms_*.txt"))):
        m = re.match(r"forgetting_ms_([a-z0-9]+)_(\d+)\.txt", os.path.basename(f))
        if not m:
            continue
        ds, seed = m.group(1), int(m.group(2))
        rows = parse_cache(f)
        if ("none", 0.0) not in rows or ("pos_shift", 0.15) not in rows:
            continue
        data.setdefault(ds, {})[seed] = (rows[("none", 0.0)], rows[("pos_shift", 0.15)])

    print("paired forgetting;  base = none w0, gfinal = pos_shift w0.15")
    hdr = "%-8s %2s  %-16s %-16s %s" % ("dataset", "n", "base mean±std", "gfinal mean±std", "paired diff")
    print(hdr)
    print("-" * len(hdr))
    for ds in ["cifar", "cifar10t", "inr", "inr10t", "ina", "vtab"]:
        if ds not in data:
            continue
        seeds = sorted(data[ds])
        base = np.array([data[ds][s][0] for s in seeds])
        gf = np.array([data[ds][s][1] for s in seeds])
        diff = gf - base
        print("%-8s %2d  %6.2f ±%-8.2f %6.2f ±%-8.2f %+5.2f ±%.2f" % (
            ds, len(seeds), base.mean(), base.std(ddof=1), gf.mean(), gf.std(ddof=1),
            diff.mean(), diff.std(ddof=1)))
        per_seed = "  ".join("%d: %.2f->%.2f" % (s, b, g) for s, (b, g) in sorted(data[ds].items()))
        print("         seeds  %s" % per_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dirs", nargs="*")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    for d in args.dump_dirs:
        d = d.rstrip("/")
        if not args.force and os.path.exists(cache_path(d)):
            print("cached, skip:", d)
            continue
        print("processing:", d, flush=True)
        try:
            res = process(d, args.lam)
        except RuntimeError as e:
            print("  SKIP:", e, file=sys.stderr)
            continue
        write_cache(d, res)
        for cfg, (fg, avg, fin) in res.items():
            print("  %-10s w=%.2f  forgetting %.4f  avg %.2f  final %.2f" % (cfg[0], cfg[1], fg, avg, fin))

    if args.table or not args.dump_dirs:
        table()


if __name__ == "__main__":
    main()
