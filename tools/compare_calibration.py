"""Offline comparison of branch-calibration schemes on dumped eval logits.

Usage:
    python tools/compare_calibration.py logs/cllora/cifar224/0/5/analysis_<prefix>_<seed> [task_id]

Reads eval_task{t}.npz produced by models/cllora.py (branch_calibration.dump_eval)
and reports top-1 / task-id accuracy for each calibration scheme, so schemes can
be selected without retraining. If task_id is omitted, the largest dumped task
is used and a per-task curve is printed for every scheme.
"""
import glob
import os
import re
import sys

import numpy as np


def cls_range(i, init_cls, inc):
    if i == 0:
        return 0, init_cls
    start = init_cls + (i - 1) * inc
    return start, start + inc


def load_eval(path):
    data = np.load(path)
    stats = {
        "pos": dict(zip(data["pos_stats_branches"].tolist(), data["pos_stats"])),
        "neg": dict(zip(data["neg_stats_branches"].tolist(), data["neg_stats"])),
    }
    return (
        data["raw_logits"].astype(np.float32),
        data["targets"].astype(np.int64),
        stats,
        int(data["init_cls"]),
        int(data["increment"]),
        int(data["cur_task"]),
    )


def _borrowed_neg(stats, i):
    """Neg stats for branch i; the newest branch (no negatives yet) borrows the
    mean of the other branches' neg stats — same cosine space, so scales match."""
    if i in stats["neg"]:
        return stats["neg"][i]
    if len(stats["neg"]) == 0:
        return None
    arr = np.array(list(stats["neg"].values()), dtype=np.float64)
    return arr[:, 0].mean(), arr[:, 1].mean()


def calibrate(logits, stats, scheme, init_cls, inc, cur_task, eps=1e-6):
    out = logits.copy()
    for i in range(cur_task + 1):
        s, e = cls_range(i, init_cls, inc)
        if scheme == "none":
            continue
        if scheme == "neg_borrow":
            borrowed = _borrowed_neg(stats, i)
            if borrowed is None:
                continue
            mu, sd = borrowed
            out[:, s:e] = (logits[:, s:e] - mu) / max(sd, eps)
            continue
        if scheme == "llr":
            # Gaussian log-likelihood ratio between the branch's positive and
            # negative score models; monotone within a branch (given sd_p<sd_n
            # ordering it stays effectively rank-preserving in practice).
            borrowed = _borrowed_neg(stats, i)
            if borrowed is None or i not in stats["pos"]:
                continue
            mu_n, sd_n = borrowed
            mu_p, sd_p = stats["pos"][i]
            sd_n, sd_p = max(sd_n, eps), max(sd_p, eps)
            z_n = (logits[:, s:e] - mu_n) / sd_n
            z_p = (logits[:, s:e] - mu_p) / sd_p
            out[:, s:e] = 0.5 * (z_n ** 2 - z_p ** 2) + np.log(sd_n / sd_p)
            continue
        if scheme in ("pos_shift", "pos_zscore", "pos_scale"):
            if i not in stats["pos"]:
                continue
            mu, sd = stats["pos"][i]
        elif scheme == "neg_zscore":
            if i in stats["neg"]:
                mu, sd = stats["neg"][i]
            elif i in stats["pos"]:
                mu, sd = stats["pos"][i]
            else:
                continue
        elif scheme == "hybrid":
            # shift by neg mean when available (OOD baseline), scale by pos std
            if i in stats["neg"] and i in stats["pos"]:
                mu = stats["neg"][i][0]
                sd = stats["pos"][i][1]
            elif i in stats["pos"]:
                mu, sd = stats["pos"][i]
            else:
                continue
        else:
            raise ValueError(scheme)
        if scheme == "pos_shift":
            out[:, s:e] = logits[:, s:e] - mu
        elif scheme == "pos_scale":
            out[:, s:e] = logits[:, s:e] / max(sd, eps)
        else:
            out[:, s:e] = (logits[:, s:e] - mu) / max(sd, eps)
    return out


def evaluate(logits, targets, init_cls, inc, cur_task):
    pred = logits.argmax(axis=1)
    top1 = (pred == targets).mean() * 100
    task_of = lambda c: np.maximum((c - init_cls) // inc + 1, 0)
    task_correct = (task_of(pred) == task_of(targets)).mean() * 100
    # oracle: restrict logits to the true task's block
    oracle_pred = np.empty_like(targets)
    for i in range(cur_task + 1):
        s, e = cls_range(i, init_cls, inc)
        mask = (targets >= s) & (targets < e)
        if mask.any():
            oracle_pred[mask] = s + logits[mask, s:e].argmax(axis=1)
    oracle = (oracle_pred == targets).mean() * 100
    return top1, task_correct, oracle


SCHEMES = ["none", "pos_shift", "pos_scale", "pos_zscore", "neg_zscore", "neg_borrow", "hybrid", "llr"]


def main():
    dump_dir = sys.argv[1]
    files = sorted(
        glob.glob(os.path.join(dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)),
    )
    if not files:
        sys.exit("no eval_task*.npz in {}".format(dump_dir))
    if len(sys.argv) > 2:
        files = [os.path.join(dump_dir, "eval_task{}.npz".format(sys.argv[2]))]

    curves = {s: [] for s in SCHEMES}
    task_curves = {s: [] for s in SCHEMES}
    for path in files:
        logits, targets, stats, init_cls, inc, cur_task = load_eval(path)
        for scheme in SCHEMES:
            cal = calibrate(logits, stats, scheme, init_cls, inc, cur_task)
            top1, task_correct, oracle = evaluate(cal, targets, init_cls, inc, cur_task)
            curves[scheme].append(top1)
            task_curves[scheme].append(task_correct)
        print("task {:2d} | oracle {:.2f} | ".format(cur_task, oracle)
              + " ".join("{}: {:.2f}/{:.2f}".format(s, curves[s][-1], task_curves[s][-1]) for s in SCHEMES))

    print("\n=== summary (top1 avg over dumped tasks / final | task-id final) ===")
    for scheme in SCHEMES:
        print("{:>12s}: avg {:.2f}  final {:.2f} | task-id final {:.2f}".format(
            scheme, float(np.mean(curves[scheme])), curves[scheme][-1], task_curves[scheme][-1]))


if __name__ == "__main__":
    main()
