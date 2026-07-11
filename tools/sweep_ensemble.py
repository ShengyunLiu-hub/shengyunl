"""Offline sweep of ridge ensemble_weight x calibration scheme on dump_eval dumps.

Usage:
    python tools/sweep_ensemble.py logs/.../analysis_<prefix>_<seed> [--lambda 1.0]

Replicates models/cllora.py inference exactly (branch-0 ridge, global z-score
ensemble, pos_shift calibration from the pos stats stored in the eval dumps)
and reports, for each (scheme, weight): per-run Average Accuracy (mean of the
top1 at every incremental step) and final-task top1.
"""
import argparse
import glob
import os
import re

import numpy as np

OUT_DIM = 768


def zscore(m, eps=1e-8):
    return (m - m.mean()) / (m.std() + eps)


def l2norm(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--weights", type=float, nargs="*",
                    default=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0])
    args = ap.parse_args()

    eval_files = sorted(
        glob.glob(os.path.join(args.dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)),
    )
    if not eval_files:
        raise SystemExit("no eval dumps in " + args.dump_dir)

    schemes = ["none", "pos_shift"]
    # acc[(scheme, w)] = list of per-task top1
    acc = {(s, w): [] for s in schemes for w in args.weights}

    G = np.zeros((OUT_DIM, OUT_DIM), dtype=np.float64)
    C = None

    for path in eval_files:
        t = int(re.search(r"eval_task(\d+)\.npz", path).group(1))
        d = np.load(path)
        init_cls, inc = int(d["init_cls"]), int(d["increment"])
        total_cls = init_cls + t * inc
        targets = d["targets"].astype(np.int64)
        raw = d["raw_logits"].astype(np.float32)

        # --- accumulate ridge stats with this task's branch-0 train dump ---
        tr = np.load(os.path.join(args.dump_dir, "train_emb_task{}_branch0.npz".format(t)))
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

        W = np.linalg.solve(G + args.lam * np.eye(OUT_DIM), C).astype(np.float32)
        x0 = l2norm(d["features"].astype(np.float32)[:, :OUT_DIM])
        ridge_logits = x0 @ W

        # --- pos_shift calibrated diagonal logits ---
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
        for scheme in schemes:
            zd = zscore(diag[scheme])
            zr = zscore(ridge_logits)
            for w in args.weights:
                scores = w * zr + (1.0 - w) * zd if w < 1.0 else ridge_logits
                top1 = (scores.argmax(1) == targets).mean() * 100
                acc[(scheme, w)].append(top1)

    print("{:<10} {:>6} {:>8} {:>8}".format("scheme", "w", "avg", "final"))
    for scheme in schemes:
        for w in args.weights:
            curve = acc[(scheme, w)]
            print("{:<10} {:>6.2f} {:>8.2f} {:>8.2f}".format(
                scheme, w, float(np.mean(curve)), curve[-1]))


if __name__ == "__main__":
    main()
