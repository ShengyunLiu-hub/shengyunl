"""Offline B2 evaluation: EASE-style off-diagonal fc completion + reweighted head.

Rebuilds the prototype fc from dump_eval train embeddings, ports
solve_similarity / solve_sim_reset / forward_reweight from models/cllora.py
and backbone/linears.py verbatim, and sweeps alpha on the eval dumps.

alpha=0 must reproduce the real run's raw diagonal top1 (tool validation).

Usage:
    python tools/eval_b2_reweight.py logs/.../analysis_<prefix>_<seed> \
        [--alphas 0 0.05 0.1 0.2 0.5] [--mode reset|newest]
"""
import argparse
import glob
import os
import re

import numpy as np

OUT_DIM = 768


def l2norm(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def softmax(m, axis=1):
    e = np.exp(m - m.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def cls_range(task_id, init_cls, inc):
    if task_id == 0:
        return 0, init_cls
    s = init_cls + (task_id - 1) * inc
    return s, s + inc


def build_prototypes(dump_dir, t, init_cls, inc, total_cls, cache):
    """fc.weight replica [total_cls, (t+1)*OUT_DIM]; rows of task j filled for
    branches 0..j (exactly what replace_fc has filled by eval time of task t)."""
    W = np.zeros((total_cls, (t + 1) * OUT_DIM), dtype=np.float32)
    for j in range(t + 1):
        for i in range(j + 1):
            key = (j, i)
            if key not in cache:
                d = np.load(os.path.join(
                    dump_dir, "train_emb_task{}_branch{}.npz".format(j, i)))
                emb = d["embeddings"].astype(np.float32)
                lab = d["labels"].astype(np.int64)
                protos = {}
                for c in np.unique(lab):
                    protos[int(c)] = emb[lab == c].mean(0)
                cache[key] = protos
            for c, p in cache[key].items():
                W[c, i * OUT_DIM:(i + 1) * OUT_DIM] = p
    return W


def solve_sim_reset(W, t, init_cls, inc, total_cls):
    """Port of models/cllora.py solve_sim_reset (use_init_ptm=False)."""
    W = W.copy()
    for task_id in range(t):
        s_cls, e_cls = cls_range(task_id, init_cls, inc)
        for dim_id in range(task_id + 1, t + 1):
            s_cls_old = init_cls + (dim_id - 1) * inc
            e_cls_old = total_cls
            sd_old, ed_old = task_id * OUT_DIM, (task_id + 1) * OUT_DIM
            sd, ed = dim_id * OUT_DIM, (dim_id + 1) * OUT_DIM
            A = W[s_cls_old:e_cls_old, sd_old:ed_old]
            B = W[s_cls_old:e_cls_old, sd:ed]
            A_hat = W[s_cls:e_cls, sd_old:ed_old]
            sim = l2norm(A_hat) @ l2norm(A).T
            sim = softmax(sim, axis=1)
            W[s_cls:e_cls, sd:ed] = sim @ B
    return W


def solve_similarity_newest(W, t, init_cls, inc, total_cls, known_cls):
    """Port of models/cllora.py solve_similarity: fill only the newest block."""
    W = W.copy()
    for task_id in range(t):
        s_cls, e_cls = cls_range(task_id, init_cls, inc)
        sd_old, ed_old = task_id * OUT_DIM, (task_id + 1) * OUT_DIM
        A = W[known_cls:, sd_old:ed_old]
        B = W[known_cls:, -OUT_DIM:]
        A_hat = W[s_cls:e_cls, sd_old:ed_old]
        sim = softmax(l2norm(A_hat) @ l2norm(A).T, axis=1)
        W[s_cls:e_cls, -OUT_DIM:] = sim @ B
    return W


def forward_reweight(features, W, t, alpha, init_cls, inc):
    """Port of backbone/linears.py forward_reweight (use_init_ptm=False)."""
    n_blocks = t + 1
    fb = [l2norm(features[:, j * OUT_DIM:(j + 1) * OUT_DIM]) for j in range(n_blocks)]
    outs = []
    for i in range(t + 1):
        s_cls, e_cls = cls_range(i, init_cls, inc)
        out = 0.0
        for j in range(n_blocks):
            wb = l2norm(W[s_cls:e_cls, j * OUT_DIM:(j + 1) * OUT_DIM])
            o = fb[j] @ wb.T
            if j != i:
                o = alpha * o / max(t, 1)
            out = out + o
        outs.append(out)
    return np.concatenate(outs, axis=1)


def zscore(m, eps=1e-8):
    return (m - m.mean()) / (m.std() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--mode", choices=["reset", "newest"], default="reset")
    ap.add_argument("--ridge-weights", type=float, nargs="*", default=[0.0],
                    help="ensemble weights for the branch-0 ridge head "
                         "(0 = reweighted diag only), lambda fixed at 1.0")
    args = ap.parse_args()

    eval_files = sorted(
        glob.glob(os.path.join(args.dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)),
    )
    if not eval_files:
        raise SystemExit("no eval dumps in " + args.dump_dir)

    proto_cache = {}
    acc = {(a, w): [] for a in args.alphas for w in args.ridge_weights}
    tc = {(a, w): [] for a in args.alphas for w in args.ridge_weights}
    raw_check = []
    G = np.zeros((OUT_DIM, OUT_DIM), dtype=np.float64)
    C = None

    for path in eval_files:
        t = int(re.search(r"eval_task(\d+)\.npz", path).group(1))
        d = np.load(path)
        init_cls, inc = int(d["init_cls"]), int(d["increment"])
        total_cls = init_cls + t * inc if init_cls != inc else (t + 1) * inc
        known_cls = total_cls - (inc if t > 0 else init_cls)
        targets = d["targets"].astype(np.int64)
        features = d["features"].astype(np.float32)
        raw = d["raw_logits"].astype(np.float32)
        raw_check.append((raw.argmax(1) == targets).mean() * 100)

        ridge_logits = None
        if any(w > 0 for w in args.ridge_weights):
            tr = np.load(os.path.join(
                args.dump_dir, "train_emb_task{}_branch0.npz".format(t)))
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
            W_r = np.linalg.solve(G + np.eye(OUT_DIM), C).astype(np.float32)
            ridge_logits = l2norm(features[:, :OUT_DIM]) @ W_r

        W0 = build_prototypes(args.dump_dir, t, init_cls, inc, total_cls, proto_cache)
        if t > 0:
            if args.mode == "reset":
                Wf = solve_sim_reset(W0, t, init_cls, inc, total_cls)
            else:
                Wf = solve_similarity_newest(W0, t, init_cls, inc, total_cls, known_cls)
        else:
            Wf = W0

        task_of_cls = np.array(
            [0 if c < init_cls else (c - init_cls) // inc + 1 for c in range(total_cls)])
        for a in args.alphas:
            diag = forward_reweight(features, Wf, t, a, init_cls, inc)
            for w in args.ridge_weights:
                if w > 0 and ridge_logits is not None:
                    scores = w * zscore(ridge_logits) + (1.0 - w) * zscore(diag)
                else:
                    scores = diag
                pred = scores.argmax(1)
                acc[(a, w)].append((pred == targets).mean() * 100)
                tc[(a, w)].append(
                    (task_of_cls[pred] == task_of_cls[targets]).mean() * 100)

    print("raw diag reference: avg {:.2f} final {:.2f}".format(
        float(np.mean(raw_check)), raw_check[-1]))
    print("{:<8} {:<8} {:>8} {:>8} {:>8} {:>8}".format(
        "alpha", "ridge_w", "avg", "final", "TCavg", "TCfin"))
    for a in args.alphas:
        for w in args.ridge_weights:
            k = (a, w)
            print("{:<8.2f} {:<8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
                a, w, float(np.mean(acc[k])), acc[k][-1],
                float(np.mean(tc[k])), tc[k][-1]))


if __name__ == "__main__":
    main()
