"""Offline comparison of alternative classification heads (B2/B3) on dumps.

Usage:
    python tools/compare_heads.py logs/.../analysis_<prefix>_<seed> [task_id]

Consumes the dumps written by models/cllora.py when branch_calibration.dump_eval
is on:
  * train_emb_task{t}_branch{i}.npz : task-t train embeddings under branch i
  * eval_task{t}.npz                : test features [N,(t+1)*768], raw logits, targets

Heads compared (all rehearsal-free, using only information available online):
  baseline      raw diagonal cosine logits (what the model does today)
  fecam_diag    per-task shared covariance, Mahalanobis in each class's own branch
  branch0_ncm   cosine NCM for ALL classes in the branch-0 feature space
  branch0_ridge ridge regression (one-hot) on branch-0 features, closed form
  ranpac0       branch-0 features -> fixed random projection + ReLU -> ridge
"""
import glob
import os
import re
import sys

import numpy as np

OUT_DIM = 768


def cls_range(i, init_cls, inc):
    if i == 0:
        return 0, init_cls
    start = init_cls + (i - 1) * inc
    return start, start + inc


def load_dumps(dump_dir, task_id=None):
    eval_files = sorted(
        glob.glob(os.path.join(dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)),
    )
    if not eval_files:
        sys.exit("no eval dumps in {}".format(dump_dir))
    if task_id is None:
        task_id = int(re.search(r"eval_task(\d+)\.npz", eval_files[-1]).group(1))
    data = np.load(os.path.join(dump_dir, "eval_task{}.npz".format(task_id)))
    if "features" not in data:
        sys.exit("eval dump has no features (older dump format)")
    test_feat = data["features"].astype(np.float32)
    test_logits = data["raw_logits"].astype(np.float32)
    targets = data["targets"].astype(np.int64)
    init_cls, inc = int(data["init_cls"]), int(data["increment"])

    train = {}  # (task, branch) -> (emb, labels)
    for path in glob.glob(os.path.join(dump_dir, "train_emb_task*_branch*.npz")):
        m = re.search(r"train_emb_task(\d+)_branch(\d+)\.npz", path)
        t, b = int(m.group(1)), int(m.group(2))
        if t > task_id:
            continue
        d = np.load(path)
        train[(t, b)] = (d["embeddings"].astype(np.float32), d["labels"].astype(np.int64))
    return test_feat, test_logits, targets, train, init_cls, inc, task_id


def l2n(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def accuracy(scores, targets, init_cls, inc):
    pred = scores.argmax(axis=1)
    top1 = (pred == targets).mean() * 100
    task_of = lambda c: np.maximum((c - init_cls) // inc + 1, 0)
    tid = (task_of(pred) == task_of(targets)).mean() * 100
    return top1, tid


def head_fecam_diag(test_feat, targets, train, init_cls, inc, T, shrink=1.0):
    """Class-mean + per-task shared covariance Mahalanobis, each class scored in
    its own task's branch block. More cross-branch comparable than cosine."""
    n_test = test_feat.shape[0]
    n_cls = init_cls + T * inc
    scores = np.full((n_test, n_cls), -np.inf, dtype=np.float32)
    for t in range(T + 1):
        if (t, t) not in train:
            continue
        emb, labels = train[(t, t)]
        emb = l2n(emb)
        x = l2n(test_feat[:, t * OUT_DIM:(t + 1) * OUT_DIM])
        centered = emb - emb.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(emb) - 1, 1)
        cov += shrink * np.trace(cov) / OUT_DIM * np.eye(OUT_DIM, dtype=np.float32)
        prec = np.linalg.inv(cov)
        for c in np.unique(labels):
            mu = emb[labels == c].mean(axis=0)
            diff = x - mu
            scores[:, c] = -np.einsum("nd,dk,nk->n", diff, prec, diff)
    return scores


def head_branch0_ncm(test_feat, train, init_cls, inc, T):
    x0 = l2n(test_feat[:, :OUT_DIM])
    n_cls = init_cls + T * inc
    protos = np.zeros((n_cls, OUT_DIM), dtype=np.float32)
    for t in range(T + 1):
        if (t, 0) not in train:
            continue
        emb, labels = train[(t, 0)]
        for c in np.unique(labels):
            protos[c] = emb[labels == c].mean(axis=0)
    return x0 @ l2n(protos).T


def _gather_branch0_train(train, T):
    xs, ys = [], []
    for t in range(T + 1):
        if (t, 0) in train:
            emb, labels = train[(t, 0)]
            xs.append(emb)
            ys.append(labels)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def head_branch0_ridge(test_feat, train, init_cls, inc, T, lam=1.0, proj=None):
    x_tr, y_tr = _gather_branch0_train(train, T)
    x_te = test_feat[:, :OUT_DIM]
    if proj is not None:
        rng = np.random.RandomState(0)
        P = rng.randn(OUT_DIM, proj).astype(np.float32) / np.sqrt(OUT_DIM)
        x_tr = np.maximum(x_tr @ P, 0)
        x_te = np.maximum(x_te @ P, 0)
    x_tr, x_te = l2n(x_tr), l2n(x_te)
    n_cls = init_cls + T * inc
    onehot = np.eye(n_cls, dtype=np.float32)[y_tr]
    d = x_tr.shape[1]
    G = x_tr.T @ x_tr + lam * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(G, x_tr.T @ onehot)
    return x_te @ W


def main():
    dump_dir = sys.argv[1]
    task_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    test_feat, test_logits, targets, train, init_cls, inc, T = load_dumps(dump_dir, task_id)
    print("evaluating at task {} | test {} samples | train dumps {}".format(
        T, len(targets), len(train)))

    results = {}
    results["baseline"] = accuracy(test_logits, targets, init_cls, inc)
    results["fecam_diag"] = accuracy(
        head_fecam_diag(test_feat, targets, train, init_cls, inc, T), targets, init_cls, inc)
    results["branch0_ncm"] = accuracy(
        head_branch0_ncm(test_feat, train, init_cls, inc, T), targets, init_cls, inc)
    for lam in (1.0, 10.0, 100.0):
        results["branch0_ridge_l{:g}".format(lam)] = accuracy(
            head_branch0_ridge(test_feat, train, init_cls, inc, T, lam=lam), targets, init_cls, inc)
    for m in (2048, 4096):
        results["ranpac0_m{}".format(m)] = accuracy(
            head_branch0_ridge(test_feat, train, init_cls, inc, T, lam=100.0, proj=m), targets, init_cls, inc)

    print("\n=== head comparison at task {} (top1 | task-id) ===".format(T))
    for name, (top1, tid) in results.items():
        print("{:>18s}: {:6.2f} | {:6.2f}".format(name, top1, tid))


if __name__ == "__main__":
    main()
