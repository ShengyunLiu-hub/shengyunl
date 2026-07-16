"""Routing-error diagnosis figure: error decomposition at the final task.

For each seed dump, replicates the final-step scoring for base (none w0) and
ours (pos_shift w0.15), splits top-1 error into
  routing error      -- predicted class lies in the wrong task's range
  within-task error  -- right task, wrong class
and computes the oracle-routing bound (argmax restricted to the true task's
class slice). Results averaged over seeds -> stacked-bar figure + a cache
(logs/figs/routing_decomp.txt) so the figure can be rebuilt without recompute.

Usage:
    python tools/fig_routing.py            # compute missing seeds + draw
    python tools/fig_routing.py --draw     # draw from cache only
"""
import argparse
import glob
import os
import re

import numpy as np

OUT_DIM = 768
CACHE = "logs/figs/routing_decomp.txt"
ROOTS = {
    "cifar": "logs/cllora/cifar224/0/5/analysis_ms_cifar_*",
    "inr": "logs/cllora/imagenetr/0/5/analysis_ms_inr_*",
    "ina": "logs/cllora/imageneta/0/20/analysis_ms_ina_*",
    "vtab": "logs/cllora/vtab/0/10/analysis_ms_vtab_*",
}
DS_LABEL = {"cifar": "CIFAR-100", "inr": "ImageNet-R", "ina": "ImageNet-A", "vtab": "VTAB"}


def zscore(m, eps=1e-8):
    return (m - m.mean()) / (m.std() + eps)


def l2norm(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def task_of(classes, init_cls, inc):
    return np.where(classes < init_cls, 0,
                    (classes - init_cls) // inc + 1) if init_cls != inc \
        else classes // inc


def process(dump_dir, lam=1.0):
    eval_files = sorted(
        glob.glob(os.path.join(dump_dir, "eval_task*.npz")),
        key=lambda p: int(re.search(r"eval_task(\d+)\.npz", p).group(1)))
    if not eval_files:
        return None
    T = int(re.search(r"eval_task(\d+)\.npz", eval_files[-1]).group(1))

    G = np.zeros((OUT_DIM, OUT_DIM), dtype=np.float64)
    C = None
    d = np.load(eval_files[-1])
    init_cls, inc = int(d["init_cls"]), int(d["increment"])
    total_cls = init_cls + T * inc
    for t in range(T + 1):
        tr = np.load(os.path.join(dump_dir, "train_emb_task{}_branch0.npz".format(t)))
        x = l2norm(tr["embeddings"].astype(np.float32))
        y = tr["labels"].astype(np.int64)
        G += x.T @ x
        if C is None:
            C = np.zeros((OUT_DIM, total_cls), dtype=np.float64)
        onehot = np.zeros((x.shape[0], total_cls), dtype=np.float64)
        onehot[np.arange(x.shape[0]), y] = 1.0
        C += x.astype(np.float64).T @ onehot

    W = np.linalg.solve(G + lam * np.eye(OUT_DIM), C).astype(np.float32)
    targets = d["targets"].astype(np.int64)
    raw = d["raw_logits"].astype(np.float32)
    ridge = l2norm(d["features"].astype(np.float32)[:, :OUT_DIM]) @ W

    calib = raw.copy()
    for b, mu in zip(d["pos_stats_branches"].astype(int), d["pos_stats"][:, 0]):
        if b > T:
            continue
        s = b * inc if init_cls == inc else (0 if b == 0 else init_cls + (b - 1) * inc)
        e = s + (init_cls if (b == 0 and init_cls != inc) else inc)
        calib[:, s:e] = raw[:, s:e] - mu

    true_task = task_of(targets, init_cls, inc)
    res = {}
    for name, scores in [
            ("base", zscore(raw)),
            ("ours", 0.15 * zscore(ridge) + 0.85 * zscore(calib))]:
        pred = scores.argmax(1)
        pred_task = task_of(pred, init_cls, inc)
        route_err = (pred_task != true_task).mean() * 100
        within_err = ((pred_task == true_task) & (pred != targets)).mean() * 100
        res[name] = (route_err, within_err)

    # oracle routing: argmax inside the true task's class slice
    oracle_wrong = 0
    for g in np.unique(true_task):
        s = g * inc if init_cls == inc else (0 if g == 0 else init_cls + (g - 1) * inc)
        e = s + (init_cls if (g == 0 and init_cls != inc) else inc)
        m = true_task == g
        pred_in = raw[m, s:e].argmax(1) + s
        oracle_wrong += (pred_in != targets[m]).sum()
    res["oracle"] = (0.0, oracle_wrong / len(targets) * 100)
    return res


def load_cache():
    rows = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            p = line.split()
            if len(p) == 5:
                rows[(p[0], p[1])] = {"base": (float(p[2].split(",")[0]), float(p[2].split(",")[1])),
                                      "ours": (float(p[3].split(",")[0]), float(p[3].split(",")[1])),
                                      "oracle": (0.0, float(p[4]))}
    return rows


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, VERM, INK, MUTED = "#0072B2", "#D55E00", "#1a1a1a", "#666666"
    plt.rcParams.update({
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, ds in zip(axes, ["cifar", "inr", "ina", "vtab"]):
        seeds = [k for k in rows if k[0] == ds]
        if not seeds:
            ax.set_visible(False)
            continue
        agg = {c: (np.mean([rows[k][c][0] for k in seeds]),
                   np.mean([rows[k][c][1] for k in seeds])) for c in ["base", "ours", "oracle"]}
        x = np.arange(3)
        route = [agg[c][0] for c in ["base", "ours", "oracle"]]
        within = [agg[c][1] for c in ["base", "ours", "oracle"]]
        ax.bar(x, within, width=0.6, color=BLUE, label="within-task error",
               edgecolor="white", linewidth=1.2, zorder=3)
        ax.bar(x, route, width=0.6, bottom=within, color=VERM,
               label="routing error", edgecolor="white", linewidth=1.2, zorder=3)
        for xi, (r, wv) in enumerate(zip(route, within)):
            if r > 0.4:
                ax.text(xi, wv + r / 2, "%.1f" % r, ha="center", va="center",
                        fontsize=6.5, color="white")
            ax.text(xi, wv / 2, "%.1f" % wv, ha="center", va="center",
                    fontsize=6.5, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["base", "ours", "oracle\nrouting"], fontsize=7)
        ax.set_title("%s (n=%d)" % (DS_LABEL[ds], len(seeds)), fontsize=8.5)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    axes[0].set_ylabel("final-task top-1 error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=7.5)
    os.makedirs("logs/figs", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("logs/figs/fig_routing_decomp.%s" % ext, dpi=300,
                    bbox_inches="tight")
    print("wrote logs/figs/fig_routing_decomp.{pdf,png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    rows = load_cache()
    if not args.draw:
        for ds, pat in ROOTS.items():
            for dump in sorted(glob.glob(pat)):
                if dump.endswith(".bak"):
                    continue
                seed = os.path.basename(dump).split("_")[-1]
                if (ds, seed) in rows:
                    continue
                # skip dumps still being written
                newest = max((os.path.getmtime(p) for p in
                              glob.glob(os.path.join(dump, "*.npz"))), default=0)
                import time
                if time.time() - newest < 600:
                    print("skip in-progress:", dump)
                    continue
                print("processing:", dump, flush=True)
                res = process(dump)
                if res is None:
                    continue
                rows[(ds, seed)] = res
                with open(CACHE, "a") as f:
                    f.write("%s %s %.4f,%.4f %.4f,%.4f %.4f\n" % (
                        ds, seed, res["base"][0], res["base"][1],
                        res["ours"][0], res["ours"][1], res["oracle"][1]))
    draw(rows)


if __name__ == "__main__":
    main()
