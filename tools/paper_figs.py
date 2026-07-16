"""Draft paper figures from the offline sweep outputs.

Outputs logs/figs/fig_gain_vs_tasks.{pdf,png}, fig_paired_slopes.{pdf,png},
fig_w_sensitivity.{pdf,png}. Rerun anytime; picks up whatever sweeps exist.

Colors: Okabe-Ito subset, CVD-validated. All series direct-labeled.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#666666"
SWEEP_DIR = "logs/multiseed/sweeps"
GRID_DIR = "logs/multiseed/sweeps_ablation"
OUT = "logs/figs"

DS_LABEL = {"cifar": "CIFAR-100", "inr": "ImageNet-R", "ina": "ImageNet-A", "vtab": "VTAB"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def parse_sweep(path):
    rows = {}
    for line in open(path):
        m = re.match(r"(none|pos_shift)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows[(m.group(1), float(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return rows


def load_pairs(metric_idx=0, datasets=r"cifar|inr|ina|vtab"):
    """{ds: {seed: (base, gfinal)}} from the main-table sweeps."""
    data = {}
    for f in sorted(glob.glob(os.path.join(SWEEP_DIR, "sweep_analysis_ms_*.txt"))):
        m = re.match(r"sweep_analysis_ms_(%s)_(\d+)\.txt" % datasets,
                     os.path.basename(f))
        if not m:
            continue
        rows = parse_sweep(f)
        if ("none", 0.0) not in rows or ("pos_shift", 0.15) not in rows:
            continue
        data.setdefault(m.group(1), {})[int(m.group(2))] = (
            rows[("none", 0.0)][metric_idx], rows[("pos_shift", 0.15)][metric_idx])
    return data


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{pdf,png}"))


# ---------------------------------------------------------------- Fig 1
def fig_gain_vs_tasks():
    """Paired avg-acc gain grows with task count (routing evidence)."""
    pairs = load_pairs(0, datasets=r"cifar10t|inr10t|cifar|inr")
    cells = [
        ("CIFAR-100", [("T=10", "cifar10t"), ("T=20", "cifar")]),
        ("ImageNet-R", [("T=10", "inr10t"), ("T=40", "inr")]),
    ]
    any_single = False
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    for ax, (title, bars) in zip(axes, cells):
        vals, errs, hatches = [], [], []
        for lbl, ds in bars:
            gains = np.array([g - b for b, g in pairs[ds].values()])
            vals.append(gains.mean())
            errs.append(gains.std(ddof=1) if len(gains) > 1 else 0)
            hatches.append(None if len(gains) > 1 else "////")
            any_single = any_single or len(gains) == 1
        x = np.arange(len(bars))
        bp = ax.bar(x, vals, width=0.55, color=VERM, zorder=3)
        for patch, h in zip(bp, hatches):
            if h:
                patch.set_hatch(h); patch.set_fill(False)
                patch.set_edgecolor(VERM); patch.set_linewidth(0.8)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2, zorder=4)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.12, "%+.2f" % v,
                    ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars])
        ax.set_title(title, fontsize=8.5)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"paired gain in $\bar{A}$ (pp)")
    if any_single:
        fig.text(0.5, -0.06, "hatched = single seed (5-seed runs in progress)",
                 ha="center", fontsize=6.5, color=MUTED)
    save(fig, "fig_gain_vs_tasks")


# ---------------------------------------------------------------- Fig 2
def fig_paired_slopes():
    """Every seed improves: base -> gfinal slope lines, 4 panels."""
    pairs = load_pairs(0)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, ds in zip(axes, ["cifar", "inr", "ina", "vtab"]):
        d = pairs[ds]
        for b, g in d.values():
            ax.plot([0, 1], [b, g], color=MUTED, lw=0.7, alpha=0.55, zorder=2)
            ax.scatter([0], [b], s=9, color=BLUE, zorder=3)
            ax.scatter([1], [g], s=9, color=VERM, zorder=3)
        base = np.array([v[0] for v in d.values()])
        gf = np.array([v[1] for v in d.values()])
        ax.plot([0, 1], [base.mean(), gf.mean()], color=INK, lw=1.6, zorder=4)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "ours"])
        ax.set_title("%s (n=%d)" % (DS_LABEL[ds], len(d)), fontsize=8.5)
        ax.text(0.04, 0.97, "%+.2f" % (gf.mean() - base.mean()),
                transform=ax.transAxes, fontsize=7.5, color=VERM, va="top")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$\bar{A}$ (%)")
    save(fig, "fig_paired_slopes")


# ---------------------------------------------------------------- Fig 3
def fig_w_sensitivity():
    """Avg accuracy vs ensemble weight w, mean across seeds, both schemes."""
    grids = {}
    for f in sorted(glob.glob(os.path.join(GRID_DIR, "grid_ms_*.txt"))):
        m = re.match(r"grid_ms_(cifar|inr|ina|vtab)_(\d+)\.txt", os.path.basename(f))
        if not m:
            continue
        grids.setdefault(m.group(1), []).append(parse_sweep(f))
    fig, axes = plt.subplots(1, len(grids), figsize=(7.0, 1.9))
    if len(grids) == 1:
        axes = [axes]
    for ax, ds in zip(axes, [d for d in ["cifar", "inr", "ina", "vtab"] if d in grids]):
        runs = grids[ds]
        ws = sorted({w for r in runs for (s, w) in r if s == "none"})
        for scheme, color, lbl in [("none", BLUE, "raw head"),
                                   ("pos_shift", VERM, "calibrated")]:
            mean = [np.mean([r[(scheme, w)][0] for r in runs if (scheme, w) in r])
                    for w in ws]
            ax.plot(ws, mean, color=color, lw=1.4, marker="o", ms=2.5,
                    zorder=3, label=lbl)
        ax.axvline(0.15, color=GREEN, lw=0.8, ls="--", zorder=2)
        ax.set_title("%s (n=%d)" % (DS_LABEL[ds], len(runs)), fontsize=8.5)
        ax.set_xlabel("ensemble weight $w$")
        ax.grid(color="#e6e6e6", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$\bar{A}$ (%)")
    axes[0].annotate("$w=0.15$ (used)", (0.15, axes[0].get_ylim()[0]),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=6.5, color=GREEN)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=7.5)
    save(fig, "fig_w_sensitivity")


if __name__ == "__main__":
    fig_gain_vs_tasks()
    fig_paired_slopes()
    fig_w_sensitivity()
