"""Single-column (2.9 in) figures for the research paper, drawn from the real result files.

Run from the project root:  python -m scripts.paper_figures
Writes docs/paper/figures/fig1..fig7.png. Figs 4-7 read outputs/results and the production
checkpoints; Figs 1-3 are diagrams of the code structure (src/, training/, app/).
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join("docs", "paper", "figures")
W = 2.9
DPI = 450
BLUE, LBLUE, ORANGE, LORANGE, GREY, GREEN = "#2b5d8a", "#dce9f5", "#c4621a", "#fbe6d3", "#6b7280", "#2f7d4f"
plt.rcParams.update({"font.size": 6.2, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})


def canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / W)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=6,
                                 lw=0.8, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


def fig1():
    h = 3.05
    fig, ax = canvas(h)
    top = 100 * h / W
    items = [
        (r"Input window $X\in\mathbb{R}^{10\times5}$" + "\n4 risk indices + Total Cost", LBLUE, BLUE),
        ("Per step: node features $[RI_i,\\ Cost]$ on the chain\nSupplier $\\to$ Manufacturer $\\to$ Distributor $\\to$ Retailer", LBLUE, BLUE),
        ("Relational GCN $\\times$2 (32 ch): self + upstream\n+ downstream messages, ReLU; + echelon embedding", LBLUE, BLUE),
        (r"Node sequences: 4 nodes $\times$ 10 steps $\times$ 32", LBLUE, BLUE),
        ("Shared 2-layer LSTM per node (hidden 64, dropout 0.2)", LBLUE, BLUE),
        (r"Head 64$\to$32$\to$1 per node:  $\Delta\hat{y}\in\mathbb{R}^{4}$", LBLUE, BLUE),
        (r"Residual skip: $\hat{y}_{t+5}=y_t+\Delta\hat{y}$", LORANGE, ORANGE),
    ]
    bh, gap, x0, bw = 8.2, 3.4, 4, 92
    y = top - 3 - bh
    ys = []
    for t, fc, ec in items:
        box(ax, x0, y, bw, bh, t, fc, ec)
        ys.append(y)
        y -= bh + gap
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, 50, a - 0.2, 50, b + bh + 0.2)
    # IG module
    yig = ys[-1] - bh - 5
    box(ax, x0, yig, bw, bh + 3, "Integrated Gradients (post hoc): 64 steps,\ntrain-mean baseline, scalar target node $k$\n"
        r"$\Rightarrow$ $Attr\in\mathbb{R}^{10\times5}$", "#e8f3ea", GREEN)
    arrow(ax, 50, ys[-1] - 0.2, 50, yig + bh + 3.2, color=GREEN, ls="--")
    fig.savefig(os.path.join(OUT, "fig1.png"), dpi=DPI)
    plt.close(fig)


def fig2():
    h = 2.85
    fig, ax = canvas(h)
    top = 100 * h / W
    items = [
        ("Raw SCRM CSV (Mendeley): 649,999 rows", LBLUE, BLUE),
        ("Clean: parse + sort, drop duplicate stamps,\nforward-fill $\\leq$5 steps, split at gaps > 6 min", LBLUE, BLUE),
        ("Chronological 80/10/10 split; MinMax scaler\nfitted on train only; windows $L$=10, $H$=5", LBLUE, BLUE),
        ("Train 4 models $\\times$ 5 seeds (Adam, early stop);\nSHA-256 manifest of all checkpoints", LBLUE, BLUE),
        ("Evaluate on test vs Persistence, Ridge-AR(10),\nLSTM: MSE, R$^2$, 3-tier severity", LORANGE, ORANGE),
        ("Explain: Integrated Gradients\n(completeness gap, deletion test)", "#e8f3ea", GREEN),
        ("Streamlit dashboard: Overview, Monitor, Why,\nBenchmarks, Export, Sandbox", LORANGE, ORANGE),
    ]
    bh, gap, x0, bw = 8.4, 3.3, 4, 92
    y = top - 3 - bh
    ys = []
    for t, fc, ec in items:
        box(ax, x0, y, bw, bh, t, fc, ec)
        ys.append(y)
        y -= bh + gap
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, 50, a - 0.2, 50, b + bh + 0.2)
    fig.savefig(os.path.join(OUT, "fig2.png"), dpi=DPI)
    plt.close(fig)


def fig3():
    h = 3.0
    fig, ax = canvas(h)
    top = 100 * h / W
    r = lambda k: top - 3 - 8 - k * 17.2
    box(ax, 33, r(0), 34, 8, "src/config.py\nConfig", "#f3f4f6", GREY)
    box(ax, 2, r(1), 29, 8, "src/dataset.py\nbuild_datasets", LBLUE, BLUE)
    box(ax, 34, r(1), 31, 8, "src/graph_builder.py\nbuild_graphs", LBLUE, BLUE)
    box(ax, 68, r(1), 30, 8, "graph_layers.py\nGraphConv", LBLUE, BLUE)
    box(ax, 3, r(2), 94, 8, "src/models/st_gcn_lstm.py\nSTGCNLSTM | LSTMBaseline | PaperHybridOverall", LBLUE, BLUE, fs=5.6)
    box(ax, 2, r(3), 27, 8, "training/train.py\ntrain_one", LORANGE, ORANGE)
    box(ax, 32, r(3), 31, 8, "training/evaluate.py\nmain", LORANGE, ORANGE)
    box(ax, 66, r(3), 32, 8, "src/explainability.py\nRiskExplainer", "#e8f3ea", GREEN, fs=5.8)
    box(ax, 3, r(4), 94, 8, "outputs/: models (.pt, scaler, SHA-256), results (.csv), figures", "#f3f4f6", GREY, fs=5.4)
    box(ax, 8, r(5), 84, 8, "app/ (Streamlit): views + utils", LORANGE, ORANGE, fs=5.8)
    for sx, tx, a, b in [(16, 16, 1, 3), (49, 49, 1, 2), (83, 60, 1, 2)]:
        pass
    arrow(ax, 49, r(0) - 0.2, 16, r(1) + 8.2)                     # config -> dataset
    arrow(ax, 49, r(1) - 0.2, 40, r(2) + 8.2)                     # graph_builder -> model
    arrow(ax, 83, r(1) - 0.2, 66, r(2) + 8.2)                     # graph_layers -> model
    arrow(ax, 30, r(2) - 0.2, 16, r(3) + 8.2)                     # model -> train
    arrow(ax, 50, r(2) - 0.2, 47, r(3) + 8.2)                     # model -> evaluate
    arrow(ax, 70, r(2) - 0.2, 80, r(3) + 8.2)                     # model -> explainability
    arrow(ax, 16, r(1) - 0.2, 9, r(3) + 8.2, ls="--")             # dataset -> train
    arrow(ax, 16, r(3) - 0.2, 22, r(4) + 8.2)
    arrow(ax, 47, r(3) - 0.2, 50, r(4) + 8.2)
    arrow(ax, 82, r(3) - 0.2, 78, r(4) + 8.2)
    arrow(ax, 50, r(4) - 0.2, 50, r(5) + 8.2)
    fig.savefig(os.path.join(OUT, "fig3.png"), dpi=DPI)
    plt.close(fig)


def fig4():
    nd = pd.read_csv("outputs/results/node_metrics.csv")
    nodes = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(4)
    wd = 0.26
    for j, (m, lab, col) in enumerate([("persistence", "Persistence", GREY), ("lstm", "LSTM", "#7aa6c9"),
                                        ("st_gcn_lstm_dir", "ST-GCN directed", ORANGE)]):
        d = nd[nd.model == m].set_index("node").reindex(nodes)
        err = d["MSE_std"].fillna(0).values * 1e4
        ax.bar(x + (j - 1) * wd, d["MSE_mean"].values * 1e4, wd, yerr=err, label=lab, color=col,
               error_kw={"lw": 0.5, "capsize": 1})
    d = nd[nd.model == "st_gcn_lstm_dir"].set_index("node").reindex(nodes)
    for i, s in enumerate(d["skill_score_mean"].values):
        ax.text(i + wd, d["MSE_mean"].values[i] * 1e4 * 1.25, f"{s * 100:.0f}%", ha="center", fontsize=5.4, color=ORANGE)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Supplier", "Manuf.", "Distrib.", "Retailer"])
    ax.set_ylabel(r"Test MSE ($\times10^{-4}$, log)")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig4.png"), dpi=DPI)
    plt.close(fig)


def fig5():
    from src.config import Config
    from src.dataset import build_datasets
    from src.models.st_gcn_lstm import build_model
    cfg = Config()
    tr, va, te, sc, info = build_datasets(cfg, save_scaler=False)
    Y, X = te.node_targets.numpy(), te.sequences.numpy()
    pers = X[:, -1, :4]
    mean_pred = {}
    for name in ("lstm", "st_gcn_lstm_dir"):
        ps = []
        for s in cfg.SEEDS:
            m = build_model(name, cfg)
            m.load_state_dict(torch.load(f"outputs/models/{name}_seed{s}.pt", map_location="cpu", weights_only=True))
            with torch.no_grad():
                ps.append(m.eval()(te.sequences).numpy())
        mean_pred[name] = np.mean(ps, axis=0)
    # window around the largest Retailer shock in the test set
    i0 = int(np.argmax(np.abs(Y[:, 3] - pers[:, 3])))
    sl = slice(max(i0 - 40, 0), i0 + 40)
    rng = sc.data_range_[:4]
    fig, axes = plt.subplots(2, 1, figsize=(W, 2.5), sharex=True)
    for ax, k, nm in zip(axes, (3, 2), ("Retailer", "Distributor")):
        t = np.arange(sl.stop - sl.start)
        ax.plot(t, Y[sl, k], color="black", lw=0.9, label="Truth")
        ax.plot(t, pers[sl, k], color=GREY, lw=0.7, ls="--", label="Persistence")
        ax.plot(t, mean_pred["lstm"][sl, k], color="#7aa6c9", lw=0.7, label="LSTM")
        ax.plot(t, mean_pred["st_gcn_lstm_dir"][sl, k], color=ORANGE, lw=0.9, label="ST-GCN directed")
        ax.set_ylabel(f"{nm}\n(scaled RI)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=5.2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28),
                   columnspacing=0.8, handlelength=1.2)
    axes[1].set_xlabel("Consecutive test windows (largest Retailer shock)")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig5.png"), dpi=DPI)
    plt.close(fig)


def fig6():
    cm = pd.read_csv("outputs/results/confusion_matrix.csv", index_col=0).values
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)", ha="center", va="center", fontsize=5.8,
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    lab = ["Low", "Medium", "High"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab)
    ax.set_yticks(range(3))
    ax.set_yticklabels(lab)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig6.png"), dpi=DPI)
    plt.close(fig)


def fig7():
    ig = pd.read_csv(os.path.join("docs", "paper", "evidence", "ig_audit_weighted.csv"))
    feats = ["S", "M", "D", "R", "Cost"]
    tg = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75), sharey=True)
    for ax, t in zip(axes, tg):
        for j, (delta, col, lab) in enumerate([(False, BLUE, "Forecast"), (True, ORANGE, r"Correction $\Delta\hat{y}$")]):
            g = ig[(ig.target == t) & (ig.delta == delta)][feats].sum()
            ax.bar(np.arange(5) + (j - 0.5) * 0.4, (g / g.sum()).values * 100, 0.4, color=col, label=lab)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["S", "M", "D", "R", "C"], fontsize=5.2)
        ax.set_title({"Supplier": "Supplier", "Manufacturer": "Manuf.", "Distributor": "Distrib.",
                      "Retailer": "Retailer"}[t] + "\n(target)", fontsize=5.6, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=1.5, labelsize=5.2)
    axes[0].set_ylabel("Share of |attribution| (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.4, loc="lower center", ncol=2, handlelength=1.2)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.82, bottom=0.30, wspace=0.12)
    fig.savefig(os.path.join(OUT, "fig7.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
        print("wrote", f.__name__)
