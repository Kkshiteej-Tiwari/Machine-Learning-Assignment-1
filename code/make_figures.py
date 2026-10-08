"""Figures for the report (reads results/*.json written by train.py)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 200,
})


def load(var):
    with open(os.path.join(RES, f"{var}_cv.json")) as f:
        return json.load(f)


def degree_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, var in zip(axes, ("var1", "var2")):
        L = load(var)
        r = L["ridge"]
        d = [x["degree"] for x in r]
        ax.plot(d, [x["train_mse_ols"] for x in r], color=AQUA, marker="o", ms=4,
                label="Train MSE (unregularised)")
        ax.plot(d, [x["cv_mse_ols"] for x in r], color=ORANGE, marker="o", ms=4,
                label="CV MSE (unregularised)")
        ax.plot(d, [x["cv_mse"] for x in r], color=BLUE, marker="o", ms=4,
                label="CV MSE (ridge, tuned)")
        b = L["best"]
        ax.axhline(b["cv_mse"], color=INK2, lw=1, ls="--")
        ax.text(d[-1], b["cv_mse"] * 0.8, f"final model, deg {b['degree']}",
                ha="right", va="top", color=INK2, fontsize=7.5)
        ax.set_yscale("log")
        ax.set_xlabel("Polynomial degree")
        ax.set_title(f"{var}", loc="left", fontweight="bold")
        ax.set_xticks(d if len(d) <= 10 else d[1::2])
        lo = min(min(x["train_mse_ols"] for x in r), b["cv_mse"]) * 0.4
        ax.set_ylim(max(lo, 1e-3), L["var_y"] * 3)
    axes[0].set_ylabel("MSE (log scale)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(RES, "fig_degree_curves.png"))


def parity():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, var in zip(axes, ("var1", "var2")):
        yo = np.load(os.path.join(RES, f"{var}_oof.npy"))
        y, p = yo[:, 0], yo[:, 1]
        b = load(var)["best"]
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.scatter(y, p, s=9, color=BLUE, alpha=0.55, edgecolors="none")
        ax.set_xlabel("Actual y")
        ax.set_title(f"{var}: out-of-fold predictions", loc="left", fontweight="bold")
        ax.text(0.03, 0.97, f"MSE = {b['oof_mse']:.3f}\nR$^2$ = {b['oof_r2']:.4f}",
                transform=ax.transAxes, va="top", color=INK)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted y")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_parity.png"))


def residuals():
    """Out-of-fold residuals (actual - predicted): against the prediction, and their distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), gridspec_kw=dict(width_ratios=(1.35, 1)))
    for row, var in zip(axes, ("var1", "var2")):
        yo = np.load(os.path.join(RES, f"{var}_oof.npy"))
        y, p = yo[:, 0], yo[:, 1]
        r = y - p
        ax = row[0]
        ax.axhline(0, color=INK2, lw=1, ls="--")
        ax.scatter(p, r, s=9, color=BLUE, alpha=0.55, edgecolors="none")
        ax.set_xlabel("Predicted y")
        ax.set_ylabel("Actual − predicted")
        ax.set_title(f"{var}: residuals vs prediction", loc="left", fontweight="bold")
        m = np.abs(r).max() * 1.08
        ax.set_ylim(-m, m)
        ax = row[1]
        ax.hist(r, bins=30, color=BLUE, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Rows")
        ax.set_title(f"{var}: residual distribution", loc="left", fontweight="bold")
        ax.text(0.03, 0.97, f"mean = {r.mean():+.3f}\nsd = {r.std():.3f}",
                transform=ax.transAxes, va="top", color=INK)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_residuals.png"))


if __name__ == "__main__":
    degree_curves()
    parity()
    residuals()
    print("figures written to", RES)
