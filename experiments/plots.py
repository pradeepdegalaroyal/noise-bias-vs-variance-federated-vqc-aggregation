"""Figures: hatched grouped bars, mean +- std line bands and waterfall charts (Tableau-muted palette)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAB = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#9C755F"]
HATCH = ["", "//", "xx", ".."]
LABEL = {"fedavg": "FedAvg", "reliability": "Reliability-weighted", "corrected": "Correction only",
         "fisher": "Fisher weights only", "proposed": "Correction + Fisher weights",
         "oracle": "Ideal devices", "classical": "Classical MLP"}
DPI = 200


def _ms(df: pd.DataFrame, col: str) -> tuple[float, float]:
    return float(df[col].mean()), float(df[col].std(ddof=1)) if len(df) > 1 else 0.0


def fig_shots(runs: pd.DataFrame, out: Path) -> None:
    d = runs[runs.experiment == "shots"]
    shots = sorted(s for s in d.shots.unique() if s != -1) + [-1]
    keys = ["fedavg", "reliability", "corrected", "proposed"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.2
    for j, k in enumerate(keys):
        m = [_ms(d[(d.method_key == k) & (d.shots == s)], "final_mse") for s in shots]
        ax.bar(np.arange(len(shots)) + (j - 1.5) * w, [a for a, _ in m], w, yerr=[b for _, b in m], capsize=3,
               color=TAB[j], edgecolor="black", hatch=HATCH[j], label=LABEL[k], linewidth=0.8)
    for k, ls, c in [("oracle", "--", TAB[4]), ("classical", ":", "black")]:
        if (d.method_key == k).any():
            ax.axhline(d[d.method_key == k].final_mse.mean(), ls=ls, color=c, lw=1.4, label=LABEL[k])
    ax.set_xticks(np.arange(len(shots)))
    ax.set_xticklabels([("exact" if s == -1 else f"{int(s)}") for s in shots])
    ax.set_xlabel("Shots per circuit (exact = infinite shots)")
    ax.set_ylabel("Ideal-device test loss (mean squared error)")
    ax.set_title("Final test loss versus measurement budget (4 noisy clients, 5 seeds)")
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_shots_grouped_bars.png", dpi=DPI)
    plt.close(fig)


def fig_alpha(runs: pd.DataFrame, out: Path) -> None:
    d = runs[(runs.experiment == "alpha")]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for corrected, c, mk, ls, name in [(False, TAB[1], "s", "--", "Uncorrected clients"),
                                       (True, TAB[0], "o", "-", "Attenuation-corrected clients")]:
        sub = d[d.corrected == corrected]
        al = sorted(sub.alpha.unique())
        m = np.array([_ms(sub[sub.alpha == a], "final_mse") for a in al])
        ax.plot(al, m[:, 0], color=c, marker=mk, ls=ls, label=name)
        ax.fill_between(al, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=c, alpha=0.18)
    ax.set_xlabel("Weight exponent alpha  (client weight = data size x calibrated attenuation^alpha)")
    ax.set_ylabel("Ideal-device test loss (mean squared error)")
    ax.set_title("Which weight exponent is best?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_alpha_sweep.png", dpi=DPI)
    plt.close(fig)


def _waterfall(ax: plt.Axes, labels: list[str], values: list[float], title: str) -> None:
    start, steps = values[0], np.diff(values)
    x = np.arange(len(values) + 0)
    ax.bar(0, start, color=TAB[7], edgecolor="black", hatch="")
    run = start
    for i, st in enumerate(steps, 1):
        ax.bar(i, st, bottom=run, color=TAB[4] if st < 0 else TAB[2], edgecolor="black", hatch="//" if st < 0 else "xx")
        ax.text(i, run + st + (0.002 if st >= 0 else -0.002), f"{st:+.3f}", ha="center",
                va="bottom" if st >= 0 else "top", fontsize=8)
        run += st
    ax.bar(len(values), run, color=TAB[0], edgecolor="black")
    ax.text(0, start + 0.002, f"{start:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(len(values), run + 0.002, f"{run:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x) + [len(values)])
    ax.set_xticklabels(labels + ["Result"], fontsize=8)
    lo = min(values) - 0.06
    ax.set_ylim(lo, max(values) + 0.04)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def fig_waterfall(runs: pd.DataFrame, base_shots: int, out: Path) -> None:
    d = runs[(runs.experiment == "shots") & (runs.shots == base_shots)]
    mse = {k: d[d.method_key == k].final_mse.mean() for k in ("fedavg", "corrected", "fisher", "proposed")}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    _waterfall(axes[0], ["FedAvg", "+ attenuation\ncorrection", "+ Fisher\nweights"],
               [mse["fedavg"], mse["corrected"], mse["proposed"]], "Path A: correct first, then weight")
    _waterfall(axes[1], ["FedAvg", "+ Fisher\nweights", "+ attenuation\ncorrection"],
               [mse["fedavg"], mse["fisher"], mse["proposed"]], "Path B: weight first, then correct")
    axes[0].set_ylabel("Ideal-device test loss (MSE; axis truncated)")
    fig.suptitle(f"Ablation: contribution of each component ({base_shots} shots per circuit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig3_ablation_waterfall.png", dpi=DPI)
    plt.close(fig)


def fig_curves(curves: pd.DataFrame, base_shots: int, out: Path) -> None:
    c = curves.copy()
    c["shots"] = c["shots"].fillna(-1)
    c = c[(c.experiment == "shots") & ((c.shots == base_shots) | c.method_key.isin(["oracle", "classical"]))]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {"fedavg": (TAB[1], "--", "s"), "reliability": (TAB[5], "-.", "^"), "proposed": (TAB[0], "-", "o"),
              "oracle": (TAB[4], ":", "D"), "classical": ("black", (0, (5, 1)), "x")}
    for k, (col, ls, mk) in styles.items():
        g = c[c.method_key == k].groupby("round").test_mse.agg(["mean", "std"]).fillna(0)
        ax.plot(g.index, g["mean"], color=col, ls=ls, marker=mk, ms=4, label=LABEL[k])
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Ideal-device test loss (mean squared error)")
    ax.set_title(f"Learning curves ({base_shots} shots per circuit)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_learning_curves.png", dpi=DPI)
    plt.close(fig)


def _band(ax: plt.Axes, d: pd.DataFrame, col: str, keys: list[str], styles: dict) -> None:
    xs = sorted(d[col].unique())
    for k in keys:
        colr, ls, mk = styles[k]
        m = np.array([_ms(d[(d.method_key == k) & (d[col] == x)], "final_mse") for x in xs])
        ax.plot(xs, m[:, 0], color=colr, ls=ls, marker=mk, label=LABEL[k])
        ax.fill_between(xs, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=colr, alpha=0.15)
    ax.grid(alpha=0.3)
    ax.set_ylabel("Ideal-device test loss (MSE)")


def fig_robustness(runs: pd.DataFrame, out: Path) -> None:
    st = {"fedavg": (TAB[1], "--", "s"), "reliability": (TAB[5], "-.", "^"), "corrected": (TAB[4], "-", "D"),
          "proposed": (TAB[0], ":", "o")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    _band(axes[0], runs[runs.experiment == "sigma"], "sigma", ["fedavg", "reliability", "corrected", "proposed"], st)
    axes[0].set_xlabel("Spread of client error rates (std of ln p)")
    axes[0].set_title("(a) Noise heterogeneity across clients", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5)
    e = runs[runs.experiment == "calerror"]
    _band(axes[1], e, "cal_error", ["corrected", "proposed"], st)
    ref = runs[(runs.experiment == "shots") & (runs.method_key == "fedavg") & (runs.shots == e.shots.iloc[0])]
    mu, sd = _ms(ref, "final_mse")
    axes[1].axhline(mu, color=TAB[1], ls="--", label=LABEL["fedavg"])
    axes[1].axhspan(mu - sd, mu + sd, color=TAB[1], alpha=0.10)
    axes[1].set_xlabel("Injected calibration error (std of ln attenuation)")
    axes[1].set_title("(b) Error in the calibrated attenuation", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5)
    _band(axes[2], runs[runs.experiment == "beta"], "beta", ["fedavg", "corrected", "proposed"], st)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Dirichlet label-split concentration (small = skewed)")
    axes[2].set_title("(c) Label skew across clients", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig5_robustness.png", dpi=DPI)
    plt.close(fig)


def make_all(runs: pd.DataFrame, curves: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    """Write all figures into ``out``."""
    base_shots = int(runs[runs.experiment == "calerror"].shots.iloc[0])
    fig_shots(runs, out)
    fig_alpha(runs, out)
    fig_waterfall(runs, base_shots, out)
    fig_curves(curves, base_shots, out)
    fig_robustness(runs, out)
