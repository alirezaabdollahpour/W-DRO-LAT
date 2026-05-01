"""Read a Monge-gap CSV produced by ``monge_gap_sweep.py`` and produce two
artifacts:

  1. ``figs/monge_gap_vs_test_loss{suffix}.pdf`` (and ``.png``)
     Scatter plot, x-axis = Monge gap, y-axis = test loss under shift.
     One marker per (method, seed); markers colored by method. Each method
     gets a per-method linear regression line for visual reference.

  2. ``monge_gap_summary{suffix}.csv``
     Per-method aggregated stats: median Monge gap and IQR, median test
     loss and IQR, Spearman rank correlation between gap and loss, and the
     number of points.

Usage:
    python monge_gap_plot.py --csv monge_gap_ls.csv --suffix _ls
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHOD_COLORS = {
    "erm":             "#7f7f7f",   # neutral grey: ERM is the no-adversary reference
    "particle_ascent": "#ff7f0e",
    "wfr":             "#2ca02c",
    "dual":            "#9467bd",
    "nn_dro":          "#8c564b",
    "ppa":             "#1f77b4",
    "npf":             "#d62728",   # ICNN-DRO, our headline method
    # RL aliases
    "nominal":         "#7f7f7f",
    "particle":        "#ff7f0e",
}
METHOD_LABELS = {
    "erm":             "ERM",
    "particle_ascent": "PA",
    "wfr":             "WFR",
    "dual":            "SDRO",
    "nn_dro":          "NN-DRO",
    "ppa":             "MPA",
    "npf":             "ICNN-DRO",   # paper display name for the `npf` codebase key
    "nominal":         "ERM",
    "particle":        "PA",
}
# Plot order matches the table in the paper: baselines first, our methods
# last so they sit on top of the legend stack.
METHOD_ORDER = [
    "erm", "nominal",                                    # ERM-equivalents
    "particle_ascent", "particle", "wfr", "dual", "nn_dro",  # baselines
    "ppa", "npf",                                        # ours
]


def _gap_column(df: pd.DataFrame) -> str:
    """Pick the gap column to use.  Hungarian backend writes ``monge_gap``;
    Sinkhorn writes ``monge_gap_debias`` and ``monge_gap_raw``."""
    for col in ("monge_gap", "monge_gap_debias", "monge_gap_raw"):
        if col in df.columns:
            return col
    raise ValueError(
        "CSV has no Monge-gap column. Expected one of "
        "{'monge_gap', 'monge_gap_debias', 'monge_gap_raw'}."
    )


def _ordered_methods(df: pd.DataFrame):
    """Iterate methods in the canonical paper order, keeping only those
    that actually appear in the CSV."""
    seen = set()
    for m in METHOD_ORDER:
        if m in df["method"].unique() and m not in seen:
            seen.add(m)
            yield m
    # Tail: any method present in the CSV that wasn't in METHOD_ORDER.
    for m in df["method"].unique():
        if m not in seen:
            seen.add(m)
            yield m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("monge_gap.csv"))
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to the output filenames "
                             "(e.g. ``_ls``).")
    parser.add_argument("--figs_dir", type=Path, default=Path("figs"))
    parser.add_argument("--exclude_methods", nargs="+", default=[],
                        help="Codebase keys to drop from the plot and summary "
                             "(e.g. ``--exclude_methods erm``).")
    parser.add_argument("--pareto", action="store_true",
                        help="Pareto-front mode: aggregate seeds per "
                             "(method, lam), connect points by lam to trace "
                             "the gap/loss trade-off curve, and overlay the "
                             "global lower-left envelope.")
    parser.add_argument("--annotate_lam", action="store_true",
                        help="In --pareto mode, annotate each marker with "
                             "its lambda value.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.exclude_methods:
        df = df[~df["method"].isin(args.exclude_methods)].reset_index(drop=True)
        print(f"Excluded methods: {args.exclude_methods}; "
              f"{len(df)} rows remain over {sorted(df['method'].unique())}.")

    # Sanity filter: ``test_loss`` is ``100 - PGD_accuracy_pct`` for
    # lr_cifar10 (must be in [0, 100]) and ``max_steps - worst_return`` for
    # rl_cartpole (still a small number). Anything outside [-1e3, 1e3] is a
    # write-side bug — drop with a warning rather than silently rescaling
    # the y-axis.
    bad = (df["test_loss"] < -1e3) | (df["test_loss"] > 1e3)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} row(s) with out-of-range "
              f"test_loss:\n{df[bad][['method','lam','seed','test_loss']]}")
        df = df[~bad].reset_index(drop=True)

    gap_col = _gap_column(df)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)

    if args.pareto:
        # One point per (method, lam): median across seeds.
        agg = (df.groupby(["method", "lam"], as_index=False)
                 .agg({gap_col: "median", "test_loss": "median"}))

        all_pts = []  # collected for the global Pareto frontier
        for method in _ordered_methods(agg):
            sub = agg[agg["method"] == method].sort_values("lam")
            if len(sub) == 0:
                continue
            color = METHOD_COLORS.get(method, "gray")
            label = METHOD_LABELS.get(method, method)
            xs = sub[gap_col].to_numpy()
            ys = sub["test_loss"].to_numpy()
            lams = sub["lam"].to_numpy()
            marker = "*" if method in ("erm", "nominal") else "o"
            ms = 12 if marker == "*" else 6
            ax.plot(xs, ys, color=color, linewidth=1.4, marker=marker,
                    markersize=ms, label=label, alpha=0.9, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.4)
            if args.annotate_lam:
                for x, y, lam in zip(xs, ys, lams):
                    ax.annotate(rf"$\lambda$={lam:g}", xy=(x, y),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=7, color=color, alpha=0.85)
            all_pts.extend(zip(xs.tolist(), ys.tolist()))

        # Global Pareto frontier: lower-left envelope (smaller gap AND smaller
        # loss is better). Sort by gap ascending, keep points whose loss is
        # strictly below the running minimum.
        all_pts.sort(key=lambda p: (p[0], p[1]))
        front = []
        best_y = float("inf")
        for x, y in all_pts:
            if y < best_y:
                front.append((x, y))
                best_y = y
        if len(front) >= 2:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color="black", linestyle="--", linewidth=1.2,
                    alpha=0.6, label="Pareto frontier", zorder=2)

        ax.text(0.02, 0.02, "← lower-left = better",
                transform=ax.transAxes, fontsize=8, alpha=0.6)
    else:
        for method in _ordered_methods(df):
            sub = df[df["method"] == method]
            if len(sub) == 0:
                continue
            color = METHOD_COLORS.get(method, "gray")
            label = METHOD_LABELS.get(method, method)
            # ERM has gap = 0 by construction; plot as a star marker.
            if method in ("erm", "nominal"):
                ax.scatter(sub[gap_col], sub["test_loss"], color=color, alpha=0.8,
                           label=label, s=80, marker="*", zorder=3,
                           edgecolors="black", linewidths=0.5)
                continue
            ax.scatter(sub[gap_col], sub["test_loss"], color=color, alpha=0.7,
                       label=label, s=30)
            # Per-method regression line (only if >= 3 points and gap has spread).
            if len(sub) >= 3 and sub[gap_col].std() > 0:
                xs = np.linspace(sub[gap_col].min(), sub[gap_col].max(), 50)
                slope, intercept = np.polyfit(sub[gap_col], sub["test_loss"], 1)
                ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.2, alpha=0.5)

    ax.set_xlabel(r"Monge gap  $\mathcal{M}^{\ell_2^2}_{\widehat{\mathbb{P}}}(T)$")
    ax.set_ylabel(r"Test loss under shift")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    args.figs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.figs_dir / f"monge_gap_vs_test_loss{args.suffix}.pdf"
    png_path = args.figs_dir / f"monge_gap_vs_test_loss{args.suffix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    print(f"Wrote {pdf_path} and {png_path}.")

    # Per-method summary table, in the same row order as the legend.
    summary = []
    for method in _ordered_methods(df):
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        if len(sub) >= 3 and sub[gap_col].std() > 0:
            rho, pval = spearmanr(sub[gap_col], sub["test_loss"])
        else:
            rho, pval = float("nan"), float("nan")
        summary.append({
            "method":            METHOD_LABELS.get(method, method),
            "code_key":          method,
            "n":                 len(sub),
            "monge_gap_median":  sub[gap_col].median(),
            "monge_gap_iqr":     sub[gap_col].quantile(0.75) - sub[gap_col].quantile(0.25),
            "test_loss_median":  sub["test_loss"].median(),
            "test_loss_iqr":     sub["test_loss"].quantile(0.75) - sub["test_loss"].quantile(0.25),
            "spearman_rho":      rho,
            "spearman_p":        pval,
        })
    out_csv = Path(f"monge_gap_summary{args.suffix}.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}.")


if __name__ == "__main__":
    main()
