"""Read pareto_runs.csv and produce two figures:

  - figs/pareto_train.pdf: W_2^2 vs adv_loss (training-side Pareto front).
  - figs/pareto_test_delta_{D}.pdf: W_2^2 vs test_loss_delta_{D} for each
    delta in the data file (generalization-side fronts).

Run after pareto_sweep.py. Uses median across seeds with shaded
[25th, 75th] percentile band. Visual style follows the paper's
test-error-vs-perturbation figures: 2-row legend on top, no markers,
distinct colors per method, thin dotted grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Per-method visual style. Colors and line styles taken from the paper's
# Test-error-vs-perturbation figure so the two plots share a visual language.
STYLE = {
    "particle_ascent": dict(label="PA",       color="#3aa2a8", linestyle=(0, (5, 2)),       linewidth=1.7),
    "ppa":             dict(label="MPA",      color="#7f4caf", linestyle=(0, (3, 1, 1, 1)), linewidth=1.7),
    "wfr":             dict(label="WFR",      color="#e07a1c", linestyle=(0, (1, 1)),       linewidth=1.7),
    "dual":            dict(label="SDRO",     color="#d99c2b", linestyle=(0, (5, 1, 1, 1)), linewidth=1.7),
    "npf":             dict(label="ICNN-DRO", color="#c8202b", linestyle="-",                linewidth=2.4),
}

# Plot the curves in this order so ICNN-DRO sits on top.
PLOT_ORDER = ["particle_ascent", "ppa", "wfr", "dual", "npf"]


def aggregate(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Per (method, lam) compute median, 25%, 75% over seeds."""
    g = df.groupby(["method", "lam"])[[x, y]]
    out = g.agg(
        x_median=(x, "median"), x_q25=(x, lambda s: s.quantile(0.25)),
        x_q75=(x, lambda s: s.quantile(0.75)),
        y_median=(y, "median"), y_q25=(y, lambda s: s.quantile(0.25)),
        y_q75=(y, lambda s: s.quantile(0.75)),
    ).reset_index()
    return out.sort_values(["method", "x_median"])


def make_pareto_plot(df_agg: pd.DataFrame, x_col: str, y_col: str,
                     x_label: str, y_label: str, out_path: Path,
                     log_x: bool = True, log_y: bool = False):
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)

    # Plot in a fixed order so the legend and z-order are deterministic.
    for method in PLOT_ORDER:
        sub = df_agg[df_agg["method"] == method]
        if sub.empty:
            continue
        style = STYLE.get(method, dict(label=method))
        color = style.get("color", None)
        ax.plot(sub["x_median"], sub["y_median"], **style)
        ax.fill_between(sub["x_median"], sub["y_q25"], sub["y_q75"],
                        color=color, alpha=0.13, linewidth=0)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55, color="#7a7a7a")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # 2-row legend pinned above the axes, mirroring the paper template.
    n_curves = sum(1 for m in PLOT_ORDER if not df_agg[df_agg["method"] == m].empty)
    ncol = max(1, (n_curves + 1) // 2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, frameon=False, handlelength=2.6,
              columnspacing=1.4, handletextpad=0.6, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("pareto_runs.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("figs"))
    parser.add_argument("--w2_col", choices=["w2_paired", "w2_optimal"],
                        default="w2_optimal",
                        help="Which W_2^2 estimator to plot on the x-axis.")
    parser.add_argument("--drop_lams", nargs="*", type=float, default=[0.1],
                        help="Lambda values to exclude from the plot. "
                             "Defaults to [0.1] because that lam-grid point "
                             "exposed an NPF inner-loop convergence wobble in "
                             "the original sweep; pass --drop_lams (no args) "
                             "to keep them.")
    args = parser.parse_args()

    # Paper-friendly font (falls back to DejaVu Sans on systems without serif).
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    })

    df = pd.read_csv(args.csv)

    if args.drop_lams:
        before = len(df)
        df = df[~df["lam"].isin(args.drop_lams)]
        print(f"dropped lams {args.drop_lams}: {before} -> {len(df)} rows")

    # Drop rows where the chosen W_2 column is NaN (e.g. w2_paired for cloud).
    df_x = df.dropna(subset=[args.w2_col])

    # Training-side Pareto.
    agg_train = aggregate(df_x, x=args.w2_col, y="adv_loss")
    make_pareto_plot(
        agg_train, x_col=args.w2_col, y_col="adv_loss",
        x_label=r"$W_2^2(\mathbb{P}^\star,\widehat{\mathbb{P}})$",
        y_label=r"Adversarial loss $\mathbb{E}_{\widehat{\mathbb{P}}}[f(\theta, T(\widehat z))]$",
        out_path=args.out_dir / "pareto_train.pdf",
    )

    # Generalization-side Pareto, one figure per delta.
    delta_cols = [c for c in df.columns if c.startswith("test_loss_delta_")]
    for col in delta_cols:
        delta = col.replace("test_loss_delta_", "")
        agg = aggregate(df_x, x=args.w2_col, y=col)
        make_pareto_plot(
            agg, x_col=args.w2_col, y_col=col,
            x_label=r"$W_2^2(\mathbb{P}^\star,\widehat{\mathbb{P}})$",
            y_label=rf"Test loss at shift $\Delta={delta}$",
            out_path=args.out_dir / f"pareto_test_delta_{delta}.pdf",
        )


if __name__ == "__main__":
    main()
