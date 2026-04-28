"""Read pareto_runs.csv and produce two figures:

  - figs/pareto_train.pdf: W_2^2 vs adv_loss (training-side Pareto front).
  - figs/pareto_test_delta_{D}.pdf: W_2^2 vs test_loss_delta_{D} for each
    delta in the data file (generalization-side fronts).

Run after pareto_sweep.py. Uses median across seeds with shaded
[25th, 75th] percentile band.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Per-method visual style.
STYLE = {
    "particle_ascent": dict(label="PA",        marker="o", linestyle="--"),
    "ppa":             dict(label="MPA",       marker="s", linestyle="-"),
    "npf":             dict(label="ICNN-DRO",  marker="D", linestyle="-",  linewidth=2.0),
    "wfr":             dict(label="WFR",       marker="^", linestyle=":"),
    "dual":            dict(label="SDRO",      marker="v", linestyle=":"),
}


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
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=150)
    for method, sub in df_agg.groupby("method"):
        style = STYLE.get(method, dict(label=method, marker="x"))
        ax.plot(sub["x_median"], sub["y_median"], **style)
        ax.fill_between(sub["x_median"], sub["y_q25"], sub["y_q75"],
                        alpha=0.15, linewidth=0)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("pareto_runs.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("figs"))
    parser.add_argument("--w2_col", choices=["w2_paired", "w2_optimal"],
                        default="w2_optimal",
                        help="Which W_2^2 estimator to plot on the x-axis.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

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
