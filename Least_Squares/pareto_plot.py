"""Read pareto_runs.csv and produce two figures:

  - figs/pareto_train.pdf: W_2^2 vs adv_loss (training-side Pareto front).
  - figs/pareto_test_delta_{D}.pdf: W_2^2 vs test_loss_delta_{D} for each
    delta in the data file (generalization-side fronts).

Run after pareto_sweep.py. Bands are bootstrap 95% CI of the median across
seeds (B=2000 by default), which represent estimator uncertainty rather
than seed-to-seed dispersion. The plot defaults restrict the displayed
range to the well-converged regime (W_2^2 <= 2e-1).

Optional --excess_over_erm rebases each method's curve to its per-seed
difference from ERM, eliminating the dominant seed-to-seed instance
variation introduced by the random A0, A1, b that is shared across
methods at fixed seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

# Default plot order. SDRO ("dual") is excluded by default per the paper
# discussion: its W_2^2 is set by the fixed Sinkhorn noise and does not
# trace a real Pareto curve in lambda, so plotting it as a curve is
# misleading.
DEFAULT_PLOT_ORDER = ["particle_ascent", "ppa", "wfr", "npf"]


# ---------------------------------------------------------------------------
# Aggregation: bootstrap 95% CI of the median across seeds
# ---------------------------------------------------------------------------


def _bootstrap_median_ci(values: np.ndarray, B: int, alpha: float,
                         rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for the median, with a (1-alpha) CI."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    idx = rng.integers(0, v.size, size=(B, v.size))
    boots = np.median(v[idx], axis=1)
    point = float(np.median(v))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, lo, hi


def aggregate(df: pd.DataFrame, x: str, y: str,
              B: int = 2000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Per (method, lam): point = median, band = bootstrap CI of the median.

    Bootstrap is over the seed axis (the unit of replication). Point and
    band are computed independently for x and y.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (method, lam), g in df.groupby(["method", "lam"]):
        x_pt, x_lo, x_hi = _bootstrap_median_ci(g[x].to_numpy(), B, alpha, rng)
        y_pt, y_lo, y_hi = _bootstrap_median_ci(g[y].to_numpy(), B, alpha, rng)
        rows.append({
            "method": method, "lam": lam,
            "x_point": x_pt, "x_lo": x_lo, "x_hi": x_hi,
            "y_point": y_pt, "y_lo": y_lo, "y_hi": y_hi,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["method", "x_point"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _autoscale_y(df_agg: pd.DataFrame, x_max: float | None,
                 pad_frac: float = 0.05) -> tuple[float, float] | None:
    """Tight y-limits: bound the band envelope inside the displayed x-range,
    with `pad_frac` extra room on each side."""
    sub = df_agg
    if x_max is not None:
        sub = sub[sub["x_point"] <= x_max]
    if sub.empty:
        return None
    lo = float(np.nanmin(sub["y_lo"]))
    hi = float(np.nanmax(sub["y_hi"]))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return None
    return lo - pad_frac * span, hi + pad_frac * span


def make_pareto_plot(df_agg: pd.DataFrame, x_label: str, y_label: str,
                     out_path: Path, plot_order: list[str],
                     log_x: bool = True, log_y: bool = False,
                     x_max: float | None = 2e-1,
                     y_pad_frac: float = 0.05,
                     y_lim: tuple[float, float] | None = None,
                     caption: str | None = None):
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)

    for method in plot_order:
        sub = df_agg[df_agg["method"] == method]
        if sub.empty:
            continue
        style = STYLE.get(method, dict(label=method))
        color = style.get("color", None)
        ax.plot(sub["x_point"], sub["y_point"], **style)
        ax.fill_between(sub["x_point"], sub["y_lo"], sub["y_hi"],
                        color=color, alpha=0.18, linewidth=0)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if x_max is not None:
        cur_lo, _ = ax.get_xlim()
        ax.set_xlim(cur_lo, x_max)

    if y_lim is None:
        y_lim = _autoscale_y(df_agg[df_agg["method"].isin(plot_order)],
                             x_max=x_max, pad_frac=y_pad_frac)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55, color="#7a7a7a")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    n_curves = sum(1 for m in plot_order if not df_agg[df_agg["method"] == m].empty)
    ncol = max(1, (n_curves + 1) // 2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, frameon=False, handlelength=2.6,
              columnspacing=1.4, handletextpad=0.6, fontsize=10)

    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top",
                 fontsize=8.5, style="italic", wrap=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Excess-over-ERM rebasing
# ---------------------------------------------------------------------------


def _erm_baseline(seeds: list[int], device: str, deltas: list[float],
                  n_test: int, dim_m: int) -> pd.DataFrame:
    """Train ERM at each seed (closed form, no lambda dependence) and return a
    long-form table with columns {seed, adv_loss_erm, test_loss_delta_*_erm}.

    "Adversarial" loss for ERM is just the clean training loss
    E_P_hat[f(theta_erm, xi_train)] — there is no adversary. This is the
    natural baseline for the (W_2^2, loss) plot.
    """
    import torch
    from algorithms.erm import solve_erm_closed_form
    from config import ULSConfig
    from utils.common import seed_everything
    from utils.data import generate_training_data, make_problem
    from utils.loss import loss_function
    from utils.pareto_metrics import evaluate_test_loss

    dev = torch.device(device)
    rows = []
    for s in seeds:
        cfg = ULSConfig(seed=s)
        seed_everything(s)
        A0, A1, b = make_problem(cfg, dev)
        xi_train = generate_training_data(cfg, dev)
        theta_erm = solve_erm_closed_form(xi_train, cfg, A0, A1, b)
        with torch.no_grad():
            adv_loss_erm = float(loss_function(
                theta_erm, xi_train, A0, A1, b, dim_m).mean().item())
        row = {"seed": s, "adv_loss_erm": adv_loss_erm}
        for d in deltas:
            row[f"test_loss_delta_{d}_erm"] = evaluate_test_loss(
                theta_erm, A0, A1, b, delta=d, dim_m=dim_m,
                n_test=n_test, seed=s + 100_000,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _rebase_to_excess(df: pd.DataFrame, deltas_in_df: list[str],
                      device: str, n_test: int, dim_m: int) -> pd.DataFrame:
    """Subtract ERM's per-seed loss from each method's loss, in place."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = sorted(df["seed"].unique().tolist())
    delta_floats = [float(d) for d in deltas_in_df]
    erm = _erm_baseline(seeds, device, delta_floats, n_test, dim_m)
    df = df.merge(erm, on="seed", how="left")
    df["adv_loss"] = df["adv_loss"] - df["adv_loss_erm"]
    for d in deltas_in_df:
        col = f"test_loss_delta_{d}"
        if col in df.columns:
            df[col] = df[col] - df[f"{col}_erm"]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("pareto_runs.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("figs"))
    parser.add_argument("--w2_col", choices=["w2_paired", "w2_optimal"],
                        default="w2_optimal",
                        help="Which W_2^2 estimator to plot on the x-axis.")
    parser.add_argument("--drop_lams", nargs="*", type=float, default=[0.1],
                        help="Lambda values to exclude. Default [0.1] removes "
                             "the NPF inner-loop wobble; pass --drop_lams to keep them.")
    parser.add_argument("--x_max", type=float, default=2e-1,
                        help="Right edge of the displayed W_2^2 range. "
                             "Default 2e-1 restricts to the well-converged regime.")
    parser.add_argument("--y_pad", type=float, default=0.05,
                        help="Fractional padding around the band envelope when "
                             "auto-scaling the y-axis.")
    parser.add_argument("--include_sdro", action="store_true",
                        help="Plot SDRO (dual). Excluded by default because its "
                             "W_2^2 is set by the fixed Sinkhorn noise level and "
                             "does not trace a true Pareto curve.")
    parser.add_argument("--bootstrap_B", type=int, default=2000,
                        help="Bootstrap resample count for CI bands.")
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--excess_over_erm", action="store_true",
                        help="Rebase each method's loss to (loss - ERM_loss) "
                             "computed at the same seed. Cancels seed-to-seed "
                             "instance variance from the random A0/A1/b.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for ERM baseline computation (auto/cuda/cpu).")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Test sample count for ERM baseline (matches sweep).")
    parser.add_argument("--dim_m", type=int, default=10,
                        help="Problem rows for ERM baseline (matches sweep).")
    args = parser.parse_args()

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

    plot_order = list(DEFAULT_PLOT_ORDER)
    if args.include_sdro:
        plot_order.append("dual")

    delta_cols = [c for c in df.columns if c.startswith("test_loss_delta_")]
    delta_strs = [c.replace("test_loss_delta_", "") for c in delta_cols]

    if args.excess_over_erm:
        print("Rebasing to excess-loss-over-ERM ...")
        df = _rebase_to_excess(df, delta_strs, args.device, args.n_test, args.dim_m)
        loss_y_label = (r"Adv loss $-$ ERM clean loss "
                        r"$\mathbb{E}_{\widehat{\mathbb{P}}}[f(\theta,T(\widehat z))]"
                        r"-\mathbb{E}_{\widehat{\mathbb{P}}}[f(\theta_{\mathrm{ERM}},\widehat z)]$")
        test_y_label_fmt = (r"Test loss $-$ ERM test loss at $\Delta={d}$")
    else:
        loss_y_label = (r"Adversarial loss "
                        r"$\mathbb{E}_{\widehat{\mathbb{P}}}[f(\theta, T(\widehat z))]$")
        test_y_label_fmt = r"Test loss at shift $\Delta={d}$"

    df_x = df.dropna(subset=[args.w2_col])

    caption_train = (
        "We restrict the displayed range to the well-converged regime; "
        "full untruncated curves in Appendix X. "
        "SDRO is omitted: its $W_2^2$ is set by the fixed Sinkhorn noise level "
        "and does not trace a true Pareto curve in $\\lambda$. "
        "Bands are bootstrap 95\\% CIs of the median across "
        f"{df_x['seed'].nunique()} seeds (B={args.bootstrap_B})."
    )
    caption_test_fmt = caption_train  # same caveats apply

    # Training-side plot.
    agg_train = aggregate(df_x, x=args.w2_col, y="adv_loss",
                          B=args.bootstrap_B, alpha=0.05,
                          seed=args.bootstrap_seed)
    make_pareto_plot(
        agg_train,
        x_label=r"$W_2^2(\mathbb{P}^\star,\widehat{\mathbb{P}})$",
        y_label=loss_y_label,
        out_path=args.out_dir / "pareto_train.pdf",
        plot_order=plot_order,
        x_max=args.x_max,
        y_pad_frac=args.y_pad,
        caption=caption_train,
    )

    # Generalization-side plots, one per delta.
    for col, d in zip(delta_cols, delta_strs):
        agg = aggregate(df_x, x=args.w2_col, y=col,
                        B=args.bootstrap_B, alpha=0.05,
                        seed=args.bootstrap_seed)
        make_pareto_plot(
            agg,
            x_label=r"$W_2^2(\mathbb{P}^\star,\widehat{\mathbb{P}})$",
            y_label=test_y_label_fmt.format(d=d),
            out_path=args.out_dir / f"pareto_test_delta_{d}.pdf",
            plot_order=plot_order,
            x_max=args.x_max,
            y_pad_frac=args.y_pad,
            caption=caption_test_fmt,
        )


if __name__ == "__main__":
    main()
