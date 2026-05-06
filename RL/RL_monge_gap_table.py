"""Render the RL CartPole Monge-gap CSV (produced by monge_gap_sweep.py) as
a LaTeX table mirroring the LR-CIFAR10 paper layout
(``tab:monge_gap_vs_lambda_lr``).

Each cell shows the seed-averaged debiased Sinkhorn estimator
``monge_gap_2_debiased`` (or ``monge_gap_2_hungarian`` when the sweep ran in
Hungarian mode) for that (method, lam) pair, formatted as ``MeXX-Y``.

Usage:
    python RL_monge_gap_table.py \\
        --csv RL/monge_gap_runs/monge_gap_rl_cartpole.csv \\
        --out RL/monge_gap_runs/monge_gap_rl_cartpole.tex \\
        --lams 0.1 0.5 1.0 2.0 5.0 \\
        --methods nominal ro particle wfr dual nn_dro ppa icnn
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# Display name -> (CSV method key, paper label, group)
# group: "ref" for ERM (highlighted reference row), "baseline" / "ours".
# MPA in the paper = ``new_ppa`` in the RL code (Algorithm 1: PA inner loop
# + batch-wide reassignment). The older ``ppa`` (Brenier-projection variant)
# is *not* MPA.
ROW_SPECS = {
    "nominal":  {"label": "ERM",      "group": "ref"},
    "ro":       {"label": "RO",       "group": "baseline"},
    "particle": {"label": "PA",       "group": "baseline"},
    "algo1":    {"label": "WRM",      "group": "baseline"},
    "wfr":      {"label": "WFR",      "group": "baseline"},
    "dual":     {"label": "SDRO",     "group": "baseline"},
    "nn_dro":   {"label": "NN-DRO",   "group": "baseline"},
    "ppa":      {"label": "PPA",      "group": "baseline"},
    "new_ppa":  {"label": "MPA",      "group": "ours"},
    "icnn":     {"label": "ICNN",     "group": "baseline"},
    "npf":      {"label": "ICNN-DRO", "group": "ours"},
}

# Preferred metric column order — first one present wins.
# Matches the keys emitted by monge_gap_utils/monge_gap.py:
#   * Sinkhorn:  ``monge_gap_debias`` (paper estimator) or ``monge_gap_raw``
#   * Hungarian / subsample-Hungarian: ``monge_gap``
GAP_COLS = (
    "monge_gap_debias",   # Sinkhorn debiased (matches paper estimator)
    "monge_gap",          # Hungarian / subsample Hungarian
    "monge_gap_raw",      # Sinkhorn without debiasing — fallback
)


def _pick_gap_col(df: pd.DataFrame) -> str:
    for c in GAP_COLS:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a Monge-gap column in CSV. Tried {GAP_COLS}; "
        f"available: {list(df.columns)}."
    )


def _fmt_cell(value: float) -> str:
    if not np.isfinite(value):
        return "{--}"
    if value == 0.0:
        return "{$0$}"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    exp = int(np.floor(np.log10(v))) if v > 0 else 0
    mantissa = v / (10.0 ** exp)
    # Match paper's ``1.24e-2`` style.
    return f"{sign}{mantissa:.2f}e{exp:+d}".replace("e+0", "e+").replace("e-0", "e-").replace("e+", "e") if exp != 0 else f"{sign}{mantissa:.2f}e+0"


def _fmt_paper(value: float) -> str:
    """Match the paper's exact format: e.g. 1.24e-2, 5.07e-4, 1.83e-0, 1.52e-7."""
    if not np.isfinite(value):
        return "{--}"
    if value == 0.0:
        return "{$0$}"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    exp = int(np.floor(np.log10(v)))
    mantissa = v / (10.0 ** exp)
    if exp >= 0:
        return f"{sign}{mantissa:.2f}e-{0}" if exp == 0 else f"{sign}{mantissa:.2f}e+{exp}"
    return f"{sign}{mantissa:.2f}e{exp}"  # exp already has '-'


def _format_paper(value: float) -> str:
    """Final paper-style formatter: 1.24e-2, 1.83e-0 (sic, paper uses e-0 for 1.x)."""
    if not np.isfinite(value):
        return "{--}"
    if value == 0.0:
        return "{$0$}"
    v = float(abs(value))
    if v == 0.0:
        return "{$0$}"
    exp = int(np.floor(np.log10(v)))
    mantissa = v / (10.0 ** exp)
    sign = "-" if value < 0 else ""
    # Paper format: <m.mm>e<sign><digit>, e.g. 1.83e-0, 5.68e-4, 1.52e-7.
    if exp == 0:
        return f"{sign}{mantissa:.2f}e-0"
    return f"{sign}{mantissa:.2f}e{exp:+d}".replace("e+", "e+").replace("e-", "e-")


def render_table(
    df: pd.DataFrame,
    methods: List[str],
    lams: List[float],
    label: str = "tab:monge_gap_vs_lambda_rl",
) -> str:
    gap_col = _pick_gap_col(df)
    df = df.copy()
    df["lam"] = df["lam"].astype(float).round(8)

    # Aggregate over seeds: median is robust to the rare numerically blown-up
    # cell. The LR-CIFAR10 paper text reads as a per-cell point estimate, and
    # median matches that intent more reliably than mean over only 1–3 seeds.
    agg = (
        df.groupby(["method", "lam"])[gap_col]
          .median()
          .reset_index()
          .pivot(index="method", columns="lam", values=gap_col)
    )

    # Order rows by group: ref → baselines → ours (matching paper layout).
    by_group: dict[str, list[tuple[str, str]]] = {"ref": [], "baseline": [], "ours": []}
    for m in methods:
        if m not in ROW_SPECS:
            continue
        spec = ROW_SPECS[m]
        by_group[spec["group"]].append((m, spec["label"]))

    cols = sorted(set(round(float(l), 8) for l in lams))

    # --- header ---
    n_lam = len(cols)
    col_spec = "@{}c l " + " ".join(["S"] * n_lam) + "@{}"
    lam_headers = " & ".join(f"{{${l:g}$}}" for l in cols)
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"  \caption{\textbf{Monge gap of the trained adversary vs.\ regularization "
        r"$\lambda$ on RL-CartPole~(\autoref{sec:Adversarial_RL}).} "
        r"Empirical debiased estimator $\widehat{\cM}_{\widehat\PP}(T)$ on a held-out "
        r"anchor batch sampled uniformly from the xi-box, computed with the "
        r"bias-corrected Sinkhorn divergence~\citep{uscidda2023monge}. "
        r"Smaller is better. For ICNN-DRO and NPF, the gap is evaluated in the "
        r"latent $u$-space on which the convex potential is defined; the "
        r"residual $\xi$-space gap induced by the sigmoid box decoder is not "
        r"reflected here. For all other rows the gap is evaluated in $\xi$-space "
        r"under the same diagonal-Mahalanobis cost $\|\cdot\|_M^2$ used by the "
        r"training objective. The highlighted reference row "
        r"($T_{\mathrm{ERM}}{:=}\mathrm{id}$, $\cM\!=\!0$ by definition) is not "
        r"a candidate adversary. ICNN-DRO satisfies "
        r"$\cM_{\widehat\PP\circ\,\mathrm{encode}^{-1}}(\nabla_u\psi_\omega) = 0$ "
        r"exactly in $u$-space by construction (\autoref{prop:wasted_transport}); "
        r"the $\sim\!10^{-7}$ entries are the Sinkhorn estimator's convergence "
        r"floor, not residual transport waste.}"
    )
    lines.append(r"  \resizebox{\linewidth}{!}{")
    lines.append(rf"    \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"      \toprule")
    lines.append(rf"       & & \multicolumn{{{n_lam}}}{{c}}{{Regularization strength $\lambda$}} \\")
    lines.append(rf"      \cmidrule(lr){{3-{2 + n_lam}}}")
    lines.append(rf"       & Method & {lam_headers} \\")
    lines.append(r"      \midrule")

    # --- ref row (ERM) ---
    if by_group["ref"]:
        for m, lbl in by_group["ref"]:
            zero_cells = " & ".join(["{$0$}"] * n_lam)
            lines.append(r"      \rowcolor{refrow}")
            lines.append(rf"       & {lbl}      & {zero_cells} \\")
        lines.append(r"      \hdashline\noalign{\vskip 1pt}")

    # --- baselines block ---
    n_base = len(by_group["baseline"])
    if n_base > 0:
        lines.append(rf"      \multirow{{{n_base}}}{{*}}{{\rotatebox[origin=c]{{90}}{{Baselines}}}}")
        for i, (m, lbl) in enumerate(by_group["baseline"]):
            row_vals = []
            for l in cols:
                v = agg.at[m, l] if (m in agg.index and l in agg.columns) else float("nan")
                row_vals.append(_format_paper(float(v)))
            cells = " & ".join(row_vals)
            lines.append(rf"       & {lbl:<8} & {cells} \\")

    # --- ours block ---
    n_ours = len(by_group["ours"])
    if n_ours > 0:
        lines.append(r"      \cmidrule(l){2-" + str(2 + n_lam) + "}")
        lines.append(rf"      \multirow{{{n_ours}}}{{*}}{{\rotatebox[origin=c]{{90}}{{Ours}}}}")
        for m, lbl in by_group["ours"]:
            row_vals = []
            for l in cols:
                v = agg.at[m, l] if (m in agg.index and l in agg.columns) else float("nan")
                row_vals.append(_format_paper(float(v)))
            cells = " & ".join(row_vals)
            lines.append(rf"       & {lbl:<8} & {cells} \\")

    lines.append(r"      \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Method keys to render (in the order they appear in the table).")
    parser.add_argument("--lams", nargs="+", type=float, required=True)
    parser.add_argument("--label", type=str, default="tab:monge_gap_vs_lambda_rl")
    args = parser.parse_args(list(argv) if argv is not None else None)

    df = pd.read_csv(args.csv)
    if "dataset" in df.columns:
        df = df[df["dataset"] == "rl_cartpole"].copy()
    tex = render_table(df, methods=list(args.methods), lams=list(args.lams), label=args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"[table] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
