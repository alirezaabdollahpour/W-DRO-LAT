"""Sanity checks for pareto_runs.csv per the experiment spec.

Run after pareto_sweep.py completes:
    python pareto_sanity_check.py --csv pareto_runs.csv

Reports pass/fail for each check; non-zero exit if any fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _print(label: str, ok: bool, msg: str = "") -> int:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + msg) if msg else ''}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("pareto_runs.csv"))
    parser.add_argument("--w2_eps", type=float, default=1e-6,
                        help="Floating-point tolerance for w2_paired vs w2_optimal.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"missing: {args.csv}")
        return 2
    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    print(f"  methods: {sorted(df['method'].unique().tolist())}")
    print(f"  lams:    {sorted(df['lam'].unique().tolist())}")
    print(f"  seeds:   {sorted(df['seed'].unique().tolist())}")

    fails = 0

    # 1. At lam=100, W_2^2 should be ~0 for all methods.
    print("\n[1] Large-lambda collapse: W_2^2 -> 0 at lam=100")
    lam_max = df["lam"].max()
    for m in sorted(df["method"].unique()):
        sub = df[(df["method"] == m) & (df["lam"] == lam_max)]
        med = sub["w2_optimal"].median()
        fails += _print(f"{m} median w2_optimal at lam={lam_max}",
                        med < 1e-2, f"got {med:.2e}")

    # 2. At lam=0.01, W_2^2 should be large and curves separated.
    print("\n[2] Small-lambda spread: W_2^2 large at lam=0.01")
    lam_min = df["lam"].min()
    spread = []
    for m in sorted(df["method"].unique()):
        sub = df[(df["method"] == m) & (df["lam"] == lam_min)]
        med = sub["w2_optimal"].median()
        spread.append((m, med))
        fails += _print(f"{m} median w2_optimal at lam={lam_min}",
                        med > 1e-3, f"got {med:.2e}")
    spreads = np.array([v for _, v in spread])
    if len(spreads) > 1:
        rel = (spreads.max() - spreads.min()) / max(spreads.max(), 1e-12)
        fails += _print("relative spread across methods at lam=0.01",
                        rel > 0.1, f"{rel:.2%}")

    # 3. NPF and PPA should have w2_paired == w2_optimal (cyclically monotone).
    print("\n[3] Cyclic monotonicity: w2_paired == w2_optimal for NPF, PPA")
    for m in ("npf", "ppa"):
        sub = df[df["method"] == m].dropna(subset=["w2_paired", "w2_optimal"])
        if len(sub) == 0:
            fails += _print(f"{m} (no data)", False)
            continue
        diff = (sub["w2_paired"] - sub["w2_optimal"]).abs().max()
        fails += _print(f"{m} max |paired - optimal|",
                        diff < args.w2_eps, f"{diff:.2e}")

    # 4. PA in m=1 (1D xi) should also have w2_paired == w2_optimal.
    print("\n[4] PA at m=1 (Prop. 3.1): w2_paired == w2_optimal")
    sub = df[df["method"] == "particle_ascent"].dropna(subset=["w2_paired", "w2_optimal"])
    if len(sub) > 0:
        diff = (sub["w2_paired"] - sub["w2_optimal"]).abs().max()
        fails += _print("PA max |paired - optimal|",
                        diff < args.w2_eps, f"{diff:.2e}")
    else:
        fails += _print("PA (no data)", False)

    # 5. Per-method monotonicity of adv_loss in W_2^2 on the training-side curve.
    print("\n[5] Training-side monotonicity: adv_loss non-decreasing in W_2^2")
    for m in sorted(df["method"].unique()):
        agg = (df[df["method"] == m]
               .groupby("lam")[["w2_optimal", "adv_loss"]].median()
               .reset_index().sort_values("w2_optimal"))
        if len(agg) < 2:
            continue
        diffs = np.diff(agg["adv_loss"].to_numpy())
        # Tolerate small numerical wobble (1% of mean adv_loss).
        tol = 0.01 * float(agg["adv_loss"].abs().mean())
        ok = bool((diffs >= -tol).all())
        n_viol = int((diffs < -tol).sum())
        fails += _print(f"{m} monotone (tol={tol:.3e})",
                        ok, f"{n_viol} violations out of {len(diffs)}")

    # 6. NPF Pareto-dominance over baselines on training-side curve.
    #    25th-percentile NPF below 75th-percentile of every baseline at every
    #    lam-grid point.
    print("\n[6] Pareto dominance: NPF q25 < baseline q75 (per lam)")
    if "npf" not in df["method"].unique():
        fails += _print("NPF data missing", False)
    else:
        baselines = [m for m in df["method"].unique() if m != "npf"]
        npf_q = (df[df["method"] == "npf"]
                 .groupby("lam")["adv_loss"].quantile([0.25, 0.75]).unstack())
        violations = []
        for base in baselines:
            base_q = (df[df["method"] == base]
                      .groupby("lam")["adv_loss"].quantile([0.25, 0.75]).unstack())
            for lam in npf_q.index:
                if lam not in base_q.index:
                    continue
                if not (npf_q.loc[lam, 0.25] < base_q.loc[lam, 0.75]):
                    violations.append((base, lam,
                                       float(npf_q.loc[lam, 0.25]),
                                       float(base_q.loc[lam, 0.75])))
        ok = len(violations) == 0
        fails += _print("NPF dominance over all baselines",
                        ok, f"{len(violations)} violations")
        for v in violations[:5]:
            print(f"      e.g. base={v[0]} lam={v[1]} npf_q25={v[2]:.4f} "
                  f"vs base_q75={v[3]:.4f}")

    print()
    if fails == 0:
        print("ALL CHECKS PASSED")
        return 0
    print(f"{fails} CHECK(S) FAILED — see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
