"""Sweep lambda over a logarithmic grid for every DRO method, record the
empirical W_2^2 of the worst-case distribution and the train/test losses,
and write tidy results to disk for plotting.

Usage (from inside Least_Squares/):
    python pareto_sweep.py \
        --methods particle_ascent ppa npf wfr dual \
        --seeds 219 220 221 222 223 224 225 226 227 228 \
        --lams 0.01 0.0316 0.1 0.316 1.0 3.16 10.0 31.6 100.0 \
        --deltas 0.0 1.0 5.0 10.0 \
        --out pareto_runs.csv
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from algorithms.dual import _solve_dual_with_zstar
from algorithms.npf import _solve_npf_icnn_map_with_zstar
from algorithms.particle_ascent import _solve_Particle_Ascent_with_zstar
from algorithms.ppa import _solve_ppa_with_zstar
from algorithms.wfr import _solve_wfr_with_zstar
from config import ULSConfig
from utils.common import seed_everything
from utils.data import generate_training_data, make_problem
from utils.loss import loss_function
from utils.pareto_metrics import (
    evaluate_test_loss,
    w2_sq_cloud,
    w2_sq_optimal,
    w2_sq_paired,
)


METHODS = {
    "particle_ascent": _solve_Particle_Ascent_with_zstar,
    "ppa":             _solve_ppa_with_zstar,
    "npf":             _solve_npf_icnn_map_with_zstar,
    "wfr":             _solve_wfr_with_zstar,
    "dual":            _solve_dual_with_zstar,
}


def make_problem_and_data(cfg: ULSConfig, device: torch.device):
    """Generate (A0, A1, b, xi_train) using the existing seeded conventions."""
    seed_everything(cfg.seed)
    A0, A1, b = make_problem(cfg, device)
    xi_train = generate_training_data(cfg, device)
    return A0, A1, b, xi_train


def run_one(method: str, cfg: ULSConfig, deltas: list, device: torch.device) -> dict:
    """Train one (method, lam, seed) configuration and return a dict of metrics."""
    A0, A1, b, xi_train = make_problem_and_data(cfg, device)
    train_fn = METHODS[method]

    t0 = time.time()
    out = train_fn(xi_train, cfg, A0, A1, b)
    train_time = time.time() - t0
    theta = out["theta"]
    z_star = out["z_star"]
    z_kind = out["z_star_kind"]

    # --- W_2^2 ---
    z_hat = xi_train.unsqueeze(-1) if xi_train.dim() == 1 else xi_train
    if z_star is None:
        w2_paired = float("nan")
        w2_optimal = float("nan")
    elif z_kind == "paired":
        z_st = z_star.unsqueeze(-1) if z_star.dim() == 1 else z_star
        w2_paired = w2_sq_paired(z_hat, z_st)
        w2_optimal = w2_sq_optimal(z_hat, z_st)
    elif z_kind == "cloud":
        z_cl = z_star.unsqueeze(-1) if z_star.dim() == 1 else z_star
        w2_paired = float("nan")  # not meaningful for clouds
        w2_optimal = w2_sq_cloud(z_hat, z_cl)
    else:
        raise ValueError(f"Unknown z_star_kind {z_kind!r}")

    # --- Adversarial training-side loss ---
    if z_star is not None and z_kind == "paired":
        with torch.no_grad():
            adv_loss = float(
                loss_function(theta, z_star.reshape(-1), A0, A1, b, cfg.dim_m).mean().item()
            )
    elif z_star is not None and z_kind == "cloud":
        with torch.no_grad():
            adv_loss = float(
                loss_function(theta, z_star.reshape(-1), A0, A1, b, cfg.dim_m).mean().item()
            )
    else:
        adv_loss = float("nan")

    # --- Test losses on shifted distributions ---
    test_losses = {}
    for delta in deltas:
        # Use a separate seed for test sampling to avoid leaking train into test.
        test_seed = cfg.seed + 100_000
        test_losses[f"test_loss_delta_{delta}"] = evaluate_test_loss(
            theta, A0, A1, b, delta=delta, dim_m=cfg.dim_m,
            n_test=cfg.n_test, seed=test_seed,
        )

    return {
        "method":     method,
        "lam":        cfg.lam,
        "seed":       cfg.seed,
        "w2_paired":  w2_paired,
        "w2_optimal": w2_optimal,
        "adv_loss":   adv_loss,
        "train_time": train_time,
        **test_losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True,
                        choices=list(METHODS.keys()))
    parser.add_argument("--seeds",   nargs="+", type=int, required=True)
    parser.add_argument("--lams",    nargs="+", type=float, required=True)
    parser.add_argument("--deltas",  nargs="+", type=float,
                        default=[0.0, 1.0, 5.0, 10.0])
    parser.add_argument("--out",     type=Path, default=Path("pareto_runs.csv"))
    parser.add_argument("--device",  type=str, default=None,
                        help="Defaults to cuda if available, else cpu.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base_cfg = ULSConfig()
    rows = []

    n_total = len(args.methods) * len(args.seeds) * len(args.lams)
    i = 0
    for method in args.methods:
        for lam in args.lams:
            for seed in args.seeds:
                cfg = replace(base_cfg, lam=lam, seed=seed)
                i += 1
                print(f"[{i}/{n_total}] method={method} lam={lam:.4g} seed={seed}",
                      flush=True)
                row = run_one(method, cfg, args.deltas, device)
                rows.append(row)
                # Persist incrementally so a crash does not lose work.
                pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"Done. Wrote {len(rows)} rows to {args.out}.")


if __name__ == "__main__":
    main()
