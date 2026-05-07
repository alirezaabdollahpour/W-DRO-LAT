"""Sweep lambda over a logarithmic grid for every DRO method, record the
empirical W_2^2 of the worst-case distribution and the train/test losses,
and write tidy results to disk for plotting.

Usage (from inside Least_Squares/):
    python pareto_sweep.py \
        --methods particle_ascent ppa npf wfr dual \
        --seeds 219 220 221 222 223 224 225 226 227 228 \
        --lams 0.01 0.0316 0.1 0.316 1.0 3.16 10.0 31.6 100.0 \
        --deltas 0.0 1.0 5.0 10.0 \
        --out pareto_runs.csv \
        --workers 8

With --workers > 1, the (method, lam, seed) tasks are distributed across
that many subprocesses, each holding its own CUDA context. For an A100
running this N=10 problem, the inner loop has so little compute that one
process saturates only ~25% of the SMs; 4-8 workers bring throughput close
to GPU-bound. If --out already contains rows, those (method, lam, seed)
triples are skipped — so a crashed sweep can be resumed by re-invoking
with the same --out path.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from algorithms.dual import _solve_dual_with_zstar
from algorithms.npf import (
    _solve_npf_icnn_map_with_zstar,
    _solve_npf_lastquad_icnn_map_with_zstar,
)
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
    "npf_lastquad":    _solve_npf_lastquad_icnn_map_with_zstar,
}

# Round lam to this many digits when matching against an existing CSV, so
# floating-point round-trip through csv.to_csv / read_csv does not cause
# spurious "this triple is missing" reruns.
_LAM_ROUND = 8


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
    if z_star is not None:
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


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------


_WORKER_READY = False


def _ensure_worker_ready(device_str: str) -> None:
    """Idempotent per-process setup: dtype + CUDA device pin. Run on first task
    rather than as a Pool initializer because initializer failures in spawn
    mode silently re-spawn the worker forever, masking bugs."""
    global _WORKER_READY
    if _WORKER_READY:
        return
    torch.set_default_dtype(torch.float64)
    if device_str.startswith("cuda"):
        # torch.cuda.set_device wants an integer index, not a bare "cuda".
        idx = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
        torch.cuda.set_device(idx)
    _WORKER_READY = True


def _worker_run(task: Tuple[str, float, int, list, str]) -> dict:
    method, lam, seed, deltas, device_str = task
    _ensure_worker_ready(device_str)
    device = torch.device(device_str)
    cfg = replace(ULSConfig(), lam=lam, seed=seed)
    return run_one(method, cfg, deltas, device)


def _load_existing(out_path: Path) -> Tuple[list, set]:
    """Load existing rows and a set of (method, lam_rounded, seed) keys."""
    if not out_path.exists():
        return [], set()
    df = pd.read_csv(out_path)
    rows = df.to_dict("records")
    done = {(r["method"], round(float(r["lam"]), _LAM_ROUND), int(r["seed"])) for r in rows}
    return rows, done


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
    parser.add_argument("--workers", type=int, default=1,
                        help="Subprocesses sharing the GPU. >1 requires the "
                             "spawn start method (handled automatically).")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    rows, done = _load_existing(args.out)

    # Build the task list, skipping triples already in --out.
    all_tasks = []
    for method in args.methods:
        for lam in args.lams:
            for seed in args.seeds:
                key = (method, round(float(lam), _LAM_ROUND), int(seed))
                if key not in done:
                    all_tasks.append((method, float(lam), int(seed), args.deltas, device_str))

    n_skipped = len(done) - sum(1 for r in rows
                                if (r["method"], round(float(r["lam"]), _LAM_ROUND), int(r["seed"]))
                                not in {(t[0], round(t[1], _LAM_ROUND), t[2]) for t in all_tasks})
    n_total = len(all_tasks)
    print(f"{len(rows)} rows already in {args.out}; {n_total} new tasks to run "
          f"({args.workers} worker(s), device={device_str}).", flush=True)

    if n_total == 0:
        print("Nothing to do.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        for i, t in enumerate(all_tasks):
            method, lam, seed, _, _ = t
            print(f"[{i+1}/{n_total}] method={method} lam={lam:.4g} seed={seed}",
                  flush=True)
            row = _worker_run(t)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    else:
        # Limit per-process intra-op threads so workers do not oversubscribe CPU.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker_run, all_tasks)):
                rows.append(row)
                print(f"[{i+1}/{n_total}] {row['method']} lam={row['lam']:.4g} "
                      f"seed={row['seed']} t={row['train_time']:.1f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"Done. Wrote {len(rows)} rows to {args.out}.")


if __name__ == "__main__":
    main()
