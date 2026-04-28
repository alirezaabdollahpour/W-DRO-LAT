"""Helpers that compute the empirical W_2^2 and the per-distribution losses
needed for the Pareto-curve experiment.

The 2-Wasserstein distance from the empirical reference
P_hat = (1/N) sum_i delta_{z_hat_i} to the worst-case empirical measure
P_star = (1/N) sum_i delta_{z_i_star} is the optimal-assignment cost,

    W_2^2 = (1/N) min_{sigma in S_N} sum_i || z_{sigma(i)}_star - z_hat_i ||^2.

Three useful quantities derive from this:

- `w2_sq_paired(z_hat, z_star)`: plug-in estimator with the identity
  assignment i -> i. Equals W_2^2 exactly when the map z_hat_i -> z_star_i
  is cyclically monotone (ICNN-DRO, MPA terminal output, PA in m=1).
  Upper-bounds W_2^2 otherwise (e.g. PA in m>=2).

- `w2_sq_optimal(z_hat, z_star)`: exact W_2^2 via Hungarian algorithm.
  O(N^3) but N = 10 here so this is free.

- `w2_sq_cloud(z_hat, particle_cloud)`: exact W_2^2 from the empirical
  reference to a particle cloud of arbitrary size, again via Hungarian.
  Used for WFR / SDRO which produce N*M particles instead of N pairs.
"""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def w2_sq_paired(z_hat: torch.Tensor, z_star: torch.Tensor) -> float:
    """Plug-in W_2^2 with identity assignment. z_hat, z_star: (N, m)."""
    assert z_hat.shape == z_star.shape, f"{z_hat.shape} vs {z_star.shape}"
    return float(((z_star - z_hat) ** 2).sum(dim=-1).mean().item())


def w2_sq_optimal(z_hat: torch.Tensor, z_star: torch.Tensor) -> float:
    """Exact W_2^2 from z_hat to z_star via Hungarian assignment.
    z_hat, z_star: (N, m)."""
    assert z_hat.shape == z_star.shape
    cost = ((z_hat.unsqueeze(1) - z_star.unsqueeze(0)) ** 2).sum(dim=-1)
    cost_np = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost_np)
    return float(cost_np[row, col].mean())


def w2_sq_cloud(z_hat: torch.Tensor, cloud: torch.Tensor) -> float:
    """Exact W_2^2 from z_hat (N, m) to a particle cloud (K, m) via Hungarian.
    Each anchor is matched to a single particle; mass is uniformly 1/N on the
    anchor side. linear_sum_assignment handles the non-square case by
    leaving K - N particles unmatched."""
    cost = ((z_hat.unsqueeze(1) - cloud.unsqueeze(0)) ** 2).sum(dim=-1)
    cost_np = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost_np)
    return float(cost_np[row, col].mean())


def evaluate_test_loss(
    theta: torch.Tensor,
    A0: torch.Tensor, A1: torch.Tensor, b: torch.Tensor,
    delta: float, dim_m: int,
    n_test: int = 1000, seed: int = 0,
) -> float:
    """Evaluate mean loss of theta on test samples drawn from
    Unif([-0.5*(1+delta), 0.5*(1+delta)]).
    Mirrors the test sampling convention already used in Least_Squares.py."""
    from utils.loss import loss_function
    g = torch.Generator(device=theta.device).manual_seed(int(seed))
    z = (torch.rand(n_test, generator=g, device=theta.device,
                    dtype=theta.dtype) - 0.5) * (1.0 + float(delta))
    return float(loss_function(theta, z, A0, A1, b, dim_m).mean().item())
