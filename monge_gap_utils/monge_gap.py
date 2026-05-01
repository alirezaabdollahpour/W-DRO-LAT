"""Compute the Monge gap of a map T on an empirical reference batch,
under the squared-Euclidean ground cost.

Definition (Uscidda & Cuturi, ICML 2023):
    M(T) = (1/N) sum_i ||T(z_i) - z_i||^2  -  W_2^2(P_hat, T_# P_hat).

The first term is the identity-coupling cost. The second is the W_2 cost
between the empirical reference and its pushforward. The gap is
non-negative; equality holds iff T is OT-optimal between P_hat and T_#P_hat.

Two backends are provided:

- ``monge_gap_hungarian(z, Tz)``: exact, O(N^3). Use only for N <= 4096.
- ``monge_gap_sinkhorn(z, Tz, eps, n_iter)``: entropic-OT approximation,
  O(N^2) per iteration. Bias scales as O(eps); the function reports both
  the raw cost and the bias-corrected estimate via Sinkhorn divergence.

For cross-method comparison, prefer ``monge_gap_sinkhorn`` with
``eps = 0.01 * median_pairwise_dist^2`` and ``n_iter = 1000``, applied
identically to every method. The absolute value of the bias is then
shared by all methods and cancels in the *ranking* of methods.
"""
from __future__ import annotations

from typing import Tuple

import torch
from scipy.optimize import linear_sum_assignment


def median_pairwise_distance_sq(z: torch.Tensor, max_pairs: int = 50_000) -> float:
    """Median of pairwise squared Euclidean distances, used as a robust
    bandwidth for Sinkhorn epsilon. Always subsamples ``max_pairs`` random
    pairs (without materializing the full (N, N, d) tensor)."""
    N = z.shape[0]
    n_pairs = min(max_pairs, N * N)
    idx_i = torch.randint(0, N, (n_pairs,), device=z.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=z.device)
    d = ((z[idx_i] - z[idx_j]) ** 2).sum(-1)
    return float(d.median().item())


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, B) squared-Euclidean cost matrix without materializing (B, B, d).

    Memory: 2 * B^2 * dtype_size for the output matrix plus the two ||.||^2
    vectors. For B = 10_000, d = 512, fp32: ~400 MiB instead of 200 GiB.
    """
    x_sq = (x * x).sum(-1)
    y_sq = (y * y).sum(-1)
    Cxy = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2.0 * (x @ y.t())
    # Numerical noise can drive an exact zero distance slightly negative.
    return Cxy.clamp_min_(0.0)


def monge_gap_hungarian(z: torch.Tensor, Tz: torch.Tensor) -> dict:
    """Exact Monge gap via Hungarian assignment. Use only for small N (<= 4096)."""
    assert z.shape == Tz.shape, f"shape mismatch: {z.shape} vs {Tz.shape}"
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    cost_id = float(((Tz - z) ** 2).sum(-1).mean().item())
    cost_pair = _pairwise_sq_dist(z, Tz)
    row, col = linear_sum_assignment(cost_pair.detach().cpu().numpy())
    w2_sq = float(cost_pair[row, col].mean().item())
    gap = max(cost_id - w2_sq, 0.0)  # guard against numerical undershoot
    return {"cost_id": cost_id, "w2_sq": w2_sq, "monge_gap": gap, "backend": "hungarian"}


def sinkhorn_w2_sq(
    z: torch.Tensor,
    Tz: torch.Tensor,
    eps: float,
    n_iter: int = 1000,
    tol: float = 1e-7,
) -> Tuple[float, float]:
    """Entropic-OT cost and Sinkhorn divergence between two empirical
    distributions of equal size, with uniform weights, squared-Euclidean cost.
    Returns ``(raw_w2_sq_eps, sinkhorn_divergence)``. The latter is bias-corrected."""
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    N = z.shape[0]
    a = torch.full((N,), 1.0 / N, device=z.device, dtype=z.dtype)
    b = a.clone()

    def transport_cost(x, y, a, b):
        Cxy = _pairwise_sq_dist(x, y)
        K = torch.exp(-Cxy / eps)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(n_iter):
            u_new = a / (K @ v + 1e-30)
            v_new = b / (K.t() @ u_new + 1e-30)
            if (u_new - u).abs().max() < tol and (v_new - v).abs().max() < tol:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        return float((plan * Cxy).sum().item())

    cost_xy = transport_cost(z, Tz, a, b)
    cost_xx = transport_cost(z, z, a, a)
    cost_yy = transport_cost(Tz, Tz, b, b)
    sinkhorn_div = cost_xy - 0.5 * (cost_xx + cost_yy)
    return cost_xy, max(sinkhorn_div, 0.0)


def monge_gap_sinkhorn(
    z: torch.Tensor,
    Tz: torch.Tensor,
    eps: float | None = None,
    n_iter: int = 1000,
) -> dict:
    """Monge gap via Sinkhorn. If ``eps`` is None, set it adaptively to
    1% of the median pairwise distance squared on the union of z and Tz."""
    assert z.shape == Tz.shape
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    if eps is None:
        eps = 0.01 * median_pairwise_distance_sq(torch.cat([z, Tz], dim=0))
        # Avoid degenerate eps when both clouds collapse onto the same point.
        if eps <= 0.0:
            eps = 1e-6
    cost_id = float(((Tz - z) ** 2).sum(-1).mean().item())
    raw_w2, sinkdiv = sinkhorn_w2_sq(z, Tz, eps=eps, n_iter=n_iter)
    return {
        "cost_id":          cost_id,
        "w2_sq_raw":        raw_w2,    # entropic, biased: ~ W_2^2 + O(eps)
        "w2_sq_debias":     sinkdiv,   # Sinkhorn divergence, debiased
        "monge_gap_raw":    max(cost_id - raw_w2, 0.0),
        "monge_gap_debias": max(cost_id - sinkdiv, 0.0),
        "epsilon":          eps,
        "backend":          "sinkhorn",
    }


def monge_gap(z: torch.Tensor, Tz: torch.Tensor, batch_size: int = 4096) -> dict:
    """Top-level convenience: choose the backend by batch size."""
    if batch_size <= 4096:
        return monge_gap_hungarian(z, Tz)
    return monge_gap_sinkhorn(z, Tz)
