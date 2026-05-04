"""Compute the Monge gap of a map T on an empirical reference batch,
under either squared-Euclidean or diagonal-Mahalanobis ground cost.

Definition (Uscidda & Cuturi, ICML 2023), generalised to a diagonal
Mahalanobis cost ``c_M(x, y) = (x - y)^T diag(M) (x - y)``:

    M(T) = (1/N) sum_i c_M(T(z_i), z_i)  -  W_2^2(P_hat, T_# P_hat; c_M).

The first term is the identity-coupling cost. The second is the OT cost
between the empirical reference and its pushforward under c_M. The gap
is non-negative; equality holds iff T is OT-optimal between P_hat and
T_#P_hat under c_M. Passing ``M_diag = None`` (the default) recovers the
squared-Euclidean cost.

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

import math
from typing import Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment


def median_pairwise_distance_sq(
    z: torch.Tensor,
    max_pairs: int = 50_000,
    M_diag: Optional[torch.Tensor] = None,
) -> float:
    """Median of pairwise (Mahalanobis-)squared distances, used as a robust
    bandwidth for Sinkhorn epsilon. Always subsamples ``max_pairs`` random
    pairs (without materializing the full (N, N, d) tensor)."""
    N = z.shape[0]
    n_pairs = min(max_pairs, N * N)
    idx_i = torch.randint(0, N, (n_pairs,), device=z.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=z.device)
    diff = z[idx_i] - z[idx_j]
    if M_diag is not None:
        d = ((diff * diff) * M_diag).sum(-1)
    else:
        d = (diff * diff).sum(-1)
    return float(d.median().item())


def _pairwise_sq_dist_direct(
    x: torch.Tensor,
    y: torch.Tensor,
    M_diag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Memory-heavy but numerically-stable direct expansion. Use when
    ``B*M*d`` fits in memory: avoids the catastrophic cancellation of
    ``x_sq + y_sq - 2 x@y.T`` for tiny ``||x - y||``."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B, M, d)
    if M_diag is not None:
        sq = (diff * diff) * M_diag
    else:
        sq = diff * diff
    return sq.sum(-1).clamp_min_(0.0)


def _pairwise_sq_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    M_diag: Optional[torch.Tensor] = None,
    *,
    max_direct_bytes: int = 2**32,
) -> torch.Tensor:
    """(B, M) (Mahalanobis-)squared distance matrix.

    With ``M_diag`` (per-coordinate weights), computes
    ``Cxy[i, j] = sum_k M_diag[k] * (x[i, k] - y[j, k])**2``.

    Uses the numerically-stable direct expansion when ``B*M*d`` fits in
    ``max_direct_bytes`` (default 4 GiB), and falls back to the memory-
    cheap outer-product expansion otherwise. The expansion suffers
    catastrophic cancellation when ``x[i] ~= y[j]``; ``clamp_min_(0.0)``
    masks but does not fix this. Float32 noise floor of the expansion is
    ``||x||^2 * 1e-7`` and is the dominant source of the residual gap
    seen at the Sinkhorn convergence floor for large-batch settings.
    """
    B, M = x.shape[0], y.shape[0]
    d = x.shape[1] if x.dim() > 1 else 1
    if B * M * d * x.element_size() <= max_direct_bytes:
        return _pairwise_sq_dist_direct(x, y, M_diag)

    if M_diag is not None:
        wx = x * M_diag
        wy = y * M_diag
        x_sq = (wx * x).sum(-1)
        y_sq = (wy * y).sum(-1)
        Cxy = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2.0 * (x @ wy.t())
    else:
        x_sq = (x * x).sum(-1)
        y_sq = (y * y).sum(-1)
        Cxy = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2.0 * (x @ y.t())
    return Cxy.clamp_min_(0.0)


def monge_gap_hungarian(
    z: torch.Tensor,
    Tz: torch.Tensor,
    M_diag: Optional[torch.Tensor] = None,
) -> dict:
    """Exact Monge gap via Hungarian assignment. Use only for small N (<= 4096).

    With ``M_diag`` set, the gap is measured under the diagonal-Mahalanobis
    cost the per-coordinate weights induce; both ``cost_id`` and the LAP
    objective use the same cost.
    """
    assert z.shape == Tz.shape, f"shape mismatch: {z.shape} vs {Tz.shape}"
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    diff = Tz - z
    if M_diag is not None:
        cost_id = float(((diff * diff) * M_diag).sum(-1).mean().item())
    else:
        cost_id = float((diff * diff).sum(-1).mean().item())
    cost_pair = _pairwise_sq_dist(z, Tz, M_diag=M_diag)
    row, col = linear_sum_assignment(cost_pair.detach().cpu().numpy())
    w2_sq = float(cost_pair[row, col].mean().item())
    gap = max(cost_id - w2_sq, 0.0)  # guard against numerical undershoot
    return {"cost_id": cost_id, "w2_sq": w2_sq, "monge_gap": gap, "backend": "hungarian"}


def _sinkhorn_log_transport_cost(
    C: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    n_iter: int,
    tol: float,
    *,
    check_every: int = 10,
) -> float:
    """Log-domain Sinkhorn iteration. Returns ``<C, P*>`` where P* is the
    entropic-OT plan with marginals ``(exp(log_a), exp(log_b))``, bandwidth
    ``eps``, and (possibly Mahalanobis) cost matrix ``C``.

    This is the numerically-stable form: every update is a logsumexp, so
    we never form ``exp(-C/eps)`` directly (which underflows whenever the
    ratio ``C/eps`` is large — e.g. at ``eps = 1%`` of median sq-dist).

    Convergence is checked every ``check_every`` iterations rather than
    every iteration to amortise the GPU↔CPU sync cost of ``.item()``.
    """
    log_K = -C / eps  # (N, M); intentionally NOT exponentiated
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for it in range(n_iter):
        # u_i: log_u_i = log_a_i - logsumexp_j (log_K_ij + log_v_j)
        log_u_new = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v_new = log_b - torch.logsumexp(log_K + log_u_new.unsqueeze(1), dim=0)
        if (it + 1) % check_every == 0:
            diff_u = (log_u_new - log_u).abs().max()
            diff_v = (log_v_new - log_v).abs().max()
            max_diff = float(torch.max(diff_u, diff_v).item())
            log_u, log_v = log_u_new, log_v_new
            if max_diff < tol:
                break
        else:
            log_u, log_v = log_u_new, log_v_new
    # Plan: P_ij = exp(log_u_i + log_K_ij + log_v_j). Marginals are (a, b).
    # Materializing P at fp32 is fine — values lie in [0, max(a, b)].
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    return float((P * C).sum().item())


def sinkhorn_w2_sq(
    z: torch.Tensor,
    Tz: torch.Tensor,
    eps: float,
    n_iter: int = 1000,
    tol: float = 1e-7,
    M_diag: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Entropic-OT cost and Sinkhorn divergence between two empirical
    distributions of equal size, with uniform weights, under either the
    squared-Euclidean or diagonal-Mahalanobis cost.

    Returns ``(raw_w2_sq_eps, sinkhorn_divergence)``. The latter is the
    bias-corrected Sinkhorn divergence
    ``S_eps(P, Q) = OT_eps(P, Q) - 0.5 * (OT_eps(P, P) + OT_eps(Q, Q))``
    and is the rigorous quantity to report.
    """
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    N = z.shape[0]
    log_a = torch.full((N,), -math.log(N), device=z.device, dtype=z.dtype)
    log_b = log_a.clone()

    def transport_cost(x, y):
        Cxy = _pairwise_sq_dist(x, y, M_diag=M_diag)
        return _sinkhorn_log_transport_cost(Cxy, log_a, log_b, eps, n_iter, tol)

    cost_xy = transport_cost(z, Tz)
    cost_xx = transport_cost(z, z)
    cost_yy = transport_cost(Tz, Tz)
    sinkhorn_div = cost_xy - 0.5 * (cost_xx + cost_yy)
    return cost_xy, max(sinkhorn_div, 0.0)


def monge_gap_sinkhorn(
    z: torch.Tensor,
    Tz: torch.Tensor,
    eps: float | None = None,
    n_iter: int = 1000,
    M_diag: Optional[torch.Tensor] = None,
) -> dict:
    """Monge gap via Sinkhorn. If ``eps`` is None, set it adaptively to
    1% of the median pairwise distance squared on the union of z and Tz.
    The bandwidth is computed under the same cost the OT uses (so when
    ``M_diag`` is set, ``eps`` scales with the Mahalanobis median)."""
    assert z.shape == Tz.shape
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    if eps is None:
        eps = 0.01 * median_pairwise_distance_sq(
            torch.cat([z, Tz], dim=0), M_diag=M_diag
        )
        # Avoid degenerate eps when both clouds collapse onto the same point.
        if eps <= 0.0:
            eps = 1e-6
    diff = Tz - z
    if M_diag is not None:
        cost_id = float(((diff * diff) * M_diag).sum(-1).mean().item())
    else:
        cost_id = float((diff * diff).sum(-1).mean().item())
    raw_w2, sinkdiv = sinkhorn_w2_sq(
        z, Tz, eps=eps, n_iter=n_iter, M_diag=M_diag,
    )
    return {
        "cost_id":          cost_id,
        "w2_sq_raw":        raw_w2,    # entropic, biased: ~ W_2^2 + O(eps)
        "w2_sq_debias":     sinkdiv,   # Sinkhorn divergence, debiased
        "monge_gap_raw":    max(cost_id - raw_w2, 0.0),
        "monge_gap_debias": max(cost_id - sinkdiv, 0.0),
        "epsilon":          eps,
        "backend":          "sinkhorn",
    }


def monge_gap_subsample_hungarian(
    z: torch.Tensor,
    Tz: torch.Tensor,
    M: int = 4096,
    K: int = 10,
    seed: int = 0,
    M_diag: Optional[torch.Tensor] = None,
) -> dict:
    """Robust Monge gap via K Hungarian solves on random subsamples of size M.

    Use this on cells where the push T may collapse or expand the cloud
    (WFR/SDRO at small lambda): adaptive-ε Sinkhorn produces estimator
    artifacts there because ε rescales with ‖Tz − z‖, while subsample-
    Hungarian has no entropic bias and a built-in IQR error bar.

    Returns median ``monge_gap`` across K draws plus quartiles, min/max,
    and the total estimator backend tag. Subsample indices are shared
    between z and Tz so each draw is a coupled empirical pair.
    """
    assert z.shape == Tz.shape
    if z.dim() == 1:
        z = z.view(-1, 1)
        Tz = Tz.view(-1, 1)
    N = z.shape[0]
    if N <= M:
        out = monge_gap_hungarian(z, Tz, M_diag=M_diag)
        out.update({
            "monge_gap_lo": out["monge_gap"], "monge_gap_hi": out["monge_gap"],
            "monge_gap_min": out["monge_gap"], "monge_gap_max": out["monge_gap"],
            "n_subsamples": 1, "subsample_size": N,
            "backend": "hungarian",
        })
        return out

    # Use the device's own generator when on CUDA so the random indices
    # don't trigger an extra CPU↔GPU sync per draw.
    try:
        g = torch.Generator(device=z.device).manual_seed(int(seed))
        on_device_gen = True
    except (RuntimeError, TypeError):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        on_device_gen = False
    gaps, cost_ids, w2s = [], [], []
    for _ in range(K):
        if on_device_gen:
            idx = torch.randperm(N, generator=g, device=z.device)[:M]
        else:
            idx = torch.randperm(N, generator=g)[:M].to(z.device)
        out = monge_gap_hungarian(z[idx], Tz[idx], M_diag=M_diag)
        gaps.append(out["monge_gap"])
        cost_ids.append(out["cost_id"])
        w2s.append(out["w2_sq"])

    gaps_t = torch.tensor(gaps)
    return {
        "cost_id":         float(torch.tensor(cost_ids).median().item()),
        "w2_sq":           float(torch.tensor(w2s).median().item()),
        "monge_gap":       float(gaps_t.median().item()),
        "monge_gap_lo":    float(gaps_t.quantile(0.25).item()),
        "monge_gap_hi":    float(gaps_t.quantile(0.75).item()),
        "monge_gap_min":   float(gaps_t.min().item()),
        "monge_gap_max":   float(gaps_t.max().item()),
        "n_subsamples":    K,
        "subsample_size":  M,
        "backend":         "hungarian_subsample",
    }


def monge_gap(
    z: torch.Tensor,
    Tz: torch.Tensor,
    batch_size: int = 4096,
    M_diag: Optional[torch.Tensor] = None,
) -> dict:
    """Top-level convenience: choose the backend by batch size."""
    if batch_size <= 4096:
        return monge_gap_hungarian(z, Tz, M_diag=M_diag)
    return monge_gap_sinkhorn(z, Tz, M_diag=M_diag)
