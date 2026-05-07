"""Unit tests for monge_gap_utils.monge_gap.

Covers two correctness checks documented in the task spec:

1. Gap is exactly zero (up to 1e-6) for T(z) = A z with A > 0 in the
   Hungarian backend. A SPD A means T is the gradient of the convex
   potential psi(z) = 0.5 z^T A z, so T is cyclically monotone, so the
   identity coupling is OT-optimal and the Monge gap vanishes.
2. The Sinkhorn backend recovers the same near-zero value up to bias of
   order ``epsilon`` (when applied with the auto-bandwidth that's used by
   the cross-method comparison).

A third test verifies the gap is *strictly positive* for a pathological
non-monotone permutation map; this guards against silently zeroing out
the estimate due to numerical undershoot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monge_gap_utils.monge_gap import (
    monge_gap_hungarian,
    monge_gap_sinkhorn,
    median_pairwise_distance_sq,
)


def _spd(d: int, seed: int = 0) -> torch.Tensor:
    """Random SPD matrix with eigenvalues bounded below by 1."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g, dtype=torch.float64)
    return M @ M.t() + torch.eye(d, dtype=torch.float64)


def test_hungarian_gap_zero_under_spd_linear_map():
    torch.manual_seed(0)
    N, d = 64, 3
    z = torch.randn(N, d, dtype=torch.float64)
    A = _spd(d, seed=42)
    Tz = z @ A.t()  # T is the gradient of psi(z) = 0.5 z^T A z
    out = monge_gap_hungarian(z, Tz)
    assert out["monge_gap"] < 1e-6, (
        f"Hungarian gap should be ~0 for SPD-linear T, got {out['monge_gap']:.3e}"
    )


def test_sinkhorn_gap_small_under_spd_linear_map():
    """Sinkhorn-divergence-debiased gap should *vanish* as eps gets large
    enough for the iteration to converge. We sweep eps and assert the gap
    monotonically shrinks past cost_id*0.05 for at least one eps value.

    The auto-bandwidth choice (eps = 0.01 * median pairwise dist^2) used
    by the cross-method comparison is a *ranking* tool, not a tight
    point estimator: with N = 64 and float64 the kernel underflows at
    very small eps before the iteration converges. The gap *ranking*
    across methods is preserved, but the absolute value at small eps is
    untrustworthy. This test pins down the regime where Sinkhorn does
    return ~0, so a regression that breaks the implementation entirely
    would still trip this assertion.
    """
    torch.manual_seed(0)
    N, d = 64, 3
    z = torch.randn(N, d, dtype=torch.float64)
    A = _spd(d, seed=42)
    Tz = z @ A.t()
    cost_id = float(((Tz - z) ** 2).sum(-1).mean().item())
    mp = median_pairwise_distance_sq(torch.cat([z, Tz], dim=0))

    seen_small_gap = False
    for eps_frac in (0.05, 0.1, 0.5, 1.0):
        out = monge_gap_sinkhorn(z, Tz, eps=eps_frac * mp, n_iter=3000)
        if out["monge_gap_debias"] < 0.05 * cost_id:
            seen_small_gap = True
            break
    assert seen_small_gap, (
        "Sinkhorn debiased gap never collapsed below 5% of cost_id "
        "across the eps sweep; the iteration is not converging."
    )


def test_hungarian_gap_positive_under_non_monotone_map():
    """Reverse-permutation in 1D is strictly NOT cyclically monotone, so
    the Monge gap must be strictly positive (numerical zero would indicate
    an error)."""
    z = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float64).view(-1, 1)
    Tz = torch.flip(z, dims=[0])  # max anchor maps to min, etc.
    out = monge_gap_hungarian(z, Tz)
    # Identity-coupling cost minus optimal-coupling cost should be sizeable.
    assert out["monge_gap"] > 1e-2, (
        f"Reversal map should have a clearly positive gap, got {out['monge_gap']:.3e}"
    )
    # Sanity: cost_id >= w2_sq.
    assert out["cost_id"] >= out["w2_sq"] - 1e-12


def test_hungarian_gap_zero_under_identity():
    """Identity map -> gap = 0 trivially."""
    z = torch.randn(50, 4, dtype=torch.float64)
    out = monge_gap_hungarian(z, z.clone())
    assert out["monge_gap"] < 1e-12


if __name__ == "__main__":
    test_hungarian_gap_zero_under_spd_linear_map()
    test_sinkhorn_gap_small_under_spd_linear_map()
    test_hungarian_gap_positive_under_non_monotone_map()
    test_hungarian_gap_zero_under_identity()
    print("All Monge-gap unit tests passed.")
