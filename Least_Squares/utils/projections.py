"""1D Brenier projection via monotone rearrangement (used by PPA)."""
from __future__ import annotations

from typing import Tuple

import torch


def brenier_projection_1d(
    z: torch.Tensor,
    xi_train: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """Monotone rearrangement of z onto the rank order of xi_train.

    Returns
    -------
    z_proj : Tensor [n]  -- reassigned z (same multiset, new ordering)
    delta  : float       -- per-sample wasted-transport gap (C_id - C_ot)
    C_id   : float       -- per-sample identity coupling cost
    """
    n = z.numel()
    if n <= 1:
        C_id = float(((z - xi_train) ** 2).sum().item()) / max(n, 1)
        return z.clone(), 0.0, C_id

    z_sorted, _ = torch.sort(z)
    xi_ranks = torch.argsort(torch.argsort(xi_train))
    z_proj = z_sorted[xi_ranks]

    C_id = float(((z - xi_train) ** 2).mean().item())
    C_ot = float(((z_proj - xi_train) ** 2).mean().item())
    delta = max(C_id - C_ot, 0.0)
    return z_proj, delta, C_id
