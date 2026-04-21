"""Within-class Brenier projection for feature-space vectors (used by PPA)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def brenier_projection_features(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    r"""Within-class Brenier projection.

    Optimally reassigns adversarial points to nominals **within each class**
    by solving one Hungarian LAP per class on the squared-Euclidean cost.

    Parameters
    ----------
    z : [N, d]  adversarial feature points
    x : [N, d]  nominal feature points
    y : [N]     labels (permuted alongside z within class)

    Returns
    -------
    z_proj  : [N, d]   optimally reassigned within each class
    y_proj  : [N]      labels preserved within class (== y)
    delta   : float    wasted-transport gap (C_id - C_ot)
    C_id    : float    identity transport cost
    C_ot    : float    optimal transport cost
    """
    from scipy.optimize import linear_sum_assignment

    N = z.size(0)
    if N <= 1:
        C_id = float(((z - x) ** 2).sum().item()) / max(N, 1)
        return z.clone(), y.clone(), 0.0, C_id, C_id

    z_flat = z.detach()
    x_flat = x.detach()

    perm_np = np.arange(N, dtype=np.int64)
    y_np = y.cpu().numpy()
    unique_classes = np.unique(y_np)

    for c in unique_classes:
        idx_c = np.where(y_np == c)[0]
        n_c = len(idx_c)
        if n_c <= 1:
            continue

        x_c = x_flat[idx_c].cpu().float()
        z_c = z_flat[idx_c].cpu().float()

        x_sq = (x_c ** 2).sum(dim=1, keepdim=True)
        z_sq = (z_c ** 2).sum(dim=1, keepdim=True)
        cross = x_c.mm(z_c.t())
        cost_c = (x_sq + z_sq.t() - 2.0 * cross).clamp(min=0.0)
        cost_c_np = cost_c.numpy()

        row_ind, col_ind = linear_sum_assignment(cost_c_np)
        local_perm = np.empty(n_c, dtype=np.int64)
        local_perm[row_ind] = col_ind
        perm_np[idx_c] = idx_c[local_perm]

    perm = torch.tensor(perm_np, device=z.device, dtype=torch.long)

    z_proj = z[perm]
    y_proj = y[perm]

    C_id = float(((z - x) ** 2).sum(dim=1).mean().item())
    C_ot = float(((z_proj - x) ** 2).sum(dim=1).mean().item())
    delta = max(C_id - C_ot, 0.0)

    return z_proj, y_proj, delta, C_id, C_ot
