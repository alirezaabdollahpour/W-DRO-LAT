"""Within-class Brenier projection + free-weight best-response projection."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def brenier_projection(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Per-class LAP (optimal reassignment restricted to samples of the same class).

    Returns (z_proj, y_proj, delta, C_id, C_ot).
    """
    N = z.size(0)
    if N <= 1:
        C_id = float(((z - x) ** 2).sum().item()) / max(N, 1)
        return z.clone(), y.clone(), 0.0, C_id, C_id

    z_flat = z.detach().view(N, -1)
    x_flat = x.detach().view(N, -1)

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

    C_id = float(((z - x) ** 2).view(N, -1).sum(dim=1).mean().item())
    C_ot = float(((z_proj - x) ** 2).view(N, -1).sum(dim=1).mean().item())
    delta = max(C_id - C_ot, 0.0)

    return z_proj, y_proj, delta, C_id, C_ot


def free_weight_projection(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    lambda_reg: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Within-class support-restricted best response for New_PPA.

    For each sample i, pick j in the same class maximising
        CE(model(z_j), y_i) - λ ||z_j - x_i||^2.
    Returns (z_proj, y_proj, gain_mean, obj_scale_mean, active_support_frac).
    """
    N = z.size(0)
    if N <= 1:
        return z.clone(), y.clone(), 0.0, 1.0, 1.0

    with torch.no_grad():
        logits = model(z)
        candidate_loss = F.cross_entropy(logits, y, reduction="none")

    z_flat = z.detach().view(N, -1)
    x_flat = x.detach().view(N, -1)
    z_proj = torch.empty_like(z)
    y_proj = y.clone()

    total_gain = 0.0
    total_scale = 0.0
    selected_counts = torch.zeros(N, device=z.device, dtype=torch.long)

    for cls in torch.unique(y, sorted=True):
        idx = (y == cls).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue

        x_c = x_flat.index_select(0, idx)
        z_c = z_flat.index_select(0, idx)
        loss_c = candidate_loss.index_select(0, idx)

        x_sq = (x_c ** 2).sum(dim=1, keepdim=True)
        z_sq = (z_c ** 2).sum(dim=1, keepdim=True)
        cross = x_c.mm(z_c.t())
        dist_c = (x_sq + z_sq.t() - 2.0 * cross).clamp(min=0.0)

        scores_c = loss_c.unsqueeze(0) - lambda_reg * dist_c
        best_cols = scores_c.argmax(dim=1)
        chosen_idx = idx.index_select(0, best_cols)

        z_proj.index_copy_(0, idx, z.index_select(0, chosen_idx))
        selected_counts.scatter_add_(
            0, chosen_idx, torch.ones_like(chosen_idx, dtype=torch.long)
        )

        old_scores = loss_c - lambda_reg * ((z_c - x_c) ** 2).sum(dim=1)
        best_scores = scores_c.gather(1, best_cols.view(-1, 1)).squeeze(1)
        total_gain += float((best_scores - old_scores).sum().item())
        total_scale += float(old_scores.abs().sum().item())

    gain_mean = total_gain / max(N, 1)
    obj_scale_mean = total_scale / max(N, 1)
    active_support_frac = float((selected_counts > 0).float().mean().item())

    return z_proj, y_proj, gain_mean, obj_scale_mean, active_support_frac
