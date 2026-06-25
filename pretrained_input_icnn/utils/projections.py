"""Free-weight (within-class best-response) projection adapted for image inputs.

Mirrors ``MNIST_Cuturi.utils.projections.free_weight_projection`` but handles
4-D image tensors (B, C, H, W) by flattening internally for the LAP cost. The
output preserves the original spatial layout via ``view_as(z)``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .adversary_loss import adversary_loss_per_sample
from .transforms import to_pixel


def free_weight_projection_images(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    classifier: nn.Module,
    lambda_reg: float,
    use_margin: bool = False,
    transport_cost: str = "normalized_mse",
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """For each sample i, pick j in the same class maximising
       primary(model(z_j), y_i) - λ c(z_j, x_i),

    where ``primary`` is CE by default and the log-sum-exp margin when
    ``use_margin=True``. The default cost is the legacy input-ICNN
    normalized-coordinate MSE; ``pixel_l2_squared`` preserves the older
    refactor's pixel-coordinate squared L2 projection score.

    Returns (z_proj, y_proj, gain_mean, obj_scale_mean, active_support_frac).
    """
    N = z.size(0)
    if N <= 1:
        return z.clone(), y.clone(), 0.0, 1.0, 1.0

    with torch.no_grad():
        logits = classifier(z)
        candidate_loss = adversary_loss_per_sample(logits, y, use_margin=use_margin)

    mode = str(transport_cost).lower()
    if mode in {"normalized_mse", "normalized", "legacy"}:
        z_flat = z.detach().view(N, -1)
        x_flat = x.detach().view(N, -1)
        cost_scale = 1.0 / float(z_flat.size(1))
    elif mode in {"pixel_l2_squared", "pixel_l2", "pixel"}:
        z_flat = to_pixel(z.detach()).view(N, -1)
        x_flat = to_pixel(x.detach()).view(N, -1)
        cost_scale = 1.0
    else:
        raise ValueError(f"Unsupported transport_cost mode: {transport_cost}")
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
        dist_c = (x_sq + z_sq.t() - 2.0 * cross).clamp(min=0.0) * cost_scale

        scores_c = loss_c.unsqueeze(0) - lambda_reg * dist_c
        best_cols = scores_c.argmax(dim=1)
        chosen_idx = idx.index_select(0, best_cols)

        z_proj.index_copy_(0, idx, z.index_select(0, chosen_idx))
        selected_counts.scatter_add_(
            0, chosen_idx, torch.ones_like(chosen_idx, dtype=torch.long)
        )

        old_scores = loss_c - lambda_reg * ((z_c - x_c) ** 2).sum(dim=1) * cost_scale
        best_scores = scores_c.gather(1, best_cols.view(-1, 1)).squeeze(1)
        total_gain += float((best_scores - old_scores).sum().item())
        total_scale += float(old_scores.abs().sum().item())

    gain_mean = total_gain / max(N, 1)
    obj_scale_mean = total_scale / max(N, 1)
    active_support_frac = float((selected_counts > 0).float().mean().item())
    return z_proj, y_proj, gain_mean, obj_scale_mean, active_support_frac
