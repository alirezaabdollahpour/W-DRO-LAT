"""Projected Particle Ascent (PPA): 1D Brenier projection interleaved with ascent."""
from __future__ import annotations

from typing import Any, Dict

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_grad_theta, loss_grad_xi
from utils.projections import brenier_projection_1d


def _solve_ppa_with_zstar(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, Any]:
    xi_train = xi_train.to(device=A0.device, dtype=A0.dtype)
    theta = torch.zeros(cfg.dim_n, device=A0.device, dtype=A0.dtype)
    z = xi_train.clone()

    for _epoch in range(cfg.epochs):
        # Round 0: replicate Particle Ascent exactly.
        z = xi_train.clone()
        for _ in range(cfg.inner_steps):
            grad = loss_grad_xi(theta, z, A0, A1, b) - 2.0 * cfg.lam * (z - xi_train)
            z = z + cfg.inner_step_size * grad
            z = z.clamp(-1.0, 1.0)

        # Rounds 1..R-1: project then refine.
        for round_idx in range(1, cfg.ppa_num_rounds):
            z, delta, C_id = brenier_projection_1d(z, xi_train)
            if (round_idx >= cfg.ppa_min_rounds
                    and delta < cfg.ppa_delta_rtol * max(C_id, 1e-12)):
                break
            for _ in range(cfg.ppa_refine_steps):
                grad = loss_grad_xi(theta, z, A0, A1, b) - 2.0 * cfg.lam * (z - xi_train)
                z = z + cfg.ppa_refine_lr * grad
                z = z.clamp(-1.0, 1.0)

        z, _, _ = brenier_projection_1d(z, xi_train)

        grads_theta = loss_grad_theta(theta, z, A0, A1, b)
        avg_grad = grads_theta.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return {"theta": theta, "z_star": z.detach(), "z_star_kind": "paired"}


def solve_ppa(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _solve_ppa_with_zstar(xi_train, cfg, A0, A1, b)["theta"]
