"""Projected-gradient-descent L2 attack used by the Madry baseline."""
from __future__ import annotations

import torch

from utils.loss import loss_function


def pgd_attack_l2(
    theta: torch.Tensor,
    xi_nominal: torch.Tensor,
    epsilon: float,
    pgd_steps: int,
    pgd_restarts: int,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
    dim_m: int,
) -> torch.Tensor:
    if epsilon <= 0.0 or pgd_steps <= 0 or pgd_restarts <= 0:
        return xi_nominal.clamp(-1.0, 1.0).detach()
    step_size = 2.5 * epsilon / float(pgd_steps)
    best_z = xi_nominal.clone()
    best_loss = loss_function(theta, xi_nominal, A0, A1, b, dim_m)

    for _ in range(pgd_restarts):
        delta = torch.empty_like(xi_nominal).uniform_(-1.0, 1.0) * epsilon
        z = (xi_nominal + delta).clamp(-1.0, 1.0)

        for _ in range(pgd_steps):
            z_var = z.clone().detach().requires_grad_(True)
            l = loss_function(theta, z_var, A0, A1, b, dim_m)
            grad_z = torch.autograd.grad(l.sum(), z_var)[0]
            grad_norm = grad_z.abs().clamp_min(1e-12)
            z = z.detach() + step_size * grad_z / grad_norm
            delta_vec = z - xi_nominal
            scale = torch.clamp(epsilon / delta_vec.abs().clamp_min(1e-12), max=1.0)
            z = (xi_nominal + delta_vec * scale).clamp(-1.0, 1.0)

        with torch.no_grad():
            current_loss = loss_function(theta, z, A0, A1, b, dim_m)
        improved = current_loss > best_loss
        best_z = torch.where(improved, z, best_z)
        best_loss = torch.where(improved, current_loss, best_loss)

    return best_z.detach()
