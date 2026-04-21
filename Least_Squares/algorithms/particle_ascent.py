"""Particle Ascent: inner gradient ascent on the Lagrangian adversary."""
from __future__ import annotations

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_grad_theta, loss_grad_xi


def solve_Particle_Ascent(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    for _epoch in range(cfg.epochs):
        z = xi_train.clone()
        for _ in range(cfg.inner_steps):
            grad = loss_grad_xi(theta, z, A0, A1, b) - 2.0 * cfg.lam * (z - xi_train)
            z = z + cfg.inner_step_size * grad
            z = z.clamp(-1.0, 1.0)
        grads_theta = loss_grad_theta(theta, z, A0, A1, b)
        avg_grad = grads_theta.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta
