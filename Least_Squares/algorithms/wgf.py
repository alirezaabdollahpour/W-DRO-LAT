"""Wasserstein Gradient Flow (Otto, 1996) adversary."""
from __future__ import annotations

import math

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_grad_theta, loss_grad_xi


def solve_wgf(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)

    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)
        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (
                particles_flat - xi_expanded
            )
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)
        particles_flat = particles.reshape(-1)
        grads = loss_grad_theta(theta, particles_flat, A0, A1, b)
        grads = grads.view(n_train, m, cfg.dim_n).mean(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta
