"""Wasserstein-Fisher-Rao (Xu, 2025) adversary."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta, loss_grad_xi


def _solve_wfr_with_zstar(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, Any]:
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)
    particles = xi_train.view(-1, 1).repeat(1, m)
    weights = torch.full((n_train, m), 1.0 / float(m), device=xi_train.device)

    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)
        weights = torch.full((n_train, m), 1.0 / float(m), device=xi_train.device)

        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (
                particles_flat - xi_expanded
            )
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)

            f_bar = loss_function(theta, particles.reshape(-1), A0, A1, b, cfg.dim_m).view(
                n_train, m
            )
            f_bar = f_bar - cfg.lam * (particles - xi_train.view(-1, 1)) ** 2

            power = 1.0 - cfg.lam * cfg.epsilon * cfg.wfr_weight_step_size
            weights = (weights.clamp_min(1e-12) ** power) * torch.exp(
                cfg.wfr_weight_step_size * f_bar
            )
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

            thr = cfg.wfr_low_weight_threshold
            low_mask = weights < thr
            rows = torch.any(low_mask, dim=1)
            if torch.any(rows):
                for i in torch.nonzero(rows, as_tuple=False).view(-1).tolist():
                    w = weights[i]
                    x = particles[i]
                    low = low_mask[i]
                    if not torch.any(low):
                        continue
                    j_max = int(torch.argmax(w).item())
                    w_max = w[j_max]
                    x_max = x[j_max].clone()
                    low_sum = torch.sum(w * low)
                    n_low = torch.sum(low).to(w.dtype)
                    avg_w = (w_max + low_sum) / (n_low + 1.0 + 1e-12)
                    update_mask = low.clone()
                    update_mask[j_max] = True
                    w = torch.where(update_mask, avg_w, w)
                    x = torch.where(low, x_max, x)
                    w = w / (w.sum() + 1e-12)
                    weights[i] = w
                    particles[i] = x

        grads = loss_grad_theta(theta, particles.reshape(-1), A0, A1, b).view(
            n_train, m, cfg.dim_n
        )
        grads = (weights.unsqueeze(-1) * grads).sum(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return {
        "theta": theta,
        "z_star": particles.reshape(-1).detach(),
        "z_star_kind": "cloud",
    }


def solve_wfr(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _solve_wfr_with_zstar(xi_train, cfg, A0, A1, b)["theta"]
