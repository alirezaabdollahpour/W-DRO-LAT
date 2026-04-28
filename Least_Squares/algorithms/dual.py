"""Dual (Wang et al., 2021) Sinkhorn / MLMC adversary."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta


def _solve_dual_with_zstar(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, Any]:
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    levels = torch.arange(cfg.sinkhorn_sample_level + 1, device=xi_train.device)
    numerators = 2.0 ** (-levels.to(torch.float64))
    denominator = 2.0 - 2.0 ** (-float(cfg.sinkhorn_sample_level))
    probs = (numerators / denominator).to(torch.float64)
    z_samples = xi_train.view(-1, 1)

    for _epoch in range(cfg.epochs):
        sampled_level = int(torch.multinomial(probs, num_samples=1).item())
        m = 2 ** sampled_level

        noise = torch.randn((n_train, m), device=xi_train.device) * math.sqrt(cfg.epsilon)
        z_samples = xi_train.view(-1, 1) + noise

        v = loss_function(theta, z_samples.reshape(-1), A0, A1, b, cfg.dim_m).view(n_train, m)
        v = v / (cfg.lam * cfg.epsilon)
        v_max = torch.max(v, dim=1, keepdim=True).values
        w = torch.exp(v - v_max)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        grads = loss_grad_theta(theta, z_samples.reshape(-1), A0, A1, b).view(
            n_train, m, cfg.dim_n
        )
        grads = (w.unsqueeze(-1) * grads).sum(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return {
        "theta": theta,
        "z_star": z_samples.reshape(-1).detach(),
        "z_star_kind": "cloud",
    }


def solve_dual(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _solve_dual_with_zstar(xi_train, cfg, A0, A1, b)["theta"]
