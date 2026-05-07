"""Madry PGD-L2 adversarial training baseline."""
from __future__ import annotations

import torch

from config import ULSConfig
from utils.common import clip_grad
from utils.loss import loss_grad_theta
from utils.pgd import pgd_attack_l2


def solve_madry_pgd(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    xi_train = xi_train.to(device=A0.device, dtype=A0.dtype)
    theta = torch.zeros(cfg.dim_n, device=A0.device, dtype=A0.dtype)

    for _epoch in range(cfg.epochs):
        z_adv = pgd_attack_l2(
            theta, xi_train, cfg.pgd_epsilon, cfg.pgd_steps, cfg.pgd_restarts,
            A0, A1, b, cfg.dim_m,
        )
        with torch.no_grad():
            grads_theta = loss_grad_theta(theta, z_adv, A0, A1, b)
            avg_grad = grads_theta.mean(dim=0)
            avg_grad = clip_grad(avg_grad, cfg.grad_clip)
            theta = theta - cfg.lr_theta * avg_grad

    return theta
