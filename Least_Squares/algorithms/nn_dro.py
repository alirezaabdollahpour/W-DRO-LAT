"""NN-DRO: vanilla MLP inner maximiser for WDRO.

Same outer loop as ICNN/NPF, but the adversary is a plain MLP applied to xi
directly — no gradient-of-potential transport, so no autograd through the
network's input.
"""
from __future__ import annotations

import torch

from config import ULSConfig
from models.nn_dro import MLPAdversary
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta


def solve_nn_dro(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    device = xi_train.device
    theta = torch.zeros(cfg.dim_n, device=device)

    adversary = MLPAdversary(
        input_dim=1,
        hidden_sizes=tuple(cfg.nn_dro_hidden),
        activation=cfg.nn_dro_activation,
        softplus_beta=cfg.nn_dro_softplus_beta,
        init_scale=cfg.nn_dro_init_scale,
    ).to(device).to(xi_train.dtype)

    inner_opt = torch.optim.Adam(adversary.parameters(), lr=cfg.nn_dro_inner_lr)
    xi_2d = xi_train.view(-1, 1)

    for _epoch in range(cfg.epochs):
        for _ in range(cfg.nn_dro_omega_steps_per_epoch):
            inner_opt.zero_grad()
            z_adv = adversary(xi_2d).view(-1)
            f = loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
            reg = cfg.lam * (z_adv - xi_train) ** 2
            obj = (f - reg).mean()
            (-obj).backward()
            inner_opt.step()

        with torch.no_grad():
            z_adv = adversary(xi_2d).view(-1).clamp(-1.0, 1.0)
            grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
            grad_theta = clip_grad(grad_theta, cfg.grad_clip)
            theta = theta - cfg.lr_theta * grad_theta

    return theta
