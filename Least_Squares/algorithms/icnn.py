"""ICNN (Amos et al.) WDRO adversary with BB+Armijo inner maximisation."""
from __future__ import annotations

from typing import Tuple

import torch

from config import ULSConfig
from models.icnn import InputConvexPotential
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta
from utils.transport import transport_map


def solve_icnn_map(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, InputConvexPotential]:
    device = xi_train.device
    theta = torch.zeros(cfg.dim_n, device=device)
    psi_omega = InputConvexPotential(
        input_dim=1,
        hidden_sizes=tuple(cfg.icnn_hidden),
        activation="softplus",
        strong_convexity=cfg.icnn_strong_convexity,
        nonneg_init="principled",
        softplus_beta=cfg.icnn_softplus_beta,
    ).to(device).to(xi_train.dtype)

    bb_state = BBArmijoState.create(
        alpha0=cfg.icnn_bb_alpha0,
        alpha_min=cfg.icnn_bb_alpha_min,
        alpha_max=cfg.icnn_bb_alpha_max,
        ls_c=cfg.icnn_bb_ls_c,
        ls_shrink=cfg.icnn_bb_ls_shrink,
        ls_max_steps=cfg.icnn_bb_ls_max_steps,
    )

    xi_2d = xi_train.view(-1, 1)

    for _epoch in range(cfg.epochs):
        for _ in range(cfg.icnn_omega_steps_per_epoch):
            def omega_objective(create_graph: bool) -> torch.Tensor:
                z_adv = transport_map(xi_2d, psi_omega, create_graph=create_graph).view(-1)
                z_adv = z_adv.clamp(-1.0, 1.0)
                f = loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                reg = cfg.lam * (z_adv - xi_train) ** 2
                return (f - reg).mean()

            _, bb_state, _, _ = bb_armijo_step_params(
                psi_omega.parameters(), omega_objective, bb_state
            )

        with torch.no_grad():
            z_adv = transport_map(xi_2d, psi_omega, create_graph=False).view(-1)
            z_adv = z_adv.clamp(-1.0, 1.0)
            grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
            grad_theta = clip_grad(grad_theta, cfg.grad_clip)
            theta = theta - cfg.lr_theta * grad_theta

    return theta, psi_omega
