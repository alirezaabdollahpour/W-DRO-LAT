"""NPF-ICNN WDRO adversary (Vesseron & Cuturi, 2024) with persistent Adam inner loop."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from config import ULSConfig
from models.npf import NPFResidualPotential
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta
from utils.transport import transport_map


def _solve_npf_icnn_map_with_zstar(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, Any]:
    device = xi_train.device
    theta = torch.zeros(cfg.dim_n, device=device)

    psi = NPFResidualPotential(
        input_dim=1,
        hidden_sizes=tuple(cfg.npf_hidden),
        outer_rank=cfg.npf_outer_rank,
        inner_rank=cfg.npf_inner_rank,
        activation=cfg.npf_activation,
        elu_alpha=cfg.npf_elu_alpha,
        softplus_beta=cfg.npf_softplus_beta,
        init_eps=cfg.npf_init_eps,
    ).to(device).to(xi_train.dtype)

    inner_opt = torch.optim.Adam(psi.parameters(), lr=cfg.inner_lr_npf)
    xi_2d = xi_train.view(-1, 1)

    diagnostics: Dict[str, list] = {
        "outer_P": [],
        "adv_loss": [],
        "mean_displacement": [],
        "max_displacement": [],
    }
    z_adv = xi_train.detach().clone()

    for _epoch in range(cfg.epochs):
        # Inner ascent on psi (Adam minimises the negated objective).
        for _ in range(cfg.npf_omega_steps_per_epoch):
            inner_opt.zero_grad()
            z_adv = transport_map(xi_2d, psi, create_graph=True).view(-1)
            f = loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
            reg = cfg.lam * (z_adv - xi_train) ** 2
            obj = (f - reg).mean()
            (-obj).backward()
            inner_opt.step()

        # Outer update on theta using the final clamped adversary.
        with torch.no_grad():
            z_adv_raw = transport_map(xi_2d, psi, create_graph=False).view(-1).detach()
            z_adv = z_adv_raw.clamp(-1.0, 1.0)
        grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
        grad_theta = clip_grad(grad_theta, cfg.grad_clip)
        theta = theta - cfg.lr_theta * grad_theta

        with torch.no_grad():
            diagnostics["outer_P"].append(float(psi.outer_P().item()))
            disp = (z_adv_raw - xi_train).abs()
            diagnostics["mean_displacement"].append(float(disp.mean().item()))
            diagnostics["max_displacement"].append(float(disp.max().item()))
            diagnostics["adv_loss"].append(
                float((loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                       - cfg.lam * (z_adv - xi_train) ** 2).mean().item())
            )

    return {
        "theta": theta,
        "psi": psi,
        "diagnostics": diagnostics,
        "z_star": z_adv.detach(),
        "z_star_kind": "paired",
    }


def solve_npf_icnn_map(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, NPFResidualPotential, Dict[str, list]]:
    out = _solve_npf_icnn_map_with_zstar(xi_train, cfg, A0, A1, b)
    return out["theta"], out["psi"], out["diagnostics"]
