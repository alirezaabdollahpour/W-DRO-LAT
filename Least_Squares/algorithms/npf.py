"""NPF-style ICNN WDRO adversary for uncertain least squares.

This mirrors the RL implementation at the algorithmic level:

* the NPF potential is initialised as an identity transport map using a fixed
  strong-convexity base plus eps-scale learnable PSD residuals;
* inner maximisation uses BB+Armijo on the potential parameters;
* because least-squares uncertainty is supported on [-1, 1], the NPF map is
  applied in an unconstrained latent coordinate and decoded with a sigmoid
  box bijection, rather than clamping the transport output.

The last point is the important setting-specific difference from CIFAR-10.
The inner objective sees the same bounded adversarial sample as the outer
theta update, but gradients are not killed by a hard clamp.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from config import ULSConfig
from models.npf import NPFInputConvexPotential
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.common import clip_grad
from utils.loss import loss_function, loss_grad_theta


_XI_LOW = -1.0
_XI_HIGH = 1.0
_BOX_EPS = 1e-6


def _encode_box(z: torch.Tensor) -> torch.Tensor:
    p = (z - _XI_LOW) / (_XI_HIGH - _XI_LOW)
    p = torch.clamp(p, _BOX_EPS, 1.0 - _BOX_EPS)
    return torch.log(p) - torch.log1p(-p)


def _decode_box(u: torch.Tensor) -> torch.Tensor:
    return _XI_LOW + (_XI_HIGH - _XI_LOW) * torch.sigmoid(u)


def _npf_box_transport(
    xi: torch.Tensor,
    psi: NPFInputConvexPotential,
    create_graph: bool,
) -> torch.Tensor:
    """Bounded map T(xi)=decode(grad_u psi(encode(xi)))."""
    u_hat = _encode_box(xi)
    u_in = u_hat.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        phi = psi(u_in)
        u_adv = torch.autograd.grad(phi.sum(), u_in, create_graph=create_graph)[0]
    u_adv = torch.nan_to_num(u_adv, nan=0.0, posinf=0.0, neginf=0.0)
    z_adv = _decode_box(u_adv).view_as(xi)
    z_adv = torch.where(torch.isfinite(z_adv), z_adv, xi)
    if not create_graph:
        z_adv = z_adv.detach()
    return z_adv


def _solve_npf_icnn_map_with_zstar(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Dict[str, Any]:
    device = xi_train.device
    xi_train = xi_train.to(device=device, dtype=A0.dtype)
    theta = torch.zeros(cfg.dim_n, device=device, dtype=A0.dtype)

    strong_convexity = float(
        getattr(cfg, "npf_strong_convexity", getattr(cfg, "icnn_strong_convexity", 1.0))
    )
    psi = NPFInputConvexPotential(
        input_dim=1,
        hidden_sizes=tuple(cfg.npf_hidden),
        outer_rank=cfg.npf_outer_rank,
        inner_rank=cfg.npf_inner_rank,
        activation=cfg.npf_activation,
        elu_alpha=cfg.npf_elu_alpha,
        softplus_beta=cfg.npf_softplus_beta,
        init_eps=cfg.npf_init_eps,
        strong_convexity=strong_convexity,
    ).to(device).to(A0.dtype)
    psi.init_as_identity()

    bb_state = BBArmijoState.create(
        alpha0=cfg.npf_bb_alpha0,
        alpha_min=cfg.npf_bb_alpha_min,
        alpha_max=cfg.npf_bb_alpha_max,
        ls_c=cfg.npf_bb_ls_c,
        ls_shrink=cfg.npf_bb_ls_shrink,
        ls_max_steps=cfg.npf_bb_ls_max_steps,
        reject_on_armijo_failure=True,
    )

    xi_2d = xi_train.view(-1, 1)
    diagnostics: Dict[str, list] = {
        "outer_P": [],
        "adv_loss": [],
        "mean_displacement": [],
        "max_displacement": [],
    }
    z_adv = xi_train.detach().clone()

    for _epoch in range(cfg.epochs):
        for _ in range(cfg.npf_omega_steps_per_epoch):

            def omega_objective(create_graph: bool) -> torch.Tensor:
                z_a = _npf_box_transport(xi_2d, psi, create_graph=create_graph).view(-1)
                f = loss_function(theta, z_a, A0, A1, b, cfg.dim_m)
                reg = cfg.lam * (z_a - xi_train) ** 2
                obj = (f - reg).mean()
                return torch.nan_to_num(obj, nan=-1e12, posinf=-1e12, neginf=-1e12)

            _, bb_state, _, _ = bb_armijo_step_params(
                psi.parameters(), omega_objective, bb_state
            )

        with torch.no_grad():
            z_adv = _npf_box_transport(xi_2d, psi, create_graph=False).view(-1)
            grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
            grad_theta = clip_grad(grad_theta, cfg.grad_clip)
            theta = theta - cfg.lr_theta * grad_theta

        with torch.no_grad():
            disp = (z_adv - xi_train).abs()
            outer_P = psi.outer_P()
            diagnostics["outer_P"].append(float(outer_P.reshape(-1)[0].item()))
            diagnostics["mean_displacement"].append(float(disp.mean().item()))
            diagnostics["max_displacement"].append(float(disp.max().item()))
            diagnostics["adv_loss"].append(
                float(
                    (
                        loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                        - cfg.lam * (z_adv - xi_train) ** 2
                    ).mean().item()
                )
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
) -> Tuple[torch.Tensor, NPFInputConvexPotential, Dict[str, list]]:
    out = _solve_npf_icnn_map_with_zstar(xi_train, cfg, A0, A1, b)
    return out["theta"], out["psi"], out["diagnostics"]
