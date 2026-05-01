"""Given a trained DRO state, evaluate the learned adversary on a held-out
anchor batch. Returns the pushforward T(z) of shape (B, d).

Each method's wrapper handles its own access pattern:
  - erm:            identity map T(z) = z. By convention only.
  - npf, nn_dro:    weight-parametric T_omega(z), called directly.
  - particle_ascent, ppa:
                    rerun the inner-loop solver from anchor z with theta
                    frozen, return the converged particle as T(z).
  - wfr, dual:      same as particle, return the *mean* of the auxiliary
                    cloud as T(z) (rationale: WFR/SDRO produce a cloud, but
                    the Monge map between empirical measures requires a
                    deterministic map; the mean is the canonical reduction).

Note: the codebase key ``npf`` is the same method called ICNN-DRO in the
paper. The display name is set in monge_gap_plot.py, not here.

Each wrapper takes a ``state`` dict that contains at minimum:
    - ``method``: the codebase key
    - ``theta`` (or model params for LR/RL): the trained outer parameter
    - method-specific extras (psi, adversary, cfg, A0/A1/b, ...)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _ensure_ls_on_path() -> None:
    """The Least_Squares folder is not a package; its modules import each
    other as flat top-level names (``from utils.loss import ...``). Push the
    folder onto sys.path so callers can import its helpers without errors.
    """
    ls_root = Path(__file__).resolve().parent.parent / "Least_Squares"
    if str(ls_root) not in sys.path:
        sys.path.insert(0, str(ls_root))


# ---------------------------------------------------------------------------
# Least-squares wrappers (1D anchor space).
# ---------------------------------------------------------------------------


def push_erm(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """ERM has no adversary; the canonical Monge gap reference point."""
    return z_hat.detach().clone()


def push_npf(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """T_omega(z) = grad_z psi_omega(z), computed via autograd.

    In paper notation this is ICNN-DRO; in code notation it is ``npf``.

    LS pipeline clamps adversary outputs to [-1, 1] at the outer step
    (see ``algorithms/npf.py``). We mirror that clamp here so the held-out
    evaluation lives in the same image as the training-time adversary.
    """
    psi = state["psi"]
    z_in = z_hat.detach().clone()
    if z_in.dim() == 1:
        z2d = z_in.view(-1, 1).requires_grad_(True)
    else:
        z2d = z_in.requires_grad_(True)
    val = psi(z2d).sum()
    grad = torch.autograd.grad(val, z2d, create_graph=False)[0]
    out = grad.detach().view_as(z_hat)
    if state.get("clamp_to_unit", True):
        out = out.clamp(-1.0, 1.0)
    return out


def push_nn_dro(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """Outer step in ``algorithms/nn_dro.py`` clamps to [-1, 1]; mirror it."""
    adversary = state["adversary"]
    z_in = z_hat.detach().clone()
    if z_in.dim() == 1:
        z2d = z_in.view(-1, 1)
    else:
        z2d = z_in
    with torch.no_grad():
        out = adversary(z2d)
    out = out.detach().view_as(z_hat)
    if state.get("clamp_to_unit", True):
        out = out.clamp(-1.0, 1.0)
    return out


def _ls_inner_grad(theta, z, A0, A1, b, lam, xi_anchor):
    """grad of (loss(theta, z) - lam (z - xi_anchor)^2) w.r.t. z, vectorized."""
    _ensure_ls_on_path()
    from utils.loss import loss_grad_xi  # type: ignore

    return loss_grad_xi(theta, z, A0, A1, b) - 2.0 * lam * (z - xi_anchor)


def push_particle(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """Rerun particle ascent from each anchor with theta frozen."""
    theta = state["theta"]
    cfg = state["cfg"]
    A0, A1, b = state["A0"], state["A1"], state["b"]
    xi_anchor = z_hat.detach().clone()
    z = xi_anchor.clone()
    for _ in range(cfg.inner_steps):
        grad = _ls_inner_grad(theta, z, A0, A1, b, cfg.lam, xi_anchor)
        z = z + cfg.inner_step_size * grad
        z = z.clamp(-1.0, 1.0)
    return z.detach()


def push_ppa(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """Rerun MPA (PPA) from each anchor with theta frozen.

    Identical to push_particle but with the pool-and-reassign rounds (Brenier
    1D projection + refinement). Used as the codebase key for MPA.
    """
    _ensure_ls_on_path()
    from utils.projections import brenier_projection_1d  # type: ignore

    theta = state["theta"]
    cfg = state["cfg"]
    A0, A1, b = state["A0"], state["A1"], state["b"]
    xi_anchor = z_hat.detach().clone()
    z = xi_anchor.clone()
    for _ in range(cfg.inner_steps):
        grad = _ls_inner_grad(theta, z, A0, A1, b, cfg.lam, xi_anchor)
        z = z + cfg.inner_step_size * grad
        z = z.clamp(-1.0, 1.0)
    for round_idx in range(1, cfg.ppa_num_rounds):
        z, delta, C_id = brenier_projection_1d(z, xi_anchor)
        if (round_idx >= cfg.ppa_min_rounds
                and delta < cfg.ppa_delta_rtol * max(C_id, 1e-12)):
            break
        for _ in range(cfg.ppa_refine_steps):
            grad = _ls_inner_grad(theta, z, A0, A1, b, cfg.lam, xi_anchor)
            z = z + cfg.ppa_refine_lr * grad
            z = z.clamp(-1.0, 1.0)
    z, _, _ = brenier_projection_1d(z, xi_anchor)
    return z.detach()


def push_wfr(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """Run the WFR sampler on each anchor; return the cloud mean as T(z).

    WFR is not a deterministic map — its output is an N x M particle cloud
    with reweighted masses. Reducing to the mean is the canonical
    deterministic-projection that lets us evaluate the Monge gap; the
    limitation is intrinsic to the method, not the estimator.
    """
    _ensure_ls_on_path()
    from utils.loss import loss_function, loss_grad_xi  # type: ignore

    theta = state["theta"]
    cfg = state["cfg"]
    A0, A1, b = state["A0"], state["A1"], state["b"]
    xi_anchor = z_hat.detach().clone()
    n = xi_anchor.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)

    particles = xi_anchor.view(-1, 1).repeat(1, m)
    weights = torch.full((n, m), 1.0 / float(m), device=xi_anchor.device, dtype=xi_anchor.dtype)

    for _ in range(cfg.inner_steps):
        flat = particles.reshape(-1)
        xi_exp = xi_anchor.repeat_interleave(m)
        grad = loss_grad_xi(theta, flat, A0, A1, b) - 2.0 * cfg.lam * (flat - xi_exp)
        noise = torch.randn_like(flat)
        flat = flat + cfg.inner_step_size * grad + noise_scale * noise
        particles = flat.view(n, m).clamp(-1.0, 1.0)

        f_bar = loss_function(theta, particles.reshape(-1), A0, A1, b, cfg.dim_m).view(n, m)
        f_bar = f_bar - cfg.lam * (particles - xi_anchor.view(-1, 1)) ** 2
        power = 1.0 - cfg.lam * cfg.epsilon * cfg.wfr_weight_step_size
        weights = (weights.clamp_min(1e-12) ** power) * torch.exp(
            cfg.wfr_weight_step_size * f_bar
        )
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    # Weighted barycenter -> deterministic map per anchor.
    z_out = (weights * particles).sum(dim=1)
    return z_out.detach()


def push_dual(state: Dict[str, Any], z_hat: torch.Tensor) -> torch.Tensor:
    """Run SDRO Sinkhorn sampler on each anchor; return the cloud mean."""
    _ensure_ls_on_path()
    from utils.loss import loss_function  # type: ignore

    theta = state["theta"]
    cfg = state["cfg"]
    A0, A1, b = state["A0"], state["A1"], state["b"]
    xi_anchor = z_hat.detach().clone()
    n = xi_anchor.numel()

    # Use the largest level for the held-out evaluation (highest-fidelity sample).
    m = 2 ** int(cfg.sinkhorn_sample_level)
    noise = torch.randn((n, m), device=xi_anchor.device, dtype=xi_anchor.dtype) * math.sqrt(cfg.epsilon)
    z_samples = xi_anchor.view(-1, 1) + noise

    v = loss_function(theta, z_samples.reshape(-1), A0, A1, b, cfg.dim_m).view(n, m)
    v = v / (cfg.lam * cfg.epsilon)
    v_max = torch.max(v, dim=1, keepdim=True).values
    w = torch.exp(v - v_max)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    z_out = (w * z_samples).sum(dim=1)
    return z_out.detach()


# ---------------------------------------------------------------------------
# Registry: maps codebase keys to push_fn(state, z_hat) -> Tz.
# Display names are handled in the plotting layer; here we use code keys.
# The RL folder uses ``nominal`` and ``particle`` instead of ``erm`` and
# ``particle_ascent``; aliases are added at the bottom.
# ---------------------------------------------------------------------------

REGISTRY = {
    "erm":             push_erm,
    "particle_ascent": push_particle,
    "ppa":             push_ppa,
    "npf":             push_npf,        # ICNN-DRO
    "wfr":             push_wfr,
    "dual":            push_dual,
    "nn_dro":          push_nn_dro,
}

# RL-folder aliases (same wrappers, different keys in algorithms/).
REGISTRY["nominal"] = REGISTRY["erm"]
REGISTRY["particle"] = REGISTRY["particle_ascent"]
