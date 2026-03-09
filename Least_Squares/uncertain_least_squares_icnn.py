#!/usr/bin/env python3
"""uncertain_least_squares_icnn.py

Reproduces **Appendix D.2 (Uncertain Least Square)** from
"GRADIENT FLOW SAMPLER-BASED DISTRIBUTIONALLY ROBUST OPTIMIZATION" and
generates a plot matching **Figure 8** (test loss vs perturbation level Δ).

In addition to the paper's baselines (ERM / Particle Ascent / WGF / WFR / Dual), this script
optionally adds an **ICNN map-based adversary** that follows the exact ICNN
implementation style used in your `moon.py`:

  - ICNN potential ψ_ω(·) convex in the input
  - Transport map T_ω(x) = ∇_x ψ_ω(x)
  - ω optimized by **BB + Armijo** ascent
  - nonnegative weights use the **principled initialization** (log-normal)

Notes
-----
* The uncertain scalar is ξ ∈ [-1, 1]. We therefore clamp adversarial ξ to [-1,1]
  (as done in the paper's particle-based inner loops).
* Default hyperparameters match the Figure 8 caption (10 epochs, λ=0.1,
  inner stepsize 1e-4 for 3000 iterations, ε=0.1, m=8 particles).

Run
---
    python uncertain_least_squares_icnn_fig8.py

This will save:
  - fig8_uncertain_least_squares.pdf           (ERM/WGF/WFR/Particle Ascent/Dual)
  - fig8_uncertain_least_squares_plus_icnn.pdf (same + ICNN)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils


# -------------------------
# Repro / utils
# -------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# -------------------------
# ICNN: BB + Armijo (ascent)
# -------------------------


@dataclass
class BBArmijoState:
    """Barzilai-Borwein step size + Armijo line search state."""

    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float = 0.0005,
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0,
        ls_c: float = 0.1,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        """Propose a BB step size (before Armijo)."""
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.shape != params_vec.shape
            or self.prev_grad_vec.shape != grad_vec.shape
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            num = torch.dot(s, s)
            cond = torch.isfinite(denom) & (torch.abs(denom) > 1e-12)
            alpha_bb = torch.where(cond, num / denom, torch.tensor(self.alpha_prev, device=denom.device))
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())

        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=alpha_clamped,
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_step_params(
    params,
    f_params,
    bb_state: BBArmijoState,
) -> Tuple[List[torch.nn.Parameter], BBArmijoState, float, float]:
    """Single BB+Armijo gradient-ascent step on a parameter collection."""

    params = list(params)
    params_vec = nn_utils.parameters_to_vector(params).detach()

    f_val = f_params(True)
    grads = torch.autograd.grad(
        f_val,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)]
    grad_vec = nn_utils.parameters_to_vector(grad_tensors)
    grad_norm = grad_vec.norm().item()

    alpha = bb_state.propose(params_vec, grad_vec)
    f_val_float = float(f_val.detach())
    g_dot_g = float(torch.dot(grad_vec, grad_vec).item())
    if g_dot_g == 0.0:
        return params, bb_state, f_val_float, grad_norm

    alpha_k = alpha
    for _ in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial = float(f_params(False).detach())
        if f_trial >= f_val_float + bb_state.ls_c * alpha_k * g_dot_g:
            break
        alpha_k *= bb_state.ls_shrink

    final_vec = params_vec + alpha_k * grad_vec
    with torch.no_grad():
        nn_utils.vector_to_parameters(final_vec, params)

    f_new = f_params(True)
    grads_new = torch.autograd.grad(
        f_new,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors_new = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads_new)]
    grad_vec_new = nn_utils.parameters_to_vector(grad_tensors_new)
    new_state = bb_state.update_history(final_vec, grad_vec_new, alpha_k)
    return params, new_state, f_val_float, grad_vec_new.norm().item()


# -------------------------
# ICNN modules (principled init)
# -------------------------


def icnn_principled_moments(fan_in: int):
    """Principled log-normal moments for positive weights."""
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NonNegativeLinear(nn.Module):
    """Linear map with strictly non-negative weights via exp/softplus."""

    def __init__(self, in_features, out_features, bias=True, init_mode="principled"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_mode = init_mode.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported init_mode '{init_mode}' for NonNegativeLinear.")
        self.parametrization = "exp" if self.init_mode == "principled" else "softplus"

        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _mu_w, _sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if tilde_sigma == 0.0:
                    self.weight_param.fill_(tilde_mu)
                else:
                    self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
                if self.bias is not None:
                    self.bias.fill_(mu_b)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.parametrization == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    """Dense ICNN potential ψ(z) that is convex in z."""

    def __init__(
        self,
        input_dim=1,
        hidden_sizes=(64, 64),
        activation="softplus",
        strong_convexity=1.0,
        nonneg_init="principled",
        softplus_beta=20.0,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.strong_convexity = strong_convexity
        self.softplus_beta = softplus_beta

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        for i, width in enumerate(hidden_sizes):
            self.z_linears.append(nn.Linear(input_dim, width, bias=True))
            if i == 0:
                self.h_linears.append(None)
            else:
                self.h_linears.append(
                    NonNegativeLinear(
                        hidden_sizes[i - 1],
                        width,
                        bias=False,
                        init_mode=nonneg_init,
                    )
                )

        self.hidden_output = NonNegativeLinear(
            hidden_sizes[-1],
            1,
            bias=True,
            init_mode=nonneg_init,
        )
        self.input_skip = nn.Linear(input_dim, 1, bias=True)

    def _activation(self):
        act_name = self.activation.lower()
        if act_name == "relu":
            return F.relu
        if act_name == "softplus":
            beta = float(self.softplus_beta)
            return lambda u: F.softplus(beta * u) / beta
        raise ValueError(f"Unsupported ICNN activation '{self.activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(x.size(0), -1)
        act = self._activation()

        h = act(self.z_linears[0](z))
        for k in range(1, len(self.z_linears)):
            z_term = self.z_linears[k](z)
            h_term = self.h_linears[k](h) if self.h_linears[k] is not None else 0.0
            h = act(z_term + h_term)

        quadratic = 0.5 * self.strong_convexity * (z**2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z) + self.hidden_output(h)
        return out.squeeze(-1)


def T_omega(x: torch.Tensor, psi_omega: nn.Module, create_graph: bool) -> torch.Tensor:
    """Transport map T_ω(x) = ∇_x ψ_ω(x)."""

    x_in = x.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi_val = psi_omega(x_in)
        grad = torch.autograd.grad(psi_val.sum(), x_in, create_graph=create_graph)[0]
    return grad.view_as(x)


# -------------------------
# Uncertain least squares: f_θ(ξ)
# -------------------------


@dataclass
class ULSConfig:
    # Problem sizes
    dim_m: int = 10
    dim_n: int = 10
    n_train: int = 10
    n_test: int = 1000

    # Delta sweep
    delta_min: float = 0.0
    delta_max: float = 10.0
    delta_steps: int = 50

    # Training hyperparams (Figure 8 caption)
    epochs: int = 10
    lr_theta: float = 1e-2
    lam: float = 0.1
    epsilon: float = 0.1
    grad_clip: float = 100.0

    # Inner-loop hyperparams
    inner_steps: int = 3000
    inner_step_size: float = 1e-4
    m_particles: int = 8

    # WFR
    wfr_weight_step_size: float = 0.0016
    wfr_low_weight_threshold: float = 1e-3

    # Dual (randomized truncation MLMC)
    sinkhorn_sample_level: int = 4

    # ICNN
    run_icnn: bool = True
    icnn_hidden: Tuple[int, ...] = (512, 512, 512, 256, 256, 128, 128, 64)
    icnn_strong_convexity: float = 1.0
    icnn_softplus_beta: float = 20.0
    icnn_bb_alpha0: float = 1e-1
    icnn_bb_alpha_min: float = 1e-6
    icnn_bb_alpha_max: float = 10.0
    icnn_bb_ls_c: float = 1e-4
    icnn_bb_ls_shrink: float = 0.5
    icnn_bb_ls_max_steps: int = 10
    icnn_omega_steps_per_epoch: int = 50

    # Repro
    seed: int = 219

    # Numerical stability
    ridge: float = 1e-6


def make_problem(cfg: ULSConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate A0, A1, b as in the paper (i.i.d. standard normal)."""
    rng = np.random.RandomState(cfg.seed)
    A0 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    A1 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    b = torch.tensor(rng.randn(cfg.dim_m), device=device)
    return A0, A1, b


def generate_training_data(cfg: ULSConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(cfg.n_train, device=device).uniform_(-0.5, 0.5)


def generate_test_data(cfg: ULSConfig, delta: float, device: torch.device) -> torch.Tensor:
    lo = -0.5 * (1.0 + float(delta))
    hi = 0.5 * (1.0 + float(delta))
    return torch.empty(cfg.n_test, device=device).uniform_(lo, hi)


def loss_function(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
    dim_m: int,
) -> torch.Tensor:
    """Vectorized f_θ(ξ) = ||A(ξ)θ - b||^2 / m."""
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)  # (B,m,n)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)  # (B,m)
    loss = (residual**2).sum(dim=1) / float(dim_m)  # (B,)
    return loss


def loss_grad_theta(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Vectorized ∇_θ f_θ(ξ). (Matches the authors' scaling.)"""
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    grad = 2.0 * torch.matmul(A_z.transpose(1, 2), residual.unsqueeze(2)).squeeze(2)  # (B,n)
    return grad


def loss_grad_xi(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Vectorized ∇_ξ f_θ(ξ). (Matches the authors' scaling.)"""
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    grad_Az = torch.matmul(A1, theta)  # (m,)
    grad = 2.0 * (residual * grad_Az.view(1, -1)).sum(dim=1)  # (B,)
    return grad


def clip_grad(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    gnorm = grad.norm()
    if gnorm > max_norm:
        grad = grad * (max_norm / (gnorm + 1e-12))
    return grad


# -------------------------
# Solvers (paper baselines)
# -------------------------


def solve_erm_closed_form(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ERM solution (quadratic objective)."""
    n = cfg.dim_n
    S = torch.zeros((n, n), device=xi_train.device)
    t = torch.zeros((n,), device=xi_train.device)
    for xi in xi_train:
        A = A0 + xi * A1
        S = S + A.transpose(0, 1) @ A
        t = t + A.transpose(0, 1) @ b
    S = S + cfg.ridge * torch.eye(n, device=xi_train.device)
    theta = torch.linalg.solve(S, t)
    return theta

# Particle Ascent solver is same as WRM solver in the paper (I just rename it)
def solve_Particle_Ascent(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Particle Ascent (ε=0): inner maximization via deterministic gradient ascent on ξ."""
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()

    for _epoch in range(cfg.epochs):
        z = xi_train.clone()
        for _ in range(cfg.inner_steps):
            grad = loss_grad_xi(theta, z, A0, A1, b) - 2.0 * cfg.lam * (z - xi_train)
            z = z + cfg.inner_step_size * grad
            z = z.clamp(-1.0, 1.0)

        grads_theta = loss_grad_theta(theta, z, A0, A1, b)  # (N,n)
        avg_grad = grads_theta.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return theta


def solve_wgf(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """WGF sampler (Algorithm 3 in the paper): Langevin-like particle updates."""
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)

    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)  # (N,m)
        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (particles_flat - xi_expanded)
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)

        particles_flat = particles.reshape(-1)
        grads = loss_grad_theta(theta, particles_flat, A0, A1, b)  # (N*m,n)
        grads = grads.view(n_train, m, cfg.dim_n).mean(dim=1)  # (N,n)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return theta


def solve_wfr(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """WFR sampler (Algorithm 4 in the paper): WGF + weight dynamics + birth-death."""
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)

    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)
        weights = torch.full((n_train, m), 1.0 / float(m), device=xi_train.device)

        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (particles_flat - xi_expanded)
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)

            # Weight update
            f_bar = loss_function(theta, particles.reshape(-1), A0, A1, b, cfg.dim_m).view(n_train, m)
            f_bar = f_bar - cfg.lam * (particles - xi_train.view(-1, 1)) ** 2

            power = 1.0 - cfg.lam * cfg.epsilon * cfg.wfr_weight_step_size
            weights = (weights.clamp_min(1e-12) ** power) * torch.exp(cfg.wfr_weight_step_size * f_bar)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

            # Birth-death: match the authors' procedure (row-wise)
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

                    # Update weights for {low} ∪ {j_max}
                    update_mask = low.clone()
                    update_mask[j_max] = True
                    w = torch.where(update_mask, avg_w, w)

                    # Move low-weight particles to the max-weight location
                    x = torch.where(low, x_max, x)
                    w = w / (w.sum() + 1e-12)

                    weights[i] = w
                    particles[i] = x

        grads = loss_grad_theta(theta, particles.reshape(-1), A0, A1, b).view(n_train, m, cfg.dim_n)
        grads = (weights.unsqueeze(-1) * grads).sum(dim=1)  # (N,n)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return theta


def solve_dual(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Dual method for SDRO (Wang et al., 2021) with randomized truncation MLMC."""
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    levels = torch.arange(cfg.sinkhorn_sample_level + 1, device=xi_train.device)
    numerators = 2.0 ** (-levels.to(torch.float64))
    denominator = 2.0 - 2.0 ** (-float(cfg.sinkhorn_sample_level))
    probs = (numerators / denominator).to(torch.float64)

    for _epoch in range(cfg.epochs):
        # sample level
        sampled_level = int(torch.multinomial(probs, num_samples=1).item())
        m = 2 ** sampled_level

        noise = torch.randn((n_train, m), device=xi_train.device) * math.sqrt(cfg.epsilon)
        z_samples = xi_train.view(-1, 1) + noise

        v = loss_function(theta, z_samples.reshape(-1), A0, A1, b, cfg.dim_m).view(n_train, m)
        v = v / (cfg.lam * cfg.epsilon)

        v_max = torch.max(v, dim=1, keepdim=True).values
        w = torch.exp(v - v_max)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        grads = loss_grad_theta(theta, z_samples.reshape(-1), A0, A1, b).view(n_train, m, cfg.dim_n)
        grads = (w.unsqueeze(-1) * grads).sum(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad

    return theta


# -------------------------
# ICNN map-based solver (optional)
# -------------------------


def solve_icnn_map(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, InputConvexPotential]:
    """Map-based adversary: z = T_ω(ξ) with BB+Armijo on ω."""

    device = xi_train.device
    theta = torch.zeros(cfg.dim_n, device=device)
    psi_omega = InputConvexPotential(
        input_dim=1,
        hidden_sizes=cfg.icnn_hidden,
        activation="softplus",
        strong_convexity=cfg.icnn_strong_convexity,
        nonneg_init="principled",
        softplus_beta=cfg.icnn_softplus_beta,
    ).to(device)

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
        # --- inner: maximize over ω ---
        for _ in range(cfg.icnn_omega_steps_per_epoch):
            def omega_objective(create_graph: bool) -> torch.Tensor:
                z_adv = T_omega(xi_2d, psi_omega, create_graph=create_graph).view(-1)
                z_adv = z_adv.clamp(-1.0, 1.0)
                f = loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                reg = cfg.lam * (z_adv - xi_train) ** 2
                return (f - reg).mean()

            _, bb_state, _, _ = bb_armijo_step_params(psi_omega.parameters(), omega_objective, bb_state)

        # --- outer: minimize over θ (envelope theorem: treat z_adv fixed) ---
        with torch.no_grad():
            z_adv = T_omega(xi_2d, psi_omega, create_graph=False).view(-1)
            z_adv = z_adv.clamp(-1.0, 1.0)
            grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
            grad_theta = clip_grad(grad_theta, cfg.grad_clip)
            theta = theta - cfg.lr_theta * grad_theta

    return theta, psi_omega


# -------------------------
# Experiment + plotting
# -------------------------


def run_experiment(cfg: ULSConfig, device: torch.device):
    set_seed(cfg.seed)
    A0, A1, b = make_problem(cfg, device)
    xi_train = generate_training_data(cfg, device)

    print("[1/2] Training models...")

    theta_erm = solve_erm_closed_form(xi_train, cfg, A0, A1, b)
    theta_Particle_Ascent = solve_Particle_Ascent(xi_train, cfg, A0, A1, b)
    theta_wgf = solve_wgf(xi_train, cfg, A0, A1, b)
    theta_wfr = solve_wfr(xi_train, cfg, A0, A1, b)
    theta_dual = solve_dual(xi_train, cfg, A0, A1, b)

    theta_icnn = None
    if cfg.run_icnn:
        theta_icnn, _psi = solve_icnn_map(xi_train, cfg, A0, A1, b)

    models: Dict[str, torch.Tensor] = {
        "ERM": theta_erm,
        "WGF(Otto, 1996)": theta_wgf,
        "WFR(Xu, 2025)": theta_wfr,
        "Particle Ascent": theta_Particle_Ascent,
        "Dual(Wang et al., 2021)": theta_dual,
    }
    if cfg.run_icnn:
        models["ICNN"] = theta_icnn

    delta_values = torch.linspace(cfg.delta_min, cfg.delta_max, cfg.delta_steps, device=device)
    results: Dict[str, List[float]] = {k: [] for k in models.keys()}

    print("[2/2] Evaluating test loss curves...")
    for delta in to_numpy(delta_values):
        xi_test = generate_test_data(cfg, float(delta), device)
        for name, theta in models.items():
            with torch.no_grad():
                test_loss = loss_function(theta, xi_test, A0, A1, b, cfg.dim_m).mean().item()
            results[name].append(float(test_loss))

    return to_numpy(delta_values), results


def plot_results(delta_values: np.ndarray, results: Dict[str, List[float]], save_path: str, *, include_icnn: bool):
    # Match the paper's compact figure style
    fig, ax = plt.subplots(figsize=(3.6, 2.613), dpi=300)

    styles = {
        "ERM": {"color": "k", "linestyle": "--"},
        "WGF(Otto, 1996)": {"color": "#9a0ba7", "linestyle": "-."},
        "WFR(Xu, 2025)": {"color": "#db2020", "linestyle": ":"},
        "Particle Ascent": {"color": "#0eaf0e", "linestyle": (0, (3, 1, 1, 1))},
        "Dual(Wang et al., 2021)": {"color": "#7A4E15", "linestyle": (0, (5, 5))},
        "ICNN": {"color": "#1f77b4", "linestyle": "-"},
    }

    plot_order = ["ERM", "WGF(Otto, 1996)", "WFR(Xu, 2025)", "Particle Ascent", "Dual(Wang et al., 2021)"]
    if include_icnn and "ICNN" in results:
        plot_order.append("ICNN")

    for name in plot_order:
        ax.plot(delta_values, results[name], linewidth=1.5, label=name, **styles[name])

    ax.set_xlabel(r"perturbation $\Delta$", fontsize=9)
    ax.set_ylabel("test loss", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(float(delta_values.min()), float(delta_values.max()))
    ax.set_ylim(0.0, 3.0)
    ax.grid(False)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    # delete top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(description="Uncertain least squares (Fig. 8) + optional ICNN map baseline")
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--delta_steps", type=int, default=50)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--inner_steps", type=int, default=3000)
    parser.add_argument("--inner_step_size", type=float, default=1e-4)
    parser.add_argument("--m_particles", type=int, default=8)
    parser.add_argument("--lr_theta", type=float, default=1e-2)
    parser.add_argument("--no_icnn", action="store_true", help="Disable ICNN baseline.")
    parser.add_argument("--icnn_omega_steps_per_epoch", type=int, default=50)
    args = parser.parse_args()

    cfg = ULSConfig(
        seed=args.seed,
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        delta_steps=args.delta_steps,
        lam=args.lam,
        epsilon=args.epsilon,
        inner_steps=args.inner_steps,
        inner_step_size=args.inner_step_size,
        m_particles=args.m_particles,
        lr_theta=args.lr_theta,
        run_icnn=not args.no_icnn,
        icnn_omega_steps_per_epoch=args.icnn_omega_steps_per_epoch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta_values, results = run_experiment(cfg, device)

    # Figure 8 replication (paper baselines only)
    plot_results(
        delta_values,
        {k: v for k, v in results.items() if k != "ICNN"},
        save_path="fig8_uncertain_least_squares.pdf",
        include_icnn=False,
    )
    print("Saved: fig8_uncertain_least_squares.pdf")

    # Optional: add ICNN overlay
    if cfg.run_icnn and "ICNN" in results:
        plot_results(
            delta_values,
            results,
            save_path="fig8_uncertain_least_squares_plus_icnn.pdf",
            include_icnn=True,
        )
        print("Saved: fig8_uncertain_least_squares_plus_icnn.pdf")


if __name__ == "__main__":
    main()
