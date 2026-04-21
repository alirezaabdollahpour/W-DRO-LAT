"""Gradient/objective primitives for xi-space adversaries (WRM, PPA, NPF, samplers)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.rollouts import evaluate_return_batch, evaluate_return_batch_pathwise


def f_values(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    *,
    n_episodes: int,
    max_steps: int,
    seed0: int,
    deterministic: bool = True,
) -> torch.Tensor:
    """Per-sample adversary objective f(θ, xi) = -J(θ, xi), no gradients."""
    with torch.no_grad():
        J = evaluate_return_batch(
            env, policy, xi,
            n_episodes=n_episodes, max_steps=max_steps,
            seed0=seed0, deterministic=deterministic,
        )
    return (-J).detach()


def f_values_pathwise(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    *,
    n_episodes: int,
    max_steps: int,
    seed0: int,
) -> torch.Tensor:
    """Differentiable surrogate f̃(θ, xi) for autograd / backprop over xi or ψ.

    Returns a tensor that retains the graph wrt `xi`.
    """
    J = evaluate_return_batch_pathwise(
        env, policy, xi,
        n_episodes=n_episodes, max_steps=max_steps,
        seed0=seed0,
    )
    return -J


def grad_f_xi(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    *,
    n_episodes: int,
    max_steps: int,
    seed0: int,
) -> torch.Tensor:
    """∇_xi f(θ, xi) via pathwise autograd. Returns a detached tensor of shape xi."""
    with torch.enable_grad():
        xi_var = xi.detach().clone().requires_grad_(True)
        f_val = f_values_pathwise(
            env, policy, xi_var,
            n_episodes=n_episodes, max_steps=max_steps, seed0=seed0,
        )
        g = torch.autograd.grad(f_val.sum(), xi_var, create_graph=False)[0]
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g.detach()


def wrm_ascent_xi(
    hat_xi: torch.Tensor,
    env: VecEnvTorch,
    policy: PolicyNet,
    *,
    Mdiag: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float,
    n_episodes: int,
    max_steps: int,
    seed0: int,
    step_offset: int = 0,
    start_xi: Optional[torch.Tensor] = None,
    anchor_xi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """WRM-style ascent in xi-space with decaying step η_s = lr / sqrt(s).

        xi <- xi + η_s (∇_xi f - 2 λ M (xi - anchor)).

    Anchors default to `start_xi` if provided, else `hat_xi`.
    """
    if num_steps == 0:
        return (start_xi if start_xi is not None else hat_xi).detach().clone()

    xi = (start_xi if start_xi is not None else hat_xi).detach().clone()
    anchor = (anchor_xi if anchor_xi is not None else hat_xi).detach()

    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        g = grad_f_xi(
            env, policy, xi,
            n_episodes=n_episodes, max_steps=max_steps,
            seed0=seed0 + 10 * s,
        )
        eta_s = lr / math.sqrt(s)
        cost_g = 2.0 * (xi - anchor) * Mdiag
        xi = xi + eta_s * (g - lambda_reg * cost_g)
        xi = torch.where(torch.isfinite(xi), xi, anchor)
        xi = torch.max(torch.min(xi, high), low)
    return xi.detach()


def wrm_ascent_xi_const_lr(
    start_xi: torch.Tensor,
    anchor_xi: torch.Tensor,
    env: VecEnvTorch,
    policy: PolicyNet,
    *,
    Mdiag: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float,
    n_episodes: int,
    max_steps: int,
    seed0: int,
) -> torch.Tensor:
    """Constant-step WRM ascent anchored at `anchor_xi` (used for PPA refine rounds)."""
    if num_steps == 0:
        return start_xi.detach().clone()
    xi = start_xi.detach().clone()
    anchor = anchor_xi.detach()
    for s in range(num_steps):
        g = grad_f_xi(
            env, policy, xi,
            n_episodes=n_episodes, max_steps=max_steps,
            seed0=seed0 + 10 * (s + 1),
        )
        cost_g = 2.0 * (xi - anchor) * Mdiag
        xi = xi + lr * (g - lambda_reg * cost_g)
        xi = torch.where(torch.isfinite(xi), xi, anchor)
        xi = torch.max(torch.min(xi, high), low)
    return xi.detach()


def repeat_xi(xi: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Repeat each xi row `num_samples` times — (B, p) -> (B*num_samples, p)."""
    return xi.repeat_interleave(num_samples, dim=0)
