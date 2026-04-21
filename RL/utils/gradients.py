"""Gradient estimators for the inner objective ∇_xi f(theta, xi)."""
from __future__ import annotations

import torch

from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.rollouts import evaluate_return_batch, evaluate_return_batch_pathwise


@torch.no_grad()
def estimate_grad_f_wrt_xi_batch(
    env: VecEnvTorch,
    policy: PolicyNet,
    xi: torch.Tensor,
    eps: float,
    n_episodes: int,
    max_steps: int,
    seed0: int,
    method: str = "pathwise",
) -> torch.Tensor:
    """Gradient of f(theta, xi) = -J(theta, xi) wrt xi (batch).

    Returns g = ∇_xi f; ascent on f decreases return.

    Methods
    -------
    pathwise (autograd):
      g = ∇_xi (-J̃), J̃ is a differentiable surrogate return.
    spsa:
      Rademacher perturbations, g ≈ (J_minus - J_plus)/(2 eps) * delta.
    fd:
      central difference per dim.
    """
    device = xi.device
    B, p = xi.shape

    if method.lower() in ("pathwise", "autograd"):
        with torch.enable_grad():
            xi_var = xi.detach().clone().requires_grad_(True)
            J = evaluate_return_batch_pathwise(
                env,
                policy,
                xi_var,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed0=seed0,
            )
            loss = -J.sum()
            g = torch.autograd.grad(loss, xi_var, create_graph=False)[0]
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return g.detach()

    if method.lower() == "spsa":
        delta = torch.empty_like(xi).bernoulli_(0.5).mul_(2).sub_(1)
        xi_plus = xi + eps * delta
        xi_minus = xi - eps * delta

        J_plus = evaluate_return_batch(env, policy, xi_plus, n_episodes=n_episodes, max_steps=max_steps, seed0=seed0 + 11)
        J_minus = evaluate_return_batch(env, policy, xi_minus, n_episodes=n_episodes, max_steps=max_steps, seed0=seed0 + 22)

        coeff = (J_minus - J_plus) / (2.0 * eps)
        g = coeff.unsqueeze(-1) * delta
        return g

    elif method.lower() == "fd":
        g = torch.zeros_like(xi)
        eye = torch.eye(p, device=device, dtype=xi.dtype)
        for j in range(p):
            ej = eye[j].unsqueeze(0)
            xi_plus = xi + eps * ej
            xi_minus = xi - eps * ej

            J_plus = evaluate_return_batch(env, policy, xi_plus, n_episodes=n_episodes, max_steps=max_steps, seed0=seed0 + 101 + j)
            J_minus = evaluate_return_batch(env, policy, xi_minus, n_episodes=n_episodes, max_steps=max_steps, seed0=seed0 + 202 + j)
            g[:, j] = (J_minus - J_plus) / (2.0 * eps)
        return g

    else:
        raise ValueError("method must be 'pathwise', 'spsa' or 'fd'")
