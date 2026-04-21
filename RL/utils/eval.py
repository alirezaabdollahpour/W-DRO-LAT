"""Outer evaluation helpers (worst-case grid, nominal return, sampling)."""
from __future__ import annotations

import torch

from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.rollouts import evaluate_return_batch


def sample_hat_xi(cfg, n: int, seed: int, device: torch.device) -> torch.Tensor:
    g = make_generator(seed, device)
    low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
    high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
    u = torch.rand(n, int(low.numel()), generator=g, device=device, dtype=torch.float32)
    return low + (high - low) * u


@torch.no_grad()
def evaluate_worst_case_grid(
    env_eval: VecEnvTorch,
    policy: PolicyNet,
    cfg,
    grid_n: int = 21,
    n_episodes: int = 1,
    max_steps: int = 300,
    seed0: int = 123,
) -> float:
    device = next(policy.parameters()).device
    low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
    high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
    d = int(low.numel())
    if d == 0:
        raise ValueError("xi dimension must be >= 1.")

    max_points = 2000
    total_grid = int(grid_n) ** int(d) if d <= 6 else max_points + 1
    if d <= 3 and total_grid <= max_points:
        axes = [torch.linspace(float(low[i]), float(high[i]), int(grid_n), device=device) for i in range(d)]
        meshes = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack([m.reshape(-1) for m in meshes], dim=-1)
    else:
        total = min(max_points, max(1, int(grid_n) ** min(int(d), 3)))
        g = make_generator(int(seed0), device)
        u = torch.rand(total, d, generator=g, device=device, dtype=torch.float32)
        grid = low + (high - low) * u

    J = evaluate_return_batch(env_eval, policy, grid, n_episodes=n_episodes, max_steps=max_steps, seed0=seed0, deterministic=True)
    worst = torch.min(J).item()
    return worst
