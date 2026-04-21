"""Dual DRO baseline: log-sum-exp smoothed worst-case adversary in xi-space.

    J_dual(xi_hat) ≈ λ ε log E_{xi ~ N(xi_hat, ε I)} exp( f(θ, xi) / (λ ε) )

The adversary emits samples drawn from a softmax distribution over Gaussian
proposals around each hat_xi, proxying the inner supremum.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.xi_ops import f_values, repeat_xi


class DualAdversary:
    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)
        self._rng = np.random.default_rng(123)

    def _clip(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(xi, self.high), self.low)

    @torch.no_grad()
    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        lvl_max = int(self.cfg.dual_sample_level)
        levels = np.arange(lvl_max + 1)
        numerators = 2.0 ** (-levels)
        probs = numerators / (2.0 - 2.0 ** (-lvl_max))
        sampled_level = int(self._rng.choice(levels, p=probs))
        m = 2 ** sampled_level

        B, p = hat_xi.shape
        hat_rep = repeat_xi(hat_xi, m)
        std = math.sqrt(float(self.cfg.dual_epsilon))

        sigma = std / self.Mdiag.clamp_min(1e-12).sqrt()
        g = make_generator(int(seed0) + 1, hat_xi.device)
        noise = torch.randn(hat_rep.shape, generator=g, device=hat_xi.device, dtype=hat_xi.dtype) * sigma
        xi = self._clip(hat_rep + noise)

        f = f_values(
            env_eval, policy, xi,
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=int(seed0) + 7, deterministic=True,
        )
        eps = max(float(self.cfg.lam) * float(self.cfg.dual_epsilon), 1e-8)
        residuals = f / eps
        residuals = residuals.view(B, m)
        weights = torch.softmax(residuals, dim=1)

        xi_grid = xi.view(B, m, p)
        xi_adv = (weights.unsqueeze(-1) * xi_grid).sum(dim=1)

        return self._clip(xi_adv).detach()
