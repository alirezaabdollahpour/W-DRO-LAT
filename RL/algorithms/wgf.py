"""WGF / Langevin-sampling WDRO adversary in xi-space.

Each hat_xi spawns N particles that evolve by Langevin dynamics on the
regularized inner objective f(θ, xi) - λ ||xi - hat||_M^2 (Mahalanobis metric),
with Gaussian noise std = √(2 η λ ε).
"""
from __future__ import annotations

import math

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.xi_ops import grad_f_xi, repeat_xi


class WGFAdversary:
    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    def _clip(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(xi, self.high), self.low)

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        num_samples = int(self.cfg.particle_num_samples)
        inner_steps = int(self.cfg.particle_inner_steps)
        inner_lr = float(self.cfg.particle_inner_lr)
        epsilon = float(self.cfg.particle_epsilon)
        lam = float(self.cfg.lam)

        anchor = repeat_xi(hat_xi.detach(), num_samples)
        xi = anchor.clone()

        noise_std = math.sqrt(max(2.0 * inner_lr * lam * epsilon, 1e-12))
        sigma = noise_std / self.Mdiag.clamp_min(1e-12).sqrt()

        for s in range(inner_steps):
            g = grad_f_xi(
                env_eval, policy, xi,
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 10 * (s + 1),
            )
            cost_g = 2.0 * (xi - anchor) * self.Mdiag
            mean = xi + inner_lr * (g - lam * cost_g)

            gen = make_generator(int(seed0) + 33 * (s + 1), xi.device)
            noise = torch.randn(xi.shape, generator=gen, device=xi.device, dtype=xi.dtype) * sigma
            xi = mean + noise
            xi = self._clip(xi)
        return xi.detach()
