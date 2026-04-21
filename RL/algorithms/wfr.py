"""WFR (Wasserstein–Fisher–Rao) WDRO adversary in xi-space.

Particles evolve under Langevin dynamics (same as WGF) while their probability
weights decay/reweight multiplicatively from the per-particle energy. Low-weight
particles are resampled towards the highest-weight member of their group.

Note: in the downstream PPO rollout the particles are treated uniformly (the
per-particle weights are used only for the intra-sampler reweighting/resampling;
the outer PPO advantage is computed per-trajectory).
"""
from __future__ import annotations

import math

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.xi_ops import f_values, grad_f_xi, repeat_xi


class WFRAdversary:
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
        B = hat_xi.shape[0]
        S = int(self.cfg.particle_num_samples)
        inner_steps = int(self.cfg.particle_inner_steps)
        inner_lr = float(self.cfg.particle_inner_lr)
        epsilon = float(self.cfg.particle_epsilon)
        lam = float(self.cfg.lam)

        anchor = repeat_xi(hat_xi.detach(), S)
        xi = anchor.clone()
        weights = torch.full((B, S), 1.0 / S, device=xi.device, dtype=xi.dtype)

        noise_std = math.sqrt(max(2.0 * inner_lr * lam * epsilon, 1e-12))
        sigma = noise_std / self.Mdiag.clamp_min(1e-12).sqrt()
        weight_exponent = 1.0 - lam * epsilon * inner_lr

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
            xi = self._clip(mean + noise)

            with torch.no_grad():
                f_curr = f_values(
                    env_eval, policy, xi,
                    n_episodes=int(self.cfg.fd_episodes),
                    max_steps=int(self.cfg.fd_horizon),
                    seed0=int(seed0) + 77 * (s + 1),
                ).view(B, S)
                diff = xi - anchor
                dist_sq = (diff * diff * self.Mdiag).sum(dim=-1).view(B, S)
                energy = f_curr - 2.0 * lam * dist_sq
                weights = weights.pow(weight_exponent) * torch.exp(
                    torch.clamp(energy * inner_lr, min=-40.0, max=40.0)
                )
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

                low_mask = weights < 1e-4
                rows_low = torch.any(low_mask, dim=1)
                if torch.any(rows_low):
                    p = xi.shape[-1]
                    xi_grid = xi.view(B, S, p)
                    max_w_vals, max_w_idx = torch.max(weights, dim=1, keepdim=True)
                    gather_idx = max_w_idx.view(B, 1, 1).expand(B, 1, p)
                    highest = torch.gather(xi_grid, 1, gather_idx)

                    low_sum = torch.sum(weights * low_mask, dim=1, keepdim=True)
                    n_low = torch.sum(low_mask, dim=1, keepdim=True, dtype=weights.dtype)
                    avg_w = (max_w_vals + low_sum) / (n_low + 1.0 + 1e-9)
                    avg_w_exp = avg_w.expand_as(weights)
                    max_mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, max_w_idx, True)
                    update_mask = (low_mask | max_mask) & rows_low.unsqueeze(1)
                    weights = torch.where(update_mask, avg_w_exp, weights)

                    replacement = highest.expand(B, S, p)
                    x_update_mask = low_mask.view(B, S, 1)
                    rows_mask = rows_low.view(B, 1, 1)
                    xi_grid = torch.where(x_update_mask & rows_mask, replacement, xi_grid)
                    xi = xi_grid.view(B * S, p)
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

        return xi.detach()
