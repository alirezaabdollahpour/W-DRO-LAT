"""RGO (Restricted Gaussian Oracle) WDRO adversary in xi-space.

Two-phase sampler (cf. MNIST_Cuturi rgo_sampler):
  1. Compute the mode xi* via gradient ascent on  f/λ - ||xi - hat||^2/ε.
  2. Reject-sample around xi* with a Gaussian proposal of variance ε and
     acceptance ratio exp(- f_L(candidate) + f_L(xi*) + ||cand - xi*||^2/(2ε)).
"""
from __future__ import annotations

import math

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.common import make_generator
from utils.xi_ops import f_values, grad_f_xi, repeat_xi


class RGOAdversary:
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
        B, p = hat_xi.shape
        S = int(self.cfg.rgo_num_samples)
        inner_steps = int(self.cfg.rgo_inner_steps)
        inner_lr = float(self.cfg.rgo_inner_lr)
        epsilon = float(self.cfg.rgo_epsilon)
        lam = float(self.cfg.lam)
        max_trials = int(self.cfg.rgo_max_trials)

        anchor = hat_xi.detach()
        xi = anchor.clone()

        for s in range(inner_steps):
            g = grad_f_xi(
                env_eval, policy, xi,
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 10 * (s + 1),
            )
            cost_g = 2.0 * (xi - anchor) * self.Mdiag
            # Ascent on  (-f / λ) - ||xi - hat||^2 / ε  equivalently: xi += lr*( f_grad/λ - (xi-hat)*2/ε)
            xi = xi + inner_lr * (g / max(lam, 1e-8) - cost_g * (1.0 / max(epsilon, 1e-12)))
            xi = torch.where(torch.isfinite(xi), xi, anchor)
            xi = self._clip(xi)

        x_opt_star = xi.detach()
        if epsilon <= 1e-12:
            return repeat_xi(x_opt_star, S).contiguous()

        std_rgo = math.sqrt(epsilon)
        sigma = std_rgo / self.Mdiag.clamp_min(1e-12).sqrt()

        f_star = f_values(
            env_eval, policy, x_opt_star,
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=int(seed0) + 101,
        )
        diff_star = (x_opt_star - anchor)
        norm_sq_star = (diff_star * diff_star * self.Mdiag).sum(dim=-1)
        f_L_star = (-f_star / max(lam * epsilon, 1e-8)) + (norm_sq_star / max(epsilon, 1e-12))

        x_opt_expand = x_opt_star.unsqueeze(1)                        # (B, 1, p)
        anchor_expand = anchor.unsqueeze(1)                           # (B, 1, p)
        f_L_star_expand = f_L_star.unsqueeze(1)                       # (B, 1)

        final_accepted = torch.zeros((B, S, p), device=hat_xi.device, dtype=hat_xi.dtype)
        active = torch.ones((B, S), dtype=torch.bool, device=hat_xi.device)

        for trial in range(max_trials):
            if not torch.any(active):
                break
            gen = make_generator(int(seed0) + 999 + trial, hat_xi.device)
            proposals = torch.randn(B * S * p, generator=gen, device=hat_xi.device, dtype=hat_xi.dtype).view(B, S, p)
            proposals = proposals * sigma
            candidates = self._clip((x_opt_expand + proposals).view(B * S, p)).view(B, S, p)

            f_cand = f_values(
                env_eval, policy, candidates.view(B * S, p),
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 1000 + trial,
            ).view(B, S)
            diff_c = candidates - anchor_expand
            norm_sq_c = (diff_c * diff_c * self.Mdiag.view(1, 1, -1)).sum(dim=-1)
            f_L_c = (-f_cand / max(lam * epsilon, 1e-8)) + (norm_sq_c / max(epsilon, 1e-12))

            prop_diff = (candidates - x_opt_expand)
            diff_sq = (prop_diff * prop_diff * self.Mdiag.view(1, 1, -1)).sum(dim=-1)
            exponent = -f_L_c + f_L_star_expand + diff_sq / (2.0 * max(epsilon, 1e-12))
            acc_prob = torch.exp(torch.clamp(exponent, max=10.0))

            gen2 = make_generator(int(seed0) + 2000 + trial, hat_xi.device)
            u = torch.rand(B * S, generator=gen2, device=hat_xi.device, dtype=hat_xi.dtype).view(B, S)
            newly_accepted = (u < acc_prob) & active
            final_accepted[newly_accepted] = prop_diff[newly_accepted]
            active[newly_accepted] = False

        sampled = x_opt_expand + final_accepted
        sampled = self._clip(sampled.view(B * S, p))
        return sampled.detach()
