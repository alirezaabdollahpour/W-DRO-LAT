"""PPA: Projected Particle Ascent in xi-space (WRM ascent + Mahalanobis-gated refinement)."""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.xi_ops import wrm_ascent_xi, wrm_ascent_xi_const_lr


class PPAAdversary:
    """Multi-round ascent in xi-space.

    Round 0 reproduces WRM ascent from hat_xi. Subsequent rounds "project" the
    iterate to the box [xi_low, xi_high] (the xi-space analogue of the in-batch
    Brenier matching used in the MNIST_Cuturi PPA) and refine with a constant-step
    WRM ascent anchored at hat_xi.
    """

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
        z = wrm_ascent_xi(
            hat_xi,
            env_eval, policy,
            Mdiag=self.Mdiag, low=self.low, high=self.high,
            lambda_reg=float(self.cfg.lam),
            num_steps=int(self.cfg.ppa_inner_steps_round0),
            lr=float(self.cfg.ppa_inner_lr_round0),
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=int(seed0),
        )

        for round_idx in range(1, int(self.cfg.ppa_num_rounds)):
            z_proj = self._clip(z)
            diff = (z_proj - z)
            delta = (diff * diff * self.Mdiag).sum(dim=-1).mean().item()

            z_norm = (z_proj * z_proj * self.Mdiag).sum(dim=-1).mean().item()
            if (
                round_idx >= int(self.cfg.ppa_min_rounds)
                and delta < float(self.cfg.ppa_delta_rtol) * max(z_norm, 1e-12)
            ):
                z = z_proj
                break

            z = wrm_ascent_xi_const_lr(
                z_proj, hat_xi,
                env_eval, policy,
                Mdiag=self.Mdiag, low=self.low, high=self.high,
                lambda_reg=float(self.cfg.lam),
                num_steps=int(self.cfg.ppa_refine_steps),
                lr=float(self.cfg.ppa_refine_lr),
                n_episodes=int(self.cfg.fd_episodes),
                max_steps=int(self.cfg.fd_horizon),
                seed0=int(seed0) + 10_000 * round_idx,
            )

        return self._clip(z).detach()
