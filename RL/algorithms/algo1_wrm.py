"""Algo1 / WRM: Sinha et al.-style inner maximization with decaying step η_s = lr / sqrt(s)."""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet
from utils.xi_ops import wrm_ascent_xi


class WRMAdversary:
    """WRM (Wasserstein Robust Method) inner maximization in xi-space.

        xi_{s+1} = xi_s + (lr/√s) (∇_xi f(θ, xi_s) - 2 λ M (xi_s - hat_xi)).
    """

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.low = torch.tensor(cfg.xi_low, device=device, dtype=torch.float32)
        self.high = torch.tensor(cfg.xi_high, device=device, dtype=torch.float32)
        self.Mdiag = torch.tensor(cfg.M_diag, device=device, dtype=torch.float32)

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        return wrm_ascent_xi(
            hat_xi,
            env_eval, policy,
            Mdiag=self.Mdiag, low=self.low, high=self.high,
            lambda_reg=float(self.cfg.lam),
            num_steps=int(self.cfg.K_algo1),
            lr=float(self.cfg.lr_algo1),
            n_episodes=int(self.cfg.fd_episodes),
            max_steps=int(self.cfg.fd_horizon),
            seed0=int(seed0),
        )
