"""Nominal (no-op) adversary — vanilla PPO baseline on P_hat."""
from __future__ import annotations

import torch

from config import InnerConfig
from envs import VecEnvTorch
from models.policy import PolicyNet


class NominalAdversary:
    """No-op adversary: always returns the nominal parameters unchanged.

    This is the non-robust baseline — vanilla PPO trained on the nominal
    physics distribution P_hat only, with no adversarial perturbation.
    """

    def __init__(self, cfg: InnerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor:
        return hat_xi.clone()
