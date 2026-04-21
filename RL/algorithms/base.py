"""Base adversary interface plus shared utilities for xi-space WDRO methods."""
from __future__ import annotations

from typing import Protocol

import torch

from envs import VecEnvTorch
from models.policy import PolicyNet


class Adversary(Protocol):
    """Contract for any adversary that maps nominal xi samples to worst-case xi samples."""

    def adversarial_xi(
        self,
        env_eval: VecEnvTorch,
        policy: PolicyNet,
        hat_xi: torch.Tensor,
        seed0: int = 0,
    ) -> torch.Tensor: ...


def project_box(xi: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(xi, high), low)


def mahalanobis_cost(xi: torch.Tensor, hat: torch.Tensor, Mdiag: torch.Tensor) -> torch.Tensor:
    """||xi - hat||_M^2 with diagonal M."""
    diff = xi - hat
    return (diff * diff * Mdiag).sum(dim=-1)


def cost_grad_xi(xi: torch.Tensor, hat: torch.Tensor, Mdiag: torch.Tensor) -> torch.Tensor:
    """∂/∂xi  ||xi - hat||_M^2 = 2 M (xi - hat)."""
    return 2.0 * (xi - hat) * Mdiag
