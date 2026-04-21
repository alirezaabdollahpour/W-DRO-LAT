"""Discrete-action policy network."""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
