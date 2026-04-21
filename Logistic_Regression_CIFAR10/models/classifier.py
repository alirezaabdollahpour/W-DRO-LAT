"""Linear classifier modules for CIFAR-10 feature-space DRO."""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Plain linear head: logits = W x + b."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LogisticRegression(nn.Module):
    """Multi-class logistic regression used by SAA baseline."""

    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
