"""Vanilla MLP adversary for NN-DRO.

Unlike ICNN (where the transport is the gradient of a convex potential), NN-DRO
parametrises the adversary directly as `z_adv(xi) = xi + h(xi)`, with `h` a
vanilla MLP. No gradient of the network w.r.t. its input is taken.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str, softplus_beta: float = 20.0):
    key = name.lower()
    if key == "relu":
        return F.relu
    if key == "elu":
        return F.elu
    if key == "tanh":
        return torch.tanh
    if key == "softplus":
        beta = float(softplus_beta)
        return lambda u: F.softplus(beta * u) / beta
    raise ValueError(f"Unsupported NN-DRO activation '{name}'.")


class MLPAdversary(nn.Module):
    """Vanilla MLP mapping xi -> z_adv = xi + h(xi)."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "relu",
        softplus_beta: float = 20.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("NN-DRO MLP requires at least one hidden layer.")
        self.input_dim = input_dim
        self.activation_name = activation
        self.softplus_beta = softplus_beta

        dims = [input_dim, *hidden_sizes]
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(hidden_sizes[-1], input_dim)

        # Initialise near identity: small output => z_adv ~= xi at start.
        with torch.no_grad():
            self.out.weight.mul_(init_scale)
            self.out.bias.zero_()

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        act = _make_activation(self.activation_name, self.softplus_beta)
        x = xi.view(xi.size(0), -1)
        h = x
        for layer in self.hidden:
            h = act(layer(h))
        delta = self.out(h)
        return (x + delta).view_as(xi)
