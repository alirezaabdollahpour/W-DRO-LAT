"""Vanilla MLP adversary for NN-DRO on CIFAR-10 features.

Unlike ICNN (where the transport is the gradient of a convex potential) or NPF
(same convex-potential formulation with per-layer quadratic re-injections),
NN-DRO parametrises the adversary directly as

    T_omega(x) = x + h_omega(x),

with h_omega a plain MLP. No autograd through the network's input is taken —
the MLP's output IS the displacement.

The final layer is initialised near zero, so T_omega(x) ≈ x at the start of
training (mirroring ``initialize_icnn_identity``).
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
    """Vanilla MLP mapping x -> T(x) = x + h(x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (512, 512, 256),
        activation: str = "relu",
        softplus_beta: float = 20.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("NN-DRO MLP requires at least one hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation_name = str(activation)
        self.softplus_beta = float(softplus_beta)

        dims = [self.input_dim, *self.hidden_sizes]
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(self.hidden_sizes[-1], self.input_dim)

        with torch.no_grad():
            self.out.weight.mul_(float(init_scale))
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = _make_activation(self.activation_name, self.softplus_beta)
        flat = x.view(x.size(0), -1)
        h = flat
        for layer in self.hidden:
            h = act(layer(h))
        delta = self.out(h)
        return (flat + delta).view_as(x)
