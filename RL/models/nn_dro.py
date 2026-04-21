"""Vanilla MLP adversary for NN-DRO.

Parametrises the transport directly as a vanilla MLP (no gradient-of-potential
formulation, unlike ICNN / NPF). The MLP maps a latent-space point u (the
logit-encoded nominal xi) to an additive displacement delta(u), so

    T(u) = u + delta_theta(u),    xi_adv = sigmoid_decode(T(u)).

Weights of the final layer are initialised near zero, so T(u) ≈ u at the
start of training and hence xi_adv ≈ hat_xi (mirroring ICNN's identity init).
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
    """Vanilla MLP producing a displacement delta(u) of the same shape as u."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "relu",
        softplus_beta: float = 20.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("NN-DRO MLP requires at least one hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = str(activation)
        self.softplus_beta = float(softplus_beta)

        dims = [self.input_dim, *self.hidden_sizes]
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(self.hidden_sizes[-1], self.input_dim)

        with torch.no_grad():
            self.out.weight.mul_(float(init_scale))
            self.out.bias.zero_()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        act = _make_activation(self.activation, self.softplus_beta)
        h = u
        for layer in self.hidden:
            h = act(layer(h))
        return self.out(h)
