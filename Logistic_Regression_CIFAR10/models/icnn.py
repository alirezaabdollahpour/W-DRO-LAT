"""ICNN potential modules (from uncertain_least_squares_icnn.py) + transport map.

Provides:
  * ``icnn_principled_moments`` — log-normal moments for positive-weight init.
  * ``NonNegativeLinear`` — linear map with strictly non-negative weights.
  * ``InputConvexPotential`` — dense ICNN potential ψ(z).
  * ``initialize_icnn_identity`` — re-init so T_ω(x) = x at t=0.
  * ``T_omega`` — transport map T_ω(x) = ∇_x ψ_ω(x).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def icnn_principled_moments(fan_in: int) -> Tuple[float, float, float, float, float]:
    """Principled log-normal moments for positive weights."""
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NonNegativeLinear(nn.Module):
    """Linear map with strictly non-negative weights via exp/softplus."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "principled",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_mode = init_mode.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported init_mode '{init_mode}' for NonNegativeLinear.")
        self.parametrization = "exp" if self.init_mode == "principled" else "softplus"

        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _mu_w, _sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if tilde_sigma == 0.0:
                    self.weight_param.fill_(tilde_mu)
                else:
                    self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
                if self.bias is not None:
                    self.bias.fill_(mu_b)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parametrization == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    """Dense ICNN potential ψ(z) that is convex in z."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "softplus",
        strong_convexity: float = 1.0,
        nonneg_init: str = "principled",
        softplus_beta: float = 20.0,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.strong_convexity = strong_convexity
        self.softplus_beta = softplus_beta

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        for i, width in enumerate(hidden_sizes):
            self.z_linears.append(nn.Linear(input_dim, width, bias=True))
            if i == 0:
                self.h_linears.append(None)
            else:
                self.h_linears.append(
                    NonNegativeLinear(
                        hidden_sizes[i - 1],
                        width,
                        bias=False,
                        init_mode=nonneg_init,
                    )
                )

        self.hidden_output = NonNegativeLinear(
            hidden_sizes[-1],
            1,
            bias=True,
            init_mode=nonneg_init,
        )
        self.input_skip = nn.Linear(input_dim, 1, bias=True)

    def _activation(self):
        act_name = self.activation.lower()
        if act_name == "relu":
            return F.relu
        if act_name == "softplus":
            beta = float(self.softplus_beta)
            return lambda u: F.softplus(beta * u) / beta
        raise ValueError(f"Unsupported ICNN activation '{self.activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(x.size(0), -1)
        act = self._activation()

        h = act(self.z_linears[0](z))
        for k in range(1, len(self.z_linears)):
            z_term = self.z_linears[k](z)
            h_term = self.h_linears[k](h) if self.h_linears[k] is not None else 0.0
            h = act(z_term + h_term)

        quadratic = 0.5 * self.strong_convexity * (z ** 2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z) + self.hidden_output(h)
        return out.squeeze(-1)


def initialize_icnn_identity(
    icnn: InputConvexPotential, strong_convexity: float = 1.0
) -> None:
    """Identity init so that T(x) = ∇ψ(x) = strong_convexity · x at t = 0."""
    if not hasattr(icnn, "z_linears") or not hasattr(icnn, "input_skip"):
        raise TypeError("initialize_icnn_identity expects an InputConvexPotential-like module.")

    with torch.no_grad():
        icnn.strong_convexity = float(strong_convexity)
        for lin in icnn.z_linears:
            lin.weight.zero_()
        icnn.input_skip.weight.zero_()
        if icnn.input_skip.bias is not None:
            icnn.input_skip.bias.zero_()
        if getattr(icnn, "hidden_output", None) is not None and getattr(icnn.hidden_output, "bias", None) is not None:
            icnn.hidden_output.bias.zero_()


def T_omega(x: torch.Tensor, psi_omega: nn.Module, create_graph: bool) -> torch.Tensor:
    """Transport map T_ω(x) = ∇_x ψ_ω(x)."""
    x_in = x.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi_val = psi_omega(x_in)
        grad = torch.autograd.grad(psi_val.sum(), x_in, create_graph=create_graph)[0]
    return grad.view_as(x)
