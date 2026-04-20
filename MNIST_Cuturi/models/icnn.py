"""Input-convex neural network (ICNN) potential + gradient map."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils._stateless import functional_call  # type: ignore


def icnn_principled_moments(fan_in: int) -> Tuple[float, float, float, float, float]:
    """Principled log-normal moments for positive-weight init."""
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


class NonNegativeDense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        init_mode: str = "principled",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.init_mode = init_mode.lower()
        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _mu_w, _sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(
                self.in_features
            )
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
        weight = (
            torch.exp(self.weight_param)
            if self.init_mode == "principled"
            else F.softplus(self.weight_param)
        )
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        strong_convexity: float = 1.0,
        nonneg_init: str = "principled",
    ):
        super().__init__()
        self.activation = activation.lower()
        self.strong_convexity = strong_convexity
        self.hidden_sizes = list(hidden_sizes)
        if len(self.hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        in_size = input_dim
        for i, width in enumerate(self.hidden_sizes):
            self.z_linears.append(nn.Linear(in_size, width))
            if i > 0:
                self.h_linears.append(
                    NonNegativeDense(
                        self.hidden_sizes[i - 1],
                        width,
                        use_bias=False,
                        init_mode=nonneg_init,
                    )
                )
        self.hidden_output = NonNegativeDense(
            self.hidden_sizes[-1], 1, init_mode=nonneg_init
        )
        self.input_skip = nn.Linear(in_size, 1)

    def init_as_identity(self):
        """Initialize so that ∇ψ(x) = x at t = 0 (requires strong_convexity = 1)."""
        with torch.no_grad():
            for z_lin in self.z_linears:
                z_lin.weight.zero_()
                z_lin.bias.zero_()
            self.input_skip.weight.zero_()
            self.input_skip.bias.zero_()

    def act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(u)
        if self.activation == "softplus":
            return F.softplus(20.0 * u) / 20.0
        raise ValueError(f"Unsupported activation {self.activation}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.view(z.size(0), -1)
        h = None
        for i, z_lin in enumerate(self.z_linears):
            z_term = z_lin(z_flat)
            if i == 0:
                h = self.act(z_term)
            else:
                h_term = self.h_linears[i - 1](h)
                h = self.act(z_term + h_term)
        assert h is not None
        quadratic = 0.5 * self.strong_convexity * (z_flat ** 2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z_flat) + self.hidden_output(h)
        return out.squeeze(1)


def icnn_gradient(
    model: InputConvexPotential,
    params: Dict[str, torch.Tensor],
    z_flat: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute T(x) = ∇ψ(x) via autograd using a functional parameter dict."""
    z_flat_req = z_flat.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        out = functional_call(model, params, (z_flat_req,))
        phi_sum = out.sum()
        grad = torch.autograd.grad(phi_sum, z_flat_req, create_graph=create_graph)[0]
    return grad
