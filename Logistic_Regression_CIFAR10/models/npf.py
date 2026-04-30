"""NPF-style ICNN potential (Vesseron & Cuturi, 2024) + transport map.

Architecture (identical to MNIST_Cuturi/models/npf.py): per-layer diagonal +
low-rank convex quadratic injections on top of a learnable PSD outer base,
with non-negative hidden-to-hidden weights and a convex non-decreasing
activation (ELU by default).
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icnn import icnn_principled_moments

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils._stateless import functional_call  # type: ignore


def _npf_softplus_inverse(y: float) -> float:
    """Numerically stable inverse of softplus: log(exp(y) - 1) for y > 0."""
    if y <= 0:
        return -1e3
    return float(math.log(math.expm1(y)))


class NPFNonNegativeDense(nn.Module):
    """Non-negative linear layer with Hoedt & Klambauer negative bias init."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        _, _, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
        with torch.no_grad():
            if tilde_sigma == 0.0:
                self.weight_param.fill_(tilde_mu)
            else:
                self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
            if self.bias is not None:
                self.bias.fill_(-mu_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.exp(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class NPFQuadraticForm(nn.Module):
    """Stack of q convex quadratic forms Q(z) = ||δ⊙z||² + ||Az||²."""

    def __init__(
        self,
        input_dim: int,
        num_forms: int,
        rank: int = 1,
        delta_init: float = 0.0,
        init_eps: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_forms = int(num_forms)
        self.rank = int(rank)
        self.init_eps = float(init_eps)
        if self.init_eps > 0.0:
            delta_raw_init = _npf_softplus_inverse(self.init_eps)
        else:
            delta_raw_init = _npf_softplus_inverse(delta_init)
        self.delta_raw = nn.Parameter(
            torch.full((self.num_forms, self.input_dim), delta_raw_init)
        )
        if self.rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(
                    max(self.rank * self.input_dim, 1)
                )
                self.A = nn.Parameter(
                    std
                    * torch.randn(self.num_forms, self.rank, self.input_dim)
                )
            else:
                self.A = nn.Parameter(
                    torch.zeros(self.num_forms, self.rank, self.input_dim)
                )
        else:
            self.register_parameter("A", None)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.delta_raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.delta
        q_diag = ((z.unsqueeze(1) * delta.unsqueeze(0)) ** 2).sum(dim=-1)
        if self.A is not None:
            Az = torch.einsum("ord,bd->bor", self.A, z)
            q_lr = (Az ** 2).sum(dim=-1)
            return q_diag + q_lr
        return q_diag


class NPFInputConvexPotential(nn.Module):
    """NPF Eq. (5) ICNN potential: learnable PSD base + deep quadratic block."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        outer_rank: int = 4,
        inner_rank: int = 1,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        softplus_beta: float = 20.0,
        init_eps: float = 1e-3,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("NPFInputConvexPotential needs at least 1 hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.outer_rank = int(outer_rank)
        self.inner_rank = int(inner_rank)
        self.activation = activation.lower()
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        self.init_eps = float(init_eps)

        self.outer_delta_raw = nn.Parameter(
            torch.full((self.input_dim,), _npf_softplus_inverse(1.0))
        )
        if self.outer_rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(
                    max(self.outer_rank * self.input_dim, 1)
                )
                self.outer_A = nn.Parameter(
                    std * torch.randn(self.outer_rank, self.input_dim)
                )
            else:
                self.outer_A = nn.Parameter(
                    torch.zeros(self.outer_rank, self.input_dim)
                )
        else:
            self.register_parameter("outer_A", None)

        self.outer_a = nn.Parameter(torch.zeros(self.input_dim))

        self.q_blocks = nn.ModuleList()
        self.b_linears = nn.ModuleList()
        self.w_linears: nn.ModuleList = nn.ModuleList()

        for l, width in enumerate(self.hidden_sizes):
            self.q_blocks.append(
                NPFQuadraticForm(
                    input_dim=self.input_dim,
                    num_forms=width,
                    rank=self.inner_rank,
                    delta_init=0.0,
                    init_eps=self.init_eps,
                )
            )
            self.b_linears.append(nn.Linear(self.input_dim, width, bias=True))
            if l == 0:
                self.w_linears.append(None)  # type: ignore[arg-type]
            else:
                self.w_linears.append(
                    NPFNonNegativeDense(
                        in_features=self.hidden_sizes[l - 1],
                        out_features=width,
                        use_bias=False,
                    )
                )

        self.w_out = NPFNonNegativeDense(
            in_features=self.hidden_sizes[-1], out_features=1, use_bias=False
        )
        self.q_out = NPFQuadraticForm(
            input_dim=self.input_dim,
            num_forms=1,
            rank=self.inner_rank,
            delta_init=0.0,
            init_eps=self.init_eps,
        )
        self.b_out = nn.Linear(self.input_dim, 1, bias=True)

    def init_as_identity(self):
        """Force ∇ψ(z) ≈ z at t=0 (live-at-init when init_eps > 0)."""
        eps = self.init_eps
        delta_raw_init = _npf_softplus_inverse(eps) if eps > 0.0 else -1e3
        with torch.no_grad():
            self.outer_delta_raw.fill_(_npf_softplus_inverse(1.0))
            if self.outer_A is not None:
                if eps > 0.0:
                    std = eps / math.sqrt(
                        max(self.outer_rank * self.input_dim, 1)
                    )
                    self.outer_A.normal_(0.0, std)
                else:
                    self.outer_A.zero_()
            self.outer_a.zero_()
            for bl in self.b_linears:
                bl.weight.zero_()
                if bl.bias is not None:
                    bl.bias.zero_()
            for q in self.q_blocks:
                q.delta_raw.fill_(delta_raw_init)
                if q.A is not None:
                    if eps > 0.0:
                        std = eps / math.sqrt(max(q.rank * q.input_dim, 1))
                        q.A.normal_(0.0, std)
                    else:
                        q.A.zero_()
            self.q_out.delta_raw.fill_(delta_raw_init)
            if self.q_out.A is not None:
                if eps > 0.0:
                    std = eps / math.sqrt(
                        max(self.q_out.rank * self.q_out.input_dim, 1)
                    )
                    self.q_out.A.normal_(0.0, std)
                else:
                    self.q_out.A.zero_()
            self.b_out.weight.zero_()
            if self.b_out.bias is not None:
                self.b_out.bias.zero_()

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(u, alpha=self.elu_alpha)
        if self.activation == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * u) / beta
        if self.activation == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported NPF activation '{self.activation}'")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.view(z.size(0), -1)

        delta_out = F.softplus(self.outer_delta_raw)
        q_diag = 0.5 * (delta_out.pow(2) * z_flat.pow(2)).sum(dim=-1)
        if self.outer_A is not None:
            Az = z_flat @ self.outer_A.t()
            q_lr = 0.5 * (Az ** 2).sum(dim=-1)
        else:
            q_lr = torch.zeros(
                z_flat.size(0), dtype=z_flat.dtype, device=z_flat.device
            )
        linear = z_flat @ self.outer_a

        q0 = self.q_blocks[0](z_flat)
        b0 = self.b_linears[0](z_flat)
        h = self._act(q0 + b0)
        for l in range(1, len(self.hidden_sizes)):
            ql = self.q_blocks[l](z_flat)
            bl = self.b_linears[l](z_flat)
            wl = self.w_linears[l](h)
            h = self._act(wl + ql + bl)

        phi = (
            self.w_out(h).squeeze(-1)
            + self.q_out(z_flat).squeeze(-1)
            + self.b_out(z_flat).squeeze(-1)
        )
        return q_diag + q_lr + linear + phi


def npf_transport_map(
    icnn_model: NPFInputConvexPotential,
    params: Dict[str, torch.Tensor],
    z_flat: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """T_ω(z) = ∇_z ψ_ω(z) via functional_call."""
    z_flat_req = z_flat.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        out = functional_call(icnn_model, params, (z_flat_req,))
        grad = torch.autograd.grad(
            out.sum(), z_flat_req, create_graph=create_graph
        )[0]
    return grad


def npf_T_omega(
    z_flat: torch.Tensor,
    icnn_model: NPFInputConvexPotential,
    create_graph: bool,
) -> torch.Tensor:
    """T_ω(z) = ∇_z ψ_ω(z) using the model's live parameters (no functional_call).

    Mirrors the ICNN ``T_omega`` helper so NPF can drive a parameter-list
    BB+Armijo step in the same idiomatic way as ICNN.
    """
    z_in = z_flat.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi_val = icnn_model(z_in)
        grad = torch.autograd.grad(
            psi_val.sum(), z_in, create_graph=create_graph
        )[0]
    return grad.view_as(z_flat)
