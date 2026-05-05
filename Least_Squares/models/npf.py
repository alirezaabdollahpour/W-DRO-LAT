"""NPF-style ICNN potential (Vesseron & Cuturi, 2024) for WDRO.

This is the same NPF parameterisation used by the RL adversary, adapted to
the least-squares scalar uncertainty setting.  The potential is

    psi(z) = 0.5*mu*||z||^2
             + 0.5 z^T (diag(delta^2) + A^T A) z + a^T z
             + phi^{NN}(z),

where phi^{NN} is an ICNN block with per-layer NPF quadratic injections.
When ``strong_convexity > 0``, the fixed ``mu I`` term carries the identity
map at initialisation and the learnable PSD residual starts at eps scale.
When ``strong_convexity == 0``, the older pure-NPF convention is recovered:
the learnable outer diagonal starts at one.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icnn import icnn_principled_moments


def _npf_softplus_inverse(y: float) -> float:
    """Numerically stable inverse of softplus for scalar initialisation."""
    if y <= 0.0:
        return -1e3
    return float(math.log(math.expm1(y)))


class NPFNonNegativeDense(nn.Module):
    """Non-negative dense layer with Hoedt-Klambauer LogNormal init."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.weight_param = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _, _, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
        with torch.no_grad():
            if tilde_sigma == 0.0:
                self.weight_param.fill_(tilde_mu)
            else:
                self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
            if self.bias is not None:
                self.bias.fill_(-mu_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.matmul(torch.exp(self.weight_param))
        if self.bias is not None:
            y = y + self.bias
        return y


class NPFQuadraticForm(nn.Module):
    """Stack of convex quadratic forms Q(z)=||delta o z||^2+||Az||^2."""

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

        delta0 = self.init_eps if self.init_eps > 0.0 else float(delta_init)
        self.delta_raw = nn.Parameter(
            torch.full((self.num_forms, self.input_dim), _npf_softplus_inverse(delta0))
        )
        if self.rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(self.rank * self.input_dim, 1))
                self.A = nn.Parameter(
                    std * torch.randn(self.num_forms, self.rank, self.input_dim)
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
        z_flat = z.view(z.size(0), -1)
        if z_flat.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {z_flat.size(1)}.")

        q_diag = ((z_flat.unsqueeze(1) * self.delta.unsqueeze(0)) ** 2).sum(dim=-1)
        if self.A is None:
            return q_diag
        Az = torch.einsum("ord,bd->bor", self.A, z_flat)
        return q_diag + (Az ** 2).sum(dim=-1)


class NPFInputConvexPotential(nn.Module):
    """NPF ICNN potential: fixed convex base + learnable PSD residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        outer_rank: int = 1,
        inner_rank: int = 1,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        softplus_beta: float = 20.0,
        init_eps: float = 1e-3,
        strong_convexity: float = 0.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("NPFInputConvexPotential needs at least one hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.outer_rank = int(outer_rank)
        self.inner_rank = int(inner_rank)
        self.activation = str(activation).lower()
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        self.init_eps = float(init_eps)
        self.strong_convexity = float(strong_convexity)

        outer_delta_init = self.init_eps if self.strong_convexity > 0.0 else 1.0
        self.outer_delta_raw = nn.Parameter(
            torch.full((self.input_dim,), _npf_softplus_inverse(outer_delta_init))
        )
        if self.outer_rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(self.outer_rank * self.input_dim, 1))
                self.outer_A = nn.Parameter(std * torch.randn(self.outer_rank, self.input_dim))
            else:
                self.outer_A = nn.Parameter(torch.zeros(self.outer_rank, self.input_dim))
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

    @property
    def outer_delta(self) -> torch.Tensor:
        return F.softplus(self.outer_delta_raw)

    def outer_P(self) -> torch.Tensor:
        eye = torch.eye(self.input_dim, dtype=self.outer_delta.dtype, device=self.outer_delta.device)
        P = self.strong_convexity * eye + torch.diag(self.outer_delta.pow(2))
        if self.outer_A is not None:
            P = P + self.outer_A.t() @ self.outer_A
        return P

    def init_as_identity(self) -> None:
        """Initialise so grad_z psi(z) is approximately z."""
        eps = self.init_eps
        delta_raw_init = _npf_softplus_inverse(eps) if eps > 0.0 else -1e3
        with torch.no_grad():
            outer_delta_init = eps if self.strong_convexity > 0.0 else 1.0
            self.outer_delta_raw.fill_(_npf_softplus_inverse(outer_delta_init))
            if self.outer_A is not None:
                if eps > 0.0:
                    std = eps / math.sqrt(max(self.outer_rank * self.input_dim, 1))
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
                    std = eps / math.sqrt(max(self.q_out.rank * self.q_out.input_dim, 1))
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
        if z_flat.size(1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {z_flat.size(1)}.")

        fixed_q = 0.5 * self.strong_convexity * z_flat.pow(2).sum(dim=-1)
        q_diag = 0.5 * (self.outer_delta.pow(2) * z_flat.pow(2)).sum(dim=-1)
        if self.outer_A is not None:
            Az = z_flat @ self.outer_A.t()
            q_lr = 0.5 * (Az ** 2).sum(dim=-1)
        else:
            q_lr = torch.zeros(z_flat.size(0), dtype=z_flat.dtype, device=z_flat.device)
        linear = z_flat @ self.outer_a

        h = self._act(self.q_blocks[0](z_flat) + self.b_linears[0](z_flat))
        for l in range(1, len(self.hidden_sizes)):
            h = self._act(
                self.w_linears[l](h)
                + self.q_blocks[l](z_flat)
                + self.b_linears[l](z_flat)
            )

        phi = (
            self.w_out(h).squeeze(-1)
            + self.q_out(z_flat).squeeze(-1)
            + self.b_out(z_flat).squeeze(-1)
        )
        return fixed_q + q_diag + q_lr + linear + phi


class NPFResidualPotential(NPFInputConvexPotential):
    """Backward-compatible name for older least-squares imports."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 512, 512, 256, 256, 128, 128, 64),
        outer_rank: int = 1,
        inner_rank: int = 1,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        softplus_beta: float = 20.0,
        init_eps: float = 1e-3,
        strong_convexity: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            outer_rank=outer_rank,
            inner_rank=inner_rank,
            activation=activation,
            elu_alpha=elu_alpha,
            softplus_beta=softplus_beta,
            init_eps=init_eps,
            strong_convexity=strong_convexity,
        )
        self.init_as_identity()
