"""NPF-style ICNN potential (Vesseron & Cuturi, 2024) for WDRO.

Potential:
    psi(z) = 0.5 * z^T (diag(delta^2) + A^T A) z + a^T z + phi^{NN}(z),
with phi^{NN} a deep ICNN block that re-injects quadratic forms at every layer.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icnn import NonNegativeLinear
from utils.common import softplus_inverse


class NPFQuadraticForm(nn.Module):
    """Stack of q convex quadratic forms  Q_{A_i, delta_i}(z) = ||delta_i o z||^2 + ||A_i z||^2."""

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
            delta_raw_init = softplus_inverse(self.init_eps)
        else:
            delta_raw_init = softplus_inverse(delta_init)
        self.delta_raw = nn.Parameter(
            torch.full((self.num_forms, self.input_dim), delta_raw_init)
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
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)
        delta = self.delta
        q_diag = ((z.unsqueeze(1) * delta.unsqueeze(0)) ** 2).sum(dim=-1)
        if self.A is not None:
            Az = torch.einsum("ord,bd->bor", self.A, z)
            q_lr = (Az ** 2).sum(dim=-1)
            return q_diag + q_lr
        return q_diag


class NPFInputConvexBlock(nn.Module):
    """Deep ICNN in the NPF style with per-layer quadratic re-injection."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...],
        rank_per_layer: int = 1,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        softplus_beta: float = 20.0,
        identity_init: bool = True,
        init_eps: float = 1e-3,
    ):
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("NPFInputConvexBlock requires at least 1 hidden layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(w) for w in hidden_sizes)
        self.rank_per_layer = int(rank_per_layer)
        self.activation_name = activation.lower()
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        self.identity_init = bool(identity_init)
        self.init_eps = float(init_eps)

        self.q_blocks = nn.ModuleList()
        self.b_linears = nn.ModuleList()
        self.w_linears: nn.ModuleList = nn.ModuleList()

        for l, width in enumerate(self.hidden_sizes):
            self.q_blocks.append(
                NPFQuadraticForm(
                    input_dim=self.input_dim,
                    num_forms=width,
                    rank=self.rank_per_layer,
                    delta_init=0.0,
                    init_eps=self.init_eps,
                )
            )
            self.b_linears.append(nn.Linear(self.input_dim, width, bias=True))
            if l == 0:
                self.w_linears.append(None)  # type: ignore[arg-type]
            else:
                self.w_linears.append(
                    NonNegativeLinear(
                        in_features=self.hidden_sizes[l - 1],
                        out_features=width,
                        bias=False,
                        init_mode="principled",
                    )
                )

        self.q_out = NPFQuadraticForm(
            input_dim=self.input_dim,
            num_forms=1,
            rank=self.rank_per_layer,
            delta_init=0.0,
            init_eps=self.init_eps,
        )
        self.b_out = nn.Linear(self.input_dim, 1, bias=True)
        self.w_out = NonNegativeLinear(
            in_features=self.hidden_sizes[-1],
            out_features=1,
            bias=False,
            init_mode="principled",
        )

        if self.identity_init:
            self._apply_identity_init()

    def _apply_identity_init(self):
        eps = self.init_eps
        delta_raw_init = softplus_inverse(eps) if eps > 0.0 else -1e3
        with torch.no_grad():
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
        if self.activation_name == "elu":
            return F.elu(u, alpha=self.elu_alpha)
        if self.activation_name == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * u) / beta
        if self.activation_name == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported activation '{self.activation_name}'.")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)

        q0 = self.q_blocks[0](z)
        b0 = self.b_linears[0](z)
        h = self._act(q0 + b0)

        for l in range(1, len(self.hidden_sizes)):
            ql = self.q_blocks[l](z)
            bl = self.b_linears[l](z)
            wl = self.w_linears[l](h)
            h = self._act(wl + ql + bl)

        out = self.w_out(h)
        out = out + self.q_out(z)
        out = out + self.b_out(z)
        return out.view(-1)


class NPFResidualPotential(nn.Module):
    """Top-level convex potential:
        psi(z) = 0.5 z^T (diag(delta^2) + A^T A) z + a^T z + phi^{NN}(z).

    Initialization follows the paper exactly: principled (Hoedt-Klambauer
    LogNormal) draws on every non-negative weight (always retained) AND
    the identity init that zeros the input-to-hidden, output linear, and
    outer/per-layer quadratic terms (down to eps scale) are applied
    JOINTLY so that T(z) = grad psi(z) ≈ z at t=0.
    """

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
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.init_eps = float(init_eps)

        self.outer_delta_raw = nn.Parameter(
            torch.full((self.input_dim,), softplus_inverse(1.0))
        )
        if outer_rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(outer_rank * self.input_dim, 1))
                self.outer_A = nn.Parameter(std * torch.randn(outer_rank, self.input_dim))
            else:
                self.outer_A = nn.Parameter(torch.zeros(outer_rank, self.input_dim))
        else:
            self.register_parameter("outer_A", None)

        self.outer_a = nn.Parameter(torch.zeros(self.input_dim))

        self.phi_nn = NPFInputConvexBlock(
            input_dim=self.input_dim,
            hidden_sizes=hidden_sizes,
            rank_per_layer=inner_rank,
            activation=activation,
            elu_alpha=elu_alpha,
            softplus_beta=softplus_beta,
            identity_init=True,
            init_eps=self.init_eps,
        )

    @property
    def outer_delta(self) -> torch.Tensor:
        return F.softplus(self.outer_delta_raw)

    def outer_P(self) -> torch.Tensor:
        P = torch.diag(self.outer_delta.pow(2))
        if self.outer_A is not None:
            P = P + self.outer_A.t() @ self.outer_A
        return P

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)

        q_diag = 0.5 * (self.outer_delta.pow(2).unsqueeze(0) * z.pow(2)).sum(dim=-1)
        if self.outer_A is not None:
            Az = z @ self.outer_A.t()
            q_lr = 0.5 * (Az ** 2).sum(dim=-1)
        else:
            q_lr = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        linear = z @ self.outer_a
        phi = self.phi_nn(z)
        return q_diag + q_lr + linear + phi
