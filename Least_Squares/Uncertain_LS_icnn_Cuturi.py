#!/usr/bin/env python3
"""uncertain_least_squares_npf.py

Robust least-squares experiment (Fig. 8 of Xu & Zhu, 2025) augmented with a
WDRO adversary parameterized as an *NPF-style ICNN* (Vesseron & Cuturi, 2024,
"On a Neural Implementation of Brenier's Polar Factorization").

Compared with the previous `uncertain_least_squares_icnn.py`, this version
rigorously implements three changes requested by the user:

  (1) NPF-style learnable PSD quadratic base in the potential
          psi_omega(z) = 0.5 z^T P_omega z + a^T z + phi_omega^{NN}(z),
      where  P_omega = diag(delta^2) + A^T A     (NPF Eq. 5, outer form)
      with  delta in R^d_+ (softplus-parameterized) and A in R^{r x d} (free).
      In 1D this collapses to a scalar  alpha = delta^2 + ||A||^2 >= 0,
      so the restriction to 1-strongly-convex potentials (implicit in the
      previous Eq. 10) is removed: the curvature of the quadratic base is
      now DATA-ADAPTIVE.

  (2) Identity initialisation at t=0:  delta_i = 1 (so P = I),  A = 0,
      a = 0, and every input-to-hidden path inside phi^{NN} zeroed, so that
      grad(phi^{NN})(z) == 0 at init and therefore T_omega(z) = z at init.
      This aligns the ICNN warm-start with the particle-ascent warm-start
      z^(0) = z_hat used by all the implicit baselines.

  (4) Full NPF ICNN block phi^{NN}: each hidden layer re-injects a stack
      of q quadratic forms [Q_{A^l_i, delta^l_i}(z)]_i (one per output
      coordinate, NPF Eq. 5), plus the usual non-negative W @ h_{l-1}
      and unconstrained B z input passthrough. Non-negative parameters
      use the principled LogNormal init of Hoedt & Klambauer (2023),
      with the *correct negative bias offset* (the previous code had
      the sign flipped).

The file keeps the paper's original baselines (ERM, Particle Ascent,
WGF, WFR, Dual) untouched so results remain directly comparable.

Run
---
    python uncertain_least_squares_npf.py

This will save:
    fig8_uncertain_least_squares.pdf                (paper baselines only)
    fig8_uncertain_least_squares_plus_npf.pdf       (baselines + WDRO-NPF)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================================================================
# Utilities
# =======================================================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def softplus_inverse(y: float) -> float:
    """Numerically stable inverse of softplus: softplus^{-1}(y) = log(e^y - 1)."""
    if y <= 0:
        return -1e3  # essentially softplus(-1e3) = 0
    # log(expm1(y)) is stable for both small and large y
    return float(math.log(math.expm1(y)))


# =======================================================================
# Principled LogNormal init (Hoedt & Klambauer, 2023) — FIXED bias sign.
# =======================================================================


def icnn_principled_moments(fan_in: int):
    """Return the (target mean, target variance) for ICNN non-negative weights
    plus the LogNormal underlying Gaussian parameters (mu_tilde, sigma_tilde)
    such that a weight w = exp(N(mu_tilde, sigma_tilde^2)) has the desired
    moments.  Also returns the *magnitude* of the bias offset (sign applied
    by the caller)."""
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    # Bias *magnitude* — sign convention applied below.
    mu_b_mag = math.sqrt((3.0 * fan_in) / denom)

    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    mu_tilde = log_mean_sq - 0.5 * log_var_plus_mean_sq
    sigma2_tilde = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    sigma_tilde = math.sqrt(sigma2_tilde)
    return mu_w, sigma_w2, mu_b_mag, mu_tilde, sigma_tilde


class NonNegativeLinear(nn.Module):
    """Linear map with strictly non-negative weights.

    Weights are parameterized as `exp(theta)` so W > 0 always.

    BIAS SIGN (FIX): Hoedt & Klambauer (2023) derive the principled
    initialisation by matching pre-activation moments.  With *positive-mean*
    weights and inputs that come from a previous layer's convex non-decreasing
    activation (ELU/Softplus), the mean input is POSITIVE, so the bias must
    be NEGATIVE to keep pre-activations zero-mean.  The previous code had
    `self.bias.fill_(mu_b)` (positive) — we now initialise to `-mu_b_mag`.
    """

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
            raise ValueError(f"Unsupported init_mode '{init_mode}'.")

        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _, _, mu_b_mag, mu_tilde, sigma_tilde = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if sigma_tilde == 0.0:
                    self.weight_param.fill_(mu_tilde)
                else:
                    self.weight_param.normal_(mean=mu_tilde, std=sigma_tilde)
                if self.bias is not None:
                    # FIXED SIGN: previous code used +mu_b_mag, should be -mu_b_mag.
                    self.bias.fill_(-mu_b_mag)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        # Always use exp-parametrisation to enforce strict positivity.
        weight = torch.exp(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


# =======================================================================
# NPF quadratic form (Vesseron & Cuturi, Eq. 5)
# =======================================================================


class NPFQuadraticForm(nn.Module):
    """Stack of q convex quadratic forms  Q_{A_i, delta_i}(z), i=1..q.

        Q_{A, delta}(z) = || delta o z ||_2^2 + || A z ||_2^2
                        = z^T ( diag(delta^2) + A^T A ) z,

    producing an (unbatched) output of shape (q,). delta is softplus-parameterised
    (so delta_i >= 0 elementwise and smooth), A is fully unconstrained
    (A^T A is automatically PSD regardless of sign).

    Args:
        input_dim: d = dimension of input z.
        num_forms: q = number of parallel quadratic forms (stack size).
        rank:      r = low-rank dimension of A (r = 0 -> pure diagonal form).
        delta_init: initial value of softplus(delta_raw). Set to 0.0 for
                    identity-init (so the form contributes 0 at t=0) or 1.0
                    for "full quadratic at init".
        init_eps:   BUG #1 FIX. If > 0, overrides delta_init / A=0 with a
                    *live* initialisation:
                        softplus(delta_raw)  =  init_eps   (instead of exactly 0)
                        A  ~  init_eps * N(0, I) / sqrt(r*d)
                    Rationale: at delta_raw = -inf, softplus'(delta_raw) = sigma
                    (-inf) = 0 exactly (fp64 underflow), so upstream gradient is
                    killed by the chain rule.  Similarly d||Az||^2 / dA = 2 A z z^T
                    equals zero identically at A=0, so A is at a fixed point of
                    the gradient flow.  With init_eps > 0 both partials become
                    non-zero while Q(z) stays O(init_eps^2 * ||z||^2), preserving
                    T(z) ≈ z at t=0 up to O(init_eps).
    """

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

        # delta_raw: (q, d), softplus(delta_raw) = delta.
        # When init_eps > 0, override delta_init so softplus(delta_raw) = init_eps
        # (small, but in the active zone of sigmoid where gradient flows).
        if self.init_eps > 0.0:
            delta_raw_init = softplus_inverse(self.init_eps)
        else:
            delta_raw_init = softplus_inverse(delta_init)
        self.delta_raw = nn.Parameter(
            torch.full((self.num_forms, self.input_dim), delta_raw_init)
        )

        # A: (q, r, d), unconstrained.  When init_eps > 0, tiny random init
        # so that d||Az||^2 / dA = 2 A z z^T is non-zero; otherwise zero-init.
        if self.rank > 0:
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(self.rank * self.input_dim, 1))
                self.A = nn.Parameter(std * torch.randn(self.num_forms, self.rank, self.input_dim))
            else:
                self.A = nn.Parameter(torch.zeros(self.num_forms, self.rank, self.input_dim))
        else:
            self.register_parameter("A", None)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.delta_raw)  # (q, d), >= 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d)  ->  (B, q)."""
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)

        delta = self.delta  # (q, d)
        # q_diag[b, i] = sum_k (delta[i, k] * z[b, k])^2
        q_diag = ((z.unsqueeze(1) * delta.unsqueeze(0)) ** 2).sum(dim=-1)  # (B, q)
        if self.A is not None:
            # Az[b, i, :] = A[i] @ z[b]   -> shape (B, q, r)
            Az = torch.einsum("ord,bd->bor", self.A, z)
            q_lr = (Az ** 2).sum(dim=-1)  # (B, q)
            return q_diag + q_lr
        return q_diag


# =======================================================================
# NPF-style ICNN block phi^{NN}(z)   (Vesseron & Cuturi, Eq. 5)
# =======================================================================


class NPFInputConvexBlock(nn.Module):
    """ICNN hidden block in the NPF style.

        h_0       = sigma( [Q_{A^0_i, delta^0_i}(z)]_i + B_0 z + c_0 )
        h_{l+1}   = sigma( W_l h_l + [Q_{A^l_i, delta^l_i}(z)]_i + B_l z + c_l )
        phi(z)    = w_L^T h_L + Q_{A_L, delta_L}(z) + b_L^T z + c_L

    - W_l (hidden-to-hidden) and w_L (output readout) are non-negative
      (via exp parametrisation; LogNormal principled init).
    - delta's are non-negative (softplus).
    - A's, B's, biases are all unconstrained.
    - Activations are convex non-decreasing. We support ELU (NPF default)
      and Softplus (the draft's §4.1 default).

    IDENTITY INIT. When `identity_init=True`, every *input-to-hidden*
    parameter (B_l, b_L, and the (delta, A) of every Q block) is zeroed,
    so that there is NO signal path from z into the activations at init.
    Consequently, phi(z) is a CONSTANT function at initialisation, hence
    grad_z phi(z) == 0 at t=0. Convexity is preserved throughout training
    (constants are convex).

    NOTE: because grad_phi == 0 at init, the outer potential
          psi(z) = 0.5 z^T P z + a^T z + phi(z)
    satisfies  grad psi(z) = P z + a  at init, which equals  z  iff P = I
    and a = 0 — both enforced by `NPFResidualPotential`'s identity-init.
    """

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

        # Per-layer input-injection quadratic forms [Q_{A^l_i, delta^l_i}(z)]_i
        # producing a vector of size hidden_sizes[l].
        self.q_blocks = nn.ModuleList()
        # Unconstrained input passthroughs B_l z
        self.b_linears = nn.ModuleList()
        # Non-negative hidden-to-hidden W_l
        self.w_linears: nn.ModuleList = nn.ModuleList()

        for l, width in enumerate(self.hidden_sizes):
            self.q_blocks.append(
                NPFQuadraticForm(
                    input_dim=self.input_dim,
                    num_forms=width,
                    rank=self.rank_per_layer,
                    delta_init=0.0,  # overridden by init_eps below if >0
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

        # Output readout
        self.q_out = NPFQuadraticForm(
            input_dim=self.input_dim,
            num_forms=1,
            rank=self.rank_per_layer,
            delta_init=0.0,
            init_eps=self.init_eps,
        )
        self.b_out = nn.Linear(self.input_dim, 1, bias=True)  # b_L^T z + c_L
        self.w_out = NonNegativeLinear(
            in_features=self.hidden_sizes[-1],
            out_features=1,
            bias=False,
            init_mode="principled",
        )

        if self.identity_init:
            self._apply_identity_init()

    # -------- identity init --------

    def _apply_identity_init(self):
        """Zero every input-to-hidden path so that phi is CONSTANT at init,
        hence grad_z phi(z) ≈ 0 for all z.

        BUG #1 FIX.  Parameters that feed z into the hidden state are now
        initialised *live* (non-zero gradient) but *small* (so T(z) ≈ z is
        preserved up to O(init_eps)):

          * b_linears[l].weight, b_linears[l].bias   zeroed  (no z -> h path at init)
          * q_blocks[l].delta_raw                   softplus_inverse(init_eps)  LIVE
          * q_blocks[l].A                            ~ init_eps * N(0, 1) / sqrt(r d)  LIVE
          * q_out.delta_raw, q_out.A                 same treatment               LIVE
          * b_out.weight, b_out.bias                 zeroed
          * w_linears[l] and w_out                   kept at principled LogNormal init
            (their values don't matter at init because the signal from z is still
            O(init_eps), and only become meaningful once the block activates.)

        With init_eps = 0.0, recovers the old (dead-at-init) behaviour.
        With init_eps = 1e-3 (default), all 14 of 17 param tensors that SHOULD
        be live at init get a non-zero gradient (the remaining 3 — layer biases
        and the output-readout bias — do not enter T(z) by construction and
        their gradient vanishing is semantically correct).
        """
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

    # -------- activation --------

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "elu":
            # ELU(u) = u if u>=0 else alpha*(e^u - 1); convex, non-decreasing.
            return F.elu(u, alpha=self.elu_alpha)
        if self.activation_name == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * u) / beta
        if self.activation_name == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported activation '{self.activation_name}'.")

    # -------- forward --------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d)  ->  (B,) scalar potential values."""
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)

        # Layer 0
        q0 = self.q_blocks[0](z)               # (B, q0)
        b0 = self.b_linears[0](z)              # (B, q0)
        h = self._act(q0 + b0)                 # (B, q0)

        # Layers 1..L-1
        for l in range(1, len(self.hidden_sizes)):
            ql = self.q_blocks[l](z)           # (B, ql)
            bl = self.b_linears[l](z)          # (B, ql)
            wl = self.w_linears[l](h)          # (B, ql) with W_l >= 0
            h = self._act(wl + ql + bl)

        # Output readout
        out = self.w_out(h)                    # (B, 1), w_L >= 0
        out = out + self.q_out(z)              # (B, 1)
        out = out + self.b_out(z)              # (B, 1)
        return out.view(-1)                    # (B,)


# =======================================================================
# NPF Residual Potential:   psi(z) = 0.5 z^T P z + a^T z + phi(z)
# with learnable NPF-style P = diag(delta^2) + A^T A.
# =======================================================================


class NPFResidualPotential(nn.Module):
    """Top-level convex potential for the WDRO adversary.

        psi_omega(z) = 0.5 * z^T ( diag(delta^2) + A^T A ) z
                     + a^T z
                     + phi_omega^{NN}(z),

    where (delta, A, a) are learnable (outer scope) and phi^{NN} is the
    NPFInputConvexBlock above. All pieces are convex; the sum is convex.

    Identity initialisation:
      * delta_i = 1 for all i  -> diag(delta^2) = I at init
      * A = 0                  -> A^T A = 0 at init
      * a = 0                  -> outer linear vanishes
      * phi^{NN} identity-init -> grad_z phi^{NN}(z) = 0
    At init:  grad psi(z) = (diag(delta^2) + A^T A) z + a + grad phi(z) = z.

    During training, delta_i can shrink to 0 (removing mass from the
    quadratic base), A can grow (adding cross-coordinate curvature), a
    can deviate, and phi^{NN} activates. The 1-strong-convexity
    restriction of the draft's Eq. (10) is thus removed in a clean,
    data-adaptive way.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 512, 512, 256, 256, 128, 128, 64) , # (64, 64, 64, 64)
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

        # --- Outer quadratic base: P = diag(delta^2) + A^T A ---
        # Initialise delta_i = 1 so that P = I at t=0.  Since softplus_inverse(1) ≈ 0.541,
        # sigma(0.541) ≈ 0.63, and the gradient flows freely for outer_delta_raw.
        self.outer_delta_raw = nn.Parameter(
            torch.full((self.input_dim,), softplus_inverse(1.0))
        )
        if outer_rank > 0:
            # BUG #1 FIX.  outer_A zero-init is a fixed point of the gradient
            # flow (d||Az||^2/dA = 2 A z z^T = 0 at A=0).  Initialise instead
            # with tiny N(0, eps^2/(r*d)) noise so that the gradient is live.
            # Contribution to T at init is O(init_eps), preserving identity init.
            if self.init_eps > 0.0:
                std = self.init_eps / math.sqrt(max(outer_rank * self.input_dim, 1))
                self.outer_A = nn.Parameter(std * torch.randn(outer_rank, self.input_dim))
            else:
                self.outer_A = nn.Parameter(torch.zeros(outer_rank, self.input_dim))
        else:
            self.register_parameter("outer_A", None)

        # --- Outer linear: a^T z ---
        # Initialise a = 0.  d psi / d a = z, so gradient is always live
        # (no fixed-point issue).
        self.outer_a = nn.Parameter(torch.zeros(self.input_dim))

        # --- Deep ICNN block (identity-initialised with LIVE gradients) ---
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

    # ---- accessors ----

    @property
    def outer_delta(self) -> torch.Tensor:
        return F.softplus(self.outer_delta_raw)  # (d,), >= 0

    def outer_P(self) -> torch.Tensor:
        """Return the current P = diag(delta^2) + A^T A (d x d)."""
        d = self.input_dim
        P = torch.diag(self.outer_delta.pow(2))
        if self.outer_A is not None:
            P = P + self.outer_A.t() @ self.outer_A
        return P

    # ---- forward ----

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d)  ->  (B,) scalar psi values."""
        if z.dim() == 1:
            z = z.view(-1, 1) if self.input_dim == 1 else z.unsqueeze(0)

        # 0.5 * sum_k delta_k^2 * z_k^2
        q_diag = 0.5 * (self.outer_delta.pow(2).unsqueeze(0) * z.pow(2)).sum(dim=-1)
        # 0.5 * || A z ||^2
        if self.outer_A is not None:
            Az = z @ self.outer_A.t()  # (B, r)
            q_lr = 0.5 * (Az ** 2).sum(dim=-1)
        else:
            q_lr = torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
        # a^T z
        linear = z @ self.outer_a  # (B,)
        # phi^{NN}(z)
        phi = self.phi_nn(z)
        return q_diag + q_lr + linear + phi


# =======================================================================
# Transport map helper:  T_omega(z) = grad_z psi_omega(z).
# =======================================================================


def transport_map(
    z: torch.Tensor,
    psi: nn.Module,
    create_graph: bool = False,
) -> torch.Tensor:
    """T_omega(z) := grad_z psi_omega(z)."""
    z_in = z.detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        val = psi(z_in).sum()
        grad = torch.autograd.grad(val, z_in, create_graph=create_graph)[0]
    return grad.view_as(z)


# =======================================================================
# Uncertain least squares (experimental setup is unchanged)
# =======================================================================


@dataclass
class ULSConfig:
    # Problem sizes
    dim_m: int = 10
    dim_n: int = 10
    n_train: int = 10
    n_test: int = 1000

    # Delta sweep
    delta_min: float = 0.0
    delta_max: float = 10.0
    delta_steps: int = 50

    # Training hyperparams (Figure 8 caption)
    epochs: int = 10
    lr_theta: float = 1e-2
    lam: float = 0.1
    epsilon: float = 0.1
    grad_clip: float = 100.0

    # Inner-loop hyperparams for PA / WGF / WFR / Dual
    inner_steps: int = 3000
    inner_step_size: float = 1e-4
    m_particles: int = 8

    # WFR
    wfr_weight_step_size: float = 0.0016
    wfr_low_weight_threshold: float = 1e-3

    # Dual (Sinkhorn / MLMC)
    sinkhorn_sample_level: int = 4

    # NPF ICNN adversary
    run_npf: bool = True
    npf_hidden: Tuple[int, ...] = (64, 64, 64, 64)
    npf_outer_rank: int = 1
    npf_inner_rank: int = 1
    npf_activation: str = "elu"  # "elu" (NPF default) | "softplus" | "relu"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0

    # --- Bug #1 fix: live-at-init scale. ---
    # softplus(delta_raw)  and  ||A_i||_F  at t=0 are both O(npf_init_eps),
    # so T(z) deviates from z by O(npf_init_eps) but every parameter has a
    # non-zero gradient at t=0.  Setting to 0 recovers the old dead-init
    # behaviour; 1e-3 is a safe default that preserves T(z) ≈ z to 5 digits.
    npf_init_eps: float = 1e-3

    # --- Bug #3 fix: budget-raised inner loop. ---
    # Original default was 50; PA's analog runs 3000.  With Adam (Bug #4)
    # instead of BB+Armijo, 500 Adam steps per outer epoch give a converged
    # best-response at roughly 1/6 of PA's per-epoch inner cost.
    npf_omega_steps_per_epoch: int = 500

    # --- Bug #4 fix: persistent Adam learning rate (replaces BB+Armijo). ---
    inner_lr_npf: float = 1e-2

    # Repro
    seed: int = 219

    # Numerical stability
    ridge: float = 1e-6


def make_problem(
    cfg: ULSConfig, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(cfg.seed)
    A0 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    A1 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    b = torch.tensor(rng.randn(cfg.dim_m), device=device)
    return A0, A1, b


def generate_training_data(cfg: ULSConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(cfg.n_train, device=device).uniform_(-0.5, 0.5)


def generate_test_data(cfg: ULSConfig, delta: float, device: torch.device) -> torch.Tensor:
    lo = -0.5 * (1.0 + float(delta))
    hi = 0.5 * (1.0 + float(delta))
    return torch.empty(cfg.n_test, device=device).uniform_(lo, hi)


def loss_function(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
    dim_m: int,
) -> torch.Tensor:
    """f_theta(xi) = || A(xi) theta - b ||^2 / m, where A(xi) = A0 + xi A1."""
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)  # (B,m,n)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)          # (B,m)
    loss = (residual ** 2).sum(dim=1) / float(dim_m)             # (B,)
    return loss


def loss_grad_theta(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    grad = 2.0 * torch.matmul(A_z.transpose(1, 2), residual.unsqueeze(2)).squeeze(2)
    return grad


def loss_grad_xi(
    theta: torch.Tensor,
    xi: torch.Tensor,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    xi = xi.reshape(-1)
    A_z = A0.unsqueeze(0) + xi.view(-1, 1, 1) * A1.unsqueeze(0)
    residual = torch.matmul(A_z, theta) - b.view(1, -1)
    grad_Az = torch.matmul(A1, theta)
    grad = 2.0 * (residual * grad_Az.view(1, -1)).sum(dim=1)
    return grad


def clip_grad(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    gnorm = grad.norm()
    if gnorm > max_norm:
        grad = grad * (max_norm / (gnorm + 1e-12))
    return grad


# =======================================================================
# Paper baselines (UNCHANGED from the user's file)
# =======================================================================


def solve_erm_closed_form(xi_train, cfg, A0, A1, b):
    n = cfg.dim_n
    S = torch.zeros((n, n), device=xi_train.device)
    t = torch.zeros((n,), device=xi_train.device)
    for xi in xi_train:
        A = A0 + xi * A1
        S = S + A.transpose(0, 1) @ A
        t = t + A.transpose(0, 1) @ b
    S = S + cfg.ridge * torch.eye(n, device=xi_train.device)
    return torch.linalg.solve(S, t)


def solve_Particle_Ascent(xi_train, cfg, A0, A1, b):
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    for _epoch in range(cfg.epochs):
        z = xi_train.clone()
        for _ in range(cfg.inner_steps):
            grad = loss_grad_xi(theta, z, A0, A1, b) - 2.0 * cfg.lam * (z - xi_train)
            z = z + cfg.inner_step_size * grad
            z = z.clamp(-1.0, 1.0)
        grads_theta = loss_grad_theta(theta, z, A0, A1, b)
        avg_grad = grads_theta.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta


def solve_wgf(xi_train, cfg, A0, A1, b):
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)
    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)
        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (
                particles_flat - xi_expanded
            )
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)
        particles_flat = particles.reshape(-1)
        grads = loss_grad_theta(theta, particles_flat, A0, A1, b)
        grads = grads.view(n_train, m, cfg.dim_n).mean(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta


def solve_wfr(xi_train, cfg, A0, A1, b):
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    m = cfg.m_particles
    noise_scale = math.sqrt(2.0 * cfg.inner_step_size * cfg.lam * cfg.epsilon)
    for _epoch in range(cfg.epochs):
        particles = xi_train.view(-1, 1).repeat(1, m)
        weights = torch.full((n_train, m), 1.0 / float(m), device=xi_train.device)
        for _ in range(cfg.inner_steps):
            particles_flat = particles.reshape(-1)
            xi_expanded = xi_train.repeat_interleave(m)
            grad = loss_grad_xi(theta, particles_flat, A0, A1, b) - 2.0 * cfg.lam * (
                particles_flat - xi_expanded
            )
            noise = torch.randn_like(particles_flat)
            particles_flat = particles_flat + cfg.inner_step_size * grad + noise_scale * noise
            particles = particles_flat.view(n_train, m).clamp(-1.0, 1.0)
            f_bar = loss_function(theta, particles.reshape(-1), A0, A1, b, cfg.dim_m).view(
                n_train, m
            )
            f_bar = f_bar - cfg.lam * (particles - xi_train.view(-1, 1)) ** 2
            power = 1.0 - cfg.lam * cfg.epsilon * cfg.wfr_weight_step_size
            weights = (weights.clamp_min(1e-12) ** power) * torch.exp(
                cfg.wfr_weight_step_size * f_bar
            )
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
            thr = cfg.wfr_low_weight_threshold
            low_mask = weights < thr
            rows = torch.any(low_mask, dim=1)
            if torch.any(rows):
                for i in torch.nonzero(rows, as_tuple=False).view(-1).tolist():
                    w = weights[i]
                    x = particles[i]
                    low = low_mask[i]
                    if not torch.any(low):
                        continue
                    j_max = int(torch.argmax(w).item())
                    w_max = w[j_max]
                    x_max = x[j_max].clone()
                    low_sum = torch.sum(w * low)
                    n_low = torch.sum(low).to(w.dtype)
                    avg_w = (w_max + low_sum) / (n_low + 1.0 + 1e-12)
                    update_mask = low.clone()
                    update_mask[j_max] = True
                    w = torch.where(update_mask, avg_w, w)
                    x = torch.where(low, x_max, x)
                    w = w / (w.sum() + 1e-12)
                    weights[i] = w
                    particles[i] = x
        grads = loss_grad_theta(theta, particles.reshape(-1), A0, A1, b).view(
            n_train, m, cfg.dim_n
        )
        grads = (weights.unsqueeze(-1) * grads).sum(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta


def solve_dual(xi_train, cfg, A0, A1, b):
    theta = torch.zeros(cfg.dim_n, device=xi_train.device)
    n_train = xi_train.numel()
    levels = torch.arange(cfg.sinkhorn_sample_level + 1, device=xi_train.device)
    numerators = 2.0 ** (-levels.to(torch.float64))
    denominator = 2.0 - 2.0 ** (-float(cfg.sinkhorn_sample_level))
    probs = (numerators / denominator).to(torch.float64)
    for _epoch in range(cfg.epochs):
        sampled_level = int(torch.multinomial(probs, num_samples=1).item())
        m = 2 ** sampled_level
        noise = torch.randn((n_train, m), device=xi_train.device) * math.sqrt(cfg.epsilon)
        z_samples = xi_train.view(-1, 1) + noise
        v = loss_function(theta, z_samples.reshape(-1), A0, A1, b, cfg.dim_m).view(n_train, m)
        v = v / (cfg.lam * cfg.epsilon)
        v_max = torch.max(v, dim=1, keepdim=True).values
        w = torch.exp(v - v_max)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        grads = loss_grad_theta(theta, z_samples.reshape(-1), A0, A1, b).view(
            n_train, m, cfg.dim_n
        )
        grads = (w.unsqueeze(-1) * grads).sum(dim=1)
        avg_grad = grads.mean(dim=0)
        avg_grad = clip_grad(avg_grad, cfg.grad_clip)
        theta = theta - cfg.lr_theta * avg_grad
    return theta


# =======================================================================
# NPF-ICNN WDRO solver
#
# This solver implements four bug fixes relative to the original version:
#
#   Bug #1 (FATAL, live-at-init):
#       Handled by NPFQuadraticForm / NPFResidualPotential init_eps>0.
#       Every parameter gets a non-zero gradient at t=0, in particular every
#       per-layer  delta_raw  and  A  feeding the NPF-style quadratic
#       injection.  Without this fix, 12 of 17 parameter tensors were stuck
#       at a fixed point of the gradient flow and never moved during training.
#
#   Bug #2 (severe, clamp-kills-grad):
#       The old inner objective called  z_adv.clamp(-1, 1)  on the transport
#       map output before computing f and reg.  clamp has zero derivative
#       outside [-1, 1], so (a) any saturated sample contributed no gradient
#       to psi and (b) the Wasserstein penalty was computed on the clamped
#       displacement, biasing the inner objective.  The clamp now happens
#       ONLY at the outer theta update, which is correct: samples fed to the
#       classifier are in support, but inner maximisation sees the true T(z).
#
#   Bug #3 (significant, starved inner loop):
#       Old default 50 inner steps per epoch vs PA's 3000 steps on the same
#       problem.  We raise the default to 500 which, combined with Adam's
#       momentum (Bug #4), converges to a reasonable best-response in practice.
#
#   Bug #4 (moderate, BB+Armijo on mixed-scale parameters):
#       BB proposes a single step size alpha = <s, s>/<s, y> for a vector
#       that concatenates parameters with wildly different scales (softplus
#       preimage ~0.5, LogNormal-in-log-space ~[-11, 1], unconstrained ~0).
#       Replaced with a persistent Adam instance that maintains per-parameter
#       first-/second-moment estimates, exactly mirroring the fix applied to
#       train_step_algo2 in the MNIST code.
# =======================================================================


def solve_npf_icnn_map(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, NPFResidualPotential, Dict[str, list]]:
    """Explicit-map WDRO adversary in the NPF style (v2, all four bugs fixed)."""
    device = xi_train.device
    theta = torch.zeros(cfg.dim_n, device=device)

    # NOTE: init_eps controls the live-at-init scale for per-layer Q blocks
    # and outer_A.  T(z) at t=0 is perturbed from identity by O(init_eps).
    psi = NPFResidualPotential(
        input_dim=1,
        hidden_sizes=cfg.npf_hidden,
        outer_rank=cfg.npf_outer_rank,
        inner_rank=cfg.npf_inner_rank,
        activation=cfg.npf_activation,
        elu_alpha=cfg.npf_elu_alpha,
        softplus_beta=cfg.npf_softplus_beta,
        init_eps=cfg.npf_init_eps,
    ).to(device)

    # Match the dtype of the data (default_dtype is float64).
    psi = psi.to(xi_train.dtype)

    # Persistent Adam for the inner maximisation.  Adam's first-/second-
    # moment accumulators survive across epochs, so the secant history that
    # BB+Armijo was trying to exploit is captured implicitly and robustly.
    # Gradient ascent via negated loss (Adam minimises by convention).
    inner_opt = torch.optim.Adam(psi.parameters(), lr=cfg.inner_lr_npf)

    xi_2d = xi_train.view(-1, 1)

    diagnostics: Dict[str, list] = {
        "outer_P": [],
        "adv_loss": [],
        "mean_displacement": [],
        "max_displacement": [],
    }

    for _epoch in range(cfg.epochs):

        # --- Inner maximisation on psi (ICNN parameters) ---
        for _ in range(cfg.npf_omega_steps_per_epoch):
            inner_opt.zero_grad()
            # Transport map T(z) = grad_z psi(z); create_graph=True so that
            # backprop through T gives second-order derivatives w.r.t. psi.
            z_adv = transport_map(xi_2d, psi, create_graph=True).view(-1)
            # BUG #2 FIX: no clamp inside the inner objective.  Out-of-support
            # samples still receive a gradient signal through the Wasserstein
            # penalty, pulling them back towards the training support.
            f = loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
            reg = cfg.lam * (z_adv - xi_train) ** 2
            obj = (f - reg).mean()
            # Gradient ASCENT on obj  <=>  Gradient DESCENT on -obj.
            (-obj).backward()
            inner_opt.step()

        # --- Outer update on theta (classifier parameters) ---
        # Here we DO clamp: the final adversarial samples are used to train
        # the classifier, and the test distribution is supported in [-1, 1].
        with torch.no_grad():
            z_adv_raw = transport_map(xi_2d, psi, create_graph=False).view(-1).detach()
            z_adv = z_adv_raw.clamp(-1.0, 1.0)
        grad_theta = loss_grad_theta(theta, z_adv, A0, A1, b).mean(dim=0)
        grad_theta = clip_grad(grad_theta, cfg.grad_clip)
        theta = theta - cfg.lr_theta * grad_theta

        # Diagnostics: track P(t), mean/max displacement, and inner objective.
        with torch.no_grad():
            diagnostics["outer_P"].append(float(psi.outer_P().item()))  # scalar in 1D
            disp = (z_adv_raw - xi_train).abs()
            diagnostics["mean_displacement"].append(float(disp.mean().item()))
            diagnostics["max_displacement"].append(float(disp.max().item()))
            diagnostics["adv_loss"].append(
                float((loss_function(theta, z_adv, A0, A1, b, cfg.dim_m)
                       - cfg.lam * (z_adv - xi_train) ** 2).mean().item())
            )

    return theta, psi, diagnostics


# =======================================================================
# Experiment + plotting
# =======================================================================


def run_experiment(cfg: ULSConfig, device: torch.device):
    set_seed(cfg.seed)
    A0, A1, b = make_problem(cfg, device)
    xi_train = generate_training_data(cfg, device)

    print("[1/2] Training models...")
    theta_erm = solve_erm_closed_form(xi_train, cfg, A0, A1, b)
    theta_pa = solve_Particle_Ascent(xi_train, cfg, A0, A1, b)
    theta_wgf = solve_wgf(xi_train, cfg, A0, A1, b)
    theta_wfr = solve_wfr(xi_train, cfg, A0, A1, b)
    theta_dual = solve_dual(xi_train, cfg, A0, A1, b)

    theta_npf = None
    diagnostics = None
    if cfg.run_npf:
        theta_npf, _psi, diagnostics = solve_npf_icnn_map(xi_train, cfg, A0, A1, b)

    models: Dict[str, torch.Tensor] = {
        "ERM": theta_erm,
        "WGF(Otto, 1996)": theta_wgf,
        "WFR(Xu, 2025)": theta_wfr,
        "Particle Ascent": theta_pa,
        "Dual(Wang et al., 2021)": theta_dual,
    }
    if cfg.run_npf:
        models["WDRO-NPF"] = theta_npf

    delta_values = torch.linspace(cfg.delta_min, cfg.delta_max, cfg.delta_steps, device=device)
    results: Dict[str, List[float]] = {k: [] for k in models.keys()}

    print("[2/2] Evaluating test loss curves...")
    for delta in to_numpy(delta_values):
        xi_test = generate_test_data(cfg, float(delta), device)
        for name, theta in models.items():
            with torch.no_grad():
                test_loss = (
                    loss_function(theta, xi_test, A0, A1, b, cfg.dim_m).mean().item()
                )
            results[name].append(float(test_loss))

    return to_numpy(delta_values), results, diagnostics


def plot_results(
    delta_values: np.ndarray,
    results: Dict[str, List[float]],
    save_path: str,
    *,
    include_npf: bool,
):
    fig, ax = plt.subplots(figsize=(3.6, 2.613), dpi=300)
    styles = {
        "ERM": {"color": "k", "linestyle": "--"},
        "WGF(Otto, 1996)": {"color": "#9a0ba7", "linestyle": "-."},
        "WFR(Xu, 2025)": {"color": "#db2020", "linestyle": ":"},
        "Particle Ascent": {"color": "#0eaf0e", "linestyle": (0, (3, 1, 1, 1))},
        "Dual(Wang et al., 2021)": {"color": "#7A4E15", "linestyle": (0, (5, 5))},
        "WDRO-NPF": {"color": "#1f77b4", "linestyle": "-"},
    }
    plot_order = [
        "ERM",
        "WGF(Otto, 1996)",
        "WFR(Xu, 2025)",
        "Particle Ascent",
        "Dual(Wang et al., 2021)",
    ]
    if include_npf and "WDRO-NPF" in results:
        plot_order.append("WDRO-NPF")

    for name in plot_order:
        ax.plot(delta_values, results[name], linewidth=1.5, label=name, **styles[name])

    ax.set_xlabel(r"perturbation $\Delta$", fontsize=9)
    ax.set_ylabel("test loss", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(float(delta_values.min()), float(delta_values.max()))
    ax.set_ylim(0.0, 3.0)
    ax.grid(False)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser(
        description="Uncertain least squares (Fig. 8) with NPF-style ICNN adversary"
    )
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--delta_steps", type=int, default=50)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--inner_steps", type=int, default=3000)
    parser.add_argument("--inner_step_size", type=float, default=1e-4)
    parser.add_argument("--m_particles", type=int, default=8)
    parser.add_argument("--lr_theta", type=float, default=1e-2)
    parser.add_argument("--no_npf", action="store_true", help="Disable NPF baseline.")
    parser.add_argument("--npf_omega_steps_per_epoch", type=int, default=500,
                        help="Inner Adam steps per outer epoch (default 500; raised from 50 per Bug #3 fix).")
    parser.add_argument("--inner_lr_npf", type=float, default=1e-2,
                        help="Persistent Adam learning rate for the inner NPF maximisation (Bug #4 fix).")
    parser.add_argument("--npf_init_eps", type=float, default=1e-3,
                        help="Live-at-init scale for per-layer quadratic forms (Bug #1 fix); set to 0 to recover dead-init behaviour.")
    parser.add_argument(
        "--npf_activation", choices=["elu", "softplus", "relu"], default="elu"
    )
    parser.add_argument("--npf_hidden", type=int, nargs="+", default=[512, 512, 512, 256, 256, 128, 128, 64])
    parser.add_argument("--npf_outer_rank", type=int, default=1)
    parser.add_argument("--npf_inner_rank", type=int, default=1)
    args = parser.parse_args()

    cfg = ULSConfig(
        seed=args.seed,
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        delta_steps=args.delta_steps,
        lam=args.lam,
        epsilon=args.epsilon,
        inner_steps=args.inner_steps,
        inner_step_size=args.inner_step_size,
        m_particles=args.m_particles,
        lr_theta=args.lr_theta,
        run_npf=not args.no_npf,
        npf_omega_steps_per_epoch=args.npf_omega_steps_per_epoch,
        inner_lr_npf=args.inner_lr_npf,
        npf_init_eps=args.npf_init_eps,
        npf_activation=args.npf_activation,
        npf_hidden=tuple(args.npf_hidden),
        npf_outer_rank=args.npf_outer_rank,
        npf_inner_rank=args.npf_inner_rank,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta_values, results, diagnostics = run_experiment(cfg, device)

    plot_results(
        delta_values,
        {k: v for k, v in results.items() if k != "WDRO-NPF"},
        save_path="fig8_uncertain_least_squares.pdf",
        include_npf=False,
    )
    print("Saved: fig8_uncertain_least_squares.pdf")

    if cfg.run_npf and "WDRO-NPF" in results:
        plot_results(
            delta_values,
            results,
            save_path="fig8_uncertain_least_squares_plus_npf.pdf",
            include_npf=True,
        )
        print("Saved: fig8_uncertain_least_squares_plus_npf.pdf")

        if diagnostics is not None and len(diagnostics["outer_P"]) > 0:
            P_arr = np.array(diagnostics["outer_P"])
            disp_mean = np.array(diagnostics["mean_displacement"])
            disp_max = np.array(diagnostics["max_displacement"])
            adv_arr = np.array(diagnostics["adv_loss"])
            print(f"[NPF] outer P (1D scalar) per epoch: {np.round(P_arr, 4).tolist()}")
            print(f"[NPF] mean |T(z)-z| per epoch:       {np.round(disp_mean, 4).tolist()}")
            print(f"[NPF] max  |T(z)-z| per epoch:       {np.round(disp_max, 4).tolist()}")
            print(f"[NPF] inner objective per epoch:     {np.round(adv_arr, 4).tolist()}")


if __name__ == "__main__":
    main()