"""OTT-style NPF ICNN potential and transport map.

This module mirrors the OTT-JAX ICNN layout used for the NPF paper:

* every input skip is a positive-definite potential block
  ``0.5 * x^T (diag(d) + A A^T) x + b^T x + c``;
* hidden-to-hidden maps are bias-free and optionally projected nonnegative;
* the scalar readout is part of the activation loop; and
* a separate final positive-definite potential is initialized to
  ``0.5 * ||x||^2`` when ``strong_convexity=1``.

The ``last_layer_diagonal`` mode is a controlled ablation: hidden input skips
are affine, while the final activated scalar layer keeps the only learned
input quadratic. The final identity positive-definite potential is still
present, matching the OTT base potential.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lecun_normal_(tensor: torch.Tensor, fan_in: int) -> None:
    """Approximate Flax/OTT's default lecun_normal initializer."""
    std = 1.0 / math.sqrt(max(int(fan_in), 1))
    nn.init.normal_(tensor, mean=0.0, std=std)


def _rectifier(name: str, x: torch.Tensor) -> torch.Tensor:
    name = name.lower()
    if name == "relu":
        return F.relu(x)
    if name == "softplus":
        return F.softplus(x)
    if name == "exp":
        # Legacy pretrained_INPUT_icnn parametrization: strictly positive,
        # never dead, gradient scale proportional to the effective weight.
        return torch.exp(x)
    raise ValueError(f"Unsupported NPF rectifier '{name}'.")


def _zero_rectifier_raw(name: str) -> float:
    name = name.lower()
    if name == "relu":
        return 0.0
    if name in {"softplus", "exp"}:
        return -1e3
    raise ValueError(f"Unsupported NPF rectifier '{name}'.")


def _softplus_inverse(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return -1e3
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def _softplus_inverse_tensor(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable elementwise inverse of the standard softplus."""
    out = torch.empty_like(value)
    large = value > 20.0
    if large.any():
        out[large] = value[large]
    if (~large).any():
        tiny = torch.finfo(value.dtype).tiny
        safe = value[~large].clamp_min(tiny)
        out[~large] = torch.log(torch.expm1(safe))
    return out


def _convex_init_corr_func(fan_in: int, target_corr: float) -> float:
    """Feature-correlation factor from Hoedt & Klambauer, Eq. 35."""
    fan_in = int(fan_in)
    rho = float(target_corr)
    mix_moment = math.sqrt(max(1.0 - rho * rho, 0.0)) + rho * math.acos(-rho)
    return fan_in * (math.pi - fan_in + (fan_in - 1.0) * mix_moment) / (
        2.0 * math.pi
    )


def convex_init_parameters(
    fan_in: int,
    *,
    target_var: float = 1.0,
    target_corr: float = 0.5,
    bias_noise: float = 0.0,
    alpha: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return principled ICNN init moments.

    The return value is ``(weight_mean_sq, weight_var, bias_mean, bias_var)``.
    It follows the public implementation accompanying arXiv:2312.12474 with
    the paper's default ``target_corr=0.5`` and deterministic bias shift.
    """
    fan_in = int(fan_in)
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive; got {fan_in}.")
    target_var = float(target_var)
    target_corr = float(target_corr)
    bias_noise = float(bias_noise)
    alpha = float(alpha)
    if target_var <= 0.0:
        raise ValueError(f"target_var must be positive; got {target_var}.")
    if not (0.0 < target_corr < 1.0):
        raise ValueError(
            f"target_corr must lie strictly between 0 and 1; got {target_corr}."
        )
    if not (0.0 <= bias_noise < 1.0):
        raise ValueError(f"bias_noise must lie in [0, 1); got {bias_noise}.")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative; got {alpha}.")

    corr_factor = _convex_init_corr_func(fan_in, target_corr)
    if corr_factor <= 0.0:
        raise ValueError(
            "convex-init correlation factor must be positive; "
            f"got {corr_factor} for fan_in={fan_in}, target_corr={target_corr}."
        )
    weight_mean_sq = target_corr / corr_factor
    relu_scale = 2.0 / (1.0 + alpha * alpha)
    weight_var = relu_scale * (1.0 - target_corr) / fan_in
    bias_var = 0.0
    if bias_noise > 0.0:
        weight_var *= 1.0 - bias_noise
        bias_var = bias_noise * (1.0 - target_corr) * target_var

    bias_mean = -fan_in * math.sqrt(weight_mean_sq * target_var / (2.0 * math.pi))
    return weight_mean_sq, weight_var, bias_mean, bias_var


def _convex_lognormal_params(fan_in: int) -> tuple[float, float, float]:
    """Return ``(log_mean, log_std, bias_mean)`` for positive ICNN weights."""
    weight_mean_sq, weight_var, bias_mean, _ = convex_init_parameters(fan_in)
    log_mom2 = math.log(weight_mean_sq + weight_var)
    log_mean = math.log(weight_mean_sq) - 0.5 * log_mom2
    log_var = max(log_mom2 - math.log(weight_mean_sq), 1e-12)
    return log_mean, math.sqrt(log_var), bias_mean


def _convex_init_positive_weight_(
    raw_weight: torch.Tensor,
    *,
    fan_in: int,
    rectifier: str,
) -> None:
    """Initialize a raw parameter so its projected weight is log-normal."""
    rectifier = rectifier.lower()
    log_mean, log_std, _ = _convex_lognormal_params(fan_in)
    projected = torch.empty_like(raw_weight)
    nn.init.normal_(projected, mean=log_mean, std=log_std)
    if rectifier == "exp":
        # exp(raw) with raw ~ N(log_mean, log_std) is exactly the target
        # log-normal weight distribution.
        raw_weight.copy_(projected)
        return
    projected.exp_()
    if rectifier == "relu":
        raw_weight.copy_(projected)
    elif rectifier == "softplus":
        raw_weight.copy_(_softplus_inverse_tensor(projected))
    else:
        raise ValueError(f"Unsupported NPF rectifier '{rectifier}'.")


def _convex_bias_mean(fan_in: int) -> float:
    _, _, bias_mean = _convex_lognormal_params(fan_in)
    return bias_mean


class NPFDense(nn.Module):
    """Bias-free dense layer with optional principled positive projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        pos_weights: bool = True,
        rectifier: str = "relu",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.pos_weights = bool(pos_weights)
        self.rectifier = str(rectifier).lower()
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.pos_weights:
                _convex_init_positive_weight_(
                    self.weight,
                    fan_in=self.in_features,
                    rectifier=self.rectifier,
                )
            else:
                _lecun_normal_(self.weight, self.in_features)

    def projected_weight(self) -> torch.Tensor:
        if self.pos_weights:
            return _rectifier(self.rectifier, self.weight)
        return self.weight

    def zero_(self) -> None:
        with torch.no_grad():
            if self.pos_weights:
                self.weight.fill_(_zero_rectifier_raw(self.rectifier))
            else:
                self.weight.zero_()

    def near_zero_(self, eps: float) -> None:
        eps = float(eps)
        if eps <= 0.0:
            self.zero_()
            return
        if not self.pos_weights:
            with torch.no_grad():
                _lecun_normal_(self.weight, self.in_features)
                self.weight.mul_(eps)
            return
        # Scale by fan-in so a wide positive matrix does not create a huge
        # constant residual path while still keeping non-zero derivatives.
        # Draw log-normal entries with mean eps/sqrt(fan_in): a constant fill
        # makes W_z exactly rank-1 and permutation-symmetric, which the inner
        # ascent then cannot break.
        target = eps / math.sqrt(max(self.in_features, 1))
        with torch.no_grad():
            log_w = torch.empty_like(self.weight)
            nn.init.normal_(log_w, mean=math.log(target) - 0.5, std=1.0)
            projected = log_w.exp()
            if self.rectifier == "relu":
                self.weight.copy_(projected)
            elif self.rectifier == "exp":
                self.weight.copy_(log_w)
            else:
                self.weight.copy_(_softplus_inverse_tensor(projected))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.projected_weight())


class NPFPosDefPotentials(nn.Module):
    """OTT PosDefPotentials in PyTorch.

    For each output potential i this computes

        0.5 * sum_j d_ji * x_j^2
        + 0.5 * sum_r (<a_ir, x>)^2
        + <b_i, x> + c_i.
    """

    def __init__(
        self,
        input_dim: int,
        num_potentials: int,
        *,
        rank: int = 0,
        use_linear: bool = True,
        use_bias: bool = True,
        diag_init: float = 1.0,
        diag_rectifier: str = "relu",
        low_rank_init: str = "lecun_normal",
        linear_init: str = "lecun_normal",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_potentials = int(num_potentials)
        self.rank = int(rank)
        self.use_linear = bool(use_linear)
        self.use_bias = bool(use_bias)
        self.diag_rectifier = str(diag_rectifier).lower()
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative; got {rank}.")
        if self.diag_rectifier not in {"relu", "softplus", "exp"}:
            raise ValueError(f"Unsupported NPF diag rectifier '{diag_rectifier}'.")

        # diag_init is the EFFECTIVE post-rectifier value for every rectifier;
        # store the matching raw parameter. With 'relu' the lambda-cost term of
        # the inner ascent can push raw entries negative, after which value and
        # gradient are exactly zero forever (a permanent dead zone that killed
        # the elementwise sample-dependent transport path in past runs); 'exp'
        # keeps the same effective init while making the diagonal never-dead.
        self.diag_kernel = nn.Parameter(
            torch.full(
                (self.input_dim, self.num_potentials),
                self._diag_raw_for(float(diag_init)),
            )
        )
        if self.rank > 0:
            self.quad_kernel = nn.Parameter(
                torch.empty(self.num_potentials, self.input_dim, self.rank)
            )
        else:
            self.register_parameter("quad_kernel", None)
        if self.use_linear:
            self.lin_kernel = nn.Parameter(torch.empty(self.input_dim, self.num_potentials))
        else:
            self.register_parameter("lin_kernel", None)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_potentials))
        else:
            self.register_parameter("bias", None)

        self.low_rank_init = str(low_rank_init).lower()
        self.linear_init = str(linear_init).lower()
        self.reset_parameters()

    def _diag_raw_for(self, effective: float) -> float:
        """Raw diag_kernel value whose rectified image equals ``effective``."""
        effective = float(effective)
        if self.diag_rectifier == "relu":
            return effective
        if self.diag_rectifier == "softplus":
            return _softplus_inverse(effective)
        if self.diag_rectifier == "exp":
            if effective <= 0.0:
                return -1e3
            return math.log(effective)
        raise ValueError(f"Unsupported NPF diag rectifier '{self.diag_rectifier}'.")

    @property
    def diag(self) -> torch.Tensor:
        return _rectifier(self.diag_rectifier, self.diag_kernel)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.quad_kernel is not None:
                if self.low_rank_init == "zeros":
                    self.quad_kernel.zero_()
                elif self.low_rank_init == "lecun_normal":
                    _lecun_normal_(self.quad_kernel, self.input_dim)
                else:
                    raise ValueError(f"Unsupported low-rank init '{self.low_rank_init}'.")
            if self.lin_kernel is not None:
                if self.linear_init == "zeros":
                    self.lin_kernel.zero_()
                elif self.linear_init == "lecun_normal":
                    _lecun_normal_(self.lin_kernel, self.input_dim)
                else:
                    raise ValueError(f"Unsupported linear init '{self.linear_init}'.")
            if self.bias is not None:
                self.bias.zero_()

    def zero_(self) -> None:
        with torch.no_grad():
            self.diag_kernel.fill_(_zero_rectifier_raw(self.diag_rectifier))
            if self.quad_kernel is not None:
                self.quad_kernel.zero_()
            if self.lin_kernel is not None:
                self.lin_kernel.zero_()
            if self.bias is not None:
                self.bias.zero_()

    def near_zero_(self, eps: float) -> None:
        eps = float(eps)
        if eps <= 0.0:
            self.zero_()
            return
        with torch.no_grad():
            self.diag_kernel.fill_(self._diag_raw_for(eps))
            if self.quad_kernel is not None:
                _lecun_normal_(self.quad_kernel, self.input_dim)
                self.quad_kernel.mul_(eps)
            if self.lin_kernel is not None:
                _lecun_normal_(self.lin_kernel, self.input_dim)
                self.lin_kernel.mul_(eps)
            if self.bias is not None:
                self.bias.zero_()

    def set_identity_(self, coefficient: float = 1.0) -> None:
        """Set a single potential to ``0.5 * coefficient * ||x||^2``."""
        if self.num_potentials != 1:
            raise ValueError("set_identity_ is only defined for one potential.")
        coefficient = float(coefficient)
        if coefficient < 0.0:
            raise ValueError(f"identity coefficient must be non-negative; got {coefficient}.")
        with torch.no_grad():
            self.diag_kernel.fill_(self._diag_raw_for(coefficient))
            if self.quad_kernel is not None:
                self.quad_kernel.zero_()
            if self.lin_kernel is not None:
                self.lin_kernel.zero_()
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.size(0), -1)
        # matmul instead of broadcast-and-sum: the latter materializes a
        # (B, input_dim, num_potentials) intermediate (multi-GB per hidden
        # layer in all_layers mode, retained for double backward).
        y = 0.5 * x_flat.pow(2).matmul(self.diag)
        if self.quad_kernel is not None:
            quad = torch.einsum("bd,odr->bor", x_flat, self.quad_kernel)
            y = y + 0.5 * quad.pow(2).sum(dim=-1)
        if self.lin_kernel is not None:
            y = y + x_flat.matmul(self.lin_kernel)
        if self.bias is not None:
            y = y + self.bias
        return y


class NPFInputConvexPotential(nn.Module):
    """NPF paper ICNN potential with an optional LastQuad ablation."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        outer_rank: int = 1,
        inner_rank: int = 1,
        output_rank: Optional[int] = None,
        quadratic_mode: str = "all_layers",
        trainable_outer_quadratic: bool = True,
        activation: str = "relu",
        elu_alpha: float = 1.0,
        softplus_beta: float = 20.0,
        init_eps: float = 0.0,
        strong_convexity: float = 1.0,
        pos_weights: bool = True,
        positive_weight_rectifier: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        if not self.hidden_sizes:
            raise ValueError("NPFInputConvexPotential needs at least one hidden layer.")
        self.outer_rank = int(outer_rank)
        self.inner_rank = int(inner_rank)
        self.quadratic_mode = str(quadratic_mode).lower()
        self.activation = str(activation).lower()
        self.elu_alpha = float(elu_alpha)
        self.softplus_beta = float(softplus_beta)
        # init_eps is no longer part of the OTT initializer. It is kept only as
        # metadata for old configs and for exact-identity ablations.
        self.init_eps = float(init_eps)
        self.strong_convexity = float(strong_convexity)
        self.trainable_outer_quadratic = bool(trainable_outer_quadratic)
        self.pos_weights = bool(pos_weights)
        self.positive_weight_rectifier = str(positive_weight_rectifier).lower()

        valid_modes = {"all_layers", "last_layer_diagonal"}
        if self.quadratic_mode not in valid_modes:
            raise ValueError(
                f"Unsupported NPF quadratic_mode '{quadratic_mode}'. "
                f"Use one of {sorted(valid_modes)}."
            )
        if self.activation not in {"elu", "softplus", "relu"}:
            raise ValueError(f"Unsupported NPF activation '{activation}'.")
        if self.activation == "softplus" and self.softplus_beta <= 0.0:
            raise ValueError(
                f"softplus_beta must be positive for convexity; got {softplus_beta}."
            )
        if self.activation == "elu" and not (0.0 <= self.elu_alpha <= 1.0):
            raise ValueError(
                "elu_alpha must lie in [0, 1] for ELU to remain convex and "
                f"non-decreasing; got {elu_alpha}."
            )
        if self.strong_convexity < 0.0:
            raise ValueError(
                f"strong_convexity must be non-negative; got {strong_convexity}."
            )
        if self.outer_rank < 0 or self.inner_rank < 0:
            raise ValueError("outer_rank and inner_rank must be non-negative.")
        if output_rank is None:
            output_rank = (
                self.inner_rank if self.quadratic_mode == "all_layers" else 0
            )
        self.output_rank = int(output_rank)
        if self.output_rank < 0:
            raise ValueError(f"output_rank must be non-negative; got {output_rank}.")

        layer_dims = self.hidden_sizes + [1]
        self.w_xs = nn.ModuleList()
        for idx, width in enumerate(layer_dims):
            is_output_layer = idx == len(layer_dims) - 1
            if self.quadratic_mode == "all_layers" or is_output_layer:
                rank = self.output_rank if is_output_layer else self.inner_rank
                self.w_xs.append(
                    NPFPosDefPotentials(
                        input_dim=self.input_dim,
                        num_potentials=width,
                        rank=rank,
                        use_linear=True,
                        use_bias=True,
                        # 'exp' keeps the diagonal strictly positive with a
                        # never-zero gradient (d diag/d raw = diag). The
                        # previous 'relu' let the lambda-cost gradient push raw
                        # entries negative, where value AND gradient are
                        # exactly zero forever — a one-way dead zone for the
                        # elementwise sample-dependent transport path.
                        # Output layer: the old 0.1269 effective init put a
                        # deterministic (1+0.1269)x global dilation into T at
                        # t=0 (measured 1.54 pixel-L2, 3x the eval radius, 94%
                        # of the init transport delta) — a sample-INDEPENDENT
                        # far-field artifact theta neutralizes in one step.
                        # 0.01 keeps the readout preactivation comfortably
                        # positive (~15) while starting T near identity.
                        # Hidden all-layers blocks keep 0.1269 = softplus(-2),
                        # the established NPF operating point (diag_init is
                        # effective-valued for every rectifier).
                        diag_init=0.01 if is_output_layer else 0.1269280110429725,
                        diag_rectifier="exp",
                        low_rank_init="lecun_normal",
                        linear_init="lecun_normal",
                    )
                )
            else:
                layer = nn.Linear(self.input_dim, width, bias=True)
                _lecun_normal_(layer.weight, self.input_dim)
                nn.init.zeros_(layer.bias)
                self.w_xs.append(layer)

        self.w_zs = nn.ModuleList()
        prev_dim = self.hidden_sizes[0]
        for width in layer_dims[1:]:
            self.w_zs.append(
                NPFDense(
                    prev_dim,
                    width,
                    pos_weights=self.pos_weights,
                    rectifier=self.positive_weight_rectifier,
                )
            )
            prev_dim = width

        self.pos_def_potential = NPFPosDefPotentials(
            input_dim=self.input_dim,
            num_potentials=1,
            rank=self.outer_rank,
            use_linear=True,
            use_bias=True,
            diag_init=self.strong_convexity,
            diag_rectifier="relu",
            low_rank_init="zeros",
            linear_init="zeros",
        )
        self._set_outer_quadratic_trainable(self.trainable_outer_quadratic)
        if self.trainable_outer_quadratic and self.outer_rank > 0:
            # grad of 0.5*||A^T x||^2 w.r.t. A vanishes identically at A=0, so
            # the zeros init above is an exact saddle: a trainable outer
            # quadratic would receive zero gradient forever. Break it with a
            # small random draw (identity-init diagnostics re-zero it via
            # set_identity_, which is intentional there).
            with torch.no_grad():
                _lecun_normal_(self.pos_def_potential.quad_kernel, self.input_dim)
                self.pos_def_potential.quad_kernel.mul_(1e-2)
        self._apply_convex_init_bias_shifts()

    @property
    def use_hidden_quadratics(self) -> bool:
        return self.quadratic_mode == "all_layers"

    @property
    def residual_output_potential(self) -> NPFPosDefPotentials:
        return self.w_xs[-1]  # type: ignore[return-value]

    def _set_outer_quadratic_trainable(self, trainable: bool) -> None:
        """Honor the trainable_outer_quadratic constructor flag."""
        for param in self.pos_def_potential.parameters():
            param.requires_grad_(bool(trainable))

    def enforce_trainable_constraints(self) -> None:
        """Restore trainability constraints after broad requires_grad toggles."""
        self._set_outer_quadratic_trainable(self.trainable_outer_quadratic)

    def _apply_convex_init_bias_shifts(self) -> None:
        """Apply the legacy-sign bias shift paired with positive hidden weights.

        The Hoedt-Klambauer convex-init derivation yields a NEGATIVE
        bias_mean, but the legacy golden run filled the corresponding biases
        with +|bias_mean| (exact sign mirror, measured +0.7394546 at fan-in
        1024). The sign decides the operating regime of the deep stack:
        negative shift leaves 49-68% of deep units active under beta-20
        softplus, positive runs them 88-100% active — the "hot" regime whose
        strong hidden-chain gradients carried the legacy adversary's
        sample-dependent attack capacity. Convexity is unaffected by bias
        sign; we follow the legacy convention.
        """
        with torch.no_grad():
            for w_z, w_x in zip(self.w_zs, self.w_xs[1:]):
                if not w_z.pos_weights:
                    continue
                bias_mean = abs(_convex_bias_mean(w_z.in_features))
                if isinstance(w_x, NPFPosDefPotentials):
                    if w_x.bias is not None:
                        w_x.bias.fill_(bias_mean)
                elif isinstance(w_x, nn.Linear):
                    if w_x.bias is not None:
                        w_x.bias.fill_(bias_mean)

    def init_as_identity(self, residual_eps: Optional[float] = None) -> None:
        """Initialize near identity and keep the OTT final identity potential."""
        eps = self.init_eps if residual_eps is None else float(residual_eps)
        with torch.no_grad():
            for module in self.w_xs:
                if isinstance(module, NPFPosDefPotentials):
                    module.near_zero_(eps)
                elif isinstance(module, nn.Linear):
                    if eps > 0.0:
                        _lecun_normal_(module.weight, self.input_dim)
                        module.weight.mul_(eps)
                    else:
                        module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
            for module in self.w_zs:
                module.near_zero_(eps)
            self.pos_def_potential.set_identity_(self.strong_convexity)
            if (
                self.trainable_outer_quadratic
                and self.outer_rank > 0
                and eps > 0.0
            ):
                # set_identity_ zeroed quad_kernel, which is an exact
                # zero-gradient saddle for a trainable outer quadratic; break
                # it at the identity-adjacent scale. Exact-identity
                # diagnostics (eps=0) keep the zeros on purpose.
                _lecun_normal_(self.pos_def_potential.quad_kernel, self.input_dim)
                self.pos_def_potential.quad_kernel.mul_(eps)

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(u, alpha=self.elu_alpha)
        if self.activation == "softplus":
            beta = self.softplus_beta
            return F.softplus(beta * u) / beta
        if self.activation == "relu":
            return F.relu(u)
        raise ValueError(f"Unsupported NPF activation '{self.activation}'.")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.reshape(z.size(0), -1)
        h = self._act(self.w_xs[0](z_flat))
        for w_z, w_x in zip(self.w_zs, self.w_xs[1:]):
            h = self._act(w_z(h) + w_x(z_flat))
        residual = h.squeeze(-1)
        base = self.pos_def_potential(z_flat).squeeze(-1)
        return residual + base


def npf_T_omega(
    z: torch.Tensor,
    icnn_model: NPFInputConvexPotential,
    create_graph: bool,
) -> torch.Tensor:
    """Return the input transport map T_omega(z) = grad_z psi_omega(z)."""
    z_in = z.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi_val = icnn_model(z_in)
        grad = torch.autograd.grad(psi_val.sum(), z_in, create_graph=create_graph)[0]
    grad = torch.where(torch.isfinite(grad), grad, z)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if not create_graph:
        grad = grad.detach()
    return grad.view_as(z)


# Backward-compatible names for older imports. Their implementation now follows
# the OTT parameterization rather than the previous squared-delta form.
NPFNonNegativeDense = NPFDense
NPFQuadraticForm = NPFPosDefPotentials
