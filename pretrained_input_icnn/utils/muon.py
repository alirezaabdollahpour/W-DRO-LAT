"""Muon inner optimizer for NPF adversary parameters.

Muon is a matrix-parameter optimizer: it applies momentum to the gradient,
orthogonalizes the resulting matrix update with Newton-Schulz iterations, and
then steps the parameter. NPF also contains vector parameters (biases, raw
diagonals, affine terms), so this utility updates matrix-like tensors with
Muon and sends lower-dimensional tensors through a small AdamW/SGD fallback.

The implementation mirrors ``bb_armijo_step_params``'s calling convention so
NPF can switch inner optimizers without changing the objective or DDP reducer
plumbing. The step is gradient ascent because the NPF adversary solves an inner
maximization problem.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import torch


@dataclass
class MuonState:
    lr: float
    momentum: float
    nesterov: bool
    ns_steps: int
    matrix_lr_scale: str
    weight_decay: float
    fallback: str
    fallback_lr: float
    fallback_weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    step_count: int = 0
    momentum_buffers: List[Optional[torch.Tensor]] = field(default_factory=list)
    adam_exp_avg: List[Optional[torch.Tensor]] = field(default_factory=list)
    adam_exp_avg_sq: List[Optional[torch.Tensor]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        lr: float = 2e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        matrix_lr_scale: str = "auto",
        weight_decay: float = 0.0,
        fallback: str = "adamw",
        fallback_lr: float = 0.0,
        fallback_weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        max_grad_norm: float = 0.0,
    ) -> "MuonState":
        lr = float(lr)
        if lr <= 0.0 or not math.isfinite(lr):
            raise ValueError(f"Muon lr must be finite and positive; got {lr}.")
        momentum = float(momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon momentum must be in [0, 1); got {momentum}.")
        matrix_lr_scale = str(matrix_lr_scale).lower()
        if matrix_lr_scale not in {"auto", "none"}:
            raise ValueError(
                f"Unsupported Muon matrix_lr_scale '{matrix_lr_scale}'. "
                "Use 'auto' or 'none'."
            )
        fallback = str(fallback).lower()
        if fallback not in {"adamw", "sgd", "none"}:
            raise ValueError(
                f"Unsupported Muon fallback '{fallback}'. "
                "Use 'adamw', 'sgd', or 'none'."
            )
        adam_beta1 = float(adam_beta1)
        adam_beta2 = float(adam_beta2)
        if not 0.0 <= adam_beta1 < 1.0 or not 0.0 <= adam_beta2 < 1.0:
            raise ValueError(
                "Muon fallback Adam betas must both be in [0, 1); "
                f"got ({adam_beta1}, {adam_beta2})."
            )
        fallback_lr = float(fallback_lr)
        if fallback_lr <= 0.0:
            fallback_lr = lr
        return cls(
            lr=lr,
            momentum=momentum,
            nesterov=bool(nesterov),
            ns_steps=int(max(1, ns_steps)),
            matrix_lr_scale=matrix_lr_scale,
            weight_decay=float(max(0.0, weight_decay)),
            fallback=fallback,
            fallback_lr=fallback_lr,
            fallback_weight_decay=float(max(0.0, fallback_weight_decay)),
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=float(max(adam_eps, 1e-16)),
            max_grad_norm=float(max(0.0, max_grad_norm)),
        )

    def ensure_slots(self, params: List[torch.nn.Parameter]) -> None:
        n = len(params)
        while len(self.momentum_buffers) < n:
            self.momentum_buffers.append(None)
        while len(self.adam_exp_avg) < n:
            self.adam_exp_avg.append(None)
        while len(self.adam_exp_avg_sq) < n:
            self.adam_exp_avg_sq.append(None)


def _profile_add(profile: Any, key: str, value: float = 1.0) -> None:
    if profile is None:
        return
    add = getattr(profile, "profile_add", None)
    if add is not None:
        add(key, value)


@contextmanager
def _profile_time(profile: Any, key: str) -> Iterator[None]:
    if profile is None:
        with nullcontext():
            yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _profile_add(profile, key, time.perf_counter() - start)


def _matrix_view(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 2:
        raise ValueError("_matrix_view requires a tensor with ndim >= 2.")
    return t.reshape(t.shape[0], -1)


def _newton_schulz_zeroth_power(
    update: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the polar factor of a 2-D update matrix.

    The quintic coefficients are the standard Muon Newton-Schulz constants.
    They intentionally over-emphasize the top singular values a little; Muon
    uses this as a fast, stable orthogonalized update rather than as a
    high-precision matrix decomposition.
    """
    if update.ndim != 2:
        raise ValueError("Newton-Schulz Muon update expects a 2-D matrix.")
    if update.numel() == 0:
        return update
    if not torch.isfinite(update).all():
        return torch.zeros_like(update)

    original_dtype = update.dtype
    original_device = update.device
    transposed = update.shape[0] > update.shape[1]
    x = update.t() if transposed else update

    work_dtype = (
        torch.bfloat16
        if original_device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    x = x.to(dtype=work_dtype)
    x = x / x.norm().clamp_min(eps)

    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(int(steps)):
        xx_t = x @ x.t()
        poly = b * xx_t + c * (xx_t @ xx_t)
        x = a * x + poly @ x

    x = x.t() if transposed else x
    return x.to(dtype=original_dtype)


def _muon_lr_scale(param: torch.Tensor, mode: str) -> float:
    if mode == "none" or param.ndim < 2:
        return 1.0
    mat = _matrix_view(param)
    rows = max(1, int(mat.shape[0]))
    cols = max(1, int(mat.shape[1]))
    return math.sqrt(max(1.0, float(rows) / float(cols)))


def _clip_grad_tensors(
    grads: List[torch.Tensor],
    max_norm: float,
) -> Tuple[List[torch.Tensor], float]:
    if len(grads) == 0:
        return grads, 0.0
    norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        norm_sq = norm_sq + g.detach().float().pow(2).sum()
    total_norm = float(torch.sqrt(norm_sq).item())
    if max_norm > 0.0 and math.isfinite(total_norm) and total_norm > max_norm:
        scale = float(max_norm) / (total_norm + 1e-12)
        grads = [g * scale for g in grads]
    return grads, total_norm


def _adamw_ascent_(
    param: torch.nn.Parameter,
    grad: torch.Tensor,
    state: MuonState,
    idx: int,
) -> None:
    exp_avg = state.adam_exp_avg[idx]
    exp_avg_sq = state.adam_exp_avg_sq[idx]
    if exp_avg is None or exp_avg.shape != param.shape:
        exp_avg = torch.zeros_like(param)
        state.adam_exp_avg[idx] = exp_avg
    if exp_avg_sq is None or exp_avg_sq.shape != param.shape:
        exp_avg_sq = torch.zeros_like(param)
        state.adam_exp_avg_sq[idx] = exp_avg_sq

    beta1 = state.adam_beta1
    beta2 = state.adam_beta2
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1 ** max(1, state.step_count)
    bias_correction2 = 1.0 - beta2 ** max(1, state.step_count)
    step_size = state.fallback_lr * math.sqrt(bias_correction2) / bias_correction1

    if state.fallback_weight_decay > 0.0:
        param.mul_(1.0 - state.fallback_lr * state.fallback_weight_decay)
    denom = exp_avg_sq.sqrt().add_(state.adam_eps)
    param.addcdiv_(exp_avg, denom, value=step_size)


def _sgd_ascent_(
    param: torch.nn.Parameter,
    grad: torch.Tensor,
    state: MuonState,
    idx: int,
) -> None:
    update = grad
    if state.momentum > 0.0:
        buf = state.momentum_buffers[idx]
        if buf is None or buf.shape != param.shape:
            buf = torch.zeros_like(param)
            state.momentum_buffers[idx] = buf
        buf.mul_(state.momentum).add_(grad)
        update = grad.add(buf, alpha=state.momentum) if state.nesterov else buf
    if state.fallback_weight_decay > 0.0:
        param.mul_(1.0 - state.fallback_lr * state.fallback_weight_decay)
    param.add_(update, alpha=state.fallback_lr)


def muon_step_params(
    params,
    f_params,
    muon_state: MuonState,
    reduce_grad_fn=None,
    reduce_scalar_fn=None,
    profile=None,
) -> Tuple[List[torch.nn.Parameter], MuonState, float, float]:
    """Single Muon gradient-ascent step on a parameter collection.

    ``f_params(create_graph: bool)`` must return the scalar objective to
    maximize. DDP reducers follow the same contract as ``bb_armijo_step_params``:
    parameter gradients and objective scalars are averaged across ranks before
    any step is taken, keeping the shared NPF adversary synchronized.
    """
    _profile_add(profile, "muon_steps", 1.0)
    with _profile_time(profile, "muon_params_prepare_s"):
        params = [p for p in list(params) if p.requires_grad]
        if len(params) == 0:
            raise ValueError("muon_step_params received an empty parameter list.")
        muon_state.ensure_slots(params)

    with _profile_time(profile, "muon_f_grad_objective_s"):
        f_val = f_params(True)
    with _profile_time(profile, "muon_grad_autograd_s"):
        grads = torch.autograd.grad(
            f_val,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

    with _profile_time(profile, "muon_grad_pack_reduce_s"):
        grad_tensors = [
            g.detach() if g is not None else torch.zeros_like(p)
            for p, g in zip(params, grads)
        ]
        if reduce_grad_fn is not None:
            grad_tensors = reduce_grad_fn(grad_tensors)
        grad_tensors, grad_norm = _clip_grad_tensors(
            grad_tensors, muon_state.max_grad_norm
        )
        if reduce_scalar_fn is not None:
            f_val_float = float(reduce_scalar_fn(f_val.detach()))
        else:
            f_val_float = float(f_val.detach())

    if (
        not math.isfinite(f_val_float)
        or not math.isfinite(grad_norm)
        or any(not torch.isfinite(g).all() for g in grad_tensors)
    ):
        _profile_add(profile, "muon_skipped_nonfinite", 1.0)
        return params, muon_state, f_val_float, grad_norm

    muon_state.step_count += 1
    with torch.no_grad():
        with _profile_time(profile, "muon_update_s"):
            for idx, (param, grad) in enumerate(zip(params, grad_tensors)):
                # Degenerate (n,1)/(1,n) "matrices" (diag/linear kernels of the
                # quadratic potentials) must not take the Newton-Schulz path:
                # orthogonalizing a rank-1 column returns the normalized
                # gradient, which combined with the sqrt(max(n,m)) lr scale
                # amplifies these sample-independent modes ~55x. Route them to
                # the elementwise fallback instead.
                is_true_matrix = (
                    grad.ndim >= 2
                    and param.ndim >= 2
                    and min(param.shape[0], max(1, param.numel() // param.shape[0])) > 1
                )
                if is_true_matrix:
                    _profile_add(profile, "muon_matrix_params", 1.0)
                    buf = muon_state.momentum_buffers[idx]
                    if buf is None or buf.shape != param.shape:
                        buf = torch.zeros_like(param)
                        muon_state.momentum_buffers[idx] = buf
                    buf.mul_(muon_state.momentum).add_(grad)
                    update = (
                        grad.add(buf, alpha=muon_state.momentum)
                        if muon_state.nesterov
                        else buf
                    )
                    update_shape = update.shape
                    with _profile_time(profile, "muon_orthogonalize_s"):
                        update = _newton_schulz_zeroth_power(
                            _matrix_view(update),
                            steps=muon_state.ns_steps,
                        ).reshape(update_shape)
                    lr = muon_state.lr * _muon_lr_scale(
                        param, muon_state.matrix_lr_scale
                    )
                    if muon_state.weight_decay > 0.0:
                        param.mul_(1.0 - muon_state.lr * muon_state.weight_decay)
                    param.add_(update, alpha=lr)
                    continue

                if muon_state.fallback == "none":
                    _profile_add(profile, "muon_fallback_skipped_params", 1.0)
                    continue

                _profile_add(profile, "muon_fallback_params", 1.0)
                with _profile_time(profile, "muon_fallback_update_s"):
                    if muon_state.fallback == "adamw":
                        _adamw_ascent_(param, grad, muon_state, idx)
                    elif muon_state.fallback == "sgd":
                        _sgd_ascent_(param, grad, muon_state, idx)
                    else:
                        raise ValueError(
                            f"Unsupported Muon fallback '{muon_state.fallback}'."
                        )

    return params, muon_state, f_val_float, grad_norm
