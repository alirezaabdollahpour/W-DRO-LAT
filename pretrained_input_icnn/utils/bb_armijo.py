"""Barzilai-Borwein step size with Armijo line search.

Vendored from ``Logistic_Regression_CIFAR10/utils/bb_armijo.py`` (the
``bb_armijo_step_params`` flavour) so the NPF inner loop in this package
runs identical inner-loop semantics to the LR-CIFAR10 reference. We keep
only the parameter-list variant since the NPF algorithm here uses live
``nn.Parameter`` lists; the flat-vector variant is unused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.utils as nn_utils


@dataclass
class BBArmijoState:
    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    reject_on_armijo_failure: bool = False
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float = 0.0005,
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0,
        ls_c: float = 0.1,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
        reject_on_armijo_failure: bool = False,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
            reject_on_armijo_failure=bool(reject_on_armijo_failure),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        """Barzilai-Borwein step proposal for *ascent*.

        For locally concave maximization with ``g = +grad f``, ``<s, y>`` is
        negative; the standard descent BB ``<s,s>/<s,y>`` would be negative
        and clamp to ``alpha_min``, disabling BB. We use the ascent BB
        ``alpha = -<s,s>/<s,y>``.
        """
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.shape != params_vec.shape
            or self.prev_grad_vec.shape != grad_vec.shape
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            num = torch.dot(s, s)
            cond = torch.isfinite(denom) & (denom < -1e-12)
            alpha_bb = torch.where(
                cond,
                -num / denom,
                torch.tensor(self.alpha_prev, device=denom.device, dtype=denom.dtype),
            )
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())
        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        return max(self.alpha_min, min(self.alpha_max, float(alpha)))

    def update_history(
        self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float
    ) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=alpha_clamped,
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            reject_on_armijo_failure=self.reject_on_armijo_failure,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_step_params(
    params,
    f_params,
    bb_state: BBArmijoState,
    reduce_grad_fn=None,
    reduce_scalar_fn=None,
) -> Tuple[List[torch.nn.Parameter], BBArmijoState, float, float]:
    """Single BB+Armijo gradient-ascent step on a parameter collection.

    ``reduce_grad_fn`` and ``reduce_scalar_fn`` are optional hooks for
    DDP. When provided they receive (a) the per-rank gradient tensor list
    and (b) a per-rank objective scalar respectively, and must return the
    cross-rank averaged versions. With both ``None`` (single-GPU) the
    function behaves exactly as before. Plumbing both reducers ensures
    every rank picks the same Armijo trial step and stays in lockstep.

    The returned ``grad_norm`` is the gradient norm at the START of this
    call. The next call recomputes the gradient at the new iterate, so an
    extra fwd+bwd just for logging would double the inner-loop wallclock.
    """
    params = list(params)
    if len(params) == 0:
        raise ValueError("bb_armijo_step_params received an empty parameter list.")
    params_vec = nn_utils.parameters_to_vector(params).detach()

    f_val = f_params(True)
    grads = torch.autograd.grad(
        f_val,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    # autograd.grad can return non-contiguous tensors (e.g. through .t() or
    # einsum in NPF); parameters_to_vector internally calls view(-1) which
    # fails on non-contiguous storage, so we reshape and concat instead.
    grad_tensors = [
        g.detach() if g is not None else torch.zeros_like(p)
        for p, g in zip(params, grads)
    ]
    if reduce_grad_fn is not None:
        grad_tensors = reduce_grad_fn(grad_tensors)
    grad_vec = torch.cat([g.reshape(-1) for g in grad_tensors])
    grad_norm = grad_vec.norm().item()
    if reduce_scalar_fn is not None:
        f_val_float = float(reduce_scalar_fn(f_val.detach()))
    else:
        f_val_float = float(f_val.detach())

    if (
        not math.isfinite(f_val_float)
        or not torch.isfinite(grad_vec).all()
        or not math.isfinite(grad_norm)
    ):
        return params, bb_state, f_val_float, grad_norm

    alpha = bb_state.propose(params_vec, grad_vec)
    directional_derivative = float(torch.dot(grad_vec, grad_vec).item())
    if directional_derivative <= 0.0 or not math.isfinite(directional_derivative):
        return params, bb_state, f_val_float, grad_norm

    alpha_k = alpha
    armijo_succeeded = False
    for i in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial_t = f_params(False).detach()
            if reduce_scalar_fn is not None:
                f_trial = float(reduce_scalar_fn(f_trial_t))
            else:
                f_trial = float(f_trial_t)
        if (
            math.isfinite(f_trial)
            and f_trial >= f_val_float + bb_state.ls_c * alpha_k * directional_derivative
        ):
            armijo_succeeded = True
            break
        if i < bb_state.ls_max_steps - 1:
            alpha_k *= bb_state.ls_shrink

    if armijo_succeeded or not bb_state.reject_on_armijo_failure:
        final_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(final_vec, params)
        new_state = bb_state.update_history(params_vec, grad_vec, alpha_k)
    else:
        with torch.no_grad():
            nn_utils.vector_to_parameters(params_vec, params)
        new_state = bb_state
    return params, new_state, f_val_float, grad_norm
