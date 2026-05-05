"""Barzilai-Borwein step size with Armijo line search.

Two flavours are provided:

* ``bb_armijo_step_params`` operates on a **list of nn.Parameters**, in
  place. Used by both the ICNN-DRO and NPF algorithms in this pipeline
  (the classifier/policy and the convex potential are nn.Modules and we update
  the potential directly via parameters_to_vector / vector_to_parameters).
* ``bb_armijo_step_vector`` operates on a **flat parameter vector**
  (paired with metadata so it can be unflattened back into a parameter
  dict for ``functional_call``). Currently unused in this pipeline; kept
  for API parity with MNIST_Cuturi which uses this pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.utils as nn_utils


@dataclass
class BBArmijoState:
    """Barzilai-Borwein step size + Armijo line search state."""

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

        For a locally concave maximization objective ``f`` with ascent
        direction ``g = +grad f``:
            s = alpha_prev * g_prev,
            y = grad_new - grad_prev ~= alpha_prev * H @ g_prev,    H < 0
        so ``<s, y> < 0``. The standard descent BB ``alpha = <s, s>/<s, y>``
        is therefore negative, and clamping to ``[alpha_min, alpha_max]``
        collapses to ``alpha_min`` - disabling BB. We use the ascent BB
        ``alpha = -<s, s> / <s, y>`` which is positive whenever ``y`` is
        meaningful and reflects the local concavity (more curvature ->
        smaller step).
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
            # Concave-ascent regime: denom < 0. Accept any sufficiently
            # negative value; fall back to alpha_prev otherwise (denom near
            # zero means y ~= 0 / numerically unreliable curvature).
            cond = torch.isfinite(denom) & (denom < -1e-12)
            alpha_bb = torch.where(
                cond,
                -num / denom,
                torch.tensor(self.alpha_prev, device=denom.device, dtype=denom.dtype),
            )
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())

        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

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
) -> Tuple[List[torch.nn.Parameter], BBArmijoState, float, float]:
    """Single BB+Armijo gradient-ascent step on a parameter collection.

    The returned ``grad_norm`` is the gradient norm at the START of this
    call. We don't run a second fwd+bwd at the post-step iterate just for
    logging - the next call recomputes that gradient anyway, so the extra
    pass is wasted work (~25-50% of inner-loop wallclock).
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
    # Use reshape(-1) + cat: autograd.grad can return non-contiguous tensors
    # (e.g. via .t()/einsum in NPF), and parameters_to_vector internally
    # calls view(-1) which fails on non-contiguous storage.
    grad_tensors = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)]
    grad_vec = torch.cat([g.reshape(-1) for g in grad_tensors])
    grad_norm = grad_vec.norm().item()
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
            f_trial = float(f_params(False).detach())
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


def bb_armijo_step_vector(
    vec: torch.Tensor,
    f_params,
    bb_state: BBArmijoState,
) -> Tuple[torch.Tensor, BBArmijoState, float]:
    """Single BB+Armijo gradient-ascent step on a flat parameter vector.

    ``f_params(vec, create_graph: bool) -> scalar tensor`` is the objective
    evaluated with the flattened parameters.
    """
    vec_det = vec.detach().requires_grad_(True)
    f_val = f_params(vec_det, True)
    grad_vec = torch.autograd.grad(f_val, vec_det, create_graph=False)[0]
    f_val_f = float(f_val.detach())
    grad_flat = grad_vec.reshape(-1)
    grad_norm = float(grad_flat.norm().item())
    if (
        not math.isfinite(f_val_f)
        or not torch.isfinite(grad_flat).all()
        or not math.isfinite(grad_norm)
    ):
        return vec_det.detach(), bb_state, f_val_f
    alpha = bb_state.propose(vec_det.reshape(-1), grad_vec.reshape(-1))
    g_dot_g = float(torch.dot(grad_flat, grad_flat).item())
    if g_dot_g <= 0.0 or not math.isfinite(g_dot_g):
        return vec_det.detach(), bb_state, f_val_f
    grad_vec_det = grad_vec.detach()
    vec_det_val = vec_det.detach()
    alpha_k = alpha
    armijo_succeeded = False
    for i in range(bb_state.ls_max_steps):
        v_trial = vec_det_val + alpha_k * grad_vec_det
        with torch.no_grad():
            f_trial = f_params(v_trial, False).item()
        if math.isfinite(float(f_trial)) and f_trial >= f_val_f + bb_state.ls_c * alpha_k * g_dot_g:
            armijo_succeeded = True
            break
        if i < bb_state.ls_max_steps - 1:
            alpha_k *= bb_state.ls_shrink

    if armijo_succeeded or not bb_state.reject_on_armijo_failure:
        v_new = (vec_det_val + alpha_k * grad_vec_det).detach()
        new_bb_state = bb_state.update_history(
            vec_det_val.reshape(-1), grad_vec_det.reshape(-1), alpha_k
        )
    else:
        v_new = vec_det_val.detach()
        new_bb_state = bb_state
    return v_new, new_bb_state, f_val_f
