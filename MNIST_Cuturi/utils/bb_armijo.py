"""Barzilai–Borwein step size with Armijo line search."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


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
        alpha0: float = 5e-4,
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0,
        ls_c: float = 0.1,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
        reject_on_armijo_failure: bool = False,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=max(alpha_min, 1e-12),
            alpha_max=max(alpha_max, alpha_min),
            alpha_prev=alpha0,
            ls_c=ls_c,
            ls_shrink=ls_shrink,
            ls_max_steps=int(max(ls_max_steps, 1)),
            reject_on_armijo_failure=bool(reject_on_armijo_failure),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        """Barzilai-Borwein step proposal for *ascent*.

        For locally concave maximization with ``g = +grad f``, ``<s, y>``
        is negative; the standard descent BB ``<s,s>/<s,y>`` would be
        negative and clamp to ``alpha_min``. We use the ascent form
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


def bb_armijo_step_params(vec: torch.Tensor, meta, f_params, bb_state: BBArmijoState):
    """Single BB+Armijo gradient-ascent step on ICNN parameter vector.

    f_params(vec, create_graph) -> scalar tensor.
    Uses torch.no_grad() during Armijo line search trials for efficiency.

    History is stored as (entry params, entry grad) so that on the next
    call ``s = params_curr - params_prev = alpha_prev * grad_prev`` is
    nonzero. Saving (post-step params, post-step grad) instead would
    give ``s = 0`` (the next call's entry equals the previous call's
    exit), which silently disables BB.
    """
    vec_det = vec.detach().requires_grad_(True)
    f_val = f_params(vec_det, True)
    grad_vec = torch.autograd.grad(f_val, vec_det, create_graph=False)[0]
    grad_vec_det = grad_vec.detach()
    vec_det_val = vec_det.detach()

    alpha = bb_state.propose(vec_det_val.reshape(-1), grad_vec_det.reshape(-1))
    f_val_f = float(f_val.item())
    g_dot_g = float(torch.dot(grad_vec_det.reshape(-1), grad_vec_det.reshape(-1)).item())
    if g_dot_g == 0.0:
        return vec_det_val, bb_state, f_val_f

    alpha_k = alpha
    armijo_succeeded = False
    for i in range(bb_state.ls_max_steps):
        v_trial = vec_det_val + alpha_k * grad_vec_det
        with torch.no_grad():
            f_trial = f_params(v_trial, False).item()
        if f_trial >= f_val_f + bb_state.ls_c * alpha_k * g_dot_g:
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
        v_new = vec_det_val
        new_bb_state = bb_state
    return v_new, new_bb_state, f_val_f
