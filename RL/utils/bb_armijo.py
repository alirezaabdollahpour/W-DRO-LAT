"""Barzilai-Borwein step-size proposal + Armijo line search."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # 0 disables; otherwise rescale to ||g|| <= grad_clip
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float = 1e-1,
        alpha_min: float = 1e-6,
        alpha_max: float = 10.0,
        ls_c: float = 1e-4,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
            weight_decay=float(max(weight_decay, 0.0)),
            grad_clip=float(max(grad_clip, 0.0)),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
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
            cond = torch.isfinite(denom) & (denom > 1e-12)
            alpha_bb = torch.where(cond, num / denom, torch.tensor(self.alpha_prev, device=denom.device))
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())

        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=alpha_clamped,
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_step_params(
    params: Any,
    f_params,
    bb_state: BBArmijoState,
) -> Tuple[Any, BBArmijoState, float, float]:
    """Single BB+Armijo gradient-ascent step on a parameter collection.

    f_params: callable(create_graph: bool) -> scalar torch tensor (maximized).
    """
    params = list(params)
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

    # Weight decay: ascending on (J - 0.5*wd*||theta||^2) means subtracting
    # wd*theta from the ascent direction. This is the standard L2-prox trick;
    # for the NPF/ICNN convex potential it prevents weight drift to the
    # corner-saturation regime when the cost penalty (lam) is small.
    if bb_state.weight_decay > 0.0:
        grad_vec = grad_vec - bb_state.weight_decay * params_vec

    # Gradient norm clipping. Caps single-step magnitude when the policy
    # term -J spikes (e.g. on near-edge anchors); has no effect inside the
    # well-behaved regime where ||g|| <= grad_clip.
    if bb_state.grad_clip > 0.0:
        gnorm = grad_vec.norm()
        if torch.isfinite(gnorm) and float(gnorm.item()) > bb_state.grad_clip:
            grad_vec = grad_vec * (bb_state.grad_clip / float(gnorm.item()))

    grad_norm = grad_vec.norm().item()

    alpha = bb_state.propose(params_vec, grad_vec)
    f_val_float = float(f_val.detach())
    g_dot_g = float(torch.dot(grad_vec, grad_vec).item())
    if g_dot_g == 0.0:
        return params, bb_state, f_val_float, grad_norm

    alpha_k = alpha
    for i in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial = float(f_params(False).detach())
        if f_trial >= f_val_float + bb_state.ls_c * alpha_k * g_dot_g:
            break
        if i < bb_state.ls_max_steps - 1:
            alpha_k *= bb_state.ls_shrink

    final_vec = params_vec + alpha_k * grad_vec
    with torch.no_grad():
        nn_utils.vector_to_parameters(final_vec, params)

    f_new = f_params(True)
    grads_new = torch.autograd.grad(
        f_new,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors_new = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads_new)]
    grad_vec_new = torch.cat([g.reshape(-1) for g in grad_tensors_new])
    new_state = bb_state.update_history(params_vec, grad_vec, alpha_k)
    return params, new_state, f_val_float, grad_vec_new.norm().item()
