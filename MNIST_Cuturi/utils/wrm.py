"""WRM inner-maximisation ascent (Sinha et al., 2018)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrm_ascent_x(
    x0: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    step_offset: int = 0,
) -> torch.Tensor:
    """WRM inner maximization: z <- z + (lr/sqrt(s)) (∇CE - 2λ(z-x))."""
    if num_steps == 0:
        return x0.detach()
    x_orig = x0.detach()
    z = x_orig.clone()
    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        eta_s = lr / math.sqrt(s)
        z = z.detach() + eta_s * (grads - 2.0 * lambda_reg * (z.detach() - x_orig))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()


def wrm_ascent_x_anchored(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    step_offset: int = 0,
) -> torch.Tensor:
    """WRM ascent starting from z0 with penalty anchored at x_anchor."""
    if num_steps == 0:
        return z0.detach()
    x_anc = x_anchor.detach()
    z = z0.detach().clone()
    for s in range(1 + step_offset, num_steps + 1 + step_offset):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        eta_s = lr / math.sqrt(s)
        z = z.detach() + eta_s * (grads - 2.0 * lambda_reg * (z.detach() - x_anc))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()


def wrm_ascent_x_anchored_const_lr(
    z0: torch.Tensor,
    x_anchor: torch.Tensor,
    model: nn.Module,
    y: torch.Tensor,
    lambda_reg: float,
    num_steps: int,
    lr: float = 0.01,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    """WRM ascent with constant step size (used for PPA refinement rounds)."""
    if num_steps == 0:
        return z0.detach()
    x_anc = x_anchor.detach()
    z = z0.detach().clone()
    for _ in range(num_steps):
        z.requires_grad_(True)
        per_sample_ce = F.cross_entropy(model(z), y, reduction="none")
        grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
        z = z.detach() + lr * (grads - 2.0 * lambda_reg * (z.detach() - x_anc))
        if clamp is not None:
            lo, hi = clamp
            z = z.clamp(lo, hi)
    return z.detach()
