"""PGD-L2 attack utilities (single-start and multi-restart)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_l2(x: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x onto the L2 ball of radius eps centred at x_orig."""
    diff = x - x_orig
    flat = diff.view(diff.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True)
    factor = (eps / (norm + 1e-12)).clamp(max=1.0)
    factor = factor.view(-1, *([1] * (x.dim() - 1)))
    return x_orig + diff * factor


def per_sample_l2_normalize(t: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    B = t.size(0)
    flat = t.view(B, -1)
    norms = flat.norm(p=2, dim=1).clamp(min=floor)
    return t / norms.view(B, *([1] * (t.dim() - 1)))


def sphere_start_l2(x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Random start on the L2 sphere of radius eps intersected with [0,1]^d.

    Project-then-normalise so ||start - x0||_2 = eps exactly even after box clipping.
    """
    z = torch.randn_like(x0)
    z = per_sample_l2_normalize(z)
    p = x0 + eps * z
    p_box = p.clamp(0.0, 1.0)
    d = p_box - x0
    B = x0.size(0)
    d_flat = d.view(B, -1)
    norms = d_flat.norm(p=2, dim=1)
    safe_norms = norms.clamp(min=1e-12)
    scale = (eps / safe_norms).view(B, *([1] * (x0.dim() - 1)))
    d_scaled = d * scale
    zero_mask = (norms < 1e-12).view(B, *([1] * (x0.dim() - 1)))
    d_final = torch.where(zero_mask, torch.zeros_like(d_scaled), d_scaled)
    return (x0 + d_final).clamp(0.0, 1.0)


def project_onto_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factors = (eps / norms).clamp(max=1.0)
    return (flat * factors).view_as(delta)


def pgd_l2_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_steps: int,
    step_size: Optional[float] = None,
) -> torch.Tensor:
    """Single-start PGD-L2 from x (no random restart)."""
    if step_size is None:
        step_size = 2.0 * float(eps) / float(max(num_steps, 1))
    elif step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size!r}.")

    was_training = model.training
    model.eval()
    try:
        adv = x.detach().clone()
        for _ in range(num_steps):
            with torch.enable_grad():
                adv = adv.detach().requires_grad_(True)
                logits = model(adv)
                loss = F.cross_entropy(logits, y, reduction="sum")
                grad = torch.autograd.grad(loss, adv, create_graph=False)[0]
            with torch.no_grad():
                adv = adv + step_size * per_sample_l2_normalize(grad)
                adv = project_l2(adv, x, eps)
                adv = adv.clamp(0.0, 1.0)
        return adv.detach()
    finally:
        model.train(was_training)


def pgd_l2_attack_restarts(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_steps: int,
    step_size: float,
    restarts: int,
) -> torch.Tensor:
    """PGD-L2 with deterministic first restart + random subsequent restarts."""
    was_training = model.training
    model.eval()
    try:
        x0 = x.detach()
        best_delta = torch.zeros_like(x0)
        best_loss = torch.full((x0.size(0),), -float("inf"), device=x0.device)

        for r in range(max(1, restarts)):
            if r == 0:
                x_adv = x0.clone().detach()
            else:
                x_adv = sphere_start_l2(x0, eps).detach()

            for _ in range(num_steps):
                with torch.enable_grad():
                    x_adv = x_adv.detach().requires_grad_(True)
                    logits = model(x_adv)
                    loss = F.cross_entropy(logits, y, reduction="sum")
                    grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
                with torch.no_grad():
                    x_adv = x_adv + step_size * per_sample_l2_normalize(grad)
                    delta = project_onto_l2_ball(x_adv - x0, eps)
                    x_adv = (x0 + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                logits = model(x_adv)
                per_sample_loss = F.cross_entropy(logits, y, reduction="none")
                delta = (x_adv - x0).detach()
                improved = per_sample_loss > best_loss
                best_loss[improved] = per_sample_loss[improved]
                best_delta[improved] = delta[improved]

        return (x0 + best_delta).clamp(0.0, 1.0).detach()
    finally:
        model.train(was_training)
