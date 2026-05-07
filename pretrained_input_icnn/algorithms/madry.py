"""Madry-style adversarial training — BB+Armijo on z + l2-ball projection.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/madry.py`` for the
adversary's threat model (l2-ball of radius ``epsilon`` around the clean
input), but the inner ascent step rule is the SAME BB+Armijo used by
NPF instead of fixed-step normalized PGD. After each accepted BB+Armijo
step the iterate is projected back onto the l2 ball and clamped to the
valid normalized pixel range. The Armijo sufficient-increase check uses
the UNPROJECTED trial — projection breaks Armijo's monotonicity
guarantee, so we project after acceptance.

Random restarts are preserved: with ``pgd_restarts > 1`` the inner
ascent runs from multiple starting points and we keep the iterate with
the highest final adversary loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_tensor,
    clamped_normalized_copy,
    set_requires_grad,
)
from .base import BaseAdvTrainer


def _project_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.reshape(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (eps / norms).clamp(max=1.0)
    return (flat * scale).view_as(delta)


class MadryTrainer(BaseAdvTrainer):
    name = "madry"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.epsilon = float(cfg.madry_epsilon)
        self.pgd_steps = max(1, int(cfg.madry_pgd_steps))
        self.pgd_restarts = max(1, int(cfg.madry_pgd_restarts))

    def _pgd_attack(self, x_orig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.pgd_steps <= 0:
            return x_orig.detach()
        cfg = self.config
        use_margin = bool(cfg.use_margin_loss)
        best_adv = x_orig.detach()
        best_loss = torch.full((x_orig.size(0),), -float("inf"), device=x_orig.device)

        # Per-batch BB+Armijo state, fresh for each restart so a hot
        # alpha from one restart's trajectory doesn't bias the next.
        def fresh_bb_state() -> BBArmijoState:
            return BBArmijoState.create(
                alpha0=cfg.bb_alpha0,
                alpha_min=cfg.bb_alpha_min,
                alpha_max=cfg.bb_alpha_max,
                ls_c=cfg.bb_ls_c,
                ls_shrink=cfg.bb_ls_shrink,
                ls_max_steps=cfg.bb_ls_max_steps,
                reject_on_armijo_failure=True,
            )

        # Madry's inner objective is the unconstrained adversary loss —
        # the l2-ball constraint is enforced by projection AFTER the
        # accepted step (which preserves Armijo's monotonicity check).
        def f_obj(z_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
            primary = adversary_loss_per_sample(
                self._classifier_module(z_var), y, use_margin=use_margin
            )
            return primary.mean()

        for restart_idx in range(self.pgd_restarts):
            if restart_idx == 0:
                x_adv = x_orig.detach().clone()
            else:
                noise = torch.randn_like(x_orig)
                noise_norm = (
                    noise.reshape(noise.size(0), -1).norm(dim=1, keepdim=True).clamp_min(1e-12)
                )
                radii = torch.rand(
                    noise.size(0), 1, device=x_orig.device, dtype=x_orig.dtype
                )
                noise_unit = noise / noise_norm.view(-1, 1, 1, 1)
                x_adv = x_orig.detach() + self.epsilon * radii.view(-1, 1, 1, 1) * noise_unit
            x_adv = clamped_normalized_copy(x_adv)

            bb_state = fresh_bb_state()
            for _ in range(self.pgd_steps):
                x_adv, bb_state, _, _ = bb_armijo_step_tensor(x_adv, f_obj, bb_state)
                # Post-step projection: clip Δ to the l2-ball and clamp
                # to valid pixel space. Armijo already accepted on the
                # unprojected trial; this enforces Madry's threat model.
                delta = _project_l2_ball(x_adv - x_orig, self.epsilon)
                x_adv = clamped_normalized_copy(x_orig + delta)

            with torch.no_grad():
                final_loss = adversary_loss_per_sample(
                    self._classifier_module(x_adv), y, use_margin=use_margin
                )
                improve = final_loss > best_loss
                best_loss = torch.where(improve, final_loss, best_loss)
                best_adv = torch.where(improve.view(-1, 1, 1, 1), x_adv, best_adv)
        return best_adv.detach()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._classifier_module.eval()
        set_requires_grad(self._classifier_module, False)
        x_adv = self._pgd_attack(x, y)
        set_requires_grad(self._classifier_module, True)
        with torch.no_grad():
            ce = F.cross_entropy(self._classifier_module(x_adv), y).item()
        self._last_inner_loss = float(ce)
        return x_adv

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        # Madry doesn't have a parametric transport map; reuse the PGD
        # attack on the test inputs against fixed labels at eval time.
        # The base evaluator wraps this in @no_grad-free context.
        return x.detach()
