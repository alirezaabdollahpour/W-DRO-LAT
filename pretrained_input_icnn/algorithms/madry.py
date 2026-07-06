"""Madry / RO adversarial training with fixed-step l2-PGD.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/madry.py`` for the
adversary's threat model: maximise the adversary loss over a pixel-space
l2 ball of radius ``epsilon`` around the clean input using projected
gradient ascent. Unlike the DRO methods, RO has no transport penalty
lambda; the constraint is the epsilon-ball projection itself.

Random restarts are preserved: with ``pgd_restarts > 1`` the inner
ascent runs from multiple starting points and we keep the iterate with
the highest final adversary loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils import (
    adversary_loss_per_sample,
    frozen_module,
    to_normalized,
    to_pixel,
)
from .base import BaseAdvTrainer


def _project_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.reshape(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (eps / norms).clamp(max=1.0)
    return (flat * scale).view_as(delta)


def _l2_normalize(v: torch.Tensor) -> torch.Tensor:
    flat = v.reshape(v.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norms).view_as(v)


class MadryTrainer(BaseAdvTrainer):
    name = "madry"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.epsilon = float(cfg.madry_epsilon)
        self.pgd_steps = int(cfg.madry_pgd_steps)
        self.pgd_restarts = max(1, int(cfg.madry_pgd_restarts))
        raw_step_size = float(cfg.madry_pgd_step_size)
        self.pgd_step_size = (
            raw_step_size
            if raw_step_size > 0.0
            else 2.0 * self.epsilon / max(1, self.pgd_steps)
        )

    def _pgd_attack(self, x_orig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.pgd_steps <= 0:
            return x_orig.detach()
        cfg = self.config
        use_margin = bool(cfg.use_margin_loss)
        x0_pix = to_pixel(x_orig).detach().clamp(0.0, 1.0)
        best_adv_pix = x0_pix.detach()
        best_loss = torch.full((x_orig.size(0),), -float("inf"), device=x_orig.device)

        for restart_idx in range(self.pgd_restarts):
            if restart_idx == 0:
                x_adv_pix = x0_pix.detach().clone()
            else:
                noise = torch.randn_like(x0_pix)
                noise_norm = (
                    noise.reshape(noise.size(0), -1).norm(dim=1, keepdim=True).clamp_min(1e-12)
                )
                radii = torch.rand(
                    noise.size(0), 1, device=x_orig.device, dtype=x_orig.dtype
                )
                noise_unit = noise / noise_norm.view(-1, 1, 1, 1)
                x_adv_pix = x0_pix + self.epsilon * radii.view(-1, 1, 1, 1) * noise_unit
            x_adv_pix = x_adv_pix.clamp(0.0, 1.0)

            for _ in range(self.pgd_steps):
                x_var = x_adv_pix.detach().clone().requires_grad_(True)
                primary = adversary_loss_per_sample(
                    self._classifier_module(to_normalized(x_var)),
                    y,
                    use_margin=use_margin,
                )
                grad = torch.autograd.grad(primary.sum(), x_var, create_graph=False)[0]
                with torch.no_grad():
                    x_adv_pix = x_var + self.pgd_step_size * _l2_normalize(grad)
                    delta = _project_l2_ball(x_adv_pix - x0_pix, self.epsilon)
                    x_adv_pix = (x0_pix + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                final_loss = adversary_loss_per_sample(
                    self._classifier_module(to_normalized(x_adv_pix)),
                    y,
                    use_margin=use_margin,
                )
                improve = final_loss > best_loss
                best_loss = torch.where(improve, final_loss, best_loss)
                best_adv_pix = torch.where(
                    improve.view(-1, 1, 1, 1), x_adv_pix, best_adv_pix
                )
        return to_normalized(best_adv_pix).detach()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with frozen_module(
            self._classifier_module,
            eval_mode=self._adversary_classifier_eval_mode(),
        ):
            x_adv = self._pgd_attack(x, y)
            with torch.no_grad():
                ce = F.cross_entropy(self._classifier_module(x_adv), y).item()
        self._last_inner_loss = float(ce)
        return x_adv

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        # Madry/RO has no persistent transport map. Robustness is measured
        # by the separate input-PGD evaluator when enabled.
        return x.detach()
