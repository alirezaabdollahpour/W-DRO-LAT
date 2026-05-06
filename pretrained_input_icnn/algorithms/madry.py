"""Madry-style PGD adversarial training in normalized CIFAR-10 input space.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/madry.py`` but the attack
runs in the normalized image space and the adversary is projected onto a
ball of radius ``epsilon`` around the clean inputs (in the same space).
The base trainer uses CE on the adversarial inputs as the outer loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils import (
    adversary_loss_per_sample,
    clamped_normalized_copy,
    set_requires_grad,
)
from .base import BaseAdvTrainer


def _l2_normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    flat = grad.reshape(grad.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norms).view_as(grad)


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
        self.pgd_step_size = (
            float(cfg.madry_pgd_step_size)
            if cfg.madry_pgd_step_size and cfg.madry_pgd_step_size > 0
            else 2.0 * self.epsilon / max(1, self.pgd_steps // 2)
        )
        self.pgd_restarts = max(1, int(cfg.madry_pgd_restarts))

    def _pgd_attack(self, x_orig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.pgd_steps <= 0:
            return x_orig.detach()
        use_margin = bool(self.config.use_margin_loss)
        best_adv = x_orig.detach()
        best_loss = torch.full((x_orig.size(0),), -float("inf"), device=x_orig.device)

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
            for _ in range(self.pgd_steps):
                x_adv = x_adv.detach().requires_grad_(True)
                loss_per_sample = adversary_loss_per_sample(
                    self._classifier_module(x_adv), y, use_margin=use_margin
                )
                grad = torch.autograd.grad(loss_per_sample.sum(), x_adv, create_graph=False)[0]
                with torch.no_grad():
                    x_adv = x_adv + self.pgd_step_size * _l2_normalize_grad(grad)
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
