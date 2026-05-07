"""WRM (Sinha et al.) adversarial training — BB+Armijo on the input.

The original LR-CIFAR10 reference uses a constant-LR ascent
``z <- z + lr * (∇primary - 2λ(z - x))``. Here we replace that with the
SAME BB+Armijo step rule used by NPF, applied to the input variable z
directly (the per-batch DRO inner objective is
``primary(classifier(z), y) - λ ||z - x||^2``).

BB state is reset every batch — z changes per-batch, so the (s, y)
history from a previous batch isn't meaningful. K=cfg.wrm_inner_steps
BB+Armijo iterations are taken per batch. We clamp z to the valid
normalized pixel range after each accepted step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_tensor,
    clamped_normalized_copy,
    frozen_module,
)
from .base import BaseAdvTrainer


class WRMTrainer(BaseAdvTrainer):
    name = "wrm"

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)

        with frozen_module(self._classifier_module):
            def f_obj(z_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
                primary = adversary_loss_per_sample(
                    self._classifier_module(z_var), y, use_margin=use_margin
                )
                cost = (z_var - x).reshape(z_var.size(0), -1).pow(2).sum(dim=1)
                return (primary - lam * cost).mean()

            # Per-batch BB+Armijo state — z changes batch-to-batch so the
            # (s, y) history doesn't carry over.
            bb_state = BBArmijoState.create(
                alpha0=cfg.bb_alpha0,
                alpha_min=cfg.bb_alpha_min,
                alpha_max=cfg.bb_alpha_max,
                ls_c=cfg.bb_ls_c,
                ls_shrink=cfg.bb_ls_shrink,
                ls_max_steps=cfg.bb_ls_max_steps,
                reject_on_armijo_failure=True,
            )

            z = x.clone().detach()
            last_f_val = 0.0
            for _ in range(int(cfg.wrm_inner_steps)):
                z, bb_state, last_f_val, _ = bb_armijo_step_tensor(z, f_obj, bb_state)
                z = clamped_normalized_copy(z)
            self._last_inner_loss = last_f_val

            with torch.no_grad():
                # CE-on-clamped-z is recorded for cross-method consistency
                # (matches the per-method "inner_loss" diagnostic from before).
                self._last_inner_loss = float(
                    F.cross_entropy(self._classifier_module(z), y).item()
                )
        return z
