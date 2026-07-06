"""WRM (Sinha et al.) adversarial training — BB+Armijo on the input.

The original LR-CIFAR10 reference uses a constant-LR ascent
``z <- z + lr * (∇primary - 2λ(z - x))``. Here we replace that with the
SAME BB+Armijo step rule used by NPF, applied to the input variable z
directly (the per-batch DRO inner objective is
``primary(classifier(z), y) - λ ||z - x||^2``).

BB state is reset every batch — z changes per-batch, so the (s, y)
history from a previous batch isn't meaningful. K=cfg.wrm_inner_steps
BB+Armijo iterations are taken per batch. We clamp z to the valid
normalized pixel range after each accepted step. The penalty uses the
configured transport-cost convention, defaulting to legacy normalized MSE.
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

        with self.profile_time("step_wall_s"):
            return self._profiled_step_impl(x, y, lam, use_margin)

    def _profiled_step_impl(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        use_margin: bool,
    ) -> torch.Tensor:
        cfg = self.config

        with frozen_module(
            self._classifier_module,
            eval_mode=self._adversary_classifier_eval_mode(),
        ):
            def f_obj(z_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
                self.profile_add("objective_calls", 1.0)
                self.profile_add(
                    "objective_create_graph_calls" if create_graph else "objective_no_graph_calls",
                    1.0,
                )
                with self.profile_time("objective_classifier_s"):
                    logits = self._classifier_module(z_var)
                with self.profile_time("objective_loss_cost_s"):
                    primary = adversary_loss_per_sample(
                        logits, y, use_margin=use_margin
                    )
                    cost = self._transport_cost(z_var, x)
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
            with self.profile_time("inner_loop_wall_s"):
                for _ in range(int(cfg.wrm_inner_steps)):
                    z, bb_state, last_f_val, _ = bb_armijo_step_tensor(
                        z,
                        f_obj,
                        bb_state,
                        profile=self if self.is_inner_profile_active else None,
                    )
                    with self.profile_time("wrm_clamp_s"):
                        z = clamped_normalized_copy(z)
            self._last_inner_loss = last_f_val

            with torch.no_grad():
                # CE-on-clamped-z is recorded for cross-method consistency
                # (matches the per-method "inner_loss" diagnostic from before).
                with self.profile_time("final_ce_s"):
                    self._last_inner_loss = float(
                        F.cross_entropy(self._classifier_module(z), y).item()
                    )
        return z
