"""Wasserstein-Fisher-Rao (WFR) sampler in normalized CIFAR-10 input space.

Adapted from ``Logistic_Regression_CIFAR10/algorithms/wfr.py``. Maintains
``num_samples`` particles per data point and reweights them by the inner
energy each step. The base trainer's outer loss uses the importance-
weighted CE on the final particle ensemble (computed inside ``step``).

For uniform comparison with the other adversaries, the deterministic
gradient component of the Langevin step is taken via the SAME
BB+Armijo step rule as NPF. The Gaussian noise (the Fisher-Rao part)
is then added on top. The deterministic drift and the particle
reweighting use the same primary - lambda * ||z - x||^2 energy so the
lambda convention matches the other DRO trainers. The squared transport
cost uses the configured convention; the Langevin noise is still injected
in pixel coordinates before converting back to normalized tensors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_tensor,
    clamped_normalized_copy,
    frozen_module,
    to_normalized,
    to_pixel,
)
from .base import BaseAdvTrainer


class WFRTrainer(BaseAdvTrainer):
    name = "wfr"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.epsilon = float(cfg.wfr_epsilon)
        self.num_samples = max(1, int(cfg.wfr_num_samples))
        self.inner_lr = float(cfg.wfr_inner_lr)
        self.inner_steps = max(1, int(cfg.wfr_inner_steps))

    def _sampler(self, x_orig: torch.Tensor, y: torch.Tensor):
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        eps = self.epsilon
        m = self.num_samples
        bs = x_orig.size(0)
        device = x_orig.device

        x_rep = x_orig.detach().unsqueeze(1).expand(-1, m, -1, -1, -1).reshape(
            bs * m, *x_orig.shape[1:]
        ).contiguous()
        y_rep = y.repeat_interleave(m, dim=0)
        x_anchor = x_rep.clone()
        z = x_rep.clone()

        weights = torch.full((bs, m), 1.0 / m, device=device, dtype=x_orig.dtype)
        wt_lr = self.inner_lr
        weight_exponent = 1.0 - lam * eps * wt_lr
        std_dev = math.sqrt(max(2.0 * self.inner_lr * lam * eps, 0.0))
        low_weight_threshold = 1e-4

        # WFR's deterministic step is gradient ascent on
        # ``CE(classifier(z), y_rep) - λ ||z - x_anchor||^2`` (this is
        # the WRM-style Langevin drift; see Sinha et al. and the WFR
        # paper). We take that step via BB+Armijo, then add Langevin
        # noise. BB state is fresh per batch.
        bb_state = BBArmijoState.create(
            alpha0=cfg.bb_alpha0,
            alpha_min=cfg.bb_alpha_min,
            alpha_max=cfg.bb_alpha_max,
            ls_c=cfg.bb_ls_c,
            ls_shrink=cfg.bb_ls_shrink,
            ls_max_steps=cfg.bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

        def f_obj(z_var: torch.Tensor, create_graph: bool) -> torch.Tensor:
            ce = adversary_loss_per_sample(
                self._classifier_module(z_var), y_rep, use_margin=use_margin
            )
            cost = self._transport_cost(z_var, x_anchor)
            return (ce - lam * cost).mean()

        for _ in range(self.inner_steps):
            z, bb_state, _, _ = bb_armijo_step_tensor(z, f_obj, bb_state)
            with torch.no_grad():
                if std_dev > 0:
                    z_pix = (to_pixel(z) + torch.randn_like(z) * std_dev).clamp(0.0, 1.0)
                    z = to_normalized(z_pix)
                else:
                    z = clamped_normalized_copy(z)

                cur_loss = adversary_loss_per_sample(
                    self._classifier_module(z), y_rep, use_margin=use_margin
                ).view(bs, m)
                dist_sq = self._transport_cost(z, x_anchor).view(bs, m)
                # Same objective as the deterministic drift:
                # primary(classifier(z), y) - lambda * ||z - x||^2.
                energy = cur_loss - lam * dist_sq

                weights.pow_(weight_exponent)
                weights.mul_(torch.exp(energy * wt_lr))
                weights.div_(weights.sum(dim=1, keepdim=True).add_(1e-9))

                low_mask = weights < low_weight_threshold
                rows_with_low = low_mask.any(dim=1, keepdim=True)
                if rows_with_low.any():
                    z_view = z.view(bs, m, *x_orig.shape[1:])
                    max_w_vals, max_w_idx = weights.max(dim=1, keepdim=True)
                    expanded_idx = max_w_idx.view(bs, 1, 1, 1, 1).expand(
                        bs, 1, *x_orig.shape[1:]
                    )
                    top_z = z_view.gather(1, expanded_idx)
                    low_sum = (weights * low_mask).sum(dim=1, keepdim=True)
                    low_count = low_mask.sum(dim=1, keepdim=True, dtype=weights.dtype)
                    avg_w = (max_w_vals + low_sum) / (low_count + 1.0 + 1e-9)
                    max_mask = torch.zeros_like(weights, dtype=torch.bool)
                    max_mask.scatter_(1, max_w_idx, True)
                    update_mask = (low_mask | max_mask) & rows_with_low
                    weights = torch.where(update_mask, avg_w, weights)
                    z_update = (low_mask & rows_with_low).view(
                        bs, m, *([1] * (z_view.dim() - 2))
                    )
                    z_view = torch.where(z_update, top_z, z_view)
                    z = z_view.reshape(bs * m, *x_orig.shape[1:])
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return z.detach(), weights.detach(), y_rep

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with frozen_module(
            self._classifier_module,
            eval_mode=self._adversary_classifier_eval_mode(),
        ):
            z, weights, y_rep = self._sampler(x, y)
            # Stash for the WFR-specific outer update below.
            self._wfr_z = z
            self._wfr_w = weights
            self._wfr_y_rep = y_rep
            with torch.no_grad():
                ce = F.cross_entropy(self._classifier_module(z), y_rep, reduction="none")
                obj = (ce.view(weights.size(0), self.num_samples) * weights).sum(dim=1).mean()
                self._last_inner_loss = float(obj.item())
        # Return the highest-weight particle per sample so the base trainer
        # can compute MSE / clean-vs-adv accuracy. The actual classifier
        # update overrides that path via ``classifier_update`` below.
        m = self.num_samples
        z_view = z.view(weights.size(0), m, *x.shape[1:])
        best_idx = weights.argmax(dim=1, keepdim=True)
        gather_idx = best_idx.view(-1, 1, 1, 1, 1).expand(-1, 1, *x.shape[1:])
        return z_view.gather(1, gather_idx).squeeze(1).detach()

    def classifier_update(self, x_adv, y):
        # Override the base trainer to use the importance-weighted loss
        # over all particles rather than only the top-weight survivor.
        self._prepare_classifier_for_update()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.classifier(self._wfr_z)
        ce = nn.CrossEntropyLoss(reduction="none")(logits, self._wfr_y_rep)
        loss = (
            ce.view(self._wfr_w.size(0), self.num_samples) * self._wfr_w
        ).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.classifier.parameters() if p.requires_grad], max_norm=10.0
        )
        self.optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == self._wfr_y_rep).float().mean().item()
        return float(loss.item()), float(acc)
