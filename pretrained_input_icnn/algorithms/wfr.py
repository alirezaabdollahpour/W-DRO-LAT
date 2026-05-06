"""Wasserstein-Fisher-Rao (WFR) sampler in normalized CIFAR-10 input space.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/wfr.py``. Maintains
``num_samples`` particles per data point and reweights them by the inner
energy each step. The base trainer's outer loss uses the importance-
weighted CE on the final particle ensemble (computed inside ``step``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import clamped_normalized_copy, set_requires_grad
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
        criterion = nn.CrossEntropyLoss(reduction="none")
        low_weight_threshold = 1e-4

        for _ in range(self.inner_steps):
            z.requires_grad_(True)
            losses = criterion(self._classifier_module(z), y_rep)
            grads = torch.autograd.grad(losses.sum(), z, create_graph=False)[0]
            with torch.no_grad():
                mean = z.detach() + self.inner_lr * (grads - 2.0 * lam * (z.detach() - x_anchor))
                noise = torch.randn_like(mean) * std_dev if std_dev > 0 else 0.0
                z = clamped_normalized_copy(mean + noise) if std_dev > 0 else clamped_normalized_copy(mean)

                cur_loss = criterion(self._classifier_module(z), y_rep).view(bs, m)
                dist_sq = (z - x_anchor).reshape(bs * m, -1).pow(2).sum(dim=1).view(bs, m)
                energy = cur_loss - 2.0 * lam * dist_sq

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
        self._classifier_module.eval()
        set_requires_grad(self._classifier_module, False)
        z, weights, y_rep = self._sampler(x, y)
        # Stash for the WFR-specific outer update below.
        self._wfr_z = z
        self._wfr_w = weights
        self._wfr_y_rep = y_rep
        with torch.no_grad():
            ce = F.cross_entropy(self._classifier_module(z), y_rep, reduction="none")
            obj = (ce.view(weights.size(0), self.num_samples) * weights).sum(dim=1).mean()
            self._last_inner_loss = float(obj.item())
        set_requires_grad(self._classifier_module, True)
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
        self.classifier.train()
        set_requires_grad(self.classifier, True)
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
