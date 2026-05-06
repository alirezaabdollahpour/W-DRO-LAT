"""Sinkhorn SDRO dual / entropy-regularised WDRO baseline.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/dual.py`` for image inputs:
the inner objective is replaced with the closed-form Gibbs-sampling dual,
``log(1/m sum_j exp(ℓ(z_j)/(λε)))`` with z_j = x + sqrt(eps) * noise. We
clamp the noisy samples back to the valid normalized pixel range so the
classifier never sees out-of-domain values.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..utils import clamped_normalized_copy, set_requires_grad
from .base import BaseAdvTrainer


class SDRODualTrainer(BaseAdvTrainer):
    name = "dual"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.epsilon = float(cfg.dual_epsilon)
        self.sample_level = max(1, int(cfg.dual_sample_level))
        # The Sinkhorn dual is a single-shot loss — we don't have a separate
        # "adversary" to update, so the inner step here only constructs the
        # noisy expansion that the classifier_update consumes.

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        levels = np.arange(self.sample_level + 1)
        numerators = 2.0 ** (-levels)
        denominator = 2.0 - 2.0 ** (-self.sample_level)
        probabilities = numerators / denominator
        sampled_level = int(np.random.choice(levels, p=probabilities))
        m = 2 ** sampled_level

        x_rep = x.repeat_interleave(m, dim=0)
        noise = torch.randn_like(x_rep) * math.sqrt(self.epsilon)
        z = clamped_normalized_copy(x_rep + noise)
        y_rep = y.repeat_interleave(m, dim=0)

        self._dual_z = z.detach()
        self._dual_y_rep = y_rep
        self._dual_m = m
        self._dual_lam_reg = float(cfg.lambda_param) * self.epsilon

        with torch.no_grad():
            ce = nn.CrossEntropyLoss()(self.classifier(z), y_rep)
            self._last_inner_loss = float(ce.item())
        # Return per-sample top-CE z for diagnostics.
        with torch.no_grad():
            ce_each = nn.CrossEntropyLoss(reduction="none")(self.classifier(z), y_rep)
            ce_view = ce_each.view(x.size(0), m)
            top = ce_view.argmax(dim=1, keepdim=True)
            z_view = z.view(x.size(0), m, *x.shape[1:])
            gather_idx = top.view(-1, 1, 1, 1, 1).expand(-1, 1, *x.shape[1:])
            return z_view.gather(1, gather_idx).squeeze(1).detach()

    def classifier_update(self, x_adv, y):
        self.classifier.train()
        set_requires_grad(self.classifier, True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.classifier(self._dual_z)
        criterion = nn.CrossEntropyLoss(reduction="none")
        residuals = criterion(logits, self._dual_y_rep) / max(self._dual_lam_reg, 1e-8)
        residual_matrix = residuals.view(-1, self._dual_m).T
        loss = (
            torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(self._dual_m))
            * self._dual_lam_reg
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.classifier.parameters() if p.requires_grad], max_norm=10.0
        )
        self.optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == self._dual_y_rep).float().mean().item()
        return float(loss.item()), float(acc)

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach()
