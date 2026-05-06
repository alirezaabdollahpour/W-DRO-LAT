"""WRM (Sinha et al.) adversarial training in normalized CIFAR-10 input space.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/wrm.py``. The inner sampler
ascends ``ℓ(f(z), y) - λ ||z - x||^2`` with a constant learning rate. We
clamp z to the valid normalized pixel range each step so the classifier
never sees out-of-domain inputs.
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


class WRMTrainer(BaseAdvTrainer):
    name = "wrm"

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)

        self._classifier_module.eval()
        set_requires_grad(self._classifier_module, False)

        z = x.clone().detach()
        for _ in range(int(cfg.wrm_inner_steps)):
            z.requires_grad_(True)
            primary = adversary_loss_per_sample(
                self._classifier_module(z), y, use_margin=use_margin
            )
            grads = torch.autograd.grad(primary.sum(), z, create_graph=False)[0]
            with torch.no_grad():
                z = z.detach() + cfg.wrm_inner_lr * (grads - 2.0 * lam * (z.detach() - x))
                z = clamped_normalized_copy(z)
        z = z.detach()
        with torch.no_grad():
            self._last_inner_loss = float(
                F.cross_entropy(self._classifier_module(z), y).item()
            )
        set_requires_grad(self._classifier_module, True)
        return z

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach()
