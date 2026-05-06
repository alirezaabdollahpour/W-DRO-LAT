"""NN-DRO competitor: vanilla MLP adversary trained with Adam ascent.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/nn_dro.py``: the adversary
is parametrised directly as ``T_ω(x) = x + h_ω(x)`` with ``h_ω`` a plain
MLP. We clamp the resulting adversarial inputs to the valid normalized
pixel range so the classifier is never fed values outside its training
support.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.optim as optim

from .. import distributed as dist_helpers
from ..models.nn_dro import MLPAdversary
from ..utils import (
    adversary_loss_per_sample,
    clamped_normalized_copy,
    set_requires_grad,
)
from .base import BaseAdvTrainer


class NNDROTrainer(BaseAdvTrainer):
    name = "nn_dro"
    has_parametric_adversary = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.input_dim = int(3 * 32 * 32)
        self.adversary = MLPAdversary(
            input_dim=self.input_dim,
            hidden_sizes=tuple(cfg.nn_dro_hidden),
            activation=cfg.nn_dro_activation,
            softplus_beta=cfg.nn_dro_softplus_beta,
            init_scale=cfg.nn_dro_init_scale,
        ).to(self.device)
        self.inner_opt = optim.Adam(self.adversary.parameters(), lr=cfg.nn_dro_inner_lr)

    def _transport(self, x: torch.Tensor) -> torch.Tensor:
        return clamped_normalized_copy(self.adversary(x))

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)

        # Inner loop: Adam ascent on ω with the classifier frozen.
        self._classifier_module.eval()
        self.adversary.train()
        set_requires_grad(self._classifier_module, False)
        set_requires_grad(self.adversary, True)

        last_obj = 0.0
        for _ in range(int(cfg.omega_steps_per_batch)):
            self.inner_opt.zero_grad(set_to_none=True)
            x_adv = self._transport(x)
            logits = self._classifier_module(x_adv)
            primary = adversary_loss_per_sample(logits, y, use_margin=use_margin)
            cost = (x_adv - x).reshape(x.size(0), -1).pow(2).sum(dim=1)
            obj = (primary - lam * cost).mean()
            (-obj).backward()
            # All-reduce ω gradients before Adam.step() — Adam's update
            # is local, so without this each rank would diverge in ω.
            dist_helpers.all_reduce_grads_(self.adversary.parameters())
            self.inner_opt.step()
            last_obj = float(obj.detach().item())
        self._last_inner_loss = last_obj

        self.adversary.eval()
        set_requires_grad(self.adversary, False)
        with torch.no_grad():
            x_adv = self._transport(x)
        return x_adv.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.adversary.training
        self.adversary.eval()
        try:
            x_adv = clamped_normalized_copy(self.adversary(x))
        finally:
            self.adversary.train(was_training)
        return x_adv.detach()

    def adversary_state_dicts(self) -> Dict[str, Any]:
        return {"adversary": self.adversary.state_dict()}
