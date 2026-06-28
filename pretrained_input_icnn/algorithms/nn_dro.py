"""NN-DRO competitor: vanilla MLP adversary, BB+Armijo ascent on ω.

Mirrors ``Logistic_Regression_CIFAR10/algorithms/nn_dro.py``: the adversary
is parametrised directly as ``T_ω(x) = x + h_ω(x)`` with ``h_ω`` a plain
MLP. We clamp the resulting adversarial inputs to the valid normalized
pixel range so the classifier is never fed values outside its training
support.

The Wasserstein penalty uses the configured transport-cost convention
while the adversary and classifier exchange normalized CIFAR tensors.
The default matches the legacy input-ICNN code: normalized-coordinate MSE.

The original LR-CIFAR10 reference used Adam on ω. Here we replace that
with the SAME BB+Armijo step rule used by NPF, so every parametric
adversary in the runtime sweep is optimised by an identical step rule
and only the adversary architecture differs.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from .. import distributed as dist_helpers
from ..models.nn_dro import MLPAdversary
from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_params,
    clamped_normalized_copy,
    frozen_module,
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
        # Persistent BB+Armijo state on the MLP adversary parameters —
        # ω lives across batches so the (s, y) history carries over.
        self.bb_state = self._new_bb_state()

    def _new_bb_state(self) -> BBArmijoState:
        cfg = self.config
        return BBArmijoState.create(
            alpha0=cfg.bb_alpha0,
            alpha_min=cfg.bb_alpha_min,
            alpha_max=cfg.bb_alpha_max,
            ls_c=cfg.bb_ls_c,
            ls_shrink=cfg.bb_ls_shrink,
            ls_max_steps=cfg.bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

    def _transport(self, x: torch.Tensor) -> torch.Tensor:
        return clamped_normalized_copy(self.adversary(x))

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lam = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)
        attack_mask = None
        if bool(getattr(cfg, "attack_clean_correct_only", True)):
            attack_mask = self._clean_correct_attack_mask(x, y)
        reset_bb_each_batch = bool(getattr(cfg, "reset_parametric_bb_each_batch", True))
        bb_state = self._new_bb_state() if reset_bb_each_batch else self.bb_state

        # Inner loop: BB+Armijo ascent on ω with the classifier frozen.
        self.adversary.train()
        set_requires_grad(self.adversary, True)

        with frozen_module(self._classifier_module):
            def omega_objective(create_graph: bool) -> torch.Tensor:
                x_adv = self._transport(x)  # MLP forward; create_graph unused
                logits = self._classifier_module(x_adv)
                primary = adversary_loss_per_sample(logits, y, use_margin=use_margin)
                cost = self._transport_cost(x_adv, x)
                obj = self._shared_adversary_masked_mean(
                    primary - lam * cost,
                    attack_mask,
                )
                return torch.nan_to_num(obj, nan=-1e12, posinf=-1e12, neginf=-1e12)

            # Same DDP reducers as NPF: ω is shared across ranks, so each
            # rank's local gradient must be averaged before the step.
            reduce_grad_fn = (
                dist_helpers.all_reduce_grad_list if self.dist.is_distributed else None
            )
            reduce_scalar_fn = (
                dist_helpers.all_reduce_scalar if self.dist.is_distributed else None
            )

            last_f_val = 0.0
            for _ in range(int(cfg.omega_steps_per_batch)):
                _, bb_state, last_f_val, _ = bb_armijo_step_params(
                    self.adversary.parameters(),
                    omega_objective,
                    bb_state,
                    reduce_grad_fn=reduce_grad_fn,
                    reduce_scalar_fn=reduce_scalar_fn,
                    max_grad_norm=getattr(cfg, "parametric_bb_max_grad_norm", 1.0),
                )
        if not reset_bb_each_batch:
            self.bb_state = bb_state
        self._last_inner_loss = last_f_val

        self.adversary.eval()
        set_requires_grad(self.adversary, False)
        with torch.no_grad():
            x_adv = self._transport(x)
            x_adv = self._keep_clean_for_unattacked(x_adv, x, attack_mask)
        return x_adv.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.adversary.training
        self.adversary.eval()
        try:
            x_adv = clamped_normalized_copy(self.adversary(x))
        finally:
            self.adversary.train(was_training)
        return x_adv.detach()

    def freeze_adversary_parameters(self) -> None:
        self.adversary.eval()
        set_requires_grad(self.adversary, False)

    def adversary_state_dicts(self) -> Dict[str, Any]:
        return {"adversary": self.adversary.state_dict()}
