"""NPF-style WDRO adversarial training with BB+Armijo inner ascent.

This is the input-space CIFAR-10 counterpart of
``Logistic_Regression_CIFAR10/algorithms/npf.py``. The NPF ICNN potential
operates on flattened CIFAR images; the transport map T_ω(x) = ∇_x ψ_ω(x)
is reshaped back to (B, 3, 32, 32) and clamped to the valid normalized
pixel range before being fed to the classifier (this is the only
modification needed for CIFAR-10 input-space — feature-space NPF skips
the clamp because features are unbounded).

Outer loop: SGD on the classifier.
Inner loop: BB+Armijo on ω optimising

    max_ω  E[ CE(f(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]

Hyperparameters mirror ``config.py`` from the LR-CIFAR10 reference: the
``cfg.npf_bb_*`` defaults and the principled LogNormal + identity init are
used so the only modeling difference between this and the LR setting is
the input domain.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from ..models.npf import NPFInputConvexPotential, npf_T_omega
from ..utils import (
    BBArmijoState,
    adversary_loss_per_sample,
    bb_armijo_step_params,
    clamped_normalized_copy,
    set_requires_grad,
)
from .base import BaseAdvTrainer


class NPFTrainer(BaseAdvTrainer):
    name = "npf"
    has_parametric_adversary = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        # Input dim = product of image dims (3 * 32 * 32 for CIFAR).
        self.input_dim = int(3 * 32 * 32)
        self.psi_omega = NPFInputConvexPotential(
            input_dim=self.input_dim,
            hidden_sizes=cfg.npf_hidden,
            outer_rank=cfg.npf_outer_rank,
            inner_rank=cfg.npf_inner_rank,
            activation=cfg.npf_activation,
            elu_alpha=cfg.npf_elu_alpha,
            softplus_beta=cfg.npf_softplus_beta,
            init_eps=cfg.npf_init_eps,
            strong_convexity=cfg.npf_strong_convexity,
        ).to(self.device)
        # Identity init applied on top of the LogNormal draws produced inside
        # the non-negative layers' constructor.
        self.psi_omega.init_as_identity()

        self.bb_state = BBArmijoState.create(
            alpha0=cfg.npf_bb_alpha0,
            alpha_min=cfg.npf_bb_alpha_min,
            alpha_max=cfg.npf_bb_alpha_max,
            ls_c=cfg.npf_bb_ls_c,
            ls_shrink=cfg.npf_bb_ls_shrink,
            ls_max_steps=cfg.npf_bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

    # ------------------------------------------------------------------
    def _transport(self, x: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        """T_ω(x) = clamp(∇ψ_ω(x))."""
        x_adv = npf_T_omega(x, self.psi_omega, create_graph=create_graph)
        return clamped_normalized_copy(x_adv)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lambda_param = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)

        # Inner loop: ω-ascent via BB+Armijo. Freeze the classifier so its
        # gradients are not accumulated; bb_armijo_step_params already
        # routes gradients only to ψ_ω parameters, but the explicit
        # freeze keeps the contract robust to future refactors.
        self.classifier.eval()
        self.psi_omega.train()
        set_requires_grad(self.classifier, False)
        set_requires_grad(self.psi_omega, True)

        def omega_objective(create_graph: bool) -> torch.Tensor:
            x_adv = self._transport(x, create_graph=create_graph)
            logits = self.classifier(x_adv)
            primary = adversary_loss_per_sample(logits, y, use_margin=use_margin)
            transport_cost = (x_adv - x).reshape(x.size(0), -1).pow(2).sum(dim=1)
            obj = (primary - lambda_param * transport_cost).mean()
            return torch.nan_to_num(obj, nan=-1e12, posinf=-1e12, neginf=-1e12)

        last_f_val = 0.0
        for _ in range(int(cfg.omega_steps_per_batch)):
            _, self.bb_state, last_f_val, _ = bb_armijo_step_params(
                self.psi_omega.parameters(),
                omega_objective,
                self.bb_state,
            )
        self._last_inner_loss = last_f_val

        # Switch contract for the outer step. The base trainer will set
        # the classifier back to train()/requires_grad=True before
        # backprop; we just need to detach the adversarial features.
        self.psi_omega.eval()
        set_requires_grad(self.psi_omega, False)
        with torch.no_grad():
            x_adv = self._transport(x, create_graph=False)
        return x_adv.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.psi_omega.training
        self.psi_omega.eval()
        try:
            x_adv = self._transport(x, create_graph=False)
        finally:
            self.psi_omega.train(was_training)
        return x_adv.detach()

    def adversary_state_dicts(self) -> Dict[str, Any]:
        return {"psi_omega": self.psi_omega.state_dict()}
