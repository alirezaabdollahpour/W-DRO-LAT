"""NPF-style WDRO adversarial training with selectable inner ascent.

This is the input-space CIFAR-10 counterpart of
``Logistic_Regression_CIFAR10/algorithms/npf.py``. The NPF ICNN potential
operates on flattened CIFAR images; the transport map T_ω(x) = ∇_x ψ_ω(x)
is reshaped back to (B, 3, 32, 32) and clamped to the valid normalized
pixel range before being fed to the classifier (this is the only
modification needed for CIFAR-10 input-space — feature-space NPF skips
the clamp because features are unbounded).

Outer loop: SGD on the classifier.
Inner loop: configurable optimizer on ω optimising

    max_ω  E[ CE(f(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]

where classifier inputs are normalized CIFAR tensors and the squared
transport cost uses the configured convention. The default is the legacy
pretrained_INPUT_icnn.py cost: per-sample mean squared difference in
normalized CIFAR coordinates.

The default inner optimizer is the existing BB+Armijo ascent. Passing
``--npf-inner-optimizer muon`` switches only the NPF / NPF-LastQuad
adversary update to Muon, leaving the objective, transport map, classifier
outer update, and evaluation path unchanged.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from .. import distributed as dist_helpers
from ..models.npf import NPFInputConvexPotential, npf_T_omega
from ..utils import (
    BBArmijoState,
    MuonState,
    adversary_loss_per_sample,
    bb_armijo_step_params,
    clamped_normalized_copy,
    frozen_module,
    muon_step_params,
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
        self.psi_omega = NPFInputConvexPotential(**self._potential_kwargs()).to(
            self.device
        )
        # Identity init applied on top of the LogNormal draws produced inside
        # the non-negative layers' constructor.
        self.psi_omega.init_as_identity()

        self.npf_inner_optimizer = str(
            getattr(cfg, "npf_inner_optimizer", "bb_armijo")
        ).lower()
        if self.npf_inner_optimizer not in {"bb_armijo", "muon"}:
            raise ValueError(
                f"Unsupported NPF inner optimizer '{self.npf_inner_optimizer}'. "
                "Use 'bb_armijo' or 'muon'."
            )

        self.bb_state = None
        self.muon_state = None
        if self.npf_inner_optimizer == "bb_armijo":
            # This state is used only when --persistent-parametric-bb is set.
            # The default legacy-compatible path resets BB history per batch.
            self.bb_state = self._new_bb_state()
        else:
            self.muon_state = MuonState.create(
                lr=cfg.npf_muon_lr,
                momentum=cfg.npf_muon_momentum,
                nesterov=cfg.npf_muon_nesterov,
                ns_steps=cfg.npf_muon_ns_steps,
                matrix_lr_scale=cfg.npf_muon_matrix_lr_scale,
                weight_decay=cfg.npf_muon_weight_decay,
                fallback=cfg.npf_muon_fallback,
                fallback_lr=cfg.npf_muon_fallback_lr,
                fallback_weight_decay=cfg.npf_muon_fallback_weight_decay,
                adam_beta1=cfg.npf_muon_adam_beta1,
                adam_beta2=cfg.npf_muon_adam_beta2,
                adam_eps=cfg.npf_muon_adam_eps,
                max_grad_norm=cfg.npf_muon_max_grad_norm,
            )

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

    def _potential_kwargs(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": cfg.npf_hidden,
            "outer_rank": cfg.npf_outer_rank,
            "inner_rank": cfg.npf_inner_rank,
            "quadratic_mode": "all_layers",
            "trainable_outer_quadratic": True,
            "activation": cfg.npf_activation,
            "elu_alpha": cfg.npf_elu_alpha,
            "softplus_beta": cfg.npf_softplus_beta,
            "init_eps": cfg.npf_init_eps,
            "strong_convexity": cfg.npf_strong_convexity,
        }

    # ------------------------------------------------------------------
    def _transport(self, x: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        """T_ω(x) = clamp(∇ψ_ω(x))."""
        x_adv = npf_T_omega(x, self.psi_omega, create_graph=create_graph)
        return clamped_normalized_copy(x_adv)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        lambda_param = float(cfg.lambda_param)
        use_margin = bool(cfg.use_margin_loss)

        with self.profile_time("step_wall_s"):
            return self._profiled_step_impl(x, y, lambda_param, use_margin)

    def _profiled_step_impl(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_param: float,
        use_margin: bool,
    ) -> torch.Tensor:
        cfg = self.config
        attack_mask = None
        if bool(getattr(cfg, "attack_clean_correct_only", True)):
            attack_mask = self._clean_correct_attack_mask(x, y)
        reset_bb_each_batch = bool(getattr(cfg, "reset_parametric_bb_each_batch", True))
        bb_state = self._new_bb_state() if reset_bb_each_batch else self.bb_state

        # Inner loop: ω-ascent. Freeze the classifier so its gradients are not
        # accumulated; the inner optimizers route gradients only to ψ_ω
        # parameters, but the explicit freeze keeps the contract robust to
        # future refactors.
        self.psi_omega.train()
        set_requires_grad(self.psi_omega, True)

        with frozen_module(self._classifier_module):
            def omega_objective(create_graph: bool) -> torch.Tensor:
                self.profile_add("objective_calls", 1.0)
                self.profile_add(
                    "objective_create_graph_calls" if create_graph else "objective_no_graph_calls",
                    1.0,
                )
                with self.profile_time("objective_transport_s"):
                    x_adv = self._transport(x, create_graph=create_graph)
                # Use the unwrapped classifier — DDP's reducer must not be
                # disturbed by the many forwards in the inner ascent.
                with self.profile_time("objective_classifier_s"):
                    logits = self._classifier_module(x_adv)
                with self.profile_time("objective_loss_cost_s"):
                    primary = adversary_loss_per_sample(logits, y, use_margin=use_margin)
                    transport_cost = self._transport_cost(x_adv, x)
                    objective_per_sample = primary - lambda_param * transport_cost
                    obj = self._masked_mean(objective_per_sample, attack_mask)
                return torch.nan_to_num(obj, nan=-1e12, posinf=-1e12, neginf=-1e12)

            # Plumb the cross-rank reducers when DDP is active so every rank
            # picks the same Armijo trial step and ω stays in sync.
            reduce_grad_fn = (
                dist_helpers.all_reduce_grad_list if self.dist.is_distributed else None
            )
            reduce_scalar_fn = (
                dist_helpers.all_reduce_scalar if self.dist.is_distributed else None
            )

            last_f_val = 0.0
            with self.profile_time("inner_loop_wall_s"):
                for _ in range(int(cfg.omega_steps_per_batch)):
                    if self.npf_inner_optimizer == "bb_armijo":
                        _, bb_state, last_f_val, _ = bb_armijo_step_params(
                            self.psi_omega.parameters(),
                            omega_objective,
                            bb_state,
                            reduce_grad_fn=reduce_grad_fn,
                            reduce_scalar_fn=reduce_scalar_fn,
                            profile=self if self.is_inner_profile_active else None,
                        )
                    else:
                        _, self.muon_state, last_f_val, _ = muon_step_params(
                            self.psi_omega.parameters(),
                            omega_objective,
                            self.muon_state,
                            reduce_grad_fn=reduce_grad_fn,
                            reduce_scalar_fn=reduce_scalar_fn,
                            profile=self if self.is_inner_profile_active else None,
                        )
        if self.npf_inner_optimizer == "bb_armijo" and not reset_bb_each_batch:
            self.bb_state = bb_state
        self._last_inner_loss = last_f_val

        # Switch contract for the outer step. The base trainer will set
        # the classifier back to train()/requires_grad=True before
        # backprop; we just need to detach the adversarial features.
        self.psi_omega.eval()
        set_requires_grad(self.psi_omega, False)
        with torch.no_grad():
            with self.profile_time("final_transport_s"):
                x_adv = self._transport(x, create_graph=False)
                x_adv = self._keep_clean_for_unattacked(x_adv, x, attack_mask)
        return x_adv.detach()

    def transport_for_eval(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.psi_omega.training
        self.psi_omega.eval()
        try:
            x_adv = self._transport(x, create_graph=False)
        finally:
            self.psi_omega.train(was_training)
        return x_adv.detach()

    def freeze_adversary_parameters(self) -> None:
        self.psi_omega.eval()
        set_requires_grad(self.psi_omega, False)

    def adversary_state_dicts(self) -> Dict[str, Any]:
        return {"psi_omega": self.psi_omega.state_dict()}

    def load_adversary_state_dicts(
        self,
        payload: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if "psi_omega" not in payload:
            raise KeyError("Checkpoint payload is missing required 'psi_omega' state.")
        self.psi_omega.load_state_dict(payload["psi_omega"], strict=strict)


class NPFLastQuadTrainer(NPFTrainer):
    """NPF variant with only the final trainable rank-0 diagonal quadratic."""

    name = "npf_lastquad"

    def _potential_kwargs(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": cfg.npf_lastquad_hidden,
            "outer_rank": 0,
            "inner_rank": 0,
            "quadratic_mode": "last_layer_diagonal",
            "trainable_outer_quadratic": False,
            "activation": cfg.npf_lastquad_activation,
            "elu_alpha": cfg.npf_lastquad_elu_alpha,
            "softplus_beta": cfg.npf_lastquad_softplus_beta,
            "init_eps": cfg.npf_lastquad_init_eps,
            "strong_convexity": cfg.npf_lastquad_strong_convexity,
        }
