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

from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

from .. import distributed as dist_helpers
from ..models.npf import NPFInputConvexPotential, npf_T_omega
from ..utils import (
    BBArmijoState,
    MuonState,
    adversary_loss_per_sample,
    bb_armijo_step_params,
    clamped_normalized_copy,
    frozen_module,
    input_pgd_best_adversary,
    muon_step_params,
    normalized_mse,
    project_normalized_to_input_ball,
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
        potential_kwargs = self._potential_kwargs()
        self.psi_omega = NPFInputConvexPotential(**potential_kwargs).to(self.device)
        # Optional identity init applied on top of the LogNormal draws produced
        # inside the non-negative layers' constructor.
        if self._identity_init_enabled():
            self.psi_omega.init_as_identity(
                residual_eps=float(potential_kwargs.get("init_eps", 0.0))
            )
            self._warn_if_identity_dead_start(potential_kwargs)
        self.reset_omega_each_batch = bool(
            getattr(cfg, "npf_reset_omega_each_batch", False)
        )
        self._initial_omega_state = None
        if self.reset_omega_each_batch:
            self._initial_omega_state = {
                key: value.detach().clone()
                for key, value in self.psi_omega.state_dict().items()
            }

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
            self.muon_state = self._new_muon_state()

    def _identity_init_enabled(self) -> bool:
        return bool(getattr(self.config, "npf_identity_init", True))

    def _warn_if_identity_dead_start(self, potential_kwargs: Dict[str, Any]) -> None:
        if not self.is_main_rank:
            return
        activation = str(potential_kwargs.get("activation", "")).lower()
        rectifier = str(potential_kwargs.get("positive_weight_rectifier", "")).lower()
        init_eps = float(potential_kwargs.get("init_eps", 0.0) or 0.0)
        if init_eps <= 0.0:
            print(
                "[npf warning] exact identity init with init_eps=0 leaves the "
                "residual ICNN at zero; some residual paths can be dead or too "
                "stiff for BB+Armijo. For training, use a small positive "
                "NPF init_eps such as 1e-3. Keep init_eps=0 only for identity "
                "diagnostics.",
                flush=True,
            )
        elif activation == "relu" and rectifier == "relu":
            print(
                "[npf warning] ReLU activation with ReLU positive-weight "
                "projection is a valid raw-OTT ablation, but it can be brittle "
                "near zero. Smooth activation/rectifier settings usually give "
                "a more reliable identity-adjacent warm start.",
                flush=True,
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

    def _new_muon_state(self) -> MuonState:
        cfg = self.config
        return MuonState.create(
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

    def _reset_omega_to_initial(self) -> None:
        if self._initial_omega_state is None:
            return
        self.psi_omega.load_state_dict(self._initial_omega_state, strict=True)

    def _potential_kwargs(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": cfg.npf_hidden,
            "outer_rank": cfg.npf_outer_rank,
            "inner_rank": cfg.npf_inner_rank,
            "output_rank": None,
            "quadratic_mode": "all_layers",
            "trainable_outer_quadratic": True,
            "activation": cfg.npf_activation,
            "elu_alpha": cfg.npf_elu_alpha,
            "softplus_beta": cfg.npf_softplus_beta,
            "init_eps": cfg.npf_init_eps,
            "strong_convexity": cfg.npf_strong_convexity,
            "pos_weights": cfg.npf_pos_weights,
            "positive_weight_rectifier": cfg.npf_positive_weight_rectifier,
        }

    def _omega_free_decay_parameters(self) -> list[nn.Parameter]:
        """FREE (unconstrained) omega kernels eligible for ascent L2 decay.

        Legacy pretrained_INPUT_icnn applied wd=1e-4 to every parameter that
        was not an exp-reparametrized nonnegative weight and not a bias —
        i.e. exactly the unconstrained input-side kernels whose growth
        produces sample-independent far-field transports (affine input-skip
        weights, PosDef lin/quad kernels). Diagonals, biases, the positive
        w_z ladder, and the frozen outer identity potential are excluded.
        """
        selected: list[nn.Parameter] = []
        for name, param in self.psi_omega.named_parameters():
            if not name.startswith("w_xs."):
                continue
            if name.endswith(".bias") or name.endswith("diag_kernel"):
                continue
            if (
                name.endswith(".weight")
                or name.endswith("lin_kernel")
                or name.endswith("quad_kernel")
            ):
                selected.append(param)
        return selected

    def _omega_weight_decay_penalty(self) -> Optional[torch.Tensor]:
        wd = float(getattr(self.config, "npf_omega_weight_decay", 0.0) or 0.0)
        if wd <= 0.0:
            return None
        params = self._omega_free_decay_parameters()
        if not params:
            return None
        total = params[0].pow(2).sum()
        for param in params[1:]:
            total = total + param.pow(2).sum()
        return 0.5 * wd * total

    # ------------------------------------------------------------------
    def _transport(self, x: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        """T_ω(x) = clamp(∇ψ_ω(x))."""
        cfg = self.config
        x_adv = npf_T_omega(x, self.psi_omega, create_graph=create_graph)
        if bool(getattr(cfg, "npf_project_to_input_ball", False)):
            p = 2 if str(getattr(cfg, "inp_p", "2")).lower() == "2" else float("inf")
            return project_normalized_to_input_ball(
                x_adv,
                x,
                eps=float(getattr(cfg, "inp_eps", 0.0)),
                p=p,
                geometry=str(getattr(cfg, "input_pgd_geometry", "pixel_l2_squared")),
            )
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
        if self.reset_omega_each_batch:
            self._reset_omega_to_initial()
            if self.npf_inner_optimizer == "muon":
                self.muon_state = self._new_muon_state()
        reset_bb_each_batch = bool(getattr(cfg, "reset_parametric_bb_each_batch", True))
        if self.reset_omega_each_batch:
            reset_bb_each_batch = True
        bb_state = self._new_bb_state() if reset_bb_each_batch else self.bb_state

        # Inner loop: ω-ascent. Freeze the classifier so its gradients are not
        # accumulated; the inner optimizers route gradients only to ψ_ω
        # parameters, but the explicit freeze keeps the contract robust to
        # future refactors.
        self.psi_omega.train()
        set_requires_grad(self.psi_omega, True)
        self.psi_omega.enforce_trainable_constraints()
        self._last_npf_pgd_anchor_mse = None

        with frozen_module(
            self._classifier_module,
            eval_mode=self._adversary_classifier_eval_mode(),
        ):
            pgd_anchor = None
            pgd_anchor_weight = float(getattr(cfg, "npf_pgd_anchor_weight", 0.0) or 0.0)
            if pgd_anchor_weight > 0.0:
                p_input = (
                    2
                    if str(getattr(cfg, "inp_p", "2")).lower() == "2"
                    else float("inf")
                )
                anchor_steps = max(
                    0,
                    int(getattr(cfg, "npf_pgd_anchor_steps", 5) or 0),
                )
                anchor_restarts = max(
                    1,
                    int(getattr(cfg, "npf_pgd_anchor_restarts", 1) or 1),
                )
                with self.profile_time("pgd_anchor_s"):
                    pgd_anchor, _ = input_pgd_best_adversary(
                        self._classifier_module,
                        x.detach(),
                        y,
                        p=p_input,
                        eps=float(getattr(cfg, "inp_eps", 0.0)),
                        steps=anchor_steps,
                        step_size=float(getattr(cfg, "inp_step_size", 0.0) or 0.0),
                        restarts=anchor_restarts,
                        loss=str(
                            getattr(
                                cfg,
                                "npf_pgd_anchor_loss",
                                getattr(cfg, "input_pgd_loss", "margin"),
                            )
                        ),
                        geometry=str(
                            getattr(cfg, "input_pgd_geometry", "pixel_l2_squared")
                        ),
                    )
                pgd_anchor = pgd_anchor.detach()

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
                    if pgd_anchor is not None and pgd_anchor_weight > 0.0:
                        anchor_cost = normalized_mse(x_adv, pgd_anchor)
                        objective_per_sample = (
                            objective_per_sample - pgd_anchor_weight * anchor_cost
                        )
                    obj = self._shared_adversary_masked_mean(
                        objective_per_sample,
                        attack_mask,
                    )
                    decay = self._omega_weight_decay_penalty()
                    if decay is not None:
                        # Identical on every DDP rank, so the mean-reduced
                        # gradient keeps exactly this decay strength.
                        obj = obj - decay
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
                            max_grad_norm=getattr(cfg, "parametric_bb_max_grad_norm", 1.0),
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
                if pgd_anchor is not None and pgd_anchor_weight > 0.0:
                    anchor_mse = normalized_mse(x_adv, pgd_anchor)
                    self._last_npf_pgd_anchor_mse = float(
                        self._masked_mean(anchor_mse, attack_mask).item()
                    )
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

    def adversary_delta_parameters(self) -> Iterator[nn.Parameter]:
        # No requires_grad filter: step() leaves psi_omega with grads disabled
        # at epoch end, which would make this empty and silently zero the
        # omega_l2_delta "is the adversary moving" diagnostic.
        return iter(self.psi_omega.parameters())

    def load_adversary_state_dicts(
        self,
        payload: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if "psi_omega" not in payload:
            raise KeyError("Checkpoint payload is missing required 'psi_omega' state.")
        self.psi_omega.load_state_dict(payload["psi_omega"], strict=strict)
        if self._initial_omega_state is not None:
            # Under per-batch omega resets, keep resuming runs on the
            # checkpointed adversary instead of silently discarding it in
            # favor of the constructor-time snapshot on the first batch.
            self._initial_omega_state = {
                key: value.detach().clone()
                for key, value in self.psi_omega.state_dict().items()
            }


class NPFLastQuadTrainer(NPFTrainer):
    """NPF variant with only the final trainable rank-0 diagonal quadratic."""

    name = "npf_lastquad"

    def _identity_init_enabled(self) -> bool:
        return bool(getattr(self.config, "npf_lastquad_identity_init", True))

    def _potential_kwargs(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": cfg.npf_lastquad_hidden,
            "outer_rank": 0,
            "inner_rank": 0,
            "output_rank": cfg.npf_lastquad_output_rank,
            "quadratic_mode": "last_layer_diagonal",
            "trainable_outer_quadratic": False,
            "activation": cfg.npf_lastquad_activation,
            "elu_alpha": cfg.npf_lastquad_elu_alpha,
            "softplus_beta": cfg.npf_lastquad_softplus_beta,
            "init_eps": cfg.npf_lastquad_init_eps,
            "strong_convexity": cfg.npf_lastquad_strong_convexity,
            "pos_weights": cfg.npf_lastquad_pos_weights,
            "positive_weight_rectifier": cfg.npf_lastquad_positive_weight_rectifier,
        }
