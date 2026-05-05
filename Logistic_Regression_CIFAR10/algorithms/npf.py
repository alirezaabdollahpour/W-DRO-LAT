"""NPF-style WDRO adversarial training (Vesseron & Cuturi, 2024).

Adapted to the CIFAR-10 feature-space logistic-regression setting:
  * Outer loop: SGD on logistic-regression classifier B
  * Inner loop: BB+Armijo on the NPF ICNN parameters ω optimising

        max_ω  E[ CE(B(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]

where T_ω(x) = ∇ψ_ω(x) is the NPF ICNN transport map (see models/npf.py).
Unlike the bounded RL/least-squares settings, CIFAR feature vectors are not
box-constrained here, so the transport is applied directly in feature space.

Initialization follows the paper exactly and is *not* a choice: principled
LogNormal draws (used inside the non-negative layers' constructor) and the
identity init that zeros the input-to-hidden, output linear, and outer/
per-layer quadratic terms are always applied JOINTLY so that
T_ω(x) ≈ x at t=0 while the non-negative cascade and readout retain
their LogNormal draws.

The inner BB+Armijo step uses the EXACT SAME hyperparameters as the
ICNN-DRO algorithm (cfg.icnn_bb_*) so the only modeling difference
between NPF and ICNN is the potential's parametrisation.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model
from models.npf import NPFInputConvexPotential, npf_T_omega
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params
from utils.common import set_requires_grad


class NPF(BaseLinearDRO):
    """NPF-style ICNN transport-map adversary with BB+Armijo inner optimiser."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 10.0,
        npf_hidden: Sequence[int] = (512, 512, 256, 128, 64),
        npf_outer_rank: int = 8,
        npf_inner_rank: int = 2,
        npf_activation: str = "softplus",
        npf_elu_alpha: float = 1.0,
        npf_softplus_beta: float = 10.0,
        npf_init_eps: float = 1e-4,
        npf_strong_convexity: float = 1.0,
        omega_steps_per_batch: int = 20,
        bb_alpha0: float = 2e-4,
        bb_alpha_min: float = 1e-7,
        bb_alpha_max: float = 0.25,
        bb_ls_c: float = 1e-4,
        bb_ls_shrink: float = 0.5,
        bb_ls_max_steps: int = 15,
        lr_B: float = 1e-3,
        weight_decay_B: float = 1e-4,
        max_itr: int = 10,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)

        self.lambda_param = float(lambda_param)
        self.omega_steps_per_batch = int(omega_steps_per_batch)
        self.lr_B = float(lr_B)
        self.weight_decay_B = float(weight_decay_B)
        self.max_itr = int(max_itr)
        self.batch_size = int(batch_size)

        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

        self.psi_omega = NPFInputConvexPotential(
            input_dim=self.input_dim,
            hidden_sizes=npf_hidden,
            outer_rank=npf_outer_rank,
            inner_rank=npf_inner_rank,
            activation=npf_activation,
            elu_alpha=npf_elu_alpha,
            softplus_beta=npf_softplus_beta,
            init_eps=npf_init_eps,
            strong_convexity=npf_strong_convexity,
        ).to(self.device)
        # Identity init applied on top of the LogNormal draws produced inside
        # the non-negative layers' constructor.
        self.psi_omega.init_as_identity()

        self.bb_state = BBArmijoState.create(
            alpha0=bb_alpha0,
            alpha_min=bb_alpha_min,
            alpha_max=bb_alpha_max,
            ls_c=bb_ls_c,
            ls_shrink=bb_ls_shrink,
            ls_max_steps=bb_ls_max_steps,
            reject_on_armijo_failure=True,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        run_id: int = 0,
    ) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)

        optimizer_B = optim.SGD(
            self.model.parameters(),
            lr=self.lr_B,
            momentum=0.9,
            weight_decay=self.weight_decay_B,
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_history: List[float] = []

        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Run {run_id+1} Training NPF Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_batch = x_original_batch.to(self.device)
                y_batch = y_original_batch.to(self.device)

                # Inner loop: ω-ascent via BB+Armijo. Explicitly freeze the
                # classifier — bb_armijo_step_params calls
                # ``torch.autograd.grad(..., params=psi.parameters())`` so
                # the classifier's .grad is not accumulated today, but the
                # explicit freeze makes the contract robust to future
                # refactors that switch to ``f.backward()``. Mirrors the
                # convention in MNIST_Cuturi.algorithms.npf.train_step_npf.
                self.model.eval()
                self.psi_omega.train()
                set_requires_grad(self.model, False)
                set_requires_grad(self.psi_omega, True)

                def omega_objective(create_graph: bool) -> torch.Tensor:
                    x_adv = npf_T_omega(
                        x_batch, self.psi_omega, create_graph=create_graph
                    )
                    logits = self.model(x_adv)
                    ce = nn.CrossEntropyLoss(reduction="none")(logits, y_batch)
                    transport_cost = torch.sum((x_adv - x_batch) ** 2, dim=1)
                    obj = (ce - self.lambda_param * transport_cost).mean()
                    return torch.nan_to_num(
                        obj, nan=-1e12, posinf=-1e12, neginf=-1e12
                    )

                for _ in range(self.omega_steps_per_batch):
                    _, self.bb_state, _, _ = bb_armijo_step_params(
                        self.psi_omega.parameters(),
                        omega_objective,
                        self.bb_state,
                    )

                # Outer loop: B-update (SGD) on adversarial features. Restore
                # train/eval modes and gradient flags on both modules.
                self.model.train()
                self.psi_omega.eval()
                set_requires_grad(self.model, True)
                set_requires_grad(self.psi_omega, False)

                with torch.no_grad():
                    adv_feats = npf_T_omega(
                        x_batch, self.psi_omega, create_graph=False
                    ).detach()

                logits_adv = self.model(adv_feats)
                loss_B = nn.CrossEntropyLoss()(logits_adv, y_batch)

                optimizer_B.zero_grad()
                loss_B.backward()
                optimizer_B.step()

                epoch_loss_record += loss_B.item()
                pbar.set_postfix(loss=float(loss_B.item()))

            avg_epoch_loss = epoch_loss_record / max(1, len(dataloader))
            loss_history.append(avg_epoch_loss)

            torch.save(
                {
                    "B_state_dict": self.model.state_dict(),
                    "psi_omega_state_dict": self.psi_omega.state_dict(),
                },
                f"{checkpoint_dir}/NPF_run{run_id}_epoch_{epoch+1}.pth",
            )

        # Diagnostic: ``outer_a`` is the linear term of the convex potential
        # and is initialized to zero. If it remains ~0 across training,
        # nothing in the inner objective is applying pressure on it and the
        # parameter is dead weight. Print the final norm so a future
        # reviewer can spot this without re-loading the checkpoint.
        try:
            outer_a_norm = float(self.psi_omega.outer_a.detach().norm().item())
            print(f"[NPF] final ||outer_a||_2 = {outer_a_norm:.4e}")
        except AttributeError:
            pass

        return loss_history
