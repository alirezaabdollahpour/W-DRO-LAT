"""ICNN-DRO competitor (transport map adversary).

Paper notation:
  * Classifier parameters: ``B``
  * Features: ``x``
  * Convex potential: ``ψ_ω`` (ICNN)
  * Transport map: ``T_ω(x) = ∇_x ψ_ω(x)``

Training scheme:
  * Outer loop: SGD on B (logistic regression)
  * Inner loop: ω-ascent via BB+Armijo on

      max_ω  E[ CE(B(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model
from models.icnn import (
    InputConvexPotential,
    T_omega,
    initialize_icnn_identity,
)
from utils.bb_armijo import BBArmijoState, bb_armijo_step_params


class ICNNDRO(BaseLinearDRO):
    """ICNN-DRO competitor for adversarial training."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 10.0,
        icnn_hidden: Tuple[int, ...] = (512, 512, 512, 256, 256, 128, 128, 64),
        icnn_strong_convexity: float = 1.0,
        icnn_softplus_beta: float = 20.0,
        icnn_init_mode: str = "identity",
        lr_B: float = 5e-3,
        weight_decay_B: float = 0.0,
        omega_steps_per_batch: int = 1,
        bb_alpha0: float = 1e-1,
        bb_alpha_min: float = 1e-6,
        bb_alpha_max: float = 10.0,
        bb_ls_c: float = 1e-4,
        bb_ls_shrink: float = 0.5,
        bb_ls_max_steps: int = 10,
        max_itr: int = 10,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)

        self.lambda_param = float(lambda_param)
        self.lr_B = float(lr_B)
        self.weight_decay_B = float(weight_decay_B)
        self.omega_steps_per_batch = int(omega_steps_per_batch)
        self.max_itr = int(max_itr)
        self.batch_size = int(batch_size)

        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

        self.psi_omega = InputConvexPotential(
            input_dim=self.input_dim,
            hidden_sizes=icnn_hidden,
            strong_convexity=icnn_strong_convexity,
            softplus_beta=icnn_softplus_beta,
            nonneg_init="principled",
        ).to(self.device)

        init_mode = str(icnn_init_mode).lower()
        if init_mode == "identity":
            initialize_icnn_identity(self.psi_omega, strong_convexity=1.0)
        elif init_mode == "principled":
            pass
        else:
            raise ValueError(
                f"Unsupported icnn_init_mode '{icnn_init_mode}'. Use 'identity' or 'principled'."
            )

        self.bb_state = BBArmijoState.create(
            alpha0=bb_alpha0,
            alpha_min=bb_alpha_min,
            alpha_max=bb_alpha_max,
            ls_c=bb_ls_c,
            ls_shrink=bb_ls_shrink,
            ls_max_steps=bb_ls_max_steps,
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
                desc=f"Run {run_id+1} Training ICNN Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)

                # Inner loop: ω-ascent via BB+Armijo
                self.model.eval()
                self.psi_omega.train()

                def omega_objective(create_graph: bool) -> torch.Tensor:
                    x_adv = T_omega(
                        x_original_batch_dev, self.psi_omega, create_graph=create_graph
                    )
                    logits = self.model(x_adv)
                    ce = nn.CrossEntropyLoss(reduction="none")(
                        logits, y_original_batch_dev
                    )
                    transport_cost = torch.sum(
                        (x_adv - x_original_batch_dev) ** 2, dim=1
                    )
                    return (ce - self.lambda_param * transport_cost).mean()

                for _ in range(self.omega_steps_per_batch):
                    _, self.bb_state, f_val, grad_norm = bb_armijo_step_params(
                        self.psi_omega.parameters(),
                        omega_objective,
                        self.bb_state,
                    )

                # Outer loop: B-update (SGD) on adversarial features
                self.model.train()
                self.psi_omega.eval()

                with torch.no_grad():
                    x_adv_det = T_omega(
                        x_original_batch_dev, self.psi_omega, create_graph=False
                    )

                logits_adv = self.model(x_adv_det)
                loss_B = nn.CrossEntropyLoss()(logits_adv, y_original_batch_dev)

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
                f"{checkpoint_dir}/ICNN_DRO_run{run_id}_epoch_{epoch+1}.pth",
            )

        return loss_history
