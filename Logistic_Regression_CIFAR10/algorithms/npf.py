"""NPF-style WDRO adversarial training (Vesseron & Cuturi, 2024).

Adapted to the CIFAR-10 feature-space logistic-regression setting:
  * Outer loop: SGD on logistic-regression classifier B
  * Inner loop: Adam on a flat parameter vector ω optimising

        max_ω  E[ CE(B(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]

where T_ω(x) = ∇ψ_ω(x) is the NPF ICNN transport map (see models/npf.py).

Core algorithm is identical to ``MNIST_Cuturi/algorithms/npf.py``; only the
outer driver is rewritten so NPF fits the ``BaseLinearDRO.fit`` interface
used by every other CIFAR-10 DRO baseline.
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
from models.npf import NPFInputConvexPotential, npf_transport_map
from utils.flatten import flatten_params, unflatten_vector


class NPF(BaseLinearDRO):
    """NPF-style ICNN transport-map adversary with Adam inner optimiser."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 10.0,
        npf_hidden: Sequence[int] = (512, 512, 256, 256, 128),
        npf_outer_rank: int = 4,
        npf_inner_rank: int = 1,
        npf_activation: str = "elu",
        npf_elu_alpha: float = 1.0,
        npf_softplus_beta: float = 20.0,
        npf_init_eps: float = 1e-3,
        inner_steps_npf: int = 20,
        inner_lr_npf: float = 1e-2,
        lr_B: float = 5e-3,
        weight_decay_B: float = 0.0,
        max_itr: int = 10,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)

        self.lambda_param = float(lambda_param)
        self.inner_steps_npf = int(inner_steps_npf)
        self.inner_lr_npf = float(inner_lr_npf)
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
        ).to(self.device)
        self.psi_omega.init_as_identity()

        params_vec, meta = flatten_params(self.psi_omega)
        self.params_vec = params_vec.to(self.device)
        self.meta = meta
        self.inner_param = self.params_vec.detach().clone().requires_grad_(True)
        self.inner_opt = optim.Adam([self.inner_param], lr=self.inner_lr_npf)

    def _adv_obj_params(
        self,
        vec: torch.Tensor,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        create_graph: bool,
    ) -> torch.Tensor:
        params_dict = unflatten_vector(vec, self.meta)
        adv = npf_transport_map(
            self.psi_omega, params_dict, x_batch, create_graph=create_graph
        )
        logits = self.model(adv)
        ce = nn.CrossEntropyLoss(reduction="none")(logits, y_batch)
        w2 = ((adv - x_batch) ** 2).sum(dim=1)
        return (ce - self.lambda_param * w2).mean()

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

                # Inner loop: Adam ascent on ω (flat parameter vector)
                self.model.eval()
                self.psi_omega.train()

                with torch.no_grad():
                    self.inner_param.data.copy_(self.params_vec.to(self.device))

                for _ in range(self.inner_steps_npf):
                    self.inner_opt.zero_grad()
                    obj = self._adv_obj_params(
                        self.inner_param, x_batch, y_batch, create_graph=True
                    )
                    (-obj).backward()
                    self.inner_opt.step()

                self.params_vec = self.inner_param.detach().clone()

                # Outer loop: B-update (SGD) on adversarial features
                self.model.train()
                self.psi_omega.eval()

                params_dict_final = unflatten_vector(
                    self.params_vec.to(self.device), self.meta
                )
                adv_feats = npf_transport_map(
                    self.psi_omega, params_dict_final, x_batch, create_graph=False
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
                    "params_vec": self.params_vec.detach().cpu(),
                },
                f"{checkpoint_dir}/NPF_run{run_id}_epoch_{epoch+1}.pth",
            )

        return loss_history
