"""NN-DRO competitor: vanilla-MLP inner maximiser (no gradient-of-potential).

Same outer loop as ICNN-DRO / NPF, but the adversary is parametrised directly
as a plain MLP applied to the feature x — no gradient-of-potential transport,
hence no autograd through the network's input.

Paper notation:
  * Classifier parameters: ``B``
  * Features: ``x``
  * Transport map: ``T_ω(x) = x + h_ω(x)`` with ``h_ω`` a vanilla MLP.

Training scheme:
  * Outer loop: SGD on B (logistic regression) over adversarial features.
  * Inner loop: Adam ascent on ω optimising

      max_ω  E[ CE(B(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ]
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model
from models.nn_dro import MLPAdversary


class NNDRO(BaseLinearDRO):
    """NN-DRO: vanilla MLP transport-map adversary with Adam inner optimiser."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 10.0,
        nn_dro_hidden: Sequence[int] = (512, 512, 256),
        nn_dro_activation: str = "relu",
        nn_dro_softplus_beta: float = 20.0,
        nn_dro_init_scale: float = 1e-3,
        omega_steps_per_batch: int = 10,
        inner_lr: float = 1e-2,
        lr_B: float = 5e-4,
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
        self.omega_steps_per_batch = int(omega_steps_per_batch)
        self.inner_lr = float(inner_lr)
        self.lr_B = float(lr_B)
        self.weight_decay_B = float(weight_decay_B)
        self.max_itr = int(max_itr)
        self.batch_size = int(batch_size)

        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

        self.adversary = MLPAdversary(
            input_dim=self.input_dim,
            hidden_sizes=tuple(nn_dro_hidden),
            activation=nn_dro_activation,
            softplus_beta=nn_dro_softplus_beta,
            init_scale=nn_dro_init_scale,
        ).to(self.device)

        self.inner_opt = optim.Adam(self.adversary.parameters(), lr=self.inner_lr)

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
                desc=f"Run {run_id+1} Training NN-DRO Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_batch = x_original_batch.to(self.device)
                y_batch = y_original_batch.to(self.device)

                # Inner loop: Adam ascent on ω
                self.model.eval()
                self.adversary.train()

                for _ in range(self.omega_steps_per_batch):
                    self.inner_opt.zero_grad()
                    x_adv = self.adversary(x_batch)
                    logits = self.model(x_adv)
                    ce = nn.CrossEntropyLoss(reduction="none")(logits, y_batch)
                    transport_cost = torch.sum(
                        (x_adv - x_batch) ** 2, dim=1
                    )
                    obj = (ce - self.lambda_param * transport_cost).mean()
                    (-obj).backward()
                    self.inner_opt.step()

                # Outer loop: B-update (SGD) on adversarial features
                self.model.train()
                self.adversary.eval()

                with torch.no_grad():
                    x_adv_det = self.adversary(x_batch).detach()

                logits_adv = self.model(x_adv_det)
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
                    "adversary_state_dict": self.adversary.state_dict(),
                },
                f"{checkpoint_dir}/NN_DRO_run{run_id}_epoch_{epoch+1}.pth",
            )

        return loss_history
