"""WRM (Sinha et al.) baseline."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class WRM(BaseLinearDRO):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 1.0,
        max_itr: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        inner_lr: float = 0.01,
        inner_itr: int = 200,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr = inner_lr
        self.inner_itr = inner_itr
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _sinha_sampler(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        epoch: int,
    ) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        num_steps = int(self.inner_itr)
        for _ in range(num_steps):
            loss_values = nn.CrossEntropyLoss(reduction="none")(
                model(x_clone), y_orig
            )
            grads, = torch.autograd.grad(
                loss_values, x_clone, grad_outputs=torch.ones_like(loss_values)
            )
            x_clone = x_clone.detach() + self.inner_lr * (
                grads - 2 * self.lambda_param * (x_clone - x_orig)
            )
            x_clone.requires_grad_(True)
        return x_clone.detach()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        run_id: int = 0,
    ) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_history: List[float] = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Run {run_id+1} Training WRM Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                self.model.eval()
                x_WRM_batch = self._sinha_sampler(
                    x_original_batch_dev, y_original_batch_dev, self.model, epoch
                )
                self.model.train()
                predictions_logits_batch = self.model(x_WRM_batch)
                optimization_loss = nn.CrossEntropyLoss()(
                    predictions_logits_batch, y_original_batch_dev
                )
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = nn.CrossEntropyLoss()(
                        self.model(x_original_batch_dev), y_original_batch_dev
                    )
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(
                    optim_loss=optimization_loss.item(),
                    record_loss=comparable_loss.item(),
                )
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/WRM_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
