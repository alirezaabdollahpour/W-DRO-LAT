"""Wasserstein gradient flow sampler (WGF) WDRO baseline."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class SDRO_WGF(BaseLinearDRO):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        epsilon: float = 0.1,
        lambda_param: float = 1.0,
        num_samples: int = 10,
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
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.inner_lr = inner_lr
        self.inner_itr = inner_itr
        self.num_samples = num_samples
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _WGF_sampler(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        epoch: int,
    ) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        x_clone = (
            x_clone.unsqueeze(1).expand(-1, self.num_samples, -1)
            .contiguous().view(-1, self.input_dim)
        )
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = (
            x_orig.unsqueeze(1).expand(-1, self.num_samples, -1)
            .reshape(-1, self.input_dim)
        )
        num_steps = int(self.inner_itr)
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            loss_values = nn.CrossEntropyLoss(reduction="none")(
                model(x_clone), y_repeated
            )
            grads, = torch.autograd.grad(
                loss_values, x_clone, grad_outputs=torch.ones_like(loss_values)
            )
            x_clone = x_clone.detach()

            mean = x_clone + self.inner_lr * (
                grads - 2 * self.lambda_param * (x_clone - x_original_expanded)
            )
            std_dev = torch.sqrt(
                torch.tensor(
                    2 * self.inner_lr * self.lambda_param * self.epsilon,
                    device=self.device,
                )
            )

            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

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
                desc=f"Run {run_id+1} Training WGF Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                self.model.eval()
                x_WGF_batch = self._WGF_sampler(
                    x_original_batch_dev, y_original_batch_dev, self.model, epoch
                )
                y_repeated_batch = y_original_batch_dev.repeat_interleave(
                    self.num_samples, dim=0
                )
                self.model.train()
                predictions_logits_batch = self.model(x_WGF_batch)
                optimization_loss = nn.CrossEntropyLoss()(
                    predictions_logits_batch, y_repeated_batch
                )
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/SDRO_WGF_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
