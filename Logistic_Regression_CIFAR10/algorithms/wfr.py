"""Wasserstein-Fisher-Rao sampler (WFR) WDRO baseline."""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class SDRO_WFR(BaseLinearDRO):
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

    def _WFR_sampler(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        epoch: int,
    ):
        batch_size = x_orig.shape[0]

        weights = torch.ones(
            (batch_size, self.num_samples), dtype=torch.float32, device=self.device
        ) / self.num_samples

        x_clone = (
            x_orig.clone().detach()
            .unsqueeze(1).expand(-1, self.num_samples, -1)
            .contiguous().view(-1, self.input_dim)
        )
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = (
            x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)
        )

        wt_lr = self.inner_lr
        weight_exponent = 1 - self.lambda_param * self.epsilon * wt_lr
        low_weight_threshold = 1e-4
        std_dev = math.sqrt(2 * self.inner_lr * self.lambda_param * self.epsilon)
        criterion = nn.CrossEntropyLoss(reduction="none")

        num_steps = int(self.inner_itr)

        param_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            for _ in range(num_steps):
                x_clone.requires_grad_(True)

                loss_values = criterion(model(x_clone), y_repeated)
                grads = torch.autograd.grad(
                    loss_values.sum(), x_clone, retain_graph=False, create_graph=False
                )[0]
                x_clone = x_clone.detach()

                with torch.no_grad():
                    mean = x_clone + self.inner_lr * (
                        grads - 2 * self.lambda_param * (x_clone - x_original_expanded)
                    )
                    x_clone = mean + torch.randn_like(mean) * std_dev

                    current_loss = criterion(model(x_clone), y_repeated).view(
                        batch_size, self.num_samples
                    )
                    dist_sq = torch.sum(
                        (x_clone - x_original_expanded) ** 2, dim=1
                    ).view(batch_size, self.num_samples)
                    energy_term = current_loss - 2 * self.lambda_param * dist_sq

                    weights.pow_(weight_exponent)
                    weights.mul_(torch.exp(energy_term * wt_lr))
                    weights.div_(torch.sum(weights, dim=1, keepdim=True).add_(1e-9))

                    low_weight_mask = weights < low_weight_threshold
                    rows_with_low_weights = low_weight_mask.any(dim=1, keepdim=True)

                    x_reshaped = x_clone.view(
                        batch_size, self.num_samples, self.input_dim
                    )
                    max_weight_vals, max_weight_indices = weights.max(dim=1, keepdim=True)
                    highest_weight_point_data = x_reshaped.gather(
                        1,
                        max_weight_indices.unsqueeze(-1).expand(-1, -1, self.input_dim),
                    )

                    low_weights_sum = torch.sum(
                        weights * low_weight_mask, dim=1, keepdim=True
                    )
                    num_low_weights = low_weight_mask.sum(
                        dim=1, keepdim=True, dtype=weights.dtype
                    )
                    avg_weight = (max_weight_vals + low_weights_sum) / (
                        num_low_weights + 1.0 + 1e-9
                    )

                    max_weight_mask = torch.zeros_like(weights, dtype=torch.bool)
                    max_weight_mask.scatter_(1, max_weight_indices, True)
                    update_mask = (
                        low_weight_mask | max_weight_mask
                    ) & rows_with_low_weights
                    weights = torch.where(update_mask, avg_weight, weights)

                    x_update_mask = low_weight_mask & rows_with_low_weights
                    x_reshaped = torch.where(
                        x_update_mask.unsqueeze(-1),
                        highest_weight_point_data,
                        x_reshaped,
                    )
                    x_clone = x_reshaped.view(-1, self.input_dim)
                    weights = weights / (
                        torch.sum(weights, dim=1, keepdim=True) + 1e-9
                    )
        finally:
            for p, req in zip(model.parameters(), param_requires_grad):
                p.requires_grad_(req)

        return x_clone.detach(), weights.detach()

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
                desc=f"Run {run_id+1} Training WFR Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                self.model.eval()
                x_WFR_batch, weights = self._WFR_sampler(
                    x_original_batch_dev, y_original_batch_dev, self.model, epoch
                )
                y_repeated_batch = y_original_batch_dev.repeat_interleave(
                    self.num_samples, dim=0
                )
                self.model.train()
                predictions_logits_batch = self.model(x_WFR_batch)
                loss_values = nn.CrossEntropyLoss(reduction="none")(
                    predictions_logits_batch, y_repeated_batch
                )
                optimization_loss = (
                    loss_values.view(-1, self.num_samples) * weights
                ).sum(dim=1).mean()
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/SDRO_WFR_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
