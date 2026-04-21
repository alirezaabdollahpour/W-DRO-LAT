"""Sinkhorn SDRO dual / entropy-regularised WDRO baseline."""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class SDRO_Dual(BaseLinearDRO):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        epsilon: float = 1e-3,
        lambda_param: float = 1e2,
        max_itr: int = 100,
        learning_rate: float = 1e-2,
        sample_level: int = 6,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.sample_level = sample_level
        self.batch_size = batch_size
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = make_linear_model(
            input_dim, num_classes, fit_intercept, self.device
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history: List[float] = []
        self.model.train()
        for epoch in range(self.max_itr):
            epoch_loss = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Run {run_id+1} Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                levels = np.arange(self.sample_level + 1)
                numerators = 2.0 ** (-levels)
                denominator = 2.0 - 2.0 ** (-self.sample_level)

                probabilities = numerators / denominator
                sampled_level = np.random.choice(levels, p=probabilities)
                m = 2 ** sampled_level

                expanded_data = data.repeat_interleave(m, dim=0)
                noise = torch.randn_like(expanded_data) * math.sqrt(self.epsilon)
                noisy_data = expanded_data + noise
                repeated_target = target.repeat_interleave(m, dim=0)

                predictions = self.model(noisy_data)
                loss = self._compute_loss(predictions, repeated_target, m, lambda_reg)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_epoch_loss)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/SDRO_Dual_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
