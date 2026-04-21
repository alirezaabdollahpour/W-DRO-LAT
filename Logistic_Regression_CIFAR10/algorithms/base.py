"""Shared base class for the CIFAR-10 feature-space DRO algorithms."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.classifier import LinearModel


class DROError(Exception):
    """Generic error from a DRO training routine."""


class BaseLinearDRO:
    """Shared scaffolding: input validation, dataloader, score, loss utilities."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool,
        sample_level: int = 6,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fit_intercept = fit_intercept
        self.model: nn.Module
        self.sample_level = sample_level

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=torch.float32)

    def _validate_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X.reshape(-1, self.input_dim)
        if y.ndim > 1:
            y = y.flatten()
        if X.shape[0] != y.shape[0]:
            raise DROError(
                f"Shapes mismatch: X {X.shape[0]}, y {y.shape[0]}"
            )
        if X.shape[1] != self.input_dim:
            raise DROError(
                f"Input dim mismatch: expected {self.input_dim}, got {X.shape[1]}"
            )
        return X, y

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int
    ) -> DataLoader:
        dataset = TensorDataset(
            self._to_tensor(X), torch.as_tensor(y, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_val, _ = self._validate_inputs(X, np.zeros(X.shape[0]))
        self.model.eval()
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            inputs = self._to_tensor(X_val).to(model_device)
            return self.model(inputs).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy on (X, y)."""
        X_val, y_val = self._validate_inputs(X, y)
        y_true = y_val.flatten()
        y_pred = np.argmax(self.predict(X_val), axis=1)
        return float(np.mean(y_true == y_pred))

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        m: int,
        lambda_reg: float,
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(reduction="none")
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(
            torch.logsumexp(residual_matrix, dim=0) - math.log(m)
        ) * lambda_reg

    def _compute_dual_loss(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        lam: float,
        epsilon: float,
    ) -> torch.Tensor:
        m = 2 ** 6
        expanded_data = data.repeat_interleave(m, dim=0)
        noise = torch.randn_like(expanded_data) * math.sqrt(epsilon)
        noisy_data = expanded_data + noise
        repeated_target = targets.repeat_interleave(m, dim=0)

        predictions = self.model(noisy_data)
        criterion = nn.CrossEntropyLoss(reduction="none")
        residuals = criterion(predictions, repeated_target) / max(lam * epsilon, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(
            torch.logsumexp(residual_matrix, dim=0) - math.log(m)
        ) * lam * epsilon


def make_linear_model(
    input_dim: int, num_classes: int, fit_intercept: bool, device: torch.device
) -> LinearModel:
    return LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(device)
