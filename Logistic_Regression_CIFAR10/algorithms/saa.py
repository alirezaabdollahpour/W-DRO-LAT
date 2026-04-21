"""SAA / ERM baseline: clean logistic regression training."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from algorithms.base import BaseLinearDRO
from models.classifier import LogisticRegression


def train_saa(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    run_id: int = 0,
) -> List[float]:
    """Clean SGD training loop for the SAA / ERM baseline."""
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history: List[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Run {run_id+1} SAA Epoch {epoch+1}/{epochs}", leave=False
        )
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        torch.save(
            model.state_dict(),
            f"{checkpoint_dir}/SAA_run{run_id}_epoch_{epoch+1}.pth",
        )
    return loss_history


class SAA(BaseLinearDRO):
    """Thin wrapper around a LogisticRegression trained with Adam + CE."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        max_itr: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_workers: int = 2,
        device: str = "cpu",
    ):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = LogisticRegression(input_dim, num_classes).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        run_id: int = 0,
    ) -> List[float]:
        X, y = self._validate_inputs(X, y)
        train_ds = TensorDataset(self._to_tensor(X), torch.as_tensor(y, dtype=torch.long))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_saa(
            self.model,
            train_loader,
            device=self.device,
            epochs=self.max_itr,
            learning_rate=self.learning_rate,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
        )
