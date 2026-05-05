"""Madry-style l2-PGD robust optimisation baseline for CIFAR-10 features."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class MadryRO(BaseLinearDRO):
    """Projected-gradient robust optimisation over an l2 feature-space ball."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        epsilon: float = 1.0,
        pgd_steps: int = 10,
        pgd_step_size: Optional[float] = None,
        pgd_restarts: int = 1,
        max_itr: int = 10,
        learning_rate: float = 5e-3,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)

        self.epsilon = float(epsilon)
        self.pgd_steps = int(pgd_steps)
        self.pgd_step_size = (
            float(pgd_step_size)
            if pgd_step_size is not None
            else self.epsilon / max(1, self.pgd_steps // 2)
        )
        self.pgd_restarts = int(pgd_restarts)
        self.max_itr = int(max_itr)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _pgd_l2_attack(self, x_orig: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.pgd_steps <= 0:
            return x_orig.detach()

        criterion = nn.CrossEntropyLoss(reduction="none")
        best_adv = x_orig.detach()
        best_loss = torch.full((x_orig.shape[0],), -torch.inf, device=x_orig.device)
        restarts = max(1, self.pgd_restarts)

        for restart_idx in range(restarts):
            if restart_idx == 0:
                x_adv = x_orig.detach().clone()
            else:
                noise = torch.randn_like(x_orig)
                noise_norm = torch.linalg.norm(noise, dim=1, keepdim=True).clamp_min(
                    1e-12
                )
                radii = torch.rand(
                    x_orig.shape[0], 1, device=x_orig.device, dtype=x_orig.dtype
                )
                x_adv = x_orig.detach() + self.epsilon * radii * noise / noise_norm

            for _ in range(self.pgd_steps):
                x_adv = x_adv.detach().requires_grad_(True)
                loss = criterion(self.model(x_adv), y).sum()
                grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
                grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True).clamp_min(
                    1e-12
                )

                with torch.no_grad():
                    x_adv = x_adv + self.pgd_step_size * grad / grad_norm
                    delta = x_adv - x_orig
                    delta_norm = torch.linalg.norm(
                        delta, dim=1, keepdim=True
                    ).clamp_min(1e-12)
                    scale = torch.minimum(
                        torch.ones_like(delta_norm), self.epsilon / delta_norm
                    )
                    x_adv = x_orig + delta * scale

            with torch.no_grad():
                final_loss = criterion(self.model(x_adv), y)
                improve = final_loss > best_loss
                best_loss = torch.where(improve, final_loss, best_loss)
                best_adv = torch.where(improve.unsqueeze(1), x_adv, best_adv)

        return best_adv.detach()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint_dir: str = "checkpoints",
        run_id: int = 0,
    ) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_history: List[float] = []

        for epoch in range(self.max_itr):
            epoch_loss = 0.0
            pbar = tqdm(
                dataloader,
                desc=f"Run {run_id+1} Training RO Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_batch, y_batch in pbar:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.model.eval()
                x_adv = self._pgd_l2_attack(x_batch, y_batch)

                self.model.train()
                logits = self.model(x_adv)
                loss = nn.CrossEntropyLoss()(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=float(loss.item()))

            avg_epoch_loss = epoch_loss / max(1, len(dataloader))
            loss_history.append(avg_epoch_loss)
            torch.save(
                self.model.state_dict(),
                f"{checkpoint_dir}/RO_run{run_id}_epoch_{epoch+1}.pth",
            )

        return loss_history
