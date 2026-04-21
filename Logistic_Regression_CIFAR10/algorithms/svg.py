"""SVGD sampler WDRO baseline."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class SDRO_SVG(BaseLinearDRO):
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
        adagrad_hist_decay: float = 0.9,
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
        self.adagrad_hist_decay = adagrad_hist_decay
        self.num_samples = num_samples
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _rbf_kernel_batched_torch(
        self, particles: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = particles.shape
        device = particles.device
        sq_dist = torch.cdist(particles, particles, p=2).pow(2)
        median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
        median_sq_dist = median_sq_dist.view(B, 1, 1)
        h_squared = median_sq_dist / (
            torch.log(torch.tensor(S, dtype=torch.float32, device=device)) + 1e-8
        )
        K = torch.exp(-sq_dist / (2 * h_squared))
        diff = particles.unsqueeze(2) - particles.unsqueeze(1)
        grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)
        return K, grad_K_x

    def _svg_sampler(
        self,
        x_original_batch: torch.Tensor,
        y_original_batch: torch.Tensor,
        model: nn.Module,
        epoch: int,
    ) -> torch.Tensor:
        if x_original_batch.shape[0] == 0:
            return torch.empty(
                0, x_original_batch.shape[1], device=x_original_batch.device
            )

        B, D = x_original_batch.shape
        S = self.num_samples

        x_orig_repeated = x_original_batch.unsqueeze(1).repeat(1, S, 1)
        y_repeated = y_original_batch.repeat_interleave(S)

        particles = x_orig_repeated.clone().detach()
        particles += torch.randn_like(particles) * 0.1
        hist_grad = torch.zeros_like(particles)

        for _ in range(int(self.inner_itr)):
            x_tensor = particles.view(B * S, D).clone().requires_grad_(True)
            x_orig_repeated_flat = x_orig_repeated.view(B * S, D)

            neg_log_likelihood = nn.CrossEntropyLoss(reduction="sum")(
                model(x_tensor), y_repeated
            )
            grad_log_py_x, = torch.autograd.grad(
                outputs=neg_log_likelihood, inputs=x_tensor
            )
            grad_log_px = -2 * self.lambda_param * (x_tensor - x_orig_repeated_flat)

            total_grad_flat = (grad_log_py_x + grad_log_px) / (
                self.lambda_param * self.epsilon
            )
            total_grad = total_grad_flat.view(B, S, D)

            K, grad_K_x = self._rbf_kernel_batched_torch(particles)
            K_grad_prod = torch.bmm(K, total_grad)
            sum_grad_K = torch.sum(grad_K_x, dim=2)

            svg_grad = (K_grad_prod + sum_grad_K) / S

            with torch.no_grad():
                hist_grad = self.adagrad_hist_decay * hist_grad + (
                    1 - self.adagrad_hist_decay
                ) * (svg_grad ** 2)
                adj_grad = svg_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += self.inner_lr * adj_grad

        return particles.view(B * S, D).detach()

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
                desc=f"Run {run_id+1} Training SVG Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)

                self.model.eval()
                x_svg_batch = self._svg_sampler(
                    x_original_batch_dev, y_original_batch_dev, self.model, epoch
                )
                y_repeated_batch = y_original_batch_dev.repeat_interleave(
                    self.num_samples, dim=0
                )

                self.model.train()
                predictions_logits_batch = self.model(x_svg_batch)
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
                f"{checkpoint_dir}/SDRO_SVG_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
