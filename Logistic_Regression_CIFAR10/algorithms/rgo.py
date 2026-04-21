"""Rejection sampling-based WDRO (RGO) baseline."""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from algorithms.base import BaseLinearDRO, make_linear_model


class SDRO_RGO(BaseLinearDRO):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        epsilon: float = 0.1,
        lambda_param: float = 1.0,
        rgo_inner_lr: float = 0.01,
        rgo_inner_steps: int = 20,
        num_samples: int = 10,
        max_itr: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        rgo_vectorized_max_trials: int = 100,
        device: str = "cpu",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.rgo_inner_lr = rgo_inner_lr
        self.rgo_inner_steps = rgo_inner_steps
        self.num_samples = num_samples
        self.max_itr = max_itr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rgo_vectorized_max_trials = rgo_vectorized_max_trials
        self.model = make_linear_model(
            self.input_dim, self.num_classes, self.fit_intercept, self.device
        )

    def _get_model_loss_value_batched(
        self,
        x_features_batch: torch.Tensor,
        y_target_batch: torch.Tensor,
        model_instance: nn.Module,
    ) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction="none")(
            model_instance(x_features_batch), y_target_batch
        )

    def _rgo_sampler_vectorized(
        self,
        x_original_batch: torch.Tensor,
        y_original_batch: torch.Tensor,
        current_model_state: nn.Module,
        num_samples_to_generate: int,
        epoch: int,
    ) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(self.rgo_inner_steps)
        for _ in range(inner_steps):
            x_pert_batch.requires_grad_(True)
            per_sample_losses = self._get_model_loss_value_batched(
                x_pert_batch, y_original_batch, current_model_state
            )
            per_sample_grads, = torch.autograd.grad(
                outputs=per_sample_losses,
                inputs=x_pert_batch,
                grad_outputs=torch.ones_like(per_sample_losses),
            )
            x_pert_batch = x_pert_batch.detach()
            grad_total = -per_sample_grads / self.lambda_param + 2 * (
                x_pert_batch - x_orig_detached_batch
            )
            x_pert_batch -= lr_inner * grad_total
        x_opt_star_batch = x_pert_batch
        var_rgo = self.epsilon
        if var_rgo <= 1e-12:
            return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)
        std_rgo = math.sqrt(var_rgo)
        f_model_loss_opt_star = self._get_model_loss_value_batched(
            x_opt_star_batch, y_original_batch, current_model_state
        )
        norm_sq_opt_star = torch.sum(
            (x_opt_star_batch - x_orig_detached_batch) ** 2, dim=1
        )
        f_L_xi_opt_star = (
            -f_model_loss_opt_star / (self.lambda_param * self.epsilon)
        ) + (norm_sq_opt_star / self.epsilon)
        x_opt_star_3d = x_opt_star_batch.unsqueeze(1)
        x_original_3d = x_orig_detached_batch.unsqueeze(1)
        f_L_xi_opt_star_3d = f_L_xi_opt_star.unsqueeze(1)
        final_accepted_perturbations = torch.zeros(
            (batch_size, num_samples_to_generate, self.input_dim), device=self.device
        )
        active_flags = torch.ones(
            (batch_size, num_samples_to_generate), dtype=torch.bool, device=self.device
        )
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any():
                break
            pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
            x_candidates = x_opt_star_3d + pert_proposals
            x_candidates_flat = x_candidates.view(-1, self.input_dim)
            y_repeated = y_original_batch.repeat_interleave(num_samples_to_generate, dim=0)
            f_model_loss_candidates = self._get_model_loss_value_batched(
                x_candidates_flat, y_repeated, current_model_state
            ).view(batch_size, num_samples_to_generate)
            norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
            f_L_xi_candidates = (
                -f_model_loss_candidates / (self.lambda_param * self.epsilon)
            ) + (norm_sq_candidates / self.epsilon)
            diff_cand_opt_norm_sq = torch.sum(pert_proposals ** 2, dim=2)
            exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
            acceptance_probs = torch.exp(
                torch.clamp(
                    -f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10
                )
            )
            newly_accepted_mask = (
                torch.rand_like(acceptance_probs) < acceptance_probs
            ) & active_flags
            final_accepted_perturbations[newly_accepted_mask] = pert_proposals[
                newly_accepted_mask
            ]
            active_flags[newly_accepted_mask] = False
        return (x_opt_star_3d + final_accepted_perturbations).view(-1, self.input_dim)

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
                desc=f"Run {run_id+1} Training RGO Epoch {epoch+1}/{self.max_itr}",
                leave=False,
            )
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(
                    x_original_batch_dev,
                    y_original_batch_dev,
                    self.model,
                    self.num_samples,
                    epoch,
                )
                y_repeated_batch = y_original_batch_dev.repeat_interleave(
                    self.num_samples, dim=0
                )
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
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
                f"{checkpoint_dir}/SDRO_RGO_run{run_id}_epoch_{epoch+1}.pth",
            )
        return loss_history
