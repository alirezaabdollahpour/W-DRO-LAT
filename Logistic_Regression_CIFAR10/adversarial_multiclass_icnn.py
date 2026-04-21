#!/usr/bin/env python3
"""adversarial_multiclass_icnn.py

Reproduces **Section 6.3 (Adversarial Multi-class Logistic Regression)** and
**Figure 6** from *GRADIENT FLOW SAMPLER-BASED DISTRIBUTIONALLY ROBUST
OPTIMIZATION*.

This script evaluates multi-class logistic regression on **CIFAR-10 ResNet-50
features (512-dim)** and reports robustness under an **l2-PGD** test-time attack.

Competitors (Figure 6) + added ICNN method
------------------------------------------
Baselines included (as in Figure 6):
  - SAA / ERM
  - Dual (entropy-regularized Wasserstein DRO dual / Sinkhorn SDRO)
  - WRM (Sinha et al.)
  - RGO (rejection sampling baseline)
  - WGF (Wasserstein gradient flow sampler)
  - WFR (Wasserstein–Fisher–Rao sampler)
  - SVG (SVGD sampler)

Added competitor:
  - ICNN-DRO (Brenier/transport-map adversary) using a convex potential
    ψ_ω parameterized by an ICNN, with transport map T_ω(x) = ∇_x ψ_ω(x).

Notation (matches `main (12).tex`; used in comments/docstrings)
--------------------------------------------------------------
  - x ∈ R^d : CIFAR-10 feature vector (d = 512)
  - y ∈ {0,…,9} : label
  - B : classifier parameters (logistic regression; PyTorch Linear weights)
  - ψ_ω : convex potential (ICNN)
  - T_ω(x) = ∇ψ_ω(x) : transport map producing adversarial features

Default hyperparameters (Figure 6 caption)
------------------------------------------
  - τ = 0.05  (converted to λ = 1/(2τ) = 10.0)
  - ε_ent ∈ {0.2, 0.02, 0.002}  (entropy regularization for SDRO methods)
  - outer epochs = 10
  - inner-loop stepsize = 0.01 and inner-loop iterations = 100
    (applied to methods with an explicit inner loop: WRM/WGF/WFR/SVG; for RGO
     we also run the inner optimizer for 100 steps for consistency)

Data / feature extraction
-------------------------
CIFAR-10 images are resized to 224×224 and normalized with ImageNet mean/std,
then passed through a ResNet-50 pretrained on ImageNet. We follow the provided
reference code (`Multiclass_classification.py`) and replace the final fc head by
a 512-dim Linear + ReLU head. Features are cached to disk to make runs
repeatable.

Implementation provenance (strict requirement)
----------------------------------------------
  - The BB+Armijo optimizer + ICNN modules (NonNegativeLinear, principled
    log-normal init, InputConvexPotential, T_ω) are copied **verbatim** from
    `uncertain_least_squares_icnn.py`.
  - The baseline SDRO / sampler implementations are adapted directly from
    `Multiclass_classification.py`.

Run
---
  python adversarial_multiclass_icnn.py --help
  python adversarial_multiclass_icnn.py

Outputs
-------
  - JSON results per ε_ent in the output directory
  - A Figure-6 style robustness plot (test error vs normalized attack strength Δ)

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception as _e:
    torchvision = None  # type: ignore
    transforms = None  # type: ignore
    _TORCHVISION_IMPORT_ERROR = _e
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def extract_features(data_loader, model, device):
    model.to(device)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)

def pgd_attack(model, features, labels, epsilon, alpha, num_iter):
    """PGD Adversarial Attack (l2 norm) - CORRECTED VERSION"""
    perturbed_features = features.clone().detach().to(DEVICE)
    perturbed_features.requires_grad = True
    original_features = features.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        if perturbed_features.grad is not None:
            perturbed_features.grad.zero_()

        outputs = model(perturbed_features)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = perturbed_features.grad.detach()
        grad_norm = torch.linalg.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True) + 1e-12
        normalized_grad = grad / grad_norm
        
        reshape_dims = [grad.shape[0]] + [1] * (grad.dim() - 1)
        
        perturbed_features.data = perturbed_features.data + alpha * normalized_grad
        
        perturbation = perturbed_features.data - original_features.data
        
        pert_norm = torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True)
        factor = epsilon / (pert_norm + 1e-12)
        factor = torch.min(torch.ones_like(factor), factor)
        
        perturbation = perturbation * factor.view(*reshape_dims)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x): return self.linear(x)

class DROError(Exception): pass

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.linear(x)

class BaseLinearDRO:
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool, sample_level: int = 6):
        self.input_dim, self.num_classes, self.fit_intercept = input_dim, num_classes, fit_intercept
        self.model: nn.Module
        self.sample_level = sample_level
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor: return torch.as_tensor(data, dtype=torch.float32)
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1: X = X.reshape(-1, self.input_dim)
        if y.ndim > 1: y = y.flatten()
        if X.shape[0] != y.shape[0]: raise DROError(f"Shapes mismatch: X {X.shape[0]}, y {y.shape[0]}")
        if X.shape[1] != self.input_dim: raise DROError(f"Input dim mismatch: expected {self.input_dim}, got {X.shape[1]}")
        return X, y
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        dataset = TensorDataset(self._to_tensor(X), torch.as_tensor(y, dtype=torch.long))
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
    
    def _compute_loss(self, predictions, targets, m, lambda_reg):
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, targets) / max(lambda_reg, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lambda_reg
    
    def _compute_dual_loss(self, data, targets, lam, epsilon):
        m = 2 ** 6
        expanded_data = data.repeat_interleave(m, dim=0)
        noise = torch.randn_like(expanded_data) * math.sqrt(epsilon)
        noisy_data = expanded_data + noise
        repeated_target = targets.repeat_interleave(m, dim=0)
        
        predictions = self.model(noisy_data)
        criterion = nn.CrossEntropyLoss(reduction='none')
        residuals = criterion(predictions, repeated_target) / max(lam * epsilon, 1e-8)
        residual_matrix = residuals.view(-1, m).T
        return torch.mean(torch.logsumexp(residual_matrix, dim=0) - math.log(m)) * lam * epsilon
    
class SDRO_Dual(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 1e-3,
                 lambda_param: float = 1e2, max_itr: int = 100, learning_rate: float = 1e-2,
                 sample_level: int = 6, batch_size: int = 64, device: str = "cpu"):
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param, self.max_itr, self.learning_rate, self.sample_level, self.batch_size = epsilon, lambda_param, max_itr, learning_rate, sample_level, batch_size
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = LinearModel(input_dim, output_dim=num_classes, bias=fit_intercept).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> None:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lambda_reg = self.lambda_param * self.epsilon
        loss_history = []
        self.model.train()
        for epoch in range(self.max_itr):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SinkhornLinearDRO Epoch {epoch+1}/{self.max_itr}", leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                levels = np.arange(self.sample_level + 1)
                numerators = 2.0**(-levels)
                denominator = 2.0 - 2.0**(-self.sample_level)
    
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
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_Dual_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history


class SDRO_RGO(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, rgo_inner_lr: float = 0.01, rgo_inner_steps: int = 20, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.rgo_inner_lr, self.rgo_inner_steps = rgo_inner_lr, rgo_inner_steps
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.rgo_vectorized_max_trials = 100 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _rgo_sampler_vectorized(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, current_model_state: nn.Module, num_samples_to_generate: int, epoch: int) -> torch.Tensor:
        batch_size = x_original_batch.size(0)
        x_orig_detached_batch = x_original_batch.detach()
        x_pert_batch = x_orig_detached_batch.clone()
        lr_inner = self.rgo_inner_lr
        inner_steps = int(self.rgo_inner_steps)
        for _ in range(inner_steps):
            x_pert_batch.requires_grad_(True)
            per_sample_losses = self._get_model_loss_value_batched(x_pert_batch, y_original_batch, current_model_state)
            per_sample_grads, = torch.autograd.grad(outputs=per_sample_losses, inputs=x_pert_batch, grad_outputs=torch.ones_like(per_sample_losses))
            x_pert_batch = x_pert_batch.detach()
            grad_total = -per_sample_grads / self.lambda_param + 2 * (x_pert_batch - x_orig_detached_batch)
            x_pert_batch -= lr_inner * grad_total
        x_opt_star_batch = x_pert_batch
        var_rgo = self.epsilon
        if var_rgo <= 1e-12: return x_opt_star_batch.repeat_interleave(num_samples_to_generate, dim=0)
        std_rgo = math.sqrt(var_rgo)
        f_model_loss_opt_star = self._get_model_loss_value_batched(x_opt_star_batch, y_original_batch, current_model_state)
        norm_sq_opt_star = torch.sum((x_opt_star_batch - x_orig_detached_batch) ** 2, dim=1)
        f_L_xi_opt_star = (-f_model_loss_opt_star / (self.lambda_param * self.epsilon)) + (norm_sq_opt_star / self.epsilon)
        x_opt_star_3d, x_original_3d, f_L_xi_opt_star_3d = x_opt_star_batch.unsqueeze(1), x_orig_detached_batch.unsqueeze(1), f_L_xi_opt_star.unsqueeze(1)
        final_accepted_perturbations = torch.zeros((batch_size, num_samples_to_generate, self.input_dim), device=self.device)
        active_flags = torch.ones((batch_size, num_samples_to_generate), dtype=torch.bool, device=self.device)
        for _ in range(self.rgo_vectorized_max_trials):
            if not active_flags.any(): break
            pert_proposals = torch.randn_like(final_accepted_perturbations) * std_rgo
            x_candidates = x_opt_star_3d + pert_proposals
            x_candidates_flat = x_candidates.view(-1, self.input_dim)
            y_repeated = y_original_batch.repeat_interleave(num_samples_to_generate, dim=0)
            f_model_loss_candidates = self._get_model_loss_value_batched(x_candidates_flat, y_repeated, current_model_state).view(batch_size, num_samples_to_generate)
            norm_sq_candidates = torch.sum((x_candidates - x_original_3d) ** 2, dim=2)
            f_L_xi_candidates = (-f_model_loss_candidates / (self.lambda_param * self.epsilon)) + (norm_sq_candidates / self.epsilon)
            diff_cand_opt_norm_sq = torch.sum(pert_proposals**2, dim=2)
            exponent_term3 = diff_cand_opt_norm_sq / (2 * var_rgo)
            acceptance_probs = torch.exp(torch.clamp(-f_L_xi_candidates + f_L_xi_opt_star_3d + exponent_term3, max=10))
            newly_accepted_mask = (torch.rand_like(acceptance_probs) < acceptance_probs) & active_flags
            final_accepted_perturbations[newly_accepted_mask] = pert_proposals[newly_accepted_mask]
            active_flags[newly_accepted_mask] = False
        return (x_opt_star_3d + final_accepted_perturbations).view(-1, self.input_dim)
    
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training RGO Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_rgo_batch = self._rgo_sampler_vectorized(x_original_batch_dev, y_original_batch_dev, self.model, self.num_samples, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_rgo_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_RGO_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history
    
class SDRO_WFR(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WFR_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        batch_size = x_orig.shape[0]

        weights = torch.ones((batch_size, self.num_samples), dtype=torch.float32, device=self.device) / self.num_samples

        x_clone = x_orig.clone().detach().unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)

        wt_lr = self.inner_lr
        weight_exponent = 1 - self.lambda_param * self.epsilon * wt_lr
        low_weight_threshold = 1e-4
        std_dev = math.sqrt(2 * self.inner_lr * self.lambda_param * self.epsilon)
        criterion = nn.CrossEntropyLoss(reduction='none')

        num_steps = int(self.inner_itr)

        # Avoid computing/accumulating parameter gradients inside the sampler (only need d/dx).
        param_requires_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)

        try:
            for _ in range(num_steps):
                x_clone.requires_grad_(True)

                loss_values = criterion(model(x_clone), y_repeated)
                grads = torch.autograd.grad(loss_values.sum(), x_clone, retain_graph=False, create_graph=False)[0]
                x_clone = x_clone.detach()

                with torch.no_grad():
                    mean = x_clone + self.inner_lr * (grads - 2 * self.lambda_param * (x_clone - x_original_expanded))
                    x_clone = mean + torch.randn_like(mean) * std_dev

                    current_loss = criterion(model(x_clone), y_repeated).view(batch_size, self.num_samples)
                    dist_sq = torch.sum((x_clone - x_original_expanded) ** 2, dim=1).view(batch_size, self.num_samples)
                    energy_term = current_loss - 2 * self.lambda_param * dist_sq

                    weights.pow_(weight_exponent)
                    weights.mul_(torch.exp(energy_term * wt_lr))
                    weights.div_(torch.sum(weights, dim=1, keepdim=True).add_(1e-9))

                    # Birth-death step (fully vectorized; no Python branching on CUDA tensors).
                    low_weight_mask = weights < low_weight_threshold
                    rows_with_low_weights = low_weight_mask.any(dim=1, keepdim=True)

                    x_reshaped = x_clone.view(batch_size, self.num_samples, self.input_dim)
                    max_weight_vals, max_weight_indices = weights.max(dim=1, keepdim=True)
                    highest_weight_point_data = x_reshaped.gather(
                        1, max_weight_indices.unsqueeze(-1).expand(-1, -1, self.input_dim)
                    )

                    low_weights_sum = torch.sum(weights * low_weight_mask, dim=1, keepdim=True)
                    num_low_weights = low_weight_mask.sum(dim=1, keepdim=True, dtype=weights.dtype)
                    avg_weight = (max_weight_vals + low_weights_sum) / (num_low_weights + 1.0 + 1e-9)

                    max_weight_mask = torch.zeros_like(weights, dtype=torch.bool)
                    max_weight_mask.scatter_(1, max_weight_indices, True)
                    update_mask = (low_weight_mask | max_weight_mask) & rows_with_low_weights
                    weights = torch.where(update_mask, avg_weight, weights)

                    x_update_mask = low_weight_mask & rows_with_low_weights
                    x_reshaped = torch.where(x_update_mask.unsqueeze(-1), highest_weight_point_data, x_reshaped)
                    x_clone = x_reshaped.view(-1, self.input_dim)
                    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-9)
        finally:
            for p, req in zip(model.parameters(), param_requires_grad):
                p.requires_grad_(req)

        return x_clone.detach(), weights.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WFR Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WFR_batch, weights = self._WFR_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WFR_batch)
                loss_values = nn.CrossEntropyLoss(reduction='none')(predictions_logits_batch, y_repeated_batch)
                optimization_loss = (loss_values.view(-1, self.num_samples) * weights).sum(dim=1).mean()
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_WFR_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SDRO_WGF(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _WGF_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        x_clone = x_clone.unsqueeze(1).expand(-1, self.num_samples, -1).contiguous().view(-1, self.input_dim)
        y_repeated = y_orig.repeat_interleave(self.num_samples, dim=0)
        x_original_expanded = x_orig.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, self.input_dim)
        num_steps = int(self.inner_itr)
        for _ in range(num_steps):
            x_clone.requires_grad_(True)
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_repeated)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach()
            
            mean = x_clone + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_original_expanded))
            std_dev = torch.sqrt(torch.tensor(2*self.inner_lr*self.lambda_param*self.epsilon, device=self.device))
            
            noise = torch.randn_like(mean) * std_dev
            x_clone = mean + noise

        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WGF Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WGF_batch = self._WGF_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                self.model.train()
                predictions_logits_batch = self.model(x_WGF_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_WGF_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history
    
class SDRO_SVG(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, epsilon: float = 0.1, 
                 lambda_param: float = 1.0, num_samples: int = 10, max_itr: int = 30, 
                 learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, 
                 inner_itr: int = 200, adagrad_hist_decay: float = 0.9, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.epsilon, self.lambda_param = epsilon, lambda_param
        self.inner_lr, self.inner_itr, self.adagrad_hist_decay = inner_lr, inner_itr, adagrad_hist_decay
        self.num_samples, self.max_itr, self.learning_rate, self.batch_size = num_samples, max_itr, learning_rate, batch_size
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _rbf_kernel_batched_torch(self, particles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = particles.shape
        device = particles.device

        sq_dist = torch.cdist(particles, particles, p=2).pow(2)

        median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0]
        median_sq_dist = median_sq_dist.view(B, 1, 1)
        h_squared = median_sq_dist / (torch.log(torch.tensor(S, dtype=torch.float32, device=device)) + 1e-8)

        K = torch.exp(-sq_dist / (2 * h_squared))

        diff = particles.unsqueeze(2) - particles.unsqueeze(1)
        grad_K_x = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)

        return K, grad_K_x

    def _svg_sampler(self, x_original_batch: torch.Tensor, y_original_batch: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        if x_original_batch.shape[0] == 0:
            return torch.empty(0, x_original_batch.shape[1], device=x_original_batch.device)

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

            neg_log_likelihood = nn.CrossEntropyLoss(reduction='sum')(model(x_tensor), y_repeated)
            grad_log_py_x, = torch.autograd.grad(outputs=neg_log_likelihood, inputs=x_tensor)
            
            grad_log_px = -2 * self.lambda_param * (x_tensor - x_orig_repeated_flat)

            total_grad_flat = (grad_log_py_x + grad_log_px) / (self.lambda_param * self.epsilon)
            total_grad = total_grad_flat.view(B, S, D)

            K, grad_K_x = self._rbf_kernel_batched_torch(particles)
            
            K_grad_prod = torch.bmm(K, total_grad)
            
            sum_grad_K = torch.sum(grad_K_x, dim=2)

            svg_grad = (K_grad_prod + sum_grad_K) / S

            with torch.no_grad():
                # Adagrad optimizer for stable updates
                hist_grad = self.adagrad_hist_decay * hist_grad + (1 - self.adagrad_hist_decay) * (svg_grad**2)
                adj_grad = svg_grad / (1e-6 + torch.sqrt(hist_grad))
                particles += self.inner_lr * adj_grad

        return particles.view(B * S, D).detach()

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training SVG Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                
                self.model.eval()
                x_svg_batch = self._svg_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                y_repeated_batch = y_original_batch_dev.repeat_interleave(self.num_samples, dim=0)
                
                self.model.train()
                predictions_logits_batch = self.model(x_svg_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_repeated_batch)
                
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()

                epoch_loss_record += optimization_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item())
            
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/SDRO_SVG_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history

class WRM(BaseLinearDRO):
    def __init__(self, input_dim: int, num_classes: int, fit_intercept: bool = True, lambda_param: float = 1.0, max_itr: int = 30, learning_rate: float = 0.01, batch_size: int = 64, inner_lr: float = 0.01, inner_itr: int = 200, device: str = "cpu"):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.max_itr, self.learning_rate, self.batch_size = max_itr, learning_rate, batch_size 
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)
    def _get_model_loss_value_batched(self, x_features_batch: torch.Tensor, y_target_batch: torch.Tensor, model_instance: nn.Module) -> torch.Tensor:
        return nn.CrossEntropyLoss(reduction='none')(model_instance(x_features_batch), y_target_batch)
    def _sinha_sampler(self, x_orig: torch.Tensor, y_orig: torch.Tensor, model: nn.Module, epoch) -> torch.Tensor:
        x_clone = x_orig.clone().detach().requires_grad_(True)
        num_steps = int(self.inner_itr)
        for _ in range(num_steps):
            loss_values = nn.CrossEntropyLoss(reduction='none')(model(x_clone), y_orig)
            grads, = torch.autograd.grad(loss_values, x_clone, grad_outputs=torch.ones_like(loss_values))
            x_clone = x_clone.detach() + self.inner_lr * (grads - 2*self.lambda_param*(x_clone - x_orig))
            x_clone.requires_grad_(True)
        return x_clone.detach()
    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training WRM Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev, y_original_batch_dev = x_original_batch.to(self.device), y_original_batch.to(self.device)
                self.model.eval()
                x_WRM_batch = self._sinha_sampler(x_original_batch_dev, y_original_batch_dev, self.model, epoch)
                self.model.train()
                predictions_logits_batch = self.model(x_WRM_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_original_batch_dev)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = nn.CrossEntropyLoss()(self.model(x_original_batch_dev), y_original_batch_dev)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)
            
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/WRM_run{run_id}_epoch_{epoch+1}.pth')
            
        return loss_history


def _brenier_projection_features(
    z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    r"""Within-class Brenier projection for feature-space vectors.

    Optimally reassigns adversarial points to nominals *within each class*
    using the Hungarian algorithm on squared-Euclidean cost.

    Adapted from MNIST.py's ``brenier_projection``.  The only structural
    difference is that features are [N, d] (not [N, C, H, W]), so no
    ``view(N, -1)`` reshaping is needed.

    Parameters
    ----------
    z : Tensor [N, d]  -- adversarial feature points
    x : Tensor [N, d]  -- nominal feature points
    y : Tensor [N]     -- labels (permuted alongside z within class)

    Returns
    -------
    z_proj  : Tensor [N, d]  -- optimally reassigned within each class
    y_proj  : Tensor [N]     -- same as y (labels preserved within class)
    delta   : float          -- wasted-transport gap  (C_id - C_ot)
    C_id    : float          -- identity transport cost
    C_ot    : float          -- optimal transport cost
    """
    from scipy.optimize import linear_sum_assignment

    N = z.size(0)
    if N <= 1:
        C_id = float(((z - x) ** 2).sum().item()) / max(N, 1)
        return z.clone(), y.clone(), 0.0, C_id, C_id

    z_flat = z.detach()   # already [N, d]
    x_flat = x.detach()   # already [N, d]

    # Build global permutation by solving per-class LAPs
    perm_np = np.arange(N, dtype=np.int64)  # identity by default
    y_np = y.cpu().numpy()
    unique_classes = np.unique(y_np)

    for c in unique_classes:
        idx_c = np.where(y_np == c)[0]
        n_c = len(idx_c)
        if n_c <= 1:
            continue

        # Extract class-c subsets on CPU
        x_c = x_flat[idx_c].cpu().float()   # [n_c, d]
        z_c = z_flat[idx_c].cpu().float()   # [n_c, d]

        # Cost matrix: cost[i, j] = ||x_{I_c[i]} - z_{I_c[j]}||^2
        x_sq = (x_c ** 2).sum(dim=1, keepdim=True)
        z_sq = (z_c ** 2).sum(dim=1, keepdim=True)
        cross = x_c.mm(z_c.t())
        cost_c = (x_sq + z_sq.t() - 2.0 * cross).clamp(min=0.0)
        cost_c_np = cost_c.numpy()

        # Solve LAP within class c
        row_ind, col_ind = linear_sum_assignment(cost_c_np)

        # Map back to global indices
        local_perm = np.empty(n_c, dtype=np.int64)
        local_perm[row_ind] = col_ind
        perm_np[idx_c] = idx_c[local_perm]

    perm = torch.tensor(perm_np, device=z.device, dtype=torch.long)

    # Apply permutation
    z_proj = z[perm]
    y_proj = y[perm]  # within-class: y_proj == y

    # Compute metrics
    C_id = float(((z - x) ** 2).sum(dim=1).mean().item())
    C_ot = float(((z_proj - x) ** 2).sum(dim=1).mean().item())
    delta = max(C_id - C_ot, 0.0)

    return z_proj, y_proj, delta, C_id, C_ot


class PPA(BaseLinearDRO):
    """Projected Particle Ascent (PPA) for distributionally robust optimization.

    Adapted from MNIST.py's ``train_step_ppa``.  PPA refines the plain WRM
    adversary by alternating Brenier (within-class optimal transport)
    projections with gradient ascent rounds:

        Round 0:   Plain WRM ascent (replicates WRM exactly — dominance condition).
        Round r>=1: (a) Within-class Brenier projection   z <- Pi(z)
                    (b) Constant-lr WRM ascent from projected z
        Final:     One last Brenier projection to capture remaining wasted transport.

    This guarantees L(theta, z_PPA) >= L(theta, z_WRM), so the PPA inner
    oracle strictly dominates the WRM inner oracle.

    In the feature-space setting (CIFAR-10 ResNet-50 features), there is no
    pixel clamping — features are unconstrained.
    """

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
        ppa_num_rounds: int = 5,
        ppa_min_rounds: int = 2,
        ppa_refine_steps: int = 15,
        ppa_refine_lr: float = 5e-3,
        ppa_delta_rtol: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)
        self.lambda_param = lambda_param
        self.inner_lr, self.inner_itr = inner_lr, inner_itr
        self.max_itr, self.learning_rate, self.batch_size = max_itr, learning_rate, batch_size
        self.ppa_num_rounds = ppa_num_rounds
        self.ppa_min_rounds = ppa_min_rounds
        self.ppa_refine_steps = ppa_refine_steps
        self.ppa_refine_lr = ppa_refine_lr
        self.ppa_delta_rtol = ppa_delta_rtol
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

    def _wrm_ascent(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        num_steps: int,
        lr: float,
    ) -> torch.Tensor:
        """WRM gradient ascent with diminishing step size (round 0).

        Maximises CE(z, y) - lambda ||z - x||^2  w.r.t. z,
        using step size lr / sqrt(s).
        """
        if num_steps == 0:
            return x_orig.detach()
        x_anc = x_orig.detach()
        z = x_anc.clone()
        for s in range(1, num_steps + 1):
            z.requires_grad_(True)
            per_sample_ce = F.cross_entropy(model(z), y_orig, reduction="none")
            grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
            eta_s = lr / math.sqrt(s)
            z = z.detach() + eta_s * (grads - 2.0 * self.lambda_param * (z.detach() - x_anc))
        return z.detach()

    def _wrm_ascent_const_lr(
        self,
        z0: torch.Tensor,
        x_anchor: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
        num_steps: int,
        lr: float,
    ) -> torch.Tensor:
        """Constant-lr WRM ascent for PPA refinement rounds.

        After Brenier projection the particles sit at an OT-optimal coupling;
        constant step size is more appropriate than diminishing because this
        is refinement from a new starting point, not a cold start.

        Maximises CE(z, y) - lambda ||z - x_anchor||^2  w.r.t. z,
        starting from z = z0 with anchor at x_anchor.
        """
        if num_steps == 0:
            return z0.detach()
        x_anc = x_anchor.detach()
        z = z0.detach().clone()
        for _ in range(num_steps):
            z.requires_grad_(True)
            per_sample_ce = F.cross_entropy(model(z), y_orig, reduction="none")
            grads = torch.autograd.grad(per_sample_ce.sum(), z, create_graph=False)[0]
            z = z.detach() + lr * (grads - 2.0 * self.lambda_param * (z.detach() - x_anc))
        return z.detach()

    def _ppa_sampler(
        self,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Full PPA inner loop: round-0 WRM + refinement rounds + final projection.

        Follows MNIST.py ``train_step_ppa`` exactly:
          Round 0:   WRM ascent (same steps/lr as plain WRM — dominance condition)
          Round r:   Brenier projection -> adaptive stopping -> const-lr ascent
          Final:     One last Brenier projection
        """
        # Round 0: replicate WRM exactly
        z = self._wrm_ascent(x_orig, y_orig, model, self.inner_itr, self.inner_lr)

        # Refinement rounds 1..R-1
        for round_idx in range(1, self.ppa_num_rounds):
            # (a) Within-class Brenier projection
            z, _y_proj, delta, _C_id, _C_ot = _brenier_projection_features(z, x_orig, y_orig)

            # (b) Adaptive stopping with minimum-rounds guarantee
            if (round_idx >= self.ppa_min_rounds
                    and delta < self.ppa_delta_rtol * max(_C_id, 1e-12)):
                break

            # (c) Constant-lr ascent from projected position
            z = self._wrm_ascent_const_lr(
                z, x_orig, y_orig, model,
                self.ppa_refine_steps, self.ppa_refine_lr,
            )

        # Final projection: captures remaining wasted transport (gain >= 0)
        z, _y_proj, _delta_final, _, _ = _brenier_projection_features(z, x_orig, y_orig)

        return z.detach()

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = 'checkpoints', run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)
        optimizer_theta = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_history = []
        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training PPA Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)
                # Inner maximization: freeze classifier
                self.model.eval()
                x_ppa_batch = self._ppa_sampler(x_original_batch_dev, y_original_batch_dev, self.model)
                # Outer minimization: unfreeze classifier
                self.model.train()
                predictions_logits_batch = self.model(x_ppa_batch)
                optimization_loss = nn.CrossEntropyLoss()(predictions_logits_batch, y_original_batch_dev)
                optimizer_theta.zero_grad()
                optimization_loss.backward()
                optimizer_theta.step()
                with torch.no_grad():
                    comparable_loss = nn.CrossEntropyLoss()(self.model(x_original_batch_dev), y_original_batch_dev)
                epoch_loss_record += comparable_loss.item()
                pbar.set_postfix(optim_loss=optimization_loss.item(), record_loss=comparable_loss.item())
            avg_epoch_loss_record = epoch_loss_record / len(dataloader)
            loss_history.append(avg_epoch_loss_record)

            torch.save(self.model.state_dict(), f'{checkpoint_dir}/PPA_run{run_id}_epoch_{epoch+1}.pth')

        return loss_history


def train_saa(model, train_loader, epochs=30, learning_rate=0.001, checkpoint_dir: str = 'checkpoints', run_id: int = 0):
    model.to(DEVICE).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Run {run_id+1} SAA Epoch {epoch+1}/{epochs}", leave=False)
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        torch.save(model.state_dict(), f'{checkpoint_dir}/SAA_run{run_id}_epoch_{epoch+1}.pth')
        
    return loss_history

def evaluate_model(model_object, test_features_np, test_labels_np, attack_fn, epsilon_list):
    pytorch_model = model_object.model.eval()
    
    clean_accuracy = model_object.score(test_features_np, test_labels_np) * 100
    results = {0.0: clean_accuracy}
    print(f"Accuracy on clean test set: {results[0.0]:.2f}%")

    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_features_np).float(), torch.from_numpy(test_labels_np).long()), batch_size=128)
    
    for epsilon in epsilon_list:
        if epsilon == 0.0: continue
        correct, total = 0, 0
        for features, labels in tqdm(test_loader, desc=f"Evaluating (epsilon={epsilon:.4f})", leave=False):
            perturbed_features = attack_fn(model=pytorch_model, features=features, labels=labels, epsilon=epsilon, alpha=epsilon / 8, num_iter=10)
            with torch.no_grad():
                outputs = pytorch_model(perturbed_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(DEVICE)).sum().item()
        results[epsilon] = 100 * correct / total
        print(f"Accuracy under PGD attack (epsilon={epsilon:.4f}): {results[epsilon]:.2f}%")
    return results


def plot_training_loss(all_runs_losses, lam, eps):
    plt.style.use('jz.mplstyle')
    plt.figure()
    
    for model_name, loss_runs in all_runs_losses.items():
        if not loss_runs:
            continue
        
        loss_array = np.array(loss_runs)
        epochs = np.arange(1, loss_array.shape[1] + 1)
        mean_loss = np.mean(loss_array, axis=0)
        std_loss = np.std(loss_array, axis=0)
        
        line, = plt.plot(epochs, mean_loss, label=model_name)
        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=line.get_color(), alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"training_loss_lam={lam}_eps={eps}.pdf")
    print(f"\nTraining loss plot saved as 'training_loss_lam={lam}_eps={eps}.pdf'")

def plot_robustness_results(all_runs_results, perturbation_levels, epsilon_values, lam, eps):
    plt.style.use('jz.mplstyle')
    plt.figure(figsize=(3.8, 2.613), dpi=300, tight_layout=True)
    
    for model_name, results_runs in all_runs_results.items():
        if not results_runs:
            continue
            
        errors_matrix = []
        query_keys = [0.0] + epsilon_values[1:]

        for single_run_result in results_runs:
            ordered_errors = [100.0 - single_run_result.get(key, 0.0) for key in query_keys]
            errors_matrix.append(ordered_errors)
        
        errors_array = np.array(errors_matrix)
        mean_errors = np.mean(errors_array, axis=0)
        std_errors = np.std(errors_array, axis=0)
        
        line, = plt.plot(perturbation_levels, mean_errors, label=model_name)
        plt.fill_between(
            perturbation_levels,
            mean_errors - std_errors,
            mean_errors + std_errors,
            color=line.get_color(),
            alpha=0.25
        )

    plt.xlabel("perturbation $\Delta$")
    plt.ylabel("test error (%)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"robustness_lam={lam}_eps={eps}.pdf")
    print("\nRobustness plot saved as 'robustness_comparison_plot.pdf'")

def get_cifar10_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader




# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds (NumPy + PyTorch) for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# CIFAR-10 -> ResNet-50 feature extraction (512-dim) with caching
# -----------------------------------------------------------------------------

def get_cifar10_dataloaders(data_dir: str, batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 dataloaders with ImageNet preprocessing (paper setup).

    CIFAR-10 images are resized to 224×224 and normalized using ImageNet mean/std.
    """
    if torchvision is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for CIFAR-10 loading / transforms, but it failed to import. "
            "Install a compatible torchvision build for your PyTorch version."
        ) from _TORCHVISION_IMPORT_ERROR

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def build_resnet50_feature_extractor(output_dim: int = 512) -> nn.Module:
    """Create the ResNet-50 feature extractor used in Section 6.3 / Figure 6."""
    if torchvision is None:
        raise RuntimeError(
            "torchvision is required to construct the ResNet-50 feature extractor, but it failed to import. "
            "Install a compatible torchvision build for your PyTorch version."
        ) from _TORCHVISION_IMPORT_ERROR
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet50(weights=weights)
    except Exception:
        # Backward-compat for older torchvision
        resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, output_dim),
        nn.ReLU(inplace=True),
    )
    return resnet


def load_or_extract_cifar10_resnet50_features(
    feature_file: Path,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    force_reextract: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached CIFAR-10 ResNet-50 features or extract and cache them."""
    if feature_file.exists() and not force_reextract:
        print(f"Loading cached features from: {feature_file}")
        data = torch.load(feature_file, map_location="cpu")
        return data["train_features"], data["train_labels"], data["test_features"], data["test_labels"]

    print("Cached feature file not found (or forced re-extract). Extracting features...")
    if torchvision is None or transforms is None:
        raise RuntimeError(
            "torchvision is required to extract CIFAR-10 ResNet-50 features, but it failed to import. "
            "If you already have a cached feature file, pass --feature_cache pointing to it."
        ) from _TORCHVISION_IMPORT_ERROR
    feature_file.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: the 512-dim head is randomly initialized; caching ensures repeatability.
    resnet_feature_extractor = build_resnet50_feature_extractor(output_dim=512)

    cifar_train_loader, cifar_test_loader = get_cifar10_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers)
    train_features, train_labels = extract_features(cifar_train_loader, resnet_feature_extractor, device)
    test_features, test_labels = extract_features(cifar_test_loader, resnet_feature_extractor, device)

    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
        },
        feature_file,
    )
    print(f"Saved cached features to: {feature_file}")

    return train_features, train_labels, test_features, test_labels


# -----------------------------------------------------------------------------
# Plotting (Figure 6 style)
# -----------------------------------------------------------------------------

def _ordered_error_curves(
    results_runs: List[Dict[float, float]],
    query_keys: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert {epsilon_attack -> accuracy} dicts into mean/std error curves."""
    errors_matrix: List[List[float]] = []
    for single_run_result in results_runs:
        ordered_errors = [100.0 - float(single_run_result.get(k, 0.0)) for k in query_keys]
        errors_matrix.append(ordered_errors)

    errors_array = np.asarray(errors_matrix, dtype=np.float64)
    mean_errors = np.mean(errors_array, axis=0)
    std_errors = np.std(errors_array, axis=0)
    return mean_errors, std_errors


def plot_figure6_panels(
    all_results_by_eps_ent: Dict[float, Dict[str, List[Dict[float, float]]]],
    perturbation_levels: List[float],
    epsilon_attack_values: List[float],
    tau: float,
    out_path: Path,
) -> None:
    """Create a multi-panel robustness plot (test error vs normalized Δ).

    Parameters
    ----------
    all_results_by_eps_ent:
        Mapping ε_ent -> {method_name -> [run_result_dict]}, where each run_result_dict
        maps attack radius ε_attack to accuracy (%). The ε_attack values should match
        `epsilon_attack_values`.
    perturbation_levels:
        Normalized perturbation Δ grid (x-axis).
    epsilon_attack_values:
        Absolute l2 PGD radii used in evaluation (includes 0.0).
    tau:
        τ from the paper; used only for titles.
    out_path:
        Output image path.
    """
    eps_ent_list = list(all_results_by_eps_ent.keys())
    eps_ent_list.sort(reverse=True)  # match typical panel ordering (0.2, 0.02, 0.002)

    query_keys = [0.0] + list(epsilon_attack_values[1:])

    fig, axes = plt.subplots(1, len(eps_ent_list), figsize=(12.5, 3.6), dpi=200, constrained_layout=True)
    if len(eps_ent_list) == 1:
        axes = [axes]

    methods_order = ["RGO", "WGF", "SAA", "Dual", "WRM", "WFR", "SVG", "ICNN", "PPA"]

    for ax, eps_ent in zip(axes, eps_ent_list):
        results_for_eps = all_results_by_eps_ent[eps_ent]

        for method in methods_order:
            if method not in results_for_eps:
                continue
            runs = results_for_eps[method]
            if not runs:
                continue

            mean_errors, std_errors = _ordered_error_curves(runs, query_keys)
            line = ax.plot(perturbation_levels, mean_errors, label=method)[0]
            if len(runs) > 1:
                ax.fill_between(
                    perturbation_levels,
                    mean_errors - std_errors,
                    mean_errors + std_errors,
                    color=line.get_color(),
                    alpha=0.20,
                )

        ax.set_xlabel("perturbation $\\Delta$")
        ax.set_ylabel("test error (%)")
        ax.set_title(f"$\\tau$={tau:g}, $\\epsilon$={eps_ent:g}")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.08))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved robustness plot: {out_path}")


def _json_safe_results(results: Dict[float, float]) -> Dict[str, float]:
    return {f"{k:.10g}": float(v) for k, v in results.items()}


@dataclass
class BBArmijoState:
    """Barzilai-Borwein step size + Armijo line search state."""

    alpha_min: float
    alpha_max: float
    alpha_prev: float
    ls_c: float
    ls_shrink: float
    ls_max_steps: int
    prev_params_vec: Optional[torch.Tensor] = None
    prev_grad_vec: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        alpha0: float = 0.0005,
        alpha_min: float = 1e-6,
        alpha_max: float = 1.0,
        ls_c: float = 0.1,
        ls_shrink: float = 0.5,
        ls_max_steps: int = 10,
    ) -> "BBArmijoState":
        alpha0 = float(max(alpha_min, min(alpha_max, alpha0)))
        return cls(
            alpha_min=float(max(alpha_min, 1e-12)),
            alpha_max=float(max(alpha_max, alpha_min)),
            alpha_prev=alpha0,
            ls_c=float(ls_c),
            ls_shrink=float(ls_shrink),
            ls_max_steps=int(max(ls_max_steps, 1)),
        )

    def propose(self, params_vec: torch.Tensor, grad_vec: torch.Tensor) -> float:
        """Propose a BB step size (before Armijo)."""
        if (
            self.prev_params_vec is None
            or self.prev_grad_vec is None
            or self.prev_params_vec.shape != params_vec.shape
            or self.prev_grad_vec.shape != grad_vec.shape
        ):
            alpha = self.alpha_prev
        else:
            s = params_vec - self.prev_params_vec
            y = grad_vec - self.prev_grad_vec
            denom = torch.dot(s, y)
            num = torch.dot(s, s)
            cond = torch.isfinite(denom) & (torch.abs(denom) > 1e-12)
            alpha_bb = torch.where(cond, num / denom, torch.tensor(self.alpha_prev, device=denom.device))
            alpha = float(alpha_bb.clamp(self.alpha_min, self.alpha_max).item())

        if not math.isfinite(alpha):
            alpha = self.alpha_prev
        alpha = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return alpha

    def update_history(self, params_vec: torch.Tensor, grad_vec: torch.Tensor, alpha: float) -> "BBArmijoState":
        alpha_clamped = max(self.alpha_min, min(self.alpha_max, float(alpha)))
        return BBArmijoState(
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_prev=alpha_clamped,
            ls_c=self.ls_c,
            ls_shrink=self.ls_shrink,
            ls_max_steps=self.ls_max_steps,
            prev_params_vec=params_vec.detach().clone(),
            prev_grad_vec=grad_vec.detach().clone(),
        )


def bb_armijo_step_params(
    params,
    f_params,
    bb_state: BBArmijoState,
) -> Tuple[List[torch.nn.Parameter], BBArmijoState, float, float]:
    """Single BB+Armijo gradient-ascent step on a parameter collection."""

    params = list(params)
    params_vec = nn_utils.parameters_to_vector(params).detach()

    f_val = f_params(True)
    grads = torch.autograd.grad(
        f_val,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads)]
    grad_vec = nn_utils.parameters_to_vector(grad_tensors)
    grad_norm = grad_vec.norm().item()

    alpha = bb_state.propose(params_vec, grad_vec)
    f_val_float = float(f_val.detach())
    g_dot_g = float(torch.dot(grad_vec, grad_vec).item())
    if g_dot_g == 0.0:
        return params, bb_state, f_val_float, grad_norm

    alpha_k = alpha
    for _ in range(bb_state.ls_max_steps):
        trial_vec = params_vec + alpha_k * grad_vec
        with torch.no_grad():
            nn_utils.vector_to_parameters(trial_vec, params)
            f_trial = float(f_params(False).detach())
        if f_trial >= f_val_float + bb_state.ls_c * alpha_k * g_dot_g:
            break
        alpha_k *= bb_state.ls_shrink

    final_vec = params_vec + alpha_k * grad_vec
    with torch.no_grad():
        nn_utils.vector_to_parameters(final_vec, params)

    f_new = f_params(True)
    grads_new = torch.autograd.grad(
        f_new,
        params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    grad_tensors_new = [g.detach() if g is not None else torch.zeros_like(p) for p, g in zip(params, grads_new)]
    grad_vec_new = nn_utils.parameters_to_vector(grad_tensors_new)
    new_state = bb_state.update_history(final_vec, grad_vec_new, alpha_k)
    return params, new_state, f_val_float, grad_vec_new.norm().item()


# -------------------------
# ICNN modules (principled init)
# -------------------------


def icnn_principled_moments(fan_in: int):
    """Principled log-normal moments for positive weights."""
    if fan_in <= 0:
        raise ValueError(f"ICNN fan-in must be positive; got {fan_in}.")
    denom_offset = 6.0 * (math.pi - 1.0)
    denom_slope = 3.0 * math.sqrt(3.0) + 2.0 * math.pi - 6.0
    denom = denom_offset + (fan_in - 1.0) * denom_slope
    mu_w = math.sqrt((6.0 * math.pi) / (fan_in * denom))
    sigma_w2 = 1.0 / float(fan_in)
    mu_b = math.sqrt((3.0 * fan_in) / denom)
    mu_w_sq = mu_w * mu_w
    log_var_plus_mean_sq = math.log(sigma_w2 + mu_w_sq)
    log_mean_sq = math.log(mu_w_sq)
    tilde_mu = log_mean_sq - 0.5 * log_var_plus_mean_sq
    tilde_sigma2 = max(log_var_plus_mean_sq - log_mean_sq, 1e-12)
    tilde_sigma = math.sqrt(tilde_sigma2)
    return mu_w, sigma_w2, mu_b, tilde_mu, tilde_sigma


class NonNegativeLinear(nn.Module):
    """Linear map with strictly non-negative weights via exp/softplus."""

    def __init__(self, in_features, out_features, bias=True, init_mode="principled"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_mode = init_mode.lower()
        if self.init_mode not in {"principled", "xavier"}:
            raise ValueError(f"Unsupported init_mode '{init_mode}' for NonNegativeLinear.")
        self.parametrization = "exp" if self.init_mode == "principled" else "softplus"

        self.weight_param = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_mode == "principled":
            _mu_w, _sigma_w2, mu_b, tilde_mu, tilde_sigma = icnn_principled_moments(self.in_features)
            with torch.no_grad():
                if tilde_sigma == 0.0:
                    self.weight_param.fill_(tilde_mu)
                else:
                    self.weight_param.normal_(mean=tilde_mu, std=tilde_sigma)
                if self.bias is not None:
                    self.bias.fill_(mu_b)
        else:
            nn.init.xavier_uniform_(self.weight_param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.parametrization == "exp":
            weight = torch.exp(self.weight_param)
        else:
            weight = F.softplus(self.weight_param)
        y = x.matmul(weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class InputConvexPotential(nn.Module):
    """Dense ICNN potential ψ(z) that is convex in z."""

    def __init__(
        self,
        input_dim=1,
        hidden_sizes=(64, 64),
        activation="softplus",
        strong_convexity=1.0,
        nonneg_init="principled",
        softplus_beta=20.0,
    ):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.strong_convexity = strong_convexity
        self.softplus_beta = softplus_beta

        self.z_linears = nn.ModuleList()
        self.h_linears = nn.ModuleList()
        for i, width in enumerate(hidden_sizes):
            self.z_linears.append(nn.Linear(input_dim, width, bias=True))
            if i == 0:
                self.h_linears.append(None)
            else:
                self.h_linears.append(
                    NonNegativeLinear(
                        hidden_sizes[i - 1],
                        width,
                        bias=False,
                        init_mode=nonneg_init,
                    )
                )

        self.hidden_output = NonNegativeLinear(
            hidden_sizes[-1],
            1,
            bias=True,
            init_mode=nonneg_init,
        )
        self.input_skip = nn.Linear(input_dim, 1, bias=True)

    def _activation(self):
        act_name = self.activation.lower()
        if act_name == "relu":
            return F.relu
        if act_name == "softplus":
            beta = float(self.softplus_beta)
            return lambda u: F.softplus(beta * u) / beta
        raise ValueError(f"Unsupported ICNN activation '{self.activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(x.size(0), -1)
        act = self._activation()

        h = act(self.z_linears[0](z))
        for k in range(1, len(self.z_linears)):
            z_term = self.z_linears[k](z)
            h_term = self.h_linears[k](h) if self.h_linears[k] is not None else 0.0
            h = act(z_term + h_term)

        quadratic = 0.5 * self.strong_convexity * (z**2).sum(dim=1, keepdim=True)
        out = quadratic + self.input_skip(z) + self.hidden_output(h)
        return out.squeeze(-1)


def initialize_icnn_identity(icnn: InputConvexPotential, strong_convexity: float = 1.0) -> None:
    """
    Identity init for the ICNN transport:
      φ(x) = 0.5 * strong_convexity * ||x||^2 + const  =>  T(x)=∇φ(x)=strong_convexity * x.
    For strong_convexity=1 this is exactly the identity map T(x)=x.

    We enforce this by:
      - setting all z_linears (x→hidden) weights/biases to 0
      - setting input_skip weights/biases to 0
      - (optionally) setting hidden_output bias to 0 (does not affect T)
    """
    if not hasattr(icnn, "z_linears") or not hasattr(icnn, "input_skip"):
        raise TypeError("initialize_icnn_identity expects an InputConvexPotential-like module.")

    with torch.no_grad():
        icnn.strong_convexity = float(strong_convexity)

        # Keep the ICNN part trainable but x-independent at init:
        #   - z_linears weights -> 0 ensures h(x) does not depend on x
        #   - biases kept as-is (from their default init) since they do not affect ∇_x
        for lin in icnn.z_linears:
            lin.weight.zero_()

        # input_skip contributes a constant vector to ∇_x; set it to 0 so T(x)=strong_convexity*x.
        icnn.input_skip.weight.zero_()
        if icnn.input_skip.bias is not None:
            icnn.input_skip.bias.zero_()

        if getattr(icnn, "hidden_output", None) is not None and getattr(icnn.hidden_output, "bias", None) is not None:
            icnn.hidden_output.bias.zero_()


def T_omega(x: torch.Tensor, psi_omega: nn.Module, create_graph: bool) -> torch.Tensor:
    """Transport map T_ω(x) = ∇_x ψ_ω(x)."""

    x_in = x.clone().detach().requires_grad_(True)
    with torch.set_grad_enabled(True):
        psi_val = psi_omega(x_in)
        grad = torch.autograd.grad(psi_val.sum(), x_in, create_graph=create_graph)[0]
    return grad.view_as(x)



# -----------------------------------------------------------------------------
# ICNN-DRO competitor (transport map adversary)
# -----------------------------------------------------------------------------

class ICNNDRO(BaseLinearDRO):
    """ICNN-DRO competitor for adversarial training.

    Paper notation:
      - Classifier parameters: **B**
      - Features: **x**
      - Convex potential: **ψ_ω**
      - Transport map: **T_ω(x) = ∇_x ψ_ω(x)**

    Training scheme (high-level):
      - Outer loop: SGD on B (logistic regression)
      - Inner loop: Ascent on ω using BB+Armijo (verbatim from source)

    Objective (inner, ascent on ω) on a minibatch:
      maximize_ω  E[ CE(B(T_ω(x)), y) - λ ||T_ω(x) - x||_2^2 ].

    Notes:
      - This corresponds to a Wasserstein-type DRO objective (no entropy term).
      - ψ_ω is parameterized by InputConvexPotential (ICNN); `icnn_init_mode="identity"` makes T_ω(x)=x at start.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        fit_intercept: bool = True,
        lambda_param: float = 10.0,
        icnn_hidden: Tuple[int, ...] = (512, 512, 512, 256, 256, 128, 128, 64),
        icnn_strong_convexity: float = 1.0,
        icnn_softplus_beta: float = 20.0,
        icnn_init_mode: str = "identity",
        lr_B: float = 5e-3,
        weight_decay_B: float = 0.0,
        omega_steps_per_batch: int = 1,
        bb_alpha0: float = 1e-1,
        bb_alpha_min: float = 1e-6,
        bb_alpha_max: float = 10.0,
        bb_ls_c: float = 1e-4,
        bb_ls_shrink: float = 0.5,
        bb_ls_max_steps: int = 10,
        max_itr: int = 10,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        super().__init__(input_dim, num_classes, fit_intercept)

        self.lambda_param = float(lambda_param)
        self.lr_B = float(lr_B)
        self.weight_decay_B = float(weight_decay_B)
        self.omega_steps_per_batch = int(omega_steps_per_batch)
        self.max_itr = int(max_itr)
        self.batch_size = int(batch_size)

        # Classifier B (logistic regression)
        self.model = LinearModel(self.input_dim, output_dim=self.num_classes, bias=self.fit_intercept).to(self.device)

        # Potential ψ_ω (ICNN) --- verbatim module imported below (InputConvexPotential)
        self.psi_omega = InputConvexPotential(
            input_dim=self.input_dim,
            hidden_sizes=icnn_hidden,
            strong_convexity=icnn_strong_convexity,
            softplus_beta=icnn_softplus_beta,
            nonneg_init="principled",
        ).to(self.device)

        init_mode = str(icnn_init_mode).lower()
        if init_mode == "identity":
            initialize_icnn_identity(self.psi_omega, strong_convexity=1.0)
        elif init_mode == "principled":
            pass
        else:
            raise ValueError(f"Unsupported icnn_init_mode '{icnn_init_mode}'. Use 'identity' or 'principled'.")

        # BB+Armijo state for ω-ascent (verbatim class/function imported below)
        self.bb_state = BBArmijoState.create(
            alpha0=bb_alpha0,
            alpha_min=bb_alpha_min,
            alpha_max=bb_alpha_max,
            ls_c=bb_ls_c,
            ls_shrink=bb_ls_shrink,
            ls_max_steps=bb_ls_max_steps,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_dir: str = "checkpoints", run_id: int = 0) -> List[float]:
        X, y = self._validate_inputs(X, y)
        dataloader = self._create_dataloader(X, y, batch_size=self.batch_size)

        optimizer_B = optim.SGD(self.model.parameters(), lr=self.lr_B, momentum=0.9, weight_decay=self.weight_decay_B)

        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_history: List[float] = []

        for epoch in range(self.max_itr):
            epoch_loss_record = 0.0
            pbar = tqdm(dataloader, desc=f"Run {run_id+1} Training ICNN Epoch {epoch+1}/{self.max_itr}", leave=False)
            for x_original_batch, y_original_batch in pbar:
                x_original_batch_dev = x_original_batch.to(self.device)
                y_original_batch_dev = y_original_batch.to(self.device)

                # -------------------------
                # Inner loop: ω-ascent via BB+Armijo
                # -------------------------
                self.model.eval()
                self.psi_omega.train()

                def omega_objective(create_graph: bool) -> torch.Tensor:
                    # Transport map T_ω(x) = ∇ψ_ω(x)
                    x_adv = T_omega(x_original_batch_dev, self.psi_omega, create_graph=create_graph)

                    logits = self.model(x_adv)
                    ce = nn.CrossEntropyLoss(reduction="none")(logits, y_original_batch_dev)

                    transport_cost = torch.sum((x_adv - x_original_batch_dev) ** 2, dim=1)
                    return (ce - self.lambda_param * transport_cost).mean()

                for _ in range(self.omega_steps_per_batch):
                    _, self.bb_state, f_val, grad_norm = bb_armijo_step_params(
                        self.psi_omega.parameters(),
                        omega_objective,
                        self.bb_state,
                    )

                # -------------------------
                # Outer loop: B-update (SGD) on adversarial features
                # -------------------------
                self.model.train()
                self.psi_omega.eval()

                with torch.no_grad():
                    x_adv_det = T_omega(x_original_batch_dev, self.psi_omega, create_graph=False)

                logits_adv = self.model(x_adv_det)
                loss_B = nn.CrossEntropyLoss()(logits_adv, y_original_batch_dev)

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
                    "psi_omega_state_dict": self.psi_omega.state_dict(),
                },
                f"{checkpoint_dir}/ICNN_DRO_run{run_id}_epoch_{epoch+1}.pth",
            )

        return loss_history


# -----------------------------------------------------------------------------
# Main experiment driver (Figure 6 replication on CIFAR-10 features)
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 6 adversarial training on CIFAR-10 features + ICNN competitor.")

    # I/O
    parser.add_argument("--data_dir", type=str, default="./data", help="Where CIFAR-10 is/will be downloaded.")
    parser.add_argument("--out_dir", type=str, default="./outputs_adversarial_multiclass_icnn", help="Output directory.")
    parser.add_argument("--feature_cache", type=str, default="cifar10_resnet50_features.pth", help="Feature cache filename (saved under out_dir unless absolute path).")
    parser.add_argument("--force_reextract", action="store_true", help="Re-extract CIFAR-10 features even if cache exists.")

    # Figure 6 hyperparams
    parser.add_argument("--tau", type=float, default=0.05, help="Figure 6: tau. We use lambda=1/(2tau).")
    parser.add_argument("--eps_ent", type=float, nargs="+", default=[0.2, 0.02, 0.002], help="Figure 6: entropy regularization eps values.")
    parser.add_argument("--epochs", type=int, default=10, help="Figure 6: train epochs.")
    parser.add_argument("--inner_lr", type=float, default=0.01, help="Figure 6: inner-loop stepsize (methods with inner loop).")
    parser.add_argument("--inner_steps", type=int, default=100, help="Figure 6: inner-loop iterations (methods with inner loop).")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of particles per data point for sampler methods.")
    parser.add_argument("--sample_level", type=int, default=5, help="Dual SDRO randomized truncation level.")

    # Optimization
    parser.add_argument("--dro_lr", type=float, default=5e-3, help="Outer learning rate for DRO baselines (reference code uses 5e-3).")
    parser.add_argument("--saa_lr", type=float, default=1e-3, help="ERM/SAA learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training DRO + SAA.")
    parser.add_argument("--feature_batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers (feature extraction + eval).")
    parser.add_argument("--seed", type=int, default=2022)

    # Evaluation (PGD)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument(
        "--perturbation_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.008, 0.016, 0.024, 0.032, 0.04, 0.048, 0.056, 0.064, 0.072, 0.08],
        help="Normalized perturbation Δ grid (Figure 6 uses 0..0.08).",
    )

    # Method selection
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["SAA", "Dual", "WRM", "WGF", "WFR", "SVG", "RGO", "ICNN", "PPA"],
        help="Which methods to run. Subset of: SAA Dual WRM WGF WFR SVG RGO ICNN PPA",
    )

    # PPA knobs (Projected Particle Ascent)
    parser.add_argument("--ppa_num_rounds", type=int, default=5, help="PPA: total rounds including round 0.")
    parser.add_argument("--ppa_min_rounds", type=int, default=2, help="PPA: minimum refinement rounds before early stopping.")
    parser.add_argument("--ppa_refine_steps", type=int, default=15, help="PPA: constant-lr ascent steps per refinement round.")
    parser.add_argument("--ppa_refine_lr", type=float, default=5e-3, help="PPA: constant lr for refinement rounds.")
    parser.add_argument("--ppa_delta_rtol", type=float, default=1e-4, help="PPA: relative Δ threshold for adaptive stopping.")

    # ICNN knobs (competitor)
    parser.add_argument("--icnn_hidden", type=int, nargs="+", default=[1024, 512, 512, 256, 128, 64])
    parser.add_argument("--icnn_strong_convexity", type=float, default=1.0)
    parser.add_argument("--icnn_softplus_beta", type=float, default=20.0)
    parser.add_argument("--icnn_lr_B", type=float, default=0.0005)
    parser.add_argument("--icnn_weight_decay_B", type=float, default=0.0)
    parser.add_argument(
        "--icnn_init_mode",
        type=str,
        default="identity",
        choices=["identity", "principled"],
        help="ICNN init for ψ_ω: 'identity' makes T_ω(x)=x at start; 'principled' leaves default init.",
    )
    parser.add_argument("--icnn_omega_steps_per_batch", type=int, default=10)
    parser.add_argument("--icnn_bb_alpha0", type=float, default=0.0005)
    parser.add_argument("--icnn_bb_alpha_min", type=float, default=1e-6)
    parser.add_argument("--icnn_bb_alpha_max", type=float, default=1.0)
    parser.add_argument("--icnn_bb_ls_c", type=float, default=0.1)
    parser.add_argument("--icnn_bb_ls_shrink", type=float, default=0.5)
    parser.add_argument("--icnn_bb_ls_max_steps", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert τ -> λ (used in the reference code as `lambda_param`)
    tau = float(args.tau)
    lambda_param = 1.0 / (2.0 * tau)

    # -------------------------
    # Load / extract features
    # -------------------------
    feature_cache_path = Path(args.feature_cache)
    if not feature_cache_path.is_absolute():
        feature_cache_path = out_dir / feature_cache_path

    train_features, train_labels, test_features, test_labels = load_or_extract_cifar10_resnet50_features(
        feature_cache_path,
        data_dir=args.data_dir,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        device=DEVICE,
        force_reextract=args.force_reextract,
    )

    train_features_np = train_features.numpy()
    train_labels_np = train_labels.numpy()
    test_features_np = test_features.numpy()
    test_labels_np = test_labels.numpy()

    input_dim = int(train_features_np.shape[1])
    num_classes = 10

    print(f"\nFeature dim: {input_dim} (expected 512)")
    print(f"Training size: {train_features_np.shape[0]}, Test size: {test_features_np.shape[0]}")
    print(f"tau={tau:g} -> lambda={lambda_param:g}")

    # PGD evaluation radii: ε_attack = Δ * E[||x||_2]
    avg_feature_norm = float(np.mean(np.linalg.norm(test_features_np, axis=1)))
    perturbation_levels = list(args.perturbation_levels)
    epsilon_attack_values = [float(level) * avg_feature_norm for level in perturbation_levels]

    print(f"Average test feature ||x||_2: {avg_feature_norm:.4f}")

    # Shared (ε_ent-independent) models
    shared_models: Dict[str, BaseLinearDRO] = {}

    # SAA / ERM
    if "SAA" in args.methods:
        print("\n--- Training SAA / ERM ---")
        saa_wrapper = BaseLinearDRO(input_dim, num_classes, True)
        saa_wrapper.model = LogisticRegression(input_dim, num_classes).to(DEVICE)
        feature_train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_saa(
            saa_wrapper.model,
            feature_train_loader,
            epochs=args.epochs,
            learning_rate=args.saa_lr,
            checkpoint_dir=str(checkpoint_dir),
            run_id=0,
        )
        shared_models["SAA"] = saa_wrapper

    # WRM
    if "WRM" in args.methods:
        print("\n--- Training WRM ---")
        wrm_model = WRM(
            input_dim,
            num_classes,
            device=DEVICE.type,
            lambda_param=lambda_param,
            max_itr=args.epochs,
            learning_rate=args.dro_lr,
            batch_size=args.batch_size,
            inner_lr=args.inner_lr,
            inner_itr=args.inner_steps,
        )
        wrm_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
        shared_models["WRM"] = wrm_model

    # ICNN-DRO
    if "ICNN" in args.methods:
        print("\n--- Training ICNN-DRO (transport map) ---")
        icnn_model = ICNNDRO(
            input_dim,
            num_classes,
            device=DEVICE.type,
            lambda_param=lambda_param,
            icnn_hidden=tuple(args.icnn_hidden),
            icnn_strong_convexity=args.icnn_strong_convexity,
            icnn_softplus_beta=args.icnn_softplus_beta,
            icnn_init_mode=args.icnn_init_mode,
            lr_B=args.icnn_lr_B,
            weight_decay_B=args.icnn_weight_decay_B,
            omega_steps_per_batch=args.icnn_omega_steps_per_batch,
            bb_alpha0=args.icnn_bb_alpha0,
            bb_alpha_min=args.icnn_bb_alpha_min,
            bb_alpha_max=args.icnn_bb_alpha_max,
            bb_ls_c=args.icnn_bb_ls_c,
            bb_ls_shrink=args.icnn_bb_ls_shrink,
            bb_ls_max_steps=args.icnn_bb_ls_max_steps,
            max_itr=args.epochs,
            batch_size=args.batch_size,
        )
        icnn_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
        shared_models["ICNN"] = icnn_model

    # PPA (Projected Particle Ascent)
    if "PPA" in args.methods:
        print("\n--- Training PPA (Projected Particle Ascent) ---")
        ppa_model = PPA(
            input_dim,
            num_classes,
            device=DEVICE.type,
            lambda_param=lambda_param,
            max_itr=args.epochs,
            learning_rate=args.dro_lr,
            batch_size=args.batch_size,
            inner_lr=args.inner_lr,
            inner_itr=args.inner_steps,
            ppa_num_rounds=args.ppa_num_rounds,
            ppa_min_rounds=args.ppa_min_rounds,
            ppa_refine_steps=args.ppa_refine_steps,
            ppa_refine_lr=args.ppa_refine_lr,
            ppa_delta_rtol=args.ppa_delta_rtol,
        )
        ppa_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
        shared_models["PPA"] = ppa_model

    # -------------------------
    # ε_ent sweep (Figure 6 panels)
    # -------------------------
    all_results_by_eps_ent: Dict[float, Dict[str, List[Dict[float, float]]]] = {}

    for eps_ent in args.eps_ent:
        eps_ent = float(eps_ent)
        print(f"\n{'='*20} ε_ent = {eps_ent:g} {'='*20}")

        models_to_evaluate: Dict[str, BaseLinearDRO] = dict(shared_models)

        # Dual
        if "Dual" in args.methods:
            print("\n--- Training Dual SDRO ---")
            dual_model = SDRO_Dual(
                input_dim,
                num_classes,
                device=DEVICE.type,
                lambda_param=lambda_param,
                epsilon=eps_ent,
                max_itr=args.epochs,
                learning_rate=args.dro_lr,
                batch_size=args.batch_size,
                sample_level=args.sample_level,
            )
            dual_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
            models_to_evaluate["Dual"] = dual_model

        # WGF
        if "WGF" in args.methods:
            print("\n--- Training WGF sampler ---")
            wgf_model = SDRO_WGF(
                input_dim,
                num_classes,
                device=DEVICE.type,
                lambda_param=lambda_param,
                epsilon=eps_ent,
                max_itr=args.epochs,
                learning_rate=args.dro_lr,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                inner_lr=args.inner_lr,
                inner_itr=args.inner_steps,
            )
            wgf_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
            models_to_evaluate["WGF"] = wgf_model

        # WFR
        if "WFR" in args.methods:
            print("\n--- Training WFR sampler ---")
            wfr_model = SDRO_WFR(
                input_dim,
                num_classes,
                device=DEVICE.type,
                lambda_param=lambda_param,
                epsilon=eps_ent,
                max_itr=args.epochs,
                learning_rate=args.dro_lr,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                inner_lr=args.inner_lr,
                inner_itr=args.inner_steps,
            )
            wfr_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
            models_to_evaluate["WFR"] = wfr_model

        # SVG
        if "SVG" in args.methods:
            print("\n--- Training SVG sampler ---")
            svg_model = SDRO_SVG(
                input_dim,
                num_classes,
                device=DEVICE.type,
                lambda_param=lambda_param,
                epsilon=eps_ent,
                max_itr=args.epochs,
                learning_rate=args.dro_lr,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                inner_lr=args.inner_lr,
                inner_itr=args.inner_steps,
            )
            svg_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
            models_to_evaluate["SVG"] = svg_model

        # RGO
        if "RGO" in args.methods:
            print("\n--- Training RGO ---")
            rgo_model = SDRO_RGO(
                input_dim,
                num_classes,
                device=DEVICE.type,
                lambda_param=lambda_param,
                epsilon=eps_ent,
                max_itr=args.epochs,
                learning_rate=args.dro_lr,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                rgo_inner_lr=args.inner_lr,
                rgo_inner_steps=args.inner_steps,
            )
            rgo_model.fit(train_features_np, train_labels_np, checkpoint_dir=str(checkpoint_dir), run_id=0)
            models_to_evaluate["RGO"] = rgo_model

        # -------------------------
        # Evaluation (l2 PGD on features)
        # -------------------------
        results_for_eps: Dict[str, List[Dict[float, float]]] = {name: [] for name in models_to_evaluate.keys()}

        for name, model_obj in models_to_evaluate.items():
            print(f"\n--- Evaluating {name} (ε_ent={eps_ent:g}) ---")
            res = evaluate_model(model_obj, test_features_np, test_labels_np, pgd_attack, epsilon_attack_values[1:])
            results_for_eps[name].append(res)

        all_results_by_eps_ent[eps_ent] = results_for_eps

        # Save JSON for this ε_ent
        json_path = out_dir / f"results_tau={tau:g}_epsent={eps_ent:g}.json"
        json_payload = {
            "tau": tau,
            "lambda": lambda_param,
            "eps_ent": eps_ent,
            "avg_feature_norm": avg_feature_norm,
            "perturbation_levels": perturbation_levels,
            "epsilon_attack_values": epsilon_attack_values,
            "results": {
                method: [_json_safe_results(run_dict) for run_dict in run_list]
                for method, run_list in results_for_eps.items()
            },
        }
        json_path.write_text(json.dumps(json_payload, indent=2))
        print(f"Saved results: {json_path}")

    # -------------------------
    # Plot combined Figure-6 style robustness panels
    # -------------------------
    plot_path = out_dir / f"figure6_tau={tau:g}_robustness.png"
    plot_figure6_panels(
        all_results_by_eps_ent,
        perturbation_levels=perturbation_levels,
        epsilon_attack_values=epsilon_attack_values,
        tau=tau,
        out_path=plot_path,
    )


if __name__ == "__main__":
    main()
