"""DRO sampler baselines: Dual / WGF / WFR / SVG / RGO."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import rbf_kernel_batched_images, repeat_particles


def dual_dro_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_reg: float,
    epsilon: float,
    sample_level: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised log-sum-exp dual objective.

    Samples m = 2^L noisy replicas with L ~ geometric on {0,...,sample_level}.
    """
    levels = np.arange(sample_level + 1)
    numerators = 2.0 ** (-levels)
    probabilities = numerators / (2.0 - 2.0 ** (-sample_level))
    sampled_level = int(np.random.choice(levels, p=probabilities))
    m = 2 ** sampled_level

    repeated_x = x.repeat_interleave(m, dim=0)
    repeated_y = y.repeat_interleave(m, dim=0)
    noisy_x = (
        repeated_x + torch.randn_like(repeated_x) * math.sqrt(epsilon)
    ).clamp(0.0, 1.0)

    logits = model(noisy_x)
    residuals = F.cross_entropy(logits, repeated_y, reduction="none") / max(
        lambda_reg * epsilon, 1e-8
    )
    residual_matrix = residuals.view(-1, m).T
    loss = torch.mean(
        torch.logsumexp(residual_matrix, dim=0) - math.log(m)
    ) * (lambda_reg * epsilon)
    return loss, noisy_x.detach(), repeated_y.detach()


def wgf_sampler(
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    model: nn.Module,
    lambda_reg: float,
    epsilon: float,
    num_samples: int,
    inner_steps: int,
    inner_lr: float,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    """Weighted gradient-flow / Langevin sampler."""
    x_clone = repeat_particles(x_orig.detach(), num_samples)
    y_repeated = y_orig.repeat_interleave(num_samples, dim=0)
    x_anchor = repeat_particles(x_orig.detach(), num_samples)

    for _ in range(inner_steps):
        x_clone.requires_grad_(True)
        loss_values = F.cross_entropy(model(x_clone), y_repeated, reduction="none")
        grads, = torch.autograd.grad(loss_values.sum(), x_clone)
        mean = x_clone.detach() + inner_lr * (
            grads - 2.0 * lambda_reg * (x_clone.detach() - x_anchor)
        )
        std = math.sqrt(max(2.0 * inner_lr * lambda_reg * epsilon, 1e-12))
        x_clone = mean + torch.randn_like(mean) * std
        if clamp is not None:
            lo, hi = clamp
            x_clone = x_clone.clamp(lo, hi)
    return x_clone.detach()


def wfr_sampler(
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    model: nn.Module,
    lambda_reg: float,
    epsilon: float,
    num_samples: int,
    inner_steps: int,
    inner_lr: float,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finite-particle WFR sampler with adaptive particle weights."""
    batch_size = x_orig.size(0)
    weights = torch.full(
        (batch_size, num_samples), 1.0 / num_samples, device=x_orig.device
    )
    x_clone = repeat_particles(x_orig.detach(), num_samples)
    x_anchor = repeat_particles(x_orig.detach(), num_samples)
    y_repeated = y_orig.repeat_interleave(num_samples, dim=0)

    weight_exponent = 1.0 - lambda_reg * epsilon * inner_lr

    for _ in range(inner_steps):
        x_clone.requires_grad_(True)
        with torch.enable_grad():
            loss_values = F.cross_entropy(
                model(x_clone), y_repeated, reduction="none"
            )
            grads, = torch.autograd.grad(loss_values.sum(), x_clone)
            mean = x_clone.detach() + inner_lr * (
                grads - 2.0 * lambda_reg * (x_clone.detach() - x_anchor)
            )
            std = math.sqrt(max(2.0 * inner_lr * lambda_reg * epsilon, 1e-12))
            x_clone = mean + torch.randn_like(mean) * std
        if clamp is not None:
            lo, hi = clamp
            x_clone = x_clone.clamp(lo, hi)

        with torch.no_grad():
            current_loss = F.cross_entropy(
                model(x_clone), y_repeated, reduction="none"
            ).view(batch_size, num_samples)
            dist_sq = (
                ((x_clone - x_anchor) ** 2)
                .view(batch_size, num_samples, -1)
                .sum(dim=2)
            )
            energy_term = current_loss - 2.0 * lambda_reg * dist_sq
            weights = weights.pow(weight_exponent) * torch.exp(
                torch.clamp(energy_term * inner_lr, min=-40.0, max=40.0)
            )
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

            low_weight_mask = weights < 1e-4
            rows_with_low_weights = torch.any(low_weight_mask, dim=1)
            if torch.any(rows_with_low_weights):
                x_reshaped = x_clone.view(
                    batch_size, num_samples, *x_orig.shape[1:]
                )
                max_weight_vals, max_weight_indices = torch.max(
                    weights, dim=1, keepdim=True
                )
                gather_idx = max_weight_indices.view(
                    batch_size, 1, *([1] * len(x_orig.shape[1:]))
                )
                gather_idx = gather_idx.expand(
                    batch_size, 1, *x_orig.shape[1:]
                )
                highest_weight_point = torch.gather(x_reshaped, 1, gather_idx)

                low_weights_sum = torch.sum(
                    weights * low_weight_mask, dim=1, keepdim=True
                )
                num_low_weights = torch.sum(
                    low_weight_mask, dim=1, keepdim=True, dtype=weights.dtype
                )
                avg_weight = (max_weight_vals + low_weights_sum) / (
                    num_low_weights + 1.0 + 1e-9
                )
                avg_weight_expanded = avg_weight.expand_as(weights)
                max_weight_mask = torch.zeros_like(
                    weights, dtype=torch.bool
                ).scatter_(1, max_weight_indices, True)
                update_mask = (
                    low_weight_mask | max_weight_mask
                ) & rows_with_low_weights.unsqueeze(1)
                weights = torch.where(
                    update_mask, avg_weight_expanded, weights
                )

                replacement = highest_weight_point.expand(
                    batch_size, num_samples, *x_orig.shape[1:]
                )
                x_update_mask = low_weight_mask.view(
                    batch_size, num_samples, *([1] * len(x_orig.shape[1:]))
                )
                rows_mask = rows_with_low_weights.view(
                    batch_size, 1, *([1] * len(x_orig.shape[1:]))
                )
                x_reshaped = torch.where(
                    x_update_mask & rows_mask, replacement, x_reshaped
                )
                x_clone = x_reshaped.view_as(x_clone)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

    return x_clone.detach(), weights.detach()


def svg_sampler(
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    model: nn.Module,
    lambda_reg: float,
    epsilon: float,
    num_samples: int,
    inner_steps: int,
    inner_lr: float,
    adagrad_hist_decay: float,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    """Stein variational gradient sampler."""
    if x_orig.size(0) == 0:
        return torch.empty_like(x_orig)

    B = x_orig.size(0)
    S = num_samples
    x_anchor = x_orig.unsqueeze(1).repeat(1, S, 1, 1, 1)
    particles = x_anchor.clone().detach() + 0.1 * torch.randn_like(x_anchor)
    if clamp is not None:
        lo, hi = clamp
        particles = particles.clamp(lo, hi)
    hist_grad = torch.zeros_like(particles)
    y_repeated = y_orig.repeat_interleave(S)

    for _ in range(inner_steps):
        x_tensor = (
            particles.view(B * S, *x_orig.shape[1:])
            .detach()
            .requires_grad_(True)
        )
        x_anchor_flat = x_anchor.view(B * S, *x_orig.shape[1:])
        neg_log_likelihood = F.cross_entropy(
            model(x_tensor), y_repeated, reduction="sum"
        )
        grad_log_py_x, = torch.autograd.grad(neg_log_likelihood, x_tensor)
        grad_log_px = -2.0 * lambda_reg * (x_tensor - x_anchor_flat)
        total_grad_flat = (grad_log_py_x + grad_log_px) / max(
            lambda_reg * epsilon, 1e-8
        )
        total_grad = total_grad_flat.view(B, S, *x_orig.shape[1:])

        K, grad_K_x = rbf_kernel_batched_images(particles)
        total_grad_vec = total_grad.view(B, S, -1)
        K_grad_prod = torch.bmm(K, total_grad_vec).view_as(total_grad)
        sum_grad_K = grad_K_x.sum(dim=2)
        svg_grad = (K_grad_prod + sum_grad_K) / S

        with torch.no_grad():
            hist_grad = adagrad_hist_decay * hist_grad + (
                1.0 - adagrad_hist_decay
            ) * (svg_grad ** 2)
            adj_grad = svg_grad / (1e-6 + torch.sqrt(hist_grad))
            particles = particles + inner_lr * adj_grad
            if clamp is not None:
                lo, hi = clamp
                particles = particles.clamp(lo, hi)

    return particles.view(B * S, *x_orig.shape[1:]).detach()


def rgo_sampler(
    x_orig: torch.Tensor,
    y_orig: torch.Tensor,
    model: nn.Module,
    lambda_reg: float,
    epsilon: float,
    num_samples: int,
    inner_steps: int,
    inner_lr: float,
    max_trials: int,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> torch.Tensor:
    """Vectorised RGO sampler."""
    batch_size = x_orig.size(0)
    x_anchor = x_orig.detach()
    x_pert = x_anchor.clone()

    for _ in range(inner_steps):
        x_pert.requires_grad_(True)
        per_sample_loss = F.cross_entropy(model(x_pert), y_orig, reduction="none")
        grads, = torch.autograd.grad(per_sample_loss.sum(), x_pert)
        x_pert = x_pert.detach() - inner_lr * (
            -grads / max(lambda_reg, 1e-8)
            + 2.0 * (x_pert.detach() - x_anchor)
        )
        if clamp is not None:
            lo, hi = clamp
            x_pert = x_pert.clamp(lo, hi)

    x_opt_star = x_pert.detach()
    if epsilon <= 1e-12:
        return x_opt_star.repeat_interleave(num_samples, dim=0)

    var_rgo = epsilon
    std_rgo = math.sqrt(var_rgo)
    f_model_loss_opt = F.cross_entropy(model(x_opt_star), y_orig, reduction="none")
    norm_sq_opt = ((x_opt_star - x_anchor) ** 2).view(batch_size, -1).sum(dim=1)
    f_L_opt = (-f_model_loss_opt / max(lambda_reg * epsilon, 1e-8)) + (
        norm_sq_opt / epsilon
    )

    x_opt_star_expand = x_opt_star.unsqueeze(1)
    x_anchor_expand = x_anchor.unsqueeze(1)
    f_L_opt_expand = f_L_opt.unsqueeze(1)

    final_accepted = torch.zeros(
        (batch_size, num_samples, *x_orig.shape[1:]), device=x_orig.device
    )
    active_flags = torch.ones(
        (batch_size, num_samples), dtype=torch.bool, device=x_orig.device
    )

    for _ in range(max_trials):
        if not active_flags.any():
            break
        proposals = torch.randn_like(final_accepted) * std_rgo
        candidates = x_opt_star_expand + proposals
        if clamp is not None:
            lo, hi = clamp
            candidates = candidates.clamp(lo, hi)
        candidates_flat = candidates.view(
            batch_size * num_samples, *x_orig.shape[1:]
        )
        y_repeated = y_orig.repeat_interleave(num_samples, dim=0)

        f_model_loss_candidates = F.cross_entropy(
            model(candidates_flat), y_repeated, reduction="none"
        ).view(batch_size, num_samples)
        norm_sq_candidates = (
            ((candidates - x_anchor_expand) ** 2)
            .view(batch_size, num_samples, -1)
            .sum(dim=2)
        )
        f_L_candidates = (
            -f_model_loss_candidates / max(lambda_reg * epsilon, 1e-8)
        ) + (norm_sq_candidates / epsilon)
        diff_cand_opt_norm_sq = (
            proposals.view(batch_size, num_samples, -1).pow(2).sum(dim=2)
        )
        exponent_term3 = diff_cand_opt_norm_sq / (2.0 * var_rgo)
        acceptance_probs = torch.exp(
            torch.clamp(
                -f_L_candidates + f_L_opt_expand + exponent_term3, max=10.0
            )
        )
        newly_accepted = (
            torch.rand_like(acceptance_probs) < acceptance_probs
        ) & active_flags
        final_accepted[newly_accepted] = proposals[newly_accepted]
        active_flags[newly_accepted] = False

    sampled = x_opt_star_expand + final_accepted
    if clamp is not None:
        lo, hi = clamp
        sampled = sampled.clamp(lo, hi)
    return sampled.view(batch_size * num_samples, *x_orig.shape[1:]).detach()
