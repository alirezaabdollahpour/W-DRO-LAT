"""Adaptive l2-PGD attack on feature-space perturbations."""
from __future__ import annotations

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float | None,
    num_iter: int,
    device: torch.device,
    restarts: int = 5,
) -> torch.Tensor:
    """Multi-restart adaptive l2-PGD attack on feature perturbations.

    The optimisation variable is the current feature-space perturbation
    ``delta``.  Each ascent step updates ``delta`` using a gradient normalised
    in the l2 norm and then projects the current ``delta`` back onto the
    radius-``epsilon`` l2 ball.  The returned adversarial features are selected
    per sample as the highest-loss point found across all restarts.
    """
    original_features = features.clone().detach().to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    batch_size = original_features.shape[0]
    flat_dim = original_features[0].numel()
    reshape_dims = [batch_size] + [1] * (original_features.dim() - 1)

    if epsilon <= 0.0 or num_iter <= 0:
        return original_features

    step_size = float(alpha) if alpha is not None else float(epsilon) / max(1, num_iter // 2)
    best_adv = original_features
    best_loss = torch.full((batch_size,), -torch.inf, device=device)

    for restart_idx in range(max(1, int(restarts))):
        if restart_idx == 0:
            delta = torch.zeros_like(original_features)
        else:
            noise = torch.randn_like(original_features)
            noise_norm = torch.linalg.norm(noise.view(batch_size, -1), dim=1).view(
                *reshape_dims
            ).clamp_min(1e-12)
            radii = torch.rand(batch_size, device=device).pow(1.0 / flat_dim).view(
                *reshape_dims
            )
            delta = epsilon * radii * noise / noise_norm

        for _ in range(num_iter):
            delta = delta.detach().requires_grad_(True)
            adv_features = original_features + delta
            losses = criterion(model(adv_features), labels)
            grad = torch.autograd.grad(losses.sum(), delta, create_graph=False)[0]

            with torch.no_grad():
                grad_norm = torch.linalg.norm(
                    grad.view(batch_size, -1), dim=1
                ).view(*reshape_dims).clamp_min(1e-12)
                delta = delta + step_size * grad / grad_norm

                delta_norm = torch.linalg.norm(
                    delta.view(batch_size, -1), dim=1
                ).view(*reshape_dims).clamp_min(1e-12)
                scale = torch.minimum(
                    torch.ones_like(delta_norm), float(epsilon) / delta_norm
                )
                delta = delta * scale

        with torch.no_grad():
            adv_features = original_features + delta
            losses = criterion(model(adv_features), labels)
            improve = losses > best_loss
            best_loss = torch.where(improve, losses, best_loss)
            best_adv = torch.where(improve.view(*reshape_dims), adv_features, best_adv)

    return best_adv.detach()
