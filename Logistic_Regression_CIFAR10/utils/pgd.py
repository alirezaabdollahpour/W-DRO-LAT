"""l2-PGD attack on feature vectors (verbatim from adversarial_multiclass_icnn.py)."""
from __future__ import annotations

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_iter: int,
    device: torch.device,
) -> torch.Tensor:
    """PGD adversarial attack (l2 norm)."""
    perturbed_features = features.clone().detach().to(device)
    perturbed_features.requires_grad = True
    original_features = features.clone().detach().to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        if perturbed_features.grad is not None:
            perturbed_features.grad.zero_()

        outputs = model(perturbed_features)
        loss = criterion(outputs, labels)
        loss.backward()

        grad = perturbed_features.grad.detach()
        grad_norm = torch.linalg.norm(
            grad.view(grad.shape[0], -1), dim=1, keepdim=True
        ) + 1e-12
        normalized_grad = grad / grad_norm

        reshape_dims = [grad.shape[0]] + [1] * (grad.dim() - 1)

        perturbed_features.data = perturbed_features.data + alpha * normalized_grad
        perturbation = perturbed_features.data - original_features.data

        pert_norm = torch.linalg.norm(
            perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True
        )
        factor = epsilon / (pert_norm + 1e-12)
        factor = torch.min(torch.ones_like(factor), factor)

        perturbation = perturbation * factor.view(*reshape_dims)
        perturbed_features.data = original_features.data + perturbation

    return perturbed_features.detach()
