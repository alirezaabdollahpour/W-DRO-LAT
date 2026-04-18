"""Small shared helpers: seeding, loss, accuracy, requires_grad toggles."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    """Seed PyTorch (CPU + all CUDA devices) and NumPy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="mean")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def adversary_loss(
    logits: torch.Tensor, labels: torch.Tensor, use_margin_adv: bool
) -> torch.Tensor:
    if use_margin_adv:
        logits_correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        margins = logits - logits_correct.unsqueeze(1)
        num_classes = logits.size(1)
        mask = F.one_hot(labels, num_classes=num_classes).bool()
        margins = torch.where(
            mask, torch.tensor(-float("inf"), device=logits.device), margins
        )
        return torch.logsumexp(margins, dim=1).mean()
    return cross_entropy_loss(logits, labels)


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def repeat_particles(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    reps = [1, num_samples] + [1] * (x.dim() - 1)
    return x.unsqueeze(1).repeat(*reps).view(-1, *x.shape[1:])


def weighted_accuracy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().view_as(weights)
    return (correct * weights).sum(dim=1).mean()


def rbf_kernel_batched_images(
    particles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RBF kernel + gradient for batched particle systems.

    particles : [B, S, C, H, W] -> (K [B,S,S], grad_K [B,S,S,C,H,W]).
    """
    B, S = particles.shape[:2]
    flat = particles.view(B, S, -1)
    sq_dist = torch.cdist(flat, flat, p=2).pow(2)
    median_sq_dist = torch.median(sq_dist.view(B, -1), dim=1, keepdim=True)[0].view(
        B, 1, 1
    )
    h_squared = median_sq_dist / (
        torch.log(torch.tensor(float(max(S, 2)), device=particles.device)) + 1e-8
    )
    h_squared = h_squared.clamp(min=1e-6)
    K = torch.exp(-sq_dist / (2.0 * h_squared))
    diff = flat.unsqueeze(2) - flat.unsqueeze(1)
    grad_K_flat = -diff / h_squared.unsqueeze(-1) * K.unsqueeze(-1)
    return K, grad_K_flat.view(B, S, S, *particles.shape[2:])


def set_deterministic_backends() -> None:
    """Enable deterministic cuDNN + PyTorch algorithms (warn-only)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
