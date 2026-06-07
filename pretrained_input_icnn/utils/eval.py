"""Clean / transport-adversary / input-PGD evaluation helpers."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .transforms import (
    pixel_l2_squared,
    to_normalized,
    to_pixel,
)


@torch.no_grad()
def evaluate_clean(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = classifier(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def evaluate_under_transport(
    classifier: nn.Module,
    transport_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    penalty_lambda: float,
) -> Tuple[float, float, float, float]:
    """Eval CE / acc / transport cost when adversary maps clean -> adversarial inputs.

    ``transport_fn`` must accept normalized inputs and return normalized
    adversarial inputs of the same shape (no grad needed).
    """
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    total_mse_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_adv = transport_fn(x).detach()
        logits = classifier(x_adv)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
        total_mse_sum += float(pixel_l2_squared(x_adv, x).sum().item())
    mean_loss = total_loss / max(1, total)
    mean_acc = total_correct / max(1, total)
    mean_mse = total_mse_sum / max(1, total)
    mean_penalty = penalty_lambda * mean_mse
    return mean_loss, mean_acc, mean_mse, mean_penalty


def _l2_normalize(grad: torch.Tensor) -> torch.Tensor:
    flat = grad.view(grad.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norms).view_as(grad)


def _project_lp(delta: torch.Tensor, eps: float, p) -> torch.Tensor:
    if p == 2:
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = (eps / norms).clamp(max=1.0)
        return (flat * scale).view_as(delta)
    return delta.clamp(min=-eps, max=eps)


def _random_start_pix(x0: torch.Tensor, eps: float, p) -> torch.Tensor:
    if p == 2:
        z = torch.randn_like(x0)
        z = _l2_normalize(z)
        radii = torch.rand(x0.size(0), 1, 1, 1, device=x0.device, dtype=x0.dtype)
        delta = z * radii * eps
    else:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
    return (x0 + delta).clamp(0.0, 1.0)


def evaluate_under_input_pgd(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    p,
    eps: float,
    steps: int,
    step_size: Optional[float],
    restarts: int = 1,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """Pixel-space PGD attack + robust accuracy.

    The threat model is measured in unnormalized pixel coordinates with
    components in [0, 1]. Robust accuracy follows the common benchmark
    convention: a clean-misclassified example is already non-robust, so it
    cannot become robust by landing on a correctly classified adversarial
    point.
    """
    classifier.eval()
    total_robust_correct = 0
    total_clean_correct = 0
    total = 0
    sum_l2 = 0.0
    sum_linf = 0.0
    max_l2 = 0.0
    max_linf = 0.0

    if step_size is None or step_size <= 0:
        step_size = 2.0 * eps / max(1, steps)
    restarts = max(1, int(restarts))

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    if max_samples is not None and total_batches is not None:
        batch_size = getattr(loader, "batch_size", None) or max_samples
        total_batches = min(total_batches, max(1, (int(max_samples) + batch_size - 1) // batch_size))
    progress = tqdm(loader, desc="Input-PGD", leave=False, total=total_batches)
    for batch_idx, (x_norm, y) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if max_samples is not None and total >= max_samples:
            break
        if max_samples is not None:
            remaining = int(max_samples) - total
            if remaining <= 0:
                break
            x_norm = x_norm[:remaining]
            y = y[:remaining]
        x_norm = x_norm.to(device)
        y = y.to(device)
        x0_pix = to_pixel(x_norm).detach()

        with torch.no_grad():
            clean_pred = classifier(x_norm).argmax(dim=1)
            clean_correct = clean_pred == y

        best_delta = torch.zeros_like(x0_pix)
        best_loss = torch.full((x0_pix.size(0),), -1e9, device=device)

        for _ in range(restarts):
            x_pix = _random_start_pix(x0_pix, eps, p).detach().requires_grad_(True)
            for _ in range(steps):
                logits = classifier(to_normalized(x_pix))
                loss_mean = F.cross_entropy(logits, y, reduction="mean")
                grad_pix = torch.autograd.grad(loss_mean, x_pix)[0]
                with torch.no_grad():
                    step = step_size * (
                        _l2_normalize(grad_pix) if p == 2 else torch.sign(grad_pix)
                    )
                    x_pix = x_pix + step
                    delta = x_pix - x0_pix
                    delta = _project_lp(delta, eps, p)
                    x_pix = (x0_pix + delta).clamp(0.0, 1.0)
                    x_pix.requires_grad_(True)

            with torch.no_grad():
                logits = classifier(to_normalized(x_pix))
                losses = F.cross_entropy(logits, y, reduction="none")
                delta_pix = (x_pix - x0_pix).detach()
                mask = losses > best_loss
                if mask.any():
                    best_loss[mask] = losses[mask]
                    best_delta[mask] = delta_pix[mask]

        with torch.no_grad():
            best_delta = _project_lp(best_delta, eps, p)
            x_adv_pix = (x0_pix + best_delta).clamp(0.0, 1.0)
            final_delta = x_adv_pix - x0_pix
            logits = classifier(to_normalized(x_adv_pix))
            adv_correct = logits.argmax(dim=1) == y
            robust_correct = clean_correct & adv_correct
            total_robust_correct += int(robust_correct.sum().item())
            total_clean_correct += int(clean_correct.sum().item())
            total += x_norm.size(0)
            l2 = final_delta.view(final_delta.size(0), -1).norm(p=2, dim=1)
            linf = final_delta.abs().view(final_delta.size(0), -1).max(dim=1)[0]
            sum_l2 += float(l2.sum().item())
            sum_linf += float(linf.sum().item())
            max_l2 = max(max_l2, float(l2.max().item()))
            max_linf = max(max_linf, float(linf.max().item()))

    progress.close()
    return total_robust_correct / max(1, total), {
        "clean_acc": total_clean_correct / max(1, total),
        "clean_correct": total_clean_correct,
        "robust_correct": total_robust_correct,
        "avg_l2": sum_l2 / max(1, total),
        "avg_linf": sum_linf / max(1, total),
        "max_l2": max_l2,
        "max_linf": max_linf,
        "samples": total,
    }
