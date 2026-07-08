"""Clean / transport-adversary / input-PGD evaluation helpers."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .transforms import (
    clamped_normalized_copy,
    normalized_mse,
    pixel_l2_squared,
    project_normalized_to_input_ball,
    to_normalized,
    to_pixel,
)


def _module_training_modes(module: nn.Module) -> Dict[nn.Module, bool]:
    return {child: child.training for child in module.modules()}


def _restore_training_modes(training_modes: Dict[nn.Module, bool]) -> None:
    for module, was_training in training_modes.items():
        module.train(was_training)


def _pgd_loss_per_sample(
    logits: torch.Tensor,
    y: torch.Tensor,
    loss: str,
) -> torch.Tensor:
    loss = str(loss).lower()
    if loss in {"ce", "cross_entropy"}:
        return F.cross_entropy(logits, y, reduction="none")
    if loss in {"margin", "logsumexp_margin", "lse_margin"}:
        true_logits = logits.gather(1, y[:, None]).squeeze(1)
        non_true = logits.masked_fill(
            F.one_hot(y, num_classes=logits.size(1)).bool(),
            float("-inf"),
        )
        return torch.logsumexp(non_true, dim=1) - true_logits
    raise ValueError("input PGD loss must be 'margin' or 'ce'.")


@torch.no_grad()
def evaluate_clean(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    training_modes = _module_training_modes(classifier)
    classifier.eval()
    try:
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
    finally:
        _restore_training_modes(training_modes)


def evaluate_under_transport(
    classifier: nn.Module,
    transport_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    penalty_lambda: float,
    cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[float, float, float, float, Dict[str, Any]]:
    """Eval CE / acc / transport cost when adversary maps clean -> adversarial inputs.

    ``transport_fn`` must accept normalized inputs and return normalized
    adversarial inputs of the same shape (no grad needed). ``max_samples``
    caps the number of evaluated test points (None/0 = full test set) — used
    when the transport is an expensive per-batch attack (e.g. MPA) rather
    than a cheap learned map.
    """
    if cost_fn is None:
        cost_fn = pixel_l2_squared
    sample_cap = int(max_samples) if max_samples else 0
    training_modes = _module_training_modes(classifier)
    classifier.eval()
    try:
        total_loss = 0.0
        total_correct = 0
        total = 0
        total_mse_sum = 0.0
        total_norm_mse_sum = 0.0
        total_l2_sum = 0.0
        total_linf_sum = 0.0
        max_norm_mse = 0.0
        max_l2 = 0.0
        max_linf = 0.0
        for x, y in loader:
            if sample_cap > 0 and total >= sample_cap:
                break
            x = x.to(device)
            y = y.to(device)
            x_adv = transport_fn(x).detach()
            logits = classifier(x_adv)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += x.size(0)
            total_mse_sum += float(cost_fn(x_adv, x).sum().item())
            norm_mse = normalized_mse(x_adv, x)
            delta_pix = to_pixel(x_adv) - to_pixel(x)
            flat_delta_pix = delta_pix.reshape(delta_pix.size(0), -1)
            l2 = flat_delta_pix.norm(p=2, dim=1)
            linf = flat_delta_pix.abs().max(dim=1)[0]
            total_norm_mse_sum += float(norm_mse.sum().item())
            total_l2_sum += float(l2.sum().item())
            total_linf_sum += float(linf.sum().item())
            max_norm_mse = max(max_norm_mse, float(norm_mse.max().item()))
            max_l2 = max(max_l2, float(l2.max().item()))
            max_linf = max(max_linf, float(linf.max().item()))
        mean_loss = total_loss / max(1, total)
        mean_acc = total_correct / max(1, total)
        mean_mse = total_mse_sum / max(1, total)
        mean_penalty = penalty_lambda * mean_mse
        info = {
            "avg_l2": total_l2_sum / max(1, total),
            "avg_linf": total_linf_sum / max(1, total),
            "max_l2": max_l2,
            "max_linf": max_linf,
            "avg_normalized_mse": total_norm_mse_sum / max(1, total),
            "max_normalized_mse": max_norm_mse,
            "samples": total,
        }
        return mean_loss, mean_acc, mean_mse, mean_penalty, info
    finally:
        _restore_training_modes(training_modes)


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


def _normalized_mse_radius(delta: torch.Tensor, budget: float) -> float:
    flat_dim = int(delta[0].numel()) if delta.numel() > 0 else 1
    return math.sqrt(max(0.0, float(budget)) * float(flat_dim))


def _project_normalized_mse(delta_norm: torch.Tensor, budget: float) -> torch.Tensor:
    radius = _normalized_mse_radius(delta_norm, budget)
    flat = delta_norm.view(delta_norm.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (radius / norms).clamp(max=1.0)
    return (flat * scale).view_as(delta_norm)


def _random_start_pix(x0: torch.Tensor, eps: float, p) -> torch.Tensor:
    if p == 2:
        z = torch.randn_like(x0)
        z = _l2_normalize(z)
        radii = torch.rand(x0.size(0), 1, 1, 1, device=x0.device, dtype=x0.dtype)
        delta = z * radii * eps
    else:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
    return (x0 + delta).clamp(0.0, 1.0)


def _random_start_normalized_mse(x0_norm: torch.Tensor, budget: float) -> torch.Tensor:
    z = torch.randn_like(x0_norm)
    z = _l2_normalize(z)
    radius = _normalized_mse_radius(x0_norm, budget)
    radii = torch.rand(x0_norm.size(0), 1, 1, 1, device=x0_norm.device, dtype=x0_norm.dtype)
    delta = z * radii * radius
    return clamped_normalized_copy(x0_norm + delta)


def _canonicalize_input_pgd_options(loss: str, geometry: str, p) -> Tuple[str, str]:
    loss = str(loss).lower()
    if loss == "cross_entropy":
        loss = "ce"
    elif loss in {"logsumexp_margin", "lse_margin"}:
        loss = "margin"
    if loss not in {"margin", "ce"}:
        raise ValueError("input PGD loss must be 'margin' or 'ce'.")
    geometry = str(geometry).lower()
    if geometry in {"pixel", "pixel_l2", "pixel_l2_squared"}:
        geometry = "pixel_l2_squared"
    elif geometry in {"normalized", "normalized_mse"}:
        geometry = "normalized_mse"
    else:
        raise ValueError("input PGD geometry must be 'pixel_l2_squared' or 'normalized_mse'.")
    if geometry == "normalized_mse" and p != 2:
        raise ValueError("normalized_mse input PGD geometry requires p=2.")
    return loss, geometry


def _input_pgd_best_adversary(
    classifier: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    p,
    eps: float,
    steps: int,
    step_size: Optional[float],
    restarts: int,
    loss: str,
    geometry: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_adv_norm, final_delta_pix)`` from the configured PGD attack."""
    user_step_size = step_size
    restarts = max(1, int(restarts))
    steps = max(0, int(steps))

    if geometry == "pixel_l2_squared":
        if step_size is None or step_size <= 0:
            step_size = 2.0 * eps / max(1, steps)
        x0_pix = to_pixel(x_norm).detach()
        best_delta = torch.zeros_like(x0_pix)
        batch_size = x0_pix.size(0)
    else:
        x0_norm = x_norm.detach()
        best_delta = torch.zeros_like(x0_norm)
        batch_size = x0_norm.size(0)
        step_size = (
            float(user_step_size)
            if user_step_size is not None and user_step_size > 0
            else 2.0 * _normalized_mse_radius(x0_norm, eps) / max(1, steps)
        )
    best_loss = torch.full((batch_size,), float("-inf"), device=x_norm.device)

    for _ in range(restarts):
        if geometry == "pixel_l2_squared":
            x_pix = _random_start_pix(x0_pix, eps, p).detach()
            with torch.no_grad():
                scores = _pgd_loss_per_sample(
                    classifier(to_normalized(x_pix)), y, loss
                )
                mask = scores > best_loss
                if mask.any():
                    best_loss[mask] = scores[mask]
                    best_delta[mask] = (x_pix - x0_pix).detach()[mask]

            x_pix = x_pix.detach().requires_grad_(True)
            for _ in range(steps):
                logits = classifier(to_normalized(x_pix))
                losses = _pgd_loss_per_sample(logits, y, loss)
                grad_pix = torch.autograd.grad(losses.mean(), x_pix)[0]
                with torch.no_grad():
                    step = step_size * (
                        _l2_normalize(grad_pix) if p == 2 else torch.sign(grad_pix)
                    )
                    x_pix = x_pix + step
                    delta = x_pix - x0_pix
                    delta = _project_lp(delta, eps, p)
                    x_pix = (x0_pix + delta).clamp(0.0, 1.0)
                    scores = _pgd_loss_per_sample(
                        classifier(to_normalized(x_pix)), y, loss
                    )
                    mask = scores > best_loss
                    if mask.any():
                        best_loss[mask] = scores[mask]
                        best_delta[mask] = (x_pix - x0_pix).detach()[mask]
                x_pix = x_pix.detach().requires_grad_(True)
        else:
            x0_norm = x_norm.detach()
            x_adv_norm = _random_start_normalized_mse(x0_norm, eps).detach()
            with torch.no_grad():
                scores = _pgd_loss_per_sample(classifier(x_adv_norm), y, loss)
                mask = scores > best_loss
                if mask.any():
                    best_loss[mask] = scores[mask]
                    best_delta[mask] = (x_adv_norm - x0_norm).detach()[mask]

            x_adv_norm = x_adv_norm.detach().requires_grad_(True)
            for _ in range(steps):
                logits = classifier(x_adv_norm)
                losses = _pgd_loss_per_sample(logits, y, loss)
                grad_norm = torch.autograd.grad(losses.mean(), x_adv_norm)[0]
                with torch.no_grad():
                    x_adv_norm = x_adv_norm + step_size * _l2_normalize(grad_norm)
                    delta = x_adv_norm - x0_norm
                    delta = _project_normalized_mse(delta, eps)
                    x_adv_norm = clamped_normalized_copy(x0_norm + delta)
                    scores = _pgd_loss_per_sample(classifier(x_adv_norm), y, loss)
                    mask = scores > best_loss
                    if mask.any():
                        best_loss[mask] = scores[mask]
                        best_delta[mask] = (x_adv_norm - x0_norm).detach()[mask]
                x_adv_norm = x_adv_norm.detach().requires_grad_(True)

    with torch.no_grad():
        if geometry == "pixel_l2_squared":
            best_delta = _project_lp(best_delta, eps, p)
            x_adv_pix = (x0_pix + best_delta).clamp(0.0, 1.0)
            x_adv_norm = to_normalized(x_adv_pix)
            final_delta_pix = x_adv_pix - x0_pix
        else:
            best_delta = _project_normalized_mse(best_delta, eps)
            x0_norm = x_norm.detach()
            x_adv_norm = clamped_normalized_copy(x0_norm + best_delta)
            final_delta_pix = to_pixel(x_adv_norm) - to_pixel(x0_norm)
    return x_adv_norm, final_delta_pix


def input_pgd_best_adversary(
    classifier: nn.Module,
    x_norm: torch.Tensor,
    y: torch.Tensor,
    *,
    p,
    eps: float,
    steps: int,
    step_size: Optional[float],
    restarts: int,
    loss: str = "margin",
    geometry: str = "pixel_l2_squared",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public helper returning the configured PGD adversarial batch.

    Unlike :func:`evaluate_under_input_pgd`, this does not alter training
    modes; callers should wrap the classifier in the desired mode first.
    """
    loss, geometry = _canonicalize_input_pgd_options(loss, geometry, p)
    return _input_pgd_best_adversary(
        classifier,
        x_norm,
        y,
        p=p,
        eps=eps,
        steps=steps,
        step_size=step_size,
        restarts=restarts,
        loss=loss,
        geometry=geometry,
    )


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
    loss: str = "margin",
    geometry: str = "pixel_l2_squared",
):
    """Input-space PGD attack + robust accuracy.

    ``geometry="pixel_l2_squared"`` is the standard CIFAR benchmark attack:
    the threat model is measured in unnormalized pixel coordinates with
    components in [0, 1], and ``eps`` is the L2/Linf norm radius.

    ``geometry="normalized_mse"`` runs L2 PGD in normalized CIFAR
    coordinates and interprets ``eps`` as the budget on
    mean((x_adv_norm - x_norm)^2). This is useful only for explicit
    compatibility diagnostics against the legacy normalized-MSE DRO cost.

    Robust accuracy follows the common benchmark convention: a
    clean-misclassified example is already non-robust, so it cannot become
    robust by landing on a correctly classified adversarial point. The
    default logsumexp-margin objective avoids the CE saturation artifact that
    can make clean fine-tuned models look spuriously robust.
    """
    loss, geometry = _canonicalize_input_pgd_options(loss, geometry, p)

    training_modes = _module_training_modes(classifier)
    classifier.eval()
    total_robust_correct = 0
    total_clean_correct = 0
    total = 0
    sum_l2 = 0.0
    sum_linf = 0.0
    max_l2 = 0.0
    max_linf = 0.0
    sum_normalized_mse = 0.0
    max_normalized_mse = 0.0

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    if max_samples is not None and total_batches is not None:
        batch_size = getattr(loader, "batch_size", None) or max_samples
        total_batches = min(
            total_batches,
            max(1, (int(max_samples) + int(batch_size) - 1) // int(batch_size)),
        )

    progress = tqdm(loader, desc=f"Input-PGD/{loss}/{geometry}", leave=False, total=total_batches)
    try:
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

            with torch.no_grad():
                clean_pred = classifier(x_norm).argmax(dim=1)
                clean_correct = clean_pred == y

            x_adv_norm, final_delta_pix = _input_pgd_best_adversary(
                classifier,
                x_norm,
                y,
                p=p,
                eps=eps,
                steps=steps,
                step_size=step_size,
                restarts=restarts,
                loss=loss,
                geometry=geometry,
            )

            with torch.no_grad():
                logits = classifier(x_adv_norm)
                adv_correct = logits.argmax(dim=1) == y
                robust_correct = clean_correct & adv_correct
                total_robust_correct += int(robust_correct.sum().item())
                total_clean_correct += int(clean_correct.sum().item())
                total += x_norm.size(0)
                l2 = final_delta_pix.view(final_delta_pix.size(0), -1).norm(p=2, dim=1)
                linf = final_delta_pix.abs().view(final_delta_pix.size(0), -1).max(dim=1)[0]
                norm_mse = normalized_mse(x_adv_norm, x_norm)
                sum_l2 += float(l2.sum().item())
                sum_linf += float(linf.sum().item())
                max_l2 = max(max_l2, float(l2.max().item()))
                max_linf = max(max_linf, float(linf.max().item()))
                sum_normalized_mse += float(norm_mse.sum().item())
                max_normalized_mse = max(max_normalized_mse, float(norm_mse.max().item()))

        return total_robust_correct / max(1, total), {
            "clean_acc": total_clean_correct / max(1, total),
            "clean_correct": total_clean_correct,
            "robust_correct": total_robust_correct,
            "avg_l2": sum_l2 / max(1, total),
            "avg_linf": sum_linf / max(1, total),
            "max_l2": max_l2,
            "max_linf": max_linf,
            "avg_normalized_mse": sum_normalized_mse / max(1, total),
            "max_normalized_mse": max_normalized_mse,
            "samples": total,
            "loss": loss,
            "geometry": geometry,
        }
    finally:
        progress.close()
        _restore_training_modes(training_modes)


def evaluate_transport_pgd_alignment(
    classifier: nn.Module,
    transport_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    *,
    p,
    eps: float,
    steps: int,
    step_size: Optional[float],
    restarts: int = 1,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
    loss: str = "margin",
    geometry: str = "pixel_l2_squared",
) -> Dict[str, Any]:
    """Compare learned transport directions against PGD directions.

    This is a diagnostic, not a robust-accuracy metric. It catches the common
    failure mode where a learned transport has large norm but points almost
    orthogonally to the local PGD maximizer.
    """
    loss, geometry = _canonicalize_input_pgd_options(loss, geometry, p)
    training_modes = _module_training_modes(classifier)
    classifier.eval()

    total = 0
    valid_cos = 0
    cos_chunks = []
    transport_correct = 0
    transport_projected_correct = 0
    pgd_correct = 0
    transport_loss_sum = 0.0
    transport_projected_loss_sum = 0.0
    pgd_loss_sum = 0.0
    transport_l2_sum = 0.0
    transport_projected_l2_sum = 0.0
    pgd_l2_sum = 0.0

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    if max_samples is not None and total_batches is not None:
        batch_size = getattr(loader, "batch_size", None) or max_samples
        total_batches = min(
            total_batches,
            max(1, (int(max_samples) + int(batch_size) - 1) // int(batch_size)),
        )

    progress = tqdm(
        loader,
        desc=f"Transport-vs-PGD/{loss}/{geometry}",
        leave=False,
        total=total_batches,
    )
    try:
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

            with torch.no_grad():
                x_transport = clamped_normalized_copy(transport_fn(x_norm).detach())
                x_transport_projected = project_normalized_to_input_ball(
                    x_transport,
                    x_norm,
                    eps=eps,
                    p=p,
                    geometry=geometry,
                )

            x_pgd, pgd_delta_pix = _input_pgd_best_adversary(
                classifier,
                x_norm,
                y,
                p=p,
                eps=eps,
                steps=steps,
                step_size=step_size,
                restarts=restarts,
                loss=loss,
                geometry=geometry,
            )

            with torch.no_grad():
                logits_transport = classifier(x_transport)
                logits_transport_projected = classifier(x_transport_projected)
                logits_pgd = classifier(x_pgd)
                transport_loss = _pgd_loss_per_sample(logits_transport, y, loss)
                transport_projected_loss = _pgd_loss_per_sample(
                    logits_transport_projected, y, loss
                )
                pgd_loss = _pgd_loss_per_sample(logits_pgd, y, loss)

                transport_correct += int((logits_transport.argmax(dim=1) == y).sum().item())
                transport_projected_correct += int(
                    (logits_transport_projected.argmax(dim=1) == y).sum().item()
                )
                pgd_correct += int((logits_pgd.argmax(dim=1) == y).sum().item())
                transport_loss_sum += float(transport_loss.sum().item())
                transport_projected_loss_sum += float(transport_projected_loss.sum().item())
                pgd_loss_sum += float(pgd_loss.sum().item())

                delta_transport_pix = to_pixel(x_transport) - to_pixel(x_norm)
                delta_transport_projected_pix = (
                    to_pixel(x_transport_projected) - to_pixel(x_norm)
                )
                transport_flat = delta_transport_pix.reshape(delta_transport_pix.size(0), -1)
                pgd_flat = pgd_delta_pix.reshape(pgd_delta_pix.size(0), -1)
                transport_l2 = transport_flat.norm(p=2, dim=1)
                transport_projected_l2 = delta_transport_projected_pix.reshape(
                    delta_transport_projected_pix.size(0), -1
                ).norm(p=2, dim=1)
                pgd_l2 = pgd_flat.norm(p=2, dim=1)
                denom = (transport_l2 * pgd_l2).clamp_min(1e-12)
                cos = (transport_flat * pgd_flat).sum(dim=1) / denom
                valid = (transport_l2 > 1e-12) & (pgd_l2 > 1e-12)

                # Only valid pairs: near-zero transports have a clamp-forced
                # cosine of ~0 that drags the mean toward zero and makes a
                # frozen adversary read as "orthogonal".
                cos_chunks.append(cos[valid].detach().cpu())
                valid_cos += int(valid.sum().item())
                transport_l2_sum += float(transport_l2.sum().item())
                transport_projected_l2_sum += float(transport_projected_l2.sum().item())
                pgd_l2_sum += float(pgd_l2.sum().item())
                total += x_norm.size(0)

        if cos_chunks:
            cos_all = torch.cat(cos_chunks)
            cos_mean = float(cos_all.mean().item())
            cos_median = float(cos_all.median().item())
            cos_p10 = float(torch.quantile(cos_all, 0.10).item())
            cos_p90 = float(torch.quantile(cos_all, 0.90).item())
        else:
            cos_mean = cos_median = cos_p10 = cos_p90 = 0.0
        n = max(1, total)
        return {
            "samples": total,
            "valid_cos_samples": valid_cos,
            "cos_mean": cos_mean,
            "cos_median": cos_median,
            "cos_p10": cos_p10,
            "cos_p90": cos_p90,
            "transport_acc": transport_correct / n,
            "transport_projected_acc": transport_projected_correct / n,
            "pgd_acc": pgd_correct / n,
            "transport_loss": transport_loss_sum / n,
            "transport_projected_loss": transport_projected_loss_sum / n,
            "pgd_loss": pgd_loss_sum / n,
            "transport_avg_l2": transport_l2_sum / n,
            "transport_projected_avg_l2": transport_projected_l2_sum / n,
            "pgd_avg_l2": pgd_l2_sum / n,
            "loss": loss,
            "geometry": geometry,
        }
    finally:
        progress.close()
        _restore_training_modes(training_modes)
