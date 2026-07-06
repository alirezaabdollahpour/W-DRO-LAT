"""CIFAR-10 normalization helpers and pixel-bound clamping for normalized space."""
from __future__ import annotations

from typing import Tuple

import torch

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)


def _mean_std(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(CIFAR10_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return mean, std


def to_pixel(x_norm: torch.Tensor) -> torch.Tensor:
    mean, std = _mean_std(x_norm)
    return x_norm * std + mean


def to_normalized(x_pix: torch.Tensor) -> torch.Tensor:
    mean, std = _mean_std(x_pix)
    return (x_pix - mean) / std


def normalized_pixel_bounds(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (lower, upper) bounds in normalized space corresponding to pixels in [0, 1]."""
    mean, std = _mean_std(x)
    return (0.0 - mean) / std, (1.0 - mean) / std


def clamp_normalized_inputs_(x: torch.Tensor) -> None:
    lower, upper = normalized_pixel_bounds(x)
    x.clamp_(min=lower, max=upper)


def clamped_normalized_copy(x: torch.Tensor) -> torch.Tensor:
    lower, upper = normalized_pixel_bounds(x)
    return torch.clamp(x, min=lower, max=upper)


def project_normalized_to_pixel_lp_ball(
    x_adv_norm: torch.Tensor,
    x_clean_norm: torch.Tensor,
    *,
    eps: float,
    p,
) -> torch.Tensor:
    """Project normalized adversarial inputs into a pixel-space Lp ball."""
    eps = float(eps)
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative; got {eps}.")
    x_clean_pix = to_pixel(x_clean_norm)
    x_adv_pix = to_pixel(x_adv_norm)
    delta = x_adv_pix - x_clean_pix
    p_label = str(p).lower()
    if p == 2 or p_label == "2":
        flat = delta.reshape(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = (eps / norms).clamp(max=1.0)
        delta = (flat * scale).view_as(delta)
    elif p == float("inf") or p_label in {"inf", "infinity"}:
        delta = delta.clamp(min=-eps, max=eps)
    else:
        raise ValueError(f"Unsupported pixel Lp projection p={p!r}.")
    return to_normalized((x_clean_pix + delta).clamp(0.0, 1.0))


def project_normalized_to_normalized_mse_ball(
    x_adv_norm: torch.Tensor,
    x_clean_norm: torch.Tensor,
    *,
    budget: float,
) -> torch.Tensor:
    """Project into mean((x_adv_norm - x_clean_norm)^2) <= budget."""
    budget = float(budget)
    if budget < 0.0:
        raise ValueError(f"budget must be non-negative; got {budget}.")
    delta = x_adv_norm - x_clean_norm
    flat = delta.reshape(delta.size(0), -1)
    radius = (budget * max(1, flat.size(1))) ** 0.5
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = (radius / norms).clamp(max=1.0)
    projected = x_clean_norm + (flat * scale).view_as(delta)
    return clamped_normalized_copy(projected)


def project_normalized_to_input_ball(
    x_adv_norm: torch.Tensor,
    x_clean_norm: torch.Tensor,
    *,
    eps: float,
    p,
    geometry: str,
) -> torch.Tensor:
    """Project normalized inputs to the configured input-adversary threat set."""
    geometry = str(geometry).lower()
    if geometry in {"pixel", "pixel_l2", "pixel_l2_squared"}:
        return project_normalized_to_pixel_lp_ball(
            x_adv_norm,
            x_clean_norm,
            eps=eps,
            p=p,
        )
    if geometry in {"normalized", "normalized_mse"}:
        if not (p == 2 or str(p).lower() == "2"):
            raise ValueError("normalized_mse projection requires p=2.")
        return project_normalized_to_normalized_mse_ball(
            x_adv_norm,
            x_clean_norm,
            budget=eps,
        )
    raise ValueError(
        "input projection geometry must be 'pixel_l2_squared' or 'normalized_mse'."
    )


def normalized_mse(
    x_adv_norm: torch.Tensor,
    x_clean_norm: torch.Tensor,
) -> torch.Tensor:
    """Per-sample mean squared distance in normalized CIFAR coordinates.

    This is the transport penalty convention used by the legacy
    ``pretrained_INPUT_icnn.py`` input-space ICNN implementation. It divides
    by the number of input coordinates and does not undo CIFAR normalization.
    """
    diff = x_adv_norm - x_clean_norm
    return diff.reshape(diff.size(0), -1).pow(2).mean(dim=1)


def pixel_l2_squared(
    x_adv_norm: torch.Tensor,
    x_clean_norm: torch.Tensor,
) -> torch.Tensor:
    """Per-sample squared L2 distance in CIFAR pixel coordinates.

    Inputs are normalized tensors because that is the package-wide
    classifier convention. The returned cost is measured after undoing
    CIFAR normalization, matching the standard CIFAR L2 robustness
    benchmark convention where eps is defined on pixels in [0, 1].
    """
    diff_pix = to_pixel(x_adv_norm) - to_pixel(x_clean_norm)
    return diff_pix.reshape(diff_pix.size(0), -1).pow(2).sum(dim=1)
