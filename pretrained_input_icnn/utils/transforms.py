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
