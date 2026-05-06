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
