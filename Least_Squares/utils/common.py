"""Seeding, tensor helpers, gradient clipping, deterministic backends."""
from __future__ import annotations

import math

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def clip_grad(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    gnorm = grad.norm()
    if not torch.isfinite(gnorm):
        return torch.zeros_like(grad)
    if max_norm > 0.0 and gnorm > max_norm:
        grad = grad * (max_norm / (gnorm + 1e-12))
    return grad


def softplus_inverse(y: float) -> float:
    """Numerically stable inverse of softplus: softplus^{-1}(y) = log(e^y - 1)."""
    if y <= 0:
        return -1e3
    if y > 20.0:
        return float(y)
    return float(math.log(math.expm1(y)))


def set_deterministic_backends() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
