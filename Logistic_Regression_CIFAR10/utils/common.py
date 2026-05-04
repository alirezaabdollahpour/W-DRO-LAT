"""Shared helpers: device, seeding."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed NumPy + PyTorch (CPU + all CUDA devices) for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Toggle requires_grad on every leaf parameter in ``module``.

    Mirrors ``MNIST_Cuturi.utils.common.set_requires_grad`` so adversarial
    inner-loop callers can freeze the classifier in a single line. Calling
    autograd helpers like ``torch.autograd.grad(..., params=...)`` already
    routes the gradient to the requested params, but the explicit freeze
    is defensive: any future refactor to ``f.backward()`` would silently
    accumulate adversary gradients into the classifier without it.
    """
    for p in module.parameters():
        p.requires_grad_(flag)
