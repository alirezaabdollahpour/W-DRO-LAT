"""Device, deterministic seeding, parameter freezing helpers."""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_seed_benchmark_mode(seed: int = 1) -> None:
    """Seed RNGs but enable cuDNN autotune and TF32 for max throughput.

    Use this in place of ``set_deterministic`` for runtime comparisons:
    seeds make the *data order* and *initialisations* reproducible across
    runs, while the kernel selection is free to pick the fastest path.
    Bit-exact reproducibility is sacrificed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def cuda_sync() -> None:
    """No-op when CUDA is unavailable; otherwise force synchronisation.

    Wallclock measurements taken without sync only count kernel-launch
    time (most CUDA work is async). Always sync immediately before
    reading the clock for accurate per-epoch / per-batch timing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


@contextmanager
def frozen_module(module: nn.Module, *, eval_mode: bool = True) -> Iterator[nn.Module]:
    """Temporarily freeze a module's parameters and restore its prior state."""
    was_training = module.training
    params = list(module.parameters())
    requires_grad = [p.requires_grad for p in params]
    if eval_mode:
        module.eval()
    for p in params:
        p.requires_grad_(False)
    try:
        yield module
    finally:
        for p, req in zip(params, requires_grad):
            p.requires_grad_(req)
        module.train(was_training)
