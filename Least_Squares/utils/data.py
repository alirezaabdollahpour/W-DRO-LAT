"""Problem sampling for the uncertain least squares experiment."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from config import ULSConfig


def make_problem(
    cfg: ULSConfig, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(cfg.seed)
    A0 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    A1 = torch.tensor(rng.randn(cfg.dim_m, cfg.dim_n), device=device)
    b = torch.tensor(rng.randn(cfg.dim_m), device=device)
    return A0, A1, b


def generate_training_data(cfg: ULSConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(cfg.n_train, device=device).uniform_(-0.5, 0.5)


def generate_test_data(cfg: ULSConfig, delta: float, device: torch.device) -> torch.Tensor:
    lo = -0.5 * (1.0 + float(delta))
    hi = 0.5 * (1.0 + float(delta))
    return torch.empty(cfg.n_test, device=device).uniform_(lo, hi)
