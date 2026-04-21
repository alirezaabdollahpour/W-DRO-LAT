"""ERM (closed-form sample-average least squares)."""
from __future__ import annotations

import torch

from config import ULSConfig


def solve_erm_closed_form(
    xi_train: torch.Tensor,
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    n = cfg.dim_n
    S = torch.zeros((n, n), device=xi_train.device)
    t = torch.zeros((n,), device=xi_train.device)
    for xi in xi_train:
        A = A0 + xi * A1
        S = S + A.transpose(0, 1) @ A
        t = t + A.transpose(0, 1) @ b
    S = S + cfg.ridge * torch.eye(n, device=xi_train.device)
    return torch.linalg.solve(S, t)
