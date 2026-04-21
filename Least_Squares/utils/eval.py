"""Evaluate trained thetas on the delta sweep."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from config import ULSConfig
from utils.common import to_numpy
from utils.data import generate_test_data
from utils.loss import loss_function


def evaluate_delta_sweep(
    models: Dict[str, torch.Tensor],
    cfg: ULSConfig,
    A0: torch.Tensor,
    A1: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, Dict[str, List[float]]]:
    delta_values = torch.linspace(
        cfg.delta_min, cfg.delta_max, cfg.delta_steps, device=device
    )
    results: Dict[str, List[float]] = {k: [] for k in models.keys()}

    for delta in to_numpy(delta_values):
        xi_test = generate_test_data(cfg, float(delta), device)
        for name, theta in models.items():
            with torch.no_grad():
                test_loss = (
                    loss_function(theta, xi_test, A0, A1, b, cfg.dim_m).mean().item()
                )
            results[name].append(float(test_loss))
    return to_numpy(delta_values), results
