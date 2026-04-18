"""Dual DRO baseline (log-sum-exp dual objective)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from algorithms.base import TrainState, train_algorithm_generic_adv
from config import TrainConfig
from utils.common import accuracy
from utils.samplers import dual_dro_loss


def train_step_dual(
    state: TrainState, batch, cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    model = state.model
    opt = state.opt
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    if x.size(0) == 0:
        zero = torch.tensor(0.0, device=device)
        return state, {
            "loss_adv": zero, "acc_clean": zero,
            "acc_adv": zero, "w2_proxy": zero,
        }

    model.train()
    with torch.enable_grad():
        loss, noisy_x, repeated_y = dual_dro_loss(
            model, x, y, cfg.lambda_reg, cfg.dual_epsilon, cfg.dual_sample_level
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        logits_adv = model(noisy_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv, repeated_y)
        factor = noisy_x.size(0) // max(x.size(0), 1)
        w2_proxy = (
            (noisy_x - x.repeat_interleave(factor, dim=0)) ** 2
        ).view(noisy_x.size(0), -1).sum(dim=1).mean()
    return state, {
        "loss_adv": loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
    }


def train_algorithm_dual(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, Any]]:
    return train_algorithm_generic_adv("dual", "Dual", cfg, device, train_step_dual)
