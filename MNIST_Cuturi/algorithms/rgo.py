"""RGO (Restricted Gaussian Oracle) DRO baseline."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from algorithms.base import TrainState, train_algorithm_generic_adv
from config import TrainConfig
from utils.common import (
    accuracy,
    repeat_particles,
    set_requires_grad,
)
from utils.samplers import rgo_sampler


def train_step_rgo(
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

    set_requires_grad(model, False)
    model.eval()
    adv_x = rgo_sampler(
        x, y, model, cfg.lambda_reg, cfg.rgo_epsilon,
        cfg.rgo_num_samples, cfg.rgo_inner_steps, cfg.rgo_inner_lr,
        cfg.rgo_vectorized_max_trials, clamp=(0.0, 1.0),
    )
    repeated_y = y.repeat_interleave(cfg.rgo_num_samples, dim=0)

    set_requires_grad(model, True)
    model.train()
    with torch.enable_grad():
        logits_adv = model(adv_x)
        loss = F.cross_entropy(logits_adv, repeated_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = accuracy(logits_adv_post, repeated_y)
        x_anchor = repeat_particles(x, cfg.rgo_num_samples)
        w2_proxy = (
            ((adv_x - x_anchor) ** 2).view(adv_x.size(0), -1).sum(dim=1).mean()
        )
    return state, {
        "loss_adv": loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
    }


def train_algorithm_rgo(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, Any]]:
    return train_algorithm_generic_adv("rgo", "RGO", cfg, device, train_step_rgo)
