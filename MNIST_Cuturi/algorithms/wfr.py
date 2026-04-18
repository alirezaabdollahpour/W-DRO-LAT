"""WFR (Wasserstein-Fisher-Rao) DRO baseline."""
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
    weighted_accuracy_from_logits,
)
from utils.samplers import wfr_sampler


def train_step_wfr(
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
    adv_x, weights = wfr_sampler(
        x, y, model, cfg.lambda_reg, cfg.particle_epsilon,
        cfg.particle_num_samples, cfg.particle_inner_steps,
        cfg.particle_inner_lr, clamp=(0.0, 1.0),
    )
    repeated_y = y.repeat_interleave(cfg.particle_num_samples, dim=0)

    set_requires_grad(model, True)
    model.train()
    with torch.enable_grad():
        logits_adv = model(adv_x)
        per_particle_loss = F.cross_entropy(
            logits_adv, repeated_y, reduction="none"
        ).view(-1, cfg.particle_num_samples)
        loss = (per_particle_loss * weights).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_clean = model(x)
        logits_adv_post = model(adv_x)
        acc_clean = accuracy(logits_clean, y)
        acc_adv = weighted_accuracy_from_logits(
            logits_adv_post, repeated_y, weights
        )
        x_anchor = repeat_particles(x, cfg.particle_num_samples)
        dist_sq = (
            ((adv_x - x_anchor) ** 2)
            .view(x.size(0), cfg.particle_num_samples, -1)
            .sum(dim=2)
        )
        w2_proxy = (dist_sq * weights).sum(dim=1).mean()
    return state, {
        "loss_adv": loss.detach(),
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "w2_proxy": w2_proxy,
    }


def train_algorithm_wfr(
    cfg: TrainConfig, device: torch.device
) -> Tuple[TrainState, Dict[str, Any]]:
    return train_algorithm_generic_adv("wfr", "WFR", cfg, device, train_step_wfr)
