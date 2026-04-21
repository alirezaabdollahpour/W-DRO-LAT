"""Algorithm registry + builders.

Separates:

* ``SHARED_ALGORITHMS`` — algorithms independent of ε_ent (SAA, WRM, ICNN, NPF,
  NN-DRO, PPA). These are trained once and evaluated for every ε_ent value.
* ``EPS_ENT_ALGORITHMS`` — algorithms whose hyperparameters depend on ε_ent
  (Dual, WGF, WFR, SVG, RGO). Re-trained for each ε_ent value.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch

from algorithms.base import BaseLinearDRO
from algorithms.dual import SDRO_Dual
from algorithms.icnn import ICNNDRO
from algorithms.nn_dro import NNDRO
from algorithms.npf import NPF
from algorithms.ppa import PPA
from algorithms.rgo import SDRO_RGO
from algorithms.saa import SAA
from algorithms.svg import SDRO_SVG
from algorithms.wfr import SDRO_WFR
from algorithms.wgf import SDRO_WGF
from algorithms.wrm import WRM


# Algorithms whose training is independent of ε_ent.
SHARED_ALGORITHMS = ("SAA", "WRM", "ICNN", "NPF", "NN-DRO", "PPA")

# Algorithms whose ε_ent is a training-time hyperparameter.
EPS_ENT_ALGORITHMS = ("Dual", "WGF", "WFR", "SVG", "RGO")

# Order shown in Figure-6 panels.
ALL_ALGORITHMS = (
    "SAA",
    "Dual",
    "WRM",
    "WGF",
    "WFR",
    "SVG",
    "RGO",
    "ICNN",
    "NPF",
    "NN-DRO",
    "PPA",
)


Builder = Callable[..., BaseLinearDRO]


def _build_saa(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SAA(
        input_dim,
        num_classes,
        device=device.type,
        max_itr=cfg.epochs,
        learning_rate=cfg.saa_lr,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )


def _build_wrm(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return WRM(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        inner_lr=cfg.inner_lr,
        inner_itr=cfg.inner_steps,
    )


def _build_icnn(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return ICNNDRO(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        icnn_hidden=tuple(cfg.icnn_hidden),
        icnn_strong_convexity=cfg.icnn_strong_convexity,
        icnn_softplus_beta=cfg.icnn_softplus_beta,
        icnn_init_mode=cfg.icnn_init_mode,
        lr_B=cfg.icnn_lr_B,
        weight_decay_B=cfg.icnn_weight_decay_B,
        omega_steps_per_batch=cfg.icnn_omega_steps_per_batch,
        bb_alpha0=cfg.icnn_bb_alpha0,
        bb_alpha_min=cfg.icnn_bb_alpha_min,
        bb_alpha_max=cfg.icnn_bb_alpha_max,
        bb_ls_c=cfg.icnn_bb_ls_c,
        bb_ls_shrink=cfg.icnn_bb_ls_shrink,
        bb_ls_max_steps=cfg.icnn_bb_ls_max_steps,
        max_itr=cfg.epochs,
        batch_size=cfg.batch_size,
    )


def _build_npf(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return NPF(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        npf_hidden=tuple(cfg.npf_hidden),
        npf_outer_rank=cfg.npf_outer_rank,
        npf_inner_rank=cfg.npf_inner_rank,
        npf_activation=cfg.npf_activation,
        npf_elu_alpha=cfg.npf_elu_alpha,
        npf_softplus_beta=cfg.npf_softplus_beta,
        npf_init_eps=cfg.npf_init_eps,
        inner_steps_npf=cfg.inner_steps_npf,
        inner_lr_npf=cfg.inner_lr_npf,
        lr_B=cfg.npf_lr_B,
        weight_decay_B=cfg.npf_weight_decay_B,
        max_itr=cfg.epochs,
        batch_size=cfg.batch_size,
    )


def _build_nn_dro(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return NNDRO(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        nn_dro_hidden=tuple(cfg.nn_dro_hidden),
        nn_dro_activation=cfg.nn_dro_activation,
        nn_dro_softplus_beta=cfg.nn_dro_softplus_beta,
        nn_dro_init_scale=cfg.nn_dro_init_scale,
        omega_steps_per_batch=cfg.nn_dro_omega_steps_per_batch,
        inner_lr=cfg.nn_dro_inner_lr,
        lr_B=cfg.nn_dro_lr_B,
        weight_decay_B=cfg.nn_dro_weight_decay_B,
        max_itr=cfg.epochs,
        batch_size=cfg.batch_size,
    )


def _build_ppa(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return PPA(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        inner_lr=cfg.inner_lr,
        inner_itr=cfg.inner_steps,
        ppa_num_rounds=cfg.ppa_num_rounds,
        ppa_min_rounds=cfg.ppa_min_rounds,
        ppa_refine_steps=cfg.ppa_refine_steps,
        ppa_refine_lr=cfg.ppa_refine_lr,
        ppa_delta_rtol=cfg.ppa_delta_rtol,
    )


def _build_dual(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SDRO_Dual(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        epsilon=eps_ent,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        sample_level=cfg.sample_level,
    )


def _build_wgf(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SDRO_WGF(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        epsilon=eps_ent,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        inner_lr=cfg.inner_lr,
        inner_itr=cfg.inner_steps,
    )


def _build_wfr(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SDRO_WFR(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        epsilon=eps_ent,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        inner_lr=cfg.inner_lr,
        inner_itr=cfg.inner_steps,
    )


def _build_svg(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SDRO_SVG(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        epsilon=eps_ent,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        inner_lr=cfg.inner_lr,
        inner_itr=cfg.inner_steps,
    )


def _build_rgo(cfg, input_dim: int, num_classes: int, device: torch.device, eps_ent: float) -> BaseLinearDRO:
    return SDRO_RGO(
        input_dim,
        num_classes,
        device=device.type,
        lambda_param=cfg.lambda_param,
        epsilon=eps_ent,
        max_itr=cfg.epochs,
        learning_rate=cfg.dro_lr,
        batch_size=cfg.batch_size,
        num_samples=cfg.num_samples,
        rgo_inner_lr=cfg.inner_lr,
        rgo_inner_steps=cfg.inner_steps,
        rgo_vectorized_max_trials=cfg.rgo_vectorized_max_trials,
    )


BUILDERS: Dict[str, Builder] = {
    "SAA": _build_saa,
    "WRM": _build_wrm,
    "ICNN": _build_icnn,
    "NPF": _build_npf,
    "NN-DRO": _build_nn_dro,
    "PPA": _build_ppa,
    "Dual": _build_dual,
    "WGF": _build_wgf,
    "WFR": _build_wfr,
    "SVG": _build_svg,
    "RGO": _build_rgo,
}


def build_algorithm(
    name: str,
    cfg,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    eps_ent: float,
) -> BaseLinearDRO:
    key = name if name in BUILDERS else None
    if key is None:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {sorted(BUILDERS.keys())}"
        )
    return BUILDERS[key](cfg, input_dim, num_classes, device, eps_ent)
