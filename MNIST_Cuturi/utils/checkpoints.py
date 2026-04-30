"""Checkpoint saving for each algorithm."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from algorithms.algo2_icnn import ICNNState
    from algorithms.base import TrainState
    from algorithms.npf import NPFICNNState
    from config import TrainConfig


CHECKPOINT_DIR = os.path.join("MNIST_checkpoint")


def _lambda_str(lam: float) -> str:
    return f"{lam:g}"


def save_checkpoint_algo1(state: "TrainState", cfg: "TrainConfig") -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"algo1_wrm_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "algo1_wrm",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[Algo1] Checkpoint saved to {path}")
    return path


def save_checkpoint_algo2(
    state: "TrainState", icnn_state: "ICNNState", cfg: "TrainConfig"
) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hidden_str = "_".join(str(h) for h in cfg.icnn_hidden_sizes)
    fname = (
        f"algo2_icnn_lambda{_lambda_str(cfg.lambda_reg)}_hidden{hidden_str}.pt"
    )
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "algo2_icnn",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "icnn_model_state_dict": icnn_state.model.state_dict(),
        "icnn_params_vec": icnn_state.params_vec,
        "lambda_reg": cfg.lambda_reg,
        "icnn_hidden_sizes": list(cfg.icnn_hidden_sizes),
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[Algo2] Checkpoint saved to {path}")
    return path


def save_checkpoint_npf(
    state: "TrainState", icnn_state: "NPFICNNState", cfg: "TrainConfig"
) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hidden_str = "_".join(str(h) for h in cfg.npf_hidden_sizes)
    fname = (
        f"npf_lambda{_lambda_str(cfg.lambda_reg)}"
        f"_rout{cfg.npf_outer_rank}_rin{cfg.npf_inner_rank}"
        f"_hidden{hidden_str}.pt"
    )
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "npf_icnn",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "icnn_model_state_dict": icnn_state.model.state_dict(),
        "icnn_params_vec": icnn_state.params_vec,
        "lambda_reg": cfg.lambda_reg,
        "npf_hidden_sizes": list(cfg.npf_hidden_sizes),
        "npf_outer_rank": cfg.npf_outer_rank,
        "npf_inner_rank": cfg.npf_inner_rank,
        "npf_activation": cfg.npf_activation,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[NPF]   Checkpoint saved to {path}")
    return path


def save_checkpoint_ppa(state: "TrainState", cfg: "TrainConfig") -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"ppa_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "ppa_projected_wrm",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[PPA]   Checkpoint saved to {path}")
    return path


def save_checkpoint_new_ppa(state: "TrainState", cfg: "TrainConfig") -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"new_ppa_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": "new_ppa_free_weight_projected_wrm",
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[New_PPA] Checkpoint saved to {path}")
    return path


def save_checkpoint_simple(
    state: "TrainState",
    cfg: "TrainConfig",
    algorithm_key: str,
    display_name: str,
) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fname = f"{algorithm_key}_lambda{_lambda_str(cfg.lambda_reg)}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save({
        "algorithm": algorithm_key,
        "display_name": display_name,
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.opt.state_dict(),
        "lambda_reg": cfg.lambda_reg,
        "hyperparameters": asdict(cfg),
    }, path)
    print(f"[{display_name}] Checkpoint saved to {path}")
    return path
