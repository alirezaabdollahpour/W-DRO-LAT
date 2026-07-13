"""Entry point for input-space adversarial training on CIFAR-10."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

# Cluster containers often ship with a tiny /dev/shm (Docker default 64 MB),
# which the default 'file_descriptor' sharing strategy exhausts via the
# DataLoader workers' SemLock + shared-tensor allocations once a few epochs
# have passed. 'file_system' uses regular files in the working tree
# instead — slightly slower but never hits "[Errno 28] No space left on
# device". Set BEFORE the first DataLoader spawns workers.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, AttributeError):
    pass

from . import distributed as dist_helpers
from .algorithms import ALGORITHMS
from .config import build_arg_parser, config_from_args
from .data import get_cifar10_loaders, get_cifar10_train_loader
from .models import load_pretrained_classifier
from .utils import (
    evaluate_clean,
    get_device,
    set_deterministic,
    set_seed_benchmark_mode,
)


_LOG_FIELDS = (
    "run_id",
    "algorithm",
    "epoch",
    "phase",
    "frozen_adversary_map_steps",
    "timestamp",
    "train_loss",
    "train_acc",
    "train_mse",
    "inner_loss",
    "adversary_loss_type",
    "train_adv_loss",
    "train_transport_cost",
    "train_weighted_penalty",
    "train_inner_objective",
    "npf_pgd_anchor_mse",
    "theta_l2_delta",
    "omega_l2_delta",
    "epoch_seconds",
    "bn_recalibration_seconds",
    "bn_recalibration_batches",
    "bn_recalibration_samples",
    "bn_online_refresh_seconds",
    "bn_online_refresh_batches",
    "bn_online_refresh_samples",
    "test_loss",
    "test_acc",
    "clean_loss",
    "clean_acc",
    "adv_loss",
    "adv_acc",
    "adv_penalty",
    "eval_transport_cost",
    "eval_transport_weighted_penalty",
    "eval_transport_avg_l2",
    "eval_transport_avg_linf",
    "eval_transport_max_l2",
    "eval_transport_max_linf",
    "eval_transport_avg_normalized_mse",
    "eval_transport_max_normalized_mse",
    "eval_transport_samples",
    "transport_adv_loss",
    "transport_adv_acc",
    "input_pgd_acc",
    "input_pgd_clean_acc",
    "input_pgd_clean_correct",
    "input_pgd_robust_correct",
    "input_pgd_avg_l2",
    "input_pgd_avg_linf",
    "input_pgd_max_l2",
    "input_pgd_max_linf",
    "input_pgd_avg_normalized_mse",
    "input_pgd_max_normalized_mse",
    "input_pgd_samples",
    "input_pgd_loss",
    "input_pgd_geometry",
    "wrm_normalized_radius_protocol",
    "wrm_pgd_eps_mode",
    "wrm_normalized_radius_requested",
    "wrm_normalized_radius",
    "wrm_cp",
    "wrm_effective_inp_eps",
    "eval_transport_pgd_alignment",
    "eval_transport_pgd_alignment_samples",
    "transport_pgd_align_samples",
    "transport_pgd_align_valid_cos_samples",
    "transport_pgd_cos_mean",
    "transport_pgd_cos_median",
    "transport_pgd_cos_p10",
    "transport_pgd_cos_p90",
    "transport_pgd_transport_acc",
    "transport_pgd_transport_projected_acc",
    "transport_pgd_pgd_acc",
    "transport_pgd_transport_loss",
    "transport_pgd_transport_projected_loss",
    "transport_pgd_pgd_loss",
    "transport_pgd_transport_avg_l2",
    "transport_pgd_transport_projected_avg_l2",
    "transport_pgd_pgd_avg_l2",
    "transport_cost",
    "attack_clean_correct_only",
    "reset_parametric_bb_each_batch",
    "npf_reset_omega_each_batch",
    "npf_project_to_input_ball",
    "npf_pgd_anchor_weight",
    "npf_pgd_anchor_steps",
    "npf_pgd_anchor_restarts",
    "npf_pgd_anchor_loss",
    "parametric_bb_max_grad_norm",
    "lambda_param",
    "lr_theta",
    "batch_size",
    "freeze_batchnorm",
    "freeze_batchnorm_affine",
    "adversary_classifier_eval",
    "online_batchnorm_refresh",
    "batchnorm_online_refresh_momentum",
    "npf_identity_init",
    "npf_lastquad_identity_init",
    "npf_pos_weights",
    "npf_lastquad_pos_weights",
    "npf_positive_weight_rectifier",
    "npf_lastquad_positive_weight_rectifier",
    "seed",
    "pretrained_path",
)


def _append_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in _LOG_FIELDS
        }
    )
    desired_fieldnames = list(_LOG_FIELDS) + extra_fields
    file_exists = path.exists()
    needs_header = not file_exists
    if file_exists:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            needs_header = reader.fieldnames is None
            fieldnames = list(reader.fieldnames or desired_fieldnames)
            existing_rows = list(reader)
        missing_fields = [
            field for field in desired_fieldnames if field not in fieldnames
        ]
        if missing_fields:
            fieldnames.extend(missing_fields)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            tmp_path.replace(path)
            needs_header = False
    else:
        fieldnames = desired_fieldnames
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False))


def _load_resume_payload(path: str) -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--resume-checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"--resume-checkpoint must contain a dict payload: {ckpt_path}")
    if "classifier" not in payload:
        raise KeyError(f"--resume-checkpoint is missing required 'classifier': {ckpt_path}")
    return payload


# The strong-convexity coefficient mu lives in the psi_omega state_dict as the
# outer PosDef diag_kernel, so on resume the checkpoint value silently wins
# over the CLI flag. For npf_lastquad the outer quadratic is frozen (never
# trained), meaning a mismatched resume would run the entire continuation at
# the old mu while every new provenance artifact (summary.json, history CSV,
# RUN_CONFIG env dump) records the new CLI value.
_RESUME_STRONG_CONVEXITY_FIELDS = {
    "npf_lastquad": ("npf_lastquad_strong_convexity",),
    "npf": ("npf_strong_convexity",),
}


def _check_resume_strong_convexity(payload: Dict[str, Any], cfg) -> None:
    stored_config = payload.get("config")
    if not isinstance(stored_config, dict):
        return
    for field in _RESUME_STRONG_CONVEXITY_FIELDS.get(cfg.algorithm, ()):
        if stored_config.get(field) is None:
            continue
        stored = float(stored_config[field])
        requested = float(getattr(cfg, field))
        if stored != requested:
            raise ValueError(
                f"--resume-checkpoint was trained with {field}={stored:g} but "
                f"this run requests {field}={requested:g}. The resumed "
                "psi_omega state dict carries the old outer PosDef diagonal, "
                "so the requested value would be silently ignored while the "
                "run's logs claim it. Re-launch with the matching value, or "
                "train the ablation arm from scratch."
            )


def _metric_definitions(use_margin_loss: bool, transport_cost: str = "normalized_mse") -> Dict[str, str]:
    adv_loss = (
        "Mean logsumexp margin on T_omega(x): logsumexp_{j != y}(logit_j - logit_y)."
        if use_margin_loss
        else "Mean cross-entropy on T_omega(x)."
    )
    cost_label = (
        "legacy normalized-coordinate mean squared distance"
        if str(transport_cost).lower() == "normalized_mse"
        else "pixel-coordinate squared L2"
    )
    return {
        "train_loss": "Outer classifier update loss: cross-entropy on adversarial training inputs.",
        "train_acc": "Outer classifier accuracy on adversarial training inputs.",
        "inner_loss": "Last optimizer-reported inner objective value for the batch, averaged over the epoch.",
        "train_adv_loss": adv_loss,
        "train_transport_cost": f"Mean raw configured transport cost on training batches ({cost_label}).",
        "train_mse": "Backward-compatible alias for train_transport_cost; name kept for old CSV consumers.",
        "train_weighted_penalty": "lambda_param * train_transport_cost.",
        "transport_cost": "Transport penalty convention used by DRO inner objectives.",
        "attack_clean_correct_only": "Whether transport adversaries are trained/applied only on clean-correct examples.",
        "reset_parametric_bb_each_batch": "Whether parametric BB+Armijo history is reset at each batch.",
        "npf_reset_omega_each_batch": "Whether NPF/NPF-LastQuad omega is reset to its identity-adjacent initialization before each batch inner maximization.",
        "npf_project_to_input_ball": "Whether NPF/NPF-LastQuad transports are projected into the configured input-PGD threat set before scoring/updating.",
        "npf_pgd_anchor_weight": "Weight for the optional PGD-anchor term in the NPF/NPF-LastQuad omega objective; 0 disables it.",
        "npf_pgd_anchor_steps": "PGD steps used to build the optional NPF anchor target.",
        "npf_pgd_anchor_restarts": "PGD restarts used to build the optional NPF anchor target.",
        "npf_pgd_anchor_loss": "Loss maximized by the optional NPF PGD anchor target.",
        "npf_pgd_anchor_mse": "Mean normalized-coordinate MSE between final NPF transport and the optional PGD anchor target.",
        "parametric_bb_max_grad_norm": "Global norm clip applied to shared parametric adversary gradients before BB+Armijo; 0 disables clipping.",
        "npf_identity_init": "Whether full NPF applies the identity-adjacent initialization after constructing the potential.",
        "npf_lastquad_identity_init": "Whether NPF-LastQuad applies the identity-adjacent initialization after constructing the potential.",
        "npf_pos_weights": "Whether full NPF projects hidden-to-hidden weights through a nonnegative rectifier.",
        "npf_lastquad_pos_weights": "Whether NPF-LastQuad projects hidden-to-hidden weights through a nonnegative rectifier.",
        "npf_positive_weight_rectifier": "Rectifier used for full NPF positive hidden-to-hidden weights.",
        "npf_lastquad_positive_weight_rectifier": "Rectifier used for NPF-LastQuad positive hidden-to-hidden weights.",
        "train_inner_objective": "train_adv_loss - train_weighted_penalty.",
        "theta_l2_delta": (
            "Epoch-to-epoch classifier parameter L2 change "
            "||theta_t - theta_{t-1}||_2, starting from the loaded checkpoint."
        ),
        "omega_l2_delta": (
            "Epoch-to-epoch NPF adversary parameter L2 change "
            "||omega_t - omega_{t-1}||_2."
        ),
        "clean_loss": "Clean test cross-entropy after the epoch.",
        "clean_acc": "Clean test accuracy after the epoch.",
        "transport_adv_loss": "Test cross-entropy under the learned transport adversary.",
        "transport_adv_acc": "Test accuracy under the learned transport adversary after the epoch.",
        "input_pgd_acc": "Robust test accuracy under the input-space PGD attack after the epoch.",
        "input_pgd_clean_acc": "Clean accuracy on exactly the examples evaluated by input-space PGD.",
        "input_pgd_clean_correct": "Number of PGD-evaluated examples classified correctly before attack.",
        "input_pgd_robust_correct": "Number of PGD-evaluated examples still correct after attack.",
        "input_pgd_avg_l2": "Average pixel-space L2 perturbation norm selected by PGD.",
        "input_pgd_avg_linf": "Average pixel-space Linf perturbation norm selected by PGD.",
        "input_pgd_max_l2": "Maximum pixel-space L2 perturbation norm selected by PGD.",
        "input_pgd_max_linf": "Maximum pixel-space Linf perturbation norm selected by PGD.",
        "input_pgd_avg_normalized_mse": "Average normalized-coordinate MSE selected by PGD.",
        "input_pgd_max_normalized_mse": "Maximum normalized-coordinate MSE selected by PGD.",
        "input_pgd_samples": "Number of examples evaluated by input-space PGD.",
        "input_pgd_loss": "Loss maximized by input-space PGD evaluation: margin or ce.",
        "input_pgd_geometry": "Input-space PGD geometry: standard pixel_l2_squared or legacy normalized_mse.",
        "wrm_normalized_radius_protocol": "Launcher metadata: whether WRM-style normalized PGD radius reporting was enabled.",
        "wrm_pgd_eps_mode": "Launcher metadata: whether the requested PGD budget was normalized rho or raw epsilon_adv.",
        "wrm_normalized_radius_requested": "Launcher metadata: original LOCAL_INP_EPS request before any rho-to-epsilon conversion.",
        "wrm_normalized_radius": "WRM-style reported PGD budget rho = epsilon_adv / C_p.",
        "wrm_cp": "WRM-style normalizer C_p = E_train ||X||_p used for rho-to-epsilon conversion.",
        "wrm_effective_inp_eps": "Raw PGD radius epsilon_adv actually passed as --inp-eps.",
        "eval_transport_pgd_alignment": "Whether the transport-vs-PGD direction alignment diagnostic is enabled.",
        "eval_transport_pgd_alignment_samples": "Sample cap used by the transport-vs-PGD direction alignment diagnostic.",
        "transport_pgd_cos_mean": "Mean cosine between learned transport delta and one-restart PGD delta on the alignment subset.",
        "transport_pgd_cos_median": "Median cosine between learned transport delta and one-restart PGD delta on the alignment subset.",
        "transport_pgd_cos_p10": "10th percentile cosine between learned transport delta and one-restart PGD delta.",
        "transport_pgd_cos_p90": "90th percentile cosine between learned transport delta and one-restart PGD delta.",
        "transport_pgd_transport_projected_acc": "Accuracy after projecting the learned transport direction into the input-PGD ball on the alignment subset.",
        "transport_pgd_pgd_acc": "Accuracy under one-restart PGD on the same alignment subset.",
        "transport_pgd_align_samples": "Number of examples in the transport-vs-PGD alignment diagnostic.",
        "eval_transport_cost": f"Mean raw configured test-set transport cost under transport_for_eval ({cost_label}).",
        "eval_transport_weighted_penalty": "lambda_param * eval_transport_cost.",
        "eval_transport_avg_l2": "Average pixel-space L2 perturbation norm of the learned transport.",
        "eval_transport_avg_linf": "Average pixel-space Linf perturbation norm of the learned transport.",
        "eval_transport_max_l2": "Maximum pixel-space L2 perturbation norm of the learned transport.",
        "eval_transport_max_linf": "Maximum pixel-space Linf perturbation norm of the learned transport.",
        "eval_transport_avg_normalized_mse": "Average normalized-coordinate MSE of the learned transport.",
        "eval_transport_max_normalized_mse": "Maximum normalized-coordinate MSE of the learned transport.",
        "eval_transport_samples": "Number of examples evaluated under the learned transport.",
        "bn_recalibration_seconds": (
            "Wallclock for the optional no-gradient clean-train BatchNorm "
            "running-stat recalibration pass performed after an outer "
            "classifier-update epoch and before evaluation."
        ),
        "bn_recalibration_batches": "Number of clean train batches used for BatchNorm recalibration.",
        "bn_recalibration_samples": "Number of clean train samples used for BatchNorm recalibration.",
        "bn_online_refresh_seconds": (
            "Wallclock for per-batch no-gradient clean BatchNorm running-stat "
            "refreshes performed before adversarial/classifier updates."
        ),
        "bn_online_refresh_batches": "Number of clean train batches used for online BatchNorm refresh.",
        "bn_online_refresh_samples": "Number of clean train samples used for online BatchNorm refresh.",
        "profile_*": (
            "Inner-maximization profiler fields. Timings are CUDA-synchronized "
            "seconds, off by default, and intended for ablation rather than "
            "throughput training. *_per_batch averages over profiled train "
            "batches; *_per_step averages over inner optimizer steps "
            "(BB+Armijo or Muon); *_per_trial averages over Armijo trial "
            "objectives."
        ),
    }


def _enrich_history_entry(
    entry: Dict[str, Any],
    *,
    cfg,
    run_id: str,
    adversary_loss_type: str,
) -> Dict[str, Any]:
    return {
        **entry,
        "run_id": run_id,
        "adversary_loss_type": adversary_loss_type,
        "transport_cost": cfg.transport_cost,
        "attack_clean_correct_only": cfg.attack_clean_correct_only,
        "reset_parametric_bb_each_batch": cfg.reset_parametric_bb_each_batch,
        "npf_reset_omega_each_batch": getattr(cfg, "npf_reset_omega_each_batch", None),
        "npf_project_to_input_ball": getattr(cfg, "npf_project_to_input_ball", None),
        "npf_pgd_anchor_weight": getattr(cfg, "npf_pgd_anchor_weight", None),
        "npf_pgd_anchor_steps": getattr(cfg, "npf_pgd_anchor_steps", None),
        "npf_pgd_anchor_restarts": getattr(cfg, "npf_pgd_anchor_restarts", None),
        "npf_pgd_anchor_loss": getattr(cfg, "npf_pgd_anchor_loss", None),
        "eval_transport_pgd_alignment": getattr(
            cfg, "eval_transport_pgd_alignment", False
        ),
        "eval_transport_pgd_alignment_samples": getattr(
            cfg, "eval_transport_pgd_alignment_samples", None
        ),
        "parametric_bb_max_grad_norm": cfg.parametric_bb_max_grad_norm,
        "npf_identity_init": getattr(cfg, "npf_identity_init", None),
        "npf_lastquad_identity_init": getattr(cfg, "npf_lastquad_identity_init", None),
        "npf_pos_weights": getattr(cfg, "npf_pos_weights", None),
        "npf_lastquad_pos_weights": getattr(cfg, "npf_lastquad_pos_weights", None),
        "npf_positive_weight_rectifier": getattr(cfg, "npf_positive_weight_rectifier", None),
        "npf_lastquad_positive_weight_rectifier": getattr(cfg, "npf_lastquad_positive_weight_rectifier", None),
        "wrm_normalized_radius_protocol": getattr(cfg, "wrm_normalized_radius_protocol", ""),
        "wrm_pgd_eps_mode": getattr(cfg, "wrm_pgd_eps_mode", ""),
        "wrm_normalized_radius_requested": getattr(cfg, "wrm_normalized_radius_requested", None),
        "wrm_normalized_radius": getattr(cfg, "wrm_normalized_radius", None),
        "wrm_cp": getattr(cfg, "wrm_cp", None),
        "wrm_effective_inp_eps": getattr(cfg, "wrm_effective_inp_eps", None),
        "lambda_param": entry.get("lambda_param", cfg.lambda_param),
    }


def _csv_row_from_entry(
    entry: Dict[str, Any],
    *,
    cfg,
    run_id: str,
    adversary_loss_type: str,
) -> Dict[str, Any]:
    row = {
        "run_id": run_id,
        "algorithm": entry.get("algorithm", cfg.algorithm),
        "epoch": entry.get("epoch"),
        "phase": entry.get("phase", "adv"),
        "frozen_adversary_map_steps": entry.get("frozen_adversary_map_steps"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_loss": entry.get("train_loss"),
        "train_acc": entry.get("train_acc"),
        "train_mse": entry.get("train_mse"),
        "inner_loss": entry.get("inner_loss"),
        "adversary_loss_type": adversary_loss_type,
        "train_adv_loss": entry.get("train_adv_loss"),
        "train_transport_cost": entry.get("train_transport_cost"),
        "train_weighted_penalty": entry.get("train_weighted_penalty"),
        "train_inner_objective": entry.get("train_inner_objective"),
        "npf_pgd_anchor_mse": entry.get("npf_pgd_anchor_mse"),
        "theta_l2_delta": entry.get("theta_l2_delta"),
        "omega_l2_delta": entry.get("omega_l2_delta"),
        "epoch_seconds": entry.get("epoch_seconds"),
        "bn_recalibration_seconds": entry.get("bn_recalibration_seconds"),
        "bn_recalibration_batches": entry.get("bn_recalibration_batches"),
        "bn_recalibration_samples": entry.get("bn_recalibration_samples"),
        "bn_online_refresh_seconds": entry.get("bn_online_refresh_seconds"),
        "bn_online_refresh_batches": entry.get("bn_online_refresh_batches"),
        "bn_online_refresh_samples": entry.get("bn_online_refresh_samples"),
        "test_loss": entry.get("test_loss"),
        "test_acc": entry.get("test_acc"),
        "clean_loss": entry.get("clean_loss", entry.get("test_loss")),
        "clean_acc": entry.get("clean_acc", entry.get("test_acc")),
        "adv_loss": entry.get("adv_loss"),
        "adv_acc": entry.get("adv_acc"),
        "adv_penalty": entry.get("adv_penalty"),
        "eval_transport_cost": entry.get("eval_transport_cost"),
        "eval_transport_weighted_penalty": entry.get("eval_transport_weighted_penalty"),
        "eval_transport_avg_l2": entry.get("eval_transport_avg_l2"),
        "eval_transport_avg_linf": entry.get("eval_transport_avg_linf"),
        "eval_transport_max_l2": entry.get("eval_transport_max_l2"),
        "eval_transport_max_linf": entry.get("eval_transport_max_linf"),
        "eval_transport_avg_normalized_mse": entry.get("eval_transport_avg_normalized_mse"),
        "eval_transport_max_normalized_mse": entry.get("eval_transport_max_normalized_mse"),
        "eval_transport_samples": entry.get("eval_transport_samples"),
        "transport_adv_loss": entry.get("transport_adv_loss", entry.get("adv_loss")),
        "transport_adv_acc": entry.get("transport_adv_acc", entry.get("adv_acc")),
        "input_pgd_acc": entry.get("input_pgd_acc"),
        "input_pgd_clean_acc": entry.get("input_pgd_clean_acc"),
        "input_pgd_clean_correct": entry.get("input_pgd_clean_correct"),
        "input_pgd_robust_correct": entry.get("input_pgd_robust_correct"),
        "input_pgd_avg_l2": entry.get("input_pgd_avg_l2"),
        "input_pgd_avg_linf": entry.get("input_pgd_avg_linf"),
        "input_pgd_max_l2": entry.get("input_pgd_max_l2"),
        "input_pgd_max_linf": entry.get("input_pgd_max_linf"),
        "input_pgd_avg_normalized_mse": entry.get("input_pgd_avg_normalized_mse"),
        "input_pgd_max_normalized_mse": entry.get("input_pgd_max_normalized_mse"),
        "input_pgd_samples": entry.get("input_pgd_samples"),
        "input_pgd_loss": entry.get("input_pgd_loss", cfg.input_pgd_loss),
        "input_pgd_geometry": entry.get(
            "input_pgd_geometry", getattr(cfg, "input_pgd_geometry", "pixel_l2_squared")
        ),
        "wrm_normalized_radius_protocol": entry.get(
            "wrm_normalized_radius_protocol",
            getattr(cfg, "wrm_normalized_radius_protocol", ""),
        ),
        "wrm_pgd_eps_mode": entry.get(
            "wrm_pgd_eps_mode", getattr(cfg, "wrm_pgd_eps_mode", "")
        ),
        "wrm_normalized_radius_requested": entry.get(
            "wrm_normalized_radius_requested",
            getattr(cfg, "wrm_normalized_radius_requested", None),
        ),
        "wrm_normalized_radius": entry.get(
            "wrm_normalized_radius", getattr(cfg, "wrm_normalized_radius", None)
        ),
        "wrm_cp": entry.get("wrm_cp", getattr(cfg, "wrm_cp", None)),
        "wrm_effective_inp_eps": entry.get(
            "wrm_effective_inp_eps", getattr(cfg, "wrm_effective_inp_eps", None)
        ),
        "eval_transport_pgd_alignment": getattr(
            cfg, "eval_transport_pgd_alignment", False
        ),
        "eval_transport_pgd_alignment_samples": getattr(
            cfg, "eval_transport_pgd_alignment_samples", None
        ),
        "transport_pgd_align_samples": entry.get("transport_pgd_align_samples"),
        "transport_pgd_align_valid_cos_samples": entry.get("transport_pgd_align_valid_cos_samples"),
        "transport_pgd_cos_mean": entry.get("transport_pgd_cos_mean"),
        "transport_pgd_cos_median": entry.get("transport_pgd_cos_median"),
        "transport_pgd_cos_p10": entry.get("transport_pgd_cos_p10"),
        "transport_pgd_cos_p90": entry.get("transport_pgd_cos_p90"),
        "transport_pgd_transport_acc": entry.get("transport_pgd_transport_acc"),
        "transport_pgd_transport_projected_acc": entry.get("transport_pgd_transport_projected_acc"),
        "transport_pgd_pgd_acc": entry.get("transport_pgd_pgd_acc"),
        "transport_pgd_transport_loss": entry.get("transport_pgd_transport_loss"),
        "transport_pgd_transport_projected_loss": entry.get("transport_pgd_transport_projected_loss"),
        "transport_pgd_pgd_loss": entry.get("transport_pgd_pgd_loss"),
        "transport_pgd_transport_avg_l2": entry.get("transport_pgd_transport_avg_l2"),
        "transport_pgd_transport_projected_avg_l2": entry.get("transport_pgd_transport_projected_avg_l2"),
        "transport_pgd_pgd_avg_l2": entry.get("transport_pgd_pgd_avg_l2"),
        "transport_cost": cfg.transport_cost,
        "attack_clean_correct_only": cfg.attack_clean_correct_only,
        "reset_parametric_bb_each_batch": cfg.reset_parametric_bb_each_batch,
        "npf_reset_omega_each_batch": getattr(cfg, "npf_reset_omega_each_batch", None),
        "npf_project_to_input_ball": getattr(cfg, "npf_project_to_input_ball", None),
        "npf_pgd_anchor_weight": getattr(cfg, "npf_pgd_anchor_weight", None),
        "npf_pgd_anchor_steps": getattr(cfg, "npf_pgd_anchor_steps", None),
        "npf_pgd_anchor_restarts": getattr(cfg, "npf_pgd_anchor_restarts", None),
        "npf_pgd_anchor_loss": getattr(cfg, "npf_pgd_anchor_loss", None),
        "parametric_bb_max_grad_norm": cfg.parametric_bb_max_grad_norm,
        "lambda_param": entry.get("lambda_param", cfg.lambda_param),
        "lr_theta": cfg.lr_theta,
        "batch_size": cfg.batch_size,
        "freeze_batchnorm": cfg.freeze_batchnorm,
        "freeze_batchnorm_affine": cfg.freeze_batchnorm_affine,
        "adversary_classifier_eval": cfg.adversary_classifier_eval,
        "online_batchnorm_refresh": cfg.online_batchnorm_refresh,
        "batchnorm_online_refresh_momentum": cfg.batchnorm_online_refresh_momentum,
        "npf_identity_init": getattr(cfg, "npf_identity_init", None),
        "npf_lastquad_identity_init": getattr(cfg, "npf_lastquad_identity_init", None),
        "npf_pos_weights": getattr(cfg, "npf_pos_weights", None),
        "npf_lastquad_pos_weights": getattr(cfg, "npf_lastquad_pos_weights", None),
        "npf_positive_weight_rectifier": getattr(cfg, "npf_positive_weight_rectifier", None),
        "npf_lastquad_positive_weight_rectifier": getattr(cfg, "npf_lastquad_positive_weight_rectifier", None),
        "seed": cfg.seed,
        "pretrained_path": cfg.pretrained_path,
    }
    row.update({k: v for k, v in entry.items() if k.startswith("profile_")})
    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)

    # Initialise DDP if torchrun set the env vars; safe no-op otherwise.
    dist_info = dist_helpers.init_distributed()
    is_main = dist_info.is_main

    if cfg.benchmark_mode:
        set_seed_benchmark_mode(cfg.seed)
        if is_main:
            print(
                f"[input-icnn] BENCHMARK MODE — cudnn.benchmark=True, "
                f"determinism off, seed={cfg.seed}"
            )
    else:
        set_deterministic(cfg.seed)

    if dist_info.is_distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_info.local_rank}")
    else:
        device = get_device()

    if is_main:
        print(
            f"[input-icnn] device={device}, algorithm={cfg.algorithm}, "
            f"lambda={cfg.lambda_param}, world_size={dist_info.world_size}"
        )

    train_loader, test_loader, train_sampler = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_root=cfg.data_dir,
        augment_train=cfg.augment_train,
        seed=cfg.seed,
        world_size=dist_info.world_size,
        rank=dist_info.rank,
    )
    bn_recalibration_loader = None
    bn_recalibration_sampler = None
    if cfg.recalibrate_batchnorm:
        bn_recalibration_loader, bn_recalibration_sampler = get_cifar10_train_loader(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_root=cfg.data_dir,
            augment_train=False,
            seed=cfg.seed,
            world_size=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=False,
            drop_last=False,
        )

    classifier = load_pretrained_classifier(
        pretrained_path=cfg.pretrained_path,
        num_classes=10,
        strict=cfg.pretrained_strict,
        device=device,
    ).to(device)

    resume_payload = None
    if cfg.resume_checkpoint:
        resume_payload = _load_resume_payload(cfg.resume_checkpoint)
        classifier.load_state_dict(resume_payload["classifier"], strict=True)
        if is_main:
            print(f"[input-icnn] resumed classifier from {cfg.resume_checkpoint}")

    if is_main:
        sanity_loss, sanity_acc = evaluate_clean(classifier, test_loader, device)
        print(f"[sanity] clean test  loss={sanity_loss:.4f}  acc={sanity_acc*100:.2f}%")
    dist_helpers.barrier()

    trainer_cls = ALGORITHMS[cfg.algorithm]
    trainer = trainer_cls(
        classifier=classifier,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        config=cfg,
        train_sampler=train_sampler,
        bn_recalibration_loader=bn_recalibration_loader,
        bn_recalibration_sampler=bn_recalibration_sampler,
    )

    if resume_payload is not None:
        _check_resume_strong_convexity(resume_payload, cfg)
        trainer.load_adversary_state_dicts(resume_payload, strict=True)
        if is_main:
            print(f"[input-icnn] resumed adversary from {cfg.resume_checkpoint}")
    dist_helpers.barrier()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    adversary_loss_type = "logsumexp_margin" if cfg.use_margin_loss else "cross_entropy"
    if is_main:
        def write_epoch_row(entry: Dict[str, Any]) -> None:
            enriched = _enrich_history_entry(
                entry,
                cfg=cfg,
                run_id=run_id,
                adversary_loss_type=adversary_loss_type,
            )
            row = _csv_row_from_entry(
                enriched,
                cfg=cfg,
                run_id=run_id,
                adversary_loss_type=adversary_loss_type,
            )
            _append_csv(Path(cfg.log_csv), [row])

        trainer.epoch_end_callback = write_epoch_row

    history = trainer.fit()
    trainer.save_final()

    # Capture the GPU peak memory from THIS rank's view. ``max_memory_allocated``
    # is per-device, so each rank reports its own peak — they should be near-
    # identical in DDP since each rank holds an identical model + per-rank
    # batch. We record rank 0's number in the summary; that's the
    # representative single-GPU memory footprint for the algorithm.
    peak_gpu_alloc_mb: float = float("nan")
    peak_gpu_reserved_mb: float = float("nan")
    if torch.cuda.is_available():
        peak_gpu_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # Only rank 0 writes CSV / summary; other ranks finalise DDP and exit.
    if not is_main:
        dist_helpers.cleanup_distributed()
        return

    history_epochs = [
        _enrich_history_entry(
            entry,
            cfg=cfg,
            run_id=run_id,
            adversary_loss_type=adversary_loss_type,
        )
        for entry in history
    ]
    metric_definitions = _metric_definitions(cfg.use_margin_loss, cfg.transport_cost)
    history_payload = {
        "run_id": run_id,
        "algorithm": cfg.algorithm,
        "adversary_loss_type": adversary_loss_type,
        "lambda_param": cfg.lambda_param,
        "metric_definitions": metric_definitions,
        "epochs": history_epochs,
        "config": cfg.to_dict(),
    }
    history_path = Path(cfg.log_csv).with_name("history.json")
    _write_json(history_path, history_payload)
    summary = {
        "run_id": run_id,
        "algorithm": cfg.algorithm,
        "adversary_loss_type": adversary_loss_type,
        "best_robust_acc": trainer.best_robust_acc,
        "best_robust_epoch": trainer.best_robust_epoch,
        "epochs_completed": trainer.last_completed_epoch,
        "peak_gpu_alloc_mb": peak_gpu_alloc_mb,
        "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
        "csv": cfg.log_csv,
        "history_json": str(history_path),
        "metric_definitions": metric_definitions,
        "final_epoch": history_epochs[-1] if history_epochs else None,
        "config": cfg.to_dict(),
    }
    # Always write a summary next to the CSV so the analyzer can find it
    # even when --save was empty (GPU memory is the new RSS replacement).
    summary_path = Path(cfg.log_csv).with_name("summary.json")
    _write_json(summary_path, summary)
    if cfg.save:
        save_summary_path = Path(cfg.save).with_suffix(".summary.json")
        save_history_path = Path(cfg.save).with_suffix(".history.json")
        _write_json(save_summary_path, summary)
        _write_json(save_history_path, history_payload)
    print(
        f"[input-icnn] done. log={cfg.log_csv}  history={history_path}  "
        f"best_robust={trainer.best_robust_acc}  "
        f"gpu_peak={peak_gpu_alloc_mb:.1f} MB"
    )
    dist_helpers.cleanup_distributed()


if __name__ == "__main__":
    main()
