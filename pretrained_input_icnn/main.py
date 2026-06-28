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
    "input_pgd_samples",
    "input_pgd_loss",
    "transport_cost",
    "attack_clean_correct_only",
    "reset_parametric_bb_each_batch",
    "parametric_bb_max_grad_norm",
    "lambda_param",
    "lr_theta",
    "batch_size",
    "freeze_batchnorm",
    "freeze_batchnorm_affine",
    "online_batchnorm_refresh",
    "batchnorm_online_refresh_momentum",
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
        "parametric_bb_max_grad_norm": "Global norm clip applied to shared parametric adversary gradients before BB+Armijo; 0 disables clipping.",
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
        "input_pgd_samples": "Number of examples evaluated by input-space PGD.",
        "input_pgd_loss": "Loss maximized by input-space PGD evaluation: margin or ce.",
        "eval_transport_cost": f"Mean raw configured test-set transport cost under transport_for_eval ({cost_label}).",
        "eval_transport_weighted_penalty": "lambda_param * eval_transport_cost.",
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
        "parametric_bb_max_grad_norm": cfg.parametric_bb_max_grad_norm,
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
        "input_pgd_samples": entry.get("input_pgd_samples"),
        "input_pgd_loss": entry.get("input_pgd_loss", cfg.input_pgd_loss),
        "transport_cost": cfg.transport_cost,
        "attack_clean_correct_only": cfg.attack_clean_correct_only,
        "reset_parametric_bb_each_batch": cfg.reset_parametric_bb_each_batch,
        "parametric_bb_max_grad_norm": cfg.parametric_bb_max_grad_norm,
        "lambda_param": entry.get("lambda_param", cfg.lambda_param),
        "lr_theta": cfg.lr_theta,
        "batch_size": cfg.batch_size,
        "freeze_batchnorm": cfg.freeze_batchnorm,
        "freeze_batchnorm_affine": cfg.freeze_batchnorm_affine,
        "online_batchnorm_refresh": cfg.online_batchnorm_refresh,
        "batchnorm_online_refresh_momentum": cfg.batchnorm_online_refresh_momentum,
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
