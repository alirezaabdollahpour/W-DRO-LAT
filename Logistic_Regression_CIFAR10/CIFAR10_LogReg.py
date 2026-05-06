"""CIFAR-10 adversarial multiclass logistic regression entry point.

Reproduces Section 6.3 / Figure 6 of "GRADIENT FLOW SAMPLER-BASED
DISTRIBUTIONALLY ROBUST OPTIMIZATION" on CIFAR-10 ResNet-50 features (512-dim)
under an l2-PGD test-time attack.
The attack is adaptive feature-space l2-PGD over the current perturbation
delta with five restarts by default.

Competitors
-----------
  * SAA / ERM
  * Sinkhorn SDRO Dual
  * WRM (Sinha et al.)
  * RO (Madry-style l2-PGD robust optimisation)
  * WGF / WFR / SVG samplers
  * RGO (rejection sampling)
  * ICNN-DRO (Brenier transport-map adversary, our method)
  * NPF (Vesseron & Cuturi, 2024 — ICNN with per-layer quadratic injections)
  * NN-DRO (vanilla-MLP adversary, no gradient-of-potential)
  * PPA (Projected Particle Ascent)

Usage
-----
    python CIFAR10_LogReg.py --help
    python CIFAR10_LogReg.py
    python CIFAR10_LogReg.py --methods SAA WRM RO ICNN NPF NN-DRO PPA
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from algorithms.registry import (
    ALL_ALGORITHMS,
    EPS_ENT_ALGORITHMS,
    SHARED_ALGORITHMS,
    build_algorithm,
)
from config import build_arg_parser, config_from_args
from utils.common import get_device, set_seed
from utils.eval import evaluate_model
from utils.features import load_or_extract_cifar10_resnet50_features
from utils.pgd import pgd_attack
from utils.plotting import plot_figure6_panels


def _json_safe_results(results: Dict[float, float]) -> Dict[str, float]:
    return {f"{k:.10g}": float(v) for k, v in results.items()}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)

    device = get_device()
    print(f"Using device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Feature cache path (absolute or relative to out_dir)
    feature_cache_path = Path(cfg.feature_cache)
    if not feature_cache_path.is_absolute():
        feature_cache_path = out_dir / feature_cache_path

    train_features, train_labels, test_features, test_labels = (
        load_or_extract_cifar10_resnet50_features(
            feature_cache_path,
            data_dir=cfg.data_dir,
            batch_size=cfg.feature_batch_size,
            num_workers=cfg.num_workers,
            device=device,
            force_reextract=cfg.force_reextract,
        )
    )

    train_features_np = train_features.numpy()
    train_labels_np = train_labels.numpy()
    test_features_np = test_features.numpy()
    test_labels_np = test_labels.numpy()

    input_dim = int(train_features_np.shape[1])
    num_classes = 10

    print(f"\nFeature dim: {input_dim} (expected 512)")
    print(
        f"Training size: {train_features_np.shape[0]}, "
        f"Test size: {test_features_np.shape[0]}"
    )
    print(f"tau={cfg.tau:g} -> lambda={cfg.lambda_param:g}")

    # PGD evaluation radii: ε_attack = Δ · E[||x||_2]
    avg_train_feature_norm = float(np.mean(np.linalg.norm(train_features_np, axis=1)))
    avg_feature_norm = float(np.mean(np.linalg.norm(test_features_np, axis=1)))
    if cfg.ro_epsilon is None:
        cfg = replace(
            cfg,
            ro_epsilon=float(cfg.ro_epsilon_level) * avg_train_feature_norm,
        )
    perturbation_levels = list(cfg.perturbation_levels)
    epsilon_attack_values = [
        float(level) * avg_feature_norm for level in perturbation_levels
    ]

    print(f"Average train feature ||x||_2: {avg_train_feature_norm:.4f}")
    print(f"Average test feature ||x||_2: {avg_feature_norm:.4f}")
    print(
        "Evaluation attack: adaptive feature-space l2-PGD "
        f"with steps={cfg.pgd_steps}, restarts={cfg.pgd_restarts}"
    )

    requested_methods = list(cfg.methods)
    unknown = [m for m in requested_methods if m not in ALL_ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unknown methods requested: {unknown}. Available: {list(ALL_ALGORITHMS)}"
        )
    if "RO" in requested_methods:
        print(
            "RO training radius: "
            f"epsilon={cfg.ro_epsilon:.6g}, "
            f"level={cfg.ro_epsilon_level:g}, "
            f"pgd_steps={cfg.ro_pgd_steps}, "
            f"pgd_restarts={cfg.ro_pgd_restarts}"
        )
    if "NPF" in requested_methods:
        print(
            "NPF architecture: "
            f"hidden={list(cfg.npf_hidden)}, "
            f"outer_rank={cfg.npf_outer_rank}, "
            f"inner_rank={cfg.npf_inner_rank}, "
            f"activation={cfg.npf_activation}"
        )

    # Train shared (ε_ent-independent) algorithms once.
    shared_models: Dict[str, Any] = {}
    train_timings: Dict[str, float] = {}
    for name in SHARED_ALGORITHMS:
        if name not in requested_methods:
            continue
        print(f"\n--- Training {name} ---")
        train_t0 = time.perf_counter()
        model_obj = build_algorithm(
            name, cfg, input_dim, num_classes, device, eps_ent=0.0
        )
        model_obj.fit(
            train_features_np,
            train_labels_np,
            checkpoint_dir=str(checkpoint_dir),
            run_id=0,
        )
        train_timings[name] = time.perf_counter() - train_t0
        print(f"Training time {name}: {train_timings[name]:.3f} seconds")
        shared_models[name] = model_obj

    # ε_ent sweep (Figure 6 panels)
    all_results_by_eps_ent: Dict[float, Dict[str, List[Dict[float, float]]]] = {}

    for eps_ent in cfg.eps_ent:
        eps_ent = float(eps_ent)
        print(f"\n{'='*20} ε_ent = {eps_ent:g} {'='*20}")

        models_to_evaluate: Dict[str, Any] = dict(shared_models)

        for name in EPS_ENT_ALGORITHMS:
            if name not in requested_methods:
                continue
            print(f"\n--- Training {name} (ε_ent={eps_ent:g}) ---")
            train_t0 = time.perf_counter()
            model_obj = build_algorithm(
                name, cfg, input_dim, num_classes, device, eps_ent=eps_ent
            )
            model_obj.fit(
                train_features_np,
                train_labels_np,
                checkpoint_dir=str(checkpoint_dir),
                run_id=0,
            )
            train_timings[name] = time.perf_counter() - train_t0
            print(f"Training time {name}: {train_timings[name]:.3f} seconds")
            models_to_evaluate[name] = model_obj

        # Evaluation (l2 PGD on features)
        results_for_eps: Dict[str, List[Dict[float, float]]] = {
            name: [] for name in models_to_evaluate.keys()
        }
        eval_timings_for_eps: Dict[str, float] = {}

        for name, model_obj in models_to_evaluate.items():
            print(f"\n--- Evaluating {name} (ε_ent={eps_ent:g}) ---")
            eval_t0 = time.perf_counter()
            res = evaluate_model(
                model_obj,
                test_features_np,
                test_labels_np,
                pgd_attack,
                epsilon_attack_values[1:],
                device=device,
                pgd_num_iter=cfg.pgd_steps,
                pgd_restarts=cfg.pgd_restarts,
            )
            eval_timings_for_eps[name] = time.perf_counter() - eval_t0
            print(
                f"Evaluation time {name} (ε_ent={eps_ent:g}): "
                f"{eval_timings_for_eps[name]:.3f} seconds"
            )
            results_for_eps[name].append(res)

        all_results_by_eps_ent[eps_ent] = results_for_eps

        # Save JSON for this ε_ent
        json_path = out_dir / f"results_tau={cfg.tau:g}_epsent={eps_ent:g}.json"
        json_payload = {
            "tau": cfg.tau,
            "lambda": cfg.lambda_param,
            "eps_ent": eps_ent,
            "avg_train_feature_norm": avg_train_feature_norm,
            "avg_feature_norm": avg_feature_norm,
            "perturbation_levels": perturbation_levels,
            "epsilon_attack_values": epsilon_attack_values,
            "evaluation_attack": {
                "name": "adaptive_l2_pgd_feature_delta",
                "pgd_steps": cfg.pgd_steps,
                "pgd_restarts": cfg.pgd_restarts,
                "step_size": "epsilon / max(1, pgd_steps // 2)",
            },
            "hyperparameters": asdict(cfg),
            "training_radii": {
                "RO": {
                    "epsilon": cfg.ro_epsilon,
                    "epsilon_level": cfg.ro_epsilon_level,
                    "scale": "epsilon_level * avg_train_feature_norm"
                    if args.ro_epsilon is None
                    else "absolute --ro_epsilon",
                }
            },
            "architectures": {
                "NPF": {
                    "hidden": list(cfg.npf_hidden),
                    "outer_rank": cfg.npf_outer_rank,
                    "inner_rank": cfg.npf_inner_rank,
                    "activation": cfg.npf_activation,
                    "elu_alpha": cfg.npf_elu_alpha,
                    "softplus_beta": cfg.npf_softplus_beta,
                    "init_eps": cfg.npf_init_eps,
                }
            },
            "method_timings": {
                method: {
                    "train_seconds": float(train_timings.get(method, 0.0)),
                    "eval_seconds": float(eval_timings_for_eps.get(method, 0.0)),
                    "total_seconds": float(train_timings.get(method, 0.0))
                    + float(eval_timings_for_eps.get(method, 0.0)),
                    "definition": (
                        "train_seconds includes algorithm construction and fit(); "
                        "eval_seconds includes clean accuracy and all configured PGD "
                        "attack radii for this eps_ent panel."
                    ),
                }
                for method in results_for_eps.keys()
            },
            "results": {
                method: [_json_safe_results(run_dict) for run_dict in run_list]
                for method, run_list in results_for_eps.items()
            },
        }
        json_path.write_text(json.dumps(json_payload, indent=2))
        print(f"Saved results: {json_path}")

    # Combined Figure-6-style robustness panels
    plot_path = out_dir / f"figure6_tau={cfg.tau:g}_robustness.png"
    plot_figure6_panels(
        all_results_by_eps_ent,
        perturbation_levels=perturbation_levels,
        epsilon_attack_values=epsilon_attack_values,
        tau=cfg.tau,
        out_path=plot_path,
    )


if __name__ == "__main__":
    # CUBLAS workspace config must be set before any CUDA kernel is launched.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
