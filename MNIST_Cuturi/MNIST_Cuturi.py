"""MNIST_Cuturi entry point.

Trains any subset of {SAA, Algo1/WRM, Algo2/ICNN, NPF, PPA, New_PPA,
Dual, WGF, WFR, SVG, RGO} on MNIST and runs a standardised clean / PGD-L2
evaluation. All hyperparameters are exposed as CLI flags; see --help.
"""
from __future__ import annotations

import json
import os
# CUBLAS workspace config must be set before any CUDA kernel is launched.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from algorithms.base import TrainState
from algorithms.registry import DOUBLE_STATE_ALGORITHMS, build_registry
from config import ALL_ALGORITHMS, build_arg_parser, config_from_args, pgd_cfg_from_args
from utils.common import seed_everything, set_deterministic_backends
from utils.data import load_mnist
from utils.eval import evaluate_clean, evaluate_pgd, evaluate_pgd_l2_sweep


def main() -> None:
    set_deterministic_backends()

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    pgd_eval_cfg = pgd_cfg_from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = build_registry()

    requested = [name.lower() for name in args.algorithms]
    if "all" in requested:
        selected = list(ALL_ALGORITHMS)
    else:
        invalid = [name for name in requested if name not in registry]
        if invalid:
            raise ValueError(f"Unknown algorithms requested: {invalid}")
        selected = requested

    trained_states: Dict[str, TrainState] = {}
    training_artifacts: Dict[str, Dict[str, Any]] = {}

    for algo_key in selected:
        display_name, train_fn, save_fn = registry[algo_key]
        print("\n" + "=" * 72)
        print(f"Training {display_name} ...")
        print("=" * 72)

        if algo_key in DOUBLE_STATE_ALGORITHMS:
            state, icnn_state, logs = train_fn(cfg, device)
            trained_states[algo_key] = state
            training_artifacts[algo_key] = {"logs": logs, "icnn_state": icnn_state}
            if save_fn is not None:
                save_fn(state, icnn_state, cfg)
        else:
            state, logs = train_fn(cfg, device)
            trained_states[algo_key] = state
            training_artifacts[algo_key] = {"logs": logs}
            if save_fn is not None:
                save_fn(state, cfg)

    if args.skip_eval:
        return

    _, test_ds = load_mnist()
    seed_everything(cfg.seed)

    fixed_pgd_kwargs = dict(
        batch_size=cfg.batch_size,
        eps=args.eval_fixed_eps,
        num_steps=pgd_eval_cfg.num_steps,
        restarts=pgd_eval_cfg.restarts,
        device=device,
    )

    summary: Dict[str, Any] = {
        "hyperparameters": asdict(cfg),
        "pgd_fixed_config": {
            **{k: v for k, v in fixed_pgd_kwargs.items() if k != "device"},
        },
        "pgd_l2_sweep_config": asdict(pgd_eval_cfg),
        "algorithms": {},
    }

    for algo_key in selected:
        state = trained_states[algo_key]
        display_name = registry[algo_key][0]
        print("\n" + "-" * 72)
        print(f"Standardized evaluation: {display_name}")
        print("-" * 72)
        clean_metrics = evaluate_clean(state, test_ds, cfg.batch_size, device)
        fixed_pgd = evaluate_pgd(state, test_ds, **fixed_pgd_kwargs)
        print(
            f"[{display_name}] fixed-eps PGD acc={fixed_pgd['acc'] * 100:.2f}% "
            f"L2={fixed_pgd['avg_l2']:.4f} Linf={fixed_pgd['avg_linf']:.4f}"
        )
        sweep = evaluate_pgd_l2_sweep(
            state, test_ds, pgd_eval_cfg, cfg.batch_size, device
        )
        summary["algorithms"][algo_key] = {
            "display_name": display_name,
            "clean_test": clean_metrics,
            "fixed_pgd": fixed_pgd,
            "pgd_l2_sweep": sweep,
        }

        result_path = Path("MNIST") / f"{algo_key}_results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                payload = json.load(f)
        else:
            payload = {
                "algorithm": algo_key,
                "display_name": display_name,
                "hyperparameters": asdict(cfg),
            }
        payload["clean_test"] = clean_metrics
        payload["fixed_pgd"] = fixed_pgd
        payload["pgd_l2_sweep"] = sweep
        payload["pgd_eval_config"] = asdict(pgd_eval_cfg)
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)

    os.makedirs("MNIST", exist_ok=True)
    with open(os.path.join("MNIST", "all_algorithm_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
