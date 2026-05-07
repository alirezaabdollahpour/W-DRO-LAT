"""Least_Squares entry point.

Runs any subset of {ERM, Particle Ascent, WGF, WFR, Dual, ICNN, Madry PGD,
PPA, WDRO-NPF, WDRO-NPF-LastQuad} on the uncertain least squares Fig. 8
benchmark, saves per-method JSON results, and optionally produces the
Fig. 8 PDFs.
"""
from __future__ import annotations

import os
from typing import Dict

import torch

from algorithms.registry import EXTENDED_RETURN_ALGORITHMS, build_registry
from config import ALL_ALGORITHMS, build_arg_parser, config_from_args
from utils.common import seed_everything
from utils.data import generate_training_data, make_problem
from utils.eval import evaluate_delta_sweep
from utils.io import save_results_json
from utils.plot import plot_results


def main() -> None:
    torch.set_default_dtype(torch.float64)

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

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

    seed_everything(cfg.seed)
    A0, A1, b = make_problem(cfg, device)
    xi_train = generate_training_data(cfg, device)

    trained: Dict[str, torch.Tensor] = {}
    display_names: Dict[str, str] = {}

    print("[1/2] Training models...")
    for algo_key in selected:
        display_name, solve_fn = registry[algo_key]
        print("\n" + "=" * 72)
        print(f"Training {display_name} ...")
        print("=" * 72)

        if algo_key in EXTENDED_RETURN_ALGORITHMS:
            out = solve_fn(xi_train, cfg, A0, A1, b)
            theta = out[0]
        else:
            theta = solve_fn(xi_train, cfg, A0, A1, b)

        trained[display_name] = theta
        display_names[algo_key] = display_name

    if args.skip_eval:
        return

    print("\n[2/2] Evaluating delta sweep...")
    delta_values, results = evaluate_delta_sweep(trained, cfg, A0, A1, b, device)

    save_dir = args.results_dir or os.path.dirname(os.path.abspath(__file__))
    save_results_json(delta_values, results, save_dir=save_dir, cfg=cfg)

    if args.plots:
        fig_dir = save_dir
        paper_plot = os.path.join(fig_dir, "fig8_uncertain_least_squares.pdf")
        plot_results(
            delta_values,
            {k: v for k, v in results.items()
             if k in {"ERM", "WGF(Otto, 1996)", "WFR(Xu, 2025)",
                      "Particle Ascent", "Dual(Wang et al., 2021)"}},
            save_path=paper_plot,
            plot_order=[
                "ERM", "WGF(Otto, 1996)", "WFR(Xu, 2025)",
                "Particle Ascent", "Dual(Wang et al., 2021)",
            ],
        )
        print(f"Saved: {paper_plot}")

        all_plot = os.path.join(fig_dir, "fig8_uncertain_least_squares_all_methods.pdf")
        plot_results(
            delta_values, results, save_path=all_plot,
            plot_order=[
                "ERM", "WGF(Otto, 1996)", "WFR(Xu, 2025)",
                "Particle Ascent", "Dual(Wang et al., 2021)",
                "ICNN", "Madry PGD", "PPA", "WDRO-NPF",
                "WDRO-NPF-LastQuad", "NN-DRO",
            ],
        )
        print(f"Saved: {all_plot}")


if __name__ == "__main__":
    main()
