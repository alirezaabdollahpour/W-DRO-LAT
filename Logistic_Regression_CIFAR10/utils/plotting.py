"""Figure-6-style training loss + robustness plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(
    all_runs_losses: Dict[str, List[List[float]]], lam: float, eps: float
) -> None:
    try:
        plt.style.use("jz.mplstyle")
    except (OSError, ValueError):
        pass
    plt.figure()
    for model_name, loss_runs in all_runs_losses.items():
        if not loss_runs:
            continue
        loss_array = np.array(loss_runs)
        epochs = np.arange(1, loss_array.shape[1] + 1)
        mean_loss = np.mean(loss_array, axis=0)
        std_loss = np.std(loss_array, axis=0)
        line, = plt.plot(epochs, mean_loss, label=model_name)
        plt.fill_between(
            epochs, mean_loss - std_loss, mean_loss + std_loss,
            color=line.get_color(), alpha=0.2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.legend()
    plt.grid(True, which="both", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"training_loss_lam={lam}_eps={eps}.pdf")
    print(f"\nTraining loss plot saved as 'training_loss_lam={lam}_eps={eps}.pdf'")


def _ordered_error_curves(
    results_runs: List[Dict[float, float]],
    query_keys: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    errors_matrix: List[List[float]] = []
    for single_run_result in results_runs:
        ordered_errors = [100.0 - float(single_run_result.get(k, 0.0)) for k in query_keys]
        errors_matrix.append(ordered_errors)
    errors_array = np.asarray(errors_matrix, dtype=np.float64)
    mean_errors = np.mean(errors_array, axis=0)
    std_errors = np.std(errors_array, axis=0)
    return mean_errors, std_errors


def plot_figure6_panels(
    all_results_by_eps_ent: Dict[float, Dict[str, List[Dict[float, float]]]],
    perturbation_levels: List[float],
    epsilon_attack_values: List[float],
    tau: float,
    out_path: Path,
) -> None:
    """Create a multi-panel robustness plot (test error vs normalised Δ)."""
    eps_ent_list = list(all_results_by_eps_ent.keys())
    eps_ent_list.sort(reverse=True)

    query_keys = [0.0] + list(epsilon_attack_values[1:])

    fig, axes = plt.subplots(
        1, len(eps_ent_list), figsize=(12.5, 3.6), dpi=200, constrained_layout=True
    )
    if len(eps_ent_list) == 1:
        axes = [axes]

    methods_order = [
        "RGO",
        "WGF",
        "SAA",
        "Dual",
        "WRM",
        "RO",
        "WFR",
        "SVG",
        "ICNN",
        "NPF",
        "NPF-LastQuad",
        "NN-DRO",
        "PPA",
    ]

    for ax, eps_ent in zip(axes, eps_ent_list):
        results_for_eps = all_results_by_eps_ent[eps_ent]
        for method in methods_order:
            if method not in results_for_eps:
                continue
            runs = results_for_eps[method]
            if not runs:
                continue
            mean_errors, std_errors = _ordered_error_curves(runs, query_keys)
            line = ax.plot(perturbation_levels, mean_errors, label=method)[0]
            if len(runs) > 1:
                ax.fill_between(
                    perturbation_levels,
                    mean_errors - std_errors,
                    mean_errors + std_errors,
                    color=line.get_color(),
                    alpha=0.20,
                )
        ax.set_xlabel("perturbation $\\Delta$")
        ax.set_ylabel("test error (%)")
        ax.set_title(f"$\\tau$={tau:g}, $\\epsilon$={eps_ent:g}")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods_order), bbox_to_anchor=(0.5, 1.08))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved robustness plot: {out_path}")
