"""Fig. 8 plotting for the ULS delta sweep."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt


STYLES = {
    "ERM": {"color": "k", "linestyle": "--"},
    "WGF(Otto, 1996)": {"color": "#9a0ba7", "linestyle": "-."},
    "WFR(Xu, 2025)": {"color": "#db2020", "linestyle": ":"},
    "Particle Ascent": {"color": "#0eaf0e", "linestyle": (0, (3, 1, 1, 1))},
    "Dual(Wang et al., 2021)": {"color": "#7A4E15", "linestyle": (0, (5, 5))},
    "ICNN": {"color": "#1f77b4", "linestyle": "-"},
    "Madry PGD": {"color": "#e67300", "linestyle": (0, (1, 1))},
    "PPA": {"color": "#00ced1", "linestyle": (0, (3, 1, 1, 1, 1, 1))},
    "WDRO-NPF": {"color": "#1f77b4", "linestyle": "-"},
    "WDRO-NPF-LastQuad": {"color": "#0072B2", "linestyle": (0, (4, 1))},
    "NN-DRO": {"color": "#8a2be2", "linestyle": (0, (5, 2, 1, 2))},
}


def plot_results(
    delta_values: np.ndarray,
    results: Dict[str, List[float]],
    save_path: str,
    plot_order: Sequence[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.613), dpi=300)
    order = list(plot_order) if plot_order is not None else list(results.keys())

    for name in order:
        if name not in results:
            continue
        style = STYLES.get(name, {"linestyle": "-"})
        ax.plot(delta_values, results[name], linewidth=1.5, label=name, **style)

    ax.set_xlabel(r"perturbation $\Delta$", fontsize=9)
    ax.set_ylabel("test loss", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.set_xlim(float(delta_values.min()), float(delta_values.max()))
    ax.set_ylim(0.0, 3.0)
    ax.grid(False)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
