#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


METHOD_ORDER = ["RGO", "WGF", "SAA", "Dual", "WRM", "RO", "WFR", "SVG", "ICNN"]
MARKER_ORDER = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
METHOD_DISPLAY_NAME: Dict[str, str] = {
    # Align display names with Least_Squares/uncertain_least_squares_icnn.py
    "SAA": "ERM",
    "WRM": "Particle Ascent",
    "RO": "RO",
    "WGF": "WGF(Otto, 1996)",
    "WFR": "WFR(Xu, 2025)",
    "Dual": "Dual(Wang et al., 2021)",
}


@dataclass(frozen=True)
class ResultsFile:
    path: Path
    tau: float
    eps_ent: float
    perturbation_levels: List[float]
    epsilon_attack_values: List[float]
    results: Dict[str, List[Dict[str, float]]]


def _as_float_list(value: Any, *, key: str) -> List[float]:
    if not isinstance(value, list):
        raise TypeError(f"Expected '{key}' to be a list, got {type(value).__name__}.")
    return [float(x) for x in value]


def _as_results_dict(value: Any) -> Dict[str, List[Dict[str, float]]]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected 'results' to be a dict, got {type(value).__name__}.")
    out: Dict[str, List[Dict[str, float]]] = {}
    for method, runs in value.items():
        if not isinstance(method, str):
            raise TypeError(f"Expected method names to be strings, got {type(method).__name__}.")
        if not isinstance(runs, list):
            raise TypeError(f"Expected runs list for '{method}', got {type(runs).__name__}.")
        parsed_runs: List[Dict[str, float]] = []
        for run in runs:
            if not isinstance(run, dict):
                raise TypeError(f"Expected run dict for '{method}', got {type(run).__name__}.")
            parsed_run: Dict[str, float] = {}
            for k, v in run.items():
                if not isinstance(k, str):
                    k = str(k)
                parsed_run[k] = float(v)
            parsed_runs.append(parsed_run)
        out[method] = parsed_runs
    return out


def load_results_file(path: Path) -> ResultsFile:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON root to be a dict in {path}.")

    try:
        tau = float(payload["tau"])
        eps_ent = float(payload["eps_ent"])
    except KeyError as e:
        raise KeyError(f"Missing key {e} in {path}.") from e

    perturbation_levels = _as_float_list(payload.get("perturbation_levels", []), key="perturbation_levels")
    epsilon_attack_values = _as_float_list(payload.get("epsilon_attack_values", []), key="epsilon_attack_values")
    results = _as_results_dict(payload.get("results", {}))

    if not epsilon_attack_values:
        # Fallback for older/hand-edited payloads: reconstruct absolute radii.
        avg_feature_norm = float(payload.get("avg_feature_norm", math.nan))
        if not perturbation_levels or not math.isfinite(avg_feature_norm):
            raise ValueError(
                f"Missing 'epsilon_attack_values' in {path} and cannot reconstruct "
                "(need 'perturbation_levels' + finite 'avg_feature_norm')."
            )
        epsilon_attack_values = [float(avg_feature_norm * d) for d in perturbation_levels]

    if not perturbation_levels:
        avg_feature_norm = float(payload.get("avg_feature_norm", math.nan))
        if not math.isfinite(avg_feature_norm) or not epsilon_attack_values:
            raise ValueError(
                f"Missing 'perturbation_levels' in {path} and cannot reconstruct "
                "(need finite 'avg_feature_norm' + 'epsilon_attack_values')."
            )
        perturbation_levels = [float(e / avg_feature_norm) for e in epsilon_attack_values]

    if len(perturbation_levels) != len(epsilon_attack_values):
        raise ValueError(
            f"Mismatched grids in {path}: len(perturbation_levels)={len(perturbation_levels)} "
            f"but len(epsilon_attack_values)={len(epsilon_attack_values)}."
        )

    return ResultsFile(
        path=path,
        tau=tau,
        eps_ent=eps_ent,
        perturbation_levels=perturbation_levels,
        epsilon_attack_values=epsilon_attack_values,
        results=results,
    )


def _key_for_epsilon(eps: float) -> str:
    # Matches formatting used by adversarial_multiclass_icnn._json_safe_results
    return f"{eps:.10g}"


def _error_curves_from_json_runs(
    runs: Sequence[Mapping[str, float]],
    epsilons: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list[{epsilon(str)->accuracy}] into mean/std error curves."""
    query_keys = [_key_for_epsilon(eps) for eps in epsilons]

    errors_matrix: List[List[float]] = []
    missing: List[str] = []
    for run in runs:
        row: List[float] = []
        for k in query_keys:
            if k not in run:
                missing.append(k)
                row.append(np.nan)
                continue
            acc = float(run[k])
            row.append(100.0 - acc)
        errors_matrix.append(row)

    errors = np.asarray(errors_matrix, dtype=np.float64)
    if np.isnan(errors).any():
        missing_unique = sorted(set(missing))
        raise KeyError(
            "Missing epsilon keys in results runs: "
            + ", ".join(missing_unique[:10])
            + (" ..." if len(missing_unique) > 10 else "")
        )

    mean = np.mean(errors, axis=0)
    std = np.std(errors, axis=0)
    return mean, std


def _expand_inputs(inputs: Sequence[str], *, base_dir: Path | None = None) -> List[Path]:
    out: List[Path] = []
    base_dir = base_dir.resolve() if base_dir is not None else None
    for item in inputs:
        raw = Path(item)
        candidates = [raw]
        if base_dir is not None and not raw.is_absolute():
            candidates.append(base_dir / raw)

        dir_path = next((c for c in candidates if c.is_dir()), None)
        if dir_path is not None:
            out.extend(sorted(dir_path.glob("results_tau=*_epsent=*.json")))
            continue

        if any(ch in item for ch in ["*", "?", "["]):
            matches = glob.glob(item)
            if not matches and base_dir is not None and not raw.is_absolute():
                matches = glob.glob(str(base_dir / item))
            out.extend(sorted(Path(m) for m in matches))
            continue

        file_path = next((c for c in candidates if c.exists()), raw)
        out.append(file_path)

    deduped: List[Path] = []
    seen: set[Path] = set()
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot robustness curves (test error vs perturbation Δ) from results_tau=*_epsent=*.json files."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        dest="inputs_flag",
        default=None,
        help="JSON file(s), glob(s), or directories (same semantics as positional inputs).",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="JSON file(s), glob(s), or directories. Defaults to *_old.json if present, else all results_tau=*_epsent=*.json.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="robustness_vs_delta.pdf",
        help="Output image path (default: robustness_vs_delta.pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Headless-safe default.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    inputs: List[str] = []
    if args.inputs:
        inputs.extend(args.inputs)
    if args.inputs_flag:
        inputs.extend(args.inputs_flag)
    if not inputs:
        default_dir = Path(__file__).resolve().parent
        old = sorted(default_dir.glob("results_tau=*_epsent=*_old.json"))
        if old:
            inputs = [str(p) for p in old]
        else:
            inputs = [str(p) for p in sorted(default_dir.glob("results_tau=*_epsent=*.json"))]

    default_dir = Path(__file__).resolve().parent
    json_paths = _expand_inputs(inputs, base_dir=default_dir)
    if not json_paths:
        raise FileNotFoundError("No input JSON files found.")

    loaded = [load_results_file(p) for p in json_paths]

    # Group by eps_ent (useful when you pass multiple result files).
    groups: MutableMapping[float, List[ResultsFile]] = {}
    for rf in loaded:
        groups.setdefault(rf.eps_ent, []).append(rf)

    eps_ent_list = sorted(groups.keys(), reverse=True)
    n_panels = len(eps_ent_list)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 3.6), dpi=int(args.dpi), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # Stable colors across panels.
    cmap = plt.get_cmap("tab10")
    method_to_color = {m: cmap(i % 10) for i, m in enumerate(METHOD_ORDER)}
    method_to_handle: Dict[str, Any] = {}

    from matplotlib.ticker import MultipleLocator

    for ax, eps_ent in zip(axes, eps_ent_list):
        files = groups[eps_ent]
        # Merge runs across files (same eps_ent).
        epsilon_attack_values = files[0].epsilon_attack_values
        perturbation_levels = files[0].perturbation_levels
        tau = files[0].tau

        for rf in files[1:]:
            if len(rf.epsilon_attack_values) != len(epsilon_attack_values) or any(
                abs(a - b) > 1e-8 for a, b in zip(rf.epsilon_attack_values, epsilon_attack_values)
            ):
                raise ValueError(
                    "Cannot merge files with different epsilon_attack_values grids:\n"
                    + f"- {files[0].path}\n"
                    + f"- {rf.path}"
                )
            if len(rf.perturbation_levels) != len(perturbation_levels) or any(
                abs(a - b) > 1e-12 for a, b in zip(rf.perturbation_levels, perturbation_levels)
            ):
                raise ValueError(
                    "Cannot merge files with different perturbation_levels grids:\n"
                    + f"- {files[0].path}\n"
                    + f"- {rf.path}"
                )

        merged_results: Dict[str, List[Dict[str, float]]] = {}
        for rf in files:
            for method, runs in rf.results.items():
                merged_results.setdefault(method, []).extend(runs)

        x_vals = perturbation_levels
        panel_min_y = math.inf
        panel_max_y = -math.inf

        for method in METHOD_ORDER:
            runs = merged_results.get(method, [])
            if not runs:
                continue

            mean_errors, std_errors = _error_curves_from_json_runs(runs, epsilon_attack_values)
            panel_min_y = min(panel_min_y, float(np.min(mean_errors)))
            panel_max_y = max(panel_max_y, float(np.max(mean_errors)))

            marker = MARKER_ORDER[METHOD_ORDER.index(method) % len(MARKER_ORDER)]
            line = ax.plot(
                x_vals,
                mean_errors,
                label=METHOD_DISPLAY_NAME.get(method, method),
                color=method_to_color.get(method),
                linewidth=2.0,
                marker=marker,
                markersize=4.0,
                markeredgewidth=0.0,
            )[0]
            if method not in method_to_handle:
                method_to_handle[method] = line
            if len(runs) > 1:
                ax.fill_between(
                    x_vals,
                    mean_errors - std_errors,
                    mean_errors + std_errors,
                    color=line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_xlabel(r"perturbation $\Delta$")
        ax.set_ylabel("test error (%)")
        if tau == 0:
            raise ValueError(f"Invalid tau=0 in results; cannot compute lambda = 1/(2*tau) for eps_ent={eps_ent:g}.")
        lambda_from_tau = 1.0 / (2.0 * tau)
        ax.set_title(fr"$\lambda$={lambda_from_tau:g}, $\epsilon$={eps_ent:g}")

        if math.isfinite(panel_min_y) and math.isfinite(panel_max_y):
            pad = max(1.5, 0.04 * (panel_max_y - panel_min_y))
            ax.set_ylim(bottom=max(0.0, panel_min_y - pad), top=min(100.0, panel_max_y + pad))

        if x_vals:
            ax.set_xlim(min(x_vals), max(x_vals))
            max_x = max(x_vals)
            if max_x <= 0.12:
                ax.xaxis.set_major_locator(MultipleLocator(0.01))
                ax.xaxis.set_minor_locator(MultipleLocator(0.005))
            elif max_x <= 0.25:
                ax.xaxis.set_major_locator(MultipleLocator(0.02))
                ax.xaxis.set_minor_locator(MultipleLocator(0.01))
            elif max_x <= 0.6:
                ax.xaxis.set_major_locator(MultipleLocator(0.05))
                ax.xaxis.set_minor_locator(MultipleLocator(0.025))
            else:
                ax.xaxis.set_major_locator(MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))

        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.30)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
        ax.tick_params(axis="both", which="both", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [method_to_handle[m] for m in METHOD_ORDER if m in method_to_handle]
    labels = [METHOD_DISPLAY_NAME.get(m, m) for m in METHOD_ORDER if m in method_to_handle]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
