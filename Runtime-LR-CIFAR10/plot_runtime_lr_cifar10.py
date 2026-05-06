#!/usr/bin/env python3
"""Plot runtime/robustness tradeoffs for Runtime-LR-CIFAR10 experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


METHOD_ORDER = ["WRM", "RO", "Dual", "WFR", "NPF"]
EXCLUDED_METHODS = {"NN-DRO"}
FAMILY_LABELS = {
    "wrm_style": "Implicit Maps",
    "icnn_style": "ICNN-DRO",
}
METHOD_LABELS = {
    "WRM": "PA",
    "Dual": "SDRO",
    "WFR": "WFR",
    "NPF": "ICNN-DRO",
}
METHOD_MARKERS = {
    "WRM": "o",
    "RO": "s",
    "Dual": "D",
    "WFR": "^",
    "NPF": "P",
}
METHOD_STYLES = {
    "WRM": {"color": "#009E73", "linestyle": (0, (5, 2))},
    "RO": {"color": "#CC79A7", "linestyle": (0, (2, 2))},
    "Dual": {"color": "#E69F00", "linestyle": (0, (3, 3, 1, 3))},
    "WFR": {"color": "#D55E00", "linestyle": (0, (1, 1))},
    "NPF": {"color": "#CA0020", "linestyle": "-"},
}
SERIES_STYLES = {
    "Implicit Maps": {"color": "#009E73", "linestyle": (0, (5, 2)), "marker": "o"},
    "ICNN-DRO": {"color": "#CA0020", "linestyle": "-", "marker": "o"},
}


def _configure_matplotlib() -> None:
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.titlesize": 16,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })


_configure_matplotlib()


@dataclass(frozen=True)
class RunRecord:
    family: str
    k: int
    seed: int
    started_utc: str
    elapsed_seconds: float
    out_dir: Path
    log_file: Path
    time_file: Path
    json_path: Path


@dataclass(frozen=True)
class MethodRecord:
    family: str
    method: str
    k: int
    seed: int
    elapsed_seconds: float
    delta_to_accuracy: Dict[float, float]


def _parse_max_k(value: str) -> Optional[int]:
    if value.lower() in {"none", "all", "inf"}:
        return None
    return int(value)


def _read_manifest(path: Path, max_k: Optional[int]) -> List[RunRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    rows: List[RunRecord] = []
    with path.open(newline="") as f:
        non_comment_lines = (line for line in f if not line.startswith("#"))
        reader = csv.DictReader(non_comment_lines, delimiter="\t")
        for row in reader:
            if row.get("status") != "completed":
                continue
            k = int(row["k"])
            if max_k is not None and k > max_k:
                continue
            out_dir = Path(row["out_dir"])
            jsons = sorted(out_dir.glob("results_tau=*_epsent=*.json"))
            if not jsons:
                continue
            rows.append(
                RunRecord(
                    family=row["family"],
                    k=k,
                    seed=int(row["seed"]),
                    started_utc=row.get("started_utc", ""),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    out_dir=out_dir,
                    log_file=Path(row.get("log_file", "")),
                    time_file=Path(row.get("time_file", "")),
                    json_path=jsons[0],
                )
            )
    rows.sort(key=lambda r: (r.family, r.k, r.seed))
    return rows


def _nearest_index(values: Sequence[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))


def _accuracy_for_epsilon(run_result: Mapping[str, float], epsilon: float) -> float:
    best_key = min(run_result.keys(), key=lambda k: abs(float(k) - float(epsilon)))
    return float(run_result[best_key])


def _iso_utc_to_timestamp(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _checkpoint_runtime_seconds(run: RunRecord, prefix: str) -> Optional[float]:
    start_ts = _iso_utc_to_timestamp(run.started_utc)
    if start_ts is None:
        return None
    ckpts = list((run.out_dir / "checkpoints").glob(f"{prefix}_run*_epoch_*.pth"))
    if not ckpts:
        return None
    last_ckpt_ts = max(p.stat().st_mtime for p in ckpts)
    elapsed = last_ckpt_ts - start_ts
    if elapsed <= 0.0 or not math.isfinite(elapsed):
        return None
    return elapsed


def _method_elapsed_seconds(run: RunRecord, method: str, payload: Mapping[str, object]) -> float:
    method_timings = payload.get("method_timings", {})
    if isinstance(method_timings, Mapping):
        method_timing = method_timings.get(method, {})
        if isinstance(method_timing, Mapping):
            total_seconds = method_timing.get("total_seconds")
            if total_seconds is not None:
                return float(total_seconds)
    if method == "NPF":
        # The icnn_style run also trained NN-DRO after NPF.  The final NPF
        # checkpoint is written before NN-DRO starts, so start->last-NPF-ckpt
        # is the cleanest runtime estimate available from existing artifacts.
        estimated = _checkpoint_runtime_seconds(run, "NPF")
        if estimated is not None:
            return estimated
    return run.elapsed_seconds


def _load_method_records(runs: Sequence[RunRecord]) -> Tuple[List[MethodRecord], List[float]]:
    records: List[MethodRecord] = []
    perturbation_levels: List[float] = []

    for run in runs:
        payload = json.loads(run.json_path.read_text())
        levels = [float(v) for v in payload["perturbation_levels"]]
        epsilons = [float(v) for v in payload["epsilon_attack_values"]]
        if not perturbation_levels:
            perturbation_levels = levels

        for method, method_runs in payload["results"].items():
            if method in EXCLUDED_METHODS:
                continue
            if not method_runs:
                continue
            result = method_runs[0]
            delta_to_accuracy = {}
            for delta, epsilon in zip(levels, epsilons):
                delta_to_accuracy[float(delta)] = _accuracy_for_epsilon(result, epsilon)
            records.append(
                MethodRecord(
                    family=run.family,
                    method=method,
                    k=run.k,
                    seed=run.seed,
                    elapsed_seconds=_method_elapsed_seconds(run, method, payload),
                    delta_to_accuracy=delta_to_accuracy,
                )
            )

    records.sort(key=lambda r: (METHOD_ORDER.index(r.method) if r.method in METHOD_ORDER else 999, r.k, r.seed))
    return records, perturbation_levels


def _method_sort_key(method: str) -> Tuple[int, str]:
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else 999, method)


def _available_methods(records: Iterable[MethodRecord]) -> List[str]:
    return sorted({r.method for r in records}, key=_method_sort_key)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_style(method: str) -> Dict[str, object]:
    return dict(METHOD_STYLES.get(method, {"color": None, "linestyle": "-"}))


def _series_style(label: str) -> Dict[str, object]:
    return dict(SERIES_STYLES.get(label, {"color": None, "linestyle": "-", "marker": "o"}))


def _percent_text(value: float) -> str:
    return f"{value:.1f}" + (r"\%" if plt.rcParams.get("text.usetex") else "%")


def _percent_axis_label(label: str) -> str:
    return rf"{label} (\%)" if plt.rcParams.get("text.usetex") else f"{label} (%)"


def _savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle=":", linewidth=0.2, color="gray", alpha=0.2)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="gray", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _set_k_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def _set_delta_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5.0))


def _paper_legend(ax: plt.Axes, **kwargs):
    defaults = dict(
        frameon=False,
        framealpha=0.95,
        edgecolor="0.75",
        fontsize=14,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.1,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _paper_figure_legend(fig: plt.Figure, handles, labels, **kwargs):
    defaults = dict(
        frameon=False,
        framealpha=0.95,
        edgecolor="0.75",
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.1,
    )
    defaults.update(kwargs)
    return fig.legend(handles, labels, **defaults)


@dataclass(frozen=True)
class RuntimePoint:
    label: str
    k: int
    elapsed_seconds: float
    robust_acc_delta: Optional[float]


def _runtime_points_for_scaling(
    runs: Sequence[RunRecord],
    records: Sequence[MethodRecord],
    target_delta: float,
    available_deltas: Sequence[float],
) -> Dict[str, List[RuntimePoint]]:
    delta = available_deltas[_nearest_index(available_deltas, target_delta)]
    points: Dict[str, List[RuntimePoint]] = {}

    for run in runs:
        if run.family.startswith("icnn_style"):
            npf_rows = [r for r in records if r.family == run.family and r.k == run.k and r.seed == run.seed and r.method == "NPF"]
            if not npf_rows:
                continue
            r = npf_rows[0]
            points.setdefault("ICNN-DRO", []).append(
                RuntimePoint("ICNN-DRO", run.k, r.elapsed_seconds, r.delta_to_accuracy[delta])
            )
        elif run.family.startswith("wrm_style"):
            rows = [r for r in records if r.family == run.family and r.k == run.k and r.seed == run.seed]
            if not rows:
                continue
            best_acc = max(r.delta_to_accuracy[delta] for r in rows)
            points.setdefault("Implicit Maps", []).append(
                RuntimePoint("Implicit Maps", run.k, run.elapsed_seconds, best_acc)
            )

    for label in points:
        points[label].sort(key=lambda p: p.k)
    return points


def _plot_family_runtime(
    runs: Sequence[RunRecord],
    records: Sequence[MethodRecord],
    available_deltas: Sequence[float],
    out_dir: Path,
) -> None:
    points = _runtime_points_for_scaling(runs, records, 0.08, available_deltas)
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300, layout="constrained")
    for label, series in points.items():
        style = _series_style(label)
        ax.plot(
            [p.k for p in series],
            [p.elapsed_seconds / 60.0 for p in series],
            marker=style.pop("marker", "o"),
            linewidth=2.0,
            label=label,
            **style,
        )
    ax.set_xlabel(r"Inner iterations $K$")
    ax.set_ylabel(r"Wall time (minutes)")
    ax.set_title("Runtime Scaling")
    _set_k_axis(ax)
    _paper_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=2)
    _style_axes(ax)
    _savefig(fig, out_dir, "01_family_runtime_vs_k")


def _annotate_runtime_points(
    ax: plt.Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    accs: Sequence[Optional[float]],
    *,
    series_label: str,
    normalized: bool = False,
) -> None:
    for x, y, acc in zip(xs, ys, accs):
        if acc is None:
            continue
        if normalized and x == min(xs) and series_label == "Implicit Maps":
            offset = (5, -16)
        elif normalized and x == min(xs):
            offset = (5, 8)
        elif series_label == "Implicit Maps":
            offset = (5, 7)
        else:
            offset = (5, 7)
        ax.annotate(
            _percent_text(acc),
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            color="#303030",
        )


def _plot_family_runtime_scaling(
    runs: Sequence[RunRecord],
    records: Sequence[MethodRecord],
    available_deltas: Sequence[float],
    out_dir: Path,
) -> None:
    points = _runtime_points_for_scaling(runs, records, 0.08, available_deltas)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=300, layout="constrained")
    all_ks: List[float] = []
    for label, series in points.items():
        if not series:
            continue
        style0 = _series_style(label)
        style1 = _series_style(label)
        ks = np.asarray([p.k for p in series], dtype=float)
        mins = np.asarray([p.elapsed_seconds / 60.0 for p in series], dtype=float)
        accs = [p.robust_acc_delta for p in series]
        all_ks.extend(ks.tolist())
        axes[0].plot(
            ks,
            mins / ks,
            marker=style0.pop("marker", "o"),
            linewidth=2.0,
            label=label,
            **style0,
        )
        axes[1].plot(
            ks,
            mins / mins[0],
            marker=style1.pop("marker", "o"),
            linewidth=2.0,
            label=label,
            **style1,
        )
        _annotate_runtime_points(axes[0], ks, mins / ks, accs, series_label=label)
        _annotate_runtime_points(
            axes[1], ks, mins / mins[0], accs, series_label=label, normalized=True
        )

    axes[0].set_xlabel(r"Inner iterations $K$")
    axes[0].set_ylabel(r"Runtime minutes / $K$")
    axes[0].set_title("Amortized Runtime Per Inner Iteration")
    axes[1].set_xlabel(r"Inner iterations $K$")
    axes[1].set_ylabel(r"Runtime / runtime at smallest $K$")
    axes[1].set_title("Normalized Runtime Growth")
    for ax in axes:
        _set_k_axis(ax)
        if all_ks:
            ax.set_xticks(sorted(set(all_ks)))
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        framealpha=0.95,
        edgecolor="0.75",
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncols=2,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.1,
    )
    fig.text(
        0.5,
        -0.02,
        r"ICNN-DRO runtime excludes NN-DRO; point labels show robust accuracy at $\Delta=0.08$",
        ha="center",
        va="top",
        fontsize=12,
    )
    _savefig(fig, out_dir, "07_family_runtime_scaling")


def _plot_accuracy_vs_k(
    records: Sequence[MethodRecord],
    requested_deltas: Sequence[float],
    available_deltas: Sequence[float],
    out_dir: Path,
) -> None:
    deltas = [available_deltas[_nearest_index(available_deltas, d)] for d in requested_deltas]
    fig, axes = plt.subplots(
        1, len(deltas), figsize=(5.4 * len(deltas), 4.8), dpi=300
    )
    fig.subplots_adjust(top=0.76, wspace=0.22)
    if len(deltas) == 1:
        axes = [axes]

    methods = _available_methods(records)
    for ax, delta in zip(axes, deltas):
        for method in methods:
            rows = sorted([r for r in records if r.method == method], key=lambda r: r.k)
            if not rows:
                continue
            ax.plot(
                [r.k for r in rows],
                [r.delta_to_accuracy[delta] for r in rows],
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=2.0,
                label=_method_label(method),
                **_method_style(method),
            )
        ax.set_xlabel(r"Inner iterations $K$")
        ax.set_ylabel(_percent_axis_label("Accuracy"))
        ax.set_title(rf"Accuracy vs $K$ at $\Delta={delta:g}$", pad=10)
        _set_k_axis(ax)
        _style_axes(ax)
    handles, labels = axes[-1].get_legend_handles_labels()
    _paper_figure_legend(
        fig,
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.98),
        ncols=min(5, len(labels)),
    )
    _savefig(fig, out_dir, "02_accuracy_vs_k_selected_deltas")


def _plot_tradeoff(
    records: Sequence[MethodRecord],
    requested_deltas: Sequence[float],
    available_deltas: Sequence[float],
    out_dir: Path,
) -> None:
    deltas = [available_deltas[_nearest_index(available_deltas, d)] for d in requested_deltas]
    fig, axes = plt.subplots(
        1, len(deltas), figsize=(5.8 * len(deltas), 5.0), dpi=300
    )
    fig.subplots_adjust(top=0.76, wspace=0.24)
    if len(deltas) == 1:
        axes = [axes]

    methods = _available_methods(records)
    for ax, delta in zip(axes, deltas):
        for method in methods:
            rows = sorted([r for r in records if r.method == method], key=lambda r: r.k)
            if not rows:
                continue
            xs = [r.elapsed_seconds / 60.0 for r in rows]
            ys = [r.delta_to_accuracy[delta] for r in rows]
            ax.plot(
                xs,
                ys,
                marker=METHOD_MARKERS.get(method, "o"),
                linewidth=1.8,
                label=_method_label(method),
                **_method_style(method),
            )
            for r, x, y in zip(rows, xs, ys):
                ax.annotate(rf"$K={r.k}$", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel(r"Wall time (minutes)")
        ax.set_ylabel(_percent_axis_label("Accuracy"))
        ax.set_title(rf"Accuracy vs Runtime at $\Delta={delta:g}$", pad=10)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        _style_axes(ax)
    handles, labels = axes[-1].get_legend_handles_labels()
    _paper_figure_legend(
        fig,
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.98),
        ncols=min(5, len(labels)),
    )
    _savefig(fig, out_dir, "03_accuracy_vs_runtime_selected_deltas")


def _plot_error_curves_at_best_k(
    records: Sequence[MethodRecord],
    available_deltas: Sequence[float],
    out_dir: Path,
    best_k: Optional[int],
) -> None:
    if best_k is None:
        best_k = max(r.k for r in records)
    rows = [r for r in records if r.k == best_k]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.0, 8.0), dpi=300)
    fig.subplots_adjust(top=0.82)
    for method in _available_methods(rows):
        method_rows = [r for r in rows if r.method == method]
        if not method_rows:
            continue
        r = method_rows[0]
        errors = [100.0 - r.delta_to_accuracy[d] for d in available_deltas]
        ax.plot(
            available_deltas,
            errors,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            label=_method_label(method),
            **_method_style(method),
        )
    ax.set_xlabel(r"Perturbation $\Delta$")
    ax.set_ylabel(_percent_axis_label("Test error"))
    ax.set_title(rf"Robustness Curves at $K={best_k}$", pad=10)
    _set_delta_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _paper_figure_legend(
        fig,
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.98),
        ncols=min(5, len(labels)),
    )
    _style_axes(ax)
    _savefig(fig, out_dir, f"04_robustness_curves_k{best_k}")


def _plot_summary_bars(
    records: Sequence[MethodRecord],
    requested_deltas: Sequence[float],
    available_deltas: Sequence[float],
    out_dir: Path,
    best_k: Optional[int],
) -> None:
    if best_k is None:
        best_k = max(r.k for r in records)
    rows = [r for r in records if r.k == best_k]
    if not rows:
        return

    deltas = [available_deltas[_nearest_index(available_deltas, d)] for d in requested_deltas]
    methods = _available_methods(rows)
    x = np.arange(len(methods))
    width = min(0.8 / max(len(deltas), 1), 0.25)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    fig.subplots_adjust(top=0.74)
    for j, delta in enumerate(deltas):
        vals = []
        for method in methods:
            r = next(rr for rr in rows if rr.method == method)
            vals.append(r.delta_to_accuracy[delta])
        ax.bar(
            x + (j - (len(deltas) - 1) / 2) * width,
            vals,
            width=width,
            label=rf"$\Delta={delta:g}$",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=0)
    ax.set_ylabel(_percent_axis_label("Accuracy"))
    ax.set_title(rf"Accuracy Summary at $K={best_k}$", pad=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    _paper_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=len(deltas))
    _style_axes(ax)
    _savefig(fig, out_dir, f"05_accuracy_bars_k{best_k}")


def _plot_runtime_efficiency(
    records: Sequence[MethodRecord],
    target_delta: float,
    available_deltas: Sequence[float],
    out_dir: Path,
) -> None:
    delta = available_deltas[_nearest_index(available_deltas, target_delta)]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    fig.subplots_adjust(top=0.76)
    for method in _available_methods(records):
        rows = sorted([r for r in records if r.method == method], key=lambda r: r.k)
        if not rows:
            continue
        eff = [r.delta_to_accuracy[delta] / max(r.elapsed_seconds / 60.0, 1e-12) for r in rows]
        ax.plot(
            [r.k for r in rows],
            eff,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            label=_method_label(method),
            **_method_style(method),
        )
    ax.set_xlabel(r"Inner iterations $K$")
    ax.set_ylabel("Accuracy points per runtime minute")
    ax.set_title(rf"Runtime Efficiency at $\Delta={delta:g}$", pad=10)
    _set_k_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _paper_figure_legend(
        fig,
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.98),
        ncols=min(5, len(labels)),
    )
    _style_axes(ax)
    _savefig(fig, out_dir, f"06_efficiency_vs_k_delta_{str(delta).replace('.', 'p')}")


def _write_summary_csv(records: Sequence[MethodRecord], available_deltas: Sequence[float], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_summary_long.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["family", "method", "k", "seed", "elapsed_seconds", "elapsed_minutes"]
            + [f"acc_delta_{d:g}" for d in available_deltas]
            + [f"err_delta_{d:g}" for d in available_deltas]
        )
        for r in records:
            accs = [r.delta_to_accuracy[d] for d in available_deltas]
            errs = [100.0 - a for a in accs]
            writer.writerow(
                [r.family, r.method, r.k, r.seed, r.elapsed_seconds, r.elapsed_seconds / 60.0]
                + accs
                + errs
            )
    return path


def _write_readme(out_dir: Path, max_k: Optional[int], records: Sequence[MethodRecord]) -> Path:
    path = out_dir / "README_runtime_plots.md"
    max_k_text = "all available K" if max_k is None else f"K <= {max_k}"
    methods = ", ".join(_method_label(m) for m in _available_methods(records))
    path.write_text(
        "\n".join(
            [
                "# Runtime LR CIFAR-10 Plots",
                "",
                f"Included runs: {max_k_text}.",
                f"Methods: {methods}.",
                "",
                "Runtime interpretation note: when `method_timings` exists in the JSON,",
                "the CSV uses explicit per-method train+evaluation time. For older combined",
                "runs without `method_timings`, the plotter falls back to run-level elapsed",
                "time, except ICNN-DRO where it uses the existing NPF checkpoint estimate.",
                "",
                "Generated figures:",
                "- `01_family_runtime_vs_k`: wall-clock scaling of each family.",
                "- `02_accuracy_vs_k_selected_deltas`: clean/robust accuracy versus K.",
                "- `03_accuracy_vs_runtime_selected_deltas`: accuracy-runtime tradeoff.",
                "- `04_robustness_curves_k*`: full error curves at the largest included K.",
                "- `05_accuracy_bars_k*`: clean/robust accuracy bars at the largest included K.",
                "- `06_efficiency_vs_k_delta_*`: accuracy points per runtime minute.",
                "- `07_family_runtime_scaling`: amortized and normalized runtime scaling, annotated with Δ=0.08 robust accuracy.",
                "- `runtime_summary_long.csv`: tidy source table for custom plots.",
                "",
            ]
        )
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime/robustness tradeoffs from Runtime-LR-CIFAR10 results."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--results_root", type=Path, default=script_dir / "results")
    parser.add_argument("--out_dir", type=Path, default=script_dir / "plots")
    parser.add_argument(
        "--max_k",
        type=_parse_max_k,
        default=25,
        help="Maximum K to include. Use 'all' or 'none' to include every completed JSON.",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.04, 0.08],
        help="Perturbation Δ values to highlight. Nearest available grid point is used.",
    )
    parser.add_argument(
        "--efficiency_delta",
        type=float,
        default=0.08,
        help="Perturbation Δ used for the runtime-efficiency plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.results_root / "run_manifest.tsv"
    runs = _read_manifest(manifest, args.max_k)
    if not runs:
        raise SystemExit(f"No completed runs with JSON results found under {args.results_root}")

    records, available_deltas = _load_method_records(runs)
    if not records:
        raise SystemExit("No method records found in completed JSON files.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = _write_summary_csv(records, available_deltas, args.out_dir)
    _plot_family_runtime(runs, records, available_deltas, args.out_dir)
    _plot_family_runtime_scaling(runs, records, available_deltas, args.out_dir)
    _plot_accuracy_vs_k(records, args.deltas, available_deltas, args.out_dir)
    _plot_tradeoff(records, args.deltas, available_deltas, args.out_dir)
    _plot_error_curves_at_best_k(records, available_deltas, args.out_dir, best_k=args.max_k)
    _plot_summary_bars(records, args.deltas, available_deltas, args.out_dir, best_k=args.max_k)
    _plot_runtime_efficiency(records, args.efficiency_delta, available_deltas, args.out_dir)
    readme = _write_readme(args.out_dir, args.max_k, records)

    print(f"Loaded {len(runs)} completed family runs and {len(records)} method records.")
    print(f"Plots saved to: {args.out_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Plot notes: {readme}")


if __name__ == "__main__":
    main()
