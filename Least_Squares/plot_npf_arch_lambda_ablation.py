"""Paper-style plots for the NPF architecture/lambda ablation.

The ablation runner can be interrupted and leave a partial JSON. This plotter
uses only complete architecture-lambda cells: every expected seed must have an
``ok`` row for every method at that cell. Incomplete cells are summarized in a
text report and masked in the heatmaps.

Example:
    python Least_Squares/plot_npf_arch_lambda_ablation.py

Outputs default to ``Least_Squares/figs/npf_arch_lambda_ablation``:
    - aggregate_summary.csv
    - completion_report.txt
    - ablation_heatmaps_<metric>.pdf/.png
    - lambda_curves_<metric>.pdf/.png
    - efficiency_tradeoff_<metric>.pdf/.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "npf_arch_lambda_ablation.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "figs" / "npf_arch_lambda_ablation"
LAM_ROUND = 12

METHOD_STYLE = {
    "npf": {
        "label": "ICNN-DRO",
        "color": "#C8202B",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "npf_lastquad": {
        "label": "ICNN-DRO (LQ)",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": (0, (4, 1.4)),
        "linewidth": 1.8,
    },
}

ARCH_COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
]


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def lam_key(lam: float) -> float:
    return round(float(lam), LAM_ROUND)


def fmt_lam(lam: float) -> str:
    return f"{float(lam):g}"


def safe_metric_name(metric: str) -> str:
    return (
        metric.replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def arch_label(arch_id: str, arch_meta: Dict[str, Any] | None = None) -> str:
    hidden = None
    if arch_meta is not None:
        hidden = arch_meta.get("hidden_sizes")
    if hidden is None:
        try:
            hidden = [int(part) for part in arch_id.split("x")]
        except ValueError:
            hidden = None
    if hidden and len(set(hidden)) == 1:
        return f"{hidden[0]} x {len(hidden)}"
    if hidden:
        return "x".join(str(v) for v in hidden)
    return arch_id


def arch_sort_key(arch_id: str, arch_meta: Dict[str, Any] | None = None) -> Tuple[int, int, Tuple[int, ...], str]:
    hidden = None
    if arch_meta is not None:
        hidden = arch_meta.get("hidden_sizes")
    if hidden is None:
        try:
            hidden = [int(part) for part in arch_id.split("x")]
        except ValueError:
            hidden = None
    if hidden:
        hidden_tuple = tuple(int(v) for v in hidden)
        return len(hidden_tuple), max(hidden_tuple), hidden_tuple, arch_id
    return 10**9, 10**9, tuple(), arch_id


def method_label(method: str, method_labels: Dict[str, str]) -> str:
    return METHOD_STYLE.get(method, {}).get("label", method_labels.get(method, method))


def metric_label(metric: str) -> str:
    if metric.endswith("__seed_centered"):
        base = metric.replace("__seed_centered", "")
        if base.startswith("test_loss_delta_"):
            delta = base.replace("test_loss_delta_", "")
            return rf"Seed-centered test loss at shift $\Delta={delta}$"
        return f"Seed-centered {metric_label(base)}"
    if metric.startswith("test_loss_delta_"):
        delta = metric.replace("test_loss_delta_", "")
        return rf"Test loss at shift $\Delta={delta}$"
    labels = {
        "adv_objective": r"Adversarial objective",
        "train_loss_z_star": r"Train loss on transported samples",
        "clean_train_loss": r"Clean train loss",
        "w2_optimal": r"$W_2^2$ optimal matching",
        "w2_paired": r"$W_2^2$ paired map",
        "train_time_wall_sec": r"Training wall time (s)",
        "train_time_cpu_sec": r"Training CPU time (s)",
        "trainable_params": r"Trainable parameters",
        "theta_norm": r"$\|\theta\|_2$",
        "z_star.mean_abs_displacement": r"Mean displacement",
        "z_star.max_abs_displacement": r"Max displacement",
    }
    return labels.get(metric, metric.replace("_", " "))


def short_metric_label(metric: str) -> str:
    if metric.endswith("__seed_centered"):
        return f"seed-centered {short_metric_label(metric.replace('__seed_centered', ''))}"
    if metric.startswith("test_loss_delta_"):
        delta = metric.replace("test_loss_delta_", "")
        return rf"test loss, $\Delta={delta}$"
    labels = {
        "train_time_wall_sec": "wall time",
        "trainable_params": "parameters",
        "adv_objective": "adv. objective",
        "w2_optimal": r"$W_2^2$",
        "z_star.mean_abs_displacement": "mean displacement",
    }
    return labels.get(metric, metric_label(metric))


def value_from_row(row: Dict[str, Any], metric: str) -> float:
    derived = row.get("_derived", {})
    if metric in derived:
        return float(derived[metric])

    if metric.startswith("test_loss_delta_"):
        suffix = metric.replace("test_loss_delta_", "")
        candidates = [f"delta_{suffix}"]
        try:
            candidates.append(f"delta_{float(suffix):g}")
        except ValueError:
            pass
        losses = row.get("test_losses", {})
        for key in candidates:
            if key in losses:
                return float(losses[key])
        return float("nan")

    cur: Any = row
    for part in metric.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return float("nan")
        cur = cur[part]
    if cur is None:
        return float("nan")
    try:
        val = float(cur)
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def available_test_metrics(rows: Sequence[Dict[str, Any]]) -> List[str]:
    metrics = set()
    for row in rows:
        for key in row.get("test_losses", {}):
            if key.startswith("delta_"):
                suffix = key.replace("delta_", "")
                metrics.add(f"test_loss_delta_{suffix}")

    def sort_key(metric: str) -> Tuple[float, str]:
        suffix = metric.replace("test_loss_delta_", "")
        try:
            return float(suffix), suffix
        except ValueError:
            return float("inf"), suffix

    return sorted(metrics, key=sort_key)


def default_metrics(rows: Sequence[Dict[str, Any]]) -> List[str]:
    base = [
        "adv_objective",
        "train_loss_z_star",
        "clean_train_loss",
        "w2_optimal",
        "w2_paired",
        "train_time_wall_sec",
        "train_time_cpu_sec",
        "trainable_params",
        "theta_norm",
        "z_star.mean_abs_displacement",
        "z_star.max_abs_displacement",
    ]
    return available_test_metrics(rows) + base


def bootstrap_median_ci(
    values: Iterable[float],
    rng: np.random.Generator,
    B: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(arr))
    if arr.size == 1 or B <= 0:
        return point, point, point
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boots = np.median(arr[idx], axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, lo, hi


def summarize_values(
    values: Iterable[float],
    rng: np.random.Generator,
    B: int,
    alpha: float,
) -> Dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    median, ci_low, ci_high = bootstrap_median_ci(arr, rng, B=B, alpha=alpha)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "median": median,
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": float(std / math.sqrt(arr.size)) if arr.size > 0 else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def metadata_order(payload: Dict[str, Any]) -> Tuple[List[str], List[float], List[int], List[str]]:
    archs = [str(a["arch_id"]) for a in payload.get("architectures", [])]
    lams = [float(v) for v in payload.get("lams", [])]
    seeds = [int(v) for v in payload.get("seeds", [])]
    methods = [str(m["key"]) for m in payload.get("methods", [])]

    runs = payload.get("runs", [])
    if not archs:
        archs = sorted({str(r.get("arch_id")) for r in runs if r.get("arch_id")})
    if not lams:
        lams = sorted({float(r.get("lam")) for r in runs if r.get("lam") is not None})
    if not seeds:
        seeds = sorted({int(r.get("seed")) for r in runs if r.get("seed") is not None})
    if not methods:
        methods = sorted({str(r.get("method")) for r in runs if r.get("method")})
    return archs, lams, seeds, methods


def complete_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    archs, lams, seeds, methods = metadata_order(payload)
    method_labels = {
        str(m["key"]): str(m.get("display_name", m["key"]))
        for m in payload.get("methods", [])
    }
    arch_meta = {str(a["arch_id"]): dict(a) for a in payload.get("architectures", [])}

    ok_rows = [r for r in payload.get("runs", []) if r.get("status") == "ok"]
    row_by_key: Dict[Tuple[str, str, float, int], Dict[str, Any]] = {}
    duplicates = 0
    for row in ok_rows:
        key = (
            str(row["method"]),
            str(row["arch_id"]),
            lam_key(float(row["lam"])),
            int(row["seed"]),
        )
        if key in row_by_key:
            duplicates += 1
        row_by_key[key] = row

    complete_cells = []
    incomplete_cells = []
    expected = len(methods) * len(seeds)
    for arch_id in archs:
        for lam in lams:
            present_by_method = {}
            present = 0
            for method in methods:
                method_seeds = [
                    seed
                    for seed in seeds
                    if (method, arch_id, lam_key(lam), int(seed)) in row_by_key
                ]
                present_by_method[method] = method_seeds
                present += len(method_seeds)
            cell = {
                "arch_id": arch_id,
                "lam": float(lam),
                "present": present,
                "expected": expected,
                "present_by_method": present_by_method,
            }
            if present == expected and expected > 0:
                complete_cells.append(cell)
            else:
                incomplete_cells.append(cell)

    complete_cell_keys = {(c["arch_id"], lam_key(c["lam"])) for c in complete_cells}
    complete_rows = [
        row
        for row in row_by_key.values()
        if (str(row["arch_id"]), lam_key(float(row["lam"]))) in complete_cell_keys
    ]
    dropped_ok_rows = len(row_by_key) - len(complete_rows)

    return {
        "archs": archs,
        "lams": lams,
        "seeds": seeds,
        "methods": methods,
        "method_labels": method_labels,
        "arch_meta": arch_meta,
        "ok_rows": ok_rows,
        "complete_rows": complete_rows,
        "complete_cells": complete_cells,
        "incomplete_cells": incomplete_cells,
        "duplicates": duplicates,
        "dropped_ok_rows": dropped_ok_rows,
        "expected_per_cell": expected,
    }


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    method_labels: Dict[str, str],
    arch_meta: Dict[str, Dict[str, Any]],
    B: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["arch_id"]), lam_key(row["lam"]))].append(row)

    out = []
    for (method, arch_id, lam), group in sorted(
        groups.items(),
        key=lambda kv: (arch_sort_key(kv[0][1], arch_meta.get(kv[0][1])), kv[0][2], kv[0][0]),
    ):
        first = group[0]
        meta = arch_meta.get(arch_id, {})
        hidden = first.get("hidden_sizes", meta.get("hidden_sizes", []))
        record: Dict[str, Any] = {
            "method": method,
            "display_name": method_label(method, method_labels),
            "arch_id": arch_id,
            "arch_label": arch_label(arch_id, meta),
            "hidden_sizes": "x".join(str(v) for v in hidden),
            "depth": first.get("depth", meta.get("depth", "")),
            "max_width": first.get("max_width", meta.get("max_width", "")),
            "lam": float(lam),
            "n_seeds": len({int(r["seed"]) for r in group}),
            "seeds": " ".join(str(int(r["seed"])) for r in sorted(group, key=lambda r: int(r["seed"]))),
        }
        for metric in metrics:
            vals = [value_from_row(row, metric) for row in group]
            stats = summarize_values(vals, rng, B=B, alpha=alpha)
            prefix = safe_metric_name(metric)
            for name, value in stats.items():
                record[f"{prefix}_{name}"] = value
        out.append(record)
    return out


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "method",
        "display_name",
        "arch_id",
        "arch_label",
        "hidden_sizes",
        "depth",
        "max_width",
        "lam",
        "n_seeds",
        "seeds",
    ]
    extra = sorted({k for row in rows for k in row if k not in base})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base + extra)
        writer.writeheader()
        writer.writerows(rows)


def write_completion_report(info: Dict[str, Any], path: Path) -> str:
    archs = info["archs"]
    lams = info["lams"]
    methods = info["methods"]
    seeds = info["seeds"]
    complete_cells = info["complete_cells"]
    incomplete_cells = info["incomplete_cells"]
    expected = info["expected_per_cell"]
    arch_meta = info["arch_meta"]
    method_labels = info["method_labels"]

    complete_by_arch: Dict[str, List[float]] = defaultdict(list)
    for cell in complete_cells:
        complete_by_arch[cell["arch_id"]].append(float(cell["lam"]))

    lines = [
        "NPF architecture/lambda ablation completion report",
        f"Methods: {', '.join(method_label(m, method_labels) for m in methods)}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Architectures in metadata: {', '.join(arch_label(a, arch_meta.get(a)) for a in archs)}",
        f"Lambdas in metadata: {', '.join(fmt_lam(l) for l in lams)}",
        "",
        (
            f"Complete architecture-lambda cells: {len(complete_cells)} / "
            f"{len(archs) * len(lams)}"
        ),
        f"Expected ok rows per complete cell: {expected}",
        f"Ok rows retained for plots: {len(info['complete_rows'])}",
        f"Ok rows dropped from incomplete cells: {info['dropped_ok_rows']}",
        f"Duplicate ok rows overwritten by latest row: {info['duplicates']}",
        "",
        "Complete cells used:",
    ]
    for arch_id in archs:
        vals = sorted(complete_by_arch.get(arch_id, []))
        if vals:
            lines.append(
                f"  {arch_label(arch_id, arch_meta.get(arch_id))}: "
                + ", ".join(fmt_lam(v) for v in vals)
            )
        else:
            lines.append(f"  {arch_label(arch_id, arch_meta.get(arch_id))}: none")

    partial = [c for c in incomplete_cells if c["present"] > 0]
    if partial:
        lines.extend(["", "Incomplete cells with partial progress:"])
        for cell in partial:
            method_parts = []
            for method, present_seeds in cell["present_by_method"].items():
                label = method_label(method, method_labels)
                seeds_s = ",".join(str(s) for s in present_seeds) or "-"
                method_parts.append(f"{label}: [{seeds_s}]")
            lines.append(
                "  "
                + f"{arch_label(cell['arch_id'], arch_meta.get(cell['arch_id']))}, "
                + f"lambda={fmt_lam(cell['lam'])}: "
                + f"{cell['present']}/{cell['expected']} ok; "
                + "; ".join(method_parts)
            )

    report = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return report


def summary_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    return {
        (str(row["method"]), str(row["arch_id"]), lam_key(float(row["lam"]))): row
        for row in rows
    }


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def metric_stat(row: Dict[str, Any], metric: str, stat: str = "median") -> float:
    key = f"{safe_metric_name(metric)}_{stat}"
    val = row.get(key, float("nan"))
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def fmt_value(v: float, signed: bool = False) -> str:
    if not math.isfinite(v):
        return ""
    sign = "+" if signed and v >= 0 else ""
    av = abs(v)
    if av >= 1000:
        return f"{sign}{v:.0f}"
    if av >= 100:
        return f"{sign}{v:.1f}"
    if av >= 10:
        return f"{sign}{v:.2f}"
    if av >= 1:
        return f"{sign}{v:.3f}"
    if av >= 1e-2:
        return f"{sign}{v:.4f}"
    return f"{sign}{v:.1e}"


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.png')}")


def common_plot_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle=":", linewidth=0.55, alpha=0.55, color="#7a7a7a")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def matrix_for(
    rows_by_key: Dict[Tuple[str, str, float], Dict[str, Any]],
    method: str,
    archs: Sequence[str],
    lams: Sequence[float],
    metric: str,
    stat: str = "median",
) -> np.ndarray:
    mat = np.full((len(archs), len(lams)), np.nan, dtype=float)
    for i, arch_id in enumerate(archs):
        for j, lam in enumerate(lams):
            row = rows_by_key.get((method, arch_id, lam_key(lam)))
            if row is not None:
                mat[i, j] = metric_stat(row, metric, stat=stat)
    return mat


def annotate_heatmap(ax: plt.Axes, mat: np.ndarray, norm: mcolors.Normalize, signed: bool = False) -> None:
    for (i, j), val in np.ndenumerate(mat):
        if not math.isfinite(float(val)):
            continue
        rgba = plt.get_cmap(ax.images[-1].cmap.name)(norm(val))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = "white" if luminance < 0.45 else "black"
        ax.text(j, i, fmt_value(float(val), signed=signed), ha="center", va="center", fontsize=6.2, color=color)


def highlight_min(ax: plt.Axes, mat: np.ndarray) -> None:
    vals = finite_values(mat.ravel())
    if vals.size == 0:
        return
    best = float(np.min(vals))
    locs = np.argwhere(np.isclose(mat, best, rtol=1e-10, atol=1e-12))
    for i, j in locs:
        ax.add_patch(
            mpatches.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="white",
                linewidth=1.8,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=0.7,
            )
        )


def plot_heatmaps(
    summary_rows: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    metric: str,
    out_path: Path,
) -> None:
    rows_by_key = summary_index(summary_rows)
    methods = [m for m in info["methods"] if any(r["method"] == m for r in summary_rows)]
    archs = [a for a in info["archs"] if any(r["arch_id"] == a for r in summary_rows)]
    lams = sorted({float(r["lam"]) for r in summary_rows})
    if not methods or not archs or not lams:
        print("No complete rows available for heatmap.")
        return

    mats = [matrix_for(rows_by_key, method, archs, lams, metric) for method in methods]
    finite = finite_values(np.concatenate([m.ravel() for m in mats]))
    if finite.size == 0:
        print(f"No finite values for {metric}; skipping heatmap.")
        return

    has_pair = len(methods) >= 2
    ncols = len(methods) + (1 if has_pair else 0)
    fig_w = max(11.5, 3.75 * ncols + 0.6)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.9), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([axes])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    method_images = []
    for ax, method, mat in zip(axes, methods, mats):
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        method_images.append(im)
        ax.set_title(method_label(method, info["method_labels"]))
        ax.set_xticks(range(len(lams)), [fmt_lam(lam) for lam in lams], rotation=35, ha="right")
        ax.set_yticks(range(len(archs)), [arch_label(a, info["arch_meta"].get(a)) for a in archs])
        ax.set_xlabel(r"$\lambda$")
        annotate_heatmap(ax, mat, norm=norm)
        highlight_min(ax, mat)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    axes[0].set_ylabel("hidden architecture")
    cbar = fig.colorbar(method_images[0], ax=axes[: len(methods)], shrink=0.86, pad=0.015)
    cbar.set_label(short_metric_label(metric))
    cbar.ax.tick_params(labelsize=7)

    if has_pair:
        diff_ax = axes[-1]
        diff = mats[1] - mats[0]
        diff_finite = finite_values(diff.ravel())
        if diff_finite.size:
            lim = float(np.max(np.abs(diff_finite)))
            if lim == 0:
                lim = 1.0
            diff_norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            diff_cmap = plt.get_cmap("RdBu_r").copy()
            diff_cmap.set_bad("#eeeeee")
            im = diff_ax.imshow(np.ma.masked_invalid(diff), cmap=diff_cmap, norm=diff_norm, aspect="auto")
            title = (
                f"{method_label(methods[1], info['method_labels'])}\n"
                f"$-$ {method_label(methods[0], info['method_labels'])}"
            )
            diff_ax.set_title(title)
            diff_ax.set_xticks(range(len(lams)), [fmt_lam(lam) for lam in lams], rotation=35, ha="right")
            diff_ax.set_yticks(range(len(archs)), [])
            diff_ax.set_xlabel(r"$\lambda$")
            annotate_heatmap(diff_ax, diff, norm=diff_norm, signed=True)
            cbar_diff = fig.colorbar(im, ax=diff_ax, shrink=0.86, pad=0.015)
            cbar_diff.set_label("difference")
            cbar_diff.ax.tick_params(labelsize=7)

    fig.suptitle(f"Complete-cell ablation: median {short_metric_label(metric)}", y=1.05, fontsize=9.5)
    save_figure(fig, out_path)


def plot_lambda_curves(
    summary_rows: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    metric: str,
    out_path: Path,
) -> None:
    rows_by_key = summary_index(summary_rows)
    methods = [m for m in info["methods"] if any(r["method"] == m for r in summary_rows)]
    archs = [a for a in info["archs"] if any(r["arch_id"] == a for r in summary_rows)]
    lams = sorted({float(r["lam"]) for r in summary_rows})
    if not methods or not archs or not lams:
        print("No complete rows available for lambda curves.")
        return

    fig_w = max(10.5, 3.35 * len(archs))
    fig, axes = plt.subplots(1, len(archs), figsize=(fig_w, 3.85), sharey=True, constrained_layout=True)
    if len(archs) == 1:
        axes = np.asarray([axes])

    all_band_values = []
    for ax, arch_id in zip(axes, archs):
        for method in methods:
            xs, ys, lo, hi = [], [], [], []
            for lam in lams:
                row = rows_by_key.get((method, arch_id, lam_key(lam)))
                if row is None:
                    continue
                y = metric_stat(row, metric, "median")
                ylo = metric_stat(row, metric, "ci_low")
                yhi = metric_stat(row, metric, "ci_high")
                if not math.isfinite(y):
                    continue
                xs.append(float(lam))
                ys.append(y)
                lo.append(ylo if math.isfinite(ylo) else y)
                hi.append(yhi if math.isfinite(yhi) else y)
            if not xs:
                continue
            style = METHOD_STYLE.get(method, {})
            color = style.get("color", None)
            ax.plot(
                xs,
                ys,
                label=method_label(method, info["method_labels"]),
                color=color,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.6),
                markersize=3.5,
            )
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)
            all_band_values.extend(lo)
            all_band_values.extend(hi)
        ax.set_xscale("log")
        ax.set_title(arch_label(arch_id, info["arch_meta"].get(arch_id)))
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(lams)
        ax.set_xticklabels([fmt_lam(lam) for lam in lams], rotation=35, ha="right")
        common_plot_axes(ax)

    axes[0].set_ylabel(metric_label(metric))
    finite_band = finite_values(all_band_values)
    if finite_band.size:
        lo, hi = float(np.min(finite_band)), float(np.max(finite_band))
        pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        axes[0].set_ylim(lo - pad, hi + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.suptitle("Lambda sensitivity over complete cells", y=1.02, fontsize=9.5)
    save_figure(fig, out_path)


def lambda_norm(lams: Sequence[float]) -> mcolors.Normalize:
    positive = [float(l) for l in lams if float(l) > 0]
    if len(positive) == len(lams) and len(set(positive)) > 1:
        return mcolors.LogNorm(vmin=min(positive), vmax=max(positive))
    return mcolors.Normalize(vmin=min(lams), vmax=max(lams))


def plot_efficiency_tradeoff(
    summary_rows: Sequence[Dict[str, Any]],
    info: Dict[str, Any],
    metric: str,
    out_path: Path,
) -> None:
    methods = [m for m in info["methods"] if any(r["method"] == m for r in summary_rows)]
    lams = sorted({float(r["lam"]) for r in summary_rows})
    if not summary_rows or not methods or not lams:
        print("No complete rows available for efficiency plot.")
        return

    norm = lambda_norm(lams)
    cmap = plt.get_cmap("viridis")
    panels = [
        ("trainable_params", "median", "Trainable parameters"),
        ("train_time_wall_sec", "median", "Training wall time (s)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15), sharey=True, constrained_layout=True)

    for ax, (x_metric, x_stat, x_label) in zip(axes, panels):
        for method in methods:
            for lam in lams:
                pts = []
                for row in summary_rows:
                    if row["method"] != method or not math.isclose(float(row["lam"]), lam):
                        continue
                    x = metric_stat(row, x_metric, x_stat)
                    y = metric_stat(row, metric, "median")
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((x, y))
                if len(pts) >= 2:
                    pts.sort(key=lambda xy: xy[0])
                    xs, ys = zip(*pts)
                    ax.plot(
                        xs,
                        ys,
                        color=cmap(norm(lam)),
                        linestyle=METHOD_STYLE.get(method, {}).get("linestyle", "-"),
                        linewidth=0.75,
                        alpha=0.34,
                    )

        for row in summary_rows:
            x = metric_stat(row, x_metric, x_stat)
            y = metric_stat(row, metric, "median")
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            method = str(row["method"])
            lam = float(row["lam"])
            ax.scatter(
                x,
                y,
                s=28,
                marker=METHOD_STYLE.get(method, {}).get("marker", "o"),
                facecolor=cmap(norm(lam)),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        common_plot_axes(ax)
    axes[0].set_ylabel(metric_label(metric))
    axes[0].set_title("Capacity")
    axes[1].set_title("Training cost")

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=METHOD_STYLE.get(method, {}).get("marker", "o"),
            linestyle="None",
            markersize=4.5,
            label=method_label(method, info["method_labels"]),
        )
        for method in methods
    ]
    axes[0].legend(handles=method_handles, loc="best", frameon=False, handletextpad=0.4)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.84, pad=0.02)
    cbar.set_label(r"$\lambda$")
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Efficiency trade-off using complete cells ({short_metric_label(metric)})", y=1.04, fontsize=9.5)
    save_figure(fig, out_path)


def choose_quality_metric(rows: Sequence[Dict[str, Any]], requested: str | None) -> str:
    if requested:
        return requested
    test_metrics = available_test_metrics(rows)
    if test_metrics:
        return test_metrics[-1]
    return "adv_objective"


def add_seed_centered_metric(rows: Sequence[Dict[str, Any]], metric: str) -> str:
    """Add metric minus each seed's median complete-cell value to rows."""
    derived_metric = f"{metric}__seed_centered"
    by_seed: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        val = value_from_row(row, metric)
        if math.isfinite(val):
            by_seed[int(row["seed"])].append(val)

    centers = {
        seed: float(np.median(values))
        for seed, values in by_seed.items()
        if len(values) > 0
    }
    for row in rows:
        seed = int(row["seed"])
        val = value_from_row(row, metric)
        centered = val - centers[seed] if seed in centers and math.isfinite(val) else float("nan")
        row.setdefault("_derived", {})[derived_metric] = centered
    return derived_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NPF architecture/lambda ablations while filtering out "
            "interrupted, incomplete grid cells."
        )
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Ablation JSON file.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for figures and tables.")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help=(
            "Quality metric to plot. Default is the largest available "
            "test_loss_delta_* metric, usually test_loss_delta_10."
        ),
    )
    parser.add_argument("--bootstrap_B", type=int, default=2000, help="Bootstrap draws for median CI bands.")
    parser.add_argument("--bootstrap_seed", type=int, default=0, help="Seed for bootstrap resampling.")
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="Alpha for bootstrap CI. Default gives 95%% CI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    payload = load_payload(args.json)
    info = complete_dataset(payload)
    quality_metric = choose_quality_metric(info["complete_rows"], args.metric)
    seed_centered_metric = add_seed_centered_metric(info["complete_rows"], quality_metric)
    metrics = default_metrics(info["complete_rows"])
    if quality_metric not in metrics:
        metrics = [quality_metric] + metrics
    if seed_centered_metric not in metrics:
        metrics = [seed_centered_metric] + metrics

    if not info["complete_rows"]:
        report = write_completion_report(info, args.out_dir / "completion_report.txt")
        print(report)
        raise SystemExit("No complete architecture-lambda cells found.")

    summary_rows = aggregate_rows(
        info["complete_rows"],
        metrics=metrics,
        method_labels=info["method_labels"],
        arch_meta=info["arch_meta"],
        B=args.bootstrap_B,
        alpha=args.ci_alpha,
        seed=args.bootstrap_seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, args.out_dir / "aggregate_summary.csv")
    report = write_completion_report(info, args.out_dir / "completion_report.txt")
    print(report)
    print(f"wrote {args.out_dir / 'aggregate_summary.csv'}")

    metric_safe = safe_metric_name(quality_metric)
    plot_heatmaps(
        summary_rows,
        info,
        quality_metric,
        args.out_dir / f"ablation_heatmaps_{metric_safe}.pdf",
    )
    plot_lambda_curves(
        summary_rows,
        info,
        quality_metric,
        args.out_dir / f"lambda_curves_{metric_safe}.pdf",
    )
    plot_lambda_curves(
        summary_rows,
        info,
        seed_centered_metric,
        args.out_dir / f"lambda_curves_seed_centered_{metric_safe}.pdf",
    )
    plot_efficiency_tradeoff(
        summary_rows,
        info,
        quality_metric,
        args.out_dir / f"efficiency_tradeoff_{metric_safe}.pdf",
    )


if __name__ == "__main__":
    main()
