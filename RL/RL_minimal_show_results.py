"""
Visualize and compare RL_minimal JSON results (ICNN vs Particle Ascent).

Example:
  python RL_minimal_show_results.py \
    RL_minimal_cartpole_both_seed0_lam_2.0_*.json \
    --show

  python RL_minimal_show_results.py \
    RL_minimal_icnn_seed0_lam_2.0_*.json \
    RL_minimal_icnn_seed8_lam_2.0_*.json \
    RL_minimal_particle_seed0_lam_2.0_*.json \
    RL_minimal_particle_seed8_lam_2.0_*.json \
    --show

If multiple seeds are provided for the same (method, lambda), the script plots the mean over
seeds (optionally with ±1 std shading) to make comparisons less noisy.

By default, if --show is not passed, the script saves a comparison figure next to this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Environment-specific xi names (mirrors RL_minimal.py ENV_SPECS)
# ---------------------------------------------------------------------------

_ENV_XI_NAMES: dict[str, tuple[str, ...]] = {
    "cartpole": ("masspole", "length"),
    "swingup_pendulum": ("m", "l", "b"),
}


def _infer_xi_names(args: dict[str, Any], cfg_inner: dict[str, Any]) -> tuple[str, ...]:
    """Infer xi dimension names from the JSON args/cfg_inner."""
    env = str(args.get("env", "")).strip().lower()
    # canonical aliases
    if env in {"pendulum", "swingup", "swingup-pendulum"}:
        env = "swingup_pendulum"
    if env in _ENV_XI_NAMES:
        return _ENV_XI_NAMES[env]
    # fallback: generate generic names from xi_low dimension
    xi_low = cfg_inner.get("xi_low") or []
    return tuple(f"xi{i}" for i in range(len(xi_low)))


@dataclass(frozen=True)
class MethodRun:
    method: str
    source_path: Path
    created_utc: str | None
    seed: int | None
    lam: float | None
    args: dict[str, Any]
    cfg_inner: dict[str, Any]
    logs: dict[str, Any]
    hat_xi_plot: np.ndarray
    xi_adv_plot: np.ndarray


@dataclass(frozen=True)
class MethodAggregate:
    method: str
    lam: float | None
    seeds: tuple[int, ...]
    source_paths: tuple[Path, ...]
    created_utcs: tuple[str | None, ...]
    args: dict[str, Any]
    cfg_inner: dict[str, Any]
    iters: np.ndarray
    j_nom_mean: np.ndarray
    j_nom_std: np.ndarray
    j_worst_mean: np.ndarray
    j_worst_std: np.ndarray
    logdet_mean: np.ndarray
    logdet_std: np.ndarray
    hat_xi_plot: np.ndarray
    xi_adv_plot: np.ndarray


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _coerce_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _load_runs(json_path: Path) -> list[MethodRun]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    args = payload.get("args") or {}
    cfg_inner = payload.get("cfg_inner") or {}
    methods = payload.get("methods") or {}
    lam = _coerce_float(args.get("lam"))

    runs: list[MethodRun] = []
    for method, md in methods.items():
        md = md or {}
        logs = md.get("logs") or {}
        hat = _as_float_array(md.get("hat_xi_plot") or [])
        adv = _as_float_array(md.get("xi_adv_plot") or [])
        runs.append(
            MethodRun(
                method=str(method),
                source_path=json_path,
                created_utc=meta.get("created_utc"),
                seed=args.get("seed"),
                lam=lam,
                args=args,
                cfg_inner=cfg_inner,
                logs=logs,
                hat_xi_plot=hat,
                xi_adv_plot=adv,
            )
        )

    if not runs:
        raise ValueError(f"No methods found in {json_path}. Expected key 'methods'.")
    return runs


def _select_latest_per_method_seed_lam(runs: list[MethodRun]) -> list[MethodRun]:
    """
    De-duplicate potentially repeated runs by keeping the latest created_utc for each
    (method, lam, seed) triple.
    """
    by_key: dict[tuple[str, float | None, int | None], list[MethodRun]] = {}
    for run in runs:
        by_key.setdefault((run.method, run.lam, run.seed), []).append(run)

    chosen: list[MethodRun] = []
    for (method, lam, seed), candidates in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1.0, kv[0][2] or -1)):
        candidates = sorted(candidates, key=lambda r: r.created_utc or "", reverse=True)
        if len(candidates) > 1:
            newest = candidates[0]
            print(
                f"[warn] method={method} lam={lam} seed={seed}: found {len(candidates)} runs; using latest created_utc={newest.created_utc}"
            )
        chosen.append(candidates[0])
    return chosen


def _infer_single_lambda(runs: list[MethodRun], *, requested_lam: float | None) -> float | None:
    """
    Determine which lambda to plot. If requested_lam is provided, return it (after validation).
    Otherwise:
      - If exactly one lambda appears in provided runs, use it.
      - If multiple lambdas appear, raise an error asking the user to specify --lam.
    """
    lams = sorted({r.lam for r in runs if r.lam is not None})

    if requested_lam is not None:
        if not lams:
            return float(requested_lam)
        if any(np.isclose(float(requested_lam), float(x), rtol=0.0, atol=1e-12) for x in lams):
            return float(requested_lam)
        raise ValueError(f"--lam={requested_lam} not found in provided runs. Available lambdas: {lams}")

    if len(lams) == 1:
        return float(lams[0])
    if len(lams) == 0:
        return None
    raise ValueError(f"Multiple lambdas found in provided runs: {lams}. Please specify --lam <value>.")


def _stack_metric_by_iter(runs: list[MethodRun], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      iters: (T,) sorted unique iteration indices across runs
      values: (N,T) where values[i,t] is the metric for run i at iters[t] (NaN if missing)
    """
    if not runs:
        return np.asarray([], dtype=int), np.asarray([[]], dtype=float)

    per_run: list[dict[int, float]] = []
    it_union: set[int] = set()
    for run in runs:
        logs = run.logs or {}
        it = np.asarray(logs.get("iter") or [], dtype=int).reshape(-1)
        vals = np.asarray(logs.get(metric_key) or [], dtype=float).reshape(-1)
        n = min(it.size, vals.size)
        it = it[:n]
        vals = vals[:n]
        d = {int(i): float(v) for i, v in zip(it.tolist(), vals.tolist())}
        per_run.append(d)
        it_union.update(d.keys())

    iters = np.asarray(sorted(it_union), dtype=int)
    values = np.full((len(runs), iters.size), np.nan, dtype=float)
    if iters.size == 0:
        return iters, values

    it_index = {int(it): j for j, it in enumerate(iters.tolist())}
    for i, d in enumerate(per_run):
        for it, v in d.items():
            j = it_index.get(int(it))
            if j is not None:
                values[i, j] = float(v)
    return iters, values


def _concat_2d(arrs: list[np.ndarray]) -> np.ndarray:
    arrs = [a for a in arrs if isinstance(a, np.ndarray) and a.size]
    if not arrs:
        return np.asarray([], dtype=float).reshape(0, 0)
    # Drop incompatible shapes (best-effort).
    d = arrs[0].shape[1] if arrs[0].ndim == 2 else None
    kept: list[np.ndarray] = []
    for a in arrs:
        if a.ndim != 2:
            continue
        if d is not None and a.shape[1] != d:
            continue
        kept.append(a)
    if not kept:
        return np.asarray([], dtype=float).reshape(0, 0)
    return np.concatenate(kept, axis=0)


def _aggregate_by_method(runs: list[MethodRun], *, lam: float | None) -> list[MethodAggregate]:
    by_method: dict[str, list[MethodRun]] = {}
    for r in runs:
        if lam is not None and (r.lam is None or not np.isclose(float(r.lam), float(lam), rtol=0.0, atol=1e-12)):
            continue
        by_method.setdefault(r.method, []).append(r)

    aggregates: list[MethodAggregate] = []
    for method in sorted(by_method.keys()):
        method_runs = sorted(by_method[method], key=lambda r: (r.seed is None, r.seed or -1))
        if not method_runs:
            continue

        it, j_nom_mat = _stack_metric_by_iter(method_runs, "J_nominal")
        _, j_worst_mat = _stack_metric_by_iter(method_runs, "J_worst_grid")
        _, logdet_mat = _stack_metric_by_iter(method_runs, "adv_cov_logdet")

        def mean_std(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if mat.size == 0:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            mean = np.where(np.isfinite(mean), mean, np.nan)
            std = np.where(np.isfinite(std), std, np.nan)
            return mean, std

        j_nom_mean, j_nom_std = mean_std(j_nom_mat)
        j_worst_mean, j_worst_std = mean_std(j_worst_mat)
        logdet_mean, logdet_std = mean_std(logdet_mat)

        seeds = tuple(sorted({int(r.seed) for r in method_runs if r.seed is not None}))
        aggregates.append(
            MethodAggregate(
                method=method,
                lam=lam if lam is not None else method_runs[0].lam,
                seeds=seeds,
                source_paths=tuple(r.source_path for r in method_runs),
                created_utcs=tuple(r.created_utc for r in method_runs),
                args=method_runs[0].args,
                cfg_inner=method_runs[0].cfg_inner,
                iters=it,
                j_nom_mean=j_nom_mean,
                j_nom_std=j_nom_std,
                j_worst_mean=j_worst_mean,
                j_worst_std=j_worst_std,
                logdet_mean=logdet_mean,
                logdet_std=logdet_std,
                hat_xi_plot=_concat_2d([r.hat_xi_plot for r in method_runs]),
                xi_adv_plot=_concat_2d([r.xi_adv_plot for r in method_runs]),
            )
        )
    return aggregates


def _safe_logdet_cov(x: np.ndarray) -> float:
    if x.size == 0 or x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, (x.shape[0] - 1))
    cov = cov + 1e-12 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    if not np.isfinite(logdet) or sign <= 0:
        return float("nan")
    return float(logdet)


def _frac_at_bounds(x: np.ndarray, low: np.ndarray, high: np.ndarray, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        d = low.shape[0]
        return np.full((d,), np.nan), np.full((d,), np.nan)
    at_low = np.mean(np.abs(x - low) <= tol, axis=0)
    at_high = np.mean(np.abs(x - high) <= tol, axis=0)
    return at_low, at_high


def _default_out_path() -> Path:
    out_dir = Path(__file__).resolve().parent
    base = out_dir / "RL_minimal_compare.png"
    if not base.exists():
        return base
    k = 1
    while True:
        cand = out_dir / f"RL_minimal_compare_{k}.png"
        if not cand.exists():
            return cand
        k += 1


def _print_summary(runs: list[MethodAggregate]) -> None:
    print(f"Loaded {len(runs)} method aggregate(s).")
    for run in runs:
        n_seeds = len(run.seeds) if run.seeds else len(run.source_paths)
        lam = run.lam
        seed_str = ",".join(str(s) for s in run.seeds) if run.seeds else "?"
        print(f"\n[{run.method}] lam={lam}  seeds(n={n_seeds})=[{seed_str}]")

        it = run.iters
        if it.size:
            print(f"  logged_iters: n={it.size}  first={int(it[0])}  last={int(it[-1])}")
        if run.j_nom_mean.size:
            print(f"  J_nominal (mean): final={run.j_nom_mean[-1]:.3f}  max={np.nanmax(run.j_nom_mean):.3f}")
        if run.j_worst_mean.size:
            print(f"  J_worst_grid (mean): final={run.j_worst_mean[-1]:.3f}  max={np.nanmax(run.j_worst_mean):.3f}")
        if run.logdet_mean.size:
            print(
                f"  adv_cov_logdet (mean): final={run.logdet_mean[-1]:.3f}  max={np.nanmax(run.logdet_mean):.3f}"
            )

        low = np.asarray((run.cfg_inner.get("xi_low") or [np.nan, np.nan]), dtype=float)
        high = np.asarray((run.cfg_inner.get("xi_high") or [np.nan, np.nan]), dtype=float)
        adv = run.xi_adv_plot
        if adv.size:
            mean = np.mean(adv, axis=0)
            std = np.std(adv, axis=0)
            logdet = _safe_logdet_cov(adv)
            at_low, at_high = _frac_at_bounds(adv, low=low, high=high)
            print(
                "  adv_xi_plot: "
                f"mean=({mean[0]:.4f},{mean[1]:.4f})  std=({std[0]:.4f},{std[1]:.4f})  logdetCov={logdet:.3f}"
            )
            if np.all(np.isfinite(at_low)) and np.all(np.isfinite(at_high)):
                print(
                    "  adv_xi_plot: "
                    f"frac_at_low=({at_low[0]:.2f},{at_low[1]:.2f})  frac_at_high=({at_high[0]:.2f},{at_high[1]:.2f})"
                )


def _build_config_subtitle(runs: list[MethodAggregate]) -> str:
    """Build a concise subtitle string from experiment configuration."""
    if not runs:
        return ""
    # Use the first run's args for shared config
    args = runs[0].args
    cfg = runs[0].cfg_inner
    parts: list[str] = []

    env = args.get("env", "?")
    parts.append(f"env={env}")

    grad = args.get("grad_method") or cfg.get("grad_method")
    if grad:
        parts.append(f"grad={grad}")

    iters = args.get("iters")
    if iters is not None:
        parts.append(f"iters={iters}")

    k_icnn = cfg.get("K_icnn")
    if k_icnn is not None:
        parts.append(f"K_icnn={k_icnn}")

    eta_icnn = cfg.get("eta_icnn")
    if eta_icnn is not None:
        parts.append(f"eta_icnn={eta_icnn}")

    beta = cfg.get("icnn_softplus_beta")
    if beta is not None:
        parts.append(f"softplus_beta={beta}")

    icnn_init = cfg.get("icnn_init")
    if icnn_init:
        parts.append(f"init={icnn_init}")

    hidden = cfg.get("icnn_hidden_sizes")
    if hidden:
        parts.append(f"hidden={hidden}")

    return "  ".join(parts)


def _build_final_metrics_text(runs_by_method: dict[str, MethodAggregate]) -> str:
    """Build a text block summarizing final metrics for each method."""
    lines: list[str] = []
    for method, run in runs_by_method.items():
        n = len(run.seeds) if run.seeds else len(run.source_paths)
        n = max(1, int(n))
        parts = [f"{method.upper()} (n={n})"]
        if run.j_nom_mean.size:
            parts.append(f"J_nom={run.j_nom_mean[-1]:.1f}")
        if run.j_worst_mean.size:
            parts.append(f"J_worst={run.j_worst_mean[-1]:.1f}")
        if run.logdet_mean.size:
            parts.append(f"logdet={run.logdet_mean[-1]:.2f}")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _plot(
    runs: list[MethodAggregate],
    *,
    out_path: Path | None,
    show: bool,
    title: str,
    dpi: int,
    shade_std: bool,
    xi_names: tuple[str, ...] | None = None,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs_by_method = {r.method: r for r in runs}
    methods = list(runs_by_method.keys())

    # Infer xi_names from data if not provided
    if xi_names is None:
        ref = runs[0] if runs else None
        if ref is not None:
            xi_names = _infer_xi_names(ref.args, ref.cfg_inner)
        else:
            xi_names = ("xi0", "xi1")

    colors = {"icnn": "#1f77b4", "particle": "#ff7f0e"}
    fallback = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, m in enumerate(methods):
        colors.setdefault(m, fallback[idx % len(fallback)])

    config_sub = _build_config_subtitle(runs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_ret, ax_div = axes[0]
    ax_dom_0, ax_dom_1 = axes[1]

    # --- Returns ---
    for method, run in runs_by_method.items():
        it = run.iters
        if it.size == 0:
            continue
        j_nom = run.j_nom_mean
        j_worst = run.j_worst_mean
        c = colors[method]
        if j_nom.size:
            ax_ret.plot(it, j_nom, color=c, linestyle="-", linewidth=2)
            if shade_std and run.j_nom_std.size == j_nom.size:
                ax_ret.fill_between(
                    it,
                    j_nom - run.j_nom_std,
                    j_nom + run.j_nom_std,
                    color=c,
                    alpha=0.15,
                    linewidth=0,
                )
        if j_worst.size:
            ax_ret.plot(it, j_worst, color=c, linestyle="--", linewidth=2)
            if shade_std and run.j_worst_std.size == j_worst.size:
                ax_ret.fill_between(
                    it,
                    j_worst - run.j_worst_std,
                    j_worst + run.j_worst_std,
                    color=c,
                    alpha=0.10,
                    linewidth=0,
                )

    method_handles = []
    for m in methods:
        r = runs_by_method[m]
        n = len(r.seeds) if r.seeds else len(r.source_paths)
        n = max(1, int(n))
        method_handles.append(Line2D([0], [0], color=colors[m], lw=2, label=f"{m} (n={n})"))
    metric_handles = [
        Line2D([0], [0], color="k", lw=2, linestyle="-", label=r"$J_{\rm nominal}$"),
        Line2D([0], [0], color="k", lw=2, linestyle="--", label=r"$J_{\rm worst}$ (grid)"),
    ]
    ax_ret.set_title("Expected Return vs Outer Iteration", fontsize=11)
    ax_ret.set_xlabel("outer iteration")
    ax_ret.set_ylabel("return")
    ax_ret.grid(alpha=0.25)
    ax_ret.legend(handles=method_handles + metric_handles, ncol=2, frameon=False, fontsize=9)

    # Final-metrics annotation on returns plot
    metrics_text = _build_final_metrics_text(runs_by_method)
    if metrics_text:
        ax_ret.text(
            0.02, 0.02, metrics_text,
            transform=ax_ret.transAxes, va="bottom", ha="left",
            fontsize=8, family="monospace",
            bbox={"facecolor": "lightyellow", "alpha": 0.85, "edgecolor": "#cccccc",
                  "boxstyle": "round,pad=0.4"},
        )

    # --- Diversity proxy ---
    for method, run in runs_by_method.items():
        it = run.iters
        logdet = run.logdet_mean
        if it.size and logdet.size:
            ax_div.plot(it, logdet, color=colors[method], linewidth=2, label=method)
            if shade_std and run.logdet_std.size == logdet.size:
                ax_div.fill_between(
                    it,
                    logdet - run.logdet_std,
                    logdet + run.logdet_std,
                    color=colors[method],
                    alpha=0.15,
                    linewidth=0,
                )

    ax_div.set_title(r"Adversarial Diversity: $\log\det\,\mathrm{Cov}(\xi_{\rm adv})$", fontsize=11)
    ax_div.set_xlabel("outer iteration")
    ax_div.set_ylabel(r"$\log\det$")
    ax_div.grid(alpha=0.25)
    if methods:
        ax_div.legend(frameon=False, fontsize=9)

    # --- Domain distributions ---
    dom_axes = [ax_dom_0, ax_dom_1]
    for ax, method in zip(dom_axes, methods, strict=False):
        run = runs_by_method[method]
        hat = run.hat_xi_plot
        adv = run.xi_adv_plot
        low = np.asarray((run.cfg_inner.get("xi_low") or [np.nan, np.nan]), dtype=float)
        high = np.asarray((run.cfg_inner.get("xi_high") or [np.nan, np.nan]), dtype=float)

        # Draw domain boundary rectangle
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (float(low[0]), float(low[1])),
                float(high[0] - low[0]), float(high[1] - low[1]),
                linewidth=1.5, edgecolor="#888888", facecolor="none",
                linestyle="--", label=r"$\Xi$ bounds",
            )
            ax.add_patch(rect)

        if hat.size:
            ax.scatter(
                hat[:, 0], hat[:, 1], s=10, alpha=0.25, color="#7f7f7f",
                label=r"$\hat{\xi} \sim \hat{P}$", zorder=2,
            )
        if adv.size:
            ax.scatter(
                adv[:, 0], adv[:, 1], s=12, alpha=0.65, color=colors[method],
                label=r"$\xi_{\rm adv}$", zorder=3,
            )

        n = len(run.seeds) if run.seeds else len(run.source_paths)
        n = max(1, int(n))

        method_label = "ICNN Transport" if method == "icnn" else "Particle Ascent"
        ax.set_title(f"Domain Distribution: {method_label} (n={n})", fontsize=11)

        # Use env-specific xi_names for axis labels
        x_label = xi_names[0] if len(xi_names) > 0 else "xi0"
        y_label = xi_names[1] if len(xi_names) > 1 else "xi1"
        ax.set_xlabel(rf"$\xi_0$ ({x_label})")
        ax.set_ylabel(rf"$\xi_1$ ({y_label})")
        ax.grid(alpha=0.25)

        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            margin_x = (high[0] - low[0]) * 0.05
            margin_y = (high[1] - low[1]) * 0.05
            ax.set_xlim(float(low[0] - margin_x), float(high[0] + margin_x))
            ax.set_ylim(float(low[1] - margin_y), float(high[1] + margin_y))

        if adv.size:
            mean = np.mean(adv, axis=0)
            std = np.std(adv, axis=0)
            logdet = _safe_logdet_cov(adv)
            at_low, at_high = _frac_at_bounds(adv, low=low, high=high)
            text_parts = [
                f"n_samples={adv.shape[0]}",
                f"mean=({mean[0]:.3f}, {mean[1]:.3f})",
                f"std =({std[0]:.3f}, {std[1]:.3f})",
                f"logdet Cov={logdet:.2f}",
            ]
            if np.all(np.isfinite(at_low)) and np.all(np.isfinite(at_high)):
                text_parts.append(
                    f"frac@low=({at_low[0]:.2f}, {at_low[1]:.2f})"
                )
                text_parts.append(
                    f"frac@high=({at_high[0]:.2f}, {at_high[1]:.2f})"
                )
            ax.text(
                0.02, 0.98, "\n".join(text_parts),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, family="monospace",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc",
                      "boxstyle": "round,pad=0.3"},
            )

        ax.legend(frameon=False, fontsize=9, loc="lower right")

    # Hide unused domain axes (if only one method provided).
    for ax in dom_axes[len(methods) :]:
        ax.axis("off")

    # Suptitle with config subtitle
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if config_sub:
        fig.text(
            0.5, 0.975, config_sub,
            ha="center", va="top", fontsize=8, color="#555555", family="monospace",
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        print(f"\nSaved figure: {out_path}")

    if show:
        if os.environ.get("DISPLAY") is None and os.name != "nt":
            print("[warn] DISPLAY is not set; matplotlib --show may not work in this environment.")
        plt.show()

    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare and visualize RL_minimal JSON results (icnn vs particle).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_paths", nargs="+", type=Path, help="Paths to RL_minimal_*.json outputs.")
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Filter runs to a specific lambda value (args.lam). Required if multiple lambdas are provided.",
    )
    parser.add_argument(
        "--no-shade",
        action="store_true",
        help="Disable ±1 std shading around mean curves (when aggregating across seeds).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output image path (png/pdf/svg).")
    parser.add_argument("--show", action="store_true", help="Show interactive figures.")
    parser.add_argument("--title", type=str, default="RL_minimal: icnn vs particle")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    all_runs: list[MethodRun] = []
    for p in args.json_paths:
        all_runs.extend(_load_runs(p))

    runs_dedup = _select_latest_per_method_seed_lam(all_runs)
    if not runs_dedup:
        raise ValueError("No runs loaded.")

    lam = _infer_single_lambda(runs_dedup, requested_lam=args.lam)
    aggregates = _aggregate_by_method(runs_dedup, lam=lam)
    if not aggregates:
        raise ValueError("No matching runs after filtering (check --lam).")

    out_path = args.out
    if out_path is None and not args.show:
        out_path = _default_out_path()

    title = str(args.title)
    if title == "RL_minimal: icnn vs particle":
        # Build an informative auto-title from the data
        ref_args = aggregates[0].args if aggregates else {}
        env = ref_args.get("env", "?")
        method_names = " vs ".join(
            ("ICNN" if a.method == "icnn" else "Particle") for a in aggregates
        )
        lam_str = f", $\\lambda$={lam}" if lam is not None else ""
        seeds_all = sorted({s for a in aggregates for s in a.seeds})
        seed_str = f", seeds={seeds_all}" if len(seeds_all) > 1 else ""
        title = f"Robust Domain Randomization: {method_names} ({env}{lam_str}{seed_str})"

    # Infer xi_names from data
    ref = aggregates[0] if aggregates else None
    xi_names = _infer_xi_names(ref.args, ref.cfg_inner) if ref else None

    _print_summary(aggregates)
    _plot(
        aggregates,
        out_path=out_path,
        show=bool(args.show),
        title=title,
        dpi=int(args.dpi),
        shade_std=not bool(args.no_shade),
        xi_names=xi_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
