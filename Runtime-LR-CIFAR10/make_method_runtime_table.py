#!/usr/bin/env python3
"""Create a per-method runtime table for Runtime-LR-CIFAR10 results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


METHOD_LABELS = {
    "WRM": "PA",
    "RO": "RO",
    "Dual": "SDRO",
    "WFR": "WFR",
    "NPF": "ICNN-DRO",
}
METHOD_ORDER = ["WRM", "RO", "Dual", "WFR", "NPF"]


def _parse_max_k(value: str) -> Optional[int]:
    if value.lower() in {"all", "none", "inf"}:
        return None
    return int(value)


def _accuracy_for_delta(payload: Mapping[str, object], method: str, target_delta: float) -> float:
    levels = [float(v) for v in payload["perturbation_levels"]]  # type: ignore[index]
    epsilons = [float(v) for v in payload["epsilon_attack_values"]]  # type: ignore[index]
    idx = min(range(len(levels)), key=lambda i: abs(levels[i] - target_delta))
    eps_key = f"{epsilons[idx]:.10g}"
    method_runs = payload["results"][method]  # type: ignore[index]
    return float(method_runs[0][eps_key])


def _timing_from_payload(payload: Mapping[str, object], method: str) -> Optional[float]:
    method_timings = payload.get("method_timings")
    if not isinstance(method_timings, Mapping):
        return None
    timing = method_timings.get(method)
    if not isinstance(timing, Mapping):
        return None
    total_seconds = timing.get("total_seconds")
    if total_seconds is None:
        return None
    return float(total_seconds)


def _methods_from_command(command: str) -> Sequence[str]:
    if "--methods" not in command:
        return []
    tail = command.split("--methods", 1)[1].strip().split()
    methods = []
    for token in tail:
        if token.startswith("--"):
            break
        methods.append(token)
    return methods


def _iter_completed_rows(manifest: Path) -> Iterable[Dict[str, str]]:
    with manifest.open(newline="") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            if row.get("status") == "completed":
                yield row


def build_rows(manifest: Path, target_delta: float, max_k: Optional[int], allow_combined_fallback: bool) -> Sequence[Dict[str, object]]:
    rows = []
    for run in _iter_completed_rows(manifest):
        k = int(run["k"])
        if max_k is not None and k > max_k:
            continue
        out_dir = Path(run["out_dir"])
        json_paths = sorted(out_dir.glob("results_tau=*_epsent=*.json"))
        if not json_paths:
            continue
        payload = json.loads(json_paths[0].read_text())
        command_methods = _methods_from_command(run.get("command", ""))
        for method, method_runs in payload.get("results", {}).items():
            if method not in METHOD_LABELS or method == "NN-DRO" or not method_runs:
                continue
            seconds = _timing_from_payload(payload, method)
            timing_source = "method_timings.total_seconds"
            if seconds is None:
                if len(command_methods) == 1 and command_methods[0] == method:
                    seconds = float(run["elapsed_seconds"])
                    timing_source = "single-method manifest elapsed_seconds"
                elif allow_combined_fallback:
                    seconds = float(run["elapsed_seconds"])
                    timing_source = "combined-run elapsed_seconds (not explicit)"
                else:
                    seconds = None
                    timing_source = "missing explicit timing"
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "k": k,
                    "runtime_min": None if seconds is None else seconds / 60.0,
                    "robust_acc": _accuracy_for_delta(payload, method, target_delta),
                    "timing_source": timing_source,
                }
            )
    return sorted(rows, key=lambda r: (METHOD_ORDER.index(r["method"]), r["k"]))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=script_dir / "results" / "run_manifest.tsv")
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--max_k", type=_parse_max_k, default=50)
    parser.add_argument(
        "--allow_combined_fallback",
        action="store_true",
        help="Use combined-run elapsed time when explicit per-method timing is unavailable.",
    )
    args = parser.parse_args()

    rows = build_rows(args.manifest, args.delta, args.max_k, args.allow_combined_fallback)
    print(f"| Method | K | Runtime (min) | Robust acc. at Δ={args.delta:g} (%) | Timing source |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        runtime = row["runtime_min"]
        runtime_text = "NA" if runtime is None else f"{runtime:.2f}"
        print(
            f"| {row['label']} | {row['k']} | {runtime_text} | "
            f"{row['robust_acc']:.2f} | {row['timing_source']} |"
        )


if __name__ == "__main__":
    main()
