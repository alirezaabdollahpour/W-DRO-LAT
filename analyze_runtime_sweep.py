#!/usr/bin/env python3
"""Aggregate runtime_sweep results into a runtime + performance table.

Usage:
    python analyze_runtime_sweep.py runtime_sweep_K20_ddp4
    python analyze_runtime_sweep.py runtime_sweep_K20_ddp4 --latex
    python analyze_runtime_sweep.py runtime_sweep_K20_ddp4 --csv summary.csv

For each completed (algorithm, seed) cell:
  * Per-epoch wallclock — median, min, max (epoch 1 dropped: cuDNN autotune
    warmup is real but not representative of steady-state).
  * Total wallclock from /usr/bin/time -v.
  * Peak resident-set memory from time -v.
  * Final test_loss / test_acc on clean inputs.
  * Final adv_loss / adv_acc under the method's own transport-for-eval.
  * Final train_loss (rough convergence diagnostic).

When multiple seeds exist, reports the median and MAD of the per-seed
medians.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_epoch_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _adv_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only the rows for the standard adversarial training phase."""
    return [r for r in rows if r.get("phase", "adv") == "adv"]


def _epoch_seconds(rows: List[Dict[str, str]]) -> List[float]:
    """Per-epoch seconds, EXCLUDING epoch 1 (cuDNN autotune warmup)."""
    out: List[float] = []
    for r in rows:
        try:
            ep = int(r.get("epoch") or 0)
            t = float(r.get("epoch_seconds") or "")
        except (TypeError, ValueError):
            continue
        if ep == 1:
            continue
        out.append(t)
    return out


def _final_metric(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    """Return the metric from the LAST adv-phase row that has a value."""
    for r in reversed(rows):
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            return float(v)
        except ValueError:
            continue
    return None


def _parse_time_v(time_path: Path) -> Dict[str, str]:
    if not time_path.exists():
        return {}
    out: Dict[str, str] = {}
    pat = re.compile(r"^\s*(.+?):\s*(.*)$")
    for line in time_path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def _wallclock_seconds(time_summary: Dict[str, str]) -> float:
    raw = time_summary.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    if not raw:
        return float("nan")
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except ValueError:
        return float("nan")
    return float("nan")


def _mad(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def _fmt(v: Optional[float], spec: str = "{:.3f}", default: str = "—") -> str:
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return default
    return spec.format(v)


def _collect_one(
    result_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, List[Dict[str, str]]]:
    manifest = result_root / "run_manifest.tsv"
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}")

    by_algo: Dict[str, List[Dict[str, Any]]] = {}
    failed: List[Dict[str, str]] = []
    n_completed = 0
    with manifest.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            status = row.get("status", "")
            if status.startswith("failed"):
                failed.append(row)
                continue
            if not status.startswith("completed"):
                continue
            n_completed += 1
            algo = row["algorithm"]
            csv_path = Path(row["csv"])
            time_path = Path(row["time_file"])
            adv = _adv_rows(_read_epoch_csv(csv_path))
            ep_secs = _epoch_seconds(adv)
            t_summary = _parse_time_v(time_path)
            try:
                peak_rss_mb = float(t_summary.get("Maximum resident set size (kbytes)", "nan")) / 1024
            except ValueError:
                peak_rss_mb = float("nan")
            # GPU peak (the meaningful number for adversarial training): main.py
            # writes a summary.json next to the per-epoch CSV with rank-0's
            # ``torch.cuda.max_memory_allocated``. Older runs (before this
            # patch) won't have it; we silently report NaN in that case.
            peak_gpu_mb = float("nan")
            summary_path = csv_path.with_name("summary.json")
            if summary_path.exists():
                try:
                    payload = json.loads(summary_path.read_text())
                    peak_gpu_mb = float(payload.get("peak_gpu_alloc_mb", "nan"))
                except (json.JSONDecodeError, TypeError, ValueError):
                    peak_gpu_mb = float("nan")
            # Wallclock: prefer /usr/bin/time -v output; fall back to the
            # manifest's elapsed_seconds (always written by the bash
            # dispatcher, even when /usr/bin/time is unavailable).
            wallclock = _wallclock_seconds(t_summary)
            if wallclock != wallclock:  # NaN
                try:
                    wallclock = float(row.get("elapsed_seconds", "nan"))
                except ValueError:
                    wallclock = float("nan")
            entry: Dict[str, Any] = {
                "seed":          int(row["seed"]),
                "n_epochs":      len(ep_secs),
                "median_ep_s":   statistics.median(ep_secs) if ep_secs else float("nan"),
                "min_ep_s":      min(ep_secs) if ep_secs else float("nan"),
                "max_ep_s":      max(ep_secs) if ep_secs else float("nan"),
                "wallclock_s":   wallclock,
                "peak_rss_mb":   peak_rss_mb,
                "peak_gpu_mb":   peak_gpu_mb,
                "final_train_loss": _final_metric(adv, "train_loss"),
                "final_train_acc":  _final_metric(adv, "train_acc"),
                "final_test_loss":  _final_metric(adv, "test_loss"),
                "final_test_acc":   _final_metric(adv, "test_acc"),
                "final_adv_loss":   _final_metric(adv, "adv_loss"),
                "final_adv_acc":    _final_metric(adv, "adv_acc"),
                "final_pgd_acc":    _final_metric(adv, "input_pgd_acc"),
            }
            by_algo.setdefault(algo, []).append(entry)
    return by_algo, n_completed, failed


def _aggregate(by_algo: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for algo, runs in by_algo.items():
        agg: Dict[str, Any] = {"algorithm": algo, "seeds": len(runs)}
        for k in (
            "median_ep_s", "min_ep_s", "max_ep_s",
            "wallclock_s", "peak_rss_mb", "peak_gpu_mb",
            "final_train_loss", "final_train_acc",
            "final_test_loss",  "final_test_acc",
            "final_adv_loss",   "final_adv_acc",
            "final_pgd_acc",
        ):
            vals = [r[k] for r in runs if r[k] == r[k]]  # filter NaN
            agg[k]       = statistics.median(vals) if vals else float("nan")
            agg[k + "_mad"] = _mad(vals) if len(vals) > 1 else 0.0
        rows.append(agg)
    rows.sort(key=lambda r: r["median_ep_s"])  # fastest first
    return rows


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------
_PRESENT_ORDER = (
    "algorithm",
    "seeds",
    "median_ep_s",
    "min_ep_s",
    "max_ep_s",
    "wallclock_s",
    "peak_gpu_mb",
    "peak_rss_mb",
    "final_train_loss",
    "final_train_acc",
    "final_test_loss",
    "final_test_acc",
    "final_adv_loss",
    "final_adv_acc",
    "final_pgd_acc",
)


def _print_console_table(rows: List[Dict[str, Any]]) -> None:
    hdr = (
        f"{'algorithm':<10}  {'seeds':>5}  {'med ep (s)':>11}  {'min ep':>8}  "
        f"{'max ep':>8}  {'wall (s)':>9}  {'GPU (MB)':>9}  {'RSS (MB)':>9}  "
        f"{'train ℓ':>9}  {'train %':>9}  "
        f"{'test ℓ':>8}  {'test %':>8}  "
        f"{'adv ℓ':>8}  {'adv %':>8}  {'PGD %':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['algorithm']:<10}  "
            f"{r['seeds']:>5d}  "
            f"{_fmt(r['median_ep_s']):>11}  "
            f"{_fmt(r['min_ep_s']):>8}  "
            f"{_fmt(r['max_ep_s']):>8}  "
            f"{_fmt(r['wallclock_s'], '{:.1f}'):>9}  "
            f"{_fmt(r['peak_gpu_mb'], '{:.0f}'):>9}  "
            f"{_fmt(r['peak_rss_mb'], '{:.1f}'):>9}  "
            f"{_fmt(r['final_train_loss']):>9}  "
            f"{_fmt(r['final_train_acc'], '{:.2%}'):>9}  "
            f"{_fmt(r['final_test_loss']):>8}  "
            f"{_fmt(r['final_test_acc'], '{:.2%}'):>8}  "
            f"{_fmt(r['final_adv_loss']):>8}  "
            f"{_fmt(r['final_adv_acc'], '{:.2%}'):>8}  "
            f"{_fmt(r['final_pgd_acc'], '{:.2%}'):>7}"
        )


def _print_latex(rows: List[Dict[str, Any]]) -> None:
    print()
    print("% ====== LaTeX (paper-ready) ======")
    print(r"\begin{table}[t] \centering")
    print(
        r"\caption{Input-space adversarial training on CIFAR-10. Median per-epoch "
        r"wallclock (epoch 1 dropped to discount cuDNN autotune warmup), final "
        r"clean and under-transport test accuracy.}"
    )
    print(r"\label{tab:input_icnn_runtime}")
    print(r"\begin{tabular}{l rrr rr rr}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Per-epoch wallclock (s)} & \multicolumn{2}{c}{Clean test} & \multicolumn{2}{c}{Adv.\ test} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    print(r"Algorithm & Median & Min & Max & Loss & Acc (\%) & Loss & Acc (\%) \\")
    print(r"\midrule")
    for r in rows:
        med = _fmt(r["median_ep_s"], "{:.2f}", "—")
        mn = _fmt(r["min_ep_s"], "{:.2f}", "—")
        mx = _fmt(r["max_ep_s"], "{:.2f}", "—")
        tl = _fmt(r["final_test_loss"], "{:.3f}", "—")
        ta = _fmt(r["final_test_acc"], "{:.2%}", "—").rstrip("%").strip()
        al = _fmt(r["final_adv_loss"], "{:.3f}", "—")
        aa = _fmt(r["final_adv_acc"], "{:.2%}", "—").rstrip("%").strip()
        print(f"{r['algorithm']} & {med} & {mn} & {mx} & {tl} & {ta} & {al} & {aa} \\\\")
    print(r"\bottomrule \end{tabular} \end{table}")


def _dump_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_PRESENT_ORDER))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in _PRESENT_ORDER})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise runtime_sweep results.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--latex", action="store_true",
                        help="Also print a LaTeX table (paper-ready).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write summary CSV to this path.")
    args = parser.parse_args()

    by_algo, n_completed, failed = _collect_one(args.result_root)
    print(f"\n[summary] {n_completed} completed runs across {len(by_algo)} algorithms.")
    if failed:
        print(f"[summary] {len(failed)} FAILED rows in manifest:")
        for row in failed:
            print(
                f"  {row['status']:>16}  algo={row['algorithm']:<8}  seed={row['seed']}  "
                f"log={row['log_file']}"
            )
    print()
    if not by_algo:
        raise SystemExit("Nothing to summarise yet.")

    rows = _aggregate(by_algo)
    _print_console_table(rows)

    if args.latex:
        _print_latex(rows)
    if args.csv is not None:
        _dump_csv(rows, args.csv)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
