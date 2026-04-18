"""Cyclical monotonicity diagnostics for WRM couplings."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch


def check_cyclical_monotonicity(
    x: torch.Tensor,
    z: torch.Tensor,
    cycle_lengths: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
    num_samples: int = 500,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, Dict[str, Any]]:
    """Estimate CM violations of the identity coupling (x_i, z_i).

    For each cycle length k, sample random k-tuples and compare identity cost
    to a cyclically shifted reassignment.
    """
    N = x.size(0)
    x_flat = x.detach().view(N, -1)
    z_flat = z.detach().view(N, -1)

    id_costs = (z_flat - x_flat).pow(2).sum(dim=1)

    results: Dict[int, Dict[str, Any]] = {}

    for k in cycle_lengths:
        if k > N:
            continue

        violations_raw = []
        violations_per_edge = []
        violations_relative = []

        for _ in range(num_samples):
            idx = torch.randperm(N, device=x.device, generator=generator)[:k]
            c_id = id_costs[idx].sum().item()
            idx_shifted = idx.roll(-1)
            c_cyc = (z_flat[idx_shifted] - x_flat[idx]).pow(2).sum().item()

            raw = c_id - c_cyc
            violations_raw.append(raw)
            violations_per_edge.append(raw / k)
            violations_relative.append(raw / max(c_id, 1e-12))

        raw_t = torch.tensor(violations_raw)
        pe_t = torch.tensor(violations_per_edge)
        rel_t = torch.tensor(violations_relative)

        results[k] = {
            "cycle_length": k,
            "mean_raw": float(raw_t.mean()),
            "std_raw": float(raw_t.std()),
            "mean_per_edge": float(pe_t.mean()),
            "std_per_edge": float(pe_t.std()),
            "mean_relative": float(rel_t.mean()),
            "std_relative": float(rel_t.std()),
            "frac_violated": float((raw_t > 0).float().mean()),
            "max_raw": float(raw_t.max()),
            "max_per_edge": float(pe_t.max()),
            "max_relative": float(rel_t.max()),
            "num_samples": num_samples,
        }

    return results


def aggregate_cm_results(
    batch_results: Sequence[Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Average CM diagnostics across mini-batches."""
    if len(batch_results) == 0:
        return {}

    all_k = sorted({k for res in batch_results for k in res})
    agg: Dict[int, Dict[str, Any]] = {}

    for k in all_k:
        entries = [res[k] for res in batch_results if k in res]
        if len(entries) == 0:
            continue
        n = len(entries)
        agg[k] = {
            "cycle_length": k,
            "mean_raw": sum(e["mean_raw"] for e in entries) / n,
            "std_raw": sum(e["std_raw"] for e in entries) / n,
            "mean_per_edge": sum(e["mean_per_edge"] for e in entries) / n,
            "std_per_edge": sum(e["std_per_edge"] for e in entries) / n,
            "mean_relative": sum(e["mean_relative"] for e in entries) / n,
            "std_relative": sum(e["std_relative"] for e in entries) / n,
            "frac_violated": sum(e["frac_violated"] for e in entries) / n,
            "max_raw": max(e["max_raw"] for e in entries),
            "max_per_edge": max(e["max_per_edge"] for e in entries),
            "max_relative": max(e["max_relative"] for e in entries),
            "num_batches": n,
        }
    return agg
