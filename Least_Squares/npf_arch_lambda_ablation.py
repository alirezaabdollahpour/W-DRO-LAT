"""Architecture/lambda ablation for WDRO-NPF and WDRO-NPF-LastQuad.

This runner is intentionally narrow: it trains only the two NPF variants,
uses the same hidden architecture for both variants at each grid point, times
training only, and writes a JSON file shaped for later ablation plotting.

Example:
    python npf_arch_lambda_ablation.py \
        --architectures 32x2 64x4 128x4 256,256,128,64 \
        --lams 0.01 0.0316 0.1 0.316 1.0 3.16 10.0 \
        --seeds 219 220 221 222 223 \
        --out results/npf_arch_lambda_ablation.json
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from algorithms.npf import (
    _solve_npf_icnn_map_with_zstar,
    _solve_npf_lastquad_icnn_map_with_zstar,
)
from config import ULSConfig, build_arg_parser, config_from_args
from utils.common import seed_everything
from utils.data import generate_training_data, make_problem
from utils.loss import loss_function
from utils.pareto_metrics import evaluate_test_loss, w2_sq_optimal, w2_sq_paired


METHODS = {
    "npf": {
        "display_name": "WDRO-NPF",
        "train_fn": _solve_npf_icnn_map_with_zstar,
    },
    "npf_lastquad": {
        "display_name": "WDRO-NPF-LastQuad",
        "train_fn": _solve_npf_lastquad_icnn_map_with_zstar,
    },
}

DEFAULT_ARCHITECTURES = ("32x2", "64x2", "64x4", "128x4")
DEFAULT_LAMS = (0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0)
DEFAULT_SEEDS = (219, 220, 221, 222, 223)
DEFAULT_DELTAS = (0.0, 1.0, 5.0, 10.0)
_LAM_ROUND = 12


def parse_architecture(spec: str) -> Tuple[int, ...]:
    """Parse '64,64,32' or shorthand '64x3' into a hidden-size tuple."""
    raw = spec.strip().lower()
    if not raw:
        raise ValueError("Empty architecture specification.")
    if "x" in raw and "," not in raw:
        width_s, depth_s = raw.split("x", 1)
        width = int(width_s)
        depth = int(depth_s)
        arch = tuple([width] * depth)
    else:
        arch = tuple(int(part) for part in raw.split(",") if part.strip())
    if len(arch) == 0:
        raise ValueError(f"Architecture {spec!r} has no layers.")
    if any(width <= 0 for width in arch):
        raise ValueError(f"Architecture {spec!r} must contain positive widths.")
    return arch


def arch_id(arch: Tuple[int, ...]) -> str:
    return "x".join(str(width) for width in arch)


def canonical_architectures(specs: Iterable[str]) -> List[Tuple[int, ...]]:
    seen = set()
    out = []
    for spec in specs:
        arch = parse_architecture(spec)
        if arch not in seen:
            seen.add(arch)
            out.append(arch)
    return out


def json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)
    tmp_path.replace(path)


def load_existing(path: Path, overwrite: bool) -> Dict[str, Any] | None:
    if overwrite or not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def make_payload(
    cfg: ULSConfig,
    architectures: List[Tuple[int, ...]],
    lams: List[float],
    seeds: List[int],
    deltas: List[float],
    device_str: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "least_squares_npf_arch_lambda_ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "WDRO-NPF vs WDRO-NPF-LastQuad over matched hidden architectures, "
            "lambda values, and seeds. train_time_wall_sec measures training "
            "only, with CUDA synchronization before and after each timed call."
        ),
        "methods": [
            {"key": key, "display_name": meta["display_name"]}
            for key, meta in METHODS.items()
        ],
        "architectures": [
            {
                "arch_id": arch_id(arch),
                "hidden_sizes": list(arch),
                "depth": len(arch),
                "max_width": max(arch),
            }
            for arch in architectures
        ],
        "lams": [float(v) for v in lams],
        "seeds": [int(v) for v in seeds],
        "deltas": [float(v) for v in deltas],
        "device": device_str,
        "timing": {
            "timer": "time.perf_counter",
            "training_only": True,
            "cuda_synchronized": device_str.startswith("cuda"),
            "sequential_execution": True,
        },
        "base_config": asdict(cfg),
        "runs": [],
    }


def _run_key(row: Dict[str, Any]) -> Tuple[str, str, float, int]:
    return (
        str(row["method"]),
        str(row["arch_id"]),
        round(float(row["lam"]), _LAM_ROUND),
        int(row["seed"]),
    )


def completed_keys(payload: Dict[str, Any]) -> set:
    return {
        _run_key(row)
        for row in payload.get("runs", [])
        if row.get("status") == "ok"
    }


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_trainable_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def make_problem_and_data(cfg: ULSConfig, device: torch.device):
    seed_everything(cfg.seed)
    A0, A1, b = make_problem(cfg, device)
    xi_train = generate_training_data(cfg, device)
    return A0, A1, b, xi_train


def run_one(
    method: str,
    arch: Tuple[int, ...],
    cfg: ULSConfig,
    deltas: List[float],
    device: torch.device,
) -> Dict[str, Any]:
    train_fn = METHODS[method]["train_fn"]
    A0, A1, b, xi_train = make_problem_and_data(cfg, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cpu_t0 = time.process_time()
    sync_if_cuda(device)
    wall_t0 = time.perf_counter()
    out = train_fn(xi_train, cfg, A0, A1, b)
    sync_if_cuda(device)
    train_wall = time.perf_counter() - wall_t0
    train_cpu = time.process_time() - cpu_t0

    theta = out["theta"]
    psi = out["psi"]
    z_star = out["z_star"]
    z_hat = xi_train.unsqueeze(-1) if xi_train.dim() == 1 else xi_train
    z_eval = z_star.unsqueeze(-1) if z_star.dim() == 1 else z_star

    with torch.no_grad():
        clean_train_loss = float(
            loss_function(theta, xi_train, A0, A1, b, cfg.dim_m).mean().item()
        )
        train_loss_z_star = float(
            loss_function(theta, z_star.reshape(-1), A0, A1, b, cfg.dim_m)
            .mean()
            .item()
        )
        adv_objective = float(
            (
                loss_function(theta, z_star.reshape(-1), A0, A1, b, cfg.dim_m)
                - cfg.lam * (z_star.reshape(-1) - xi_train.reshape(-1)) ** 2
            )
            .mean()
            .item()
        )

    test_losses = {
        f"delta_{delta:g}": evaluate_test_loss(
            theta,
            A0,
            A1,
            b,
            delta=float(delta),
            dim_m=cfg.dim_m,
            n_test=cfg.n_test,
            seed=cfg.seed + 100_000,
        )
        for delta in deltas
    }

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    diagnostics = out.get("diagnostics", {})
    return {
        "status": "ok",
        "method": method,
        "display_name": METHODS[method]["display_name"],
        "arch_id": arch_id(arch),
        "hidden_sizes": list(arch),
        "depth": len(arch),
        "max_width": max(arch),
        "lam": float(cfg.lam),
        "seed": int(cfg.seed),
        "train_time_wall_sec": float(train_wall),
        "train_time_cpu_sec": float(train_cpu),
        "cuda_peak_memory_bytes": peak_memory,
        "trainable_params": count_trainable_params(psi),
        "theta_norm": float(theta.norm().item()),
        "w2_paired": w2_sq_paired(z_hat, z_eval),
        "w2_optimal": w2_sq_optimal(z_hat, z_eval),
        "clean_train_loss": clean_train_loss,
        "train_loss_z_star": train_loss_z_star,
        "adv_objective": adv_objective,
        "test_losses": test_losses,
        "z_star": {
            "mean": float(z_star.mean().item()),
            "std": float(z_star.std(unbiased=False).item()),
            "min": float(z_star.min().item()),
            "max": float(z_star.max().item()),
            "mean_abs_displacement": float((z_star - xi_train).abs().mean().item()),
            "max_abs_displacement": float((z_star - xi_train).abs().max().item()),
        },
        "diagnostics": diagnostics,
    }


def build_parser():
    parser = build_arg_parser()
    parser.description = (
        "Run a matched architecture/lambda ablation for WDRO-NPF and "
        "WDRO-NPF-LastQuad only."
    )
    parser.set_defaults(
        algorithms=["npf", "npf_lastquad"],
        skip_eval=True,
        plots=False,
        results_dir=None,
    )

    ablation = parser.add_argument_group("npf architecture/lambda ablation")
    ablation.add_argument(
        "--architectures",
        nargs="+",
        default=list(DEFAULT_ARCHITECTURES),
        help=(
            "Hidden architectures shared by NPF and NPF-LastQuad. "
            "Use comma lists like 64,64,32 or shorthand like 64x4."
        ),
    )
    ablation.add_argument(
        "--lams",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMS),
        help="Lambda values to sweep.",
    )
    ablation.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to sweep. Default is five seeds: 219..223.",
    )
    ablation.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DELTAS),
        help="Test-distribution deltas to record after training.",
    )
    ablation.add_argument(
        "--out",
        type=Path,
        default=Path("npf_arch_lambda_ablation.json"),
        help="Output JSON path.",
    )
    ablation.add_argument(
        "--device",
        type=str,
        default=None,
        help="Defaults to cuda if available, else cpu.",
    )
    ablation.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard any existing output JSON instead of resuming it.",
    )
    ablation.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run count and grid without training.",
    )
    ablation.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed training run.",
    )
    return parser


def main() -> None:
    torch.set_default_dtype(torch.float64)
    parser = build_parser()
    args = parser.parse_args()
    cfg_base = config_from_args(args)

    architectures = canonical_architectures(args.architectures)
    lams = [float(v) for v in args.lams]
    seeds = [int(v) for v in args.seeds]
    deltas = [float(v) for v in args.deltas]
    if any(lam < 0.0 for lam in lams):
        raise ValueError("All lambda values must be non-negative.")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    n_total = len(METHODS) * len(architectures) * len(lams) * len(seeds)
    print(
        f"NPF ablation: {len(METHODS)} methods x {len(architectures)} architectures "
        f"x {len(lams)} lambdas x {len(seeds)} seeds = {n_total} trainings "
        f"(device={device_str}).",
        flush=True,
    )
    print(
        "Architectures: "
        + ", ".join(f"{arch_id(arch)}={list(arch)}" for arch in architectures),
        flush=True,
    )

    if args.dry_run:
        return

    existing = load_existing(args.out, args.overwrite)
    if existing is None:
        payload = make_payload(cfg_base, architectures, lams, seeds, deltas, device_str)
    else:
        payload = existing
        payload.setdefault("runs", [])
        payload["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()

    done = completed_keys(payload)
    for arch in architectures:
        for lam in lams:
            for seed in seeds:
                cfg = replace(
                    cfg_base,
                    seed=seed,
                    lam=lam,
                    npf_hidden=arch,
                    npf_lastquad_hidden=arch,
                )
                for method in METHODS:
                    key = (method, arch_id(arch), round(float(lam), _LAM_ROUND), seed)
                    if key in done:
                        print(
                            f"skip method={method} arch={arch_id(arch)} "
                            f"lam={lam:g} seed={seed}",
                            flush=True,
                        )
                        continue

                    print(
                        f"run method={method} arch={arch_id(arch)} "
                        f"lam={lam:g} seed={seed}",
                        flush=True,
                    )
                    row_base = {
                        "method": method,
                        "display_name": METHODS[method]["display_name"],
                        "arch_id": arch_id(arch),
                        "hidden_sizes": list(arch),
                        "lam": float(lam),
                        "seed": int(seed),
                    }
                    try:
                        row = run_one(method, arch, cfg, deltas, device)
                        print(
                            f"  ok t={row['train_time_wall_sec']:.6f}s "
                            f"params={row['trainable_params']}",
                            flush=True,
                        )
                    except Exception as exc:
                        row = {
                            **row_base,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                        if args.fail_fast:
                            payload["runs"].append(row)
                            write_json(args.out, payload)
                            raise

                    payload["runs"].append(row)
                    write_json(args.out, payload)
                    if row.get("status") == "ok":
                        done.add(key)

    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out, payload)
    n_ok = sum(1 for row in payload["runs"] if row.get("status") == "ok")
    n_failed = sum(1 for row in payload["runs"] if row.get("status") == "failed")
    print(f"Done. Wrote {args.out} ({n_ok} ok, {n_failed} failed).", flush=True)


if __name__ == "__main__":
    main()
