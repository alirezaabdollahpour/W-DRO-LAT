"""
Evaluate trained RL policies across CartPole environment variants and produce
a table (text + LaTeX) similar to:

    Environment | Regular        | Particle       | Robust (ICNN)
    Original    | 399.7 ± 0.1   | 400.0 ± 0.0   | 400.0 ± 0.0
    Heavy       | 150.1 ± 4.7   | 280.2 ± 5.1   | 334.0 ± 3.7
    ...

Usage:
  # Evaluate three checkpoints (regular vs particle vs icnn) across all variants:
  python RL_eval_table.py \
    --regular  RL_minimal_..._nominal_policy.pt \
    --particle RL_minimal_..._particle_policy.pt \
    --robust   RL_minimal_..._icnn_policy.pt \
    --trials 1000 --max-steps 500

  # Evaluate a single checkpoint:
  python RL_eval_table.py \
    --robust RL_minimal_..._icnn_policy.pt \
    --trials 1000

  # Custom variants via JSON:
  python RL_eval_table.py \
    --robust checkpoint.pt \
    --variants-json my_variants.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Replicate the minimal policy/env definitions from RL_minimal.py so that
# this script is self-contained (no circular imports).
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_policy_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[PolicyNet, dict[str, Any]]:
    """Load a PolicyNet plus metadata from a checkpoint saved by RL_minimal.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy = PolicyNet(
        obs_dim=int(ckpt["obs_dim"]),
        hidden=int(ckpt["hidden"]),
        act_dim=int(ckpt["act_dim"]),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, ckpt


# ---------------------------------------------------------------------------
# CartPole environment (standalone, supports gravity override)
# ---------------------------------------------------------------------------

def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed % (2**63))
    return g


class CartPoleEval:
    """Vectorized CartPole for evaluation. Supports per-variant gravity."""

    def __init__(
        self,
        max_steps: int = 500,
        gravity: float = 9.8,
        masscart: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.max_steps = int(max_steps)
        self.gravity = torch.tensor(gravity, device=device)
        self.masscart = torch.tensor(masscart, device=device)
        self.force_mag = torch.tensor(10.0, device=device)
        self.tau = torch.tensor(0.02, device=device)
        self.action_values = torch.tensor([-10.0, 10.0], device=device)
        self.x_threshold = torch.tensor(2.4, device=device)
        self.theta_threshold_radians = torch.tensor(12 * math.pi / 180, device=device)
        self.state: Optional[torch.Tensor] = None
        self.steps: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, seed: int = 0) -> torch.Tensor:
        g = _make_generator(seed, self.device)
        self.state = (torch.rand(batch_size, 4, generator=g, device=self.device) * 0.1 - 0.05).float()
        self.steps = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        return self.state

    def step(self, action: torch.Tensor, xi: torch.Tensor):
        force = self.action_values[action]
        x, x_dot, theta, theta_dot = (
            self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3],
        )
        m_p = xi[:, 0].clamp_min(1e-6)
        l = xi[:, 1].clamp_min(1e-6)
        total_mass = self.masscart + m_p
        polemass_length = m_p * l
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            l * (4.0 / 3.0 - m_p * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = torch.stack([x, x_dot, theta, theta_dot], dim=-1)
        self.steps += 1
        terminated = (
            (x < -self.x_threshold) | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
        )
        truncated = self.steps >= self.max_steps
        done = terminated | truncated
        reward = torch.where(terminated, torch.zeros_like(x), torch.ones_like(x))
        return self.state, reward, done, {"terminated": terminated, "truncated": truncated}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_episode_lengths(
    policy: PolicyNet,
    *,
    masspole: float,
    length: float,
    gravity: float = 9.8,
    masscart: float = 1.0,
    n_trials: int = 1000,
    max_steps: int = 500,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Run n_trials episodes and return an array of episode lengths.

    Episode length = number of environment transitions until termination
    (including the terminating transition), or max_steps if the pole never
    falls. This is the conventional CartPole episode length.
    """
    env = CartPoleEval(max_steps=max_steps, gravity=gravity, masscart=masscart, device=device)
    xi = torch.tensor([[masspole, length]], device=device).expand(n_trials, -1).float()

    env.reset(n_trials, seed=seed)
    alive = torch.ones(n_trials, device=device, dtype=torch.bool)
    lengths = torch.zeros(n_trials, device=device, dtype=torch.float32)

    for t in range(max_steps):
        obs = env.state
        logits = policy(obs)
        a = torch.argmax(logits, dim=-1)
        _, r, d, _ = env.step(a, xi)
        lengths += alive.float()
        alive = alive & (~d)
        if not torch.any(alive):
            break

    return lengths.cpu().numpy()


# ---------------------------------------------------------------------------
# Environment variants
# ---------------------------------------------------------------------------

@dataclass
class EnvVariant:
    name: str
    masspole: float
    length: float
    gravity: float = 9.8
    masscart: float = 1.0
    category: str = ""  # "easier" or "harder" or ""


# Default CartPole variants inspired by the competitor's table.
# "Original" uses the standard Gym defaults: masspole=0.1, length=0.5, g=9.8
# Easier variants make balancing trivially easy.
# Harder variants push outside the nominal training range.
DEFAULT_VARIANTS = [
    EnvVariant("Original",  masspole=0.1,  length=0.5,  gravity=9.8),
    EnvVariant("Light",     masspole=0.05, length=0.5,  gravity=9.8,  category="easier"),
    EnvVariant("Long",      masspole=0.1,  length=1.0,  gravity=9.8,  category="easier"),
    EnvVariant("Soft Gravity",  masspole=0.1,  length=0.5,  gravity=5.0,  category="easier"),
    EnvVariant("Heavy",     masspole=0.5,  length=0.5,  gravity=9.8,  category="harder"),
    EnvVariant("Short",     masspole=0.1,  length=0.25, gravity=9.8,  category="harder"),
    EnvVariant("Strong Gravity",masspole=0.1,  length=0.5,  gravity=15.0, category="harder"),
]


def load_variants_from_json(path: Path) -> list[EnvVariant]:
    data = json.loads(path.read_text())
    variants = []
    for d in data:
        variants.append(EnvVariant(
            name=d["name"],
            masspole=float(d["masspole"]),
            length=float(d["length"]),
            gravity=float(d.get("gravity", 9.8)),
            masscart=float(d.get("masscart", 1.0)),
            category=d.get("category", ""),
        ))
    return variants


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt_mean_se(lengths: np.ndarray) -> str:
    mean = np.mean(lengths)
    se = np.std(lengths, ddof=1) / np.sqrt(len(lengths))
    return f"{mean:.1f} ± {se:.1f}"


def _fmt_latex_cell(lengths: np.ndarray) -> str:
    mean = np.mean(lengths)
    se = np.std(lengths, ddof=1) / np.sqrt(len(lengths))
    return f"${mean:.1f} \\pm {se:.1f}$"


def _fmt_latex_mean_se(mean: float, se: float, *, bold: bool = False) -> str:
    body = f"{mean:.1f} \\pm {se:.1f}"
    return f"$\\mathbf{{{mean:.1f}}} \\pm {se:.1f}$" if bold else f"${body}$"


def print_text_table(
    results: dict[str, dict[str, np.ndarray]],
    variants: list[EnvVariant],
    columns: list[str],
) -> None:
    """Print a simple text table."""
    col_width = 18
    header = f"{'Environment':<16}" + "".join(f"{c:>{col_width}}" for c in columns)
    print(header)
    print("-" * len(header))

    prev_cat = ""
    for v in variants:
        if v.category and v.category != prev_cat:
            cat_label = f"{'Easier' if v.category == 'easier' else 'Harder'} environments"
            print(f"\n  {cat_label}")
            prev_cat = v.category
        row = f"{v.name:<16}"
        for col in columns:
            arr = results.get(col, {}).get(v.name)
            if arr is not None:
                row += f"{_fmt_mean_se(arr):>{col_width}}"
            else:
                row += f"{'—':>{col_width}}"
        print(row)


def generate_latex_table(
    results: dict[str, dict[str, np.ndarray]],
    variants: list[EnvVariant],
    columns: list[str],
    *,
    max_steps: int = 500,
    n_trials: int = 1000,
) -> str:
    """Generate LaTeX table source."""
    n_cols = len(columns)
    col_spec = "c|" + "c" * n_cols
    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{" + col_spec + "}")

    header = "Environment & " + " & ".join(columns) + r" \\"
    lines.append(header)
    lines.append(r"\hline \hline")

    prev_cat = ""
    for i, v in enumerate(variants):
        if v.category and v.category != prev_cat:
            if prev_cat:
                lines.append(r"\hline")
            cat_label = "Easier" if v.category == "easier" else "Harder"
            lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{c}{\textit{" + cat_label + r" environments}} \\")
            prev_cat = v.category

        cells = [v.name]
        for col in columns:
            arr = results.get(col, {}).get(v.name)
            if arr is not None:
                cells.append(_fmt_latex_cell(arr))
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Episode length over "
        + str(n_trials)
        + r" trials (mean $\pm$ standard error). Max episode length = "
        + str(max_steps)
        + r".}"
    )
    lines.append(r"\label{tab:eval_variants}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_table_method_rows(
    results: dict[str, dict[str, np.ndarray]],
    variants: list[EnvVariant],
    columns: list[str],
    *,
    max_steps: int = 500,
    n_trials: int = 1000,
    bold_best: bool = True,
) -> str:
    """Generate the paper-style table: methods as rows, variants as columns."""
    if len(variants) != 7:
        raise ValueError(
            "method-rows layout expects exactly 7 variants: Original, three easier, three harder."
        )

    variant_names = [v.name for v in variants]
    means: dict[str, dict[str, float]] = {}
    ses: dict[str, dict[str, float]] = {}
    for method in columns:
        means[method] = {}
        ses[method] = {}
        for v in variants:
            arr = results[method][v.name]
            means[method][v.name] = float(np.mean(arr))
            ses[method][v.name] = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    best_by_variant: dict[str, float] = {}
    if bold_best and len(columns) > 1:
        for v_name in variant_names:
            best_by_variant[v_name] = max(means[m][v_name] for m in columns)

    lines: list[str] = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"    \centering")
    lines.append(r"    \renewcommand{\arraystretch}{1.15}")
    lines.append(
        r"    \caption{Mean episode length $\pm$~standard error over "
        + str(n_trials)
        + r" deterministic rollouts. Max episode length = "
        + str(max_steps)
        + r".}"
    )
    lines.append(r"    \resizebox{\linewidth}{!}{")
    lines.append(r"        \begin{tabular}{l c ccc ccc}")
    lines.append(r"            \toprule")
    lines.append(
        r"            \multirow{2.5}{*}{Method} & \multicolumn{1}{c}{Original} "
        r"& \multicolumn{3}{c}{Easier Environments} "
        r"& \multicolumn{3}{c}{Harder Environments} \\"
    )
    lines.append(r"            \cmidrule(lr){2-2} \cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(
        r"            & Original & Light & Long & Soft Gravity "
        r"& Heavy & Short & Strong Gravity \\"
    )
    lines.append(r"            \midrule")

    ours_labels = {"MPA", "ICNN-DRO"}
    inserted_ours_rule = False
    for method in columns:
        if method in ours_labels and not inserted_ours_rule:
            lines.append(r"            \midrule")
            inserted_ours_rule = True
        row_cells = [method]
        for v_name in variant_names:
            mean = means[method][v_name]
            se = ses[method][v_name]
            is_best = (
                bold_best
                and len(columns) > 1
                and abs(mean - best_by_variant[v_name]) <= 1e-9
            )
            row_cells.append(_fmt_latex_mean_se(mean, se, bold=is_best))
        lines.append("            " + " & ".join(row_cells) + r" \\")

    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}")
    lines.append(r"    }")
    lines.append(r"    \label{tab:eval_lambda_sensitivity}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RL policies across CartPole variants and produce a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--regular", type=Path, default=None,
                        help="Checkpoint (.pt) for the 'Regular' (nominal, non-robust) policy.")
    parser.add_argument("--particle", type=Path, default=None,
                        help="Checkpoint (.pt) for the 'Particle' (particle-ascent robust) policy.")
    parser.add_argument("--robust", type=Path, default=None,
                        help="Checkpoint (.pt) for the 'Robust (ICNN)' (WDRO transport-map) policy.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=None,
                        help="Arbitrary checkpoint(s). Column names are inferred from filenames.")
    parser.add_argument("--column-names", type=str, nargs="+", default=None,
                        help="Column names for --checkpoints (must match count).")

    parser.add_argument("--trials", type=int, default=1000, help="Number of evaluation episodes per variant.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max episode length.")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--variants-json", type=Path, default=None,
                        help="JSON file defining custom variants (list of {name, masspole, length, gravity, category}).")

    parser.add_argument("--out-latex", type=Path, default=None,
                        help="Save LaTeX table to file.")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Save raw results to JSON.")
    parser.add_argument(
        "--layout",
        choices=["variant-rows", "method-rows"],
        default="variant-rows",
        help="LaTeX table orientation. Use method-rows for the paper-style table.",
    )
    parser.add_argument(
        "--no-bold-best",
        dest="bold_best",
        action="store_false",
        default=True,
        help="Do not bold the best mean in each variant column for method-rows layout.",
    )

    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build column list
    columns: list[str] = []
    policies: dict[str, PolicyNet] = {}
    checkpoint_meta: dict[str, dict[str, Any]] = {}

    if args.regular is not None:
        columns.append("Regular")
        policies["Regular"], checkpoint_meta["Regular"] = load_policy_checkpoint(args.regular, device)
    if args.particle is not None:
        columns.append("Particle")
        policies["Particle"], checkpoint_meta["Particle"] = load_policy_checkpoint(args.particle, device)
    if args.robust is not None:
        columns.append("Robust (ICNN)")
        policies["Robust (ICNN)"], checkpoint_meta["Robust (ICNN)"] = load_policy_checkpoint(args.robust, device)
    if args.checkpoints is not None:
        names = args.column_names or [p.stem for p in args.checkpoints]
        if len(names) != len(args.checkpoints):
            parser.error("--column-names count must match --checkpoints count.")
        for name, ckpt_path in zip(names, args.checkpoints):
            columns.append(name)
            policies[name], checkpoint_meta[name] = load_policy_checkpoint(ckpt_path, device)

    if not policies:
        parser.error("Provide at least one of --regular, --particle, --robust, or --checkpoints.")

    # Load variants
    if args.variants_json is not None:
        variants = load_variants_from_json(args.variants_json)
    else:
        variants = DEFAULT_VARIANTS

    print(f"Evaluating {len(policies)} policies across {len(variants)} variants "
          f"({args.trials} trials, max_steps={args.max_steps}, seed={args.seed})")
    for col in columns:
        meta = checkpoint_meta.get(col, {})
        ckpt_env = meta.get("env")
        ckpt_method = meta.get("method")
        ckpt_seed = meta.get("seed")
        ckpt_lam = meta.get("lam")
        if ckpt_env is not None and str(ckpt_env) != "cartpole":
            raise ValueError(f"Checkpoint for column {col!r} was trained on env={ckpt_env!r}, not cartpole.")
        print(
            f"  [ckpt] {col}: method={ckpt_method} seed={ckpt_seed} "
            f"lam={ckpt_lam}"
        )
    print()

    # Evaluate
    results: dict[str, dict[str, np.ndarray]] = {col: {} for col in columns}
    for v in variants:
        for col in columns:
            policy = policies[col]
            lengths = evaluate_episode_lengths(
                policy,
                masspole=v.masspole,
                length=v.length,
                gravity=v.gravity,
                masscart=v.masscart,
                n_trials=args.trials,
                max_steps=args.max_steps,
                seed=args.seed,
                device=device,
            )
            results[col][v.name] = lengths
            mean = np.mean(lengths)
            se = np.std(lengths, ddof=1) / np.sqrt(len(lengths))
            print(f"  {v.name:<16} {col:<12} {mean:7.1f} ± {se:.1f}")

    print()
    print_text_table(results, variants, columns)

    # LaTeX
    if args.layout == "method-rows":
        latex = generate_latex_table_method_rows(
            results, variants, columns,
            max_steps=args.max_steps, n_trials=args.trials,
            bold_best=bool(args.bold_best),
        )
    else:
        latex = generate_latex_table(
            results, variants, columns,
            max_steps=args.max_steps, n_trials=args.trials,
        )
    print(f"\n{'='*60}")
    print("LaTeX:")
    print(f"{'='*60}")
    print(latex)

    if args.out_latex is not None:
        args.out_latex.parent.mkdir(parents=True, exist_ok=True)
        args.out_latex.write_text(latex, encoding="utf-8")
        print(f"\nSaved LaTeX: {args.out_latex}")

    if args.out_json is not None:
        json_out: dict[str, Any] = {
            "config": {
                "trials": args.trials,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "layout": args.layout,
            },
            "checkpoints": {
                col: {
                    "method": checkpoint_meta.get(col, {}).get("method"),
                    "seed": checkpoint_meta.get(col, {}).get("seed"),
                    "lam": checkpoint_meta.get(col, {}).get("lam"),
                    "env": checkpoint_meta.get(col, {}).get("env"),
                    "iters": checkpoint_meta.get(col, {}).get("iters"),
                }
                for col in columns
            },
            "variants": [
                {"name": v.name, "masspole": v.masspole, "length": v.length,
                 "gravity": v.gravity, "masscart": v.masscart, "category": v.category}
                for v in variants
            ],
            "results": {},
        }
        for col in columns:
            json_out["results"][col] = {}
            for v in variants:
                arr = results[col][v.name]
                json_out["results"][col][v.name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)),
                    "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(arr),
                }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(json_out, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
