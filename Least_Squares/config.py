"""Training configuration dataclass + CLI parser for the ULS suite."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Sequence


@dataclass(frozen=True)
class ULSConfig:
    # Problem sizes
    dim_m: int = 10
    dim_n: int = 10
    n_train: int = 10
    n_test: int = 1000

    # Delta sweep used for evaluation
    delta_min: float = 0.0
    delta_max: float = 10.0
    delta_steps: int = 50

    # Outer training hyperparams
    epochs: int = 10
    lr_theta: float = 1e-2
    lam: float = 0.1
    epsilon: float = 0.1
    grad_clip: float = 100.0

    # Inner-loop hyperparams (PA / WGF / WFR / Dual)
    inner_steps: int = 3000
    inner_step_size: float = 1e-4
    m_particles: int = 8

    # WFR specifics
    wfr_weight_step_size: float = 0.0016
    wfr_low_weight_threshold: float = 1e-3

    # Dual (Sinkhorn / MLMC)
    sinkhorn_sample_level: int = 4

    # ICNN (Uncertain_LS_icnn baseline)
    icnn_hidden: Sequence[int] = (512, 512, 512, 256, 256, 128, 128, 64)
    icnn_strong_convexity: float = 1.0
    icnn_softplus_beta: float = 20.0
    icnn_bb_alpha0: float = 1e-1
    icnn_bb_alpha_min: float = 1e-6
    icnn_bb_alpha_max: float = 10.0
    icnn_bb_ls_c: float = 1e-4
    icnn_bb_ls_shrink: float = 0.5
    icnn_bb_ls_max_steps: int = 10
    icnn_omega_steps_per_epoch: int = 50

    # Madry PGD
    pgd_epsilon: float = 0.316
    pgd_steps: int = 50
    pgd_restarts: int = 5

    # PPA (1D Brenier projection + refinement)
    ppa_num_rounds: int = 5
    ppa_min_rounds: int = 2
    ppa_refine_steps: int = 500
    ppa_refine_lr: float = 1e-4
    ppa_delta_rtol: float = 1e-4

    # NPF-ICNN adversary (Vesseron & Cuturi, 2024)
    # BB+Armijo defaults mirror icnn_bb_* per "same parameters" spec.
    npf_hidden: Sequence[int] = (64, 64, 64, 64)
    npf_outer_rank: int = 1
    npf_inner_rank: int = 1
    npf_activation: str = "elu"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0
    npf_init_eps: float = 1e-3
    npf_omega_steps_per_epoch: int = 500
    inner_lr_npf: float = 1e-2  # deprecated (Adam) — kept for CLI back-compat
    npf_bb_alpha0: float = 1e-1
    npf_bb_alpha_min: float = 1e-6
    npf_bb_alpha_max: float = 10.0
    npf_bb_ls_c: float = 1e-4
    npf_bb_ls_shrink: float = 0.5
    npf_bb_ls_max_steps: int = 10

    # NN-DRO (vanilla MLP adversary)
    nn_dro_hidden: Sequence[int] = (64, 64, 64, 64)
    nn_dro_activation: str = "softplus"
    nn_dro_softplus_beta: float = 20.0
    nn_dro_init_scale: float = 1e-3
    nn_dro_omega_steps_per_epoch: int = 500
    nn_dro_inner_lr: float = 1e-2

    # Repro / numerics
    seed: int = 219
    ridge: float = 1e-6


ALL_ALGORITHMS = (
    "erm",
    "particle_ascent",
    "wgf",
    "wfr",
    "dual",
    "icnn",
    "madry",
    "ppa",
    "npf",
    "nn_dro",
)


def _int_tuple(s: str) -> Sequence[int]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Uncertain least squares (Fig. 8) comparison suite: "
            "ERM / Particle Ascent / WGF / WFR / Dual / ICNN / Madry PGD / PPA / NPF / NN-DRO"
        )
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help=f"Algorithms to run. Choices: all, {', '.join(ALL_ALGORITHMS)}",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the delta-sweep test-loss evaluation stage.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to write per-method JSON results (default: script directory).",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="After evaluation, save the Fig. 8 PDFs.",
    )

    core = parser.add_argument_group("core")
    core.add_argument("--seed", type=int, default=219)
    core.add_argument("--epochs", type=int, default=10)
    core.add_argument("--n-train", type=int, default=10, dest="n_train")
    core.add_argument("--n-test", type=int, default=1000, dest="n_test")
    core.add_argument("--dim-m", type=int, default=10, dest="dim_m")
    core.add_argument("--dim-n", type=int, default=10, dest="dim_n")
    core.add_argument("--lam", type=float, default=0.1)
    core.add_argument("--epsilon", type=float, default=0.1)
    core.add_argument("--lr-theta", type=float, default=1e-2, dest="lr_theta")
    core.add_argument("--grad-clip", type=float, default=100.0, dest="grad_clip")
    core.add_argument("--delta-min", type=float, default=0.0, dest="delta_min")
    core.add_argument("--delta-max", type=float, default=10.0, dest="delta_max")
    core.add_argument("--delta-steps", type=int, default=50, dest="delta_steps")

    inner = parser.add_argument_group("inner-loop (PA / WGF / WFR)")
    inner.add_argument("--inner-steps", type=int, default=3000, dest="inner_steps")
    inner.add_argument("--inner-step-size", type=float, default=1e-4, dest="inner_step_size")
    inner.add_argument("--m-particles", type=int, default=8, dest="m_particles")

    icnn = parser.add_argument_group("icnn")
    icnn.add_argument(
        "--icnn-hidden", type=_int_tuple,
        default=(512, 512, 512, 256, 256, 128, 128, 64),
        dest="icnn_hidden",
        help="Comma-separated widths",
    )
    icnn.add_argument("--icnn-strong-convexity", type=float, default=1.0, dest="icnn_strong_convexity")
    icnn.add_argument("--icnn-softplus-beta", type=float, default=20.0, dest="icnn_softplus_beta")
    icnn.add_argument("--icnn-omega-steps-per-epoch", type=int, default=50, dest="icnn_omega_steps_per_epoch")

    pgd = parser.add_argument_group("madry")
    pgd.add_argument("--pgd-epsilon", type=float, default=0.316, dest="pgd_epsilon")
    pgd.add_argument("--pgd-steps", type=int, default=50, dest="pgd_steps")
    pgd.add_argument("--pgd-restarts", type=int, default=5, dest="pgd_restarts")

    ppa = parser.add_argument_group("ppa")
    ppa.add_argument("--ppa-num-rounds", type=int, default=5, dest="ppa_num_rounds")
    ppa.add_argument("--ppa-min-rounds", type=int, default=2, dest="ppa_min_rounds")
    ppa.add_argument("--ppa-refine-steps", type=int, default=500, dest="ppa_refine_steps")
    ppa.add_argument("--ppa-refine-lr", type=float, default=1e-4, dest="ppa_refine_lr")
    ppa.add_argument("--ppa-delta-rtol", type=float, default=1e-4, dest="ppa_delta_rtol")

    npf = parser.add_argument_group("npf")
    npf.add_argument("--npf-hidden", type=_int_tuple, default=(64, 64, 64, 64), dest="npf_hidden")
    npf.add_argument("--npf-outer-rank", type=int, default=1, dest="npf_outer_rank")
    npf.add_argument("--npf-inner-rank", type=int, default=1, dest="npf_inner_rank")
    npf.add_argument(
        "--npf-activation", choices=["elu", "softplus", "relu"],
        default="elu", dest="npf_activation",
    )
    npf.add_argument("--npf-elu-alpha", type=float, default=1.0, dest="npf_elu_alpha")
    npf.add_argument("--npf-softplus-beta", type=float, default=20.0, dest="npf_softplus_beta")
    npf.add_argument("--npf-init-eps", type=float, default=1e-3, dest="npf_init_eps")
    npf.add_argument("--npf-omega-steps-per-epoch", type=int, default=500, dest="npf_omega_steps_per_epoch")
    npf.add_argument("--inner-lr-npf", type=float, default=1e-2, dest="inner_lr_npf",
                     help="Deprecated: NPF now uses BB+Armijo, not Adam.")
    npf.add_argument("--npf-bb-alpha0", type=float, default=1e-1, dest="npf_bb_alpha0",
                     help="NPF BB+Armijo alpha0 (default matches --icnn-bb-alpha0 conceptually).")
    npf.add_argument("--npf-bb-alpha-min", type=float, default=1e-6, dest="npf_bb_alpha_min")
    npf.add_argument("--npf-bb-alpha-max", type=float, default=10.0, dest="npf_bb_alpha_max")
    npf.add_argument("--npf-bb-ls-c", type=float, default=1e-4, dest="npf_bb_ls_c")
    npf.add_argument("--npf-bb-ls-shrink", type=float, default=0.5, dest="npf_bb_ls_shrink")
    npf.add_argument("--npf-bb-ls-max-steps", type=int, default=10, dest="npf_bb_ls_max_steps")

    nn_dro = parser.add_argument_group("nn_dro")
    nn_dro.add_argument("--nn-dro-hidden", type=_int_tuple, default=(64, 64, 64, 64), dest="nn_dro_hidden")
    nn_dro.add_argument(
        "--nn-dro-activation", choices=["relu", "elu", "tanh", "softplus"],
        default="softplus", dest="nn_dro_activation",
    )
    nn_dro.add_argument("--nn-dro-softplus-beta", type=float, default=20.0, dest="nn_dro_softplus_beta")
    nn_dro.add_argument("--nn-dro-init-scale", type=float, default=1e-3, dest="nn_dro_init_scale")
    nn_dro.add_argument("--nn-dro-omega-steps-per-epoch", type=int, default=500, dest="nn_dro_omega_steps_per_epoch")
    nn_dro.add_argument("--nn-dro-inner-lr", type=float, default=1e-2, dest="nn_dro_inner_lr")

    return parser


def config_from_args(args: argparse.Namespace) -> ULSConfig:
    field_names = {f.name for f in fields(ULSConfig)}
    kwargs = {}
    for name in field_names:
        if hasattr(args, name):
            val = getattr(args, name)
            if isinstance(val, list):
                val = tuple(val)
            kwargs[name] = val
    return ULSConfig(**kwargs)
