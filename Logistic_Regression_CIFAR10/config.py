"""Training configuration dataclass + CLI parser builder.

All hyperparameters of the original ``adversarial_multiclass_icnn.py`` are
preserved. Extra NPF options are added alongside the ICNN options.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from typing import List, Optional, Sequence, Tuple

from algorithms.registry import ALL_ALGORITHMS


@dataclass(frozen=True)
class TrainConfig:
    # I/O
    data_dir: str = "./data"
    out_dir: str = "./outputs_adversarial_multiclass_icnn"
    feature_cache: str = "cifar10_resnet50_features.pth"
    force_reextract: bool = False

    # Figure 6 core hyperparameters
    tau: float = 0.05
    lambda_param: float = 10.0  # derived: 1 / (2 * tau)
    eps_ent: Tuple[float, ...] = (0.2) # (0.2, 0.02, 0.002)
    epochs: int = 10
    inner_lr: float = 0.01
    inner_steps: int = 100
    num_samples: int = 16
    sample_level: int = 5

    # Optimization
    dro_lr: float = 5e-3
    saa_lr: float = 1e-3
    batch_size: int = 128
    feature_batch_size: int = 128
    num_workers: int = 2
    seed: int = 2022

    # Evaluation (PGD)
    pgd_steps: int = 10
    pgd_restarts: int = 5
    perturbation_levels: Tuple[float, ...] = (
        0.0, 0.008, 0.016, 0.024, 0.032, 0.04, 0.048, 0.056, 0.064, 0.072, 0.08,
    )

    # Method selection
    methods: Tuple[str, ...] = ALL_ALGORITHMS

    # PPA knobs
    ppa_num_rounds: int = 5
    ppa_min_rounds: int = 2
    ppa_refine_steps: int = 15
    ppa_refine_lr: float = 5e-3
    ppa_delta_rtol: float = 1e-4

    # RO knobs (Madry-style l2-PGD robust optimisation in feature space)
    ro_epsilon_level: float = 0.04
    ro_epsilon: Optional[float] = None
    ro_pgd_steps: int = 10
    ro_pgd_step_size: Optional[float] = None
    ro_pgd_restarts: int = 1
    ro_lr: float = 5e-3

    # ICNN knobs
    icnn_hidden: Tuple[int, ...] = (1024, 512, 512, 256, 128, 64)
    icnn_strong_convexity: float = 1.0
    icnn_softplus_beta: float = 20.0
    icnn_lr_B: float = 5e-4
    icnn_weight_decay_B: float = 0.0
    icnn_init_mode: str = "identity"
    icnn_omega_steps_per_batch: int = 10
    icnn_bb_alpha0: float = 5e-4
    icnn_bb_alpha_min: float = 1e-6
    icnn_bb_alpha_max: float = 1.0
    icnn_bb_ls_c: float = 0.1
    icnn_bb_ls_shrink: float = 0.5
    icnn_bb_ls_max_steps: int = 10

    # NPF knobs tuned for CIFAR-10 ResNet-50 features. This profile keeps the
    # identity-start transport stable, gives the convex potential enough
    # low-rank curvature to move feature directions, and uses a less brittle
    # Armijo condition than the initial conservative setting.
    npf_hidden: Tuple[int, ...] = (512, 512, 256, 128, 64)
    npf_outer_rank: int = 8
    npf_inner_rank: int = 2
    npf_activation: str = "softplus"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 10.0
    npf_init_eps: float = 1e-4
    inner_steps_npf: int = 20
    inner_lr_npf: float = 1e-2  # deprecated (Adam) — kept for CLI back-compat
    npf_lr_B: float = 1e-3
    npf_weight_decay_B: float = 1e-4
    npf_bb_alpha0: float = 2e-4
    npf_bb_alpha_min: float = 1e-7
    npf_bb_alpha_max: float = 0.25
    npf_bb_ls_c: float = 1e-4
    npf_bb_ls_shrink: float = 0.5
    npf_bb_ls_max_steps: int = 15

    # NPF variant: no hidden-layer quadratic injections and one trainable
    # rank-0 diagonal quadratic form at the output layer only.
    npf_lastquad_hidden: Tuple[int, ...] = (512, 512, 256, 128, 64)
    npf_lastquad_activation: str = "softplus"
    npf_lastquad_elu_alpha: float = 1.0
    npf_lastquad_softplus_beta: float = 10.0
    npf_lastquad_init_eps: float = 1e-4
    inner_steps_npf_lastquad: int = 20
    # Deprecated Adam lr, kept for CLI back-compat.
    inner_lr_npf_lastquad: float = 1e-2
    npf_lastquad_lr_B: float = 1e-3
    npf_lastquad_weight_decay_B: float = 1e-4
    npf_lastquad_bb_alpha0: float = 2e-4
    npf_lastquad_bb_alpha_min: float = 1e-7
    npf_lastquad_bb_alpha_max: float = 0.25
    npf_lastquad_bb_ls_c: float = 1e-4
    npf_lastquad_bb_ls_shrink: float = 0.5
    npf_lastquad_bb_ls_max_steps: int = 15

    # NN-DRO knobs (vanilla-MLP adversary, no gradient-of-potential)
    nn_dro_hidden: Tuple[int, ...] = (512, 512, 256, 256, 128)
    nn_dro_activation: str = "relu"
    nn_dro_softplus_beta: float = 20.0
    nn_dro_init_scale: float = 1e-3
    nn_dro_omega_steps_per_batch: int = 10
    nn_dro_inner_lr: float = 1e-2
    nn_dro_lr_B: float = 5e-4
    nn_dro_weight_decay_B: float = 0.0

    # RGO vectorised rejection-sampling trials cap
    rgo_vectorized_max_trials: int = 100


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI exposing every TrainConfig hyperparameter."""
    parser = argparse.ArgumentParser(
        description=(
            "Figure 6 adversarial multiclass logistic regression on CIFAR-10 "
            "(ResNet-50 features). Competitors: SAA, Dual, WRM, RO, WGF, WFR, "
            "SVG, RGO, ICNN, NPF, NPF-LastQuad, NN-DRO, PPA."
        )
    )

    # I/O
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Where CIFAR-10 is/will be downloaded.")
    parser.add_argument("--out_dir", type=str, default="./outputs_adversarial_multiclass_icnn",
                        help="Output directory.")
    parser.add_argument("--feature_cache", type=str, default="cifar10_resnet50_features.pth",
                        help="Feature cache filename (saved under out_dir unless absolute).")
    parser.add_argument("--force_reextract", action="store_true",
                        help="Re-extract CIFAR-10 features even if cache exists.")

    # Figure 6 core hyperparameters
    parser.add_argument("--tau", type=float, default=0.05,
                        help="Figure 6 tau (we use lambda = 1 / (2 tau)).")
    parser.add_argument("--eps_ent", type=float, nargs="+",
                        default=[0.2, 0.02, 0.002],
                        help="Figure 6 entropy regularisation eps values.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Figure 6 training epochs.")
    parser.add_argument("--inner_lr", type=float, default=0.01,
                        help="Inner-loop stepsize for methods with an inner loop.")
    parser.add_argument("--inner_steps", type=int, default=100,
                        help="Inner-loop iterations for methods with an inner loop.")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of particles per data point for sampler methods.")
    parser.add_argument("--sample_level", type=int, default=5,
                        help="Dual SDRO randomised truncation level.")

    # Optimization
    parser.add_argument("--dro_lr", type=float, default=5e-3,
                        help="Outer learning rate for DRO baselines (reference code uses 5e-3).")
    parser.add_argument("--saa_lr", type=float, default=1e-3, help="ERM/SAA learning rate.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training DRO + SAA.")
    parser.add_argument("--feature_batch_size", type=int, default=256,
                        help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Dataloader workers (feature extraction + eval).")
    parser.add_argument("--seed", type=int, default=2022)

    # Evaluation
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--pgd_restarts", type=int, default=5,
                        help="Number of restarts for adaptive l2-PGD evaluation.")
    parser.add_argument(
        "--perturbation_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.008, 0.016, 0.024, 0.032, 0.04, 0.048, 0.056, 0.064, 0.072, 0.08],
        help="Normalized perturbation Δ grid (Figure 6 uses 0..0.08).",
    )

    # Method selection
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(ALL_ALGORITHMS),
        help=f"Which methods to run. Choices: {' '.join(ALL_ALGORITHMS)}",
    )

    # PPA knobs
    ppa = parser.add_argument_group("ppa")
    ppa.add_argument("--ppa_num_rounds", type=int, default=5,
                     help="PPA: total rounds including round 0.")
    ppa.add_argument("--ppa_min_rounds", type=int, default=2,
                     help="PPA: minimum refinement rounds before early stopping.")
    ppa.add_argument("--ppa_refine_steps", type=int, default=15,
                     help="PPA: constant-lr ascent steps per refinement round.")
    ppa.add_argument("--ppa_refine_lr", type=float, default=5e-3,
                     help="PPA: constant lr for refinement rounds.")
    ppa.add_argument("--ppa_delta_rtol", type=float, default=1e-4,
                     help="PPA: relative Δ threshold for adaptive stopping.")

    # RO knobs
    ro = parser.add_argument_group("ro")
    ro.add_argument("--ro_epsilon_level", type=float, default=0.04,
                    help=(
                        "RO training radius as a normalized feature perturbation "
                        "level. The script converts it to --ro_epsilon_level * "
                        "mean_train_feature_norm unless --ro_epsilon is set."
                    ))
    ro.add_argument("--ro_epsilon", type=float, default=None,
                    help="Absolute l2 feature-space radius for RO training.")
    ro.add_argument("--ro_pgd_steps", type=int, default=10,
                    help="RO inner l2-PGD ascent steps per batch.")
    ro.add_argument("--ro_pgd_step_size", type=float, default=None,
                    help="RO inner l2-PGD step size. Default: ro_epsilon / max(1, ro_pgd_steps // 2).")
    ro.add_argument("--ro_pgd_restarts", type=int, default=1,
                    help="RO inner l2-PGD random restarts; first restart starts at x.")
    ro.add_argument("--ro_lr", type=float, default=5e-3,
                    help="RO outer learning rate.")

    # ICNN knobs
    icnn = parser.add_argument_group("icnn")
    icnn.add_argument("--icnn_hidden", type=int, nargs="+",
                      default=[1024, 512, 512, 256, 128, 64])
    icnn.add_argument("--icnn_strong_convexity", type=float, default=1.0)
    icnn.add_argument("--icnn_softplus_beta", type=float, default=20.0)
    icnn.add_argument("--icnn_lr_B", type=float, default=5e-4)
    icnn.add_argument("--icnn_weight_decay_B", type=float, default=0.0)
    icnn.add_argument("--icnn_init_mode", type=str, default="identity",
                      choices=["identity", "principled"],
                      help="ICNN init for ψ_ω: 'identity' makes T_ω(x)=x at start.")
    icnn.add_argument("--icnn_omega_steps_per_batch", type=int, default=10)
    icnn.add_argument("--icnn_bb_alpha0", type=float, default=5e-4)
    icnn.add_argument("--icnn_bb_alpha_min", type=float, default=1e-6)
    icnn.add_argument("--icnn_bb_alpha_max", type=float, default=1.0)
    icnn.add_argument("--icnn_bb_ls_c", type=float, default=0.1)
    icnn.add_argument("--icnn_bb_ls_shrink", type=float, default=0.5)
    icnn.add_argument("--icnn_bb_ls_max_steps", type=int, default=10)

    # NPF knobs
    npf = parser.add_argument_group("npf")
    npf.add_argument("--npf_hidden", type=int, nargs="+",
                     default=[512, 512, 256, 128, 64])
    npf.add_argument("--npf_outer_rank", type=int, default=8)
    npf.add_argument("--npf_inner_rank", type=int, default=2)
    npf.add_argument("--npf_activation", type=str, default="softplus",
                     choices=["elu", "softplus", "relu"])
    npf.add_argument("--npf_elu_alpha", type=float, default=1.0)
    npf.add_argument("--npf_softplus_beta", type=float, default=10.0)
    npf.add_argument("--npf_init_eps", type=float, default=1e-4)
    npf.add_argument("--inner_steps_npf", type=int, default=20)
    npf.add_argument("--inner_lr_npf", type=float, default=1e-2,
                     help="Deprecated: Adam lr is no longer used (now BB+Armijo).")
    npf.add_argument("--npf_lr_B", type=float, default=1e-3)
    npf.add_argument("--npf_weight_decay_B", type=float, default=1e-4)
    npf.add_argument("--npf_bb_alpha0", type=float, default=2e-4,
                     help="NPF BB+Armijo initial step size.")
    npf.add_argument("--npf_bb_alpha_min", type=float, default=1e-7)
    npf.add_argument("--npf_bb_alpha_max", type=float, default=0.25)
    npf.add_argument("--npf_bb_ls_c", type=float, default=1e-4)
    npf.add_argument("--npf_bb_ls_shrink", type=float, default=0.5)
    npf.add_argument("--npf_bb_ls_max_steps", type=int, default=15)

    # NPF last-layer diagonal quadratic variant knobs
    npf_lq = parser.add_argument_group("npf_lastquad")
    npf_lq.add_argument("--npf_lastquad_hidden", type=int, nargs="+",
                        default=[512, 512, 256, 128, 64])
    npf_lq.add_argument("--npf_lastquad_activation", type=str, default="softplus",
                        choices=["elu", "softplus", "relu"])
    npf_lq.add_argument("--npf_lastquad_elu_alpha", type=float, default=1.0)
    npf_lq.add_argument("--npf_lastquad_softplus_beta", type=float, default=10.0)
    npf_lq.add_argument("--npf_lastquad_init_eps", type=float, default=1e-4)
    npf_lq.add_argument("--inner_steps_npf_lastquad", type=int, default=20)
    npf_lq.add_argument("--inner_lr_npf_lastquad", type=float, default=1e-2,
                        help="Deprecated: Adam lr is no longer used (now BB+Armijo).")
    npf_lq.add_argument("--npf_lastquad_lr_B", type=float, default=1e-3)
    npf_lq.add_argument("--npf_lastquad_weight_decay_B", type=float, default=1e-4)
    npf_lq.add_argument("--npf_lastquad_bb_alpha0", type=float, default=2e-4,
                        help="NPF-LastQuad BB+Armijo initial step size.")
    npf_lq.add_argument("--npf_lastquad_bb_alpha_min", type=float, default=1e-7)
    npf_lq.add_argument("--npf_lastquad_bb_alpha_max", type=float, default=0.25)
    npf_lq.add_argument("--npf_lastquad_bb_ls_c", type=float, default=1e-4)
    npf_lq.add_argument("--npf_lastquad_bb_ls_shrink", type=float, default=0.5)
    npf_lq.add_argument("--npf_lastquad_bb_ls_max_steps", type=int, default=15)

    # NN-DRO knobs
    nn_dro = parser.add_argument_group("nn_dro")
    nn_dro.add_argument("--nn_dro_hidden", type=int, nargs="+",
                        default=[512, 512, 256])
    nn_dro.add_argument("--nn_dro_activation", type=str, default="relu",
                        choices=["relu", "elu", "tanh", "softplus"])
    nn_dro.add_argument("--nn_dro_softplus_beta", type=float, default=20.0)
    nn_dro.add_argument("--nn_dro_init_scale", type=float, default=1e-3,
                        help="Init scale for MLP final layer (near-identity start).")
    nn_dro.add_argument("--nn_dro_omega_steps_per_batch", type=int, default=10)
    nn_dro.add_argument("--nn_dro_inner_lr", type=float, default=1e-2)
    nn_dro.add_argument("--nn_dro_lr_B", type=float, default=5e-4)
    nn_dro.add_argument("--nn_dro_weight_decay_B", type=float, default=0.0)

    # RGO knob
    rgo = parser.add_argument_group("rgo")
    rgo.add_argument("--rgo_vectorized_max_trials", type=int, default=100)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Project CLI args into a TrainConfig; also set derived fields."""
    lambda_param = 1.0 / (2.0 * float(args.tau))

    kwargs = {}
    field_names = {f.name for f in fields(TrainConfig)}
    for name in field_names:
        if name == "lambda_param":
            kwargs[name] = lambda_param
            continue
        if name == "eps_ent":
            kwargs[name] = tuple(args.eps_ent)
            continue
        if name == "perturbation_levels":
            kwargs[name] = tuple(args.perturbation_levels)
            continue
        if name == "methods":
            kwargs[name] = tuple(args.methods)
            continue
        if name == "icnn_hidden":
            kwargs[name] = tuple(args.icnn_hidden)
            continue
        if name == "npf_hidden":
            kwargs[name] = tuple(args.npf_hidden)
            continue
        if name == "npf_lastquad_hidden":
            kwargs[name] = tuple(args.npf_lastquad_hidden)
            continue
        if name == "nn_dro_hidden":
            kwargs[name] = tuple(args.nn_dro_hidden)
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    return TrainConfig(**kwargs)
