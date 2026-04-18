"""Training configuration dataclasses + CLI parser builder."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Optional, Sequence


@dataclass(frozen=True)
class TrainConfig:
    # Core training
    batch_size: int = 256
    num_epochs: int = 3
    lr_cls: float = 1e-3
    lr_cls_drop_epoch: int = 30
    lr_cls_drop_factor: float = 0.1
    lambda_reg: float = 100
    epoch_clean: int = 0

    # Per-algorithm step caps (None = no cap)
    max_steps_algo1: Optional[int] = None
    max_steps_algo2: Optional[int] = None
    max_steps_ppa: Optional[int] = None
    max_steps_new_ppa: Optional[int] = None
    max_steps_npf: Optional[int] = None
    max_steps_saa: Optional[int] = None
    max_steps_dual: Optional[int] = None
    max_steps_wgf: Optional[int] = None
    max_steps_wfr: Optional[int] = None
    max_steps_svg: Optional[int] = None
    max_steps_rgo: Optional[int] = None

    # Algo1 / Algo2 inner loops
    use_margin_adv_algo1: bool = False
    use_margin_adv_algo2: bool = False
    inner_steps_algo1: int = 5
    inner_steps_algo2: int = 5
    inner_lr_algo1: float = 1e-2
    bb_alpha0_icnn: float = 5e-4
    icnn_hidden_sizes: Sequence[int] = (64, 64, 64, 64)

    # PPA
    inner_steps_ppa_round0: int = 75
    inner_lr_ppa_round0: float = 1e-2
    ppa_num_rounds: int = 5
    ppa_min_rounds: int = 2
    ppa_refine_steps: int = 15
    ppa_refine_lr: float = 5e-3
    ppa_delta_rtol: float = 1e-4

    # New_PPA
    inner_steps_new_ppa_round0: int = 75
    inner_lr_new_ppa_round0: float = 1e-2
    new_ppa_num_rounds: int = 5
    new_ppa_min_rounds: int = 2
    new_ppa_refine_steps: int = 15
    new_ppa_refine_lr: float = 5e-3
    new_ppa_gain_rtol: float = 1e-4

    # Sampling-based DRO baselines
    dual_epsilon: float = 5e-2
    dual_sample_level: int = 5
    particle_epsilon: float = 5e-2
    particle_num_samples: int = 8
    particle_inner_steps: int = 30
    particle_inner_lr: float = 5e-3
    svg_adagrad_hist_decay: float = 0.9
    rgo_epsilon: float = 5e-2
    rgo_num_samples: int = 8
    rgo_inner_steps: int = 20
    rgo_inner_lr: float = 5e-3
    rgo_vectorized_max_trials: int = 100

    # Diagnostics + seed
    cm_diagnostics: bool = False
    seed: int = 0

    # Checkpoint-selection / early stopping
    es_val_frac: float = 0.0833
    es_patience: int = 0
    es_pgd_eps: float = 1.5
    es_pgd_steps: int = 20
    es_pgd_restarts: int = 5
    es_clean_weight: float = 0.0

    # NPF-style ICNN
    npf_hidden_sizes: Sequence[int] = (512, 512, 256, 256, 128)
    npf_outer_rank: int = 4
    npf_inner_rank: int = 1
    npf_activation: str = "elu"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0
    npf_init_eps: float = 1e-3
    inner_steps_npf: int = 20
    inner_lr_npf: float = 1e-2
    use_margin_adv_npf: bool = False


@dataclass(frozen=True)
class PGDEvalConfig:
    epsilons: Sequence[float] = (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
    num_steps: int = 40
    step_size: Optional[float] = None
    restarts: int = 5


ALL_ALGORITHMS = (
    "saa",
    "algo1",
    "algo2",
    "npf",
    "ppa",
    "new_ppa",
    "dual",
    "wgf",
    "wfr",
    "svg",
    "rgo",
)


def _int_tuple(s: str) -> Sequence[int]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _float_tuple(s: str) -> Sequence[float]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the whole MNIST_Cuturi suite.

    Every TrainConfig field is exposed as ``--<field>``; defaults are kept in
    sync with the TrainConfig dataclass.
    """
    parser = argparse.ArgumentParser(
        description=(
            "MNIST comparison suite: "
            "SAA / WRM / ICNN / NPF / PPA / New_PPA / Dual / WGF / WFR / SVG / RGO"
        )
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["all"],
        help=f"Algorithms to train. Choices: all, {', '.join(ALL_ALGORITHMS)}",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the standardised clean / PGD evaluation stage.",
    )
    parser.add_argument(
        "--cm-diagnostics",
        action="store_true",
        help="Enable cyclical monotonicity diagnostics for Algorithm 1.",
    )

    # Core
    core = parser.add_argument_group("core")
    core.add_argument("--batch-size", type=int, default=128)
    core.add_argument("--num-epochs", type=int, default=50)
    core.add_argument("--lr-cls", type=float, default=1e-2)
    core.add_argument("--lr-cls-drop-epoch", type=int, default=30)
    core.add_argument("--lr-cls-drop-factor", type=float, default=0.1)
    core.add_argument("--lambda-reg", type=float, default=0.01)
    core.add_argument("--epoch-clean", type=int, default=0)
    core.add_argument("--seed", type=int, default=0)

    # Max steps per algorithm
    caps = parser.add_argument_group("max-steps")
    for key in ALL_ALGORITHMS:
        caps.add_argument(f"--max-steps-{key.replace('_', '-')}", type=int, default=None)

    # Algo1 (Duchi) / Algo2 (my ICNN)
    algo = parser.add_argument_group("algo1 / algo2")
    algo.add_argument("--use-margin-adv-algo1", action="store_true")
    algo.add_argument("--use-margin-adv-algo2", action="store_true")
    algo.add_argument("--inner-steps-algo1", type=int, default=75)
    algo.add_argument("--inner-steps-algo2", type=int, default=20)
    algo.add_argument("--inner-lr-algo1", type=float, default=1e-2)
    algo.add_argument("--bb-alpha0-icnn", type=float, default=1e-2)
    algo.add_argument(
        "--icnn-hidden-sizes",
        type=_int_tuple,
        default=(512, 512, 256, 256, 128),
        help="Comma-separated list, e.g. 512,512,256,256,128",
    )

    # PPA / New_PPA (in paper we just have new_PPA which is MPA)
    ppa = parser.add_argument_group("ppa / new_ppa")
    ppa.add_argument("--inner-steps-ppa-round0", type=int, default=75)
    ppa.add_argument("--inner-lr-ppa-round0", type=float, default=1e-2)
    ppa.add_argument("--ppa-num-rounds", type=int, default=5)
    ppa.add_argument("--ppa-min-rounds", type=int, default=2)
    ppa.add_argument("--ppa-refine-steps", type=int, default=15)
    ppa.add_argument("--ppa-refine-lr", type=float, default=5e-3)
    ppa.add_argument("--ppa-delta-rtol", type=float, default=1e-4)
    ppa.add_argument("--inner-steps-new-ppa-round0", type=int, default=75)
    ppa.add_argument("--inner-lr-new-ppa-round0", type=float, default=1e-2)
    ppa.add_argument("--new-ppa-num-rounds", type=int, default=5)
    ppa.add_argument("--new-ppa-min-rounds", type=int, default=2)
    ppa.add_argument("--new-ppa-refine-steps", type=int, default=15)
    ppa.add_argument("--new-ppa-refine-lr", type=float, default=5e-3)
    ppa.add_argument("--new-ppa-gain-rtol", type=float, default=1e-4)

    # NPF
    npf = parser.add_argument_group("npf")
    npf.add_argument("--npf-hidden-sizes", type=_int_tuple, default=(512, 512, 256, 256, 128))
    npf.add_argument("--npf-outer-rank", type=int, default=4)
    npf.add_argument("--npf-inner-rank", type=int, default=1)
    npf.add_argument("--npf-activation", type=str, default="elu", choices=["elu", "softplus", "relu"])
    npf.add_argument("--npf-elu-alpha", type=float, default=1.0)
    npf.add_argument("--npf-softplus-beta", type=float, default=20.0)
    npf.add_argument("--npf-init-eps", type=float, default=1e-3)
    npf.add_argument("--inner-steps-npf", type=int, default=20)
    npf.add_argument("--inner-lr-npf", type=float, default=1e-2)
    npf.add_argument("--use-margin-adv-npf", action="store_true")

    # Sampling-based baselines (I got it from JJ paper)
    base = parser.add_argument_group("dual / particle / rgo")
    base.add_argument("--dual-epsilon", type=float, default=5e-4)
    base.add_argument("--dual-sample-level", type=int, default=5)
    base.add_argument("--particle-epsilon", type=float, default=5e-4)
    base.add_argument("--particle-num-samples", type=int, default=8)
    base.add_argument("--particle-inner-steps", type=int, default=75)
    base.add_argument("--particle-inner-lr", type=float, default=5e-3)
    base.add_argument("--svg-adagrad-hist-decay", type=float, default=0.9)
    base.add_argument("--rgo-epsilon", type=float, default=5e-4)
    base.add_argument("--rgo-num-samples", type=int, default=8)
    base.add_argument("--rgo-inner-steps", type=int, default=75)
    base.add_argument("--rgo-inner-lr", type=float, default=5e-3)
    base.add_argument("--rgo-vectorized-max-trials", type=int, default=100)

    # Early stopping
    es = parser.add_argument_group("early-stopping")
    es.add_argument("--es-val-frac", type=float, default=0.0833)
    es.add_argument("--es-patience", type=int, default=0)
    es.add_argument("--es-pgd-eps", type=float, default=1.5)
    es.add_argument("--es-pgd-steps", type=int, default=20)
    es.add_argument("--es-pgd-restarts", type=int, default=5)
    es.add_argument("--es-clean-weight", type=float, default=0.0)

    # Evaluation
    ev = parser.add_argument_group("eval")
    ev.add_argument(
        "--eval-epsilons",
        type=_float_tuple,
        default=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0),
        help="PGD-L2 sweep epsilons (comma-separated)",
    )
    ev.add_argument("--eval-num-steps", type=int, default=40)
    ev.add_argument("--eval-restarts", type=int, default=5)
    ev.add_argument("--eval-fixed-eps", type=float, default=2.0)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Project CLI args into a TrainConfig dataclass."""
    field_names = {f.name for f in fields(TrainConfig)}
    kwargs = {}
    for name in field_names:
        if name == "cm_diagnostics":
            kwargs[name] = bool(getattr(args, "cm_diagnostics", False))
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    return TrainConfig(**kwargs)


def pgd_cfg_from_args(args: argparse.Namespace) -> PGDEvalConfig:
    return PGDEvalConfig(
        epsilons=tuple(args.eval_epsilons),
        num_steps=args.eval_num_steps,
        step_size=None,
        restarts=args.eval_restarts,
    )
