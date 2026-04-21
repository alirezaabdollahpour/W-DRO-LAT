"""CLI / dataclass configuration for the RL WDRO pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

from envs import ENV_SPECS, canonical_env_name
from utils.common import IntListAction


# ---------------------------------------------------------------------------
# Inner-loop / adversary hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class InnerConfig:
    lam: float = 2.0
    # Mahalanobis diag weights in xi-space (defaults match the default CartPole xi box).
    M_diag: Tuple[float, ...] = (1.0 / (0.2 - 0.05) ** 2, 1.0 / (0.75 - 0.25) ** 2)
    xi_low: Tuple[float, ...] = (0.05, 0.25)
    xi_high: Tuple[float, ...] = (0.2, 0.75)

    # --- particle ascent ---
    K_part: int = 10
    eta_part: float = 0.05

    # --- ICNN ascent ---
    K_icnn: int = 10
    eta_icnn: float = 5e-2
    icnn_hidden_sizes: Tuple[int, ...] = (1024, 512, 512, 256, 128, 64)
    icnn_activation: str = "softplus"
    icnn_strong_convexity: float = 1.0
    icnn_nonneg_init: str = "principled"  # "principled" (exp) or "xavier" (softplus)
    icnn_softplus_beta: float = 20.0
    icnn_init: str = "identity"

    # BB + Armijo (for ICNN ascent)
    bb_alpha_min: float = 0.0005
    bb_alpha_max: float = 0.01
    bb_ls_c: float = 0.1
    bb_ls_shrink: float = 0.5
    bb_ls_max_steps: int = 10

    # inner gradient estimation params
    grad_method: str = "pathwise"
    fd_eps: float = 1e-2
    fd_episodes: int = 1
    fd_horizon: int = 200

    # --- algo1 / WRM ascent (decaying step η_s = lr / sqrt(s)) ---
    K_algo1: int = 10
    lr_algo1: float = 0.05

    # --- PPA (xi variant): WRM ascent + Mahalanobis refine rounds ---
    ppa_num_rounds: int = 3
    ppa_min_rounds: int = 1
    ppa_inner_steps_round0: int = 10
    ppa_inner_lr_round0: float = 0.05
    ppa_refine_steps: int = 5
    ppa_refine_lr: float = 0.05
    ppa_delta_rtol: float = 1e-3

    # --- new_ppa: WRM ascent + box best-response projection refine rounds ---
    new_ppa_num_rounds: int = 3
    new_ppa_min_rounds: int = 1
    new_ppa_inner_steps_round0: int = 10
    new_ppa_inner_lr_round0: float = 0.05
    new_ppa_refine_steps: int = 5
    new_ppa_refine_lr: float = 0.05
    new_ppa_gain_rtol: float = 1e-3

    # --- dual (log-sum-exp smoothed worst-case) ---
    dual_epsilon: float = 1e-2
    dual_sample_level: int = 3

    # --- wgf / wfr / svg / rgo (particle samplers) ---
    particle_epsilon: float = 1e-2
    particle_num_samples: int = 8
    particle_inner_steps: int = 10
    particle_inner_lr: float = 0.05
    svg_adagrad_hist_decay: float = 0.9

    # --- rgo ---
    rgo_epsilon: float = 1e-2
    rgo_num_samples: int = 8
    rgo_inner_steps: int = 10
    rgo_inner_lr: float = 0.05
    rgo_max_trials: int = 10

    # --- npf (Vesseron & Cuturi, 2024): NPF ICNN potential + persistent Adam ---
    K_npf: int = 10
    lr_npf: float = 5e-2
    npf_hidden_sizes: Tuple[int, ...] = (512, 512, 256, 128, 64)
    npf_outer_rank: int = 4
    npf_inner_rank: int = 1
    npf_activation: str = "elu"  # ELU is the NPF paper's default
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0
    npf_init_eps: float = 1e-3

    # --- nn_dro (vanilla MLP adversary, no gradient-of-potential) ---
    K_nn_dro: int = 10
    lr_nn_dro: float = 5e-2
    nn_dro_hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    nn_dro_activation: str = "relu"
    nn_dro_softplus_beta: float = 20.0
    nn_dro_init_scale: float = 1e-3


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 0.0
    train_epochs: int = 5
    minibatch_size: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


# ---------------------------------------------------------------------------
# Algorithm registry keys (used by --method and the registry)
# ---------------------------------------------------------------------------

ALL_ALGORITHMS = (
    "nominal",
    "particle",
    "icnn",
    "algo1",
    "npf",
    "ppa",
    "new_ppa",
    "dual",
    "wgf",
    "wfr",
    "svg",
    "rgo",
    "nn_dro",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WDRO reinforcement learning (particle ascent, ICNN transport, and Cuturi baselines).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_inner = InnerConfig()
    default_ppo = PPOConfig()

    # --- run control ---
    parser.add_argument(
        "--env",
        choices=["cartpole", "swingup_pendulum", "pendulum", "swingup"],
        default="cartpole",
        help="Which environment to run.",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=["both"],
        help=("Algorithm(s) to run. Accepts any subset of "
              f"{list(ALL_ALGORITHMS)} plus convenience groups "
              "{'both' = particle+icnn, 'all' = all registered methods}."),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--n-hat", type=int, default=64)
    parser.add_argument("--steps-per-xi", type=int, default=64)
    parser.add_argument("--env-max-steps", type=int, default=500)
    parser.add_argument("--policy-hidden", type=int, default=64)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-samples", type=int, default=512)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-json", dest="save_json", action="store_true", default=True)
    parser.add_argument("--no-save-json", dest="save_json", action="store_false")
    parser.add_argument("--json-path", type=str, default=None)

    # --- GIF rendering ---
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--gif-every", type=int, default=10)
    parser.add_argument("--gif-max-steps", type=int, default=500)
    parser.add_argument("--gif-fps", type=int, default=20)

    # --- outer-loop evaluation ---
    parser.add_argument("--eval-nom-episodes", type=int, default=3)
    parser.add_argument("--eval-nom-horizon", type=int, default=500)
    parser.add_argument("--eval-seed-nom", type=int, default=999)
    parser.add_argument("--worst-grid-n", type=int, default=21)
    parser.add_argument("--eval-worst-episodes", type=int, default=1)
    parser.add_argument("--eval-worst-horizon", type=int, default=300)
    parser.add_argument("--eval-seed-worst", type=int, default=123)

    # --- pendulum-specific physics / reward ---
    parser.add_argument("--pendulum-dt", type=float, default=0.1)
    parser.add_argument("--pendulum-u-max", type=float, default=8.0)
    parser.add_argument("--pendulum-max-speed", type=float, default=8.0)
    parser.add_argument("--pendulum-theta-tol", type=float, default=0.2)
    parser.add_argument("--pendulum-vel-tol", type=float, default=1.0)
    parser.add_argument("--pendulum-actions", type=int, choices=[2, 3], default=3)

    # --- inner objective / xi domain ---
    parser.add_argument("--lam", type=float, default=default_inner.lam)
    parser.add_argument("--xi-low", type=float, nargs="+", default=None)
    parser.add_argument("--xi-high", type=float, nargs="+", default=None)
    parser.add_argument("--m-diag", type=float, nargs="+", default=None)

    parser.add_argument("--k-part", type=int, default=default_inner.K_part)
    parser.add_argument("--eta-part", type=float, default=default_inner.eta_part)
    parser.add_argument("--k-icnn", type=int, default=default_inner.K_icnn)
    parser.add_argument("--eta-icnn", type=float, default=default_inner.eta_icnn)
    parser.add_argument("--grad-method", choices=["pathwise", "spsa", "fd"], default=default_inner.grad_method)
    parser.add_argument("--fd-eps", type=float, default=default_inner.fd_eps)
    parser.add_argument("--fd-episodes", type=int, default=default_inner.fd_episodes)
    parser.add_argument("--fd-horizon", type=int, default=default_inner.fd_horizon)

    # --- ICNN ---
    parser.add_argument(
        "--icnn-hidden-sizes",
        nargs="+",
        action=IntListAction,
        default=list(default_inner.icnn_hidden_sizes),
    )
    parser.add_argument("--icnn-activation", choices=["softplus", "relu"], default=default_inner.icnn_activation)
    parser.add_argument("--icnn-strong-convexity", type=float, default=default_inner.icnn_strong_convexity)
    parser.add_argument("--icnn-nonneg-init", choices=["principled", "xavier"], default=default_inner.icnn_nonneg_init)
    parser.add_argument("--icnn-softplus-beta", type=float, default=default_inner.icnn_softplus_beta)
    parser.add_argument("--icnn-init", choices=["identity", "default"], default=default_inner.icnn_init)
    parser.add_argument("--bb-alpha-min", type=float, default=default_inner.bb_alpha_min)
    parser.add_argument("--bb-alpha-max", type=float, default=default_inner.bb_alpha_max)
    parser.add_argument("--bb-ls-c", type=float, default=default_inner.bb_ls_c)
    parser.add_argument("--bb-ls-shrink", type=float, default=default_inner.bb_ls_shrink)
    parser.add_argument("--bb-ls-max-steps", type=int, default=default_inner.bb_ls_max_steps)

    # --- algo1 WRM (xi) ---
    parser.add_argument("--k-algo1", type=int, default=default_inner.K_algo1)
    parser.add_argument("--lr-algo1", type=float, default=default_inner.lr_algo1)

    # --- ppa (xi) ---
    parser.add_argument("--ppa-num-rounds", type=int, default=default_inner.ppa_num_rounds)
    parser.add_argument("--ppa-min-rounds", type=int, default=default_inner.ppa_min_rounds)
    parser.add_argument("--ppa-inner-steps-round0", type=int, default=default_inner.ppa_inner_steps_round0)
    parser.add_argument("--ppa-inner-lr-round0", type=float, default=default_inner.ppa_inner_lr_round0)
    parser.add_argument("--ppa-refine-steps", type=int, default=default_inner.ppa_refine_steps)
    parser.add_argument("--ppa-refine-lr", type=float, default=default_inner.ppa_refine_lr)
    parser.add_argument("--ppa-delta-rtol", type=float, default=default_inner.ppa_delta_rtol)

    # --- new_ppa (xi) ---
    parser.add_argument("--new-ppa-num-rounds", type=int, default=default_inner.new_ppa_num_rounds)
    parser.add_argument("--new-ppa-min-rounds", type=int, default=default_inner.new_ppa_min_rounds)
    parser.add_argument("--new-ppa-inner-steps-round0", type=int, default=default_inner.new_ppa_inner_steps_round0)
    parser.add_argument("--new-ppa-inner-lr-round0", type=float, default=default_inner.new_ppa_inner_lr_round0)
    parser.add_argument("--new-ppa-refine-steps", type=int, default=default_inner.new_ppa_refine_steps)
    parser.add_argument("--new-ppa-refine-lr", type=float, default=default_inner.new_ppa_refine_lr)
    parser.add_argument("--new-ppa-gain-rtol", type=float, default=default_inner.new_ppa_gain_rtol)

    # --- dual ---
    parser.add_argument("--dual-epsilon", type=float, default=default_inner.dual_epsilon)
    parser.add_argument("--dual-sample-level", type=int, default=default_inner.dual_sample_level)

    # --- particle samplers (wgf/wfr/svg) ---
    parser.add_argument("--particle-epsilon", type=float, default=default_inner.particle_epsilon)
    parser.add_argument("--particle-num-samples", type=int, default=default_inner.particle_num_samples)
    parser.add_argument("--particle-inner-steps", type=int, default=default_inner.particle_inner_steps)
    parser.add_argument("--particle-inner-lr", type=float, default=default_inner.particle_inner_lr)
    parser.add_argument("--svg-adagrad-hist-decay", type=float, default=default_inner.svg_adagrad_hist_decay)

    # --- rgo ---
    parser.add_argument("--rgo-epsilon", type=float, default=default_inner.rgo_epsilon)
    parser.add_argument("--rgo-num-samples", type=int, default=default_inner.rgo_num_samples)
    parser.add_argument("--rgo-inner-steps", type=int, default=default_inner.rgo_inner_steps)
    parser.add_argument("--rgo-inner-lr", type=float, default=default_inner.rgo_inner_lr)
    parser.add_argument("--rgo-max-trials", type=int, default=default_inner.rgo_max_trials)

    # --- npf ---
    parser.add_argument("--k-npf", type=int, default=default_inner.K_npf)
    parser.add_argument("--lr-npf", type=float, default=default_inner.lr_npf)
    parser.add_argument(
        "--npf-hidden-sizes", nargs="+", action=IntListAction,
        default=list(default_inner.npf_hidden_sizes),
        help="NPF ICNN hidden widths (separate from --icnn-hidden-sizes).",
    )
    parser.add_argument("--npf-outer-rank", type=int, default=default_inner.npf_outer_rank)
    parser.add_argument("--npf-inner-rank", type=int, default=default_inner.npf_inner_rank)
    parser.add_argument(
        "--npf-activation", choices=["elu", "softplus", "relu"],
        default=default_inner.npf_activation,
    )
    parser.add_argument("--npf-elu-alpha", type=float, default=default_inner.npf_elu_alpha)
    parser.add_argument("--npf-softplus-beta", type=float, default=default_inner.npf_softplus_beta)
    parser.add_argument(
        "--npf-init-eps", type=float, default=default_inner.npf_init_eps,
        help="Live-at-init scale for per-layer quadratic forms (0 to disable).",
    )

    # --- nn_dro (vanilla MLP adversary) ---
    parser.add_argument("--k-nn-dro", type=int, default=default_inner.K_nn_dro)
    parser.add_argument("--lr-nn-dro", type=float, default=default_inner.lr_nn_dro)
    parser.add_argument(
        "--nn-dro-hidden-sizes", nargs="+", action=IntListAction,
        default=list(default_inner.nn_dro_hidden_sizes),
        help="Hidden widths for the NN-DRO MLP adversary.",
    )
    parser.add_argument(
        "--nn-dro-activation", choices=["relu", "elu", "tanh", "softplus"],
        default=default_inner.nn_dro_activation,
    )
    parser.add_argument("--nn-dro-softplus-beta", type=float, default=default_inner.nn_dro_softplus_beta)
    parser.add_argument("--nn-dro-init-scale", type=float, default=default_inner.nn_dro_init_scale)

    # --- PPO ---
    parser.add_argument("--ppo-gamma", type=float, default=default_ppo.gamma)
    parser.add_argument("--ppo-gae-lambda", type=float, default=default_ppo.gae_lambda)
    parser.add_argument("--ppo-clip-eps", type=float, default=default_ppo.clip_eps)
    parser.add_argument("--ppo-pi-lr", type=float, default=default_ppo.pi_lr)
    parser.add_argument("--ppo-vf-lr", type=float, default=default_ppo.vf_lr)
    parser.add_argument("--ppo-sgd-momentum", type=float, default=default_ppo.sgd_momentum)
    parser.add_argument("--ppo-sgd-weight-decay", type=float, default=default_ppo.sgd_weight_decay)
    parser.add_argument("--ppo-train-epochs", type=int, default=default_ppo.train_epochs)
    parser.add_argument("--ppo-minibatch-size", type=int, default=default_ppo.minibatch_size)
    parser.add_argument("--ppo-ent-coef", type=float, default=default_ppo.ent_coef)
    parser.add_argument("--ppo-vf-coef", type=float, default=default_ppo.vf_coef)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=default_ppo.max_grad_norm)

    return parser


def build_inner_config_from_args(args: argparse.Namespace, *, env_name: str) -> InnerConfig:
    env_name = canonical_env_name(env_name)
    spec = ENV_SPECS[env_name]

    if (args.xi_low is None) != (args.xi_high is None):
        raise ValueError("Provide both --xi-low and --xi-high (or omit both to use env defaults).")

    if args.xi_low is None:
        xi_low = tuple(float(v) for v in spec.xi_low)
        xi_high = tuple(float(v) for v in spec.xi_high)
    else:
        xi_low = tuple(float(v) for v in args.xi_low)
        xi_high = tuple(float(v) for v in args.xi_high)

    if len(xi_low) != spec.xi_dim or len(xi_high) != spec.xi_dim:
        raise ValueError(
            f"For env='{env_name}', expected {spec.xi_dim} values for --xi-low/--xi-high "
            f"(xi order={spec.xi_names}). Got xi_low={xi_low} xi_high={xi_high}."
        )
    if any(lo >= hi for lo, hi in zip(xi_low, xi_high)):
        raise ValueError(f"Expected xi_low < xi_high elementwise; got xi_low={xi_low} xi_high={xi_high}.")

    if args.m_diag is None:
        M_diag = tuple(1.0 / (hi - lo) ** 2 for lo, hi in zip(xi_low, xi_high))
    else:
        m_diag = tuple(float(v) for v in args.m_diag)
        if len(m_diag) != spec.xi_dim:
            raise ValueError(
                f"For env='{env_name}', expected {spec.xi_dim} values for --m-diag; got {m_diag}."
            )
        M_diag = m_diag

    return InnerConfig(
        lam=float(args.lam),
        M_diag=M_diag,
        xi_low=xi_low,
        xi_high=xi_high,
        K_part=int(args.k_part),
        eta_part=float(args.eta_part),
        K_icnn=int(args.k_icnn),
        eta_icnn=float(args.eta_icnn),
        icnn_hidden_sizes=tuple(int(v) for v in args.icnn_hidden_sizes),
        icnn_activation=str(args.icnn_activation),
        icnn_strong_convexity=float(args.icnn_strong_convexity),
        icnn_nonneg_init=str(args.icnn_nonneg_init),
        icnn_softplus_beta=float(args.icnn_softplus_beta),
        icnn_init=str(args.icnn_init),
        bb_alpha_min=float(args.bb_alpha_min),
        bb_alpha_max=float(args.bb_alpha_max),
        bb_ls_c=float(args.bb_ls_c),
        bb_ls_shrink=float(args.bb_ls_shrink),
        bb_ls_max_steps=int(args.bb_ls_max_steps),
        grad_method=str(args.grad_method),
        fd_eps=float(args.fd_eps),
        fd_episodes=int(args.fd_episodes),
        fd_horizon=int(args.fd_horizon),
        K_algo1=int(args.k_algo1),
        lr_algo1=float(args.lr_algo1),
        ppa_num_rounds=int(args.ppa_num_rounds),
        ppa_min_rounds=int(args.ppa_min_rounds),
        ppa_inner_steps_round0=int(args.ppa_inner_steps_round0),
        ppa_inner_lr_round0=float(args.ppa_inner_lr_round0),
        ppa_refine_steps=int(args.ppa_refine_steps),
        ppa_refine_lr=float(args.ppa_refine_lr),
        ppa_delta_rtol=float(args.ppa_delta_rtol),
        new_ppa_num_rounds=int(args.new_ppa_num_rounds),
        new_ppa_min_rounds=int(args.new_ppa_min_rounds),
        new_ppa_inner_steps_round0=int(args.new_ppa_inner_steps_round0),
        new_ppa_inner_lr_round0=float(args.new_ppa_inner_lr_round0),
        new_ppa_refine_steps=int(args.new_ppa_refine_steps),
        new_ppa_refine_lr=float(args.new_ppa_refine_lr),
        new_ppa_gain_rtol=float(args.new_ppa_gain_rtol),
        dual_epsilon=float(args.dual_epsilon),
        dual_sample_level=int(args.dual_sample_level),
        particle_epsilon=float(args.particle_epsilon),
        particle_num_samples=int(args.particle_num_samples),
        particle_inner_steps=int(args.particle_inner_steps),
        particle_inner_lr=float(args.particle_inner_lr),
        svg_adagrad_hist_decay=float(args.svg_adagrad_hist_decay),
        rgo_epsilon=float(args.rgo_epsilon),
        rgo_num_samples=int(args.rgo_num_samples),
        rgo_inner_steps=int(args.rgo_inner_steps),
        rgo_inner_lr=float(args.rgo_inner_lr),
        rgo_max_trials=int(args.rgo_max_trials),
        K_npf=int(args.k_npf),
        lr_npf=float(args.lr_npf),
        npf_hidden_sizes=tuple(int(v) for v in args.npf_hidden_sizes),
        npf_outer_rank=int(args.npf_outer_rank),
        npf_inner_rank=int(args.npf_inner_rank),
        npf_activation=str(args.npf_activation),
        npf_elu_alpha=float(args.npf_elu_alpha),
        npf_softplus_beta=float(args.npf_softplus_beta),
        npf_init_eps=float(args.npf_init_eps),
        K_nn_dro=int(args.k_nn_dro),
        lr_nn_dro=float(args.lr_nn_dro),
        nn_dro_hidden_sizes=tuple(int(v) for v in args.nn_dro_hidden_sizes),
        nn_dro_activation=str(args.nn_dro_activation),
        nn_dro_softplus_beta=float(args.nn_dro_softplus_beta),
        nn_dro_init_scale=float(args.nn_dro_init_scale),
    )


def build_ppo_config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        gamma=float(args.ppo_gamma),
        gae_lambda=float(args.ppo_gae_lambda),
        clip_eps=float(args.ppo_clip_eps),
        pi_lr=float(args.ppo_pi_lr),
        vf_lr=float(args.ppo_vf_lr),
        sgd_momentum=float(args.ppo_sgd_momentum),
        sgd_weight_decay=float(args.ppo_sgd_weight_decay),
        train_epochs=int(args.ppo_train_epochs),
        minibatch_size=int(args.ppo_minibatch_size),
        ent_coef=float(args.ppo_ent_coef),
        vf_coef=float(args.ppo_vf_coef),
        max_grad_norm=float(args.ppo_max_grad_norm),
    )


def resolve_methods(requested: list[str]) -> list[str]:
    """Resolve --method tokens into a concrete list of algorithm keys."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in requested:
        key = str(tok).strip().lower()
        if key == "":
            continue
        if key == "all":
            for m in ALL_ALGORITHMS:
                if m not in seen:
                    out.append(m); seen.add(m)
        elif key == "both":
            for m in ("particle", "icnn"):
                if m not in seen:
                    out.append(m); seen.add(m)
        else:
            if key not in ALL_ALGORITHMS:
                raise ValueError(
                    f"Unknown method '{tok}'. Choices: {list(ALL_ALGORITHMS)} plus {{'both','all'}}."
                )
            if key not in seen:
                out.append(key); seen.add(key)
    if not out:
        raise ValueError("No methods resolved from --method.")
    return out
