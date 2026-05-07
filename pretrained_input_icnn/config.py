"""Training configuration dataclass + CLI parser.

Defaults follow ``Logistic_Regression_CIFAR10/config.py`` for the NPF
hyperparameters (the user identified that as the more principled
reference) and ``run_pretrained_input_icnn.sh`` for the
classifier-side defaults specific to the CIFAR-10 ResNet setting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

ALL_ALGORITHMS: Tuple[str, ...] = (
    "npf",
    "nn_dro",
    "madry",
    "wrm",
    "wfr",
    "dual",
    "new_ppa",
)


@dataclass(frozen=True)
class TrainConfig:
    # --- I/O ---
    data_dir: str = "./data"
    pretrained_path: str = ""
    pretrained_strict: bool = False
    save: str = ""
    save_best_robust: str = ""
    log_csv: str = "./runs_log_input_icnn.csv"

    # --- Algorithm selection ---
    algorithm: str = "npf"

    # --- Outer optimisation (classifier) ---
    epochs_adv: int = 30
    # Warmup: train *only* the adversary (e.g. NPF ω) for this many epochs
    # before the regular minimax loop. The classifier stays frozen during
    # warmup. Mirrors ``--epochs-icnn-pretrain`` from the legacy
    # ``pretrained_INPUT_icnn.py``. For stateless attacks (Madry / WRM /
    # WFR / Dual / New_PPA) warmup is a no-op since they don't have
    # persistent adversary state to carry across batches.
    epochs_icnn_pretrain: int = 0
    batch_size: int = 256
    num_workers: int = 2
    augment_train: bool = True
    lr_theta: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 1

    # --- DRO core hyperparameters ---
    # The penalty weight λ scales ||T(x) - x||^2 in every adversary's
    # objective. The CLI exposes both --penalty-lambda (the ICNN/NPF
    # convention from the original script) and --tau (the Figure-6
    # convention, λ = 1 / (2τ)). Whichever is passed wins.
    lambda_param: float = 30.0
    # When True, replace the per-sample CE in the adversary's primary loss
    # with the log-sum-exp margin: logsumexp_{j≠y}(logit_j - logit_y).
    # Mirrors the legacy ``--use-margin-loss`` flag in
    # ``pretrained_INPUT_icnn.py``. Applied to the adversarial primary
    # loss in every method; the outer classifier update remains CE for
    # primal adversarial-training methods.
    use_margin_loss: bool = False

    # --- Inner-loop budget shared by NPF / NN-DRO ---
    omega_steps_per_batch: int = 10

    # --- Shared BB+Armijo step rule ---
    # Applied UNIFORMLY across all adversaries that have an inner ascent:
    #   * NPF      — BB+Armijo on ω parameters (parametric variant)
    #   * NN-DRO   — BB+Armijo on MLP adversary parameters (replaces Adam)
    #   * WRM      — BB+Armijo on z (input-space variant)
    #   * Madry / RO is exempt: it uses fixed-step pixel-space l2-PGD
    #     with epsilon and no lambda penalty.
    #   * WFR      — BB+Armijo on the deterministic gradient step;
    #                Langevin noise injected after
    #   * New_PPA  — BB+Armijo replaces the legacy WRM ascent inside each
    #                round; projection rounds unchanged
    # Dual is exempt — its inner work is Langevin/MALA sampling of the
    # entropic Gibbs target rather than BB ascent.
    # Defaults mirror NPF's settings from the LR-CIFAR10 reference so a
    # cross-method runtime comparison reflects only the per-method
    # objective cost, not the step rule.
    bb_alpha0: float = 2e-4
    bb_alpha_min: float = 1e-7
    bb_alpha_max: float = 0.25
    bb_ls_c: float = 1e-4
    bb_ls_shrink: float = 0.5
    bb_ls_max_steps: int = 15

    # --- NPF hyperparameters (LR-CIFAR10 defaults) ---
    npf_hidden: Tuple[int, ...] = (512, 512, 256, 128, 64)
    npf_outer_rank: int = 8
    npf_inner_rank: int = 2
    npf_activation: str = "softplus"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 10.0
    npf_init_eps: float = 1e-4
    npf_strong_convexity: float = 1.0
    npf_bb_alpha0: float = 2e-4
    npf_bb_alpha_min: float = 1e-7
    npf_bb_alpha_max: float = 0.25
    npf_bb_ls_c: float = 1e-4
    npf_bb_ls_shrink: float = 0.5
    npf_bb_ls_max_steps: int = 15

    # --- NN-DRO hyperparameters ---
    nn_dro_hidden: Tuple[int, ...] = (512, 512, 256, 256, 128)
    nn_dro_activation: str = "relu"
    nn_dro_softplus_beta: float = 20.0
    nn_dro_init_scale: float = 1e-3
    nn_dro_inner_lr: float = 1e-2

    # --- Madry hyperparameters ---
    madry_epsilon: float = 0.5
    madry_pgd_steps: int = 10
    madry_pgd_step_size: float = 0.0
    madry_pgd_restarts: int = 1

    # --- WRM hyperparameters ---
    wrm_inner_steps: int = 100
    wrm_inner_lr: float = 1e-2

    # --- WFR hyperparameters ---
    wfr_epsilon: float = 0.1
    wfr_num_samples: int = 8
    wfr_inner_steps: int = 50
    wfr_inner_lr: float = 1e-2

    # --- Sinkhorn dual hyperparameters ---
    # The one-shot implementation draws m=2^sample_level Gaussian particles
    # around x and computes the same-lambda closed-form entropic dual on those.
    # Option D replaces the Gaussian-prior particles with honest samples
    # from the Gibbs target via a Langevin chain — set
    # ``dual_langevin_steps > 0`` to enable.
    dual_epsilon: float = 1e-3
    dual_sample_level: int = 5
    # Number of Langevin iterations per particle per batch. 0 keeps the
    # one-shot Gaussian behaviour.
    dual_langevin_steps: int = 0
    # Langevin step size η. Conservative default — start small; with
    # MALA enabled the accept rate is the right tuning signal (target
    # 0.5–0.8). Without MALA, η must be tiny to keep ULA's bias bounded.
    dual_langevin_step_size: float = 1e-3
    # When True, accept/reject each Langevin proposal via the
    # Metropolis-Hastings ratio — gives unbiased samples from the Gibbs
    # target at the cost of one extra fwd+bwd per Langevin step (to
    # evaluate the drift at the proposal). Default True for "rigorous"
    # Option D; pass --no-dual-mala for plain ULA (cheaper, biased).
    dual_mala: bool = True
    # Std of the initial Gaussian draw. None → sqrt(epsilon) (matches
    # the one-shot Gaussian init exactly, so K=0 keeps that behaviour).
    dual_init_noise_scale: Optional[float] = None
    # Number of leading Langevin iterations to discard as burn-in
    # before the dual loss is computed. The discarded steps are still
    # taken (they advance the chain), but the final loss is computed
    # on the post-burn-in state. 0 = no burn-in.
    dual_burn_in: int = 0

    # --- New_PPA hyperparameters ---
    ppa_num_rounds: int = 5
    ppa_min_rounds: int = 2
    ppa_round0_steps: int = 30
    ppa_round0_lr: float = 1e-2
    ppa_refine_steps: int = 15
    ppa_refine_lr: float = 5e-3
    ppa_gain_rtol: float = 1e-4

    # --- Benchmarking ---
    # When True, ``set_deterministic`` is skipped (we still seed RNGs but
    # don't force ``use_deterministic_algorithms(True)``) and
    # ``torch.backends.cudnn.benchmark`` is turned on so kernel autotune
    # picks the fastest path. Use for runtime measurements; leave off for
    # bit-exact reproducibility.
    benchmark_mode: bool = False
    # When True, skip per-epoch input-space PGD evaluation (it's the same
    # cost for every algorithm and dominates wallclock for short runs, so
    # leaving it on biases comparisons toward eval throughput rather than
    # training throughput). Final eval still runs after fit().
    skip_pgd_during_train: bool = False

    # --- Evaluation (input-space PGD) ---
    eval_input_pgd: bool = True
    eval_input_pgd_samples: int = 1000
    inp_p: str = "2"
    inp_eps: float = 0.5
    inp_steps: int = 20
    inp_step_size: float = 0.0
    inp_restarts: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Input-space adversarial training on CIFAR-10. Replaces the legacy "
            "ICNN with NPF and adds parity baselines (NN-DRO, Madry, WRM, WFR, "
            "Sinkhorn Dual, New_PPA) faithful to the LR-CIFAR10 implementations."
        )
    )

    # I/O
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--pretrained-path", type=str, default="")
    parser.add_argument("--pretrained-strict", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--save-best-robust", type=str, default="")
    parser.add_argument("--log-csv", type=str, default="./runs_log_input_icnn.csv")

    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="npf",
        choices=list(ALL_ALGORITHMS),
        help="Which adversarial training algorithm to run.",
    )

    # Outer optimisation
    parser.add_argument("--epochs-adv", type=int, default=30)
    parser.add_argument(
        "--epochs-icnn-pretrain",
        type=int,
        default=0,
        help=(
            "Warmup epochs spent training only the adversary (e.g. NPF ω) "
            "before the standard adversarial training kicks in. The "
            "classifier remains frozen during these epochs. Stateless "
            "attacks (Madry / WRM / WFR / Dual / New_PPA) ignore this knob."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-augment", dest="augment_train", action="store_false")
    parser.set_defaults(augment_train=True)
    parser.add_argument("--lr-theta", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)

    # λ
    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=None,
        help="DRO penalty λ multiplying ||T(x) - x||^2.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Alternative parameterisation: λ = 1/(2τ). Overrides --penalty-lambda when both are set.",
    )

    # Inner-loop budget
    parser.add_argument("--omega-steps-per-batch", type=int, default=10)
    # Shared BB+Armijo step rule (DRO inner ascent only; Madry/RO uses PGD).
    bb = parser.add_argument_group("bb_armijo (shared step rule)")
    bb.add_argument("--bb-alpha0", type=float, default=2e-4)
    bb.add_argument("--bb-alpha-min", type=float, default=1e-7)
    bb.add_argument("--bb-alpha-max", type=float, default=0.25)
    bb.add_argument("--bb-ls-c", type=float, default=1e-4)
    bb.add_argument("--bb-ls-shrink", type=float, default=0.5)
    bb.add_argument("--bb-ls-max-steps", type=int, default=15)
    parser.add_argument(
        "--use-margin-loss",
        dest="use_margin_loss",
        action="store_true",
        help=(
            "Use log-sum-exp margin objective for the adversary "
            "(logsumexp_{j!=y}(logit_j - logit_y)) instead of cross-entropy. "
            "Applied to every method's adversarial primary loss."
        ),
    )
    parser.add_argument(
        "--no-use-margin-loss",
        dest="use_margin_loss",
        action="store_false",
    )
    parser.set_defaults(use_margin_loss=False)

    # NPF
    npf = parser.add_argument_group("npf")
    npf.add_argument("--npf-hidden", type=int, nargs="+", default=[512, 512, 256, 128, 64])
    npf.add_argument("--npf-outer-rank", type=int, default=8)
    npf.add_argument("--npf-inner-rank", type=int, default=2)
    npf.add_argument(
        "--npf-activation", type=str, default="softplus",
        choices=["elu", "softplus", "relu"],
    )
    npf.add_argument("--npf-elu-alpha", type=float, default=1.0)
    npf.add_argument("--npf-softplus-beta", type=float, default=10.0)
    npf.add_argument("--npf-init-eps", type=float, default=1e-4)
    npf.add_argument("--npf-strong-convexity", type=float, default=1.0)
    npf.add_argument("--npf-bb-alpha0", type=float, default=2e-4)
    npf.add_argument("--npf-bb-alpha-min", type=float, default=1e-7)
    npf.add_argument("--npf-bb-alpha-max", type=float, default=0.25)
    npf.add_argument("--npf-bb-ls-c", type=float, default=1e-4)
    npf.add_argument("--npf-bb-ls-shrink", type=float, default=0.5)
    npf.add_argument("--npf-bb-ls-max-steps", type=int, default=15)

    # NN-DRO
    nn_dro = parser.add_argument_group("nn_dro")
    nn_dro.add_argument("--nn-dro-hidden", type=int, nargs="+", default=[512, 512, 256, 256, 128])
    nn_dro.add_argument("--nn-dro-activation", type=str, default="relu")
    nn_dro.add_argument("--nn-dro-softplus-beta", type=float, default=20.0)
    nn_dro.add_argument("--nn-dro-init-scale", type=float, default=1e-3)
    nn_dro.add_argument("--nn-dro-inner-lr", type=float, default=1e-2)

    # Madry
    madry = parser.add_argument_group("madry")
    madry.add_argument("--madry-epsilon", type=float, default=0.5)
    madry.add_argument("--madry-pgd-steps", type=int, default=10)
    madry.add_argument("--madry-pgd-step-size", type=float, default=0.0)
    madry.add_argument("--madry-pgd-restarts", type=int, default=1)

    # WRM
    wrm = parser.add_argument_group("wrm")
    wrm.add_argument("--wrm-inner-steps", type=int, default=100)
    wrm.add_argument("--wrm-inner-lr", type=float, default=1e-2)

    # WFR
    wfr = parser.add_argument_group("wfr")
    wfr.add_argument("--wfr-epsilon", type=float, default=0.1)
    wfr.add_argument("--wfr-num-samples", type=int, default=8)
    wfr.add_argument("--wfr-inner-steps", type=int, default=50)
    wfr.add_argument("--wfr-inner-lr", type=float, default=1e-2)

    # Sinkhorn
    dual = parser.add_argument_group("dual")
    dual.add_argument("--dual-epsilon", type=float, default=1e-3)
    dual.add_argument("--dual-sample-level", type=int, default=5)
    dual.add_argument(
        "--dual-langevin-steps", type=int, default=0,
        help=(
            "Number of Langevin (or MALA) iterations per particle per batch "
            "for Option-D rigorous inner sampling. 0 = one-shot Gaussian."
        ),
    )
    dual.add_argument("--dual-langevin-step-size", type=float, default=1e-3)
    dual.add_argument(
        "--dual-mala", dest="dual_mala", action="store_true",
        help="MH-correct the Langevin proposals (unbiased; one extra fwd+bwd per step).",
    )
    dual.add_argument(
        "--no-dual-mala", dest="dual_mala", action="store_false",
        help="Use plain ULA (biased but cheaper).",
    )
    parser.set_defaults(dual_mala=True)
    dual.add_argument(
        "--dual-init-noise-scale", type=float, default=None,
        help="Std of the initial Gaussian particle draw. Default sqrt(epsilon).",
    )
    dual.add_argument("--dual-burn-in", type=int, default=0)

    # New_PPA
    ppa = parser.add_argument_group("new_ppa")
    ppa.add_argument("--ppa-num-rounds", type=int, default=5)
    ppa.add_argument("--ppa-min-rounds", type=int, default=2)
    ppa.add_argument("--ppa-round0-steps", type=int, default=30)
    ppa.add_argument("--ppa-round0-lr", type=float, default=1e-2)
    ppa.add_argument("--ppa-refine-steps", type=int, default=15)
    ppa.add_argument("--ppa-refine-lr", type=float, default=5e-3)
    ppa.add_argument("--ppa-gain-rtol", type=float, default=1e-4)

    # Benchmarking
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "Enable cuDNN autotune (benchmark=True) and skip "
            "torch.use_deterministic_algorithms(True) for max speed. "
            "RNG seeds are still set so results are comparable across runs."
        ),
    )
    parser.add_argument(
        "--skip-pgd-during-train",
        action="store_true",
        help=(
            "Skip per-epoch input-space PGD evaluation. Useful for runtime "
            "comparisons where eval cost would dominate. Clean / "
            "transport-adversary evals still run (they are O(test-set))."
        ),
    )

    # Evaluation
    parser.add_argument("--eval-input-pgd", dest="eval_input_pgd", action="store_true")
    parser.add_argument("--no-eval-input-pgd", dest="eval_input_pgd", action="store_false")
    parser.set_defaults(eval_input_pgd=True)
    parser.add_argument("--eval-input-pgd-samples", type=int, default=1000)
    parser.add_argument("--inp-p", type=str, default="2", choices=["2", "inf"])
    parser.add_argument("--inp-eps", type=float, default=0.5)
    parser.add_argument("--inp-steps", type=int, default=20)
    parser.add_argument("--inp-step-size", type=float, default=0.0)
    parser.add_argument("--inp-restarts", type=int, default=5)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    if args.tau is not None:
        lambda_param = 1.0 / (2.0 * float(args.tau))
    elif args.penalty_lambda is not None:
        lambda_param = float(args.penalty_lambda)
    else:
        lambda_param = TrainConfig.lambda_param

    field_names = {f.name for f in fields(TrainConfig)}
    kwargs: Dict[str, Any] = {}
    cli_to_field = {
        "data_dir": "data_dir",
        "pretrained_path": "pretrained_path",
        "pretrained_strict": "pretrained_strict",
        "save": "save",
        "save_best_robust": "save_best_robust",
        "log_csv": "log_csv",
        "algorithm": "algorithm",
        "epochs_adv": "epochs_adv",
        "epochs_icnn_pretrain": "epochs_icnn_pretrain",
        "benchmark_mode": "benchmark_mode",
        "skip_pgd_during_train": "skip_pgd_during_train",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "augment_train": "augment_train",
        "lr_theta": "lr_theta",
        "momentum": "momentum",
        "weight_decay": "weight_decay",
        "seed": "seed",
        "omega_steps_per_batch": "omega_steps_per_batch",
        "bb_alpha0": "bb_alpha0",
        "bb_alpha_min": "bb_alpha_min",
        "bb_alpha_max": "bb_alpha_max",
        "bb_ls_c": "bb_ls_c",
        "bb_ls_shrink": "bb_ls_shrink",
        "bb_ls_max_steps": "bb_ls_max_steps",
        "use_margin_loss": "use_margin_loss",
        "npf_hidden": "npf_hidden",
        "npf_outer_rank": "npf_outer_rank",
        "npf_inner_rank": "npf_inner_rank",
        "npf_activation": "npf_activation",
        "npf_elu_alpha": "npf_elu_alpha",
        "npf_softplus_beta": "npf_softplus_beta",
        "npf_init_eps": "npf_init_eps",
        "npf_strong_convexity": "npf_strong_convexity",
        "npf_bb_alpha0": "npf_bb_alpha0",
        "npf_bb_alpha_min": "npf_bb_alpha_min",
        "npf_bb_alpha_max": "npf_bb_alpha_max",
        "npf_bb_ls_c": "npf_bb_ls_c",
        "npf_bb_ls_shrink": "npf_bb_ls_shrink",
        "npf_bb_ls_max_steps": "npf_bb_ls_max_steps",
        "nn_dro_hidden": "nn_dro_hidden",
        "nn_dro_activation": "nn_dro_activation",
        "nn_dro_softplus_beta": "nn_dro_softplus_beta",
        "nn_dro_init_scale": "nn_dro_init_scale",
        "nn_dro_inner_lr": "nn_dro_inner_lr",
        "madry_epsilon": "madry_epsilon",
        "madry_pgd_steps": "madry_pgd_steps",
        "madry_pgd_step_size": "madry_pgd_step_size",
        "madry_pgd_restarts": "madry_pgd_restarts",
        "wrm_inner_steps": "wrm_inner_steps",
        "wrm_inner_lr": "wrm_inner_lr",
        "wfr_epsilon": "wfr_epsilon",
        "wfr_num_samples": "wfr_num_samples",
        "wfr_inner_steps": "wfr_inner_steps",
        "wfr_inner_lr": "wfr_inner_lr",
        "dual_epsilon": "dual_epsilon",
        "dual_sample_level": "dual_sample_level",
        "dual_langevin_steps": "dual_langevin_steps",
        "dual_langevin_step_size": "dual_langevin_step_size",
        "dual_mala": "dual_mala",
        "dual_init_noise_scale": "dual_init_noise_scale",
        "dual_burn_in": "dual_burn_in",
        "ppa_num_rounds": "ppa_num_rounds",
        "ppa_min_rounds": "ppa_min_rounds",
        "ppa_round0_steps": "ppa_round0_steps",
        "ppa_round0_lr": "ppa_round0_lr",
        "ppa_refine_steps": "ppa_refine_steps",
        "ppa_refine_lr": "ppa_refine_lr",
        "ppa_gain_rtol": "ppa_gain_rtol",
        "eval_input_pgd": "eval_input_pgd",
        "eval_input_pgd_samples": "eval_input_pgd_samples",
        "inp_p": "inp_p",
        "inp_eps": "inp_eps",
        "inp_steps": "inp_steps",
        "inp_step_size": "inp_step_size",
        "inp_restarts": "inp_restarts",
    }
    for cli, dest in cli_to_field.items():
        if not hasattr(args, cli):
            continue
        val = getattr(args, cli)
        if dest in {"npf_hidden", "nn_dro_hidden"}:
            val = tuple(val)
        kwargs[dest] = val

    kwargs["lambda_param"] = lambda_param
    # Drop any kwargs not known to TrainConfig (defensive against future drift).
    kwargs = {k: v for k, v in kwargs.items() if k in field_names}
    return TrainConfig(**kwargs)
