"""Training configuration dataclass + CLI parser.

Defaults follow ``Logistic_Regression_CIFAR10/config.py`` for the NPF
hyperparameters (the user identified that as the more principled
reference) and ``run_pretrained_input_icnn.sh`` for the
classifier-side defaults specific to the CIFAR-10 ResNet setting.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

ALL_ALGORITHMS: Tuple[str, ...] = (
    "npf",
    "npf_lastquad", # this is going to be our main algorithm, because it just injects quadratic terms at the last layer---reduces computional cost of npf(injecting quadratics at every layer)---we stick to this algorithm.
    "nn_dro",
    "madry", # famous PGD-based adversarial training method from Madry et al. (2018); it's a primal method that optimizes the inner maximization by running projected gradient ascent in the input space with a fixed step size and projection radius (epsilon) for a fixed number of steps and restarts.
    "wrm",
    "wfr",
    "dual",
    "new_ppa", # is our MPA version: MPA approximates the regularized Wasserstein DRO inner maximization, i.e. role of the adversary, by alternating between parallel local optimization of $B$ adversarial samples~$\mathcal Z=\{z_i\}_{i=1}^B$ ($B$ is a batch size) and batch-wide reassignment of each empirical sample~$\hat z_i$~(i.e., nominal sample) to the best adversarial sample in~$\mathcal Z$, thereby restoring assignment stationarity of the implicit map; see the inner loop in Algorithm~\ref{alg:implicit_mpa}, which runs over~$R$ rounds. Note that MPA collapses to PA if~$R=1$, while $R>1$ activates the reassignment steps.
)


@dataclass(frozen=True)
class TrainConfig:
    # --- I/O ---
    data_dir: str = "./data"
    pretrained_path: str = ""
    pretrained_strict: bool = False
    save: str = ""
    save_best_robust: str = ""
    resume_checkpoint: str = ""
    save_every_epoch_dir: str = ""
    checkpoint_epoch_offset: int = 0
    log_csv: str = "./runs_log_input_icnn.csv"

    # --- Algorithm selection ---
    algorithm: str = "npf_lastquad"

    # --- Outer optimisation (classifier) ---
    epochs_adv: int = 50
    # Warmup: train *only* the adversary (e.g. NPF ω) for this many epochs
    # before the regular minimax loop. The classifier stays frozen during
    # warmup. Mirrors ``--epochs-icnn-pretrain`` from the legacy
    # ``pretrained_INPUT_icnn.py``. For stateless attacks (Madry / WRM /
    # WFR / Dual / New_PPA) warmup is a no-op since they don't have
    # persistent adversary state to carry across batches.
    epochs_icnn_pretrain: int = 2
    # Optional post phase: after ordinary adversarial training, freeze the
    # learned parametric adversary and continue updating only the classifier on
    # T_omega^m(x). This is useful for testing whether the learned NPF map can
    # be reused as a fixed data transform.
    frozen_adversary_epochs: int = 0
    frozen_adversary_map_steps: int = 1
    batch_size: int = 512
    num_workers: int = 2
    augment_train: bool = True
    lr_theta: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    freeze_batchnorm: bool = False
    freeze_batchnorm_affine: bool = False
    # Stable legacy input-ICNN behavior keeps the classifier in train mode
    # during adversary-side forwards in minimax epochs, while freezing only
    # parameter gradients. That lets BatchNorm running stats adapt online.
    # The default here is the safer eval-mode adversary objective used by the
    # newer runtime code; pass --adversary-classifier-train to match legacy.
    adversary_classifier_eval: bool = True
    online_batchnorm_refresh: bool = False
    batchnorm_online_refresh_momentum: Optional[float] = None
    recalibrate_batchnorm: bool = False
    batchnorm_recalibration_batches: int = 0
    batchnorm_recalibration_reset: bool = True
    batchnorm_recalibration_momentum: Optional[float] = None
    seed: int = 1

    # --- DRO core hyperparameters ---
    # The penalty weight λ scales the configured transport cost in every
    # adversary's objective. The default cost matches the legacy
    # pretrained_INPUT_icnn.py implementation: mean squared difference in
    # normalized CIFAR coordinates. Classifier inputs remain normalized
    # tensors throughout. The CLI exposes both --penalty-lambda (the ICNN/NPF convention from
    # the original script) and --tau (the Figure-6 convention,
    # λ = 1 / (2τ)). Whichever is passed wins.
    lambda_param: float = 30.0
    # Optional piecewise-constant lambda schedule for adversarial epochs.
    # When non-empty, lambda_schedule[0] is used for adversarial epochs
    # 1..lambda_stage_epochs, lambda_schedule[1] for the next block, etc.
    # This is intentionally handled in-process so optimizer state, Muon state,
    # LR scheduler state, and DDP sampler epoch progression remain continuous.
    lambda_schedule: Tuple[float, ...] = ()
    lambda_stage_epochs: int = 0
    # When True, replace the per-sample CE in the adversary's primary loss
    # with the log-sum-exp margin: logsumexp_{j≠y}(logit_j - logit_y).
    # Mirrors the legacy ``--use-margin-loss`` flag in
    # ``pretrained_INPUT_icnn.py``. Applied to the adversarial primary
    # loss in every method; the outer classifier update remains CE for
    # primal adversarial-training methods.
    use_margin_loss: bool = True
    # Legacy pretrained_INPUT_icnn.py semantics: build adversarial transports
    # only for examples currently classified correctly by the clean classifier.
    # Clean-misclassified examples stay at x in the outer update.
    attack_clean_correct_only: bool = True
    # Legacy BB+Armijo semantics for parametric adversaries: reset the BB
    # secant history at each batch because the stochastic objective changes
    # with the batch and with the classifier. Persistent BB can shrink to a
    # weak adversary late in training.
    reset_parametric_bb_each_batch: bool = True
    # Transport penalty convention for DRO inner objectives. The default
    # matches legacy pretrained_INPUT_icnn.py: mean squared difference in
    # normalized CIFAR coordinates. Use pixel_l2_squared only for experiments
    # that intentionally want standard pixel-space squared L2 scaling.
    transport_cost: str = "normalized_mse"

    # --- Inner-loop budget shared by NPF / NN-DRO ---
    omega_steps_per_batch: int = 10

    # --- NPF inner optimizer selection ---
    # ``bb_armijo`` preserves the existing line-searched ascent. ``muon`` uses
    # momentum plus Newton-Schulz orthogonalized matrix updates for the NPF ICNN
    # parameters, with an AdamW/SGD fallback for vector parameters.
    npf_inner_optimizer: str = "bb_armijo"
    # Reset the NPF potential to its identity-adjacent initialization before
    # every batch's inner maximization. This is useful when the classifier
    # quickly overfits a persistent learned map and the inner loop needs fresh
    # restarts, closer in spirit to per-batch PGD.
    npf_reset_omega_each_batch: bool = False
    # Project the learned NPF transport output into the configured input-PGD
    # threat set before scoring the inner objective and before classifier
    # updates. This is off by default for unconstrained WDRO experiments, but
    # useful when directly targeting PGD robustness at cfg.inp_eps.
    npf_project_to_input_ball: bool = False
    # Optional amortization/anchor term for PGD-targeted NPF training:
    # the inner objective additionally encourages T_omega(x) to match a small
    # per-batch PGD adversary. Weight 0 keeps the pure NPF objective.
    npf_pgd_anchor_weight: float = 0.0
    npf_pgd_anchor_steps: int = 5
    npf_pgd_anchor_restarts: int = 1
    npf_pgd_anchor_loss: str = "margin"
    # L2 decay applied inside the omega ascent objective to the FREE
    # (unconstrained) input-side kernels only: hidden affine input skips and
    # the PosDef lin/quad kernels. Legacy pretrained_INPUT_icnn folded exactly
    # this (1e-4) into every accepted BB step; over ~30k persistent-omega
    # steps it is the slow force that taxes unbounded growth of the far-field
    # translation/rescale kernels while leaving the positive-reparametrized
    # ladder and biases untouched.
    npf_omega_weight_decay: float = 1e-4

    # --- Shared BB+Armijo step rule ---
    # Applied UNIFORMLY across all adversaries that have an inner ascent:
    #   * NPF      — BB+Armijo on ω parameters when
    #                npf_inner_optimizer="bb_armijo" (default)
    #   * NN-DRO   — BB+Armijo on MLP adversary parameters (replaces Adam)
    #   * WRM      — only with --wrm-step-rule bb_armijo (default is the
    #                MPA-parity per-sample const_lr ascent)
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
    bb_alpha0: float = 5e-4
    bb_alpha_min: float = 1e-6
    bb_alpha_max: float = 1.0
    bb_ls_c: float = 0.1
    bb_ls_shrink: float = 0.5
    bb_ls_max_steps: int = 10
    # Legacy pretrained_INPUT_icnn.py clipped ICNN adversary gradients before
    # the BB proposal and Armijo trials. Applies only to shared parametric
    # BB adversaries (NPF / NPF-LastQuad / NN-DRO). Set 0 to disable.
    parametric_bb_max_grad_norm: float = 1.0

    # --- NPF hyperparameters (OTT-style ICNN defaults) ---
    npf_hidden: Tuple[int, ...] = (1024, 512, 512, 256, 128, 64)
    npf_outer_rank: int = 1
    npf_inner_rank: int = 1
    npf_activation: str = "softplus"
    npf_elu_alpha: float = 1.0
    npf_softplus_beta: float = 20.0
    # Retained for checkpoint/config compatibility. The faithful OTT
    # constructor does not use this scale unless exact identity init is enabled.
    npf_init_eps: float = 1e-2
    npf_identity_init: bool = True
    npf_strong_convexity: float = 1.0
    # OTT exposes pos_weights=False, assuming a separate clipping/regularizing
    # path. This trainer has no such projection pass, so keep projection on by
    # default to preserve ICNN convexity.
    npf_pos_weights: bool = True
    npf_positive_weight_rectifier: str = "exp"
    npf_bb_alpha0: float = 2e-4
    npf_bb_alpha_min: float = 1e-7
    npf_bb_alpha_max: float = 0.25
    npf_bb_ls_c: float = 1e-4
    npf_bb_ls_shrink: float = 0.5
    npf_bb_ls_max_steps: int = 15
    npf_muon_lr: float = 2e-4
    npf_muon_momentum: float = 0.95
    npf_muon_nesterov: bool = True
    npf_muon_ns_steps: int = 5
    npf_muon_matrix_lr_scale: str = "auto"
    npf_muon_weight_decay: float = 0.0
    npf_muon_fallback: str = "adamw"
    npf_muon_fallback_lr: float = 0.0
    npf_muon_fallback_weight_decay: float = 0.0
    npf_muon_adam_beta1: float = 0.9
    npf_muon_adam_beta2: float = 0.999
    npf_muon_adam_eps: float = 1e-8
    npf_muon_max_grad_norm: float = 0.0

    # --- NPF last-quadratic-only variant ---
    # This shares the OTT-style NPF trainer. Hidden input skips are affine, and
    # the final activated scalar input skip is the only learned quadratic block.
    # The separate final identity positive-definite potential remains present.
    npf_lastquad_hidden: Tuple[int, ...] = (1024, 512, 512, 256, 128, 64)
    # Rank of the final trainable input quadratic. The 2026-07-06 probe and
    # the ablC training run proved this is the decisive capacity knob: rank 2
    # confines the ascent to 1-2 shared far-field directions (top-1 SVD ~46%
    # of delta energy, transported acc barely moves), while rank 64 makes the
    # same K=10 ascent drive transported acc 94%->17% with far more
    # sample-dependent deltas (top-1 SVD ~26%) and lifted PGD@0.5 from ~10%
    # to 45%+ in full training.
    npf_lastquad_output_rank: int = 64
    npf_lastquad_activation: str = "softplus"
    npf_lastquad_elu_alpha: float = 1.0
    npf_lastquad_softplus_beta: float = 20.0
    npf_lastquad_init_eps: float = 1e-2
    npf_lastquad_identity_init: bool = False
    npf_lastquad_strong_convexity: float = 1.0
    npf_lastquad_pos_weights: bool = True
    npf_lastquad_positive_weight_rectifier: str = "exp"

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
    # WRM == MPA with R=1 (pure K-step per-sample particle ascent, no
    # reassignment), sharing lambda_param / transport_cost / use_margin_loss
    # / attack_clean_correct_only with MPA.
    #   * "const_lr" (default): per-sample ascent on the WRM diminishing
    #     schedule eta_s = wrm_inner_lr / sqrt(s) — the exact same code path
    #     as MPA's round-0 burst, so WRM(K, lr) is bitwise MPA(R=1,
    #     round0_steps=K, round0_lr=lr). Shard-invariant under DDP.
    #   * "bb_armijo": the legacy shared line-searched rule on the
    #     batch-mean objective (cross-method parity arm).
    wrm_step_rule: str = "const_lr"
    wrm_inner_steps: int = 100
    # Calibrated like MPA's ppa_round0_lr on the current 95.3%-clean R2.pth
    # (margin, lambda=30 normalized_mse, K=20): 0.03 -> inner obj ~46 at
    # mean pixel-L2 ~0.59 (the eval-radius scale).
    wrm_inner_lr: float = 0.03

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

    # --- New_PPA (MPA) hyperparameters ---
    # Step rule for the per-round ascent bursts:
    #   * "bb_armijo" — shared BB+Armijo line search on the batch-mean
    #     objective (uniform with the other DRO adversaries here). The
    #     ppa_*_lr fields are unused.
    #   * "const_lr" — faithful MNIST_Cuturi / Algorithm-1 semantics:
    #     per-sample updates with round 0 on the WRM diminishing schedule
    #     eta_s = ppa_round0_lr / sqrt(s) and refinement rounds at the
    #     constant ppa_refine_lr. No cross-sample coupling, so the ascent is
    #     invariant to how the batch is sharded across DDP ranks.
    ppa_step_rule: str = "bb_armijo"
    # Within-round order:
    #   * "ascent_first" (Algorithm 1): rounds of {K ascent steps; reassign};
    #     R=1 is plain PA (no reassignment).
    #   * "reassign_first": mirrored rounds of {reassign; K ascent steps},
    #     closed by a final reassignment (R=3: R-A-R-A-R-A-R) — the round-0
    #     reassignment fires at z=x (best same-class training sample as the
    #     ascent start) and, like ascent_first, the schedule ends with a
    #     reassignment so the theta-update sees within-class best responses.
    #   * "reassign_first_v2": identical to reassign_first except the ROUND-0
    #     reassignment is priced at λ₀ = λ / ppa_round0_lambda_damping
    #     (graduated multi-start). At full λ the round-0 reassignment at z=x
    #     is inert (measured 0.3% jumps at λ=30 on the 95.3% R2.pth); the
    #     damped score activates confidence-driven jumps while the retained
    #     cost term keeps the argmax anchor-dependent, preserving particle
    #     diversity. Everything else (rounds >= 1, closing reassignment,
    #     ascent objective, logged metrics) uses the full λ.
    # All other knobs (steps, LRs, min_rounds, gain_rtol) are shared.
    ppa_round_order: str = "ascent_first"
    # α for reassign_first_v2's round-0 penalty discount: λ₀ = λ/α. Read
    # ONLY by reassign_first_v2 (ascent_first and reassign_first ignore
    # it); 1.0 reduces v2 to reassign_first exactly. Measured round-0
    # activation on the current R2.pth at λ=30 (z=x): α=6 (λ₀=5) -> 52.8%
    # of samples jump; α=10 (λ₀=3) -> 72.7%.
    ppa_round0_lambda_damping: float = 6.0
    ppa_num_rounds: int = 5
    ppa_min_rounds: int = 2
    ppa_round0_steps: int = 30
    # const_lr defaults calibrated on the CURRENT R2.pth (2026-07-08 swap:
    # 95.3%-clean PreActResNet18, 'last' weights; margin loss, lambda=30
    # normalized_mse, K=20). This model is much steeper than the old
    # 87.3% R2_Old.pth: round0 lr=0.03 reaches inner obj ~46 at mean
    # pixel-L2 ~0.59 (the eval-radius scale); 0.1 -> obj 91 / L2 1.28 /
    # acc 0%. BB+Armijo at the same K stays at obj ~1.2 / L2 0.16.
    # Refine keeps the MNIST 2:1 round0:refine ratio. Old-model calibration
    # (R2_Old.pth) was 0.1 / 0.05.
    ppa_round0_lr: float = 0.03
    ppa_refine_steps: int = 15
    ppa_refine_lr: float = 0.015
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
    # When True, record synchronized, part-by-part timings for the inner
    # maximization. This intentionally inserts CUDA synchronizations around
    # small regions, so use it for ablations/profiling rather than production
    # training throughput.
    profile_inner: bool = False
    # Number of train batches per epoch to profile. 0 means every batch.
    profile_inner_batches: int = 0

    # --- Evaluation (input-space PGD) ---
    eval_input_pgd: bool = True
    eval_input_pgd_samples: int = 1024
    # Cap on test points for the per-epoch transport-adversary evaluation.
    # 0 = full test set (fine for cheap learned maps like NPF; the legacy
    # behavior). Set to e.g. 1024 for transductive attacks (new_ppa/MPA)
    # where transport_for_eval runs a full inner maximization per batch —
    # uncapped it serializes minutes of attack compute onto rank 0 every
    # epoch while other DDP ranks wait at the barrier.
    eval_transport_samples: int = 0
    eval_transport_pgd_alignment: bool = False
    eval_transport_pgd_alignment_samples: int = 256
    input_pgd_loss: str = "ce"
    input_pgd_geometry: str = "pixel_l2_squared"
    inp_p: str = "2"
    inp_eps: float = 0.5
    inp_steps: int = 20
    inp_step_size: float = 0.0
    inp_restarts: int = 5
    wrm_normalized_radius_protocol: str = ""
    wrm_pgd_eps_mode: str = ""
    wrm_normalized_radius_requested: Optional[float] = None
    wrm_normalized_radius: Optional[float] = None
    wrm_cp: Optional[float] = None
    wrm_effective_inp_eps: Optional[float] = None

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
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help=(
            "Resume classifier and parametric adversary weights from a previous "
            "pretrained_input_icnn checkpoint. Optimizer state is intentionally "
            "not restored; use --lambda-schedule for exact in-process lambda "
            "scheduling."
        ),
    )
    parser.add_argument(
        "--save-every-epoch-dir",
        type=str,
        default="",
        help=(
            "When set, write a classifier + adversary checkpoint at the end of "
            "every epoch into this directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-epoch-offset",
        type=int,
        default=0,
        help=(
            "Global epoch offset used in per-epoch checkpoint filenames, useful "
            "for staged schedules."
        ),
    )
    parser.add_argument("--log-csv", type=str, default="./runs_log_input_icnn.csv")

    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="npf_lastquad",
        choices=list(ALL_ALGORITHMS),
        help="Which adversarial training algorithm to run.",
    )

    # Outer optimisation
    parser.add_argument("--epochs-adv", type=int, default=50)
    parser.add_argument(
        "--epochs-icnn-pretrain",
        type=int,
        default=2,
        help=(
            "Warmup epochs spent training only the adversary (e.g. NPF ω) "
            "before the standard adversarial training kicks in. The "
            "classifier remains frozen during these epochs. Stateless "
            "attacks (Madry / WRM / WFR / Dual / New_PPA) ignore this knob."
        ),
    )
    parser.add_argument(
        "--frozen-adversary-epochs",
        type=int,
        default=0,
        help=(
            "After --epochs-adv, freeze the learned parametric adversary and "
            "continue training only the classifier for this many epochs."
        ),
    )
    parser.add_argument(
        "--frozen-adversary-map-steps",
        type=int,
        default=1,
        help=(
            "Number of times to apply the frozen adversary map per sample "
            "during --frozen-adversary-epochs, e.g. 2 means T_omega(T_omega(x))."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-augment", dest="augment_train", action="store_false")
    parser.set_defaults(augment_train=True)
    parser.add_argument("--lr-theta", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--freeze-batchnorm",
        dest="freeze_batchnorm",
        action="store_true",
        help=(
            "Keep BatchNorm layers in eval mode during classifier updates, "
            "preserving pretrained running statistics."
        ),
    )
    parser.add_argument(
        "--no-freeze-batchnorm",
        dest="freeze_batchnorm",
        action="store_false",
        help="Allow BatchNorm running statistics to update during classifier training.",
    )
    parser.set_defaults(freeze_batchnorm=False)
    parser.add_argument(
        "--freeze-batchnorm-affine",
        dest="freeze_batchnorm_affine",
        action="store_true",
        help=(
            "Also freeze BatchNorm affine weight/bias parameters so pretrained "
            "normalization is fixed apart from optional running-stat "
            "recalibration."
        ),
    )
    parser.add_argument(
        "--no-freeze-batchnorm-affine",
        dest="freeze_batchnorm_affine",
        action="store_false",
        help="Allow BatchNorm affine weight/bias parameters to train.",
    )
    parser.set_defaults(freeze_batchnorm_affine=False)
    parser.add_argument(
        "--adversary-classifier-eval",
        dest="adversary_classifier_eval",
        action="store_true",
        help=(
            "Put the classifier in eval mode for adversary-side forwards "
            "(clean-correct mask and inner maximization). This preserves "
            "BatchNorm running statistics during the adversary objective."
        ),
    )
    parser.add_argument(
        "--adversary-classifier-train",
        dest="adversary_classifier_eval",
        action="store_false",
        help=(
            "Legacy stable input-ICNN behavior: freeze classifier parameter "
            "gradients during adversary-side forwards but preserve train/eval "
            "mode, allowing BatchNorm running statistics to update in "
            "adversarial epochs. Warmup still forces eval mode."
        ),
    )
    parser.set_defaults(adversary_classifier_eval=True)
    parser.add_argument(
        "--online-batchnorm-refresh",
        dest="online_batchnorm_refresh",
        action="store_true",
        help=(
            "Before each classifier-updating training batch, run a no-gradient "
            "clean forward that updates only BatchNorm running statistics. "
            "When --freeze-batchnorm is also set, the subsequent adversary and "
            "classifier updates still keep BatchNorm layers in eval mode; only "
            "this clean refresh pass mutates running statistics."
        ),
    )
    parser.add_argument(
        "--no-online-batchnorm-refresh",
        dest="online_batchnorm_refresh",
        action="store_false",
        help="Disable per-batch clean BatchNorm running-stat refresh.",
    )
    parser.set_defaults(online_batchnorm_refresh=False)
    parser.add_argument(
        "--batchnorm-online-refresh-momentum",
        type=float,
        default=None,
        help=(
            "Optional BN momentum used only for the per-batch clean online "
            "refresh. Omit to keep each BatchNorm module's configured momentum."
        ),
    )
    parser.add_argument(
        "--recalibrate-batchnorm",
        dest="recalibrate_batchnorm",
        action="store_true",
        help=(
            "After each classifier-updating epoch, update BatchNorm running "
            "statistics in a no-gradient pass over clean training data before "
            "evaluation/checkpointing."
        ),
    )
    parser.add_argument(
        "--no-recalibrate-batchnorm",
        dest="recalibrate_batchnorm",
        action="store_false",
    )
    parser.set_defaults(recalibrate_batchnorm=False)
    parser.add_argument(
        "--batchnorm-recalibration-batches",
        type=int,
        default=0,
        help="Number of clean train batches for BN recalibration; 0 means one full pass.",
    )
    parser.add_argument(
        "--batchnorm-recalibration-reset",
        dest="batchnorm_recalibration_reset",
        action="store_true",
        help=(
            "Reset BatchNorm running statistics before the clean recalibration "
            "pass. This gives exact fresh stats when momentum is omitted."
        ),
    )
    parser.add_argument(
        "--no-batchnorm-recalibration-reset",
        dest="batchnorm_recalibration_reset",
        action="store_false",
        help=(
            "Keep existing BatchNorm running statistics and refresh them during "
            "the clean recalibration pass."
        ),
    )
    parser.set_defaults(batchnorm_recalibration_reset=True)
    parser.add_argument(
        "--batchnorm-recalibration-momentum",
        type=float,
        default=None,
        help=(
            "Optional BN momentum used only during recalibration. Omit for "
            "PyTorch's cumulative average; use a small value such as 0.001 "
            "with --no-batchnorm-recalibration-reset for a gentle refresh."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)

    # λ
    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=30.0,
        help="DRO penalty λ multiplying ||T(x) - x||^2.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Alternative parameterisation: λ = 1/(2τ). Overrides --penalty-lambda when both are set.",
    )
    parser.add_argument(
        "--lambda-schedule",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Piecewise-constant lambda schedule for adversarial epochs, e.g. "
            "--lambda-schedule 5 10 15 20 30 --lambda-stage-epochs 10. "
            "Runs in one continuous training process."
        ),
    )
    parser.add_argument(
        "--lambda-stage-epochs",
        type=int,
        default=0,
        help=(
            "Number of adversarial epochs per lambda in --lambda-schedule. "
            "If omitted, epochs_adv must divide evenly by len(lambda_schedule)."
        ),
    )

    # Inner-loop budget
    parser.add_argument("--omega-steps-per-batch", type=int, default=10)
    parser.add_argument(
        "--npf-inner-optimizer",
        type=str,
        default="bb_armijo",
        choices=["bb_armijo", "muon"],
        help=(
            "Optimizer for the NPF/NPF-LastQuad adversary inner maximization. "
            "'bb_armijo' preserves the existing Barzilai-Borwein + Armijo "
            "line search; 'muon' uses Muon on matrix parameters with the "
            "configured fallback for vector parameters."
        ),
    )
    parser.add_argument(
        "--npf-reset-omega-each-batch",
        dest="npf_reset_omega_each_batch",
        action="store_true",
        help=(
            "Reset NPF/NPF-LastQuad omega to its identity-adjacent "
            "initialization before each batch inner maximization. This makes "
            "the learned transport attack behave like a fresh per-batch "
            "inner optimizer instead of a persistent global map."
        ),
    )
    parser.add_argument(
        "--no-npf-reset-omega-each-batch",
        dest="npf_reset_omega_each_batch",
        action="store_false",
        help="Keep NPF/NPF-LastQuad omega persistent across batches.",
    )
    parser.set_defaults(npf_reset_omega_each_batch=False)
    parser.add_argument(
        "--npf-project-to-input-ball",
        dest="npf_project_to_input_ball",
        action="store_true",
        help=(
            "Project NPF/NPF-LastQuad transports into the configured input-PGD "
            "threat set before the inner objective and classifier update. "
            "For pixel_l2_squared PGD this enforces pixel-space ||delta||_p <= inp_eps."
        ),
    )
    parser.add_argument(
        "--no-npf-project-to-input-ball",
        dest="npf_project_to_input_ball",
        action="store_false",
        help="Leave NPF/NPF-LastQuad transports unconstrained except for image box clamping.",
    )
    parser.set_defaults(npf_project_to_input_ball=False)
    parser.add_argument(
        "--npf-omega-weight-decay",
        type=float,
        default=1e-4,
        help=(
            "L2 decay on the FREE omega kernels (affine input skips, PosDef "
            "lin/quad) inside the ascent objective. Legacy golden-run parity "
            "is 1e-4; 0 disables."
        ),
    )
    parser.add_argument(
        "--npf-pgd-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an optional per-batch PGD anchor term in the NPF "
            "omega objective. 0 disables it. A positive value encourages "
            "T_omega(x) to align with a local PGD maximizer."
        ),
    )
    parser.add_argument(
        "--npf-pgd-anchor-steps",
        type=int,
        default=5,
        help="PGD steps used for the optional NPF anchor target.",
    )
    parser.add_argument(
        "--npf-pgd-anchor-restarts",
        type=int,
        default=1,
        help="PGD restarts used for the optional NPF anchor target.",
    )
    parser.add_argument(
        "--npf-pgd-anchor-loss",
        type=str,
        default="margin",
        choices=["margin", "ce"],
        help="Loss maximized by the optional NPF PGD anchor target.",
    )
    # Shared BB+Armijo step rule (DRO inner ascent only; Madry/RO uses PGD).
    bb = parser.add_argument_group("bb_armijo (shared step rule)")
    bb.add_argument("--bb-alpha0", type=float, default=5e-4)
    bb.add_argument("--bb-alpha-min", type=float, default=1e-6)
    bb.add_argument("--bb-alpha-max", type=float, default=1.0)
    bb.add_argument("--bb-ls-c", type=float, default=0.1)
    bb.add_argument("--bb-ls-shrink", type=float, default=0.5)
    bb.add_argument("--bb-ls-max-steps", type=int, default=10)
    bb.add_argument(
        "--parametric-bb-max-grad-norm",
        type=float,
        default=1.0,
        help=(
            "Global norm clip applied to shared parametric adversary gradients "
            "before BB+Armijo. Matches legacy ICNN max_norm=1.0; set 0 to "
            "disable for ablations."
        ),
    )
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
    parser.set_defaults(use_margin_loss=True)
    parser.add_argument(
        "--attack-clean-correct-only",
        dest="attack_clean_correct_only",
        action="store_true",
        help=(
            "Match legacy pretrained_INPUT_icnn.py: train/apply the transport "
            "adversary only on clean-correct examples; clean-misclassified "
            "examples remain untransported in the outer update."
        ),
    )
    parser.add_argument(
        "--attack-all-samples",
        dest="attack_clean_correct_only",
        action="store_false",
        help="Ablation: train/apply the transport adversary on every sample.",
    )
    parser.set_defaults(attack_clean_correct_only=True)
    parser.add_argument(
        "--reset-parametric-bb-each-batch",
        dest="reset_parametric_bb_each_batch",
        action="store_true",
        help=(
            "Reset BB+Armijo history each batch for parametric adversaries "
            "(legacy input-ICNN behavior; default)."
        ),
    )
    parser.add_argument(
        "--persistent-parametric-bb",
        dest="reset_parametric_bb_each_batch",
        action="store_false",
        help="Ablation: carry BB+Armijo secant history across batches.",
    )
    parser.set_defaults(reset_parametric_bb_each_batch=True)
    parser.add_argument(
        "--transport-cost",
        type=str,
        default="normalized_mse",
        choices=["normalized_mse", "pixel_l2_squared"],
        help=(
            "Transport penalty convention for DRO objectives. normalized_mse "
            "matches legacy pretrained_INPUT_icnn.py; pixel_l2_squared uses "
            "standard CIFAR pixel-coordinate squared L2."
        ),
    )

    # NPF
    npf = parser.add_argument_group("npf")
    npf.add_argument("--npf-hidden", type=int, nargs="+", default=[1024, 512, 512, 256, 128, 64])
    npf.add_argument("--npf-outer-rank", type=int, default=1)
    npf.add_argument("--npf-inner-rank", type=int, default=1)
    npf.add_argument(
        "--npf-activation", type=str, default="softplus",
        choices=["elu", "softplus", "relu"],
    )
    npf.add_argument("--npf-elu-alpha", type=float, default=1.0)
    npf.add_argument("--npf-softplus-beta", type=float, default=20.0)
    npf.add_argument(
        "--npf-init-eps",
        type=float,
        default=1e-2,
        help=(
            "Backward-compatible metadata for older NPF configs. The faithful "
            "OTT initializer does not use this value unless identity init is enabled."
        ),
    )
    npf.add_argument(
        "--npf-identity-init",
        dest="npf_identity_init",
        action="store_true",
        help=(
            "Apply exact identity initialization after constructing the OTT-style "
            "potential: zero the residual ICNN and keep only the final "
            "0.5 * strong_convexity * ||x||^2 potential."
        ),
    )
    npf.add_argument(
        "--no-npf-identity-init",
        dest="npf_identity_init",
        action="store_false",
        help=(
            "Use the faithful OTT constructor initialization instead of zeroing "
            "the residual ICNN."
        ),
    )
    parser.set_defaults(npf_identity_init=True)
    npf.add_argument("--npf-strong-convexity", type=float, default=1.0)
    npf.add_argument(
        "--npf-pos-weights",
        dest="npf_pos_weights",
        action="store_true",
        help=(
            "Project hidden-to-hidden NPF weights through a nonnegative "
            "rectifier. This is the safe default because the trainer has no "
            "separate weight-clipping pass."
        ),
    )
    npf.add_argument(
        "--no-npf-pos-weights",
        dest="npf_pos_weights",
        action="store_false",
        help=(
            "Match OTT's unprojected Dense option. Use only with an external "
            "nonnegative-weight clipping/regularization path."
        ),
    )
    parser.set_defaults(npf_pos_weights=True)
    npf.add_argument(
        "--npf-positive-weight-rectifier",
        type=str,
        default="exp",
        choices=["relu", "softplus", "exp"],
        help=(
            "Reparametrization for nonnegative hidden weights. 'exp' matches "
            "the legacy pretrained_INPUT_icnn NonNegativeLinear exactly "
            "(strictly positive, never-dead gradients)."
        ),
    )
    npf.add_argument("--npf-bb-alpha0", type=float, default=2e-4)
    npf.add_argument("--npf-bb-alpha-min", type=float, default=1e-7)
    npf.add_argument("--npf-bb-alpha-max", type=float, default=0.25)
    npf.add_argument("--npf-bb-ls-c", type=float, default=1e-4)
    npf.add_argument("--npf-bb-ls-shrink", type=float, default=0.5)
    npf.add_argument("--npf-bb-ls-max-steps", type=int, default=15)
    npf.add_argument("--npf-muon-lr", type=float, default=2e-4)
    npf.add_argument("--npf-muon-momentum", type=float, default=0.95)
    npf.add_argument(
        "--npf-muon-nesterov",
        dest="npf_muon_nesterov",
        action="store_true",
        help="Use Nesterov-style momentum in the NPF Muon inner optimizer.",
    )
    npf.add_argument(
        "--no-npf-muon-nesterov",
        dest="npf_muon_nesterov",
        action="store_false",
    )
    parser.set_defaults(npf_muon_nesterov=True)
    npf.add_argument("--npf-muon-ns-steps", type=int, default=5)
    npf.add_argument(
        "--npf-muon-matrix-lr-scale",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help=(
            "Muon matrix learning-rate scaling. 'auto' applies the common "
            "sqrt(max(1, rows / cols)) factor after flattening tensors to "
            "(shape[0], -1); 'none' uses exactly --npf-muon-lr."
        ),
    )
    npf.add_argument("--npf-muon-weight-decay", type=float, default=0.0)
    npf.add_argument(
        "--npf-muon-fallback",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "none"],
        help=(
            "Update rule for NPF vector/scalar parameters when "
            "--npf-inner-optimizer=muon. Muon itself is used only for "
            "matrix-like tensors."
        ),
    )
    npf.add_argument(
        "--npf-muon-fallback-lr",
        type=float,
        default=0.0,
        help="Fallback lr for vector parameters. 0 reuses --npf-muon-lr.",
    )
    npf.add_argument("--npf-muon-fallback-weight-decay", type=float, default=0.0)
    npf.add_argument("--npf-muon-adam-beta1", type=float, default=0.9)
    npf.add_argument("--npf-muon-adam-beta2", type=float, default=0.999)
    npf.add_argument("--npf-muon-adam-eps", type=float, default=1e-8)
    npf.add_argument(
        "--npf-muon-max-grad-norm",
        type=float,
        default=0.0,
        help="Global norm clip for NPF Muon gradients. 0 disables clipping.",
    )

    # NPF last-quadratic-only variant
    npf_lq = parser.add_argument_group("npf_lastquad")
    npf_lq.add_argument(
        "--npf-lastquad-hidden",
        type=int,
        nargs="+",
        default=[1024, 512, 512, 256, 128, 64],
    )
    npf_lq.add_argument(
        "--npf-lastquad-activation",
        type=str,
        default="softplus",
        choices=["elu", "softplus", "relu"],
    )
    npf_lq.add_argument("--npf-lastquad-elu-alpha", type=float, default=1.0)
    npf_lq.add_argument(
        "--npf-lastquad-output-rank",
        type=int,
        default=64,
        help=(
            "Rank of the learnable low-rank quadratic factor in the final "
            "LastQuad output block. 0 keeps the diagonal-only LastQuad; the "
            "measured default 64 is what gives the ascent enough "
            "sample-dependent directions (rank 2 collapses to shared "
            "far-field deltas and the classifier neutralizes them)."
        ),
    )
    npf_lq.add_argument("--npf-lastquad-softplus-beta", type=float, default=20.0)
    npf_lq.add_argument(
        "--npf-lastquad-init-eps",
        type=float,
        default=1e-2,
        help=(
            "Residual ICNN scale used when --npf-lastquad-identity-init is "
            "enabled. Values that are too small, especially with a softplus "
            "positive-weight rectifier, can saturate the residual path and "
            "leave LastQuad almost exactly at T(x)=x."
        ),
    )
    npf_lq.add_argument(
        "--npf-lastquad-identity-init",
        dest="npf_lastquad_identity_init",
        action="store_true",
        help=(
            "Apply exact identity initialization after constructing LastQuad: "
            "zero the residual ICNN and keep only the final identity "
            "positive-definite potential."
        ),
    )
    npf_lq.add_argument(
        "--no-npf-lastquad-identity-init",
        dest="npf_lastquad_identity_init",
        action="store_false",
        help=(
            "Use the faithful OTT-style constructor initialization for LastQuad."
        ),
    )
    parser.set_defaults(npf_lastquad_identity_init=False)
    npf_lq.add_argument("--npf-lastquad-strong-convexity", type=float, default=1.0)
    npf_lq.add_argument(
        "--npf-lastquad-pos-weights",
        dest="npf_lastquad_pos_weights",
        action="store_true",
        help="Project LastQuad hidden-to-hidden weights through a nonnegative rectifier.",
    )
    npf_lq.add_argument(
        "--no-npf-lastquad-pos-weights",
        dest="npf_lastquad_pos_weights",
        action="store_false",
        help="Use unprojected hidden-to-hidden Dense layers in LastQuad.",
    )
    parser.set_defaults(npf_lastquad_pos_weights=True)
    npf_lq.add_argument(
        "--npf-lastquad-positive-weight-rectifier",
        type=str,
        default="exp",
        choices=["relu", "softplus", "exp"],
        help=(
            "Reparametrization for nonnegative hidden weights. 'exp' matches "
            "the legacy pretrained_INPUT_icnn NonNegativeLinear exactly "
            "(strictly positive, never-dead gradients)."
        ),
    )

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
    wrm.add_argument(
        "--wrm-step-rule",
        type=str,
        default="const_lr",
        choices=["const_lr", "bb_armijo"],
        help=(
            "const_lr = per-sample ascent on the WRM diminishing schedule "
            "wrm-inner-lr/sqrt(s), byte-identical to MPA's round-0 burst "
            "(WRM == MPA with R=1). bb_armijo = legacy shared line-searched "
            "rule on the batch-mean objective."
        ),
    )
    wrm.add_argument(
        "--wrm-inner-steps",
        type=int,
        default=100,
        help="K: number of gradient-ascent steps per batch.",
    )
    wrm.add_argument(
        "--wrm-inner-lr",
        type=float,
        default=0.03,
        help=(
            "const_lr rule only: base LR for the diminishing schedule "
            "lr/sqrt(s). Calibrated like MPA's --ppa-round0-lr on the "
            "current R2.pth."
        ),
    )

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
    ppa.add_argument(
        "--ppa-step-rule",
        type=str,
        default="bb_armijo",
        choices=["bb_armijo", "const_lr"],
        help=(
            "Ascent step rule for MPA rounds. bb_armijo = shared line-searched "
            "step on the batch-mean objective; const_lr = per-sample "
            "MNIST-style ascent (round 0: ppa-round0-lr/sqrt(s) diminishing, "
            "refinement: constant ppa-refine-lr)."
        ),
    )
    ppa.add_argument(
        "--ppa-round-order",
        type=str,
        default="ascent_first",
        choices=["ascent_first", "reassign_first", "reassign_first_v2"],
        help=(
            "Within-round order for MPA. ascent_first = Algorithm 1 rounds "
            "of {K ascent steps; reassign}. reassign_first = mirrored rounds "
            "of {reassign; K ascent steps} closed by a final reassignment "
            "(R=3: R-A-R-A-R-A-R); its round-0 reassignment at z=x starts "
            "each ascent from the best same-class training sample. "
            "reassign_first_v2 = same, but the round-0 reassignment is "
            "priced at lambda/--ppa-round0-lambda-damping so confidence-"
            "driven multi-start jumps activate at z=x; the damping affects "
            "ONLY this order. All other hyperparameters are shared."
        ),
    )
    ppa.add_argument(
        "--ppa-round0-lambda-damping",
        type=float,
        default=6.0,
        help=(
            "alpha for reassign_first_v2 ONLY (ascent_first/reassign_first "
            "ignore it): the round-0 reassignment scores candidates with "
            "lambda/alpha while everything else keeps the full lambda. "
            "1.0 = identical to reassign_first. On the current R2.pth at "
            "lambda=30: alpha=6 activates ~53%% of round-0 jumps, "
            "alpha=10 ~73%%."
        ),
    )
    ppa.add_argument("--ppa-num-rounds", type=int, default=5)
    ppa.add_argument("--ppa-min-rounds", type=int, default=2)
    ppa.add_argument("--ppa-round0-steps", type=int, default=30)
    ppa.add_argument(
        "--ppa-round0-lr",
        type=float,
        default=0.03,
        help=(
            "const_lr rule only: round-0 base LR for the WRM diminishing "
            "schedule lr/sqrt(s). Calibrated for margin+lambda=30 nmse on "
            "the current 95.3%%-clean R2.pth (old R2_Old.pth wanted 0.1)."
        ),
    )
    ppa.add_argument("--ppa-refine-steps", type=int, default=15)
    ppa.add_argument(
        "--ppa-refine-lr",
        type=float,
        default=0.015,
        help="const_lr rule only: constant LR for refinement-round ascent.",
    )
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
    parser.add_argument(
        "--profile-inner",
        action="store_true",
        help=(
            "Record synchronized part-by-part timings for inner maximization. "
            "Adds overhead; intended for NPF/WRM ablations."
        ),
    )
    parser.add_argument(
        "--profile-inner-batches",
        type=int,
        default=0,
        help="Number of train batches per epoch to profile. 0 profiles every batch.",
    )

    # Evaluation
    parser.add_argument("--eval-input-pgd", dest="eval_input_pgd", action="store_true")
    parser.add_argument("--no-eval-input-pgd", dest="eval_input_pgd", action="store_false")
    parser.set_defaults(eval_input_pgd=True)
    parser.add_argument("--eval-input-pgd-samples", type=int, default=1024)
    parser.add_argument(
        "--eval-transport-samples",
        type=int,
        default=0,
        help=(
            "Cap on test samples for the per-epoch transport-adversary "
            "evaluation. 0 evaluates the full test set (legacy behavior; "
            "fine for cheap learned maps). Use ~1024 for transductive "
            "attacks like new_ppa where the eval transport is a full inner "
            "maximization."
        ),
    )
    parser.add_argument(
        "--eval-transport-pgd-alignment",
        dest="eval_transport_pgd_alignment",
        action="store_true",
        help=(
            "Run a small diagnostic PGD attack and log cosine alignment between "
            "the learned transport delta and PGD delta."
        ),
    )
    parser.add_argument(
        "--no-eval-transport-pgd-alignment",
        dest="eval_transport_pgd_alignment",
        action="store_false",
        help="Disable the transport-vs-PGD direction alignment diagnostic.",
    )
    parser.set_defaults(eval_transport_pgd_alignment=False)
    parser.add_argument("--eval-transport-pgd-alignment-samples", type=int, default=256)
    parser.add_argument(
        "--input-pgd-loss",
        type=str,
        default="ce",
        choices=["margin", "ce"],
        help=(
            "Loss maximized by input-space PGD evaluation. The default "
            "logsumexp margin avoids CE saturation artifacts; pass ce only "
            "for legacy comparisons."
        ),
    )
    parser.add_argument(
        "--input-pgd-geometry",
        type=str,
        default="pixel_l2_squared",
        choices=["pixel_l2_squared", "normalized_mse"],
        help=(
            "Geometry used by input-space PGD evaluation. pixel_l2_squared "
            "is the standard CIFAR pixel-space L2/Linf threat model where "
            "--inp-eps is a norm radius. normalized_mse runs L2 PGD in "
            "normalized CIFAR coordinates and interprets --inp-eps as the "
            "budget on mean((x_adv_norm - x_norm)^2)."
        ),
    )
    parser.add_argument("--inp-p", type=str, default="2", choices=["2", "inf"])
    parser.add_argument("--inp-eps", type=float, default=0.5)
    parser.add_argument("--inp-steps", type=int, default=20)
    parser.add_argument("--inp-step-size", type=float, default=0.0)
    parser.add_argument("--inp-restarts", type=int, default=5)
    parser.add_argument(
        "--wrm-normalized-radius-protocol",
        type=str,
        default="",
        help=(
            "Metadata for WRM-style PGD reporting. When set by a launcher, "
            "--inp-eps is the raw PGD radius obtained from rho*C_p."
        ),
    )
    parser.add_argument("--wrm-pgd-eps-mode", type=str, default="")
    parser.add_argument("--wrm-normalized-radius-requested", type=float, default=None)
    parser.add_argument("--wrm-normalized-radius", type=float, default=None)
    parser.add_argument("--wrm-cp", type=float, default=None)
    parser.add_argument("--wrm-effective-inp-eps", type=float, default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    if args.tau is not None:
        lambda_param = 1.0 / (2.0 * float(args.tau))
    elif args.penalty_lambda is not None:
        lambda_param = float(args.penalty_lambda)
    else:
        lambda_param = TrainConfig.lambda_param

    lambda_schedule = tuple(float(x) for x in (args.lambda_schedule or ()))
    lambda_stage_epochs = int(args.lambda_stage_epochs or 0)
    if int(args.batchnorm_recalibration_batches) < 0:
        raise ValueError("--batchnorm-recalibration-batches must be non-negative.")
    frozen_adversary_epochs = int(args.frozen_adversary_epochs)
    frozen_adversary_map_steps = int(args.frozen_adversary_map_steps)
    if frozen_adversary_epochs < 0:
        raise ValueError("--frozen-adversary-epochs must be non-negative.")
    if frozen_adversary_epochs > 0 and frozen_adversary_map_steps < 1:
        raise ValueError(
            "--frozen-adversary-map-steps must be at least 1 when "
            "--frozen-adversary-epochs is positive."
        )
    if frozen_adversary_epochs == 0 and frozen_adversary_map_steps < 1:
        # The map-step count is unused when the frozen-adversary phase is
        # disabled, so tolerate legacy launch commands that set both to 0.
        args.frozen_adversary_map_steps = 1
    if args.batchnorm_recalibration_momentum is not None:
        bn_momentum = float(args.batchnorm_recalibration_momentum)
        if not math.isfinite(bn_momentum) or bn_momentum < 0.0 or bn_momentum > 1.0:
            raise ValueError(
                "--batchnorm-recalibration-momentum must be finite and in [0, 1]."
            )
    if args.batchnorm_online_refresh_momentum is not None:
        online_bn_momentum = float(args.batchnorm_online_refresh_momentum)
        if (
            not math.isfinite(online_bn_momentum)
            or online_bn_momentum < 0.0
            or online_bn_momentum > 1.0
        ):
            raise ValueError(
                "--batchnorm-online-refresh-momentum must be finite and in [0, 1]."
            )
    if args.input_pgd_geometry == "normalized_mse" and args.inp_p != "2":
        raise ValueError("--input-pgd-geometry normalized_mse requires --inp-p 2.")
    if not math.isfinite(float(args.inp_eps)) or float(args.inp_eps) < 0.0:
        raise ValueError("--inp-eps must be non-negative and finite.")
    if not math.isfinite(lambda_param) or lambda_param <= 0.0:
        raise ValueError(f"lambda_param must be positive and finite, got {lambda_param}.")
    if lambda_schedule:
        bad_lambdas = [x for x in lambda_schedule if not math.isfinite(x) or x <= 0.0]
        if bad_lambdas:
            raise ValueError(
                "All values in --lambda-schedule must be positive and finite; "
                f"got {bad_lambdas}."
            )
        if lambda_stage_epochs <= 0:
            if int(args.epochs_adv) % len(lambda_schedule) != 0:
                raise ValueError(
                    "--lambda-stage-epochs is required when --epochs-adv is not "
                    "divisible by len(--lambda-schedule)."
                )
            lambda_stage_epochs = int(args.epochs_adv) // len(lambda_schedule)
        expected_epochs = len(lambda_schedule) * lambda_stage_epochs
        if int(args.epochs_adv) != expected_epochs:
            raise ValueError(
                "Lambda schedule length mismatch: expected "
                f"epochs_adv={expected_epochs} from len(lambda_schedule)="
                f"{len(lambda_schedule)} and lambda_stage_epochs={lambda_stage_epochs}, "
                f"but got epochs_adv={args.epochs_adv}."
            )
        lambda_param = float(lambda_schedule[0])

    field_names = {f.name for f in fields(TrainConfig)}
    kwargs: Dict[str, Any] = {}
    cli_to_field = {
        "data_dir": "data_dir",
        "pretrained_path": "pretrained_path",
        "pretrained_strict": "pretrained_strict",
        "save": "save",
        "save_best_robust": "save_best_robust",
        "resume_checkpoint": "resume_checkpoint",
        "save_every_epoch_dir": "save_every_epoch_dir",
        "checkpoint_epoch_offset": "checkpoint_epoch_offset",
        "log_csv": "log_csv",
        "algorithm": "algorithm",
        "epochs_adv": "epochs_adv",
        "epochs_icnn_pretrain": "epochs_icnn_pretrain",
        "frozen_adversary_epochs": "frozen_adversary_epochs",
        "frozen_adversary_map_steps": "frozen_adversary_map_steps",
        "benchmark_mode": "benchmark_mode",
        "skip_pgd_during_train": "skip_pgd_during_train",
        "profile_inner": "profile_inner",
        "profile_inner_batches": "profile_inner_batches",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "augment_train": "augment_train",
        "lr_theta": "lr_theta",
        "momentum": "momentum",
        "weight_decay": "weight_decay",
        "freeze_batchnorm": "freeze_batchnorm",
        "freeze_batchnorm_affine": "freeze_batchnorm_affine",
        "adversary_classifier_eval": "adversary_classifier_eval",
        "online_batchnorm_refresh": "online_batchnorm_refresh",
        "batchnorm_online_refresh_momentum": "batchnorm_online_refresh_momentum",
        "recalibrate_batchnorm": "recalibrate_batchnorm",
        "batchnorm_recalibration_batches": "batchnorm_recalibration_batches",
        "batchnorm_recalibration_reset": "batchnorm_recalibration_reset",
        "batchnorm_recalibration_momentum": "batchnorm_recalibration_momentum",
        "seed": "seed",
        "omega_steps_per_batch": "omega_steps_per_batch",
        "npf_inner_optimizer": "npf_inner_optimizer",
        "npf_reset_omega_each_batch": "npf_reset_omega_each_batch",
        "npf_project_to_input_ball": "npf_project_to_input_ball",
        "npf_omega_weight_decay": "npf_omega_weight_decay",
        "npf_pgd_anchor_weight": "npf_pgd_anchor_weight",
        "npf_pgd_anchor_steps": "npf_pgd_anchor_steps",
        "npf_pgd_anchor_restarts": "npf_pgd_anchor_restarts",
        "npf_pgd_anchor_loss": "npf_pgd_anchor_loss",
        "bb_alpha0": "bb_alpha0",
        "bb_alpha_min": "bb_alpha_min",
        "bb_alpha_max": "bb_alpha_max",
        "bb_ls_c": "bb_ls_c",
        "bb_ls_shrink": "bb_ls_shrink",
        "bb_ls_max_steps": "bb_ls_max_steps",
        "parametric_bb_max_grad_norm": "parametric_bb_max_grad_norm",
        "use_margin_loss": "use_margin_loss",
        "attack_clean_correct_only": "attack_clean_correct_only",
        "reset_parametric_bb_each_batch": "reset_parametric_bb_each_batch",
        "transport_cost": "transport_cost",
        "lambda_stage_epochs": "lambda_stage_epochs",
        "npf_hidden": "npf_hidden",
        "npf_outer_rank": "npf_outer_rank",
        "npf_inner_rank": "npf_inner_rank",
        "npf_activation": "npf_activation",
        "npf_elu_alpha": "npf_elu_alpha",
        "npf_softplus_beta": "npf_softplus_beta",
        "npf_init_eps": "npf_init_eps",
        "npf_identity_init": "npf_identity_init",
        "npf_strong_convexity": "npf_strong_convexity",
        "npf_pos_weights": "npf_pos_weights",
        "npf_positive_weight_rectifier": "npf_positive_weight_rectifier",
        "npf_bb_alpha0": "npf_bb_alpha0",
        "npf_bb_alpha_min": "npf_bb_alpha_min",
        "npf_bb_alpha_max": "npf_bb_alpha_max",
        "npf_bb_ls_c": "npf_bb_ls_c",
        "npf_bb_ls_shrink": "npf_bb_ls_shrink",
        "npf_bb_ls_max_steps": "npf_bb_ls_max_steps",
        "npf_muon_lr": "npf_muon_lr",
        "npf_muon_momentum": "npf_muon_momentum",
        "npf_muon_nesterov": "npf_muon_nesterov",
        "npf_muon_ns_steps": "npf_muon_ns_steps",
        "npf_muon_matrix_lr_scale": "npf_muon_matrix_lr_scale",
        "npf_muon_weight_decay": "npf_muon_weight_decay",
        "npf_muon_fallback": "npf_muon_fallback",
        "npf_muon_fallback_lr": "npf_muon_fallback_lr",
        "npf_muon_fallback_weight_decay": "npf_muon_fallback_weight_decay",
        "npf_muon_adam_beta1": "npf_muon_adam_beta1",
        "npf_muon_adam_beta2": "npf_muon_adam_beta2",
        "npf_muon_adam_eps": "npf_muon_adam_eps",
        "npf_muon_max_grad_norm": "npf_muon_max_grad_norm",
        "npf_lastquad_hidden": "npf_lastquad_hidden",
        "npf_lastquad_output_rank": "npf_lastquad_output_rank",
        "npf_lastquad_activation": "npf_lastquad_activation",
        "npf_lastquad_elu_alpha": "npf_lastquad_elu_alpha",
        "npf_lastquad_softplus_beta": "npf_lastquad_softplus_beta",
        "npf_lastquad_init_eps": "npf_lastquad_init_eps",
        "npf_lastquad_identity_init": "npf_lastquad_identity_init",
        "npf_lastquad_strong_convexity": "npf_lastquad_strong_convexity",
        "npf_lastquad_pos_weights": "npf_lastquad_pos_weights",
        "npf_lastquad_positive_weight_rectifier": "npf_lastquad_positive_weight_rectifier",
        "nn_dro_hidden": "nn_dro_hidden",
        "nn_dro_activation": "nn_dro_activation",
        "nn_dro_softplus_beta": "nn_dro_softplus_beta",
        "nn_dro_init_scale": "nn_dro_init_scale",
        "nn_dro_inner_lr": "nn_dro_inner_lr",
        "madry_epsilon": "madry_epsilon",
        "madry_pgd_steps": "madry_pgd_steps",
        "madry_pgd_step_size": "madry_pgd_step_size",
        "madry_pgd_restarts": "madry_pgd_restarts",
        "wrm_step_rule": "wrm_step_rule",
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
        "ppa_step_rule": "ppa_step_rule",
        "ppa_round_order": "ppa_round_order",
        "ppa_round0_lambda_damping": "ppa_round0_lambda_damping",
        "ppa_num_rounds": "ppa_num_rounds",
        "ppa_min_rounds": "ppa_min_rounds",
        "ppa_round0_steps": "ppa_round0_steps",
        "ppa_round0_lr": "ppa_round0_lr",
        "ppa_refine_steps": "ppa_refine_steps",
        "ppa_refine_lr": "ppa_refine_lr",
        "ppa_gain_rtol": "ppa_gain_rtol",
        "eval_input_pgd": "eval_input_pgd",
        "eval_input_pgd_samples": "eval_input_pgd_samples",
        "eval_transport_samples": "eval_transport_samples",
        "eval_transport_pgd_alignment": "eval_transport_pgd_alignment",
        "eval_transport_pgd_alignment_samples": "eval_transport_pgd_alignment_samples",
        "input_pgd_loss": "input_pgd_loss",
        "input_pgd_geometry": "input_pgd_geometry",
        "inp_p": "inp_p",
        "inp_eps": "inp_eps",
        "inp_steps": "inp_steps",
        "inp_step_size": "inp_step_size",
        "inp_restarts": "inp_restarts",
        "wrm_normalized_radius_protocol": "wrm_normalized_radius_protocol",
        "wrm_pgd_eps_mode": "wrm_pgd_eps_mode",
        "wrm_normalized_radius_requested": "wrm_normalized_radius_requested",
        "wrm_normalized_radius": "wrm_normalized_radius",
        "wrm_cp": "wrm_cp",
        "wrm_effective_inp_eps": "wrm_effective_inp_eps",
    }
    for cli, dest in cli_to_field.items():
        if not hasattr(args, cli):
            continue
        val = getattr(args, cli)
        if dest in {"npf_hidden", "npf_lastquad_hidden", "nn_dro_hidden"}:
            val = tuple(val)
        kwargs[dest] = val

    kwargs["lambda_param"] = lambda_param
    kwargs["lambda_schedule"] = lambda_schedule
    kwargs["lambda_stage_epochs"] = lambda_stage_epochs
    # Drop any kwargs not known to TrainConfig (defensive against future drift).
    kwargs = {k: v for k, v in kwargs.items() if k in field_names}
    return TrainConfig(**kwargs)
