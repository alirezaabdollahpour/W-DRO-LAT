#!/bin/bash
# =============================================================================
# Input-ICNN Runtime Sweep — DDP edition
# =============================================================================
#
# Multi-GPU DDP launcher for the pretrained input-ICNN adversarial
# training experiments. By default this script follows
# run_pretrained_input_icnn.sh and runs only NPF-LastQuad on CIFAR-10,
# distributed across NPROC GPUs via torchrun + DistributedDataParallel.
#
# Adversaries:
#   * NPF      — NPF ICNN potential, selectable BB+Armijo or Muon on
#                ω parameters
#   * NPF-LastQuad — NPF variant with only the final rank-0 diagonal
#                quadratic layer trainable; uses the same NPF optimizer
#                selector
#   * NN-DRO   — vanilla MLP adversary, BB+Armijo on ω parameters
#                (replaces the legacy Adam ascent)
#   * Madry/RO — fixed-step pixel-space l2-PGD threat model with
#                epsilon-ball projection + clamp; no lambda penalty
#   * WRM      — WRM inner objective on z, BB+Armijo step rule
#                (replaces constant-LR ascent)
#   * WFR      — particle ensemble + reweighting, BB+Armijo on the
#                deterministic gradient component, Langevin noise added
#                after (replaces constant-LR Langevin)
#   * New_PPA  — BB+Armijo ascent inside each round (replaces WRM ascent),
#                free-weight projection rounds unchanged
#   * Dual     — Sinkhorn entropic dual. Two operating modes:
#                  – DUAL_LANGEVIN_STEPS=0: one-shot Gaussian estimator
#                    (Gaussian particles around x). Fast but the
#                    samples come from the prior, not the Gibbs target,
#                    so the inner integral isn't actually solved.
#                  – DUAL_LANGEVIN_STEPS>0: Option-D rigorous inner
#                    sampling. Each particle is refined via a Langevin
#                    chain on the Gibbs target π(z|x) ∝ exp(U(z)),
#                    optionally MH-corrected (DUAL_MALA=1, default) for
#                    unbiased samples. The Sinkhorn dual loss is then
#                    computed on the refined particles. The one-shot
#                    estimator is recovered at K=0.
#
# All BB+Armijo hyperparameters (alpha0, alpha_min, alpha_max, ls_c,
# ls_shrink, ls_max_steps) are SHARED across the DRO inner-ascent
# methods that use BB. NPF / NPF-LastQuad can instead use Muon by setting
# NPF_INNER_OPTIMIZER=muon.
# Madry/RO is intentionally excluded: it is standard pixel-space l2-PGD with
# epsilon, not a lambda-penalised DRO objective.
#
# ---- Parallelism strategy ----
#
# Each algorithm uses DDP across NPROC GPUs (default 4). The classifier
# is wrapped in DDP; on classifier_update the gradient all-reduce fires
# automatically. Inner adversary loops use the unwrapped module so DDP's
# reducer state isn't corrupted by forward-without-backward.
#
# For the parametric adversaries (NPF, NN-DRO) the BB+Armijo step
# all-reduces ω-gradients across ranks so every rank picks the same
# Armijo trial step. For the input-space DRO adversaries (WRM/WFR/PPA)
# each rank runs an independent BB+Armijo ascent on its local batch
# shard — no cross-rank sync is meaningful since z lives on the shard.
#
# Per-rank batch_size = BATCH_SIZE / NPROC, so the GLOBAL effective batch
# matches the single-GPU configuration for fair throughput comparison.
#
# Other algorithms are kept below as commented reference for later runtime
# sweeps, but they are intentionally disabled for the current launch.
#
# ---- Run defaults baked in ----
#
#   * Same outer hyperparameters (epochs, lr_theta, λ for DRO methods,
#     global batch). DRO methods default to the legacy normalized-MSE
#     transport penalty from pretrained_INPUT_icnn.py. Madry/RO uses
#     epsilon instead of λ.
#   * Same K=20 inner-iteration budget (per-algorithm interpretation
#     applied: NPF/NN-DRO omega_steps=K, WRM/WFR inner_steps=K,
#     Madry/RO pgd_steps=K, New_PPA round0_steps=refine_steps=K, Dual
#     langevin_steps=K via the Option-D MALA sampler).
#   * Same BB+Armijo step rule and hyperparameters across the DRO
#     inner-ascent methods when they use BB. For NPF / NPF-LastQuad,
#     NPF_INNER_OPTIMIZER=muon swaps only the NPF ω-ascent rule. Madry/RO
#     uses pixel-space l2-PGD; Dual uses the Langevin/MALA sampler for its
#     entropic Gibbs target.
#   * Input PGD evaluation is enabled by default, matching
#     run_pretrained_input_icnn.sh. Set SKIP_PGD_DURING_TRAIN=1 only for
#     pure throughput timing.
#   * Optional frozen-adversary post phase: set FROZEN_ADVERSARY_EPOCHS>0
#     and FROZEN_ADVERSARY_MAP_STEPS=m to freeze the learned NPF/NN-DRO map
#     after EPOCHS_ADV and continue classifier SGD on T_omega^m(x).
#   * Online BatchNorm refresh can be combined with FREEZE_BATCHNORM=1: the
#     clean refresh pass updates BN running stats, then adversary generation
#     and classifier SGD keep BN layers in eval mode.
#   * BENCHMARK_MODE=0 by default, matching run_pretrained_input_icnn.sh.
#     Set BENCHMARK_MODE=1 for cuDNN autotune + TF32 timing sweeps.
#   * Sequential execution if more than one algorithm is enabled later.
#   * Per-epoch CUDA-synced wallclock; analyzer drops epoch 1.
#
# ---- Memory safety ----
#
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less fragmentation).
#   * Per-algorithm BATCH override paths exist but unused at default config.
#   * 2-level OOM retry halves batch_size and proportionally adjusts inner
#     budget — preserves global throughput at the cost of GPU memory pressure.
#
# Usage:
#   bash run_runtime_sweep_ddp.sh [SPLIT] [SEED] [NPROC] [K] [EPOCHS] [OUTPUT_FOLDER_NAME]
#
#   SPLIT 0 or 1 — run npf_lastquad
#   SPLIT 2..4   — no-op while only npf_lastquad is enabled
#
# Output location:
#   RESULTS_ROOT defaults to ${SRC_DIR}/input_icnn_ddp_runs.
#   OUTPUT_FOLDER_NAME sets the subfolder under RESULTS_ROOT. This is the
#   easiest way to start a fresh run when an old run already has .completed.
#   RESULTS_DIR is still accepted as a full explicit output path.
#
# The old full-suite split map is intentionally commented out for now:
#   SPLIT 1  — npf
#   SPLIT 2  — nn_dro, madry
#   SPLIT 3  — wrm, wfr
#   SPLIT 4  — dual, new_ppa
#
# Cluster submission example (one 4-GPU NPF-LastQuad job):
#   ./csub.py -n input-icnn-npf-lastquad-ddp -g 4 -t 1d --train \
#     --command "cd /path/to/LAT && bash run_runtime_sweep_ddp.sh 0 1 4 20 50"
#
# Fresh output folder example:
#   OUTPUT_FOLDER_NAME=debug_muon_$(date -u +%Y%m%dT%H%M%SZ) \
#     bash run_runtime_sweep_ddp.sh 0 1 1 20 1
# =============================================================================

# Don't use set -e — we want OOM retry on individual algorithms.

SPLIT=${1:-${SPLIT:-0}}
SEED=${2:-${SEED:-1}}
NPROC=${3:-${NPROC:-4}}
K=${4:-${OMEGA_STEPS:-${K:-20}}}
# Backward-compatible epoch aliases:
#   ADV_EPOCHS controls classifier-updating adversarial epochs.
#   WARMUP_EPOCHS controls fixed-theta ICNN/NPF adversary warmup epochs.
# Positional arg 5 and the older EPOCHS_ADV/EPOCHS_ICNN_PRETRAIN names keep
# precedence so existing launch commands remain unchanged.
EPOCHS_ADV=${5:-${EPOCHS_ADV:-${ADV_EPOCHS:-50}}}
OUTPUT_FOLDER_NAME=${6:-${OUTPUT_FOLDER_NAME:-${RESULTS_SUBDIR:-}}}

# ---- Paths (override with env vars) ----
SRC_DIR="${SRC_DIR:-/mloscratch/homes/aabdolla/LAT}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${SRC_DIR}/ResNet_checkpoints/R2.pth}"
DATA_DIR="${DATA_DIR:-${SRC_DIR}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${SRC_DIR}/input_icnn_ddp_runs}"
RESULTS_DIR_OVERRIDE="${RESULTS_DIR:-}"

cd "$SRC_DIR"

# Memory + thread safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ---- Shared training hyperparameters ----
COMMON_BATCH=${COMMON_BATCH:-${BATCH_SIZE:-512}}  # global effective batch (split across ranks)
PENALTY_LAMBDA=${PENALTY_LAMBDA:-30}
TRANSPORT_COST=${TRANSPORT_COST:-normalized_mse}
LAMBDA_SCHEDULE=${LAMBDA_SCHEDULE:-}
LAMBDA_STAGE_EPOCHS=${LAMBDA_STAGE_EPOCHS:-0}
LR_THETA=${LR_THETA:-0.05}  # flagship recipe
ATTACK_CLEAN_CORRECT_ONLY=${ATTACK_CLEAN_CORRECT_ONLY:-1}
RESET_PARAMETRIC_BB_EACH_BATCH=${RESET_PARAMETRIC_BB_EACH_BATCH:-1}
FREEZE_BATCHNORM=${FREEZE_BATCHNORM:-0}  # legacy/flagship: BN participates
FREEZE_BATCHNORM_AFFINE=${FREEZE_BATCHNORM_AFFINE:-${FREEZE_BN_AFFINE:-$FREEZE_BATCHNORM}}
ADVERSARY_CLASSIFIER_EVAL=${ADVERSARY_CLASSIFIER_EVAL:-1}
BATCHNORM_ONLINE_REFRESH=${BATCHNORM_ONLINE_REFRESH:-${ONLINE_BATCHNORM_REFRESH:-0}}
BATCHNORM_ONLINE_REFRESH_MOMENTUM=${BATCHNORM_ONLINE_REFRESH_MOMENTUM:-${ONLINE_BATCHNORM_REFRESH_MOMENTUM:-}}
RECALIBRATE_BATCHNORM=${RECALIBRATE_BATCHNORM:-0}
BATCHNORM_RECALIBRATION_BATCHES=${BATCHNORM_RECALIBRATION_BATCHES:-0}
BATCHNORM_RECALIBRATION_RESET=${BATCHNORM_RECALIBRATION_RESET:-1}
BATCHNORM_RECALIBRATION_MOMENTUM=${BATCHNORM_RECALIBRATION_MOMENTUM:-}
INP_P=${INP_P:-2}
INP_EPS=${INP_EPS:-0.5}
INP_STEPS=${INP_STEPS:-20}
INP_RESTARTS=${INP_RESTARTS:-5}
EVAL_PGD_SAMPLES=${EVAL_PGD_SAMPLES:-1024}
# 0 = evaluate the transport adversary on the full test set (cheap for
# learned maps). new_ppa/MPA runs a full inner maximization per eval batch,
# so its launcher caps this (e.g. 1024) to keep rank-0 eval time bounded.
EVAL_TRANSPORT_SAMPLES=${EVAL_TRANSPORT_SAMPLES:-0}
EVAL_TRANSPORT_PGD_ALIGNMENT=${EVAL_TRANSPORT_PGD_ALIGNMENT:-0}
EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES=${EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES:-256}
INPUT_PGD_LOSS=${INPUT_PGD_LOSS:-ce}  # legacy 57% benchmark convention
INPUT_PGD_GEOMETRY=${INPUT_PGD_GEOMETRY:-pixel_l2_squared}
MADRY_PGD_STEP_SIZE=${MADRY_PGD_STEP_SIZE:-0}  # 0 => trainer default 2*epsilon/K
USE_MARGIN_LOSS=${USE_MARGIN_LOSS:-1}
SKIP_PGD_DURING_TRAIN=${SKIP_PGD_DURING_TRAIN:-0}
BENCHMARK_MODE=${BENCHMARK_MODE:-0}
PROFILE_INNER=${PROFILE_INNER:-0}
PROFILE_INNER_BATCHES=${PROFILE_INNER_BATCHES:-0}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
SAVE_EVERY_EPOCH_DIR=${SAVE_EVERY_EPOCH_DIR:-}
CHECKPOINT_EPOCH_OFFSET=${CHECKPOINT_EPOCH_OFFSET:-0}
# Conservative default — cluster containers typically ship with a tiny
# /dev/shm (Docker's 64 MB default) and 4 workers/rank × 4 ranks = 16
# worker processes is enough to exhaust it. CIFAR-10 fits in RAM and the
# GPU adversary work dominates wallclock, so 2 workers/rank is plenty.
# Set NUM_WORKERS=0 if you still hit "No space left on device".
NUM_WORKERS=${NUM_WORKERS:-2}
EPOCHS_ICNN_PRETRAIN=${EPOCHS_ICNN_PRETRAIN:-${WARMUP_EPOCHS:-2}}
FROZEN_ADVERSARY_EPOCHS=${FROZEN_ADVERSARY_EPOCHS:-${FROZEN_MAP_EPOCHS:-${POST_MAP_EPOCHS:-0}}}
FROZEN_ADVERSARY_MAP_STEPS=${FROZEN_ADVERSARY_MAP_STEPS:-${FROZEN_MAP_STEPS:-${POST_MAP_STEPS:-1}}}

# ---- Shared BB+Armijo step rule (overridable; defaults match NPF's) ----
BB_ALPHA0=${BB_ALPHA0:-5e-4}
BB_ALPHA_MIN=${BB_ALPHA_MIN:-1e-6}
BB_ALPHA_MAX=${BB_ALPHA_MAX:-1.0}
BB_LS_C=${BB_LS_C:-0.1}
BB_LS_SHRINK=${BB_LS_SHRINK:-0.5}
BB_LS_MAX_STEPS=${BB_LS_MAX_STEPS:-10}
PARAMETRIC_BB_MAX_GRAD_NORM=${PARAMETRIC_BB_MAX_GRAD_NORM:-1.0}
BB_ARGS=(
  --bb-alpha0 "${BB_ALPHA0}"
  --bb-alpha-min "${BB_ALPHA_MIN}"
  --bb-alpha-max "${BB_ALPHA_MAX}"
  --bb-ls-c "${BB_LS_C}"
  --bb-ls-shrink "${BB_LS_SHRINK}"
  --bb-ls-max-steps "${BB_LS_MAX_STEPS}"
  --parametric-bb-max-grad-norm "${PARAMETRIC_BB_MAX_GRAD_NORM}"
)

# ---- Dual Option-D Langevin / MALA inner sampler ----
# DUAL_LANGEVIN_STEPS=0 keeps the one-shot Gaussian estimator
# (closed-form Sinkhorn dual). >0 enables iterative MCMC refinement of
# each particle on the Gibbs target. Defaults below assume DUAL_EPSILON=0.05;
# η ≈ ε is a good starting point — MALA's accept rate is the right
# tuning signal (target ~0.5–0.8). Set DUAL_MALA=0 for plain ULA
# (cheaper, biased).
DUAL_EPSILON=${DUAL_EPSILON:-0.05}
DUAL_LANGEVIN_STEPS=${DUAL_LANGEVIN_STEPS:-${K}}
DUAL_LANGEVIN_STEP_SIZE=${DUAL_LANGEVIN_STEP_SIZE:-5e-3}
DUAL_MALA=${DUAL_MALA:-1}
DUAL_BURN_IN=${DUAL_BURN_IN:-0}
DUAL_SAMPLE_LEVEL=${DUAL_SAMPLE_LEVEL:-3}  # m=8 — keeps wallclock comparable

# ---- New_PPA (MPA) knobs ----
# MPA = Algorithm 1 with R rounds of {K-step BB+Armijo ascent on z anchored at
# the original x; within-class free-weight reassignment}. R=1 collapses to PA.
# The exact 2-round MPA schedule (K ascent -> reassign -> K ascent -> reassign)
# needs PPA_NUM_ROUNDS=2 PPA_MIN_ROUNDS=2 so the gain-based early stop can
# never skip the post-reassignment ascent burst. Round0/refine steps default
# to the shared inner budget K.
PPA_NUM_ROUNDS=${PPA_NUM_ROUNDS:-5}
PPA_MIN_ROUNDS=${PPA_MIN_ROUNDS:-2}
PPA_ROUND0_STEPS=${PPA_ROUND0_STEPS:-}
PPA_REFINE_STEPS=${PPA_REFINE_STEPS:-}
PPA_GAIN_RTOL=${PPA_GAIN_RTOL:-1e-4}
# bb_armijo = shared line-searched step (uniform with other DRO methods);
# const_lr = per-sample MNIST/paper ascent (round0 diminishing lr/sqrt(s),
# refinement constant lr). const_lr makes the ascent invariant to DDP
# sharding; the reassignment pool is all-gathered to the global batch either
# way, so MPA's multi-start pool is COMMON_BATCH regardless of NPROC.
PPA_STEP_RULE=${PPA_STEP_RULE:-bb_armijo}
PPA_ROUND0_LR=${PPA_ROUND0_LR:-0.03}
PPA_REFINE_LR=${PPA_REFINE_LR:-0.015}
# ascent_first = Algorithm 1 rounds {ascend; reassign};
# reassign_first = mirrored rounds {reassign; ascend} closed by a final
# reassignment (R=3: R-A-R-A-R-A-R); round-0 reassignment at z=x picks each
# sample's best same-class start.
# reassign_first_v2 = reassign_first with the round-0 reassignment (at z=x)
# priced at lambda/PPA_ROUND0_LAMBDA_DAMPING (graduated multi-start;
# damping 1 = v1). The damping is read ONLY by reassign_first_v2.
PPA_ROUND_ORDER=${PPA_ROUND_ORDER:-ascent_first}
PPA_ROUND0_LAMBDA_DAMPING=${PPA_ROUND0_LAMBDA_DAMPING:-6.0}

# ---- WRM knobs (WRM == MPA with R=1: K-step per-sample ascent, no
# reassignment; shares lambda / transport cost / margin / mask with MPA) ----
# const_lr = MPA-parity diminishing schedule wrm_inner_lr/sqrt(s);
# bb_armijo = legacy shared line-searched rule.
WRM_STEP_RULE=${WRM_STEP_RULE:-const_lr}
WRM_INNER_LR=${WRM_INNER_LR:-0.03}

# ---- NPF architecture knobs ----
RUN_ONLY_ALGO=${RUN_ONLY_ALGO:-}
NPF_INNER_OPTIMIZER=${NPF_INNER_OPTIMIZER:-bb_armijo}  # bb_armijo | muon (flagship: bb_armijo)
NPF_RESET_OMEGA_EACH_BATCH=${NPF_RESET_OMEGA_EACH_BATCH:-0}
NPF_PROJECT_TO_INPUT_BALL=${NPF_PROJECT_TO_INPUT_BALL:-0}
NPF_OMEGA_WEIGHT_DECAY=${NPF_OMEGA_WEIGHT_DECAY:-1e-4}  # legacy golden-run parity: L2 on free omega kernels
NPF_PGD_ANCHOR_WEIGHT=${NPF_PGD_ANCHOR_WEIGHT:-0.0}
NPF_PGD_ANCHOR_STEPS=${NPF_PGD_ANCHOR_STEPS:-5}
NPF_PGD_ANCHOR_RESTARTS=${NPF_PGD_ANCHOR_RESTARTS:-1}
NPF_PGD_ANCHOR_LOSS=${NPF_PGD_ANCHOR_LOSS:-margin}
NPF_MUON_LR=${NPF_MUON_LR:-2e-4}
NPF_MUON_MOMENTUM=${NPF_MUON_MOMENTUM:-0.95}
NPF_MUON_NESTEROV=${NPF_MUON_NESTEROV:-1}
NPF_MUON_NS_STEPS=${NPF_MUON_NS_STEPS:-5}
NPF_MUON_MATRIX_LR_SCALE=${NPF_MUON_MATRIX_LR_SCALE:-auto}
NPF_MUON_WEIGHT_DECAY=${NPF_MUON_WEIGHT_DECAY:-0.0}
NPF_MUON_FALLBACK=${NPF_MUON_FALLBACK:-adamw}
NPF_MUON_FALLBACK_LR=${NPF_MUON_FALLBACK_LR:-0.0}  # 0 => reuse NPF_MUON_LR
NPF_MUON_FALLBACK_WEIGHT_DECAY=${NPF_MUON_FALLBACK_WEIGHT_DECAY:-0.0}
NPF_MUON_ADAM_BETA1=${NPF_MUON_ADAM_BETA1:-0.9}
NPF_MUON_ADAM_BETA2=${NPF_MUON_ADAM_BETA2:-0.999}
NPF_MUON_ADAM_EPS=${NPF_MUON_ADAM_EPS:-1e-8}
NPF_MUON_MAX_GRAD_NORM=${NPF_MUON_MAX_GRAD_NORM:-0.0}
NPF_HIDDEN=${NPF_HIDDEN:-1024 512 512 256 128 64}
NPF_OUTER_RANK=${NPF_OUTER_RANK:-1}
NPF_INNER_RANK=${NPF_INNER_RANK:-1}
NPF_ACTIVATION=${NPF_ACTIVATION:-softplus}
NPF_ELU_ALPHA=${NPF_ELU_ALPHA:-1.0}
NPF_SOFTPLUS_BETA=${NPF_SOFTPLUS_BETA:-20.0}
NPF_INIT_EPS=${NPF_INIT_EPS:-1e-2}
NPF_IDENTITY_INIT=${NPF_IDENTITY_INIT:-1}  # identity-adjacent start (non-identity all-layers init measured 12.9 pixel-L2 off)
NPF_STRONG_CONVEXITY=${NPF_STRONG_CONVEXITY:-1.0}
NPF_POS_WEIGHTS=${NPF_POS_WEIGHTS:-1}
NPF_POSITIVE_WEIGHT_RECTIFIER=${NPF_POSITIVE_WEIGHT_RECTIFIER:-exp}
NPF_LASTQUAD_HIDDEN=${NPF_LASTQUAD_HIDDEN:-1024 512 512 256 128 64}
NPF_LASTQUAD_OUTPUT_RANK=${NPF_LASTQUAD_OUTPUT_RANK:-64}  # decisive capacity knob (2026-07-06)
# NPF_LASTQUAD_HIDDEN=${NPF_LASTQUAD_HIDDEN:-128 128 128 128}
NPF_LASTQUAD_ACTIVATION=${NPF_LASTQUAD_ACTIVATION:-softplus}
NPF_LASTQUAD_ELU_ALPHA=${NPF_LASTQUAD_ELU_ALPHA:-1.0}
NPF_LASTQUAD_SOFTPLUS_BETA=${NPF_LASTQUAD_SOFTPLUS_BETA:-20.0}
NPF_LASTQUAD_INIT_EPS=${NPF_LASTQUAD_INIT_EPS:-1e-2}
NPF_LASTQUAD_IDENTITY_INIT=${NPF_LASTQUAD_IDENTITY_INIT:-0}
NPF_LASTQUAD_STRONG_CONVEXITY=${NPF_LASTQUAD_STRONG_CONVEXITY:-1.0}
NPF_LASTQUAD_POS_WEIGHTS=${NPF_LASTQUAD_POS_WEIGHTS:-1}
NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER=${NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER:-exp}

HIDDEN_TAG="$(printf "%s" "$NPF_LASTQUAD_HIDDEN" | tr -s '[:space:]' 'x' | tr -cd '[:alnum:]_.=-')"
OPT_TAG=""
if [ "$NPF_INNER_OPTIMIZER" != "bb_armijo" ]; then
    MUON_LR_TAG="$(printf "%s" "$NPF_MUON_LR" | tr -cd '[:alnum:]_.=-')"
    OPT_TAG="_opt${NPF_INNER_OPTIMIZER}_muonlr${MUON_LR_TAG}"
fi
WARM_TAG=""
if [ "${EPOCHS_ICNN_PRETRAIN}" != "0" ]; then
    WARM_TAG="_warm${EPOCHS_ICNN_PRETRAIN}"
fi
FROZEN_TAG=""
if [ "${FROZEN_ADVERSARY_EPOCHS}" != "0" ]; then
    FROZEN_TAG="_frozenadv${FROZEN_ADVERSARY_EPOCHS}x${FROZEN_ADVERSARY_MAP_STEPS}"
fi
ANCHOR_TAG="$(printf "%s" "$NPF_PGD_ANCHOR_WEIGHT" | tr -cd '[:alnum:]_.=-' | sed -e 's/[.]/p/g' -e 's/-/m/g')"
AUTO_RUN_NAME="npf_lastquad_seed${SEED}_ep${EPOCHS_ADV}${WARM_TAG}${FROZEN_TAG}_K${K}_ddp${NPROC}_lr${LR_THETA}_lam${PENALTY_LAMBDA}_eps${INP_EPS}_margin${USE_MARGIN_LOSS}_act${NPF_LASTQUAD_ACTIVATION}_init${NPF_LASTQUAD_INIT_EPS}_idinit${NPF_LASTQUAD_IDENTITY_INIT}_proj${NPF_PROJECT_TO_INPUT_BALL}_anch${ANCHOR_TAG}_hidden${HIDDEN_TAG}${OPT_TAG}"
RUN_NAME="${RUN_NAME:-${AUTO_RUN_NAME}}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-${RUN_NAME}}"
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
else
    RESULTS_DIR="${RESULTS_ROOT}/${OUTPUT_FOLDER_NAME}"
fi
LOG_DIR="${RESULTS_DIR}/logs"
TIME_DIR="${RESULTS_DIR}/time"
MANIFEST="${RESULTS_DIR}/run_manifest.tsv"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$TIME_DIR"
if [ ! -f "$MANIFEST" ]; then
    printf "status\trun_name\talgorithm\tseed\tworld_size\tk\tepochs\tstarted_utc\tended_utc\telapsed_seconds\tout_dir\tcheckpoint_last\tcheckpoint_best\tlog_file\ttime_file\tcsv\tsummary_json\thistory_json\n" > "$MANIFEST"
fi

DEFAULT_ALGOS=(npf_lastquad)
# Full runtime sweep disabled for the current NPF-LastQuad launch:
# DEFAULT_ALGOS=(npf nn_dro madry wrm wfr dual new_ppa)
KNOWN_ALGOS=(npf npf_lastquad new_ppa wrm)
# KNOWN_ALGOS=(npf npf_lastquad nn_dro madry wrm wfr dual new_ppa)

is_known_algo() {
    local needle="$1" algo
    for algo in "${KNOWN_ALGOS[@]}"; do
        if [ "$algo" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

join_by_comma() {
    local first=1 item
    for item in "$@"; do
        if [ "$first" -eq 1 ]; then
            printf "%s" "$item"
            first=0
        else
            printf ",%s" "$item"
        fi
    done
}

if [ -n "$RUN_ONLY_ALGO" ] && ! is_known_algo "$RUN_ONLY_ALGO"; then
    echo "[FATAL] Unknown RUN_ONLY_ALGO=${RUN_ONLY_ALGO}. Known: ${KNOWN_ALGOS[*]}"
    exit 1
fi
if [[ "$NPF_INNER_OPTIMIZER" != "bb_armijo" && "$NPF_INNER_OPTIMIZER" != "muon" ]]; then
    echo "[FATAL] Unknown NPF_INNER_OPTIMIZER=${NPF_INNER_OPTIMIZER}. Use bb_armijo or muon."
    exit 1
fi

if [ -n "$RUN_ONLY_ALGO" ]; then
    ACTIVE_ALGOS=("$RUN_ONLY_ALGO")
else
    ACTIVE_ALGOS=("${DEFAULT_ALGOS[@]}")
fi
ACTIVE_ALGOS_CSV="$(join_by_comma "${ACTIVE_ALGOS[@]}")"

NPF_MUON_NESTEROV_FLAG="--npf-muon-nesterov"
case "$NPF_MUON_NESTEROV" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
        NPF_MUON_NESTEROV_FLAG="--no-npf-muon-nesterov"
        ;;
esac
NPF_OPT_ARGS=(
  --npf-inner-optimizer "${NPF_INNER_OPTIMIZER}"
  --npf-muon-lr "${NPF_MUON_LR}"
  --npf-muon-momentum "${NPF_MUON_MOMENTUM}"
  "${NPF_MUON_NESTEROV_FLAG}"
  --npf-muon-ns-steps "${NPF_MUON_NS_STEPS}"
  --npf-muon-matrix-lr-scale "${NPF_MUON_MATRIX_LR_SCALE}"
  --npf-muon-weight-decay "${NPF_MUON_WEIGHT_DECAY}"
  --npf-muon-fallback "${NPF_MUON_FALLBACK}"
  --npf-muon-fallback-lr "${NPF_MUON_FALLBACK_LR}"
  --npf-muon-fallback-weight-decay "${NPF_MUON_FALLBACK_WEIGHT_DECAY}"
  --npf-muon-adam-beta1 "${NPF_MUON_ADAM_BETA1}"
  --npf-muon-adam-beta2 "${NPF_MUON_ADAM_BETA2}"
  --npf-muon-adam-eps "${NPF_MUON_ADAM_EPS}"
  --npf-muon-max-grad-norm "${NPF_MUON_MAX_GRAD_NORM}"
  --npf-omega-weight-decay "${NPF_OMEGA_WEIGHT_DECAY}"
  --npf-pgd-anchor-weight "${NPF_PGD_ANCHOR_WEIGHT}"
  --npf-pgd-anchor-steps "${NPF_PGD_ANCHOR_STEPS}"
  --npf-pgd-anchor-restarts "${NPF_PGD_ANCHOR_RESTARTS}"
  --npf-pgd-anchor-loss "${NPF_PGD_ANCHOR_LOSS}"
)
case "$NPF_RESET_OMEGA_EACH_BATCH" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        NPF_OPT_ARGS+=(--npf-reset-omega-each-batch)
        ;;
    *)
        NPF_OPT_ARGS+=(--no-npf-reset-omega-each-batch)
        ;;
esac
case "$NPF_PROJECT_TO_INPUT_BALL" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        NPF_OPT_ARGS+=(--npf-project-to-input-ball)
        ;;
    *)
        NPF_OPT_ARGS+=(--no-npf-project-to-input-ball)
        ;;
esac

# ---- Pre-flight ----
echo ""
echo "================================================================"
echo "  Input-ICNN NPF-LastQuad — DDP"
echo "  Run name: ${RUN_NAME}"
echo "  Output folder: ${OUTPUT_FOLDER_NAME}"
echo "  SPLIT=${SPLIT}  SEED=${SEED}  NPROC=${NPROC}  K=${K}"
echo "  Warmup epochs=${EPOCHS_ICNN_PRETRAIN}  Adv epochs=${EPOCHS_ADV}  Batch=${COMMON_BATCH} (global)  λ=${PENALTY_LAMBDA}"
echo "  Classifier LR: ${LR_THETA}"
echo "  Transport cost: ${TRANSPORT_COST}"
echo "  Attack clean-correct only: ${ATTACK_CLEAN_CORRECT_ONLY}"
echo "  Reset parametric BB each batch: ${RESET_PARAMETRIC_BB_EACH_BATCH}"
echo "  Frozen-adversary post phase: epochs=${FROZEN_ADVERSARY_EPOCHS}  map_steps=${FROZEN_ADVERSARY_MAP_STEPS}"
echo "  Freeze BatchNorm running stats: ${FREEZE_BATCHNORM}"
echo "  Freeze BatchNorm affine params: ${FREEZE_BATCHNORM_AFFINE}"
echo "  Adversary-side classifier eval mode: ${ADVERSARY_CLASSIFIER_EVAL}"
echo "  Online clean BatchNorm refresh: ${BATCHNORM_ONLINE_REFRESH}  momentum=${BATCHNORM_ONLINE_REFRESH_MOMENTUM:-module-default}"
echo "  Recalibrate BatchNorm after adv epochs: ${RECALIBRATE_BATCHNORM}  batches=${BATCHNORM_RECALIBRATION_BATCHES}  reset=${BATCHNORM_RECALIBRATION_RESET}  momentum=${BATCHNORM_RECALIBRATION_MOMENTUM:-cumulative}"
if [ -n "$LAMBDA_SCHEDULE" ]; then
    echo "  Lambda schedule: ${LAMBDA_SCHEDULE}  stage_epochs=${LAMBDA_STAGE_EPOCHS}"
fi
echo "  Active algos: ${ACTIVE_ALGOS[*]}"
echo "  Margin loss: ${USE_MARGIN_LOSS}"
echo "  Input-PGD eval loss: ${INPUT_PGD_LOSS}"
echo "  Input-PGD geometry: ${INPUT_PGD_GEOMETRY}"
echo "  Input PGD eval: p=${INP_P} eps=${INP_EPS} steps=${INP_STEPS} restarts=${INP_RESTARTS} samples=${EVAL_PGD_SAMPLES}"
if [ -n "${WRM_NORMALIZED_RADIUS_PROTOCOL:-}" ]; then
    echo "  WRM normalized-radius protocol: ${WRM_NORMALIZED_RADIUS_PROTOCOL} mode=${WRM_PGD_EPS_MODE:-} C_p=${WRM_CP:-} rho=${WRM_NORMALIZED_RADIUS:-} effective_eps=${WRM_EFFECTIVE_INP_EPS:-${INP_EPS}}"
fi
echo "  Transport-vs-PGD alignment: ${EVAL_TRANSPORT_PGD_ALIGNMENT} samples=${EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES}"
echo "  Skip PGD during train: ${SKIP_PGD_DURING_TRAIN}  Benchmark mode: ${BENCHMARK_MODE}"
echo "  Inner profiling: ${PROFILE_INNER}  batches=${PROFILE_INNER_BATCHES}"
echo "  NPF optimizer: ${NPF_INNER_OPTIMIZER}"
echo "  Reset NPF omega each batch: ${NPF_RESET_OMEGA_EACH_BATCH}"
echo "  Project NPF transport to input-PGD ball: ${NPF_PROJECT_TO_INPUT_BALL}"
echo "  NPF PGD anchor: weight=${NPF_PGD_ANCHOR_WEIGHT} steps=${NPF_PGD_ANCHOR_STEPS} restarts=${NPF_PGD_ANCHOR_RESTARTS} loss=${NPF_PGD_ANCHOR_LOSS}"
echo "  BB+Armijo:   alpha0=${BB_ALPHA0}  alpha=[${BB_ALPHA_MIN}, ${BB_ALPHA_MAX}]"
echo "                ls_c=${BB_LS_C}  shrink=${BB_LS_SHRINK}  ls_max=${BB_LS_MAX_STEPS}"
echo "                parametric_grad_clip=${PARAMETRIC_BB_MAX_GRAD_NORM}"
if [ "$NPF_INNER_OPTIMIZER" = "muon" ]; then
    echo "  Muon:        lr=${NPF_MUON_LR}  momentum=${NPF_MUON_MOMENTUM}  nesterov=${NPF_MUON_NESTEROV}"
    echo "                ns_steps=${NPF_MUON_NS_STEPS}  scale=${NPF_MUON_MATRIX_LR_SCALE}  wd=${NPF_MUON_WEIGHT_DECAY}"
    echo "                fallback=${NPF_MUON_FALLBACK}  fallback_lr=${NPF_MUON_FALLBACK_LR}  max_grad_norm=${NPF_MUON_MAX_GRAD_NORM}"
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" madry "* ]]; then
    echo "  Madry/RO:    pixel l2-PGD eps=${INP_EPS}  step_size=${MADRY_PGD_STEP_SIZE:-0} (0=auto)"
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" dual "* ]]; then
    if [ "${DUAL_LANGEVIN_STEPS}" -gt 0 ]; then
        echo "  Dual mode:   Option-D Langevin/MALA sampler"
        echo "                eps=${DUAL_EPSILON}  m=2^${DUAL_SAMPLE_LEVEL}  K_langevin=${DUAL_LANGEVIN_STEPS}"
        echo "                eta=${DUAL_LANGEVIN_STEP_SIZE}  MALA=${DUAL_MALA}  burn_in=${DUAL_BURN_IN}"
    else
        echo "  Dual mode:   one-shot Gaussian (closed-form Sinkhorn)"
    fi
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" npf_lastquad "* ]]; then
    echo "  NPF-LastQuad: hidden=${NPF_LASTQUAD_HIDDEN}"
    echo "                output_rank=${NPF_LASTQUAD_OUTPUT_RANK}"
    echo "                activation=${NPF_LASTQUAD_ACTIVATION} beta=${NPF_LASTQUAD_SOFTPLUS_BETA}"
    echo "                init_eps=${NPF_LASTQUAD_INIT_EPS} identity_init=${NPF_LASTQUAD_IDENTITY_INIT} strong_convexity=${NPF_LASTQUAD_STRONG_CONVEXITY}"
    echo "                pos_weights=${NPF_LASTQUAD_POS_WEIGHTS} rectifier=${NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER}"
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" new_ppa "* ]]; then
    echo "  New_PPA/MPA:  step_rule=${PPA_STEP_RULE} round_order=${PPA_ROUND_ORDER} rounds=${PPA_NUM_ROUNDS} min_rounds=${PPA_MIN_ROUNDS}"
    if [ "${PPA_ROUND_ORDER}" = "reassign_first_v2" ]; then
        echo "                round0_lambda_damping=${PPA_ROUND0_LAMBDA_DAMPING} (round-0 reassignment priced at lambda/damping)"
    fi
    echo "                round0_steps=${PPA_ROUND0_STEPS:-${K}} refine_steps=${PPA_REFINE_STEPS:-${K}} gain_rtol=${PPA_GAIN_RTOL}"
    echo "                round0_lr=${PPA_ROUND0_LR} refine_lr=${PPA_REFINE_LR} (const_lr rule only)"
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" wrm "* ]]; then
    echo "  WRM (MPA R=1): step_rule=${WRM_STEP_RULE} K=${K} inner_lr=${WRM_INNER_LR} (const_lr: lr/sqrt(s) diminishing)"
fi
if [[ " ${ACTIVE_ALGOS[*]} " == *" npf "* ]]; then
    echo "  NPF:          hidden=${NPF_HIDDEN}"
    echo "                outer_rank=${NPF_OUTER_RANK} inner_rank=${NPF_INNER_RANK}"
    echo "                activation=${NPF_ACTIVATION} elu_alpha=${NPF_ELU_ALPHA} beta=${NPF_SOFTPLUS_BETA}"
    echo "                init_eps=${NPF_INIT_EPS} identity_init=${NPF_IDENTITY_INIT} strong_convexity=${NPF_STRONG_CONVEXITY}"
    echo "                pos_weights=${NPF_POS_WEIGHTS} rectifier=${NPF_POSITIVE_WEIGHT_RECTIFIER}"
fi
echo "  Results: ${RESULTS_DIR}"
echo "  Started: $(date)"
echo "================================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU pre-check:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo
fi

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "[FATAL] PRETRAINED_PATH not found: $PRETRAINED_PATH"
    exit 1
fi


# ======================================================================
# Per-algorithm config keyed off K (the shared inner-iteration budget).
# ======================================================================
algo_args() {
    # Per-algorithm flags. The shared --bb-* hyperparameters are added
    # by the runner outside of this function so every adversary uses
    # the SAME BB+Armijo step rule for DRO ascent methods. Madry/RO is
    # standard pixel-space l2-PGD, so --madry-pgd-step-size is honoured when set
    # (>0) and otherwise defaults to 2*epsilon/K.
    local algo="$1" k="$2"
    local npf_identity_arg="--npf-identity-init"
    case "$NPF_IDENTITY_INIT" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF)
            npf_identity_arg="--no-npf-identity-init"
            ;;
    esac
    local npf_pos_weights_arg="--npf-pos-weights"
    case "$NPF_POS_WEIGHTS" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF)
            npf_pos_weights_arg="--no-npf-pos-weights"
            ;;
    esac
    local npf_lastquad_identity_arg="--npf-lastquad-identity-init"
    case "$NPF_LASTQUAD_IDENTITY_INIT" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF)
            npf_lastquad_identity_arg="--no-npf-lastquad-identity-init"
            ;;
    esac
    local npf_lastquad_pos_weights_arg="--npf-lastquad-pos-weights"
    case "$NPF_LASTQUAD_POS_WEIGHTS" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF)
            npf_lastquad_pos_weights_arg="--no-npf-lastquad-pos-weights"
            ;;
    esac
    case "$algo" in
        npf)
            echo "--omega-steps-per-batch ${k} \
                --npf-hidden ${NPF_HIDDEN} \
                --npf-outer-rank ${NPF_OUTER_RANK} --npf-inner-rank ${NPF_INNER_RANK} \
                --npf-activation ${NPF_ACTIVATION} --npf-elu-alpha ${NPF_ELU_ALPHA} \
                --npf-softplus-beta ${NPF_SOFTPLUS_BETA} \
                --npf-init-eps ${NPF_INIT_EPS} ${npf_identity_arg} \
                --npf-strong-convexity ${NPF_STRONG_CONVEXITY} \
                ${npf_pos_weights_arg} \
                --npf-positive-weight-rectifier ${NPF_POSITIVE_WEIGHT_RECTIFIER}"
            ;;
        npf_lastquad)
            echo "--omega-steps-per-batch ${k} \
                --npf-lastquad-hidden ${NPF_LASTQUAD_HIDDEN} \
                --npf-lastquad-output-rank ${NPF_LASTQUAD_OUTPUT_RANK} \
                --npf-lastquad-activation ${NPF_LASTQUAD_ACTIVATION} \
                --npf-lastquad-elu-alpha ${NPF_LASTQUAD_ELU_ALPHA} \
                --npf-lastquad-softplus-beta ${NPF_LASTQUAD_SOFTPLUS_BETA} \
                --npf-lastquad-init-eps ${NPF_LASTQUAD_INIT_EPS} \
                ${npf_lastquad_identity_arg} \
                --npf-lastquad-strong-convexity ${NPF_LASTQUAD_STRONG_CONVEXITY} \
                ${npf_lastquad_pos_weights_arg} \
                --npf-lastquad-positive-weight-rectifier ${NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER}"
            ;;
        nn_dro)
            echo "--omega-steps-per-batch ${k} \
                --nn-dro-hidden 512 512 256 256 128 \
                --nn-dro-activation relu --nn-dro-softplus-beta 20.0 \
                --nn-dro-init-scale 1e-3"
            ;;
        madry)
            echo "--madry-epsilon ${INP_EPS} --madry-pgd-steps ${k} \
                --madry-pgd-step-size ${MADRY_PGD_STEP_SIZE} --madry-pgd-restarts 1"
            ;;
        wrm)
            echo "--wrm-step-rule ${WRM_STEP_RULE} \
                --wrm-inner-steps ${k} \
                --wrm-inner-lr ${WRM_INNER_LR}"
            ;;
        wfr)
            echo "--wfr-inner-steps ${k} --wfr-num-samples 8 --wfr-epsilon 0.1"
            ;;
        dual)
            # Option-D rigorous inner sampling: each particle is refined
            # via a Langevin chain on the Gibbs target before the
            # Sinkhorn dual loss. With DUAL_LANGEVIN_STEPS=0 the run
            # falls back to the one-shot closed-form estimator
            # (no inner ascent — Dual is then the only method without
            # iterative inner work, kept for back-compat and as a
            # fast-but-biased reference).
            local mala_flag="--no-dual-mala"
            if [ "$DUAL_MALA" = "1" ]; then mala_flag="--dual-mala"; fi
            echo "--dual-epsilon ${DUAL_EPSILON} \
                --dual-sample-level ${DUAL_SAMPLE_LEVEL} \
                --dual-langevin-steps ${DUAL_LANGEVIN_STEPS} \
                --dual-langevin-step-size ${DUAL_LANGEVIN_STEP_SIZE} \
                --dual-burn-in ${DUAL_BURN_IN} \
                ${mala_flag}"
            ;;
        new_ppa)
            echo "--ppa-step-rule ${PPA_STEP_RULE} \
                --ppa-round-order ${PPA_ROUND_ORDER} \
                --ppa-round0-lambda-damping ${PPA_ROUND0_LAMBDA_DAMPING} \
                --ppa-num-rounds ${PPA_NUM_ROUNDS} --ppa-min-rounds ${PPA_MIN_ROUNDS} \
                --ppa-round0-steps ${PPA_ROUND0_STEPS:-${k}} \
                --ppa-refine-steps ${PPA_REFINE_STEPS:-${k}} \
                --ppa-round0-lr ${PPA_ROUND0_LR} \
                --ppa-refine-lr ${PPA_REFINE_LR} \
                --ppa-gain-rtol ${PPA_GAIN_RTOL}"
            ;;
    esac
}


# ======================================================================
# Runner with skip-if-completed + 2-level OOM retry.
# ======================================================================
run_algo() {
    local ALGO=$1
    local OUT_DIR="${RESULTS_DIR}/${ALGO}/seed_${SEED}"
    local LOG_FILE="${LOG_DIR}/${ALGO}_seed${SEED}.log"
    local TIME_FILE="${TIME_DIR}/${ALGO}_seed${SEED}.time.txt"
    local FINAL_CKPT="${OUT_DIR}/${ALGO}_seed${SEED}_last.pth"
    local BEST_CKPT="${OUT_DIR}/${ALGO}_seed${SEED}_best_robust.pth"
    local SUMMARY_JSON="${OUT_DIR}/${ALGO}_seed${SEED}_last.summary.json"
    local HISTORY_JSON="${OUT_DIR}/${ALGO}_seed${SEED}_last.history.json"
    local RUN_CONFIG="${OUT_DIR}/run_config.env"
    local DONE="${OUT_DIR}/.completed"
    local CSV="${OUT_DIR}/epoch_log.csv"

    if [ -f "$DONE" ]; then
        echo "[SKIP] ${ALGO} seed=${SEED} — already completed at ${OUT_DIR}"
        echo "       Use OUTPUT_FOLDER_NAME=<new-name> or RESULTS_DIR=<full-path> for a fresh run."
        return 0
    fi

    if [ -d "$OUT_DIR" ]; then
        local STALE_DIR="${OUT_DIR}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
        echo "[INFO] Existing incomplete output found; moving it to ${STALE_DIR}"
        mv "$OUT_DIR" "$STALE_DIR"
    fi
    mkdir -p "$OUT_DIR"

    local BATCH_OVR=$COMMON_BATCH
    local KEFF=$K

    local STARTED_UTC T0 ENDED_UTC T1 ELAPSED STATUS
    local ATTEMPT=0
    local MAX_ATTEMPTS=3

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))

        echo ""
        echo "----------------------------------------------------------------"
        echo "  [Attempt ${ATTEMPT}] ${ALGO}  seed=${SEED}  K=${KEFF}  batch=${BATCH_OVR}  GPUs=${NPROC}"
        echo "  log=${LOG_FILE}"
        echo "  Started: $(date)"
        echo "----------------------------------------------------------------"

        STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        T0=$(date +%s)

        local EXTRA_ARGS
        EXTRA_ARGS=$(algo_args "$ALGO" "$KEFF")

        {
            printf "RUN_NAME=%q\n" "$RUN_NAME"
            printf "OUTPUT_FOLDER_NAME=%q\n" "$OUTPUT_FOLDER_NAME"
            printf "ALGORITHM=%q\n" "$ALGO"
            printf "SEED=%q\n" "$SEED"
            printf "NPROC=%q\n" "$NPROC"
            printf "K=%q\n" "$KEFF"
            printf "EPOCHS_ADV=%q\n" "$EPOCHS_ADV"
            printf "FROZEN_ADVERSARY_EPOCHS=%q\n" "$FROZEN_ADVERSARY_EPOCHS"
            printf "FROZEN_ADVERSARY_MAP_STEPS=%q\n" "$FROZEN_ADVERSARY_MAP_STEPS"
            printf "COMMON_BATCH=%q\n" "$BATCH_OVR"
            printf "LR_THETA=%q\n" "$LR_THETA"
            printf "ATTACK_CLEAN_CORRECT_ONLY=%q\n" "$ATTACK_CLEAN_CORRECT_ONLY"
            printf "RESET_PARAMETRIC_BB_EACH_BATCH=%q\n" "$RESET_PARAMETRIC_BB_EACH_BATCH"
            printf "FREEZE_BATCHNORM=%q\n" "$FREEZE_BATCHNORM"
            printf "FREEZE_BATCHNORM_AFFINE=%q\n" "$FREEZE_BATCHNORM_AFFINE"
            printf "ADVERSARY_CLASSIFIER_EVAL=%q\n" "$ADVERSARY_CLASSIFIER_EVAL"
            printf "BATCHNORM_ONLINE_REFRESH=%q\n" "$BATCHNORM_ONLINE_REFRESH"
            printf "BATCHNORM_ONLINE_REFRESH_MOMENTUM=%q\n" "$BATCHNORM_ONLINE_REFRESH_MOMENTUM"
            printf "RECALIBRATE_BATCHNORM=%q\n" "$RECALIBRATE_BATCHNORM"
            printf "BATCHNORM_RECALIBRATION_BATCHES=%q\n" "$BATCHNORM_RECALIBRATION_BATCHES"
            printf "BATCHNORM_RECALIBRATION_RESET=%q\n" "$BATCHNORM_RECALIBRATION_RESET"
            printf "BATCHNORM_RECALIBRATION_MOMENTUM=%q\n" "$BATCHNORM_RECALIBRATION_MOMENTUM"
            printf "PENALTY_LAMBDA=%q\n" "$PENALTY_LAMBDA"
            printf "TRANSPORT_COST=%q\n" "$TRANSPORT_COST"
            printf "LAMBDA_SCHEDULE=%q\n" "$LAMBDA_SCHEDULE"
            printf "LAMBDA_STAGE_EPOCHS=%q\n" "$LAMBDA_STAGE_EPOCHS"
            printf "USE_MARGIN_LOSS=%q\n" "$USE_MARGIN_LOSS"
            printf "INP_P=%q\n" "$INP_P"
            printf "INP_EPS=%q\n" "$INP_EPS"
            printf "WRM_NORMALIZED_RADIUS_PROTOCOL=%q\n" "${WRM_NORMALIZED_RADIUS_PROTOCOL:-}"
            printf "WRM_PGD_EPS_MODE=%q\n" "${WRM_PGD_EPS_MODE:-}"
            printf "WRM_NORMALIZED_RADIUS_REQUESTED=%q\n" "${WRM_NORMALIZED_RADIUS_REQUESTED:-}"
            printf "WRM_NORMALIZED_RADIUS=%q\n" "${WRM_NORMALIZED_RADIUS:-}"
            printf "WRM_CP=%q\n" "${WRM_CP:-}"
            printf "WRM_EFFECTIVE_INP_EPS=%q\n" "${WRM_EFFECTIVE_INP_EPS:-}"
            printf "INP_STEPS=%q\n" "$INP_STEPS"
            printf "INP_RESTARTS=%q\n" "$INP_RESTARTS"
            printf "EVAL_PGD_SAMPLES=%q\n" "$EVAL_PGD_SAMPLES"
            printf "EVAL_TRANSPORT_SAMPLES=%q\n" "$EVAL_TRANSPORT_SAMPLES"
            printf "EVAL_TRANSPORT_PGD_ALIGNMENT=%q\n" "$EVAL_TRANSPORT_PGD_ALIGNMENT"
            printf "EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES=%q\n" "$EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES"
            printf "INPUT_PGD_LOSS=%q\n" "$INPUT_PGD_LOSS"
            printf "INPUT_PGD_GEOMETRY=%q\n" "$INPUT_PGD_GEOMETRY"
            printf "PROFILE_INNER=%q\n" "$PROFILE_INNER"
            printf "PROFILE_INNER_BATCHES=%q\n" "$PROFILE_INNER_BATCHES"
            printf "RESUME_CHECKPOINT=%q\n" "$RESUME_CHECKPOINT"
            printf "SAVE_EVERY_EPOCH_DIR=%q\n" "$SAVE_EVERY_EPOCH_DIR"
            printf "CHECKPOINT_EPOCH_OFFSET=%q\n" "$CHECKPOINT_EPOCH_OFFSET"
            printf "NPF_INNER_OPTIMIZER=%q\n" "$NPF_INNER_OPTIMIZER"
            printf "NPF_RESET_OMEGA_EACH_BATCH=%q\n" "$NPF_RESET_OMEGA_EACH_BATCH"
            printf "NPF_PROJECT_TO_INPUT_BALL=%q\n" "$NPF_PROJECT_TO_INPUT_BALL"
            printf "NPF_OMEGA_WEIGHT_DECAY=%q\n" "$NPF_OMEGA_WEIGHT_DECAY"
            printf "NPF_PGD_ANCHOR_WEIGHT=%q\n" "$NPF_PGD_ANCHOR_WEIGHT"
            printf "NPF_PGD_ANCHOR_STEPS=%q\n" "$NPF_PGD_ANCHOR_STEPS"
            printf "NPF_PGD_ANCHOR_RESTARTS=%q\n" "$NPF_PGD_ANCHOR_RESTARTS"
            printf "NPF_PGD_ANCHOR_LOSS=%q\n" "$NPF_PGD_ANCHOR_LOSS"
            printf "NPF_MUON_LR=%q\n" "$NPF_MUON_LR"
            printf "NPF_MUON_MOMENTUM=%q\n" "$NPF_MUON_MOMENTUM"
            printf "NPF_MUON_NESTEROV=%q\n" "$NPF_MUON_NESTEROV"
            printf "NPF_MUON_NS_STEPS=%q\n" "$NPF_MUON_NS_STEPS"
            printf "NPF_MUON_MATRIX_LR_SCALE=%q\n" "$NPF_MUON_MATRIX_LR_SCALE"
            printf "NPF_MUON_WEIGHT_DECAY=%q\n" "$NPF_MUON_WEIGHT_DECAY"
            printf "NPF_MUON_FALLBACK=%q\n" "$NPF_MUON_FALLBACK"
            printf "NPF_MUON_FALLBACK_LR=%q\n" "$NPF_MUON_FALLBACK_LR"
            printf "NPF_MUON_FALLBACK_WEIGHT_DECAY=%q\n" "$NPF_MUON_FALLBACK_WEIGHT_DECAY"
            printf "NPF_MUON_MAX_GRAD_NORM=%q\n" "$NPF_MUON_MAX_GRAD_NORM"
            printf "NPF_HIDDEN=%q\n" "$NPF_HIDDEN"
            printf "NPF_OUTER_RANK=%q\n" "$NPF_OUTER_RANK"
            printf "NPF_INNER_RANK=%q\n" "$NPF_INNER_RANK"
            printf "NPF_ACTIVATION=%q\n" "$NPF_ACTIVATION"
            printf "NPF_ELU_ALPHA=%q\n" "$NPF_ELU_ALPHA"
            printf "NPF_SOFTPLUS_BETA=%q\n" "$NPF_SOFTPLUS_BETA"
            printf "NPF_INIT_EPS=%q\n" "$NPF_INIT_EPS"
            printf "NPF_IDENTITY_INIT=%q\n" "$NPF_IDENTITY_INIT"
            printf "NPF_STRONG_CONVEXITY=%q\n" "$NPF_STRONG_CONVEXITY"
            printf "NPF_POS_WEIGHTS=%q\n" "$NPF_POS_WEIGHTS"
            printf "NPF_POSITIVE_WEIGHT_RECTIFIER=%q\n" "$NPF_POSITIVE_WEIGHT_RECTIFIER"
            printf "NPF_LASTQUAD_HIDDEN=%q\n" "$NPF_LASTQUAD_HIDDEN"
            printf "NPF_LASTQUAD_OUTPUT_RANK=%q\n" "$NPF_LASTQUAD_OUTPUT_RANK"
            printf "NPF_LASTQUAD_ACTIVATION=%q\n" "$NPF_LASTQUAD_ACTIVATION"
            printf "NPF_LASTQUAD_SOFTPLUS_BETA=%q\n" "$NPF_LASTQUAD_SOFTPLUS_BETA"
            printf "NPF_LASTQUAD_INIT_EPS=%q\n" "$NPF_LASTQUAD_INIT_EPS"
            printf "NPF_LASTQUAD_IDENTITY_INIT=%q\n" "$NPF_LASTQUAD_IDENTITY_INIT"
            printf "NPF_LASTQUAD_STRONG_CONVEXITY=%q\n" "$NPF_LASTQUAD_STRONG_CONVEXITY"
            printf "NPF_LASTQUAD_POS_WEIGHTS=%q\n" "$NPF_LASTQUAD_POS_WEIGHTS"
            printf "NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER=%q\n" "$NPF_LASTQUAD_POSITIVE_WEIGHT_RECTIFIER"
            printf "BB_ALPHA0=%q\n" "$BB_ALPHA0"
            printf "BB_ALPHA_MIN=%q\n" "$BB_ALPHA_MIN"
            printf "BB_ALPHA_MAX=%q\n" "$BB_ALPHA_MAX"
            printf "BB_LS_C=%q\n" "$BB_LS_C"
            printf "BB_LS_SHRINK=%q\n" "$BB_LS_SHRINK"
            printf "BB_LS_MAX_STEPS=%q\n" "$BB_LS_MAX_STEPS"
            printf "PARAMETRIC_BB_MAX_GRAD_NORM=%q\n" "$PARAMETRIC_BB_MAX_GRAD_NORM"
            printf "PPA_STEP_RULE=%q\n" "$PPA_STEP_RULE"
            printf "PPA_ROUND_ORDER=%q\n" "$PPA_ROUND_ORDER"
            printf "PPA_ROUND0_LAMBDA_DAMPING=%q\n" "$PPA_ROUND0_LAMBDA_DAMPING"
            printf "WRM_STEP_RULE=%q\n" "$WRM_STEP_RULE"
            printf "WRM_INNER_LR=%q\n" "$WRM_INNER_LR"
            printf "PPA_NUM_ROUNDS=%q\n" "$PPA_NUM_ROUNDS"
            printf "PPA_MIN_ROUNDS=%q\n" "$PPA_MIN_ROUNDS"
            printf "PPA_ROUND0_STEPS=%q\n" "${PPA_ROUND0_STEPS:-$KEFF}"
            printf "PPA_REFINE_STEPS=%q\n" "${PPA_REFINE_STEPS:-$KEFF}"
            printf "PPA_ROUND0_LR=%q\n" "$PPA_ROUND0_LR"
            printf "PPA_REFINE_LR=%q\n" "$PPA_REFINE_LR"
            printf "PPA_GAIN_RTOL=%q\n" "$PPA_GAIN_RTOL"
            printf "RESULTS_DIR=%q\n" "$RESULTS_DIR"
            printf "OUT_DIR=%q\n" "$OUT_DIR"
            printf "FINAL_CHECKPOINT=%q\n" "$FINAL_CKPT"
            printf "BEST_ROBUST_CHECKPOINT=%q\n" "$BEST_CKPT"
            printf "CSV=%q\n" "$CSV"
            printf "SUMMARY_JSON=%q\n" "$SUMMARY_JSON"
            printf "HISTORY_JSON=%q\n" "$HISTORY_JSON"
            printf "EXTRA_ARGS=%q\n" "$EXTRA_ARGS"
        } > "$RUN_CONFIG"

        local TIME_CMD=""
        if command -v /usr/bin/time >/dev/null 2>&1; then
            TIME_CMD="/usr/bin/time -v -o ${TIME_FILE}"
        fi

        local -a COMMON_EXTRA_FLAGS=()
        if [ "$USE_MARGIN_LOSS" = "1" ]; then
            COMMON_EXTRA_FLAGS+=(--use-margin-loss)
        else
            # Explicit both ways: the argparse default is margin=True, so a
            # falsy USE_MARGIN_LOSS must pass the negative flag rather than
            # silently inherit the default.
            COMMON_EXTRA_FLAGS+=(--no-use-margin-loss)
        fi
        if [ "$SKIP_PGD_DURING_TRAIN" = "1" ]; then
            COMMON_EXTRA_FLAGS+=(--skip-pgd-during-train)
        fi
        if [ "$BENCHMARK_MODE" = "1" ]; then
            COMMON_EXTRA_FLAGS+=(--benchmark-mode)
        fi
        case "$ATTACK_CLEAN_CORRECT_ONLY" in
            0|false|False|FALSE|no|No|NO|off|Off|OFF)
                COMMON_EXTRA_FLAGS+=(--attack-all-samples)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--attack-clean-correct-only)
                ;;
        esac
        case "$RESET_PARAMETRIC_BB_EACH_BATCH" in
            0|false|False|FALSE|no|No|NO|off|Off|OFF)
                COMMON_EXTRA_FLAGS+=(--persistent-parametric-bb)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--reset-parametric-bb-each-batch)
                ;;
        esac
        case "$FREEZE_BATCHNORM" in
            0|false|False|FALSE|no|No|NO|off|Off|OFF)
                COMMON_EXTRA_FLAGS+=(--no-freeze-batchnorm)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--freeze-batchnorm)
                ;;
        esac
        case "$FREEZE_BATCHNORM_AFFINE" in
            1|true|True|TRUE|yes|Yes|YES|on|On|ON)
                COMMON_EXTRA_FLAGS+=(--freeze-batchnorm-affine)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--no-freeze-batchnorm-affine)
                ;;
        esac
        case "$ADVERSARY_CLASSIFIER_EVAL" in
            0|false|False|FALSE|no|No|NO|off|Off|OFF)
                COMMON_EXTRA_FLAGS+=(--adversary-classifier-train)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--adversary-classifier-eval)
                ;;
        esac
        case "$BATCHNORM_ONLINE_REFRESH" in
            1|true|True|TRUE|yes|Yes|YES|on|On|ON)
                COMMON_EXTRA_FLAGS+=(--online-batchnorm-refresh)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--no-online-batchnorm-refresh)
                ;;
        esac
        if [ -n "$BATCHNORM_ONLINE_REFRESH_MOMENTUM" ]; then
            COMMON_EXTRA_FLAGS+=(--batchnorm-online-refresh-momentum "$BATCHNORM_ONLINE_REFRESH_MOMENTUM")
        fi
        case "$RECALIBRATE_BATCHNORM" in
            1|true|True|TRUE|yes|Yes|YES|on|On|ON)
                COMMON_EXTRA_FLAGS+=(--recalibrate-batchnorm)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--no-recalibrate-batchnorm)
                ;;
        esac
        COMMON_EXTRA_FLAGS+=(--batchnorm-recalibration-batches "$BATCHNORM_RECALIBRATION_BATCHES")
        case "$BATCHNORM_RECALIBRATION_RESET" in
            0|false|False|FALSE|no|No|NO|off|Off|OFF)
                COMMON_EXTRA_FLAGS+=(--no-batchnorm-recalibration-reset)
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--batchnorm-recalibration-reset)
                ;;
        esac
        if [ -n "$BATCHNORM_RECALIBRATION_MOMENTUM" ]; then
            COMMON_EXTRA_FLAGS+=(--batchnorm-recalibration-momentum "$BATCHNORM_RECALIBRATION_MOMENTUM")
        fi
        if [ "$PROFILE_INNER" = "1" ]; then
            COMMON_EXTRA_FLAGS+=(--profile-inner --profile-inner-batches "$PROFILE_INNER_BATCHES")
        fi
        case "$EVAL_TRANSPORT_PGD_ALIGNMENT" in
            1|true|True|TRUE|yes|Yes|YES|on|On|ON)
                COMMON_EXTRA_FLAGS+=(
                    --eval-transport-pgd-alignment
                    --eval-transport-pgd-alignment-samples "$EVAL_TRANSPORT_PGD_ALIGNMENT_SAMPLES"
                )
                ;;
            *)
                COMMON_EXTRA_FLAGS+=(--no-eval-transport-pgd-alignment)
                ;;
        esac
        if [ -n "$LAMBDA_SCHEDULE" ]; then
            LAMBDA_SCHEDULE_RAW="${LAMBDA_SCHEDULE//,/ }"
            read -r -a LAMBDA_SCHEDULE_ARGS <<< "$LAMBDA_SCHEDULE_RAW"
            COMMON_EXTRA_FLAGS+=(--lambda-schedule "${LAMBDA_SCHEDULE_ARGS[@]}")
            if [ "${LAMBDA_STAGE_EPOCHS}" != "0" ]; then
                COMMON_EXTRA_FLAGS+=(--lambda-stage-epochs "$LAMBDA_STAGE_EPOCHS")
            fi
        fi
        if [ -n "$RESUME_CHECKPOINT" ]; then
            COMMON_EXTRA_FLAGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
        fi
        if [ -n "$SAVE_EVERY_EPOCH_DIR" ]; then
            COMMON_EXTRA_FLAGS+=(--save-every-epoch-dir "$SAVE_EVERY_EPOCH_DIR")
        fi
        COMMON_EXTRA_FLAGS+=(--checkpoint-epoch-offset "$CHECKPOINT_EPOCH_OFFSET")
        if [ -n "${WRM_NORMALIZED_RADIUS_PROTOCOL:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-normalized-radius-protocol "$WRM_NORMALIZED_RADIUS_PROTOCOL")
        fi
        if [ -n "${WRM_PGD_EPS_MODE:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-pgd-eps-mode "$WRM_PGD_EPS_MODE")
        fi
        if [ -n "${WRM_NORMALIZED_RADIUS_REQUESTED:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-normalized-radius-requested "$WRM_NORMALIZED_RADIUS_REQUESTED")
        fi
        if [ -n "${WRM_NORMALIZED_RADIUS:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-normalized-radius "$WRM_NORMALIZED_RADIUS")
        fi
        if [ -n "${WRM_CP:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-cp "$WRM_CP")
        fi
        if [ -n "${WRM_EFFECTIVE_INP_EPS:-}" ]; then
            COMMON_EXTRA_FLAGS+=(--wrm-effective-inp-eps "$WRM_EFFECTIVE_INP_EPS")
        fi

        # The shared --bb-* knobs are passed to every algorithm. They
        # are consumed only by methods using BB+Armijo (NPF, NN-DRO,
        # WRM, WFR, New_PPA); Madry/RO and Dual parse them and ignore
        # them because they use PGD and Langevin/MALA respectively.
        $TIME_CMD torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
            -m pretrained_input_icnn.main \
            --algorithm "$ALGO" \
            --pretrained-path "$PRETRAINED_PATH" \
            --data-dir "$DATA_DIR" \
            --epochs-adv "$EPOCHS_ADV" \
            --epochs-icnn-pretrain "$EPOCHS_ICNN_PRETRAIN" \
            --frozen-adversary-epochs "$FROZEN_ADVERSARY_EPOCHS" \
            --frozen-adversary-map-steps "$FROZEN_ADVERSARY_MAP_STEPS" \
            --batch-size "$BATCH_OVR" \
            --num-workers "$NUM_WORKERS" \
            --lr-theta "$LR_THETA" \
            --penalty-lambda "$PENALTY_LAMBDA" \
            --transport-cost "$TRANSPORT_COST" \
            --inp-p "$INP_P" \
            --inp-eps "$INP_EPS" \
            --inp-steps "$INP_STEPS" \
            --inp-restarts "$INP_RESTARTS" \
            --eval-input-pgd \
            --eval-input-pgd-samples "$EVAL_PGD_SAMPLES" \
            --eval-transport-samples "$EVAL_TRANSPORT_SAMPLES" \
            --input-pgd-loss "$INPUT_PGD_LOSS" \
            --input-pgd-geometry "$INPUT_PGD_GEOMETRY" \
            "${COMMON_EXTRA_FLAGS[@]}" \
            "${BB_ARGS[@]}" \
            "${NPF_OPT_ARGS[@]}" \
            --seed "$SEED" \
            --save "$FINAL_CKPT" \
            --save-best-robust "$BEST_CKPT" \
            --log-csv "$CSV" \
            $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"

        STATUS=${PIPESTATUS[0]}
        T1=$(date +%s)
        ENDED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        ELAPSED=$((T1 - T0))

        if [ $STATUS -eq 0 ]; then break; fi

        # OOM detection
        if grep -q -E "CUDA out of memory|OutOfMemoryError" "$LOG_FILE" 2>/dev/null; then
            local NEW_BATCH=$((BATCH_OVR / 2))
            if [ $NEW_BATCH -lt $((NPROC * 4)) ]; then
                echo "[FATAL OOM] ${ALGO}: cannot halve batch below ${NPROC}*4 = $((NPROC*4))"
                break
            fi
            echo "[OOM] retrying with batch_size ${BATCH_OVR} -> ${NEW_BATCH}"
            mv "$LOG_FILE" "${LOG_FILE}.attempt${ATTEMPT}"
            BATCH_OVR=$NEW_BATCH
            rm -rf "$OUT_DIR"
            mkdir -p "$OUT_DIR"
            continue
        fi

        echo "[FAIL] ${ALGO} seed=${SEED} non-OOM error (status=${STATUS}); see ${LOG_FILE}"
        break
    done

    if [ $STATUS -eq 0 ]; then
        touch "$DONE"
        printf "completed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$RUN_NAME" "$ALGO" "$SEED" "$NPROC" "$K" "$EPOCHS_ADV" \
            "$STARTED_UTC" "$ENDED_UTC" "$ELAPSED" "$OUT_DIR" \
            "$FINAL_CKPT" "$BEST_CKPT" "$LOG_FILE" "$TIME_FILE" "$CSV" \
            "$SUMMARY_JSON" "$HISTORY_JSON" >> "$MANIFEST"
        echo "[OK] ${ALGO} seed=${SEED} elapsed=${ELAPSED}s"
        echo "[OK] last checkpoint: ${FINAL_CKPT}"
        echo "[OK] best robust checkpoint: ${BEST_CKPT}"
    else
        printf "failed_%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$STATUS" "$RUN_NAME" "$ALGO" "$SEED" "$NPROC" "$K" "$EPOCHS_ADV" \
            "$STARTED_UTC" "$ENDED_UTC" "$ELAPSED" "$OUT_DIR" \
            "$FINAL_CKPT" "$BEST_CKPT" "$LOG_FILE" "$TIME_FILE" "$CSV" \
            "$SUMMARY_JSON" "$HISTORY_JSON" >> "$MANIFEST"
    fi

    # Reset CUDA caches between algorithms (each is its own torchrun
    # process, so memory IS released — this is just diagnostic).
    sleep 2
}


# ======================================================================
# Split dispatcher
# ======================================================================
DID_RUN_ALGO=0
if [ -n "$RUN_ONLY_ALGO" ]; then
    case $SPLIT in
        0|1)
            run_algo "$RUN_ONLY_ALGO"
            DID_RUN_ALGO=1
            ;;
        2|3|4)
            echo "[SKIP] RUN_ONLY_ALGO=${RUN_ONLY_ALGO}; split ${SPLIT} has no assigned work."
            ;;
        *)
            echo "[FATAL] Unknown SPLIT=${SPLIT}. Use 0 (sequential) or 1-4."
            exit 1
            ;;
    esac
else
    case $SPLIT in
        0|1)
            for algo in "${ACTIVE_ALGOS[@]}"; do
                run_algo "$algo"
                DID_RUN_ALGO=1
            done
            ;;
        2|3|4)
            echo "[SKIP] Only NPF-LastQuad is enabled; split ${SPLIT} has no assigned work."
            ;;
        # Full runtime-sweep split map is intentionally disabled:
        # 1) npf
        # 2) nn_dro, madry
        # 3) wrm, wfr
        # 4) dual, new_ppa
        *)
            echo "[FATAL] Unknown SPLIT=${SPLIT}. Use 0 (sequential) or 1-4."
            exit 1
            ;;
    esac
fi

echo ""
echo "================================================================"
echo "  Split ${SPLIT} complete at $(date)"
echo "================================================================"

if [ "$DID_RUN_ALGO" -eq 0 ]; then
    exit 0
fi

# ======================================================================
# Aggregator: prints a clean comparison table + LaTeX once all active
# algorithms in the current seed have finished.
# ======================================================================
SPLIT="$SPLIT" SEED="$SEED" RESULTS_DIR="$RESULTS_DIR" K="$K" \
ACTIVE_ALGOS="$ACTIVE_ALGOS_CSV" \
NPROC="$NPROC" EPOCHS_ADV="$EPOCHS_ADV" python - << 'PYTHON_COLLECTOR'
import csv, json, os, statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ["RESULTS_DIR"])
SEED        = int(os.environ["SEED"])
K           = int(os.environ["K"])
NPROC       = int(os.environ["NPROC"])
EPOCHS_ADV  = int(os.environ["EPOCHS_ADV"])
ALGOS       = [
    a for a in os.environ.get(
        "ACTIVE_ALGOS",
        "npf_lastquad",
    ).replace(",", " ").split()
    if a
]

manifest = RESULTS_DIR / "run_manifest.tsv"
done = set()
if manifest.exists():
    with manifest.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("status", "").startswith("completed") and int(r["seed"]) == SEED:
                done.add(r["algorithm"])

print(f"\n[COLLECTOR] {len(done)}/{len(ALGOS)} completed at SEED={SEED}, K={K}, GPUs={NPROC}")
if len(done) < len(ALGOS):
    print(f"[COLLECTOR] Missing: {sorted(set(ALGOS) - done)}")
    print("[COLLECTOR] Re-run after remaining splits complete for full table.")
    raise SystemExit(0)

def per_epoch_stats(csv_path: Path):
    if not csv_path.exists():
        return None
    epoch_secs, train_losses, train_accs = [], [], []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("phase", "adv") != "adv":
                continue
            try:
                ep = int(r["epoch"]); t = float(r["epoch_seconds"])
            except (KeyError, ValueError, TypeError):
                continue
            if ep == 1:  # drop cuDNN autotune warmup epoch
                continue
            epoch_secs.append(t)
            try:    train_losses.append(float(r["train_loss"]))
            except: pass
            try:    train_accs.append(float(r["train_acc"]))
            except: pass
    if not epoch_secs:
        return None
    return {
        "median_ep_s": statistics.median(epoch_secs),
        "min_ep_s":    min(epoch_secs),
        "max_ep_s":    max(epoch_secs),
        "n_epochs":    len(epoch_secs),
        "final_loss":  train_losses[-1] if train_losses else None,
        "final_acc":   train_accs[-1]   if train_accs   else None,
    }

def parse_time_v(p: Path):
    if not p.exists(): return {}
    out = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            out[k.strip()] = v.strip()
    return out

results = {}
for a in ALGOS:
    csvp = RESULTS_DIR / a / f"seed_{SEED}" / "epoch_log.csv"
    timep = RESULTS_DIR / "time" / f"{a}_seed{SEED}.time.txt"
    s = per_epoch_stats(csvp) or {}
    t = parse_time_v(timep)
    results[a] = {
        **s,
        "wallclock":  t.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "—"),
        "max_rss_kb": t.get("Maximum resident set size (kbytes)", "—"),
    }

print()
print("=" * 110)
print(f"  Input-space Adversarial Training — Runtime  (CIFAR-10, K={K}, seed {SEED}, {NPROC}-GPU DDP, {EPOCHS_ADV} ep)")
print("=" * 110)
hdr = (f"{'Algorithm':<10}  {'med ep (s)':>12}  {'min ep (s)':>12}  "
       f"{'max ep (s)':>12}  {'n ep':>5}  {'wallclock':>12}  {'RSS (MB)':>10}  "
       f"{'final_loss':>11}  {'final_acc':>10}")
print(hdr); print("-" * len(hdr))
for a in ALGOS:
    r = results[a]
    med = f"{r.get('median_ep_s', float('nan')):.3f}" if r.get("median_ep_s") is not None else "—"
    mn  = f"{r.get('min_ep_s', float('nan')):.3f}"    if r.get("min_ep_s")    is not None else "—"
    mx  = f"{r.get('max_ep_s', float('nan')):.3f}"    if r.get("max_ep_s")    is not None else "—"
    n   = r.get("n_epochs", "—")
    wc  = r.get("wallclock", "—")
    try:    rss = f"{float(r['max_rss_kb'])/1024:.1f}"
    except: rss = "—"
    fl  = f"{r.get('final_loss', float('nan')):.3f}" if r.get("final_loss") is not None else "—"
    fa  = f"{100*r.get('final_acc'):.2f}%" if r.get("final_acc") is not None else "—"
    print(f"{a:<10}  {med:>12}  {mn:>12}  {mx:>12}  {n!s:>5}  {wc:>12}  {rss:>10}  {fl:>11}  {fa:>10}")

# LaTeX table
print()
print("% ===== LaTeX table (paper-ready) =====")
print(r"\begin{table}[t] \centering")
print(f"\\caption{{Per-epoch wallclock for input-space adversarial training on CIFAR-10 (K={K}, "
      f"{NPROC}-GPU DDP, seed {SEED}). Median over {EPOCHS_ADV - 1} steady-state epochs (epoch 1 dropped).}}")
print(r"\label{tab:input_icnn_runtime}")
print(r"\begin{tabular}{l rrr r}")
print(r"\toprule")
print(r"Algorithm & Median (s) & Min (s) & Max (s) & Wallclock \\")
print(r"\midrule")
for a in ALGOS:
    r = results[a]
    med = f"{r['median_ep_s']:.2f}" if r.get("median_ep_s") is not None else "—"
    mn  = f"{r['min_ep_s']:.2f}"    if r.get("min_ep_s")    is not None else "—"
    mx  = f"{r['max_ep_s']:.2f}"    if r.get("max_ep_s")    is not None else "—"
    print(f"{a} & {med} & {mn} & {mx} & {r.get('wallclock', '—')} \\\\")
print(r"\bottomrule \end{tabular} \end{table}")

(RESULTS_DIR / f"summary_seed{SEED}.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: {RESULTS_DIR / f'summary_seed{SEED}.json'}")
PYTHON_COLLECTOR
