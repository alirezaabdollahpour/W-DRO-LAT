#!/usr/bin/env bash
set -euo pipefail

# Drives the new pretrained_input_icnn package.
#
# Usage:
#   bash run_pretrained_input_icnn.sh                     # npf_lastquad (default)
#   ALGORITHM=madry bash run_pretrained_input_icnn.sh     # baseline competitor
#   ALGORITHM=nn_dro EPOCHS_ADV=20 bash run_pretrained_input_icnn.sh
#
# Environment overrides (defaults shown):
#   ALGORITHM             npf_lastquad (default), npf, nn_dro, madry, wrm, wfr, dual, new_ppa
#   PRETRAINED_PATH       /mloscratch/homes/aabdolla/LAT/ResNet_checkpoints/R2.pth
#   EPOCHS_ADV            50
#   BATCH_SIZE            512
#   PENALTY_LAMBDA        10
#   LR_THETA              0.003
#   OMEGA_STEPS           20        (NPF / NN-DRO)
#   NPF_LASTQUAD_HIDDEN   "1024 512 512 256 128 64"
#   INP_P                 2
#   INP_EPS               0.5
#   USE_MARGIN_LOSS       0         (set to 1 to use logsumexp margin objective
#                                    for the adversary on NPF/NN-DRO/WRM/Madry/PPA)
#   FREEZE_BATCHNORM      1         (keep BN running stats fixed during adversarial updates)
#   RECALIBRATE_BATCHNORM 0         (after each adversarial epoch, recompute BN stats on clean train data)
#   BATCHNORM_RECALIBRATION_BATCHES 0  (0 = full clean train pass)
#   BATCHNORM_RECALIBRATION_RESET 1    (1 = exact fresh stats; 0 = refresh existing stats)
#   BATCHNORM_RECALIBRATION_MOMENTUM "" (empty = cumulative average)
#   PROFILE_INNER         0         (set to 1 for synchronized inner-loop timing)
#   PROFILE_INNER_BATCHES 0         (0 = profile every train batch)
#   EPOCHS_ICNN_PRETRAIN  0         (warmup: train only the adversary (NPF/NN-DRO)
#                                    for this many epochs before the regular minimax
#                                    schedule kicks in; classifier stays frozen)
#
# The script forwards algorithm-specific defaults from
# Logistic_Regression_CIFAR10/config.py and Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh,
# so e.g. NPF uses npf_outer_rank=8, npf_inner_rank=2, init_eps=1e-4.
# The default npf_lastquad run uses only the final diagonal quadratic term.

ALGORITHM="${ALGORITHM:-npf_lastquad}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mloscratch/homes/aabdolla/LAT/ResNet_checkpoints/R2.pth}"
EPOCHS_ADV="${EPOCHS_ADV:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PENALTY_LAMBDA="${PENALTY_LAMBDA:-10}"
LR_THETA="${LR_THETA:-0.003}"
OMEGA_STEPS="${OMEGA_STEPS:-20}"
INP_P="${INP_P:-2}"
INP_EPS="${INP_EPS:-0.5}"
INP_STEPS="${INP_STEPS:-20}"
INP_RESTARTS="${INP_RESTARTS:-5}"
EVAL_PGD_SAMPLES="${EVAL_PGD_SAMPLES:-1000}"
SEED="${SEED:-1}"
USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-0}"
FREEZE_BATCHNORM="${FREEZE_BATCHNORM:-1}"
RECALIBRATE_BATCHNORM="${RECALIBRATE_BATCHNORM:-0}"
BATCHNORM_RECALIBRATION_BATCHES="${BATCHNORM_RECALIBRATION_BATCHES:-0}"
BATCHNORM_RECALIBRATION_RESET="${BATCHNORM_RECALIBRATION_RESET:-1}"
BATCHNORM_RECALIBRATION_MOMENTUM="${BATCHNORM_RECALIBRATION_MOMENTUM:-}"
PROFILE_INNER="${PROFILE_INNER:-0}"
PROFILE_INNER_BATCHES="${PROFILE_INNER_BATCHES:-0}"
SKIP_PGD_DURING_TRAIN="${SKIP_PGD_DURING_TRAIN:-0}"
BENCHMARK_MODE="${BENCHMARK_MODE:-0}"
EPOCHS_ICNN_PRETRAIN="${EPOCHS_ICNN_PRETRAIN:-0}"
NPF_LASTQUAD_HIDDEN="${NPF_LASTQUAD_HIDDEN:-1024 512 512 256 128 64}"
read -r -a NPF_LASTQUAD_HIDDEN_ARGS <<< "${NPF_LASTQUAD_HIDDEN}"

# Per-algorithm save tag.
SAVE_PATH="${SAVE_PATH:-results/${ALGORITHM}_lambda${PENALTY_LAMBDA}_${EPOCHS_ADV}ep.pth}"
LOG_CSV="${LOG_CSV:-./runs_log_input_icnn.csv}"

COMMON_ARGS=(
  --algorithm "${ALGORITHM}"
  --pretrained-path "${PRETRAINED_PATH}"
  --epochs-adv "${EPOCHS_ADV}"
  --epochs-icnn-pretrain "${EPOCHS_ICNN_PRETRAIN}"
  --batch-size "${BATCH_SIZE}"
  --lr-theta "${LR_THETA}"
  --penalty-lambda "${PENALTY_LAMBDA}"
  --inp-p "${INP_P}"
  --inp-eps "${INP_EPS}"
  --inp-steps "${INP_STEPS}"
  --inp-restarts "${INP_RESTARTS}"
  --eval-input-pgd
  --eval-input-pgd-samples "${EVAL_PGD_SAMPLES}"
  --seed "${SEED}"
  --save "${SAVE_PATH}"
  --log-csv "${LOG_CSV}"
)
if [[ "${USE_MARGIN_LOSS}" == "1" ]]; then
  COMMON_ARGS+=(--use-margin-loss)
fi
case "${FREEZE_BATCHNORM}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--no-freeze-batchnorm)
    ;;
  *)
    COMMON_ARGS+=(--freeze-batchnorm)
    ;;
esac
case "${RECALIBRATE_BATCHNORM}" in
  1|true|True|TRUE|yes|Yes|YES)
    COMMON_ARGS+=(--recalibrate-batchnorm)
    ;;
  *)
    COMMON_ARGS+=(--no-recalibrate-batchnorm)
    ;;
esac
COMMON_ARGS+=(--batchnorm-recalibration-batches "${BATCHNORM_RECALIBRATION_BATCHES}")
case "${BATCHNORM_RECALIBRATION_RESET}" in
  0|false|False|FALSE|no|No|NO)
    COMMON_ARGS+=(--no-batchnorm-recalibration-reset)
    ;;
  *)
    COMMON_ARGS+=(--batchnorm-recalibration-reset)
    ;;
esac
if [[ -n "${BATCHNORM_RECALIBRATION_MOMENTUM}" ]]; then
  COMMON_ARGS+=(--batchnorm-recalibration-momentum "${BATCHNORM_RECALIBRATION_MOMENTUM}")
fi
if [[ "${PROFILE_INNER}" == "1" ]]; then
  COMMON_ARGS+=(--profile-inner --profile-inner-batches "${PROFILE_INNER_BATCHES}")
fi
if [[ "${SKIP_PGD_DURING_TRAIN}" == "1" ]]; then
  COMMON_ARGS+=(--skip-pgd-during-train)
fi
if [[ "${BENCHMARK_MODE}" == "1" ]]; then
  COMMON_ARGS+=(--benchmark-mode)
fi

case "${ALGORITHM}" in
  npf)
    EXTRA_ARGS=(
      --omega-steps-per-batch "${OMEGA_STEPS}"
      --npf-hidden 1024 512 512 256 128 64
      --npf-outer-rank "${NPF_OUTER_RANK:-8}"
      --npf-inner-rank "${NPF_INNER_RANK:-2}"
      --npf-activation "${NPF_ACTIVATION:-softplus}"
      --npf-softplus-beta "${NPF_SOFTPLUS_BETA:-10.0}"
      --npf-init-eps "${NPF_INIT_EPS:-1e-4}"
      --npf-strong-convexity "${NPF_STRONG_CONVEXITY:-1.0}"
      --bb-alpha0 "${NPF_BB_ALPHA0:-2e-4}"
      --bb-alpha-min "${NPF_BB_ALPHA_MIN:-1e-7}"
      --bb-alpha-max "${NPF_BB_ALPHA_MAX:-0.25}"
      --bb-ls-c "${NPF_BB_LS_C:-1e-4}"
      --bb-ls-shrink "${NPF_BB_LS_SHRINK:-0.5}"
      --bb-ls-max-steps "${NPF_BB_LS_MAX_STEPS:-15}"
    )
    ;;
  npf_lastquad)
    EXTRA_ARGS=(
      --omega-steps-per-batch "${OMEGA_STEPS}"
      --npf-lastquad-hidden "${NPF_LASTQUAD_HIDDEN_ARGS[@]}"
      --npf-lastquad-activation "${NPF_LASTQUAD_ACTIVATION:-${NPF_ACTIVATION:-softplus}}"
      --npf-lastquad-elu-alpha "${NPF_LASTQUAD_ELU_ALPHA:-1.0}"
      --npf-lastquad-softplus-beta "${NPF_LASTQUAD_SOFTPLUS_BETA:-${NPF_SOFTPLUS_BETA:-10.0}}"
      --npf-lastquad-init-eps "${NPF_LASTQUAD_INIT_EPS:-${NPF_INIT_EPS:-1e-4}}"
      --npf-lastquad-strong-convexity "${NPF_LASTQUAD_STRONG_CONVEXITY:-${NPF_STRONG_CONVEXITY:-1.0}}"
      --bb-alpha0 "${NPF_LASTQUAD_BB_ALPHA0:-${NPF_BB_ALPHA0:-2e-4}}"
      --bb-alpha-min "${NPF_LASTQUAD_BB_ALPHA_MIN:-${NPF_BB_ALPHA_MIN:-1e-7}}"
      --bb-alpha-max "${NPF_LASTQUAD_BB_ALPHA_MAX:-${NPF_BB_ALPHA_MAX:-0.25}}"
      --bb-ls-c "${NPF_LASTQUAD_BB_LS_C:-${NPF_BB_LS_C:-1e-4}}"
      --bb-ls-shrink "${NPF_LASTQUAD_BB_LS_SHRINK:-${NPF_BB_LS_SHRINK:-0.5}}"
      --bb-ls-max-steps "${NPF_LASTQUAD_BB_LS_MAX_STEPS:-${NPF_BB_LS_MAX_STEPS:-15}}"
    )
    ;;
  nn_dro)
    EXTRA_ARGS=(
      --omega-steps-per-batch "${OMEGA_STEPS}"
      --nn-dro-hidden 512 512 256 256 128
      --nn-dro-activation relu
      --nn-dro-softplus-beta "${NN_DRO_SOFTPLUS_BETA:-20.0}"
      --nn-dro-init-scale "${NN_DRO_INIT_SCALE:-1e-3}"
      --nn-dro-inner-lr "${NN_DRO_INNER_LR:-1e-2}"
    )
    ;;
  madry)
    EXTRA_ARGS=(
      --madry-epsilon "${MADRY_EPSILON:-${INP_EPS}}"
      --madry-pgd-steps "${MADRY_PGD_STEPS:-10}"
      --madry-pgd-restarts "${MADRY_PGD_RESTARTS:-1}"
    )
    ;;
  wrm)
    EXTRA_ARGS=(
      --wrm-inner-steps "${WRM_INNER_STEPS:-${OMEGA_STEPS}}"
      --wrm-inner-lr "${WRM_INNER_LR:-1e-2}"
    )
    ;;
  wfr)
    EXTRA_ARGS=(
      --wfr-epsilon "${WFR_EPSILON:-0.1}"
      --wfr-num-samples "${WFR_NUM_SAMPLES:-8}"
      --wfr-inner-steps "${WFR_INNER_STEPS:-50}"
      --wfr-inner-lr "${WFR_INNER_LR:-1e-2}"
    )
    ;;
  dual)
    EXTRA_ARGS=(
      --dual-epsilon "${DUAL_EPSILON:-1e-3}"
      --dual-sample-level "${DUAL_SAMPLE_LEVEL:-5}"
    )
    ;;
  new_ppa)
    EXTRA_ARGS=(
      --ppa-num-rounds "${PPA_NUM_ROUNDS:-5}"
      --ppa-min-rounds "${PPA_MIN_ROUNDS:-2}"
      --ppa-round0-steps "${PPA_ROUND0_STEPS:-30}"
      --ppa-round0-lr "${PPA_ROUND0_LR:-1e-2}"
      --ppa-refine-steps "${PPA_REFINE_STEPS:-15}"
      --ppa-refine-lr "${PPA_REFINE_LR:-5e-3}"
      --ppa-gain-rtol "${PPA_GAIN_RTOL:-1e-4}"
    )
    ;;
  *)
    echo "ERROR: unknown ALGORITHM=${ALGORITHM}" >&2
    echo "Choose one of: npf npf_lastquad nn_dro madry wrm wfr dual new_ppa" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${SAVE_PATH}")"

python -m pretrained_input_icnn.main \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
