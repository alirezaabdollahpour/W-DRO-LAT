#!/usr/bin/env bash
set -euo pipefail

# Drives the new pretrained_input_icnn package.
#
# Usage:
#   bash run_pretrained_input_icnn.sh                     # NPF (default)
#   ALGORITHM=madry bash run_pretrained_input_icnn.sh     # baseline competitor
#   ALGORITHM=nn_dro EPOCHS_ADV=20 bash run_pretrained_input_icnn.sh
#
# Environment overrides (defaults shown):
#   ALGORITHM             npf
#   PRETRAINED_PATH       /mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth
#   EPOCHS_ADV            30
#   BATCH_SIZE            512
#   PENALTY_LAMBDA        30
#   LR_THETA              0.1
#   OMEGA_STEPS           10        (NPF / NN-DRO)
#   INP_P                 2
#   INP_EPS               0.5
#   USE_MARGIN_LOSS       0         (set to 1 to use logsumexp margin objective
#                                    for the adversary on NPF/NN-DRO/WRM/Madry/PPA)
#   EPOCHS_ICNN_PRETRAIN  0         (warmup: train only the adversary (NPF/NN-DRO)
#                                    for this many epochs before the regular minimax
#                                    schedule kicks in; classifier stays frozen)
#
# The script forwards algorithm-specific defaults from
# Logistic_Regression_CIFAR10/config.py and Runtime-LR-CIFAR10/run_runtime_lr_cifar10.sh,
# so e.g. NPF uses npf_outer_rank=8, npf_inner_rank=2, init_eps=1e-4.

ALGORITHM="${ALGORITHM:-npf}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/lts4/scratch/students/aabdolla/LAT/ResNet_checkpoints/R2.pth}"
EPOCHS_ADV="${EPOCHS_ADV:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PENALTY_LAMBDA="${PENALTY_LAMBDA:-30}"
LR_THETA="${LR_THETA:-0.1}"
OMEGA_STEPS="${OMEGA_STEPS:-10}"
INP_P="${INP_P:-2}"
INP_EPS="${INP_EPS:-0.5}"
INP_STEPS="${INP_STEPS:-20}"
INP_RESTARTS="${INP_RESTARTS:-5}"
EVAL_PGD_SAMPLES="${EVAL_PGD_SAMPLES:-1000}"
SEED="${SEED:-1}"
USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-1}"
EPOCHS_ICNN_PRETRAIN="${EPOCHS_ICNN_PRETRAIN:-0}"

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
      --npf-bb-alpha0 "${NPF_BB_ALPHA0:-2e-4}"
      --npf-bb-alpha-min "${NPF_BB_ALPHA_MIN:-1e-7}"
      --npf-bb-alpha-max "${NPF_BB_ALPHA_MAX:-0.25}"
      --npf-bb-ls-c "${NPF_BB_LS_C:-1e-4}"
      --npf-bb-ls-shrink "${NPF_BB_LS_SHRINK:-0.5}"
      --npf-bb-ls-max-steps "${NPF_BB_LS_MAX_STEPS:-15}"
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
      --wrm-inner-steps "${WRM_INNER_STEPS:-100}"
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
    echo "Choose one of: npf nn_dro madry wrm wfr dual new_ppa" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${SAVE_PATH}")"

python -m pretrained_input_icnn.main \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
