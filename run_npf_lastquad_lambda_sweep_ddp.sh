#!/bin/bash
# =============================================================================
# NPF-LastQuad lambda ablation sweep
# =============================================================================
#
# This wrapper launches a controlled lambda grid through run_runtime_sweep_ddp.sh.
# It has two modes:
#
#   FREEZE_THETA=1  Train only the NPF map for each lambda against the same
#                   frozen classifier. This is the cleanest setting for
#                   studying T_lambda as a function of lambda. In this mode
#                   ADV_EPOCHS must be 0; WARMUP_EPOCHS defaults to MAP_EPOCHS.
#
#   FREEZE_THETA=0  Run ordinary adversarial training for each lambda. This is
#                   useful as a secondary "does the effect survive end-to-end?"
#                   check, but T_lambda is then confounded with classifier drift.
#                   In this mode ADV_EPOCHS defaults to MAP_EPOCHS and
#                   WARMUP_EPOCHS defaults to EPOCHS_ICNN_PRETRAIN or 0.
#
# Optional post phase:
#   FROZEN_ADVERSARY_EPOCHS=10 FROZEN_ADVERSARY_MAP_STEPS=2
#                   After ordinary adversarial training, freeze the learned
#                   NPF map and continue classifier-only training on
#                   T_omega^2(x). Aliases POST_MAP_EPOCHS and POST_MAP_STEPS
#                   are also accepted.
#
# Example:
#   LAMBDAS="3 5 10 20 30 60" \
#   FREEZE_THETA=1 MAP_EPOCHS=50 \
#   LR_THETA=0.003 USE_MARGIN_LOSS=0 NPF_INNER_OPTIMIZER=muon NPF_MUON_LR=2e-4 \
#   bash run_npf_lastquad_lambda_sweep_ddp.sh 1 4 20
#
# Arguments:
#   SEED  default 1
#   NPROC default 4
#   K     default 20
# =============================================================================

SEED=${1:-${SEED:-1}}
NPROC=${2:-${NPROC:-4}}
K=${3:-${K:-20}}

SRC_DIR="${SRC_DIR:-/mloscratch/homes/aabdolla/LAT}"
RESULTS_ROOT="${RESULTS_ROOT:-${SRC_DIR}/input_icnn_ddp_runs}"
LAMBDAS=${LAMBDAS:-"3 5 10 20 30 60"}
MAP_EPOCHS=${MAP_EPOCHS:-50}
FREEZE_THETA=${FREEZE_THETA:-1}
FROZEN_ADVERSARY_EPOCHS=${FROZEN_ADVERSARY_EPOCHS:-${FROZEN_MAP_EPOCHS:-${POST_MAP_EPOCHS:-0}}}
FROZEN_ADVERSARY_MAP_STEPS=${FROZEN_ADVERSARY_MAP_STEPS:-${FROZEN_MAP_STEPS:-${POST_MAP_STEPS:-1}}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-npf_lastquad_lambda_ablation}
SKIP_PGD_DURING_TRAIN=${SKIP_PGD_DURING_TRAIN:-1}
EVAL_PGD_SAMPLES=${EVAL_PGD_SAMPLES:-1000}

sanitize_float() {
    printf "%s" "$1" | sed -e 's/-/m/g' -e 's/\./p/g' -e 's/+//g'
}

is_nonnegative_int() {
    case "$1" in
        ""|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

for name in SEED NPROC K MAP_EPOCHS FROZEN_ADVERSARY_EPOCHS FROZEN_ADVERSARY_MAP_STEPS; do
    value="${!name}"
    if ! is_nonnegative_int "$value"; then
        echo "[FATAL] ${name} must be a non-negative integer, got '${value}'."
        exit 1
    fi
done
if [ "$NPROC" -lt 1 ]; then
    echo "[FATAL] NPROC must be >= 1, got '${NPROC}'."
    exit 1
fi
if [ "$K" -lt 1 ]; then
    echo "[FATAL] K must be >= 1, got '${K}'."
    exit 1
fi
if [ "$FROZEN_ADVERSARY_MAP_STEPS" -lt 1 ]; then
    echo "[FATAL] FROZEN_ADVERSARY_MAP_STEPS must be >= 1, got '${FROZEN_ADVERSARY_MAP_STEPS}'."
    exit 1
fi

case "$FREEZE_THETA" in
    1|true|True|TRUE|yes|Yes|YES)
        FREEZE_THETA=1
        MODE_TAG="fixedtheta"
        ADV_EPOCHS=${ADV_EPOCHS:-0}
        WARMUP_EPOCHS=${WARMUP_EPOCHS:-${EPOCHS_ICNN_PRETRAIN:-$MAP_EPOCHS}}
        if [ "$ADV_EPOCHS" != "0" ]; then
            echo "[FATAL] FREEZE_THETA=1 means classifier theta stays frozen, so ADV_EPOCHS must be 0."
            echo "        Use WARMUP_EPOCHS=<n> for map-only epochs, or set FREEZE_THETA=0 for adversarial-training epochs."
            exit 1
        fi
        ;;
    0|false|False|FALSE|no|No|NO)
        FREEZE_THETA=0
        MODE_TAG="end2end"
        ADV_EPOCHS=${ADV_EPOCHS:-$MAP_EPOCHS}
        WARMUP_EPOCHS=${WARMUP_EPOCHS:-${EPOCHS_ICNN_PRETRAIN:-0}}
        ;;
    *)
        echo "[FATAL] FREEZE_THETA must be 0/1, true/false, or yes/no; got '${FREEZE_THETA}'."
        exit 1
        ;;
esac

for name in ADV_EPOCHS WARMUP_EPOCHS; do
    value="${!name}"
    if ! is_nonnegative_int "$value"; then
        echo "[FATAL] ${name} must be a non-negative integer, got '${value}'."
        exit 1
    fi
done
if [ "$ADV_EPOCHS" -eq 0 ] && [ "$WARMUP_EPOCHS" -eq 0 ]; then
    echo "[FATAL] Both ADV_EPOCHS and WARMUP_EPOCHS are 0; nothing would train."
    exit 1
fi

echo ""
echo "================================================================"
echo "  NPF-LastQuad lambda sweep"
echo "  mode=${MODE_TAG}  seed=${SEED}  GPUs=${NPROC}  K=${K}"
echo "  lambdas=${LAMBDAS}"
echo "  map_epochs=${MAP_EPOCHS}  adv_epochs=${ADV_EPOCHS}  warmup_epochs=${WARMUP_EPOCHS}"
echo "  frozen_adversary_epochs=${FROZEN_ADVERSARY_EPOCHS}  frozen_adversary_map_steps=${FROZEN_ADVERSARY_MAP_STEPS}"
echo "  results_root=${RESULTS_ROOT}"
echo "================================================================"

for LAM in $LAMBDAS; do
    LAM_TAG=$(sanitize_float "$LAM")
    FROZEN_TAG=""
    if [ "$FROZEN_ADVERSARY_EPOCHS" != "0" ]; then
        FROZEN_TAG="_frozenadv${FROZEN_ADVERSARY_EPOCHS}x${FROZEN_ADVERSARY_MAP_STEPS}"
    fi
    OUTPUT_FOLDER_NAME="${OUTPUT_PREFIX}_${MODE_TAG}_lam${LAM_TAG}_seed${SEED}_K${K}_warm${WARMUP_EPOCHS}_adv${ADV_EPOCHS}${FROZEN_TAG}"
    echo ""
    echo "----------------------------------------------------------------"
    echo "  lambda=${LAM} -> ${OUTPUT_FOLDER_NAME}"
    echo "----------------------------------------------------------------"
    PENALTY_LAMBDA="$LAM" \
    OUTPUT_FOLDER_NAME="$OUTPUT_FOLDER_NAME" \
    RESULTS_ROOT="$RESULTS_ROOT" \
    EPOCHS_ICNN_PRETRAIN="$WARMUP_EPOCHS" \
    FROZEN_ADVERSARY_EPOCHS="$FROZEN_ADVERSARY_EPOCHS" \
    FROZEN_ADVERSARY_MAP_STEPS="$FROZEN_ADVERSARY_MAP_STEPS" \
    SKIP_PGD_DURING_TRAIN="$SKIP_PGD_DURING_TRAIN" \
    EVAL_PGD_SAMPLES="$EVAL_PGD_SAMPLES" \
    bash "${SRC_DIR}/run_runtime_sweep_ddp.sh" 0 "$SEED" "$NPROC" "$K" "$ADV_EPOCHS"
done

echo ""
echo "================================================================"
echo "  Lambda sweep complete"
echo "  Analyze with:"
echo "    python -m pretrained_input_icnn.lambda_ablation \\"
echo "      --runs-root ${RESULTS_ROOT} \\"
echo "      --run-name-contains ${OUTPUT_PREFIX}_${MODE_TAG}_ \\"
echo "      --checkpoint-kind last --seed-filter --seed ${SEED} \\"
echo "      --data-dir ${SRC_DIR}/data \\"
echo "      --output-dir ${RESULTS_ROOT}/${OUTPUT_PREFIX}_${MODE_TAG}_analysis_seed${SEED}"
echo "================================================================"
