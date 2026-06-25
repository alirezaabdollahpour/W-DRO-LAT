#!/bin/bash
# =============================================================================
# NPF-LastQuad native lambda scheduling experiment
# =============================================================================
#
# Runs one continuous adversarial-training job with a piecewise-constant
# lambda schedule. Unlike the old staged wrapper, this does NOT restart Python
# between lambda changes, so classifier optimizer state, cosine LR scheduler
# state, Muon/BB state, and DDP sampler epoch progression remain continuous.
#
# Defaults:
#   lambdas:            5 10 15 20 30
#   epochs per lambda:  10
#   total adv epochs:   50
#
# Usage:
#   bash run_npf_lastquad_lambda_schedule_ddp.sh [SEED] [NPROC] [K]
# =============================================================================

SEED=${1:-${SEED:-1}}
NPROC=${2:-${NPROC:-4}}
K=${3:-${K:-20}}

SRC_DIR="${SRC_DIR:-/mloscratch/homes/aabdolla/LAT}"
RESULTS_ROOT="${RESULTS_ROOT:-${SRC_DIR}/input_icnn_ddp_runs}"
LAMBDA_SCHEDULE=${LAMBDA_SCHEDULE:-"5 10 15 20 30"}
EPOCHS_PER_LAMBDA=${EPOCHS_PER_LAMBDA:-10}
OUTPUT_FOLDER_NAME=${OUTPUT_FOLDER_NAME:-}
RUN_NAME=${RUN_NAME:-}

SKIP_PGD_DURING_TRAIN=${SKIP_PGD_DURING_TRAIN:-0}
EVAL_PGD_SAMPLES=${EVAL_PGD_SAMPLES:-1000}
EPOCHS_ICNN_PRETRAIN=${EPOCHS_ICNN_PRETRAIN:-0}
FORCE_SCHEDULE=${FORCE_SCHEDULE:-0}
FREEZE_BATCHNORM=${FREEZE_BATCHNORM:-1}
FREEZE_BATCHNORM_AFFINE=${FREEZE_BATCHNORM_AFFINE:-${FREEZE_BN_AFFINE:-$FREEZE_BATCHNORM}}
BATCHNORM_ONLINE_REFRESH=${BATCHNORM_ONLINE_REFRESH:-${ONLINE_BATCHNORM_REFRESH:-0}}
BATCHNORM_ONLINE_REFRESH_MOMENTUM=${BATCHNORM_ONLINE_REFRESH_MOMENTUM:-${ONLINE_BATCHNORM_REFRESH_MOMENTUM:-}}
RECALIBRATE_BATCHNORM=${RECALIBRATE_BATCHNORM:-0}
BATCHNORM_RECALIBRATION_BATCHES=${BATCHNORM_RECALIBRATION_BATCHES:-0}
BATCHNORM_RECALIBRATION_RESET=${BATCHNORM_RECALIBRATION_RESET:-1}
BATCHNORM_RECALIBRATION_MOMENTUM=${BATCHNORM_RECALIBRATION_MOMENTUM:-}

is_positive_int() {
    case "$1" in
        ""|*[!0-9]*) return 1 ;;
        0) return 1 ;;
        *) return 0 ;;
    esac
}

for name in SEED NPROC K EPOCHS_PER_LAMBDA; do
    value="${!name}"
    if ! is_positive_int "$value"; then
        echo "[FATAL] ${name} must be a positive integer, got '${value}'."
        exit 1
    fi
done
if ! [[ "$BATCHNORM_RECALIBRATION_BATCHES" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] BATCHNORM_RECALIBRATION_BATCHES must be a non-negative integer, got '${BATCHNORM_RECALIBRATION_BATCHES}'."
    exit 1
fi
if [ -n "$BATCHNORM_RECALIBRATION_MOMENTUM" ]; then
    python - "$BATCHNORM_RECALIBRATION_MOMENTUM" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
    if [ "$?" != "0" ]; then
        echo "[FATAL] BATCHNORM_RECALIBRATION_MOMENTUM must be finite and in [0, 1], got '${BATCHNORM_RECALIBRATION_MOMENTUM}'."
        exit 1
    fi
fi

LAMBDA_SCHEDULE_RAW="${LAMBDA_SCHEDULE//,/ }"
read -r -a LAMBDAS <<< "$LAMBDA_SCHEDULE_RAW"
if [ "${#LAMBDAS[@]}" -lt 1 ]; then
    echo "[FATAL] LAMBDA_SCHEDULE is empty."
    exit 1
fi

TOTAL_EPOCHS=$(( ${#LAMBDAS[@]} * EPOCHS_PER_LAMBDA ))
if [ -z "$OUTPUT_FOLDER_NAME" ]; then
    OUTPUT_FOLDER_NAME="npf_lq_lambda_schedule_native_seed${SEED}_K${K}_ep${TOTAL_EPOCHS}"
fi
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="$OUTPUT_FOLDER_NAME"
fi

SCHEDULE_DIR="${RESULTS_ROOT}/${OUTPUT_FOLDER_NAME}"
EPOCH_CKPT_DIR="${SCHEDULE_DIR}/epoch_checkpoints"
SCHEDULE_MANIFEST="${SCHEDULE_DIR}/lambda_schedule_manifest.tsv"
FINAL_CKPT="${SCHEDULE_DIR}/npf_lastquad/seed_${SEED}/npf_lastquad_seed${SEED}_last.pth"
DONE_FILE="${SCHEDULE_DIR}/.lambda_schedule_completed"

is_number() {
    case "$1" in
        ""|*[!0-9.eE+-]*) return 1 ;;
    esac
    python - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0.0 else 1)
PY
}

for lam in "${LAMBDAS[@]}"; do
    if ! is_number "$lam"; then
        echo "[FATAL] Every lambda must be a positive finite number, got '${lam}'."
        exit 1
    fi
done

if [ "$FORCE_SCHEDULE" = "1" ] && [ -d "$SCHEDULE_DIR" ]; then
    STALE_DIR="${SCHEDULE_DIR}.force_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    echo "[INFO] FORCE_SCHEDULE=1; moving existing schedule output to ${STALE_DIR}"
    mv "$SCHEDULE_DIR" "$STALE_DIR"
fi

mkdir -p "$SCHEDULE_DIR" "$EPOCH_CKPT_DIR"

if [ "$FORCE_SCHEDULE" != "1" ] && [ -f "$DONE_FILE" ] && [ -f "$FINAL_CKPT" ]; then
    echo "[SKIP] Native lambda schedule already completed at ${SCHEDULE_DIR}"
    echo "       Set FORCE_SCHEDULE=1 or OUTPUT_FOLDER_NAME=<new-name> for a fresh run."
    exit 0
fi

cat > "$SCHEDULE_MANIFEST" <<EOF
mode	lambda_schedule	epochs_per_lambda	total_epochs	seed	world_size	k	freeze_batchnorm	freeze_batchnorm_affine	bn_online_refresh	bn_online_refresh_momentum	recalibrate_batchnorm	bn_recalibration_batches	bn_recalibration_reset	bn_recalibration_momentum	results_dir	epoch_checkpoint_dir	final_checkpoint
native	${LAMBDA_SCHEDULE_RAW}	${EPOCHS_PER_LAMBDA}	${TOTAL_EPOCHS}	${SEED}	${NPROC}	${K}	${FREEZE_BATCHNORM}	${FREEZE_BATCHNORM_AFFINE}	${BATCHNORM_ONLINE_REFRESH}	${BATCHNORM_ONLINE_REFRESH_MOMENTUM:-module-default}	${RECALIBRATE_BATCHNORM}	${BATCHNORM_RECALIBRATION_BATCHES}	${BATCHNORM_RECALIBRATION_RESET}	${BATCHNORM_RECALIBRATION_MOMENTUM:-cumulative}	${SCHEDULE_DIR}	${EPOCH_CKPT_DIR}	${FINAL_CKPT}
EOF

echo ""
echo "================================================================"
echo "  NPF-LastQuad native lambda scheduling — DDP"
echo "  seed=${SEED}  GPUs=${NPROC}  K=${K}"
echo "  lambdas=${LAMBDA_SCHEDULE_RAW}"
echo "  epochs_per_lambda=${EPOCHS_PER_LAMBDA}  total_adv_epochs=${TOTAL_EPOCHS}"
echo "  BatchNorm: freeze=${FREEZE_BATCHNORM}  freeze_affine=${FREEZE_BATCHNORM_AFFINE}  online_refresh=${BATCHNORM_ONLINE_REFRESH}  online_momentum=${BATCHNORM_ONLINE_REFRESH_MOMENTUM:-module-default}  recalibrate=${RECALIBRATE_BATCHNORM}  batches=${BATCHNORM_RECALIBRATION_BATCHES}  reset=${BATCHNORM_RECALIBRATION_RESET}  momentum=${BATCHNORM_RECALIBRATION_MOMENTUM:-cumulative}"
echo "  PGD during train: skip=${SKIP_PGD_DURING_TRAIN} samples=${EVAL_PGD_SAMPLES}"
echo "  results_dir=${SCHEDULE_DIR}"
echo "  per_epoch_checkpoints=${EPOCH_CKPT_DIR}"
echo "================================================================"

RUN_NAME="$RUN_NAME" \
OUTPUT_FOLDER_NAME="$OUTPUT_FOLDER_NAME" \
RESULTS_DIR="$SCHEDULE_DIR" \
LAMBDA_SCHEDULE="$LAMBDA_SCHEDULE_RAW" \
LAMBDA_STAGE_EPOCHS="$EPOCHS_PER_LAMBDA" \
SAVE_EVERY_EPOCH_DIR="$EPOCH_CKPT_DIR" \
CHECKPOINT_EPOCH_OFFSET=0 \
SKIP_PGD_DURING_TRAIN="$SKIP_PGD_DURING_TRAIN" \
EVAL_PGD_SAMPLES="$EVAL_PGD_SAMPLES" \
EPOCHS_ICNN_PRETRAIN="$EPOCHS_ICNN_PRETRAIN" \
FREEZE_BATCHNORM="$FREEZE_BATCHNORM" \
FREEZE_BATCHNORM_AFFINE="$FREEZE_BATCHNORM_AFFINE" \
BATCHNORM_ONLINE_REFRESH="$BATCHNORM_ONLINE_REFRESH" \
BATCHNORM_ONLINE_REFRESH_MOMENTUM="$BATCHNORM_ONLINE_REFRESH_MOMENTUM" \
RECALIBRATE_BATCHNORM="$RECALIBRATE_BATCHNORM" \
BATCHNORM_RECALIBRATION_BATCHES="$BATCHNORM_RECALIBRATION_BATCHES" \
BATCHNORM_RECALIBRATION_RESET="$BATCHNORM_RECALIBRATION_RESET" \
BATCHNORM_RECALIBRATION_MOMENTUM="$BATCHNORM_RECALIBRATION_MOMENTUM" \
bash "${SRC_DIR}/run_runtime_sweep_ddp.sh" 0 "$SEED" "$NPROC" "$K" "$TOTAL_EPOCHS"

if [ ! -f "$FINAL_CKPT" ]; then
    echo "[FATAL] Expected final checkpoint not found: ${FINAL_CKPT}"
    exit 1
fi

touch "$DONE_FILE"

echo ""
echo "================================================================"
echo "  Native lambda scheduling complete"
echo "  final_checkpoint=${FINAL_CKPT}"
echo "  per_epoch_checkpoints=${EPOCH_CKPT_DIR}"
echo "  manifest=${SCHEDULE_MANIFEST}"
echo "================================================================"
