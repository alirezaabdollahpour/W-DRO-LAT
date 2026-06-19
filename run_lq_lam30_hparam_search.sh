#!/usr/bin/env bash
# Fixed-lambda NPF-LastQuad hyperparameter sweep.
#
# Default behavior is dry-run: print csub commands only.
# To submit: SUBMIT=1 bash run_lq_lam30_hparam_search.sh phase1

set -euo pipefail

MODE="${1:-phase1}"
SUBMIT="${SUBMIT:-0}"

SRC_DIR="${SRC_DIR:-/mloscratch/homes/aabdolla/LAT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mloscratch/homes/aabdolla/optiselect/.venv/bin/activate}"
CSUB="${CSUB:-python csub.py}"
NODE_TYPE="${NODE_TYPE:-h100}"
TRAIN_TIME="${TRAIN_TIME:-1d}"
OOD_TIME="${OOD_TIME:-8h}"
GPUS="${GPUS:-2}"
NPROC="${NPROC:-2}"
SEED="${SEED:-1}"

run_or_print() {
    local cmd="$1"
    if [ "${SUBMIT}" = "1" ]; then
        echo "[submit] ${cmd}"
        eval "${cmd}"
    else
        echo "${cmd}"
        echo
    fi
}

submit_train() {
    local out="$1"
    local job="$2"
    local hidden="$3"
    local k="$4"
    local lr_theta="$5"
    local muon_lr="$6"
    local momentum="$7"
    local grad_norm="$8"
    local eval_samples="${9:-2000}"
    local fallback_lr="${10:-0.0}"

    local command
    command="cd ${SRC_DIR} && \
source ${VENV_ACTIVATE} && \
RUN_NAME=${out}_seed${SEED} \
LR_THETA=${lr_theta} \
PENALTY_LAMBDA=30 \
LAMBDA_SCHEDULE='' \
LAMBDA_STAGE_EPOCHS=0 \
USE_MARGIN_LOSS=1 \
NPF_LASTQUAD_HIDDEN='${hidden}' \
NPF_LASTQUAD_ACTIVATION=softplus \
NPF_LASTQUAD_INIT_EPS=1e-4 \
NPF_INNER_OPTIMIZER=muon \
NPF_MUON_LR=${muon_lr} \
NPF_MUON_MOMENTUM=${momentum} \
NPF_MUON_NESTEROV=1 \
NPF_MUON_NS_STEPS=5 \
NPF_MUON_MATRIX_LR_SCALE=auto \
NPF_MUON_WEIGHT_DECAY=0.0 \
NPF_MUON_MAX_GRAD_NORM=${grad_norm} \
NPF_MUON_FALLBACK=adamw \
NPF_MUON_FALLBACK_LR=${fallback_lr} \
NPF_MUON_FALLBACK_WEIGHT_DECAY=0.0 \
NPF_MUON_ADAM_BETA1=0.9 \
NPF_MUON_ADAM_BETA2=0.999 \
NPF_MUON_ADAM_EPS=1e-8 \
EVAL_PGD_SAMPLES=${eval_samples} \
INP_STEPS=20 \
INP_RESTARTS=5 \
OMEGA_STEPS=${k} \
OUTPUT_FOLDER_NAME=${out} \
bash run_runtime_sweep_ddp.sh 0 ${SEED} ${NPROC} ${k} 50"

    run_or_print "${CSUB} -n ${job} -g ${GPUS} -t ${TRAIN_TIME} --train --large-shm --node-type ${NODE_TYPE} --command \"${command}\""
}

submit_ood_existing() {
    local out="$1"
    local job="$2"
    local k="$3"

    local results_dir="${SRC_DIR}/input_icnn_ddp_runs/${out}"
    local ckpt="${results_dir}/npf_lastquad/seed_${SEED}/npf_lastquad_seed${SEED}_best_robust.pth"
    local command
    command="cd ${SRC_DIR} && \
source ${VENV_ACTIVATE} && \
RESULTS_DIR=${results_dir} \
CKPT=${ckpt} \
CHECKPOINT_KIND=best_robust \
STORE_EVAL_WITH_CKPT=1 \
FORCE_EVAL=1 \
RUN_AUTOATTACK=0 \
SKIP_CIFAR10W=0 \
SKIP_CIFAR10C=0 \
INP_STEPS=40 \
INP_RESTARTS=5 \
INP_EPS_SWEEP='0.1 0.2 0.3 0.4 0.5 0.6 0.7' \
bash run_OOD_sweep.sh 0 ${SEED} ${k}"

    run_or_print "${CSUB} -n ${job} -g 1 -t ${OOD_TIME} --train --large-shm --node-type ${NODE_TYPE} --command \"${command}\""
}

echo "# MODE=${MODE} SUBMIT=${SUBMIT}"
echo "# Lambda and epochs are fixed: PENALTY_LAMBDA=30, epochs_adv=50."
echo "# Set SUBMIT=1 to actually call csub.py."
echo

case "${MODE}" in
    phase0|ood-existing)
        submit_ood_existing "Muon_lam30_1024_logsum_lr0p0015_K2_bnfix" "lq-lam30-1024-k2-ood" 2
        submit_ood_existing "Muon_lam30_512_logsum_warm0_adv50_K30" "lq-lam30-512-k30-ood" 30
        ;;
    phase1)
        submit_ood_existing "Muon_lam30_1024_logsum_lr0p0015_K2_bnfix" "lq-lam30-1024-k2-ood" 2
        submit_ood_existing "Muon_lam30_512_logsum_warm0_adv50_K30" "lq-lam30-512-k30-ood" 30
        submit_train "Muon_lam30_512_logsum_lr0p0015_K5_bnfix" "lq-lam30-512-k5" "512 512 512 512" 5 0.0015 3e-4 0.90 5.0
        submit_train "Muon_lam30_128_logsum_lr0p0015_K3_bnfix" "lq-lam30-128-k3" "128 128 128 128" 3 0.0015 3e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0015_K3_bnfix" "lq-lam30-512-k3" "512 512 512 512" 3 0.0015 3e-4 0.90 5.0
        submit_train "Muon_lam30_1024_logsum_lr0p0015_K3_bnfix" "lq-lam30-1024-k3" "1024 512 512 256 128 64" 3 0.0015 3e-4 0.90 5.0
        ;;
    phase2)
        submit_train "Muon_lam30_1024_logsum_lr0p0015_K5_bnfix" "lq-lam30-1024-k5" "1024 512 512 256 128 64" 5 0.0015 3e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0010_K3_bnfix" "lq-lam30-512-k3-lr1e3" "512 512 512 512" 3 0.0010 3e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0020_K3_bnfix" "lq-lam30-512-k3-lr2e3" "512 512 512 512" 3 0.0020 3e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0015_K3_muon2e4_bnfix" "lq-lam30-512-k3-muon2e4" "512 512 512 512" 3 0.0015 2e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0015_K3_muon5e4_bnfix" "lq-lam30-512-k3-muon5e4" "512 512 512 512" 3 0.0015 5e-4 0.90 5.0
        submit_train "Muon_lam30_512_logsum_lr0p0015_K3_fb1p5e4_bnfix" "lq-lam30-512-k3-fb1p5e4" "512 512 512 512" 3 0.0015 3e-4 0.90 5.0 2000 1.5e-4
        submit_train "Muon_lam30_512_logsum_lr0p0015_K3_fb1e4_bnfix" "lq-lam30-512-k3-fb1e4" "512 512 512 512" 3 0.0015 3e-4 0.90 5.0 2000 1e-4
        ;;
    all)
        "$0" phase1
        "$0" phase2
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Use: phase0, phase1, phase2, or all" >&2
        exit 2
        ;;
esac
