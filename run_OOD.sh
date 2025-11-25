#!/usr/bin/env bash
set -euo pipefail

# Arguments used for OOD evaluation (kept as vars so we can embed them in the save-json path)
CKPT=/mnt/lts4/scratch/students/aabdolla/LAT/R2_INPUT_icnn_lambda_30_epochs_adv_5_l2_PGD_without_amortization.pth
INP_P=2
INP_EPS=0.5
INP_STEPS=20
INP_RESTARTS=5
AA_VERSION=custom
AA_ATTACKS=("apgd-dlr" "apgd-ce")
AA_NORM=L2
AA_EPS=0.5
AA_BS=128
# CIFAR10C_CORR=all
CIFAR10C_SEVERITIES=(1 2 3 4 5)

CKPT_NAME="$(basename "${CKPT%.*}")"
ATTACKS_TAG=$(IFS=-; echo "${AA_ATTACKS[*]}")
SAVE_JSON="results/ood_${CKPT_NAME}_inpP-${INP_P}_eps-${INP_EPS}_steps-${INP_STEPS}_restarts-${INP_RESTARTS}_aa-${AA_VERSION}-${ATTACKS_TAG}_norm-${AA_NORM}_aaeps-${AA_EPS}_bs-${AA_BS}.json"
mkdir -p "$(dirname "$SAVE_JSON")"

python evaluate_wrm_lat_cifar10_variants.py \
    --ckpt "$CKPT" \
    --inp-p "$INP_P" \
    --inp-eps "$INP_EPS" --inp-steps "$INP_STEPS" --inp-restarts "$INP_RESTARTS" \
    --autoattack \
    --autoattack-bs "$AA_BS" \
    --autoattack-version "$AA_VERSION" --autoattack-attacks "${AA_ATTACKS[@]}" \
    --autoattack-norm "$AA_NORM" \
    --autoattack-eps "$AA_EPS" \
    --save-json "$SAVE_JSON" \
    --cifar10c-severities "${CIFAR10C_SEVERITIES[@]}" \
    --cifar10w-root /mnt/lts4/scratch/students/aabdolla/LAT/cifar10w

# Optional toggles you can add to the command:
#   --autoattack-only
#   --skip-cifar10w
#   --skip-cifar10c
#  --cifar10c-corruptions "$CIFAR10C_CORR" \