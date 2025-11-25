#!/usr/bin/env bash
set -euo pipefail

python evaluate_wrm_lat_cifar10_variants.py \
    --ckpt /mnt/lts4/scratch/students/aabdolla/LAT/R2_INPUT_icnn_lambda_30_epochs_adv_5_l2_PGD_without_amortization.pth \
    --skip-cifar10w \
    --skip-cifar10c \
    --inp-p "2" \
    --inp-eps 0.5 --inp-steps 20 --inp-restarts 5 \
    # --cifar10c-corruptions all \
    # --cifar10w-root /mnt/lts4/scratch/students/aabdolla/LAT/cifar10w \
    # --autoattack \
    # --autoattack-only \
    # --autoattack-bs 128 \
    # --autoattack-version custom --autoattack-attacks apgd-ce apgd-dlr \
    # --autoattack-norm "L2" \
    # --autoattack-eps 0.5 \


